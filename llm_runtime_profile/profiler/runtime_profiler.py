import json
from pathlib import Path
import tempfile
import warnings

import torch
from torch.profiler import profile, ProfilerActivity

from ..instrumentation.quantization import QUANT_CATEGORIES
from .cuda_timer import CUDAEventTimer
from .kernel_classifier import OperationInstrumentation
from .trace_analysis import analyze_trace, UnresolvedProfile

LEAF_CATEGORIES = ("linear", "softmax", "rope", "silu", "rmsnorm", "attention_matmul",
                   "other_compute", *QUANT_CATEGORIES)


def derive_metrics(row):
    result = dict(row)
    result["nonlinear_ms"] = sum(result[f"{name}_ms"] for name in ("softmax", "rope", "silu", "rmsnorm"))
    result["quantization_overhead_ms"] = sum(result[f"{name}_ms"] for name in QUANT_CATEGORIES)
    result["other_ms"] = result["attention_matmul_ms"] + result["other_compute_ms"]
    result["compute_total_ms"] = result["linear_ms"] + result["nonlinear_ms"] + result["other_ms"]
    if result["compute_total_ms"] <= 0:
        raise ValueError("Non-positive compute runtime")
    for name in ("linear", "nonlinear", "other"):
        result[f"{name}_pct"] = result[f"{name}_ms"] / result["compute_total_ms"] * 100.0
    return result


def validate_input(model, input_ids, seq_len, device):
    if input_ids.shape != (1, seq_len) or input_ids.dtype != torch.long:
        raise ValueError(f"Expected int64 input (1, {seq_len}), got {input_ids.shape}/{input_ids.dtype}")
    if input_ids.device != torch.device(device):
        raise ValueError("Input device mismatch")
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.device != input_ids.device:
            raise ValueError(f"Model device mismatch at {name}: {value.device}")
        if value.is_floating_point() and value.dtype not in (torch.float16, torch.float32):
            raise ValueError(f"Unexpected model dtype at {name}: {value.dtype}")
    if model.get_input_embeddings().weight.dtype != torch.float16:
        raise ValueError("Model must use FP16 activations")


def validate_output(output, input_ids, vocab_size):
    if output.logits.shape != (*input_ids.shape, vocab_size):
        raise ValueError(f"Unexpected logits shape: {output.logits.shape}")
    if output.logits.device != input_ids.device or not output.logits.is_floating_point():
        raise ValueError("Output device/dtype mismatch")
    if not torch.isfinite(output.logits).all().item():
        raise ValueError("Non-finite logits; this configuration cannot produce paper data")


class RuntimeProfiler:
    def __init__(self, model, adapter, backend, device):
        self.model = model
        self.adapter = adapter
        self.backend = backend
        self.device = torch.device(device)

    def forward(self, input_ids):
        return self.model(input_ids=input_ids, attention_mask=None, use_cache=False,
                          output_attentions=False, output_hidden_states=False, return_dict=True)

    @torch.inference_mode()
    def warmup(self, input_ids, count=5):
        validate_input(self.model, input_ids, input_ids.shape[1], self.device)
        for _ in range(count):
            output = self.forward(input_ids)
            del output
        torch.cuda.synchronize(self.device)
        output = self.forward(input_ids)
        validate_output(output, input_ids, self.model.config.vocab_size)
        del output

    @torch.inference_mode()
    def measure(self, input_ids, trace_path=None):
        # A separate identical, uninstrumented forward measures physical runtime.
        # Event/dispatch/CUPTI overhead must not inflate the reported physical time.
        torch.cuda.synchronize(self.device)
        physical = CUDAEventTimer(self.device)
        with physical.measure("physical"):
            output = self.forward(input_ids)
        physical_ms = physical.collect()["physical"]
        validate_output(output, input_ids, self.model.config.vocab_size)
        del output

        timer = CUDAEventTimer(self.device)
        torch.cuda.synchronize(self.device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with timer.measure("profiled_physical"):
                with self.adapter.instrument(), OperationInstrumentation(timer):
                    output = self.forward(input_ids)
            torch.cuda.synchronize(self.device)
        event_ms = timer.collect(synchronize=False)
        validate_output(output, input_ids, self.model.config.vocab_size)
        del output
        if trace_path is not None:
            path = Path(trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(path))
            with path.open() as handle:
                trace = json.load(handle)
        else:
            with tempfile.TemporaryDirectory(prefix="llm-profile-trace-") as directory:
                path = Path(directory) / "trace.json"
                prof.export_chrome_trace(str(path))
                with path.open() as handle:
                    trace = json.load(handle)
        totals, gpu_total, kernels = analyze_trace(trace, self.backend.precision)
        for required in ("linear", "softmax", "rope", "silu", "rmsnorm", "attention_matmul"):
            if totals.get(required, 0) <= 0:
                raise UnresolvedProfile(f"Missing required GPU category: {required}")
        row = derive_metrics({f"{name}_ms": totals.get(name, 0.0) for name in LEAF_CATEGORIES})
        kernel_total = sum(record["duration_ms"] for record in kernels if record["activity"] == "kernel")
        row.update(physical_total_ms=physical_ms, profiled_physical_ms=event_ms["profiled_physical"],
                   gpu_kernel_total_ms=kernel_total, gpu_activity_total_ms=gpu_total)
        independently_measured_compute = gpu_total - row["quantization_overhead_ms"]
        discrepancy = abs(row["compute_total_ms"] - independently_measured_compute) / independently_measured_compute
        if discrepancy > 0.05:
            warnings.warn(f"Measured operator categories differ from compute total by {discrepancy:.1%}. "
                          "Profiling may contain missing or double-counted kernels.")
            raise UnresolvedProfile("Runtime consistency check failed")
        row["consistency_error_pct"] = discrepancy * 100
        self.backend.evidence["gemm_kernels"] = sorted({k["kernel"] for k in kernels if k["category"] == "linear"})
        return row, {"cuda_event_spans_ms": event_ms, "kernels": kernels,
                     "kernel_sum_ms": kernel_total, "cuda_activity_sum_ms": gpu_total,
                     "measurement_kind": "measured"}
