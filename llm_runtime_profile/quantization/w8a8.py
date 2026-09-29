"""Dynamic per-token INT8 activations, static per-row INT8 weights, INT32 GEMM."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.profiler import profile, ProfilerActivity

from ..instrumentation.scopes import scope
from ..profiler.kernel_classifier import GEMM_KERNEL
from .base import BackendUnavailable, QuantBackend, replace_linears


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


class W8A8Linear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.padded_k = round_up(self.in_features, 32)
        self.padded_n = round_up(self.out_features, 8)
        # Setup-only quantization. Never included in per-forward runtime.
        weight = original.weight.detach().float()
        scale = weight.abs().amax(dim=1).clamp_min(1e-12) / 127.0
        quantized = (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        quantized = F.pad(quantized, (0, self.padded_k - self.in_features,
                                    0, self.padded_n - self.out_features))
        self.register_buffer("weight_int8", quantized.contiguous())
        self.register_buffer("weight_scale", F.pad(scale, (0, self.padded_n - self.out_features), value=1))
        self.register_buffer("bias", original.bias.detach().clone() if original.bias is not None else None)

    def forward(self, x):
        with scope("conversion"):
            flat = x.reshape(-1, self.in_features).float()
        rows = flat.shape[0]
        with scope("scale"):
            scale = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        with scope("quantize"):
            quantized = (flat / scale).round().clamp(-127, 127).to(torch.int8)
        with scope("packing"):
            quantized = F.pad(quantized, (0, self.padded_k - self.in_features,
                                        0, max(32, round_up(rows, 8)) - rows))
            weight = self.weight_int8.t()
        # This call accepts ONLY int8 x int8 and returns int32. No float fallback.
        with scope("linear_module"):
            accumulated = torch._int_mm(quantized, weight)
        with scope("dequantize"):
            output = accumulated[:rows].float() * scale * self.weight_scale[None, :]
            output = output[:, :self.out_features].to(x.dtype)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*x.shape[:-1], self.out_features)


class W8A8Backend(QuantBackend):
    precision = "w8a8"
    weight_precision = "INT8"
    activation_precision = "INT8"

    def backend_name(self):
        return "PyTorch torch._int_mm / CUDA cuBLAS INT8 x INT8 -> INT32"

    @torch.inference_mode()
    def validate(self):
        self.check_cuda()
        if not hasattr(torch, "_int_mm"):
            raise BackendUnavailable("True W8A8 CUDA backend is unavailable: torch._int_mm missing")
        try:
            generator = torch.Generator().manual_seed(42)
            a_cpu = torch.randint(-127, 128, (32, 64), dtype=torch.int8, generator=generator)
            b_cpu = torch.randint(-127, 128, (48, 64), dtype=torch.int8, generator=generator)
            a = a_cpu.to(self.device)
            b = b_cpu.to(self.device).t()
            torch._int_mm(a, b)
            torch.cuda.synchronize(self.device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                actual = torch._int_mm(a, b)
                torch.cuda.synchronize(self.device)
            expected = (a_cpu.long() @ b_cpu.long().t()).int()
            if actual.dtype != torch.int32 or not torch.equal(actual.cpu(), expected):
                raise BackendUnavailable("INT8 GEMM failed exact INT32 arithmetic validation")
            names = sorted({event.name for event in prof.events()
                            if event.device_type == torch.autograd.DeviceType.CUDA
                            and GEMM_KERNEL.search(event.name)})
            if not names:
                raise BackendUnavailable("Cannot prove a real CUDA INT8 GEMM kernel")
        except (RuntimeError, AttributeError) as exc:
            raise BackendUnavailable(f"True W8A8 CUDA backend is unavailable: {exc}") from exc
        self.evidence = {"operator": "aten::_int_mm", "input_dtype": "torch.int8",
                         "weight_dtype": "torch.int8", "output_dtype": "torch.int32",
                         "exact_integer_check": True, "gemm_kernels": names}
        return self.evidence

    def quantize_model(self, model):
        return replace_linears(model, W8A8Linear)
