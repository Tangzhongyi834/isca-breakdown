#!/usr/bin/env python3
"""Profile actual WikiText-2 prefill computation on one CUDA GPU."""

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
import uuid


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", choices=("fp16", "w8a8", "w4a4"), default="fp16")
    parser.add_argument("--dataset", choices=("Salesforce/wikitext",), default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", choices=("wikitext-2-raw-v1",), default="wikitext-2-raw-v1")
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--phase", choices=("prefill",), default="prefill")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--cache-dir", default="data/wikitext2")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--save-all-traces", action="store_true", help="Otherwise keep the first full trace of every length")
    args = parser.parse_args(argv)
    if any(value <= 0 or value > 4096 for value in args.seq_lens) or len(set(args.seq_lens)) != len(args.seq_lens):
        parser.error("Sequence lengths must be unique and within [1, 4096]")
    if args.num_samples < 1 or args.repeats < 1 or args.warmup < 0 or args.token_offset < 0:
        parser.error("Samples/repeats must be positive; warmup/offset must be nonnegative")
    args.seq_lens.sort()
    return args


def print_summary(row):
    print("\n========================================\nExperiment\n========================================")
    for label, field in (("Model", "model"), ("Precision", "precision"), ("Sequence Length", "seq_len"), ("Samples", "num_samples")):
        print(f"{label:20}: {row[field]}")
    for label, field in (("Linear GEMM", "linear"), ("Nonlinear", "nonlinear"), ("    Softmax", "softmax"),
                         ("    RoPE", "rope"), ("    SiLU", "silu"), ("    RMSNorm", "rmsnorm"),
                         ("Attention MatMul", "attention_matmul"), ("Other Compute", "other_compute"),
                         ("Quantization", "quantization_overhead"), ("Compute Total", "compute_total"),
                         ("Physical Total", "physical_total")):
        print(f"{label:20}: {row[field + '_ms']:.6f} ms")
    for label in ("linear", "nonlinear", "other"):
        print(f"{label.title() + ' Fraction':20}: {row[label + '_pct']:.3f} %")
    print("========================================", flush=True)


def run(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from llm_runtime_profile.data.wikitext2 import prepare_samples
    from llm_runtime_profile.environment import capture_environment, gpu_lock
    from llm_runtime_profile.models import get_adapter
    from llm_runtime_profile.profiler.result_aggregator import reduce_measurements, publish_configuration, write_csv, write_json
    from llm_runtime_profile.profiler.runtime_profiler import RuntimeProfiler
    from llm_runtime_profile.quantization import create_backend

    backend = create_backend(args.precision, args.device)
    # Fail before downloading gated or multi-GB models for an unavailable backend.
    if args.precision == "w4a4" and not args.prepare_data_only:
        backend.validate()
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, local_files_only=args.local_files_only)
    if config.model_type not in ("llama", "mistral"):
        raise ValueError(f"Unsupported model_type: {config.model_type}")
    if max(args.seq_lens) > config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the checkpoint's declared context window")
    samples, dataset_metadata = prepare_samples(args.model, config.model_type, args.num_samples,
                                                args.token_offset, args.cache_dir, args.revision, args.local_files_only)
    print("Sample 0:")
    for seq_len in args.seq_lens:
        print(f"{seq_len:4} -> {samples[0, :seq_len].numel()} tokens")
    if args.prepare_data_only:
        return
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("CUDA is required for measured profiling")
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    backend.device = device
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    with gpu_lock(device) as gpu:
        backend.validate()
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, attn_implementation="eager",
            revision=args.revision, local_files_only=args.local_files_only, low_cpu_mem_usage=True)
        model = model.to(device).eval()
        model.config.use_cache = False
        adapter = get_adapter(model)
        backend.quantize_model(model)
        torch.cuda.synchronize(device)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
        root = Path(args.output_dir)
        run_dir = root / "runs" / run_id
        environment = capture_environment(args, model, backend, dataset_metadata, run_id, gpu)
        write_json(run_dir / "environment.json", environment)
        write_json(run_dir / "status.json", {"status": "running", "completed_lengths": []})
        runner = RuntimeProfiler(model, adapter, backend, device)
        completed = []
        try:
            for seq_len in args.seq_lens:
                input_ids = samples[0, :seq_len].unsqueeze(0).to(device)
                runner.warmup(input_ids, args.warmup)
                repeats, sample_rows = [], []
                identity = {"model": args.model, "precision": args.precision.upper(), "seq_len": seq_len,
                            "measurement_kind": "measured", "validation_status": "passed",
                            "fairness_id": environment["fairness_id"], "token_sha256": dataset_metadata["token_sha256"],
                            "run_id": run_id}
                for sample_id in range(args.num_samples):
                    input_ids = samples[sample_id, :seq_len].unsqueeze(0).to(device)
                    sample_repeats = []
                    for repeat_id in range(args.repeats):
                        tag = f"L{seq_len}_sample{sample_id:03d}_repeat{repeat_id}"
                        trace_path = run_dir / "traces" / f"{tag}.json" if args.save_all_traces or (sample_id == 0 and repeat_id == 0) else None
                        row, evidence = runner.measure(input_ids, trace_path)
                        sample_repeats.append(row)
                        repeats.append({**identity, "sample_id": sample_id, "repeat_id": repeat_id, **row})
                        write_json(run_dir / "kernels" / f"{tag}.json", evidence)
                    sample_rows.append({**identity, "sample_id": sample_id, **reduce_measurements(sample_repeats)})
                    print(f"[{args.precision.upper()} L={seq_len}] sample {sample_id + 1}/{args.num_samples}", flush=True)
                backend.describe()
                environment["backend_evidence"] = backend.evidence
                write_json(run_dir / "environment.json", environment)
                write_csv(run_dir / "repeats" / f"L{seq_len}.csv", repeats)
                raw_path = run_dir / "raw" / f"L{seq_len}.csv"
                write_csv(raw_path, sample_rows)
                summary = publish_configuration(root, raw_path, identity)
                print_summary(summary)
                completed.append(seq_len)
                write_json(run_dir / "status.json", {"status": "running", "completed_lengths": completed})
            write_json(run_dir / "status.json", {"status": "complete", "completed_lengths": completed})
        except BaseException as exc:
            write_json(run_dir / "status.json", {"status": "failed", "completed_lengths": completed, "error": str(exc)})
            raise
        finally:
            del runner, adapter, model
            gc.collect()
            torch.cuda.empty_cache()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
