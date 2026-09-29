from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess

import torch


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_identity(device):
    properties = torch.cuda.get_device_properties(device)
    uuid = str(getattr(properties, "uuid", ""))
    if not uuid:
        # Without a physical identity, CUDA_VISIBLE_DEVICES aliases cannot be locked safely.
        raise RuntimeError("PyTorch must expose the GPU UUID; use the tested PyTorch 2.6 environment")
    return {"name": properties.name, "uuid": uuid,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory}


@contextmanager
def gpu_lock(device):
    identity = gpu_identity(device)
    path = Path("/tmp") / f"llm-runtime-profile-{identity['uuid']}.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another experiment holds GPU {identity['uuid']}") from exc
        try:
            yield identity
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def capture_environment(args, model, backend, dataset_metadata, run_id, gpu):
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unavailable"
    packages = {name: package_version(name) for name in ("torch", "transformers", "datasets", "torchao", "accelerate")}
    fairness = {"model": args.model, "model_revision": args.revision,
                "model_commit": getattr(model.config, "_commit_hash", None),
                "model_config": model.config.to_dict(), "gpu": gpu, "driver": driver,
                "packages": packages, "cuda_version": torch.version.cuda,
                "token_sha256": dataset_metadata["token_sha256"],
                "batch_size": 1, "seq_lens": args.seq_lens, "warmup": args.warmup,
                "repeats": args.repeats, "num_samples": args.num_samples,
                "attention_implementation": "eager", "use_cache": False,
                "profiler_version": "0.1.0", "phase": "prefill"}
    fairness_id = hashlib.sha256(json.dumps(fairness, sort_keys=True).encode()).hexdigest()
    return {"run_id": run_id, "model": args.model, "model_type": model.config.model_type,
            "precision": args.precision.upper(), "quantization_backend": backend.backend_name(),
            "gemm_backend": backend.backend_name(), "backend_evidence": backend.evidence,
            "python": platform.python_version(), "gpu": gpu, "nvidia_driver": driver,
            "cuda_version": torch.version.cuda, "packages": packages,
            "attention_implementation": "eager", "batch_size": 1, "sequence_lengths": args.seq_lens,
            "num_samples": args.num_samples, "warmup": args.warmup, "repeats": args.repeats,
            "phase": "prefill", "use_cache": False, "dataset": dataset_metadata,
            "fairness_id": fairness_id, "fairness_settings": fairness,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "timing": "CUPTI kernel durations; CUDA Events for uninstrumented physical runtime and diagnostic spans",
            "physical_runtime_method": "separate identical forward per repetition",
            "aggregation": "median per leaf across repeats, then median per leaf across samples; derived totals recomputed",
            "static_weight_quantization": "setup only; excluded from per-forward runtime"}
