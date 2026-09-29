"""Attribute every CUPTI kernel once, without summing nested profiler events."""

from bisect import bisect_right
from collections import defaultdict
import math

from .kernel_classifier import GEMM_KERNEL, PREFIX


class UnresolvedProfile(RuntimeError):
    pass


def analyze_trace(trace, precision):
    events = trace["traceEvents"]
    ranges = defaultdict(list)
    cpu_ops = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        if event.get("name", "").startswith(PREFIX):
            ranges[(event["pid"], event["tid"])].append(event)
        if event.get("cat") == "cpu_op" and "External id" in event.get("args", {}):
            cpu_ops[event["args"]["External id"]] = event
    indexes = {}
    for thread, scopes in ranges.items():
        scopes.sort(key=lambda event: event["ts"])
        indexes[thread] = [event["ts"] for event in scopes]

    totals = defaultdict(float)
    inventory = defaultdict(lambda: {"calls": 0, "duration_ms": 0.0})
    activities = [event for event in events
                  if event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and event.get("ph") == "X"]
    if not any(event["cat"] == "kernel" for event in activities):
        raise UnresolvedProfile("No CUPTI CUDA kernels captured; cannot produce measured results")
    for kernel in activities:
        external_id = kernel.get("args", {}).get("External id")
        op = cpu_ops.get(external_id)
        if op is None:
            raise UnresolvedProfile(f"Uncorrelated CUDA kernel: {kernel['name']}")
        thread = (op["pid"], op["tid"])
        scopes = ranges.get(thread, [])
        starts = indexes.get(thread, [])
        position = bisect_right(starts, op["ts"]) - 1
        marker = None
        if position >= 0:
            candidate = scopes[position]
            if candidate["ts"] + candidate["dur"] >= op["ts"] + op["dur"] - 0.1:
                marker = candidate
        # Some PyTorch versions attach the kernel to the outer dispatch event.
        if marker is None:
            inside = bisect_right(starts, op["ts"] - 0.001)
            if inside < len(scopes):
                candidate = scopes[inside]
                end = op["ts"] + op["dur"]
                if candidate["ts"] + candidate["dur"] <= end + 0.1:
                    if inside + 1 < len(scopes) and scopes[inside + 1]["ts"] < end - 0.1:
                        raise UnresolvedProfile(f"Ambiguous kernel scope: {kernel['name']}")
                    marker = candidate
        if marker is None:
            raise UnresolvedProfile(f"CUDA kernel outside measured categories: {kernel['name']} ({op['name']})")
        category = marker["name"][len(PREFIX):]
        if kernel["cat"] != "kernel" and category in ("linear", "attention_matmul"):
            # cuBLAS workspace initialization and DMA are not GEMM computation.
            category = "conversion"
        if category == "attention_matmul" and not GEMM_KERNEL.search(kernel["name"]):
            if any(word in kernel["name"].lower() for word in ("copy", "elementwise", "transpose")):
                category = "conversion"
            else:
                raise UnresolvedProfile(f"Unresolved attention GEMM auxiliary kernel: {kernel['name']}")
        if category == "linear":
            expected_op = "aten::_int_mm" if precision == "w8a8" else "aten::mm"
            if op["name"] != expected_op or not GEMM_KERNEL.search(kernel["name"]):
                raise UnresolvedProfile(
                    f"UNRESOLVED_FUSED_QUANT_GEMM: {op['name']} / {kernel['name']}")
        duration = kernel["dur"] / 1000.0
        if not math.isfinite(duration) or duration < 0:
            raise UnresolvedProfile("Invalid kernel duration")
        totals[category] += duration
        key = (category, op["name"], kernel["name"], kernel["cat"])
        inventory[key]["calls"] += 1
        inventory[key]["duration_ms"] += duration

    # Independent sum over raw CUDA activities detects missing/double attribution.
    gpu_total = sum(event["dur"] for event in activities) / 1000.0
    if not math.isclose(sum(totals.values()), gpu_total, rel_tol=0.000001, abs_tol=1e-9):
        raise UnresolvedProfile("Kernel accounting mismatch")
    records = [{"category": c, "operator": op, "kernel": name, "activity": activity, **values}
               for (c, op, name, activity), values in sorted(inventory.items())]
    return dict(totals), gpu_total, records
