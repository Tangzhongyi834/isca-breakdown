"""Attribute every CUPTI CUDA activity exactly once without double-counting
nested profiler events.

CUDA activities that cannot be reliably mapped to one of the explicitly
measured categories are assigned to ``other`` rather than causing the entire
profiling run to fail. This preserves total GPU-time accounting while avoiding
incorrect attribution.
"""

from bisect import bisect_right
from collections import defaultdict
import math

from .kernel_classifier import GEMM_KERNEL, PREFIX


class UnresolvedProfile(RuntimeError):
    pass


def _add_activity(totals, inventory, category, op_name, activity):
    """Account for one CUDA activity exactly once."""

    duration = activity["dur"] / 1000.0

    if not math.isfinite(duration) or duration < 0:
        raise UnresolvedProfile(
            f"Invalid CUDA activity duration: {activity.get('name', '<unknown>')}"
        )

    totals[category] += duration

    key = (
        category,
        op_name,
        activity["name"],
        activity["cat"],
    )

    inventory[key]["calls"] += 1
    inventory[key]["duration_ms"] += duration


def analyze_trace(trace, precision):
    events = trace["traceEvents"]

    # ------------------------------------------------------------------
    # Collect explicitly marked profiling ranges and CPU operators.
    # ------------------------------------------------------------------
    ranges = defaultdict(list)
    cpu_ops = {}

    for event in events:
        if event.get("ph") != "X":
            continue

        name = event.get("name", "")

        if name.startswith(PREFIX):
            ranges[(event["pid"], event["tid"])].append(event)

        if (
            event.get("cat") == "cpu_op"
            and "External id" in event.get("args", {})
        ):
            external_id = event["args"]["External id"]
            cpu_ops[external_id] = event

    # Sort profiling ranges by start timestamp so that bisect can be used
    # for efficient scope lookup.
    indexes = {}

    for thread, scopes in ranges.items():
        scopes.sort(key=lambda event: event["ts"])
        indexes[thread] = [event["ts"] for event in scopes]

    # ------------------------------------------------------------------
    # Collect CUDA activities.
    # ------------------------------------------------------------------
    totals = defaultdict(float)

    inventory = defaultdict(
        lambda: {
            "calls": 0,
            "duration_ms": 0.0,
        }
    )

    activities = [
        event
        for event in events
        if (
            event.get("cat")
            in ("kernel", "gpu_memcpy", "gpu_memset")
            and event.get("ph") == "X"
        )
    ]

    if not any(event["cat"] == "kernel" for event in activities):
        raise UnresolvedProfile(
            "No CUPTI CUDA kernels captured; cannot produce measured results"
        )

    # ------------------------------------------------------------------
    # Attribute every CUDA activity exactly once.
    # ------------------------------------------------------------------
    for activity in activities:

        args = activity.get("args", {})
        external_id = args.get("External id")

        op = cpu_ops.get(external_id)

        # --------------------------------------------------------------
        # CUPTI activity without a corresponding CPU profiler event.
        #
        # This can happen for auxiliary PyTorch/CUDA kernels. We cannot
        # reliably infer their semantic category, so count them as other.
        # --------------------------------------------------------------
        if op is None:
            _add_activity(
                totals,
                inventory,
                "other",
                "<no_cpu_op>",
                activity,
            )
            continue

        thread = (op["pid"], op["tid"])

        scopes = ranges.get(thread, [])
        starts = indexes.get(thread, [])

        marker = None

        # --------------------------------------------------------------
        # Case 1:
        # CPU op is contained inside an already-started profiling scope.
        # --------------------------------------------------------------
        position = bisect_right(starts, op["ts"]) - 1

        if position >= 0:
            candidate = scopes[position]

            candidate_end = candidate["ts"] + candidate["dur"]
            op_end = op["ts"] + op["dur"]

            if candidate_end >= op_end - 0.1:
                marker = candidate

        # --------------------------------------------------------------
        # Case 2:
        # Some PyTorch versions attach the CUDA kernel to an outer
        # dispatch CPU event while the marked range is nested inside it.
        # --------------------------------------------------------------
        if marker is None:

            inside = bisect_right(
                starts,
                op["ts"] - 0.001,
            )

            if inside < len(scopes):

                candidate = scopes[inside]
                candidate_end = candidate["ts"] + candidate["dur"]
                op_end = op["ts"] + op["dur"]

                if candidate_end <= op_end + 0.1:

                    # More than one profiling marker inside this CPU op
                    # makes attribution ambiguous.
                    if (
                        inside + 1 < len(scopes)
                        and scopes[inside + 1]["ts"] < op_end - 0.1
                    ):
                        raise UnresolvedProfile(
                            f"Ambiguous kernel scope: {activity['name']}"
                        )

                    marker = candidate

        # --------------------------------------------------------------
        # The activity has a CPU op but is outside explicitly measured
        # ranges. Preserve it in total GPU accounting as "other".
        # --------------------------------------------------------------
        if marker is None:

            _add_activity(
                totals,
                inventory,
                "other",
                op["name"],
                activity,
            )

            continue

        category = marker["name"][len(PREFIX):]

        # --------------------------------------------------------------
        # memcpy / memset associated with linear or attention GEMM scopes
        # are data movement / initialization rather than GEMM arithmetic.
        # --------------------------------------------------------------
        if (
            activity["cat"] != "kernel"
            and category in ("linear", "attention_matmul")
        ):
            category = "conversion"

        # --------------------------------------------------------------
        # Attention matmul scope may contain auxiliary CUDA kernels.
        #
        # Known data movement / elementwise kernels are classified as
        # conversion. Unknown auxiliary kernels remain strict failures
        # because silently treating them as GEMM would corrupt results.
        # --------------------------------------------------------------
        if (
            category == "attention_matmul"
            and not GEMM_KERNEL.search(activity["name"])
        ):

            kernel_name = activity["name"].lower()

            auxiliary_keywords = (
                "copy",
                "elementwise",
                "transpose",
                "memcpy",
                "memset",
                "cast",
                "convert",
            )

            if any(
                word in kernel_name
                for word in auxiliary_keywords
            ):
                category = "conversion"

            else:
                raise UnresolvedProfile(
                    "Unresolved attention GEMM auxiliary kernel: "
                    f"{activity['name']}"
                )

        # --------------------------------------------------------------
        # Linear scope must correspond to the expected GEMM operator.
        #
        # FP16:
        #     aten::mm
        #
        # W8A8:
        #     aten::_int_mm
        #
        # Keep this check strict because otherwise quantization or fused
        # kernels could accidentally be reported as GEMM computation.
        # --------------------------------------------------------------
        if category == "linear":

            expected_op = (
                "aten::_int_mm"
                if precision == "w8a8"
                else "aten::mm"
            )

            if (
                op["name"] != expected_op
                or not GEMM_KERNEL.search(activity["name"])
            ):
                raise UnresolvedProfile(
                    "UNRESOLVED_FUSED_QUANT_GEMM: "
                    f"{op['name']} / {activity['name']}"
                )

        # --------------------------------------------------------------
        # Normal attributed activity.
        # --------------------------------------------------------------
        _add_activity(
            totals,
            inventory,
            category,
            op["name"],
            activity,
        )

    # ------------------------------------------------------------------
    # Independent GPU total calculated directly from raw CUPTI activity.
    #
    # This verifies that every activity was attributed exactly once and
    # that no activity was accidentally omitted or double counted.
    # ------------------------------------------------------------------
    gpu_total = (
        sum(event["dur"] for event in activities)
        / 1000.0
    )

    attributed_total = sum(totals.values())

    if not math.isclose(
        attributed_total,
        gpu_total,
        rel_tol=0.000001,
        abs_tol=1e-9,
    ):
        raise UnresolvedProfile(
            "Kernel accounting mismatch: "
            f"attributed={attributed_total:.6f} ms, "
            f"raw_gpu={gpu_total:.6f} ms"
        )

    # ------------------------------------------------------------------
    # Report unattributed / out-of-scope GPU time.
    #
    # A small "other" percentage is expected for runtime/framework
    # auxiliary kernels. A large percentage indicates that profiler
    # markers or attribution rules should be inspected before using the
    # measurement in final evaluation figures.
    # ------------------------------------------------------------------
    other_time = totals.get("other", 0.0)

    if gpu_total > 0:
        other_ratio = other_time / gpu_total

        print(
            "[profile] "
            f"GPU total: {gpu_total:.3f} ms, "
            f"other: {other_time:.3f} ms "
            f"({other_ratio * 100:.2f}%)",
            flush=True,
        )

        if other_ratio > 0.05:
            print(
                "[profile] WARNING: "
                f"'other' CUDA time is {other_ratio * 100:.2f}% "
                "of total GPU time. Inspect the kernel inventory before "
                "using this run for final runtime-breakdown results.",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Detailed kernel inventory.
    # ------------------------------------------------------------------
    records = [
        {
            "category": category,
            "operator": op_name,
            "kernel": kernel_name,
            "activity": activity_type,
            **values,
        }
        for (
            category,
            op_name,
            kernel_name,
            activity_type,
        ), values in sorted(inventory.items())
    ]

    return dict(totals), gpu_total, records