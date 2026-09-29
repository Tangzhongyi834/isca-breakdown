"""Hierarchical medians with additive totals recomputed from leaf categories."""

import csv
from collections import defaultdict
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import statistics

from .runtime_profiler import LEAF_CATEGORIES, derive_metrics

LEAF_FIELDS = tuple(f"{name}_ms" for name in LEAF_CATEGORIES)
INDEPENDENT_FIELDS = ("physical_total_ms", "profiled_physical_ms", "gpu_kernel_total_ms",
                      "gpu_activity_total_ms", "consistency_error_pct")
IDENTITY_FIELDS = ("model", "precision", "seq_len", "measurement_kind", "validation_status",
                   "fairness_id", "token_sha256", "run_id")


def reduce_measurements(rows):
    if not rows:
        raise ValueError("No measurements to reduce")
    values = {field: statistics.median(float(row[field]) for row in rows)
              for field in (*LEAF_FIELDS, *INDEPENDENT_FIELDS)}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Invalid measurement value")
    return derive_metrics(values)


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def statistics_for(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0,
                min=min(values), max=max(values), p10=percentile(values, .1), p90=percentile(values, .9))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty results CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


@contextmanager
def results_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".results.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_summaries(raw_rows):
    groups = defaultdict(list)
    fairness = defaultdict(set)
    for row in raw_rows:
        if row.get("measurement_kind") != "measured" or row.get("validation_status") != "passed":
            raise ValueError("Projected/unvalidated data cannot enter paper summaries")
        key = (row["model"], row["precision"], int(row["seq_len"]))
        groups[key].append(row)
        fairness[row["model"]].add(row["fairness_id"])
    if any(len(ids) != 1 for ids in fairness.values()):
        raise ValueError("Incompatible experimental settings/token samples; use a separate --output-dir")
    summary, overhead, figures, stats = [], [], [], []
    for (model, precision, seq_len), rows in sorted(groups.items()):
        ids = [int(row["sample_id"]) for row in rows]
        if sorted(ids) != list(range(len(rows))):
            raise ValueError("Missing or duplicate sample IDs")
        for name in IDENTITY_FIELDS:
            if len({str(row[name]) for row in rows}) != 1:
                raise ValueError(f"Conflicting {name} within a configuration")
        metadata = {name: rows[0][name] for name in IDENTITY_FIELDS}
        metadata["seq_len"] = seq_len
        reduced = reduce_measurements(rows)
        summary.append({**metadata, "num_samples": len(rows), **reduced})
        overhead.append({**metadata, **{name: reduced[name] for name in LEAF_FIELDS
                                       if name.removesuffix("_ms") in ("quantize", "dequantize", "requantize", "scale", "packing", "conversion")},
                         "total_quantization_overhead_ms": reduced["quantization_overhead_ms"]})
        fields = ("linear_ms", "nonlinear_ms", "other_ms", "compute_total_ms", "linear_pct", "nonlinear_pct", "other_pct")
        figures.append({**metadata, **{field: reduced[field] for field in fields}})
        for metric in reduced:
            values = [float(row[metric]) for row in rows]
            stats.append({**metadata, "num_samples": len(rows), "metric": metric, **statistics_for(values)})
    return summary, overhead, figures, stats


def publish_configuration(root, raw_path, metadata):
    root = Path(root)
    with results_lock(root):
        index_path = root / "index.json"
        index = json.loads(index_path.read_text()) if index_path.exists() else {}
        key = json.dumps([metadata["model"], metadata["precision"], metadata["seq_len"]])
        candidate = {**index, key: str(Path(raw_path).resolve())}
        rows = [row for path in candidate.values() for row in read_csv(path)]
        summary, overhead, figures, stats = build_summaries(rows)
        write_csv(root / "raw" / "runtime_samples.csv", rows)
        write_csv(root / "summary" / "runtime_by_length.csv", summary)
        write_csv(root / "summary" / "quantization_overhead.csv", overhead)
        write_csv(root / "summary" / "figure1_data.csv", figures)
        write_csv(root / "summary" / "runtime_statistics.csv", stats)
        write_json(index_path, candidate)
        environments = {}
        for path in candidate.values():
            run_dir = Path(path).parent.parent
            env = json.loads((run_dir / "environment.json").read_text())
            environments[env["run_id"]] = env
        write_json(root / "metadata" / "environment.json", {"schema_version": 1, "runs": list(environments.values())})
    return next(row for row in summary if row["model"] == metadata["model"] and
                row["precision"] == metadata["precision"] and int(row["seq_len"]) == metadata["seq_len"])
