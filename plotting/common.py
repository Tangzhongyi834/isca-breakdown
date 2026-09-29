import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRECISIONS = ("FP16", "W8A8", "W4A4")
COLORS = {"Linear": "#547AA5", "Nonlinear": "#C55A3A", "Other": "#B7BCC2"}
LINE_COLORS = {"FP16": "#547AA5", "W8A8": "#C55A3A", "W4A4": "#52856D"}


def parser(description):
    cli = argparse.ArgumentParser(description=description)
    cli.add_argument("--input", default="results/summary/figure1_data.csv")
    cli.add_argument("--output-dir", default="figures")
    cli.add_argument("--model", action="append", help="Filter a model (repeatable)")
    cli.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    cli.add_argument("--allow-incomplete", action="store_true", help="Show missing measurements explicitly")
    return cli


def load_rows(args):
    with Path(args.input).open(newline="") as handle:
        source = list(csv.DictReader(handle))
    rows, seen, fairness = [], set(), {}
    for row in source:
        if args.model and row.get("model") not in args.model:
            continue
        if int(row["seq_len"]) not in args.seq_lens:
            continue
        if row.get("measurement_kind") != "measured" or row.get("validation_status") != "passed":
            raise ValueError("Figures require validated measured data; projected/unresolved rows are forbidden")
        if row["precision"] not in PRECISIONS or not row.get("fairness_id") or not row.get("token_sha256"):
            raise ValueError("Missing experimental identity or unsupported precision")
        identity = (row["fairness_id"], row["token_sha256"])
        if row["model"] in fairness and fairness[row["model"]] != identity:
            raise ValueError("Different precisions must share samples and experimental settings")
        fairness[row["model"]] = identity
        row["seq_len"] = int(row["seq_len"])
        key = (row["model"], row["precision"], row["seq_len"])
        if key in seen:
            raise ValueError(f"Duplicate measurement: {key}")
        seen.add(key)
        if "other_ms" not in row:
            row["other_ms"] = float(row["attention_matmul_ms"]) + float(row["other_compute_ms"])
        for field in ("linear_ms", "nonlinear_ms", "other_ms", "compute_total_ms", "linear_pct", "nonlinear_pct", "other_pct"):
            row[field] = float(row[field])
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError(f"Invalid measured field: {field}")
        total = row["compute_total_ms"]
        if total <= 0 or not math.isclose(sum(row[f"{name}_ms"] for name in ("linear", "nonlinear", "other")), total, rel_tol=1e-6):
            raise ValueError("Runtime totals are inconsistent")
        for name in ("linear", "nonlinear", "other"):
            if not math.isclose(row[f"{name}_pct"], row[f"{name}_ms"] / total * 100, abs_tol=1e-5):
                raise ValueError("Fractions must use compute runtime, excluding quantization")
        rows.append(row)
    if not rows:
        raise ValueError("No measurements match the requested model/length")
    models = sorted({row["model"] for row in rows})
    if args.model and set(args.model) - set(models):
        raise ValueError("A requested model has no measurements")
    missing = [(m, p, length) for m in models for p in PRECISIONS for length in args.seq_lens
               if (m, p, length) not in seen]
    if missing and not args.allow_incomplete:
        raise ValueError(f"Incomplete measured matrix ({len(missing)} missing). Use --allow-incomplete to label missing data.")
    return rows, models, bool(missing)


def style():
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})


def save(fig, output_dir, name):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def missing_note(fig, incomplete):
    if incomplete:
        fig.text(.5, -.03, "Incomplete measured matrix: unavailable configurations have no measured values.",
                 ha="center", fontsize=9)
