#!/usr/bin/env python3
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting.common import PRECISIONS, load_rows, missing_note, parser, plt, save, style


def main(argv=None):
    cli = parser("Figure 1(c): measured cross-model nonlinear fractions")
    cli.add_argument("--seq-len", type=int, default=4096)
    args = cli.parse_args(argv)
    args.seq_lens = [args.seq_len]
    rows, models, incomplete = load_rows(args)
    if len(models) < 2:
        raise ValueError("Cross-model comparison requires at least two measured models")
    style()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    lookup = {(row["model"], row["precision"]): row for row in rows}
    width = .8 / len(models)
    for index, model in enumerate(models):
        for column, precision in enumerate(PRECISIONS):
            x = column + (index - (len(models) - 1) / 2) * width
            row = lookup.get((model, precision))
            if row is None:
                ax.text(x, 1, "N/A", ha="center", fontsize=8, rotation=90)
            else:
                ax.bar(x, row["nonlinear_pct"], width, color=f"C{index}",
                       label=model.rsplit("/", 1)[-1])
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False)
    ax.set_xticks(range(3), PRECISIONS)
    ax.set_ylabel("Nonlinear Runtime Fraction (%)")
    ax.set_title(f"Sequence Length = {args.seq_len}")
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    missing_note(fig, incomplete)
    fig.tight_layout()
    save(fig, args.output_dir, "fig1c_cross_model")


if __name__ == "__main__":
    main()
