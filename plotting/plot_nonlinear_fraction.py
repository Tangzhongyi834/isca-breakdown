#!/usr/bin/env python3
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting.common import LINE_COLORS, PRECISIONS, load_rows, missing_note, parser, plt, save, style


def main(argv=None):
    args = parser("Figure 1(b): measured nonlinear fraction versus sequence length").parse_args(argv)
    rows, models, incomplete = load_rows(args)
    style()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 3.7), squeeze=False, sharey=True)
    lookup = {(row["model"], row["precision"], row["seq_len"]): row for row in rows}
    for ax, model in zip(axes[0], models):
        for precision in PRECISIONS:
            values = [lookup.get((model, precision, length), {}).get("nonlinear_pct", float("nan"))
                      for length in args.seq_lens]
            # NaN breaks a line at missing measurements; it never invents a value.
            ax.plot(args.seq_lens, values, marker="o", color=LINE_COLORS[precision], label=precision)
        ax.set_xscale("log", base=2)
        ax.set_xticks(args.seq_lens, args.seq_lens)
        ax.set_xlabel("Sequence Length")
        ax.set_title(model.rsplit("/", 1)[-1])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    axes[0, 0].set_ylabel("Nonlinear Runtime Fraction (%)")
    missing_note(fig, incomplete)
    fig.tight_layout()
    save(fig, args.output_dir, "fig1b_nonlinear_fraction")


if __name__ == "__main__":
    main()
