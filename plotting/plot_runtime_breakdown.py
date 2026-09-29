#!/usr/bin/env python3
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting.common import COLORS, PRECISIONS, load_rows, missing_note, parser, plt, save, style


def main(argv=None):
    args = parser("Figure 1(a): measured compute runtime composition").parse_args(argv)
    rows, models, incomplete = load_rows(args)
    style()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 4), squeeze=False, sharey=True)
    lookup = {(row["model"], row["precision"], row["seq_len"]): row for row in rows}
    for ax, model in zip(axes[0], models):
        ticks, labels = [], []
        for index, seq_len in enumerate(args.seq_lens):
            for offset, precision in enumerate(PRECISIONS):
                x = index * 4 + offset
                ticks.append(x)
                labels.append(precision)
                row = lookup.get((model, precision, seq_len))
                if row is None:
                    ax.text(x, 5, "N/A", ha="center", va="bottom", fontsize=8, rotation=90)
                    continue
                bottom = 0
                for category in ("Linear", "Nonlinear", "Other"):
                    value = row[category.lower() + "_pct"]
                    ax.bar(x, value, bottom=bottom, width=.82, color=COLORS[category], label=category)
                    bottom += value
            ax.text(index * 4 + 1, -.13, str(seq_len), transform=ax.get_xaxis_transform(), ha="center")
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.set_xlim(-.7, (len(args.seq_lens) - 1) * 4 + 2.7)
        ax.set_title(model.rsplit("/", 1)[-1])
        ax.set_ylim(0, 100)
        ax.set_xlabel("Sequence Length", labelpad=30)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("Compute Runtime Fraction (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[key]) for key in COLORS]
    fig.legend(handles, list(COLORS), ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.06), frameon=False)
    missing_note(fig, incomplete)
    fig.tight_layout()
    save(fig, args.output_dir, "fig1a_runtime_breakdown")


if __name__ == "__main__":
    main()
