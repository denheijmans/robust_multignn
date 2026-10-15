"""Cleveland dot plot of per-task robustness under the PR-BCD attack.

One row per subtask (+ macro average), showing the absolute PR-AUC at six attack
ratios (including 0%, the unattacked accuracy). The x-axis is *reversed* -- it
decreases from 100 left-to-right -- so the figure reads from best (clean accuracy,
left) to most-degraded (right). Rows are ordered by robustness (sensitivity to the
attack, i.e. the PR-AUC drop at 10%); the horizontal range line per row spans its
lowest-to-highest accuracy, so its length encodes that drop. Attack ratio is
encoded twice -- by a light-to-dark sequential colour ramp and by dot size -- so
the progression stays legible in print and in grayscale.

Reads the same W&B runs as the attack-robustness scatter (shared tag and
aggregation) so the two figures are guaranteed to use identical data.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

# Reuse the scatter's data handling, task ordering and labels so the figures stay
# consistent. Importing also applies the shared rcParams style.
from analysis.scripts.plot_attack_robustness_scatter import (
    OUTPUT_DIR,
    SERIES,
    TASK_LABELS,
    TASK_NAMES,
    _aggregate,
)
from analysis.utils import load_or_fetch_runs

COLUMNWIDTH_IN = 248 / 72
FIG_W = COLUMNWIDTH_IN
FIG_H = 3.5  # compact; 12 rows + top legend

# Excluded from this figure only (kept in the shared scatter plot).
EXCLUDED_TASKS = {"deg_out", "fan_out"}
ROWS = [t for t in TASK_NAMES if t not in EXCLUDED_TASKS]

ATTACK_RATIOS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
RATIO_LABELS = ["0\\%", "0.5\\%", "1\\%", "2\\%", "5\\%", "10\\%"]

# Light-to-dark warm ramp (yellow -> dark red), truncated so the lightest 0%
# dot stays visible on white. Darker = more attack = lower accuracy.
_CMAP = matplotlib.colormaps["YlOrRd"]
RATIO_COLORS = [_CMAP(x) for x in np.linspace(0.25, 1.0, len(ATTACK_RATIOS))]

# Dot sizes (scatter ``s``, i.e. area). Spaced linearly in *diameter* so the
# perceived size grows evenly from 0% to 10%.
_DIAM = np.linspace(np.sqrt(18.0), np.sqrt(85.0), len(ATTACK_RATIOS))
RATIO_SIZES = _DIAM**2


def _load_mean() -> "object":
    """Mean test PR-AUC, indexed by attack_ratio, columns = SERIES (prbcd only)."""
    filters = {"tags": {"$in": ["attack effectiveness plot"]}}
    run_df = load_or_fetch_runs(path="robust_multignn", filters=filters)
    agg = _aggregate(run_df)
    prbcd = agg[agg["attack_type"] == "prbcd"]
    return prbcd.pivot(index="attack_ratio", columns="task", values="auprc_mean")[
        SERIES
    ]


def main():
    mean = _load_mean()
    baseline = mean.loc[0.0]

    # Absolute PR-AUC in percent, per ratio and task (what we plot).
    val = {r: mean.loc[r] * 100 for r in ATTACK_RATIOS}

    # Drop at 10% (pp below each task's own clean accuracy) -- used only to order
    # rows by robustness; the range-line length then mirrors this drop.
    drop10 = (baseline - mean.loc[0.1]) * 100

    # Sort most-robust to least-robust (smallest drop at 10% at the top).
    ordered = drop10[ROWS].sort_values(ascending=True).index.tolist()

    n = len(ordered)
    y_of = {task: n - 1 - i for i, task in enumerate(ordered)}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for task in ordered:
        y = y_of[task]
        vals = [val[r][task] for r in ATTACK_RATIOS]

        # Connector spans this task's lowest to highest accuracy.
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [y, y], ls="-", color="0.40", lw=0.8, zorder=1)

        # Largest (darkest) dot first so the smaller, lighter dots stay on top
        # where they overlap.
        for j in reversed(range(len(ATTACK_RATIOS))):
            ax.scatter(
                vals[j],
                y,
                s=RATIO_SIZES[j],
                color=RATIO_COLORS[j],
                edgecolors="none",
                zorder=3,
            )

    # y-axis: black task labels; no axis label.
    yticks = [y_of[t] for t in ordered]
    ax.set_yticks(yticks)
    ax.set_yticklabels([TASK_LABELS[t] for t in ordered])
    for tick in ax.get_yticklabels():
        tick.set_color("black")
    ax.set_ylim(-0.6, n - 1 + 0.6)
    ax.tick_params(axis="y", length=3)

    # x-axis: absolute PR-AUC, reversed so it decreases left-to-right (best accuracy
    # at the left, most-degraded at the right). First tick anchored at exactly 100;
    # 5-point ticks down to the 5-multiple at or below the lowest accuracy. Extra
    # padding on the degraded (right) side where the larger, darker dots live.
    vmin = min(min(val[r][ROWS]) for r in ATTACK_RATIOS)
    pad = 0.03 * (100.0 - vmin)
    ax.set_xlim(100.0 + pad, vmin - 2 * pad)  # left > right -> descending axis
    ax.set_xticks(np.arange(np.floor(vmin / 5) * 5, 100.0 + 1, 5))
    ax.set_xlabel(r"$\mathrm{PR\text{-}AUC}\ [\%]$")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"${v:g}$"))

    # Minimal grid: keep vertical guides only.
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    # Legend (above): the six ratio dots (colour + size).
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=RATIO_COLORS[j],
            markeredgecolor="none",
            markersize=np.sqrt(RATIO_SIZES[j]),
            label=f"${RATIO_LABELS[j]}$",
        )
        for j in range(len(ATTACK_RATIOS))
    ]

    fig.tight_layout()
    fig.canvas.draw()  # finalize axes position so we can center over the box
    pos = ax.get_position()
    x_center = (pos.x0 + pos.x1) / 2
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(x_center, 0.975),
        ncol=6,
        frameon=False,
        handletextpad=0.1,
        columnspacing=0.25,
        borderaxespad=0.0,
        fontsize=matplotlib.rcParams["legend.fontsize"] - 1,
        title="evaluation attack ratio",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "robustness_cleveland"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
