"""Cleveland dot plot of robustness vs. adversarial-training strength.

One row per adversarial-training condition (training attack ratio: 0%, 0.5%, 1%,
2%, 5%, 10%), in fixed order with the 0% baseline pinned to the top. Each dot is
the absolute macro PR-AUC at one of six test attack ratios. The x-axis is
*reversed* -- it decreases left-to-right -- so the figure still reads from best
(clean accuracy, left) to most-degraded (right), while the ticks report the actual
accuracy each row achieves. Every dot is on one shared absolute scale, so the
clean-accuracy cost of adversarial training is directly visible (e.g. the 10% row
starts slightly below the undefended baseline before any attack).

The horizontal range line per row spans that row's lowest-to-highest accuracy.

Reuses the robustness Cleveland plot's style, dot sizing and legend layout so the
two figures stay consistent, but swaps the warm (YlOrRd) ramp for a cool Blues
ramp to set this figure apart. Test attack ratio is encoded twice -- by the
light-to-dark colour ramp and by dot size.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Reuse the robustness Cleveland plot's column width, output dir and (transitively)
# its shared rcParams style so the two figures stay visually consistent.
from analysis.scripts.plot_robustness_cleveland import (
    COLUMNWIDTH_IN,
    OUTPUT_DIR,
)
from analysis.utils import load_or_fetch_runs

FIG_W = COLUMNWIDTH_IN
FIG_H = 2.0  # compact; 5 rows + top legend

# Six test attack ratios, including 0% (clean). Drops are measured against the one
# global baseline, so the 0% dot is meaningful for adversarially trained rows.
ATTACK_RATIOS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
RATIO_LABELS = ["0\\%", "0.5\\%", "1\\%", "2\\%", "5\\%", "10\\%"]

# Light-to-dark blue ramp, to set this figure apart from the robustness Cleveland
# plot's warm (YlOrRd) ramp while keeping the same encoding (darker = more attack).
# Truncated at 0.25 so the lightest 0% dot stays visible on white.
_CMAP = matplotlib.colormaps["Blues"]
RATIO_COLORS = [_CMAP(x) for x in np.linspace(0.25, 1.0, len(ATTACK_RATIOS))]

# Dot sizes (scatter ``s``, i.e. area), spaced linearly in *diameter* so the
# perceived size grows evenly from 0% to 10%. Matches the robustness Cleveland ramp.
_DIAM = np.linspace(np.sqrt(18.0), np.sqrt(85.0), len(ATTACK_RATIOS))
RATIO_SIZES = _DIAM**2

# Training conditions, top-to-bottom. 0% (undefended baseline) pinned at the top.
ADV_TRAIN_RATIOS = [0.0, 0.005, 0.01, 0.05, 0.1]
ADV_TRAIN_LABELS = ["0\\%", "0.5\\%", "1\\%", "5\\%", "10\\%"]


def _load_mean() -> pd.DataFrame:
    """Mean test macro PR-AUC, indexed by adv_train_ratio x test attack_ratio.

    Runs are selected by the shared "robustness plot" tag. The training condition
    is read from ``adv_train_params.attack_ratio`` (absent for the undefended
    baseline -> 0%); the test attack ratio from ``attack_params.attack_ratio``.
    Means are over the nine seeds (model_seed x seed) per condition.
    """
    filters = {"tags": {"$in": ["robustness plot"]}}
    run_df = load_or_fetch_runs(path="robust_multignn", filters=filters)

    config_df = pd.json_normalize(run_df["config"])
    summary_df = pd.json_normalize(run_df["summary"])

    df = pd.DataFrame(
        {
            "attack_type": config_df["attack_type"],
            "adv_train_ratio": config_df.get("adv_train_params.attack_ratio"),
            "attack_ratio": config_df["attack_params.attack_ratio"].astype(float),
            "auprc_macro": summary_df["test/auprc_macro"],
        }
    )

    # Undefended baseline runs carry no adv_train_params -> 0% adversarial training.
    df["adv_train_ratio"] = df["adv_train_ratio"].fillna(0.0).astype(float)

    # The adversarial-training grid is evaluated under PR-BCD only.
    df = df[df["attack_type"] == "prbcd"]

    grouped = df.groupby(["adv_train_ratio", "attack_ratio"], as_index=False).agg(
        auprc_mean=("auprc_macro", "mean")
    )
    return grouped.pivot(
        index="adv_train_ratio", columns="attack_ratio", values="auprc_mean"
    )


def main():
    mean = _load_mean()

    # Absolute macro PR-AUC in percent, per test attack ratio and training condition.
    # Plotted on a *descending* x-axis (high accuracy left, low right) so the figure
    # still reads left-to-right from best to most-degraded, while the ticks show the
    # actual accuracy each row achieves -- comparable across rows on one scale.
    val = {r: mean[r] * 100 for r in ATTACK_RATIOS}  # val[r] indexed by adv_train_ratio

    n = len(ADV_TRAIN_RATIOS)
    # Fixed order, no sorting: 0% on top (highest y) down to 10% at the bottom.
    y_of = {ratio: n - 1 - i for i, ratio in enumerate(ADV_TRAIN_RATIOS)}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for cond in ADV_TRAIN_RATIOS:
        y = y_of[cond]
        vals = [val[r][cond] for r in ATTACK_RATIOS]

        # Connector spans this row's lowest to highest accuracy.
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

    # Thin rule separating the 0% baseline row from the adversarially trained rows.
    ax.axhline(y_of[0.0] - 0.5, color="black", lw=0.6, zorder=1)

    # y-axis: training-ratio labels, baseline (0%) bold to stand out.
    yticks = [y_of[c] for c in ADV_TRAIN_RATIOS]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"${lab}$" for lab in ADV_TRAIN_LABELS])
    for tick, cond in zip(ax.get_yticklabels(), ADV_TRAIN_RATIOS):
        tick.set_color("black")
        if cond == 0.0:
            tick.set_fontweight("bold")
    ax.set_ylim(-0.6, n - 1 + 0.6)
    ax.set_ylabel(r"$\mathrm{adv.\ training\ attack\ ratio}$")
    ax.tick_params(axis="y", length=3)

    # x-axis: absolute macro PR-AUC, reversed so it decreases left-to-right (best
    # accuracy at the left, most-degraded at the right). Does not start at 0 -- all
    # values sit in the high 80s-to-99 range. Extra padding on the degraded (right)
    # side where the larger, darker dots live. Math-font tick labels.
    vmax = max(max(val[r]) for r in ATTACK_RATIOS)
    vmin = min(min(val[r]) for r in ATTACK_RATIOS)
    pad = 0.03 * (100.0 - vmin)
    # Anchor the left (first) tick at exactly 100; step down in 5-point ticks to the
    # 5-multiple at or below the lowest accuracy (no ticks past the data).
    ax.set_xlim(100.0 + pad, vmin - 2 * pad)  # left > right -> descending axis
    ax.set_xticks(np.arange(np.floor(vmin / 5) * 5, 100.0 + 1, 5))
    ax.set_xlabel(r"$\mathrm{macro\ PR\text{-}AUC}\ [\%]$")
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
        handletextpad=0.2,
        columnspacing=0.7,
        borderaxespad=0.0,
        fontsize=matplotlib.rcParams["legend.fontsize"] - 1,
        title="evaluation attack ratio",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "adv_training_cleveland"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
