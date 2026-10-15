from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from analysis.utils import load_or_fetch_runs

COLUMNWIDTH_IN = 248 / 72
TEXTWIDTH_IN = COLUMNWIDTH_IN
FONTSIZE = 8  # pt

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "mathtext.fontset": "cm",
        "lines.markersize": 5.5,
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.bottom": True,
        "ytick.left": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.linewidth": 0.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.2,
    }
)

OUTPUT_DIR = Path(__file__).parent.parent / "figures"

TASK_NAMES = [
    "deg_in",
    "deg_out",
    "fan_in",
    "fan_out",
    "cycle2",
    "cycle3",
    "cycle4",
    "cycle5",
    "cycle6",
    "scatter_gather",
    "biclique",
]
SERIES = TASK_NAMES + ["macro"]

TASK_LABELS = {
    "deg_in": "in-degree",
    "deg_out": "out-degree",
    "fan_in": "fan-in",
    "fan_out": "fan-out",
    "cycle2": "2-cycle",
    "cycle3": "3-cycle",
    "cycle4": "4-cycle",
    "cycle5": "5-cycle",
    "cycle6": "6-cycle",
    "scatter_gather": "scatter-gather",
    "biclique": "biclique",
    "macro": "macro avg.",
}

# One color per task family, marker shape distinguishes members within a family.
# Keeps the palette small (4 colors + black) and disambiguates similar hues.
# Two-member families use a strongly contrasting pair: upright plus ("P",
# Swiss-flag cross) and circle. Cycles 3-6 use a regular polygon whose corner
# count equals the cycle length; the 2-cycle uses a plus. Macro is a star.
TASK_STYLE = {
    "deg_in": ("C0", "P"),
    "deg_out": ("C0", "o"),
    "fan_in": ("C1", "P"),
    "fan_out": ("C1", "o"),
    "cycle2": ("C2", "P"),  # plus (no 2-sided polygon)
    "cycle3": ("C2", (3, 0, 0)),  # triangle
    "cycle4": ("C2", (4, 0, 45)),  # square
    "cycle5": ("C2", (5, 0, 0)),  # pentagon
    "cycle6": ("C2", (6, 0, 30)),  # hexagon (flat side at bottom)
    "scatter_gather": ("C3", "P"),
    "biclique": ("C3", "o"),
    "macro": ("black", "*"),
}
COLORS = {task: style[0] for task, style in TASK_STYLE.items()}
MARKERS = {task: style[1] for task, style in TASK_STYLE.items()}

SIZE_BOOST = 1.25  # overall marker enlargement
ALPHA = 0.8  # marker transparency in the plot (legend stays opaque)


def _equal_area(marker, base_area):
    """Scale scatter ``s`` so every marker has ~the same *visual* filled area.

    matplotlib's ``s`` is the bounding-box area, but shapes fill that box to
    different fractions (a triangle ~0.32, a circle ~0.79), so equal ``s`` looks
    unequal. Dividing by the fill fraction (normalised to the circle) compensates.
    """
    if isinstance(marker, tuple):  # regular polygon (n_sides, 0, angle)
        n = marker[0]
        frac = 0.125 * n * np.sin(2 * np.pi / n)
    else:
        frac = {"o": 0.785, "P": 0.556, "*": 0.33}.get(marker, 0.785)
    return base_area * (0.785 / frac)


ATTACK_TYPES = ["uniform", "prbcd"]
PLOT_RATIOS = [0.05, 0.1]


def _aggregate(run_df: pd.DataFrame) -> pd.DataFrame:
    config_df = pd.json_normalize(run_df["config"])
    summary_df = pd.json_normalize(run_df["summary"])

    base = pd.DataFrame(
        {
            "attack_type": config_df["attack_type"],
            "attack_ratio": config_df["attack_params.attack_ratio"].astype(float),
        }
    )
    for task in SERIES:
        base[task] = summary_df[f"test/auprc_{task}"]

    long_df = base.melt(
        id_vars=["attack_type", "attack_ratio"],
        value_vars=SERIES,
        var_name="task",
        value_name="auprc",
    )
    return (
        long_df.groupby(["attack_type", "attack_ratio", "task"], as_index=False)
        .agg(auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"))
        .fillna({"auprc_std": 0.0})
    )


def _compute_drops(
    agg: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Returns {attack_type: (mean_df, std_df)} in percentage points.

    std propagated via sqrt(σ_baseline² + σ_ratio²).
    """
    result = {}
    for attack_type in ATTACK_TYPES:
        sub = agg[agg["attack_type"] == attack_type]
        mean_piv = sub.pivot(index="attack_ratio", columns="task", values="auprc_mean")[
            SERIES
        ]
        std_piv = sub.pivot(index="attack_ratio", columns="task", values="auprc_std")[
            SERIES
        ]

        baseline_mean = mean_piv.loc[0.0]
        baseline_std = std_piv.loc[0.0]

        drop_mean = (baseline_mean - mean_piv).loc[PLOT_RATIOS] * 100
        drop_std = (np.sqrt(baseline_std**2 + std_piv**2)).loc[PLOT_RATIOS] * 100
        result[attack_type] = (drop_mean, drop_std)
    return result


def _draw_panel(ax, ratio, drops, lim, ticks, show_ylabel):
    base_area = matplotlib.rcParams["lines.markersize"] ** 2 * SIZE_BOOST
    macro_size = (
        _equal_area(MARKERS["macro"], base_area) * 1.3
    )  # highlight a touch larger

    naive_mean, naive_std = drops["uniform"]
    prbcd_mean, prbcd_std = drops["prbcd"]

    n_mean = naive_mean.loc[ratio]
    n_std = naive_std.loc[ratio]
    p_mean = prbcd_mean.loc[ratio]
    p_std = prbcd_std.loc[ratio]

    handles = []
    for task in TASK_NAMES:
        color = COLORS[task]
        h = ax.scatter(
            n_mean[task],
            p_mean[task],
            color=color,
            marker=MARKERS[task],
            s=_equal_area(MARKERS[task], base_area),
            label=TASK_LABELS[task],
            zorder=3,
            alpha=ALPHA,
            linewidths=0,
        )
        handles.append(h)

    h_macro = ax.scatter(
        n_mean["macro"],
        p_mean["macro"],
        color=COLORS["macro"],
        marker=MARKERS["macro"],
        s=macro_size,
        label=TASK_LABELS["macro"],
        zorder=4,
        linewidths=0,
    )
    handles.append(h_macro)

    ax.plot([0, lim], [0, lim], "--", color="gray", zorder=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"${v:g}$"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"${v:g}$"))

    # Restore the full box (seaborn-whitegrid drops the top/right spines)
    for spine in ax.spines.values():
        spine.set_visible(True)

    if show_ylabel:
        ax.set_ylabel(r"$\mathrm{PR\text{-}BCD\ (our\ adaptation)}$")

    # Attack ratio label below the diagonal, in the lower-right empty region
    ax.text(
        0.97,
        0.04,
        rf"${ratio * 100:g}\%\ \mathrm{{attack\ ratio}}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONTSIZE - 1,
        color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
    )

    return handles


def main():
    filters = {"tags": {"$in": ["attack effectiveness plot"]}}
    run_df = load_or_fetch_runs(path="robust_multignn", filters=filters)
    agg = _aggregate(run_df)
    drops = _compute_drops(agg)

    all_means = pd.concat([m for m, _ in drops.values()])
    max_drop = float(all_means.max().max())
    lim = max_drop * 1.08
    step = 5 if lim <= 35 else 10
    ticks = np.arange(0, lim, step)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.62),
        sharey=True,
    )

    legend_handles = None
    for idx, (ax, ratio) in enumerate(zip(axes, PLOT_RATIOS)):
        handles = _draw_panel(ax, ratio, drops, lim, ticks, show_ylabel=(idx == 0))
        if legend_handles is None:
            legend_handles = handles

    # sharey hides right-panel y-tick labels automatically; suppress the label too
    axes[1].set_ylabel("")

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(wspace=0.06, bottom=0.30, top=0.86)
    fig.canvas.draw()  # finalize axes positions (square aspect shrinks the box)

    # Center over the two panels (not the whole figure, which is offset by the
    # y-axis label/ticks on the left).
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    x_center = (pos0.x0 + pos1.x1) / 2

    # Single x-axis label just below the tick labels, legend below that.
    fig.text(
        x_center,
        pos0.y0 - 0.13,
        r"$\mathrm{Uniform\ (naive)}$",
        ha="center",
        va="center",
    )

    # Shared PR-AUC metric label above both panels (larger than the axis labels
    # since it applies to both x and y).
    fig.text(
        x_center,
        pos0.y1 + 0.05,
        r"$\mathrm{PR\text{-}AUC}\,\downarrow\ [\%]$",
        ha="center",
        va="center",
        fontsize=FONTSIZE + 2,
    )

    legend = fig.legend(
        handles=legend_handles,
        labels=[TASK_LABELS[t] for t in SERIES],
        loc="upper center",
        bbox_to_anchor=(0.5, pos0.y0 - 0.15),
        frameon=False,
        ncol=4,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "attack_robustness_scatter"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(
        stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01, transparent=True
    )
    plt.show()


if __name__ == "__main__":
    main()
