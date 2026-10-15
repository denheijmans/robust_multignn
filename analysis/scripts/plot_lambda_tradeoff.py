"""Dual-axis tradeoff plot for the cycle-loss regularizer weight sweep.

A single panel at a fixed attack ratio (10%) showing, as the attacker's
cycle-consistency regularizer weight lambda grows:

  * left axis  -- the C2, C3 and macro PR-AUC [%] (three lines), and
  * right axis -- the sqrt of the cycle loss (coral).

All metrics are lines with dots and per-point +/- std bands (across the
9 runs = 3 model seeds x 3 data seeds). lambda is treated as categorical so the
logarithmically spaced values (0, 0.002, 0.01, 0.05, 0.2, 1) are evenly spaced.

The left axis carries three curves: C2 (teal) and C3 (indigo) are the per-cycle
detail lines, and macro (neutral dark grey, slightly heavier) is the aggregate
"headline" curve. They are told apart by an in-axes legend whose entry text is
coloured to match each line. The right axis (line, label and tick colour) is
coloured coral to match the cycle-loss metric. The left axis auto-ranges to the
PR-AUC data; the right axis starts at 0. Note the two axes therefore encode
"better" in opposite screen directions (higher PR-AUC vs lower cycle loss).

Loads the same way as the rest of the codebase, via a single W&B tag.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from analysis.utils import load_or_fetch_runs

COLUMNWIDTH_IN = 248 / 72
FONTSIZE = 8  # pt

# Match the attack-robustness scatter / cleveland style: white background with
# light grey gridlines (seaborn whitegrid) and a full black box around the axes.
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

ATTACK_RATIO = 0.1  # representative attack ratio for the panel

# All three left-axis PR-AUC lines share one teal hue at slightly different
# brightnesses, so they read as a group belonging to the (teal) left axis; they
# are told apart by their marker symbol. Darkest -> 2-cycle, ... lightest -> macro.
C2_COLOR = "#46b3bd"  # light teal   -- 2-cycle PR-AUC
C3_COLOR = "#1c8a94"  # mid teal     -- 3-cycle PR-AUC
MACRO_COLOR = "#0d5a61"  # dark teal  -- macro avg. PR-AUC
CYCLE_COLOR = "#d95f02"  # muted coral -- right axis (sqrt cycle loss)

# Marker symbol per line (filled plus / triangle / circle). "P" is the thick
# filled plus (matching the attack-robustness scatter), not the thin "+".
C2_MARKER = "P"
C3_MARKER = "^"
MACRO_MARKER = "o"


def _equal_area_ms(marker: str, base_ms: float) -> float:
    """Markersize that gives ``marker`` ~the same filled area as a circle.

    ``plot``'s ``markersize`` is a diameter (area ~ ms^2), and shapes fill their
    bounding box to different fractions, so equal markersize looks unequal. Scale
    by sqrt(fill fraction ratio), the diameter analogue of the scatter's
    ``_equal_area`` (which works in area units). Fractions match the scatter.
    """
    frac = {"o": 0.785, "P": 0.556, "^": 0.5, "s": 1.0}.get(marker, 0.785)
    return base_ms * np.sqrt(0.785 / frac)

# Teal used for the left axis label so it matches the line group.
PRAUC_LABEL_COLOR = "#13666d"  # legible teal for the left y-axis label
CYCLE_LABEL_COLOR = "#cc5902"  # marginally darker coral
BAND_ALPHA = 0.10  # transparency of the +/- std band around each curve (3 overlap)


def _load_runs() -> pd.DataFrame:
    """Per-run C2/C3 PR-AUC, cycle loss and lambda at the chosen attack ratio."""
    filters = {"tags": {"$in": ["weight sweep plot"]}}
    run_df = load_or_fetch_runs(path="robust_multignn", filters=filters)

    config_df = pd.json_normalize(run_df["config"])
    summary_df = pd.json_normalize(run_df["summary"])

    df = pd.DataFrame(
        {
            "cycle_weight": config_df.get(
                "attack_params.regularizer_terms.cycle_loss"
            ),
            "attack_ratio": config_df["attack_params.attack_ratio"].astype(float),
            "auprc_cycle2": summary_df["test/auprc_cycle2"],
            "auprc_cycle3": summary_df["test/auprc_cycle3"],
            "auprc_macro": summary_df["test/auprc_macro"],
            "cycle_loss": summary_df["test/cycle_loss"],
        }
    )
    # Keep only cycle-loss sweep runs (have a regularizer weight) at this ratio.
    df = df.dropna(subset=["cycle_weight"]).copy()
    df["cycle_weight"] = df["cycle_weight"].astype(float)
    df = df[np.isclose(df["attack_ratio"], ATTACK_RATIO)].copy()
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per lambda with the means/stds needed for both lines."""
    grouped = (
        df.groupby("cycle_weight", as_index=False)
        .agg(
            c2_mean=("auprc_cycle2", "mean"),
            c2_std=("auprc_cycle2", "std"),
            c3_mean=("auprc_cycle3", "mean"),
            c3_std=("auprc_cycle3", "std"),
            macro_mean=("auprc_macro", "mean"),
            macro_std=("auprc_macro", "std"),
            cycle_loss_mean=("cycle_loss", "mean"),
            cycle_loss_std=("cycle_loss", "std"),
        )
        .fillna(
            {
                "c2_std": 0.0,
                "c3_std": 0.0,
                "macro_std": 0.0,
                "cycle_loss_std": 0.0,
            }
        )
        .sort_values("cycle_weight")
        .reset_index(drop=True)
    )
    return grouped


def _lambda_label(w: float) -> str:
    return rf"${w:g}$"


def main():
    df = _load_runs()
    grouped = _aggregate(df)

    weights = grouped["cycle_weight"].tolist()
    x = np.arange(len(weights))

    fig, ax_l = plt.subplots(figsize=(3.5, 3.0))
    ax_r = ax_l.twinx()

    # --- Left axis: 2-cycle, 3-cycle and macro PR-AUC [%], each with a band. ---
    # Same teal hue (different brightness) + a distinct marker per line.
    def _plot_prauc(mean_key, std_key, color, marker, *, zorder, label):
        mean = grouped[mean_key].values * 100
        std = grouped[std_key].values * 100
        ax_l.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=zorder,
        )
        (line,) = ax_l.plot(
            x,
            mean,
            marker=marker,
            markersize=_equal_area_ms(marker, matplotlib.rcParams["lines.markersize"]),
            color=color,
            linestyle="-",
            linewidth=1.4,
            zorder=zorder + 1,
            label=label,
        )
        return line

    c2_line = _plot_prauc(
        "c2_mean", "c2_std", C2_COLOR, C2_MARKER, zorder=3, label=r"$2\text{-cycle}$"
    )
    c3_line = _plot_prauc(
        "c3_mean", "c3_std", C3_COLOR, C3_MARKER, zorder=4, label=r"$3\text{-cycle}$"
    )
    macro_line = _plot_prauc(
        "macro_mean", "macro_std", MACRO_COLOR, MACRO_MARKER, zorder=5,
        label=r"$\mathrm{macro\ avg.}$",
    )

    # --- Right axis: sqrt cycle loss, sqrt-transformed +/- std band. ---
    cl_mean = grouped["cycle_loss_mean"].values
    cl_std = grouped["cycle_loss_std"].values
    cl_point = np.sqrt(np.clip(cl_mean, 0, None))
    cl_lower = np.sqrt(np.clip(cl_mean - cl_std, 0, None))
    cl_upper = np.sqrt(np.clip(cl_mean + cl_std, 0, None))
    ax_r.fill_between(
        x,
        cl_lower,
        cl_upper,
        color=CYCLE_COLOR,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=2,
    )
    ax_r.plot(x, cl_point, marker="o", color=CYCLE_COLOR, linestyle="-", zorder=3)

    # --- x axis: categorical lambda values. ---
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([_lambda_label(w) for w in weights])
    ax_l.set_xlabel(r"$\mathrm{Cycle\ Loss\ Weight}\ \lambda$")
    ax_l.set_xlim(-0.3, len(weights) - 0.7)

    # --- Left y axis styling: teal label (matches the line group), auto-range. ---
    ax_l.set_ylabel(r"$\mathrm{PR\text{-}AUC}\ [\%]$")
    ax_l.yaxis.label.set_color(PRAUC_LABEL_COLOR)
    ax_l.tick_params(axis="y", color="black", labelcolor="black")
    ax_l.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"${v:g}$"))

    # In-axes legend with line+marker handles; the markers carry the meaning since
    # the three lines share a hue. Nudged in from the left spine. Light translucent
    # frame so it stays legible over the gridlines.
    legend = ax_l.legend(
        handles=[c2_line, c3_line, macro_line],
        loc="upper left",
        bbox_to_anchor=(0.14, 0.99),
        frameon=True,
        framealpha=0.85,
        edgecolor="0.6",
        fancybox=False,
        borderpad=0.5,
        labelspacing=0.35,
    )
    legend.get_frame().set_linewidth(0.6)
    for text, color in zip(
        legend.get_texts(), [C2_COLOR, C3_COLOR, MACRO_COLOR]
    ):
        text.set_color(color)

    # --- Right y axis styling (coral): starts at 0. ---
    ax_r.set_ylabel(r"$\sqrt{\mathrm{Cycle\ Loss}}$")
    ax_r.yaxis.label.set_color(CYCLE_LABEL_COLOR)
    ax_r.tick_params(axis="y", color="black", labelcolor="black")
    ax_r.set_ylim(0, cl_upper.max() * 1.08)
    ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"${v:g}$"))

    # Mark the selected operating point: lambda = 0.05 is the strongest attack
    # (lowest C2/C3 PR-AUC) while the cycle loss has already reached its floor.
    sel = int(np.argmin(np.abs(np.asarray(weights) - 0.05)))
    ax_l.axvline(sel, color="0.4", linestyle="--", linewidth=0.9, zorder=1.5)
    ax_l.annotate(
        r"$\mathrm{selected}\ \lambda = 0.05$",
        xy=(sel, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FONTSIZE - 1,
        color="0.25",
    )

    # Grid from the left axis only (the right twin would otherwise draw a second,
    # misaligned set of horizontal lines). Restore the full box that seaborn's
    # whitegrid style drops, so the edges match the scatter/cleveland figures.
    ax_l.grid(True)
    ax_r.grid(False)
    ax_l.set_axisbelow(True)
    for spine in ax_l.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax_r.spines["right"].set_visible(True)
    ax_r.spines["right"].set_color("black")

    fig.tight_layout(pad=0.4)

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / "lambda_tradeoff"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.01, transparent=True)
    plt.show()


if __name__ == "__main__":
    main()
