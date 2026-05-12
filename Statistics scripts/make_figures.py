"""Generate all 9 figures for the thesis from the artefacts produced by
``scripts/run_statistics.py``.

Run from the project root:

    python scripts/make_figures.py

Outputs are written as PDFs to ``figures/`` next to the project root:

    bootstrap_density.pdf
    forest_plot.pdf
    per_field_heatmap.pdf
    per_field_bars.pdf
    per_manual_distribution.pdf
    cost_accuracy_pareto.pdf
    coverage_vs_error.pdf
    mcnemar_contingency_grid.pdf
    abstention_by_field.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from accuracy import PARAMETER_NAMES, EXCLUDED_KEYS  # noqa: E402

# ----------------------------------------------------------------------
# Paths and globals
# ----------------------------------------------------------------------

RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

ARTEFACTS = {
    "long":  RESULTS_DIR / "predictions_long.csv",
    "boot":  RESULTS_DIR / "bootstrap_results.csv",
    "dist":  RESULTS_DIR / "bootstrap_distributions.json",
    "mcn":   RESULTS_DIR / "mcnemar_results.csv",
}

METHOD_ORDER = [
    "naive_user", "configurator_user", "engineered_prompt",
    "naive_rag", "modular_rag",
]

# tab10-derived colour-blind-safe palette, fixed across all figures.
_TAB10 = plt.get_cmap("tab10")
METHOD_COLOURS: dict[str, tuple] = {
    "naive_user":        _TAB10(0),  # blue
    "configurator_user": _TAB10(1),  # orange
    "engineered_prompt": _TAB10(2),  # green
    "naive_rag":         _TAB10(3),  # red
    "modular_rag":       _TAB10(4),  # purple
}

# Field display order (13 scored fields).
SCORED_FIELDS = [name for key, name in PARAMETER_NAMES.items() if key not in EXCLUDED_KEYS]

# Cost (USD per 51 manuals) from usage_logger.py.
METHOD_COST_USD = {
    "naive_user":        1.24,
    "configurator_user": 1.22,
    "engineered_prompt": 1.57,
    "naive_rag":         1.03,
    "modular_rag":       1.16,
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11


# ----------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------

def load_artefacts() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    missing = [str(p) for p in ARTEFACTS.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "ERROR: data artefacts not found. Run `python scripts/run_statistics.py` first.\n"
            "Missing: " + ", ".join(missing)
        )
    long_df = pd.read_csv(ARTEFACTS["long"])
    boot_df = pd.read_csv(ARTEFACTS["boot"])
    with ARTEFACTS["dist"].open(encoding="utf-8") as fh:
        dist = {k: np.array(v, dtype=np.float64) for k, v in json.load(fh).items()}
    mcn_df = pd.read_csv(ARTEFACTS["mcn"])
    return long_df, boot_df, dist, mcn_df


def kde_eval(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Lightweight Gaussian KDE without scipy."""
    n = samples.size
    sigma = samples.std(ddof=1)
    if sigma <= 0:
        sigma = 1e-6
    bw = 1.06 * sigma * n ** (-1 / 5)  # Silverman's rule
    diffs = (grid[:, None] - samples[None, :]) / bw
    return np.exp(-0.5 * diffs ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))


# ----------------------------------------------------------------------
# Figure 1: bootstrap density
# ----------------------------------------------------------------------

def fig_bootstrap_density(boot_df: pd.DataFrame, dist: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = min(d.min() for d in dist.values())
    hi = max(d.max() for d in dist.values())
    grid = np.linspace(lo - 1, hi + 1, 600)

    for method in METHOD_ORDER:
        samples = dist[method]
        density = kde_eval(samples, grid)
        col = METHOD_COLOURS[method]
        ax.plot(grid, density, color=col, label=method, linewidth=1.8)
        row = boot_df[boot_df["method"] == method].iloc[0]
        for x in (row["ci_lower"], row["ci_upper"]):
            ax.axvline(x, color=col, linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Standard accuracy (%)")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap distribution of standard accuracy (10 000 cluster resamples)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)
    save(fig, "bootstrap_density.pdf")


# ----------------------------------------------------------------------
# Figure 2: forest plot
# ----------------------------------------------------------------------

def fig_forest(boot_df: pd.DataFrame) -> None:
    sorted_df = boot_df.sort_values("point_estimate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(sorted_df))
    for i, row in sorted_df.iterrows():
        col = METHOD_COLOURS[row["method"]]
        ax.errorbar(
            row["point_estimate"], i,
            xerr=[[row["point_estimate"] - row["ci_lower"]],
                  [row["ci_upper"] - row["point_estimate"]]],
            fmt="o", color=col, ecolor=col, elinewidth=2, capsize=4, markersize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_df["method"])
    ax.set_xlabel("Standard accuracy (%) with 95 % bootstrap CI")
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Per-method standard accuracy (forest plot)")
    save(fig, "forest_plot.pdf")


# ----------------------------------------------------------------------
# Helpers for per-field statistics
# ----------------------------------------------------------------------

def per_field_accuracy(long_df: pd.DataFrame) -> pd.DataFrame:
    """Returns matrix [method x field] of accuracy in %."""
    pivot = long_df.pivot_table(
        index="method", columns="field", values="correct", aggfunc="mean",
    ) * 100.0
    pivot = pivot.reindex(index=METHOD_ORDER, columns=SCORED_FIELDS)
    return pivot


# ----------------------------------------------------------------------
# Figure 3: per-field heatmap
# ----------------------------------------------------------------------

def fig_per_field_heatmap(long_df: pd.DataFrame) -> None:
    mat = per_field_accuracy(long_df)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat.values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            colour = "white" if v < 55 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", color=colour, fontsize=9)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.85)
    ax.set_title("Per-field accuracy (%)")
    save(fig, "per_field_heatmap.pdf")


# ----------------------------------------------------------------------
# Figure 4: per-field grouped bars
# ----------------------------------------------------------------------

def fig_per_field_bars(long_df: pd.DataFrame) -> None:
    mat = per_field_accuracy(long_df)
    fig, ax = plt.subplots(figsize=(14, 6))
    n_methods = len(METHOD_ORDER)
    n_fields = len(SCORED_FIELDS)
    x = np.arange(n_fields)
    width = 0.16
    for i, method in enumerate(METHOD_ORDER):
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset, mat.loc[method].values, width=width,
            color=METHOD_COLOURS[method], label=method,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(SCORED_FIELDS, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Per-field accuracy by method")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "per_field_bars.pdf")


# ----------------------------------------------------------------------
# Figure 5: per-manual distribution
# ----------------------------------------------------------------------

def fig_per_manual_distribution(long_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.default_rng(7)
    box_data = []
    for i, method in enumerate(METHOD_ORDER):
        sub = long_df[long_df["method"] == method]
        per_manual = sub.groupby("manual")["correct"].mean() * 100.0
        box_data.append(per_manual.values)
        jitter = rng.uniform(-0.18, 0.18, size=per_manual.size)
        ax.scatter(
            np.full(per_manual.size, i) + jitter, per_manual.values,
            color=METHOD_COLOURS[method], alpha=0.55, s=22, edgecolor="white", linewidth=0.4,
        )
    bp = ax.boxplot(
        box_data, positions=range(len(METHOD_ORDER)), widths=0.55,
        showfliers=False, patch_artist=True,
    )
    for patch, method in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor("none")
        patch.set_edgecolor(METHOD_COLOURS[method])
        patch.set_linewidth(1.4)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Per-manual accuracy (%)")
    ax.set_title("Distribution of per-manual standard accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "per_manual_distribution.pdf")


# ----------------------------------------------------------------------
# Figure 6: cost vs. accuracy with Pareto frontier
# ----------------------------------------------------------------------

def fig_cost_accuracy_pareto(boot_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    pts = []
    for method in METHOD_ORDER:
        cost = METHOD_COST_USD[method]
        acc = float(boot_df.loc[boot_df["method"] == method, "point_estimate"].iloc[0])
        pts.append((method, cost, acc))
        ax.scatter(cost, acc, s=110, color=METHOD_COLOURS[method], zorder=3, edgecolor="black", linewidth=0.6)
        ax.annotate(
            method, (cost, acc), xytext=(8, 6), textcoords="offset points", fontsize=10,
        )

    # Pareto frontier (upper-left envelope: minimise cost, maximise accuracy).
    sorted_pts = sorted(pts, key=lambda p: p[1])  # by cost
    frontier: list[tuple[str, float, float]] = []
    best_acc = -np.inf
    for name, c, a in sorted_pts:
        if a > best_acc:
            frontier.append((name, c, a))
            best_acc = a
    if len(frontier) >= 2:
        xs = [p[1] for p in frontier]
        ys = [p[2] for p in frontier]
        ax.plot(xs, ys, color="black", linewidth=1, linestyle=":", zorder=2, label="Pareto frontier")
        ax.legend(frameon=False, loc="lower right")

    ax.set_xlabel("Total cost (USD per 51 manuals)")
    ax.set_ylabel("Standard accuracy (%)")
    ax.set_title("Cost vs. accuracy")
    ax.grid(True, alpha=0.3)
    save(fig, "cost_accuracy_pareto.pdf")


# ----------------------------------------------------------------------
# Figure 7: coverage vs. error rate when answering
# ----------------------------------------------------------------------

def fig_coverage_vs_error(long_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHOD_ORDER:
        sub = long_df[long_df["method"] == method]
        n_total = len(sub)
        n_nf = int(sub["is_not_found"].sum())
        n_attempted = n_total - n_nf
        n_correct = int(sub["correct"].sum())
        n_wrong_value = n_attempted - n_correct
        coverage = 100.0 * n_attempted / n_total
        error = 100.0 * n_wrong_value / max(n_attempted, 1)
        ax.scatter(coverage, error, s=120, color=METHOD_COLOURS[method], edgecolor="black", linewidth=0.6, zorder=3)
        ax.annotate(method, (coverage, error), xytext=(8, 6), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Coverage / answer rate (%)")
    ax.set_ylabel("Error rate when answering (%)")
    ax.set_title("Coverage vs. error rate")
    ax.grid(True, alpha=0.3)
    save(fig, "coverage_vs_error.pdf")


# ----------------------------------------------------------------------
# Figure 8: McNemar contingency grid
# ----------------------------------------------------------------------

def fig_mcnemar_grid(mcn_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, mcn_df.iterrows()):
        table = np.array([
            [int(row["both_correct"]), int(row["A_only_correct"])],
            [int(row["B_only_correct"]), int(row["both_wrong"])],
        ])
        im = ax.imshow(table, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(table[i, j]), ha="center", va="center",
                        fontsize=14, color="black")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f"{row['method_B']}\ncorrect", f"{row['method_B']}\nwrong"], fontsize=9)
        ax.set_yticklabels([f"{row['method_A']}\ncorrect", f"{row['method_A']}\nwrong"], fontsize=9)
        p = row["p_value"]
        p_str = f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
        ax.set_title(f"{row['method_A']}  vs.  {row['method_B']}\np = {p_str}", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("McNemar contingency tables", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "mcnemar_contingency_grid.pdf")


# ----------------------------------------------------------------------
# Figure 9: abstention by field (5 vertically-stacked subplots)
# ----------------------------------------------------------------------

def fig_abstention_by_field(long_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    x = np.arange(len(SCORED_FIELDS))
    for ax, method in zip(axes, METHOD_ORDER):
        sub = long_df[long_df["method"] == method]
        correct = []
        wrong = []
        not_found = []
        for f in SCORED_FIELDS:
            fsub = sub[sub["field"] == f]
            n_correct = int(fsub["correct"].sum())
            n_nf = int(fsub["is_not_found"].sum())
            n_wrong = len(fsub) - n_correct - n_nf
            correct.append(n_correct)
            wrong.append(n_wrong)
            not_found.append(n_nf)
        correct = np.array(correct); wrong = np.array(wrong); not_found = np.array(not_found)
        ax.bar(x, correct, color="#2ca02c", label="correct")
        ax.bar(x, wrong, bottom=correct, color="#d62728", label="wrong")
        ax.bar(x, not_found, bottom=correct + wrong, color="#7f7f7f", label="not found")
        ax.set_ylabel(method, fontsize=10)
        ax.set_ylim(0, 55)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(SCORED_FIELDS, rotation=30, ha="right")
    fig.suptitle("Per-field outcome breakdown (counts out of 51 manuals)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "abstention_by_field.pdf")


# ----------------------------------------------------------------------
# Save helper
# ----------------------------------------------------------------------

def save(fig, name: str) -> None:
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures/{name}")


def main() -> None:
    long_df, boot_df, dist, mcn_df = load_artefacts()

    fig_bootstrap_density(boot_df, dist)
    fig_forest(boot_df)
    fig_per_field_heatmap(long_df)
    fig_per_field_bars(long_df)
    fig_per_manual_distribution(long_df)
    fig_cost_accuracy_pareto(boot_df)
    fig_coverage_vs_error(long_df)
    fig_mcnemar_grid(mcn_df)
    fig_abstention_by_field(long_df)

    print("\nAll figures written to figures/:")
    for p in sorted(FIG_DIR.glob("*.pdf")):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
