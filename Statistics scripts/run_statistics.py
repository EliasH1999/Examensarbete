"""Statistical analyses for the Modbus RTU extraction evaluation.

Runs the four analyses described in the methodology:

    1. Builds a long-format predictions DataFrame from the five canonical
       result CSVs.
    2. Runs four pairwise McNemar exact tests.
    3. Runs a 10 000-iteration cluster bootstrap (resampling at the manual
       level) for each method's standard accuracy and reports a 95 %
       percentile CI.
    4. Prints a Markdown summary of both result tables.

Outputs:
    results/predictions_long.csv
    results/mcnemar_results.csv
    results/bootstrap_results.csv
    results/bootstrap_distributions.json

Run from the project root:
    python scripts/run_statistics.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# Make `accuracy.py` importable from the project root for the field list.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from accuracy import PARAMETER_NAMES, EXCLUDED_KEYS  # noqa: E402

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR  # write artefacts back to results/

METHOD_FILES: dict[str, Path] = {
    "naive_user":         RESULTS_DIR / "naive_user"        / "results_naive_user_20260427_121940.csv",
    "configurator_user":  RESULTS_DIR / "configurator_user" / "results_configurator_user_20260427_121221.csv",
    "engineered_prompt":  RESULTS_DIR / "engineered_prompt" / "results_engineered_prompt_20260427_120239.csv",
    "naive_rag":          RESULTS_DIR / "naive_rag"         / "results_naive_rag_20260429_123742.csv",
    "modular_rag":        RESULTS_DIR / "modular_rag"       / "results_modular_rag_20260429_103953.csv",
}

METHOD_ORDER = list(METHOD_FILES.keys())

# Friendly field name -> config key (for the 13 scored fields).
FIELD_KEY_BY_NAME: dict[str, str] = {
    name: key for key, name in PARAMETER_NAMES.items() if key not in EXCLUDED_KEYS
}
SCORED_FIELD_NAMES: list[str] = list(FIELD_KEY_BY_NAME.keys())

# Pairwise comparisons for McNemar.
MCNEMAR_PAIRS: list[tuple[str, str]] = [
    ("engineered_prompt", "modular_rag"),
    ("modular_rag", "naive_rag"),
    ("configurator_user", "naive_user"),
    ("engineered_prompt", "configurator_user"),
]

# Reported Table 5.1 reference points (% standard accuracy).
TABLE_5_1_REFERENCE = {
    "naive_user":        66.02,
    "configurator_user": 82.27,
    "engineered_prompt": 89.29,
    "naive_rag":         83.98,
    "modular_rag":       85.38,
}

# CSV-row regex (mapped form): "Status: Field Name (subnetwork.path=value)"
STATUS_FIELD_RE = re.compile(
    r"^(Correct|Non-default match|Wrong|Excluded):\s*(.+?)\s*\(subnetwork\.[^=]+=",
)
# CSV-row regex (unmapped form): "Status: subnetwork.path=value"
# Used when the manual's golden file contains a key that has no friendly name
# in PARAMETER_NAMES (e.g. FC=1 Read-Coils response keys). accuracy.py still
# scores these against the denominator, so we must parse them too.
STATUS_RAWKEY_RE = re.compile(
    r"^(Correct|Non-default match|Wrong|Excluded):\s*(subnetwork\.[^=]+)=",
)
NOT_FOUND_RE = re.compile(r"^\s*(predicted:\s*)?not\s*found", re.IGNORECASE)


# ----------------------------------------------------------------------
# Step 1 - long-format DataFrame
# ----------------------------------------------------------------------

def parse_method_csv(method: str, path: Path) -> pd.DataFrame:
    """Parse a single results CSV into one row per (manual, field) prediction."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for {method!r}: {path}")

    rows: list[dict] = []
    cur_manual: str | None = None
    in_avg_block = False
    with path.open(encoding="utf-8") as fh:
        for csv_row in csv.DictReader(fh):
            man = (csv_row.get("manual") or "").strip()
            if man:
                # Per-manual summary row -> start a new manual block.
                in_avg_block = man.upper().startswith("AVERAGE")
                cur_manual = None if in_avg_block else man
                continue
            if in_avg_block or cur_manual is None:
                continue
            acc_cell = csv_row.get("accuracy") or ""
            pred_cell = csv_row.get("accuracy_percent") or ""
            m = STATUS_FIELD_RE.match(acc_cell)
            if m:
                status_label, field_name = m.group(1), m.group(2)
            else:
                # Unmapped key (no friendly name in PARAMETER_NAMES).
                # accuracy.py still scores these, so we must too.
                m2 = STATUS_RAWKEY_RE.match(acc_cell)
                if not m2:
                    continue
                status_label = m2.group(1)
                field_name = m2.group(2)  # full subnetwork.* key path
            # Excluded rows do not contribute to numerator or denominator.
            if status_label == "Excluded":
                continue
            correct = 1 if status_label in ("Correct", "Non-default match") else 0
            is_not_found = bool(NOT_FOUND_RE.match(pred_cell))
            rows.append({
                "method":       method,
                "manual":       cur_manual,
                "field":        field_name,
                "status":       status_label,
                "correct":      correct,
                "is_not_found": int(is_not_found),
            })
    return pd.DataFrame(rows)


def build_long_df() -> pd.DataFrame:
    """Build the long-format DataFrame for all methods."""
    parts = []
    for method, path in METHOD_FILES.items():
        df = parse_method_csv(method, path)
        parts.append(df)
        print(f"  parsed {method:22s} rows={len(df):4d}  manuals={df['manual'].nunique():3d}  file={path.name}")
    return pd.concat(parts, ignore_index=True)


def sanity_check_row_counts(df: pd.DataFrame) -> None:
    """Stop if methods disagree on the (manual, field) coverage."""
    counts = df.groupby("method").size()
    print("\nRow counts per method:")
    print(counts.to_string())
    if counts.nunique() != 1:
        print("\nERROR: methods have inconsistent (manual, field) coverage. Diff:")
        all_pairs = set(zip(df["manual"], df["field"]))
        for method in counts.index:
            sub = df[df["method"] == method]
            present = set(zip(sub["manual"], sub["field"]))
            missing = all_pairs - present
            extra = present - all_pairs
            print(f"  {method}: missing {len(missing)} pairs, extra {len(extra)} pairs")
            for man, field in sorted(missing)[:10]:
                print(f"     missing {man!r} :: {field!r}")
        sys.exit(2)
    n = int(counts.iloc[0])
    if n != 664:
        print(f"\nWARN: per-method row count is {n} (expected 664). Continuing.")


# ----------------------------------------------------------------------
# Step 2 - McNemar
# ----------------------------------------------------------------------

def run_mcnemar(df: pd.DataFrame) -> pd.DataFrame:
    """Run pairwise exact McNemar tests on (manual, field)-paired predictions."""
    # Pivot to (manual, field) x method binary correctness matrix.
    wide = df.pivot_table(
        index=["manual", "field"], columns="method", values="correct", aggfunc="first"
    )
    rows = []
    for a, b in MCNEMAR_PAIRS:
        paired = wide[[a, b]].dropna()
        a_arr = paired[a].astype(int).to_numpy()
        b_arr = paired[b].astype(int).to_numpy()
        n11 = int(((a_arr == 1) & (b_arr == 1)).sum())  # both correct
        n10 = int(((a_arr == 1) & (b_arr == 0)).sum())  # A only
        n01 = int(((a_arr == 0) & (b_arr == 1)).sum())  # B only
        n00 = int(((a_arr == 0) & (b_arr == 0)).sum())  # both wrong
        table = np.array([[n11, n10], [n01, n00]])
        result = mcnemar(table, exact=True)
        winner = a if n10 > n01 else (b if n01 > n10 else "tie")
        rows.append({
            "method_A":          a,
            "method_B":          b,
            "n_paired":          int(paired.shape[0]),
            "both_correct":      n11,
            "A_only_correct":    n10,
            "B_only_correct":    n01,
            "both_wrong":        n00,
            "agreements":        n11 + n00,
            "disagreements":     n10 + n01,
            "p_value":           float(result.pvalue),
            "more_wins":         winner,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Step 3 - Cluster bootstrap
# ----------------------------------------------------------------------

def cluster_bootstrap(
    df: pd.DataFrame, n_iter: int = 10_000, seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Cluster bootstrap at the manual level; 95 % percentile CI."""
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {}
    summary_rows: list[dict] = []

    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        # Group correctness arrays by manual once, up front.
        by_manual: dict[str, np.ndarray] = {
            man: g["correct"].to_numpy(dtype=np.int64)
            for man, g in sub.groupby("manual", sort=True)
        }
        manuals = np.array(list(by_manual.keys()))
        n_manuals = len(manuals)

        # Point estimate = macro mean of per-manual accuracies (matches Table 5.1).
        per_manual_acc = np.array([
            arr.sum() / arr.size for arr in by_manual.values()
        ])
        point = 100.0 * per_manual_acc.mean()

        # Cluster bootstrap: resample manuals with replacement, recompute.
        accs = np.empty(n_iter, dtype=np.float64)
        for i in range(n_iter):
            idx = rng.integers(0, n_manuals, size=n_manuals)
            picked = [by_manual[manuals[j]] for j in idx]
            # Macro mean of per-manual accuracies in this bootstrap sample.
            sample_per_manual = np.array([a.sum() / a.size for a in picked])
            accs[i] = 100.0 * sample_per_manual.mean()

        ci_lo, ci_hi = np.percentile(accs, [2.5, 97.5])
        summary_rows.append({
            "method":         method,
            "point_estimate": point,
            "ci_lower":       float(ci_lo),
            "ci_upper":       float(ci_hi),
            "ci_width":       float(ci_hi - ci_lo),
        })
        distributions[method] = accs.tolist()
        print(
            f"  {method:22s} point={point:6.2f}%  "
            f"95% CI=[{ci_lo:6.2f}, {ci_hi:6.2f}]  width={ci_hi - ci_lo:5.2f} pp"
        )

    return pd.DataFrame(summary_rows), distributions


# ----------------------------------------------------------------------
# Step 4 - Markdown summary
# ----------------------------------------------------------------------

def fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_markdown_summary(
    mcnemar_df: pd.DataFrame, bootstrap_df: pd.DataFrame,
) -> None:
    print("\n## McNemar exact-test results")
    print()
    print("| Comparison (A vs. B) | n paired | A-only correct | B-only correct | Agreements | Disagreements | More wins | p-value |")
    print("|---|---:|---:|---:|---:|---:|---|---:|")
    for _, r in mcnemar_df.iterrows():
        print(
            f"| {r['method_A']} vs. {r['method_B']} "
            f"| {int(r['n_paired'])} "
            f"| {int(r['A_only_correct'])} "
            f"| {int(r['B_only_correct'])} "
            f"| {int(r['agreements'])} "
            f"| {int(r['disagreements'])} "
            f"| {r['more_wins']} "
            f"| {fmt_p(r['p_value'])} |"
        )

    print("\n## Cluster bootstrap (95 % percentile CI, 10 000 iterations, manual-level resampling)")
    print()
    print("| Method | Point estimate | 95 % CI lower | 95 % CI upper | CI width |")
    print("|---|---:|---:|---:|---:|")
    for _, r in bootstrap_df.iterrows():
        print(
            f"| {r['method']} "
            f"| {r['point_estimate']:.2f} % "
            f"| {r['ci_lower']:.2f} % "
            f"| {r['ci_upper']:.2f} % "
            f"| {r['ci_width']:.2f} pp |"
        )


# ----------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------

def run_sanity_checks(
    df: pd.DataFrame, mcnemar_df: pd.DataFrame, bootstrap_df: pd.DataFrame,
) -> None:
    print("\n--- Sanity checks ---")

    # 1. McNemar engineered_prompt vs. naive_user should be near-zero.
    wide = df.pivot_table(
        index=["manual", "field"], columns="method", values="correct", aggfunc="first"
    ).dropna(subset=["engineered_prompt", "naive_user"])
    a = wide["engineered_prompt"].astype(int).to_numpy()
    b = wide["naive_user"].astype(int).to_numpy()
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    p_extreme = mcnemar(np.array([[0, n10], [n01, 0]]), exact=True).pvalue
    status = "OK" if p_extreme < 1e-6 else "WARN"
    print(f"  [{status}] engineered_prompt vs. naive_user McNemar p = {fmt_p(p_extreme)}")

    # 2. CI widths roughly 4-8 pp; warn if any narrower than 1 pp.
    too_narrow = bootstrap_df[bootstrap_df["ci_width"] < 1.0]
    if too_narrow.empty:
        print(f"  [OK]   bootstrap CI widths range "
              f"{bootstrap_df['ci_width'].min():.2f}-{bootstrap_df['ci_width'].max():.2f} pp")
    else:
        print(f"  [WARN] CI widths narrower than 1 pp (suspect field-level resample):")
        print(too_narrow.to_string(index=False))

    # 3. Each point estimate within 0.5 pp of Table 5.1 reference.
    for _, r in bootstrap_df.iterrows():
        ref = TABLE_5_1_REFERENCE[r["method"]]
        diff = abs(r["point_estimate"] - ref)
        if diff > 0.5:
            print(f"  [WARN] {r['method']}: point={r['point_estimate']:.2f}% vs Table 5.1 {ref:.2f}% (Δ={diff:.2f} pp)")
            if r["method"] == "modular_rag" and abs(r["point_estimate"] - 84.79) < 0.3:
                print("         -> looks like the older modular_rag CSV was loaded; stop and check METHOD_FILES.")
                sys.exit(3)
        else:
            print(f"  [OK]   {r['method']}: point={r['point_estimate']:.2f}% (Table 5.1 {ref:.2f}%, Δ={diff:.2f} pp)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    print("=== Step 1: build long-format DataFrame ===")
    df = build_long_df()
    sanity_check_row_counts(df)

    long_path = OUT_DIR / "predictions_long.csv"
    df.to_csv(long_path, index=False)
    print(f"\nWrote {long_path.relative_to(ROOT)}  ({len(df)} rows)")

    print("\n=== Step 2: McNemar exact tests ===")
    mcnemar_df = run_mcnemar(df)
    mcnemar_path = OUT_DIR / "mcnemar_results.csv"
    mcnemar_df.to_csv(mcnemar_path, index=False)
    print(f"Wrote {mcnemar_path.relative_to(ROOT)}")
    print(mcnemar_df.to_string(index=False))

    print("\n=== Step 3: cluster bootstrap (manual-level, 10 000 iter, seed=42) ===")
    bootstrap_df, distributions = cluster_bootstrap(df, n_iter=10_000, seed=42)
    boot_path = OUT_DIR / "bootstrap_results.csv"
    bootstrap_df.to_csv(boot_path, index=False)
    print(f"Wrote {boot_path.relative_to(ROOT)}")
    dist_path = OUT_DIR / "bootstrap_distributions.json"
    with dist_path.open("w", encoding="utf-8") as fh:
        json.dump(distributions, fh)
    print(f"Wrote {dist_path.relative_to(ROOT)}")

    print("\n=== Step 4: Markdown summary ===")
    print_markdown_summary(mcnemar_df, bootstrap_df)

    run_sanity_checks(df, mcnemar_df, bootstrap_df)


if __name__ == "__main__":
    main()
