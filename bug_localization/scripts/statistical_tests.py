"""
Statistical tests for bug localization experiment results.
Example:
  python scripts/statistical_tests.py --exp Baseline="output/bm25/swe/baseline_100percent/results.csv" \
    --exp AllField="../output/swe/w_extracted/bm25-retriever_50_all_field/results.csv" \
    --exp AllCode="../output/swe/w_extracted/bm25-retriever_50_all_code/results.csv" \
    --exp Expl="../output/swe/w_extracted/bm25-retriever_50_explanation/results.csv" \
    --exp ExpIdSnippets="../output/swe/w_extracted/bm25-retriever_50_explanation_id_snippets/results.csv" \
    --exp IdSnippets="../output/swe/w_extracted/bm25-retriever_50_id_snippets/results.csv" \
    --exp ExpId=../output/swe/w_extracted/bm25-retriever_explanation_id/results.csv \
    --tag=SWE
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
from ast import literal_eval
from scipy.stats import mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar  #McNemar for paired binary Hit@k

from src.baselines.metrics import classification_metrics as metrics

def removeDuplicates(final_files: list) -> list:
    # Deduplicate topk files list (predicted) as many chunks can match in the same file
    # Remove duplicate predictions while preserving order.
    if not isinstance(final_files, list):
        return []

    seen = set()
    deduped_predictions = []
    for pred in final_files:
        if isinstance(pred, dict):
            continue
        if pred not in seen:
            seen.add(pred)
            deduped_predictions.append(pred)
    return deduped_predictions

def calculateResults(column: str,
                     df: pd.DataFrame,
                     k_col: str = "top_k",
                     column_suffix: str = "", 
                     column_prefix: str = ""):
    """Compute all metrics, using df[k_col] as the *per‑row* k."""

    # ----- helpers ---------------------------------------------------------
    def k(row):
        return int(row[k_col])          # convenience, k varies by row

    def preds(row):
        """
        Deduplicate and truncate to the row-specific k.
        Guarantees len(preds(row)) == k(row)  (unless the list is shorter).
        """
        dedup = removeDuplicates(row[column])
        return dedup[: k(row)]

    # single‑row metrics ----------------------------------------------------
    df[f'{column_prefix}precision@2{column_suffix}'] = df.apply(
        lambda r: metrics.compute_precision_at_2_single(r['changed_files'],
                                                        preds(r)),
        axis=1
    )
    df[f'{column_prefix}precision@k{column_suffix}'] = df.apply(
        lambda r: metrics.compute_precision_at_k(r['changed_files'],
                                                 preds(r),
                                                 k=k(r)),
        axis=1
    )
    df[f'{column_prefix}recall@1{column_suffix}'] = df.apply(
        lambda r: metrics.compute_recall_at_1_single(r['changed_files'],
                                                     preds(r)),
        axis=1
    )
    df[f'{column_prefix}recall@2{column_suffix}'] = df.apply(
        lambda r: metrics.compute_recall_at_2_single(r['changed_files'],
                                                     preds(r)),
        axis=1
    )
    df[f'{column_prefix}MAP{column_suffix}'] = df.apply(
        lambda r: metrics.compute_average_precision(r['changed_files'],
                                                    preds(r)),
        axis=1
    )
    df[f'{column_prefix}recall@k{column_suffix}'] = df.apply(
        lambda r: metrics.compute_recall_at_k(r['changed_files'],
                                              preds(r),
                                              k=k(r)),
        axis=1
    )
    df[f'{column_prefix}f1@k{column_suffix}'] = df.apply(
        lambda r: metrics.compute_f1_at_k(r[f'recall@k{column_suffix}'],
                                          r[f'precision@k{column_suffix}'],
                                          k=k(r)),
        axis=1
    )
    df[f'{column_prefix}hit_rate@k{column_suffix}'] = df.apply(
        lambda r: metrics.hit_rate_at_k(r['changed_files'],
                                        preds(r),
                                        k=k(r)),
        axis=1
    )
    df[f'{column_prefix}all_files_predicted{column_suffix}'] = df.apply(
        lambda r: metrics.all_files_in_predicted(r['changed_files'],
                                                 preds(r)),
        axis=1
    )
    df[f'{column_prefix}MRR{column_suffix}'] = df.apply(
        lambda r: metrics.mean_reciprocal_rank(r['changed_files'],
                                               preds(r)),
        axis=1
    )

# ----------------- Cliff's delta -----------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1 = x.size
    n2 = y.size
    if n1 == 0 or n2 == 0:
        return np.nan
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    import bisect
    greater = 0
    equal = 0
    for xi in x_sorted:
        pos_le = bisect.bisect_right(y_sorted, xi)
        pos_lt = bisect.bisect_left(y_sorted, xi)
        greater += pos_lt
        equal += (pos_le - pos_lt)
    p_gt = greater / (n1 * n2)
    p_eq = equal / (n1 * n2)
    return 2 * p_gt + p_eq - 1

def cliffs_magnitude(delta: float) -> str:
    if np.isnan(delta):
        return "NA"
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    elif a < 0.33:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"

# ----------------- IO & Testing -----------------

def load_exp_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, converters={
        "changed_files": literal_eval,
        "final_files": literal_eval
    })
    if "text_id" not in df and "instance_id" not in df:
        raise ValueError(f"{path} missing 'text_id' or 'instance_id' column")
    
    id_field = "instance_id"
    if "text_id" in df:
        id_field = "text_id"
    return df, id_field

def compute_map_hit(df: pd.DataFrame, k: int, id_field : str) -> pd.DataFrame:
    # Set per-row k exactly as user's computeMetrics -- using topk fixed
    df = df.copy()
    df["top_k"] = k
    # Run calculation on 'final_files' with same defaults
    calculateResults('final_files', df, k_col='top_k')
    # Return just the columns we need
    out = df[[id_field, "MAP", "hit_rate@k"]].copy()
    out = out.rename(columns={"hit_rate@k":"Hit"})
    return out

# ----------------- Pairwise testing -----------------

def mcnemar_for_hit(sa: pd.Series, sb: pd.Series):
    """Run McNemar on paired binary series A (sa) vs B (sb)."""
    both = pd.DataFrame({"A": sa.astype(int), "B": sb.astype(int)}).dropna()
    b01 = int(((both["A"] == 0) & (both["B"] == 1)).sum())  # A miss, B hit
    b10 = int(((both["A"] == 1) & (both["B"] == 0)).sum())  # A hit,  B miss
    table = [[0, b01],
             [b10, 0]]
    res = mcnemar(table, exact=False, correction=True)
    p = float(res.pvalue)
    # Extra interpretable stats for paired binary:
    n_disc = b01 + b10
    net_gain = b01 - b10
    cohens_g = (net_gain / n_disc) if n_disc > 0 else 0.0  # effect on discordants
    odds_ratio = (b01 / b10) if b10 > 0 else (float('inf') if b01 > 0 else np.nan)
    return p, b01, b10, net_gain, cohens_g, odds_ratio

def pairwise_per_k(per_exp_data: dict, metric: str, id_field: str):
    labels = sorted(per_exp_data.keys())
    rows = []
    p_pairs = []
    pmat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    dmat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)

    def series(label):
        return per_exp_data[label].set_index(id_field)[metric]

    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            a, b = labels[i], labels[j]
            sa, sb = series(a), series(b)
            common = sa.index.intersection(sb.index)
            xa = sa.loc[common].values
            yb = sb.loc[common].values

            n_common = len(common)
            mean_a = float(np.nanmean(xa)) if n_common else np.nan
            mean_b = float(np.nanmean(yb)) if n_common else np.nan
            diff_mean = (mean_a - mean_b) if n_common else np.nan

            if metric == "MAP":
                if n_common == 0:
                    U, p = np.nan, np.nan
                    delta = np.nan
                else:
                    U, p = mannwhitneyu(xa, yb, alternative="two-sided", method="auto")
                    delta = cliffs_delta(xa, yb)
                b01 = b10 = net_gain = cohens_g = odds_ratio = np.nan  # not applicable
            else:  # Hit -> McNemar
                if n_common == 0:
                    U = np.nan
                    p = np.nan
                    delta = np.nan
                    b01 = b10 = net_gain = cohens_g = odds_ratio = np.nan
                else:
                    # McNemar uses paired 0/1 series:
                    sA = sa.loc[common].astype(int)
                    sB = sb.loc[common].astype(int)
                    p, b01, b10, net_gain, cohens_g, odds_ratio = mcnemar_for_hit(sA, sB)
                    U = np.nan  # not used for Hit
                    # Optional: still provide Cliff's δ on binary for practical effect magnitude
                    delta = cliffs_delta(sA.values, sB.values)

            rows.append({
                "A": a, "B": b, "metric": metric,
                "n_common": n_common,
                "mean_A": mean_a, "mean_B": mean_b, "diff_mean": diff_mean,
                "U": U, "p_raw": p,
                "cliffs_delta": delta, "delta_magnitude": cliffs_magnitude(delta),
                # McNemar-specific diagnostics (NaN for MAP):
                "b01_A0_B1": b01, "b10_A1_B0": b10,
                "net_gain_B_minus_A": net_gain,
                "cohens_g_on_discordants": cohens_g,
                "discordant_odds_ratio": odds_ratio,
            })
            p_pairs.append(((a,b), float(p) if not np.isnan(p) else np.nan))
            pmat.loc[a,b] = pmat.loc[b,a] = p
            dmat.loc[a,b] = delta
            dmat.loc[b,a] = -delta if not np.isnan(delta) else np.nan

    np.fill_diagonal(pmat.values, 0.0)
    np.fill_diagonal(dmat.values, 0.0)

    return pd.DataFrame(rows), pmat, dmat

def main():
    ap = argparse.ArgumentParser(description="Pairwise sigtests: Mann–Whitney for MAP, McNemar for Hit@k (paired binary).")
    ap.add_argument("--exp", action="append", required=True, help="LABEL=PATH/to/results.csv  (repeatable)")
    ap.add_argument("--k", type=int, nargs="+", default=[1,5,10], help="Top-k values to compare.")
    ap.add_argument("--tag", type=str, default="ALL", help="Tag used in output filenames, e.g., LCA or SWE.")
    ap.add_argument("--outdir", type=str, default="sigtests_outputs_same_metrics")
    args = ap.parse_args()

    # Parse experiments
    exps = {}
    for spec in args.exp:
        if "=" not in spec:
            raise ValueError(f"--exp must be LABEL=PATH, got: {spec}")
        label, path = spec.split("=", 1)
        exps[label.strip()] = path.strip()

    os.makedirs(args.outdir, exist_ok=True)

    for k in args.k:
        # Load and compute per-item MAP & Hit with user's calculations
        per_exp = {}
        for label, path in exps.items():
            df, id_field = load_exp_results(path)
            vals = compute_map_hit(df, k, id_field)
            per_exp[label] = vals

        # Pairwise tests
        for metric in ["MAP", "Hit"]:
            df_summary, pmat, dmat = pairwise_per_k(per_exp, metric, id_field)
            base = f"{args.tag}_k{k}_{metric}"
            df_summary.sort_values(by=["p_raw","cliffs_delta"], ascending=[True,False]).to_csv(
                os.path.join(args.outdir, f"summary_{base}.csv"), index=False
            )
            pmat.to_csv(os.path.join(args.outdir, f"pvalues_{base}.csv"))
            dmat.to_csv(os.path.join(args.outdir, f"cliffs_delta_{base}.csv"))
            print(f"[k={k} {metric}] wrote: summary_{base}.csv, pvalues_{base}.csv, cliffs_delta_{base}.csv")

if __name__ == "__main__":
    main()
