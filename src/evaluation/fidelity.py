"""
Statistical fidelity metrics for synthetic vs. real data.

Measures:
  - Column-wise Jensen-Shannon Divergence (JSD)
  - Column-wise Wasserstein distance
  - Pairwise correlation matrix difference
  - Basic marginal statistics comparison
"""

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def _safe_histogram(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Normalised histogram with Laplace smoothing."""
    counts = np.histogram(values, bins=bins)[0].astype(np.float64) + 1e-10
    return counts / counts.sum()


def column_jsd(real: pd.DataFrame, syn: pd.DataFrame,
               cols: list, n_bins: int = 50) -> dict:
    """Per-column Jensen-Shannon Divergence."""
    scores = {}
    for col in cols:
        if col not in real.columns or col not in syn.columns:
            continue
        r, s = real[col].values.astype(float), syn[col].values.astype(float)
        lo = min(r.min(), s.min())
        hi = max(r.max(), s.max())
        if hi - lo < 1e-10:
            scores[col] = 0.0
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p = _safe_histogram(r, bins)
        q = _safe_histogram(s, bins)
        scores[col] = float(jensenshannon(p, q) ** 2)  # JSD (squared)
    return scores


def column_wasserstein(real: pd.DataFrame, syn: pd.DataFrame,
                       cols: list) -> dict:
    """Per-column Wasserstein-1 distance."""
    scores = {}
    for col in cols:
        if col not in real.columns or col not in syn.columns:
            continue
        r = real[col].values.astype(float)
        s = syn[col].values.astype(float)
        scores[col] = float(wasserstein_distance(r, s))
    return scores


def correlation_difference(real: pd.DataFrame, syn: pd.DataFrame,
                           cols: list) -> dict:
    """Frobenius norm and mean absolute difference of correlation matrices."""
    common = [c for c in cols if c in real.columns and c in syn.columns]
    if len(common) < 2:
        return {"mean_abs_diff": 0.0, "max_abs_diff": 0.0, "frobenius_norm": 0.0}

    corr_r = real[common].corr().values
    corr_s = syn[common].corr().values
    diff = np.abs(corr_r - corr_s)
    triu = np.triu_indices_from(diff, k=1)
    return {
        "mean_abs_diff": float(diff[triu].mean()),
        "max_abs_diff": float(diff[triu].max()),
        "frobenius_norm": float(np.linalg.norm(diff, "fro")),
    }


def marginal_stats(real: pd.DataFrame, syn: pd.DataFrame,
                   cols: list) -> pd.DataFrame:
    """Compare mean, std, min, max per column between real and synthetic."""
    rows = []
    for col in cols:
        if col not in real.columns or col not in syn.columns:
            continue
        r, s = real[col], syn[col]
        rows.append({
            "column": col,
            "real_mean": r.mean(), "syn_mean": s.mean(),
            "real_std": r.std(), "syn_std": s.std(),
            "mean_diff%": abs(r.mean() - s.mean()) / (abs(r.mean()) + 1e-8) * 100,
            "std_diff%": abs(r.std() - s.std()) / (abs(r.std()) + 1e-8) * 100,
        })
    return pd.DataFrame(rows)


def evaluate_fidelity(real_df: pd.DataFrame, syn_df: pd.DataFrame,
                      cols: list, label: str = ""):
    """Run all fidelity metrics and print a summary."""
    jsd = column_jsd(real_df, syn_df, cols)
    wass = column_wasserstein(real_df, syn_df, cols)
    corr = correlation_difference(real_df, syn_df, cols)

    avg_jsd = np.mean(list(jsd.values())) if jsd else 0.0
    avg_wass = np.mean(list(wass.values())) if wass else 0.0

    print(f"  Fidelity [{label}]:  "
          f"avg_JSD={avg_jsd:.4f}  avg_Wass={avg_wass:.4f}  "
          f"corr_diff={corr['mean_abs_diff']:.4f}  "
          f"corr_frob={corr['frobenius_norm']:.4f}")

    return {
        "jsd": jsd,
        "wasserstein": wass,
        "correlation": corr,
        "avg_jsd": avg_jsd,
        "avg_wasserstein": avg_wass,
    }
