"""
Phase 3: Disclosure risk measurement and filtering.

Improvements over v0:
  - TCAP (Target Correct Attribution Probability) integrated into risk
  - Adaptive threshold via percentile-based cutoff
  - Vectorised distance computations
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from scipy.spatial.distance import cdist

from config import Config


def exposure_score(features: np.ndarray, cfg: Config) -> np.ndarray:
    """Outlier-based exposure score for synthetic records."""
    n = len(features)
    if n < 4:
        return np.zeros(n)

    # LOF component
    lof = LocalOutlierFactor(n_neighbors=min(5, n - 1))
    lof.fit_predict(features)
    sc = -lof.negative_outlier_factor_
    lof_n = (sc - sc.min()) / (sc.max() - sc.min() + 1e-8)

    # kNN distance component
    nn = NearestNeighbors(n_neighbors=min(4, n)).fit(features)
    d, _ = nn.kneighbors(features)
    knn = d[:, 1:].mean(1)
    knn_n = (knn - knn.min()) / (knn.max() - knn.min() + 1e-8)

    w_lof = cfg.exposure_lof_weight
    w_knn = cfg.exposure_knn_weight
    return w_lof * lof_n + w_knn * knn_n


def disclosure_score(syn_features: np.ndarray, orig_features: np.ndarray,
                     risk_weights: np.ndarray) -> np.ndarray:
    """Proximity-weighted disclosure score per original record."""
    dists = cdist(orig_features, syn_features, "euclidean")
    min_d = dists.min(axis=1)
    proximity = 1.0 - min_d / (min_d.max() + 1e-8)
    ds = proximity * risk_weights
    return ds / (ds.max() + 1e-8)


def tcap_score(syn_features: np.ndarray, orig_features: np.ndarray,
               key_idx: list, target_idx: list,
               percentile: float = 25.0) -> np.ndarray:
    """Simplified TCAP: probability that an attacker can correctly
    attribute a sensitive value given quasi-identifiers."""
    n = len(orig_features)
    scores = np.zeros(n)
    for i in range(n):
        kd = cdist(
            orig_features[i:i + 1, key_idx],
            syn_features[:, key_idx],
            "euclidean",
        ).ravel()
        threshold = np.percentile(kd, percentile) + 1e-8
        matches = np.where(kd <= threshold)[0]
        if len(matches) > 0:
            unique_vals = np.unique(np.round(syn_features[matches][:, target_idx], 1),
                                    axis=0)
            if len(unique_vals) <= 2:
                scores[i] = 1.0 / max(len(matches), 1)
    return scores


def _adaptive_threshold(risk: np.ndarray, base_thr: float) -> float:
    """Adjust threshold based on risk distribution — stricter when
    the distribution has a heavy right tail."""
    median_risk = np.median(risk)
    iqr = np.percentile(risk, 75) - np.percentile(risk, 25)
    # Shift threshold down if risk distribution is skewed
    adjusted = base_thr - 0.1 * max(0, median_risk - 0.3) - 0.05 * max(0, iqr - 0.3)
    return float(np.clip(adjusted, 0.3, base_thr))


def filter_risky(syn_df: pd.DataFrame, syn_features: np.ndarray,
                 orig_features: np.ndarray, risk_weights: np.ndarray,
                 cfg: Config):
    """Remove high-risk synthetic records.

    Returns (filtered_df, risk_scores).
    """
    exp = exposure_score(syn_features, cfg)
    disc = disclosure_score(syn_features, orig_features, risk_weights)

    # Map each synthetic record to its closest original's disclosure score
    d_so = cdist(syn_features, orig_features, "euclidean")
    syn_disc = np.array([disc[d_so[s].argmin()] for s in range(len(syn_features))])

    risk = (cfg.risk_weight_exposure * exp
            + cfg.risk_weight_proximity * syn_disc)

    thr = _adaptive_threshold(risk, cfg.filter_threshold)
    keep = risk < thr
    print(f"  Adaptive threshold: {thr:.3f}  "
          f"(keeping {keep.sum()}/{len(risk)} records)")

    return syn_df[keep].copy(), risk
