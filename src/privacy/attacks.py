"""
Privacy attack evaluation.

Four attacks:
  1. Linkability  — can an attacker link a synthetic record to a real one?
  2. Attribute     — can an attacker infer a sensitive attribute?
  3. Class         — can an attacker infer the fraud label?
  4. Membership    — can an attacker determine if a record was in training data?
"""

import numpy as np
from scipy.spatial.distance import cdist

from config import Config


def _attack_cols(syn_df, orig_df, cfg: Config):
    """Determine columns available to the attacker."""
    emb = [c for c in syn_df.columns if c.startswith("emb_")]
    base = [c for c in cfg.attack_key_cols
            if c in syn_df.columns and c in orig_df.columns]
    return base + [c for c in emb if c in orig_df.columns]


def _attack_targets(risk: dict, cfg: Config):
    """Select high-risk records to attack."""
    return [
        (addr, r)
        for addr, r in sorted(risk.items(), key=lambda x: -x[1])
        if r > cfg.attack_risk_threshold
    ][:cfg.attack_top_n]


def attack_linkability(syn_df, orig_df, risk, cols, cfg: Config):
    targets = _attack_targets(risk, cfg)
    results = {}
    sv = syn_df[cols].values
    for addr, r in targets:
        row = orig_df[orig_df["_addr_id"] == addr]
        if row.empty:
            continue
        ds = cdist(row[cols].values, sv, "euclidean").ravel()
        ratio = ds.min() / (np.median(ds) + 1e-8)
        results[addr] = dict(risk=r, ratio=float(ratio), linked=bool(ratio < 0.3))
    return results


def attack_attribute(syn_df, orig_df, risk, cols, cfg: Config,
                     target: str = "total ether balance"):
    targets = _attack_targets(risk, cfg)
    results = {}
    keys = [c for c in cols if c != target and c in syn_df.columns]
    if target not in syn_df.columns or target not in orig_df.columns:
        return results

    vals = orig_df[target].values
    nonzero = vals[vals != 0] if (vals != 0).any() else vals
    q5 = np.quantile(nonzero, [0.2, 0.4, 0.6, 0.8])
    sk = syn_df[keys].values

    for addr, r in targets:
        row = orig_df[orig_df["_addr_id"] == addr]
        if row.empty:
            continue
        true_bucket = int(np.searchsorted(q5, row[target].values[0]))
        ds = cdist(row[keys].values, sk, "euclidean").ravel()
        threshold = np.percentile(ds, 20) + 1e-8
        matches = np.where(ds < threshold)[0]
        if len(matches):
            pred_bucket = int(np.searchsorted(
                q5, syn_df.iloc[matches][target].median()))
            correct = pred_bucket == true_bucket
        else:
            correct = False
        results[addr] = dict(risk=r, correct=bool(correct))
    return results


def attack_class(syn_df, orig_df, risk, cols, cfg: Config):
    label = cfg.label_col
    targets = _attack_targets(risk, cfg)
    results = {}
    sv = syn_df[cols].values

    for addr, r in targets:
        row = orig_df[orig_df["_addr_id"] == addr]
        if row.empty:
            continue
        true_label = int(row[label].values[0])
        ds = cdist(row[cols].values, sv, "euclidean").ravel()
        threshold = np.percentile(ds, 20) + 1e-8
        matches = np.where(ds < threshold)[0]
        if len(matches):
            mode_vals = syn_df.iloc[matches][label].round().mode().values
            correct = int(mode_vals[0]) == true_label if len(mode_vals) else False
        else:
            correct = False
        results[addr] = dict(risk=r, correct=bool(correct))
    return results


def attack_membership(syn_df, orig_df, risk, cols, cfg: Config):
    targets = _attack_targets(risk, cfg)
    results = {}
    rng = np.random.RandomState(cfg.seed)
    sv = syn_df[cols].values

    for addr, r in targets:
        row = orig_df[orig_df["_addr_id"] == addr]
        if row.empty:
            continue
        pat = row[cols].values
        min_d = cdist(pat, sv, "euclidean").ravel().min()
        std = np.abs(pat).mean() * 0.1 + 1e-4
        shadow_dists = [
            cdist(pat + rng.normal(0, std, pat.shape), sv, "euclidean").ravel().min()
            for _ in range(50)
        ]
        sm, ss = np.mean(shadow_dists), np.std(shadow_dists) + 1e-8
        z_score = (sm - min_d) / ss
        results[addr] = dict(risk=r, z_score=float(z_score),
                             inferred=bool(min_d < sm - ss))
    return results


def run_all_attacks(syn_df, orig_df, risk, cfg: Config, label: str = ""):
    """Run all 4 privacy attacks and return success rates."""
    cols = _attack_cols(syn_df, orig_df, cfg)
    if not cols:
        cols = [c for c in cfg.attack_key_cols
                if c in syn_df.columns and c in orig_df.columns]

    link = attack_linkability(syn_df, orig_df, risk, cols, cfg)
    attr = attack_attribute(syn_df, orig_df, risk, cols, cfg)
    cls = attack_class(syn_df, orig_df, risk, cols, cfg)
    memb = attack_membership(syn_df, orig_df, risk, cols, cfg)

    def rate(d, key):
        return sum(1 for v in d.values() if v.get(key, False)) / len(d) if d else 0

    rates = dict(
        link=rate(link, "linked"),
        attr=rate(attr, "correct"),
        cls=rate(cls, "correct"),
        memb=rate(memb, "inferred"),
    )
    print(f"  Privacy attacks [{label}]:  "
          f"Link={rates['link']:.0%}  Attr={rates['attr']:.0%}  "
          f"Class={rates['cls']:.0%}  Memb={rates['memb']:.0%}")
    return rates
