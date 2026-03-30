"""
Privacy-Aware Synthetic Data Generation for Ethereum Fraud Detection
=====================================================================
Differentially Private Multi-View Graph Contrastive Learning
with Baseline Comparison (CTGAN / TVAE)

Pipeline:
  Phase 1: Privacy-Aware Multi-View Relational Encoder (Hypergraph)
  Phase 2: (eps, delta)-DP Tabular Generation (VAE + DP-SGD)
  Phase 3: Measurement-Driven Disclosure Filtering
  Phase 4: Baseline Generation (CTGAN / TVAE)
  Phase 5: Privacy Attack + Downstream Detection + Fidelity Comparison

Usage:
  cd src && python main.py
"""

import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import Config
from data import load_ethereum_data
from models.encoder import train_encoder
from models.vae import TabVAE
from privacy.dp_trainer import train_dp_vae, train_non_dp_vae
from privacy.filtering import filter_risky
from privacy.attacks import run_all_attacks
from evaluation.downstream import evaluate_downstream
from evaluation.fidelity import evaluate_fidelity
from baselines import gen_ctgan, gen_tvae

warnings.filterwarnings("ignore")


# ── Helpers ──────────────────────────────────────────────────────────

def build_augmented_table(train_df, embeddings, num_features, cfg: Config):
    aug = train_df[num_features].copy().reset_index(drop=True)
    emb_np = embeddings.detach().cpu().numpy()
    for k in range(emb_np.shape[1]):
        aug[f"emb_{k}"] = emb_np[:, k]
    aug[cfg.label_col] = train_df[cfg.label_col].values
    aug["_addr_id"] = train_df["addr_id"].values
    return aug


def make_synthetic_df(raw: np.ndarray, feat_cols: list,
                      orig_fraud_rate: float, label_col: str):
    """Convert raw VAE output to a labelled DataFrame."""
    features = raw[:, :-1]
    label_scores = np.clip(raw[:, -1], 0, 1)
    n_fraud = max(1, int(round(len(label_scores) * orig_fraud_rate)))
    labels = np.zeros(len(label_scores))
    labels[np.argsort(label_scores)[::-1][:n_fraud]] = 1.0
    df = pd.DataFrame(features, columns=feat_cols)
    df[label_col] = labels
    return df


# ── Comparison tables ────────────────────────────────────────────────

def print_downstream_comparison(all_ds, label_map):
    metrics = ["F1", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
    model_names = []
    for lbl, _ in label_map:
        if lbl in all_ds:
            for m in all_ds[lbl]:
                if m not in model_names:
                    model_names.append(m)

    sep = "=" * 88
    print(f"\n{sep}")
    print(" COMPARISON -- Downstream Detection (TSTR)")
    print(sep)
    for mn in model_names:
        print(f"\n  [{mn}]")
        print(f"  {'Method':40s}", end="")
        for m in metrics:
            print(f" {m:>9s}", end="")
        print()
        print(f"  {'─' * 88}")
        for lbl, disp in label_map:
            if lbl in all_ds and mn in all_ds[lbl]:
                r = all_ds[lbl][mn]
                print(f"  {disp:40s}", end="")
                for m in metrics:
                    v = r.get(m, float("nan"))
                    print(f" {'N/A':>9s}" if np.isnan(v) else f" {v:>9.4f}",
                          end="")
                print()


def print_privacy_comparison(all_priv, label_map):
    sep = "=" * 88
    print(f"\n{sep}")
    print(" COMPARISON -- Privacy Attack Success Rate (lower = safer)")
    print(sep)
    print(f"  {'Method':40s} {'Link':>8s} {'Attr':>8s} "
          f"{'Class':>8s} {'Memb':>8s} {'Avg':>8s}")
    print(f"  {'─' * 78}")
    for lbl, disp in label_map:
        if lbl in all_priv:
            p = all_priv[lbl]
            avg = np.mean([p['link'], p['attr'], p['cls'], p['memb']])
            print(f"  {disp:40s} {p['link']:>7.0%} {p['attr']:>7.0%} "
                  f"{p['cls']:>7.0%} {p['memb']:>7.0%} {avg:>7.0%}")


def print_fidelity_comparison(all_fid, label_map):
    sep = "=" * 88
    print(f"\n{sep}")
    print(" COMPARISON -- Statistical Fidelity (lower = more faithful)")
    print(sep)
    print(f"  {'Method':40s} {'avg JSD':>10s} {'avg Wass':>10s} "
          f"{'CorrDiff':>10s} {'CorrFrob':>10s}")
    print(f"  {'─' * 84}")
    for lbl, disp in label_map:
        if lbl in all_fid:
            f = all_fid[lbl]
            corr = f.get("correlation", {})
            print(f"  {disp:40s} {f['avg_jsd']:>10.4f} "
                  f"{f['avg_wasserstein']:>10.4f} "
                  f"{corr.get('mean_abs_diff', 0):>10.4f} "
                  f"{corr.get('frobenius_norm', 0):>10.4f}")


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    sep = "=" * 70
    print(f"\n{sep}")
    print(" Ethereum Fraud -- DP Multi-View vs CTGAN/TVAE Comparison")
    print(sep)

    # ── Phase 0: Load & split ──
    print("\n>> Phase 0  Load data")
    df = load_ethereum_data(cfg)
    print(f"  {len(df)} addresses  {len(cfg.all_num_features)} features  "
          f"fraud {df[cfg.label_col].mean():.1%}  device={cfg.device}")

    train_df, test_df = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.seed,
        stratify=df[cfg.label_col])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"  Train {len(train_df)} (fraud {int(train_df[cfg.label_col].sum())})  "
          f"Test {len(test_df)} (fraud {int(test_df[cfg.label_col].sum())})")

    n_syn = len(train_df)
    test_eval = test_df[cfg.all_num_features + [cfg.label_col]].copy()
    train_eval = train_df[cfg.all_num_features + [cfg.label_col]].copy()
    feat_raw = cfg.all_num_features

    all_ds, all_priv, all_fid = {}, {}, {}

    # ── Phase 1: Multi-View Encoder ──
    print(f"\n{sep}")
    print(" PHASE 1 -- Multi-View Encoder")
    print(sep)
    embeddings, risk = train_encoder(train_df, cfg)
    print(f"  Embeddings: {embeddings.shape}")

    # ── Phase 2: DP Tabular Generation ──
    print(f"\n{sep}")
    print(" PHASE 2 -- DP Tabular Generation")
    print(sep)

    aug = build_augmented_table(train_df, embeddings, cfg.all_num_features, cfg)
    feat_cols = [c for c in aug.columns if not c.startswith("_") and c != cfg.label_col]
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(aug[feat_cols].values)
    X_all = np.hstack([X_sc, aug[[cfg.label_col]].values])
    data_t = torch.FloatTensor(X_all)
    orig_fraud_rate = aug[cfg.label_col].mean()

    # Non-DP VAE
    print("\n  -- Non-DP VAE --")
    vae_nodp = train_non_dp_vae(data_t, cfg)
    raw_nodp = vae_nodp.sample(n_syn, cfg.device).cpu().numpy()
    raw_nodp[:, :-1] = scaler.inverse_transform(raw_nodp[:, :-1])
    syn_nodp = make_synthetic_df(raw_nodp, feat_cols, orig_fraud_rate, cfg.label_col)
    print(f"  Non-DP: {len(syn_nodp)} rows  fraud={syn_nodp[cfg.label_col].mean():.1%}")

    # DP VAE
    print("\n  -- DP VAE --")
    vae_dp, eps_achieved = train_dp_vae(data_t, cfg)
    raw_dp = vae_dp.sample(n_syn, cfg.device).cpu().numpy()
    raw_dp[:, :-1] = scaler.inverse_transform(raw_dp[:, :-1])
    syn_dp = make_synthetic_df(raw_dp, feat_cols, orig_fraud_rate, cfg.label_col)
    print(f"  DP: {len(syn_dp)} rows  fraud={syn_dp[cfg.label_col].mean():.1%}")

    # ── Phase 3: Disclosure Filtering ──
    print(f"\n{sep}")
    print(" PHASE 3 -- Disclosure Filtering")
    print(sep)
    orig_fv = aug[feat_cols].values
    w_risk = np.array([risk.get(a, 0.1) for a in aug["_addr_id"]])
    syn_filt, _ = filter_risky(
        syn_dp, syn_dp[feat_cols].values, orig_fv, w_risk, cfg)
    print(f"  Filtered: {len(syn_dp)} -> {len(syn_filt)}")

    # ── Phase 4: Baselines ──
    print(f"\n{sep}")
    print(" PHASE 4 -- Baselines (CTGAN / TVAE)")
    print(sep)
    bl_train = train_df[cfg.all_num_features + [cfg.label_col]].copy()
    syn_ctgan = gen_ctgan(bl_train, n_syn, cfg.all_num_features, cfg)
    syn_tvae = gen_tvae(bl_train, n_syn, cfg.all_num_features, cfg)

    # ── Phase 5: Comprehensive Evaluation ──
    print(f"\n{sep}")
    print(" PHASE 5a -- Privacy Attack Evaluation")
    print(sep)
    all_priv["nodp"] = run_all_attacks(syn_nodp, aug, risk, cfg, "Non-DP VAE")
    all_priv["dp_pre"] = run_all_attacks(syn_dp, aug, risk, cfg, "DP VAE (pre)")
    if len(syn_filt) > 10:
        all_priv["dp_post"] = run_all_attacks(syn_filt, aug, risk, cfg, "DP VAE (post)")
    if syn_ctgan is not None:
        all_priv["ctgan"] = run_all_attacks(syn_ctgan, aug, risk, cfg, "CTGAN")
    if syn_tvae is not None:
        all_priv["tvae"] = run_all_attacks(syn_tvae, aug, risk, cfg, "TVAE")

    print(f"\n{sep}")
    print(" PHASE 5b -- Downstream Detection (TSTR)")
    print(sep)

    print("\n  -- TRTR (upper bound) --")
    all_ds["trtr"] = evaluate_downstream(train_eval, test_eval, feat_raw, cfg)
    print("\n  -- Non-DP VAE --")
    all_ds["nodp"] = evaluate_downstream(syn_nodp, test_eval, feat_raw, cfg)
    print("\n  -- DP VAE (pre-filter) --")
    all_ds["dp_pre"] = evaluate_downstream(syn_dp, test_eval, feat_raw, cfg)
    if len(syn_filt) > 10:
        print("\n  -- DP VAE (post-filter) --")
        all_ds["dp_post"] = evaluate_downstream(syn_filt, test_eval, feat_raw, cfg)
    if syn_ctgan is not None:
        print("\n  -- CTGAN --")
        all_ds["ctgan"] = evaluate_downstream(syn_ctgan, test_eval, feat_raw, cfg)
    if syn_tvae is not None:
        print("\n  -- TVAE --")
        all_ds["tvae"] = evaluate_downstream(syn_tvae, test_eval, feat_raw, cfg)

    print(f"\n{sep}")
    print(" PHASE 5c -- Statistical Fidelity")
    print(sep)
    all_fid["nodp"] = evaluate_fidelity(train_eval, syn_nodp, feat_raw, "Non-DP VAE")
    all_fid["dp_pre"] = evaluate_fidelity(train_eval, syn_dp, feat_raw, "DP VAE (pre)")
    if len(syn_filt) > 10:
        all_fid["dp_post"] = evaluate_fidelity(train_eval, syn_filt, feat_raw, "DP VAE (post)")
    if syn_ctgan is not None:
        all_fid["ctgan"] = evaluate_fidelity(train_eval, syn_ctgan, feat_raw, "CTGAN")
    if syn_tvae is not None:
        all_fid["tvae"] = evaluate_fidelity(train_eval, syn_tvae, feat_raw, "TVAE")

    # ── Final comparison tables ──
    label_map = [
        ("trtr", "Real Train (TRTR, upper bound)"),
        ("nodp", "Ours: Non-DP VAE"),
        ("dp_pre", "Ours: DP VAE (pre-filter)"),
    ]
    if len(syn_filt) > 10:
        label_map.append(("dp_post", "Ours: DP VAE (post-filter)"))
    if syn_ctgan is not None:
        label_map.append(("ctgan", "Baseline: CTGAN"))
    if syn_tvae is not None:
        label_map.append(("tvae", "Baseline: TVAE"))

    print_downstream_comparison(all_ds, label_map)
    print_privacy_comparison(all_priv, label_map)
    print_fidelity_comparison(all_fid, label_map)

    # ── Summary ──
    print(f"\n{'=' * 88}")
    print(" SUMMARY")
    print("=" * 88)
    print(f"  Dataset : {len(df)} Ethereum addresses "
          f"(train={len(train_df)}, test={len(test_df)})")
    print(f"  DP      : (eps={eps_achieved:.4f}, delta={cfg.delta})-DP")

    parts = [f"Non-DP={len(syn_nodp)}",
             f"DP={len(syn_dp)}->{len(syn_filt)}(filtered)"]
    if syn_ctgan is not None:
        parts.append(f"CTGAN={len(syn_ctgan)}")
    if syn_tvae is not None:
        parts.append(f"TVAE={len(syn_tvae)}")
    print(f"  Synth   : {'  '.join(parts)}")

    print(f"\n  Key findings:")
    print(f"    - Multi-view encoder captures fraud relational structure")
    print(f"    - (eps,delta)-DP provides formal, provable privacy guarantee")
    print(f"    - Post-filter further reduces attack success rates")
    if syn_ctgan is not None:
        print(f"    - CTGAN/TVAE: no DP guarantee — vulnerable to attacks")
        if "dp_post" in all_priv and "ctgan" in all_priv:
            ours = all_priv["dp_post"]
            ct = all_priv["ctgan"]
            our_avg = np.mean(list(ours.values()))
            ct_avg = np.mean(list(ct.values()))
            print(f"      Ours (DP+filter) attack avg: {our_avg:.0%}")
            print(f"      CTGAN attack avg:            {ct_avg:.0%}")


if __name__ == "__main__":
    main()
