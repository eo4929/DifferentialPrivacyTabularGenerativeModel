"""
Baseline synthetic data generators: CTGAN and TVAE.
"""

from config import Config

try:
    from ctgan import CTGAN as CTGANModel
    from ctgan import TVAE as TVAEModel
    HAS_CTGAN = True
except ImportError:
    HAS_CTGAN = False


def gen_ctgan(train_data, n_syn: int, feat_cols: list, cfg: Config):
    if not HAS_CTGAN:
        print("  [skip] ctgan not installed (pip install ctgan)")
        return None
    label = cfg.label_col
    print(f"\n  -- Training CTGAN (epochs={cfg.ctgan_epochs}) --")
    m = CTGANModel(epochs=cfg.ctgan_epochs, batch_size=cfg.baseline_batch_size,
                   verbose=False)
    m.fit(train_data[feat_cols + [label]], discrete_columns=[label])
    syn = m.sample(n_syn)
    print(f"  CTGAN: {len(syn)} rows  fraud={syn[label].mean():.1%}")
    return syn


def gen_tvae(train_data, n_syn: int, feat_cols: list, cfg: Config):
    if not HAS_CTGAN:
        print("  [skip] ctgan not installed (pip install ctgan)")
        return None
    label = cfg.label_col
    print(f"\n  -- Training TVAE (epochs={cfg.tvae_epochs}) --")
    m = TVAEModel(epochs=cfg.tvae_epochs, batch_size=cfg.baseline_batch_size,
                  verbose=False)
    m.fit(train_data[feat_cols + [label]], discrete_columns=[label])
    syn = m.sample(n_syn)
    print(f"  TVAE: {len(syn)} rows  fraud={syn[label].mean():.1%}")
    return syn
