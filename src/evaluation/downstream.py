"""
Downstream fraud detection evaluation (TSTR — Train on Synthetic, Test on Real).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              roc_auc_score, average_precision_score)

from config import Config

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False


def _compute_metrics(y_true, y_pred, y_prob):
    m = dict(
        F1=f1_score(y_true, y_pred, zero_division=0),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
    )
    if len(np.unique(y_true)) > 1 and y_prob is not None:
        m["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        m["PR-AUC"] = average_precision_score(y_true, y_prob)
    else:
        m["ROC-AUC"] = m["PR-AUC"] = float("nan")
    return m


def evaluate_downstream(syn_df, test_df, feat_cols, cfg: Config):
    """Train classifiers on synthetic data, evaluate on real test data."""
    label = cfg.label_col
    Xs = syn_df[feat_cols].values.astype(np.float32)
    ys = syn_df[label].values.astype(int)
    Xt = test_df[feat_cols].values.astype(np.float32)
    yt = test_df[label].values.astype(int)

    if len(np.unique(ys)) < 2:
        print("    [warn] single-class synthetic data — skipping TSTR")
        return {}

    models = {}
    models["RF"] = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=cfg.seed,
        class_weight="balanced")
    if HAS_XGB:
        n_pos = max(ys.sum(), 1)
        n_neg = max(len(ys) - n_pos, 1)
        models["XGB"] = XGBClassifier(
            n_estimators=100, max_depth=5, scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss", random_state=cfg.seed, verbosity=0)
    if HAS_TABNET and len(np.unique(ys)) >= 2:
        models["TabNet"] = TabNetClassifier(verbose=0, seed=cfg.seed)
    if HAS_TABPFN:
        try:
            models["TabPFN"] = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, n_estimators=4, device="cpu",
                ignore_pretraining_limits=True)
        except Exception:
            pass

    results = {}
    for name, clf in models.items():
        try:
            if name == "TabNet":
                clf.fit(Xs, ys, eval_set=[(Xt, yt)], max_epochs=50,
                        patience=10, batch_size=min(256, len(Xs)))
            else:
                clf.fit(Xs, ys)
            yp = clf.predict(Xt)
            ypr = None
            if hasattr(clf, "predict_proba"):
                pr = clf.predict_proba(Xt)
                if pr.shape[1] >= 2:
                    ypr = pr[:, 1]
            results[name] = _compute_metrics(yt, yp, ypr)
        except Exception as e:
            print(f"    [error] {name}: {e}")
            results[name] = {k: float("nan")
                             for k in ("F1", "Precision", "Recall", "ROC-AUC", "PR-AUC")}
    return results
