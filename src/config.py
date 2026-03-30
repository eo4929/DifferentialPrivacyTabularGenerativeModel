"""
Configuration for DP Multi-View Synthetic Data Pipeline.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict

# Fix for non-ASCII temp directory paths (e.g. Korean Windows usernames)
if any(ord(c) > 127 for c in tempfile.gettempdir()):
    _ascii_tmp = os.path.join("C:\\", "tmp_joblib")
    os.makedirs(_ascii_tmp, exist_ok=True)
    os.environ["JOBLIB_TEMP_FOLDER"] = _ascii_tmp

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    # --- Paths ---
    data_path: str = (
        r"C:\Users\AI기술팀\Documents\DifferentialPrivacyTabularGenerativeModel"
        r"\data\preprocessed_Ethereum_cleaned_v2.csv"
    )
    label_col: str = "Fraud_Label"

    # --- Device / Seed ---
    device: torch.device = field(default_factory=get_device)
    seed: int = 42

    # --- Encoder (Phase 1) ---
    embed_dim: int = 16
    hidden_dim: int = 32
    proj_dim: int = 16
    max_graph_nodes: int = 2000
    encoder_epochs: int = 100
    encoder_lr: float = 5e-3
    encoder_k: int = 5
    lambda_sensitivity: float = 0.3
    lambda_task: float = 0.1
    contrastive_tau: float = 0.5

    # --- VAE (Phase 2) ---
    vae_hidden: int = 64
    vae_latent: int = 32
    vae_lr: float = 1e-3

    # --- DP ---
    target_epsilon: float = 10.0
    delta: float = 1e-5
    dp_batch_size: int = 64
    dp_epochs: int = 60
    dp_clip_norm: float = 1.0

    # --- Non-DP VAE ---
    non_dp_epochs: int = 120

    # --- Filtering (Phase 3) ---
    filter_threshold: float = 0.55
    exposure_lof_weight: float = 0.6
    exposure_knn_weight: float = 0.4
    risk_weight_exposure: float = 0.5
    risk_weight_proximity: float = 0.5

    # --- Baselines ---
    ctgan_epochs: int = 300
    tvae_epochs: int = 300
    baseline_batch_size: int = 500

    # --- Evaluation ---
    test_size: float = 0.3
    attack_top_n: int = 50
    attack_risk_threshold: float = 0.3

    # --- Feature groups ---
    view_txn: List[str] = field(default_factory=lambda: [
        "Avg min between sent tnx", "Avg min between received tnx",
        "Time Diff between first and last (Mins)", "Sent tnx", "Received Tnx",
        "Number of Created Contracts", "Unique Received From Addresses",
        "Unique Sent To Addresses",
        "total transactions (including tnx to create contract",
    ])
    view_ether: List[str] = field(default_factory=lambda: [
        "min value received", "max value received", "avg val received",
        "min val sent", "max val sent", "avg val sent",
        "min value sent to contract", "max val sent to contract",
        "avg value sent to contract",
        "total Ether sent", "total ether received",
        "total ether sent contracts", "total ether balance",
    ])
    view_erc20: List[str] = field(default_factory=lambda: [
        "Total ERC20 tnxs", "ERC20 total Ether received",
        "ERC20 total ether sent", "ERC20 total Ether sent contract",
        "ERC20 uniq sent addr", "ERC20 uniq rec addr",
        "ERC20 uniq sent addr.1", "ERC20 uniq rec contract addr",
        "ERC20 avg time between sent tnx", "ERC20 avg time between rec tnx",
        "ERC20 avg time between rec 2 tnx", "ERC20 avg time between contract tnx",
        "ERC20 min val rec", "ERC20 max val rec", "ERC20 avg val rec",
        "ERC20 min val sent", "ERC20 max val sent", "ERC20 avg val sent",
        "ERC20 min val sent contract", "ERC20 max val sent contract",
        "ERC20 avg val sent contract", "ERC20 uniq sent token name",
        "ERC20 uniq rec token name",
    ])
    attack_key_cols: List[str] = field(default_factory=lambda: [
        "Sent tnx", "Received Tnx", "Unique Received From Addresses",
        "Unique Sent To Addresses", "total Ether sent", "total ether received",
        "total ether balance", "Total ERC20 tnxs",
    ])

    @property
    def all_views(self) -> Dict[str, List[str]]:
        return {"txn": self.view_txn, "ether": self.view_ether, "erc20": self.view_erc20}

    @property
    def all_num_features(self) -> List[str]:
        return self.view_txn + self.view_ether + self.view_erc20
