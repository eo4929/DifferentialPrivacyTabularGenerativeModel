"""
Phase 1: Privacy-Aware Multi-View Hypergraph Encoder.

Key improvements over v0:
  - Degree-normalized hypergraph incidence (HGNN-style)
  - LayerNorm + residual connections in MixedAttentionLayer
  - Gated fusion of hypergraph / graph branches
  - Vectorised sensitivity_loss (no Python for-loop)
  - Cosine-annealing LR schedule
  - GPU support
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import Config


# ── Hypergraph construction ──────────────────────────────────────────

class HypergraphBuilder:
    """Build a hypergraph from multi-view kNN neighbourhoods."""

    def __init__(self, view_features: dict, n_nodes: int, k: int = 5):
        self.views = view_features
        self.n_nodes = n_nodes
        self.k = min(k, n_nodes - 1)

    def build_hypergraph(self, device: torch.device = torch.device("cpu")):
        hyperedges, he_types = [], []
        for vn, feats in self.views.items():
            nn_m = NearestNeighbors(n_neighbors=self.k + 1).fit(feats)
            _, idx = nn_m.kneighbors(feats)
            for i in range(self.n_nodes):
                hyperedges.append(idx[i].tolist())
                he_types.append(vn)

        n_he = len(hyperedges)
        H = torch.zeros(self.n_nodes, n_he, device=device)
        for ei, members in enumerate(hyperedges):
            for ni in members:
                H[ni, ei] = 1.0

        info = {"n_he": n_he}
        for vn in self.views:
            info[f"n_{vn}"] = he_types.count(vn)
        return H, hyperedges, info

    @staticmethod
    def degree_normalise(H: torch.Tensor) -> torch.Tensor:
        """D_v^{-1/2} H D_e^{-1} — used for spectral-style propagation."""
        D_v = H.sum(dim=1).clamp(min=1)
        D_e = H.sum(dim=0).clamp(min=1)
        return D_v.pow(-0.5).unsqueeze(1) * H * D_e.pow(-1).unsqueeze(0)

    @staticmethod
    def build_clique_adjacency(H: torch.Tensor) -> torch.Tensor:
        adj = torch.mm(H, H.t())
        adj = (adj > 0).float()
        adj.fill_diagonal_(0)
        return adj


# ── Attention modules ────────────────────────────────────────────────

class HyperedgeAttention(nn.Module):
    """Aggregate node features into hyperedge representations."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.u = nn.Parameter(torch.randn(dim))
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x: torch.Tensor, H: torch.Tensor):
        h = self.W(x)
        N, E = H.shape
        sc = torch.matmul(h, self.u) / math.sqrt(h.size(1))
        sc = sc.unsqueeze(1).expand(N, E)
        mask = H > 0
        sc = sc.masked_fill(~mask, -1e9)
        A = F.softmax(sc, dim=0) * mask.float()
        return torch.mm(A.t(), h), A


class NodeAttention(nn.Module):
    """Aggregate hyperedge representations back to nodes."""

    def __init__(self, dim: int):
        super().__init__()
        self.W_e = nn.Linear(dim, dim, bias=False)
        self.W_n = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.W_e.weight)
        nn.init.xavier_uniform_(self.W_n.weight)

    def forward(self, x: torch.Tensor, e_repr: torch.Tensor, H: torch.Tensor):
        he = self.W_e(e_repr)
        hn = self.W_n(x)
        B = torch.mm(hn, he.t()) / math.sqrt(he.size(1))
        mask = H > 0
        B = B.masked_fill(~mask, -1e9)
        B = F.softmax(B, dim=1) * mask.float()
        return F.elu(torch.mm(B, self.W_e(e_repr))), B


class GraphAttention(nn.Module):
    """Standard GAT-style attention on clique-expanded adjacency."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Linear(2 * dim, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        h = self.W(x)
        N = h.size(0)
        hi = h.unsqueeze(1).expand(N, N, -1)
        hj = h.unsqueeze(0).expand(N, N, -1)
        e = F.leaky_relu(self.attn(torch.cat([hi, hj], -1)).squeeze(-1), 0.2)
        conn = adj + torch.eye(N, device=adj.device)
        e = e.masked_fill(conn == 0, -1e9)
        G = F.softmax(e, dim=1)
        return F.elu(torch.mm(G, h)), G


# ── Mixed attention layer ────────────────────────────────────────────

class MixedAttentionLayer(nn.Module):
    """Cross-view mixed attention with residual connections and LayerNorm."""

    def __init__(self, dim: int):
        super().__init__()
        self.he_attn = HyperedgeAttention(dim)
        self.node_attn = NodeAttention(dim)
        self.graph_attn = GraphAttention(dim)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: torch.Tensor, adj: torch.Tensor):
        # Hyperedge path: node -> hyperedge -> node
        e_repr, A = self.he_attn(x, H)
        _, B = self.node_attn(x, e_repr, H)

        # Graph path
        _, G = self.graph_attn(x, adj)

        # Cross-view modulation
        mask_H = (H > 0).float()
        he_sz = mask_H.sum(0).clamp(min=1)
        M = torch.mm(G, mask_H) / he_sz.unsqueeze(0)

        A_hat = F.softmax(A.t() + M.t(), dim=1) * mask_H.t()
        e_mix = torch.mm(A_hat, self.he_attn.W(x))
        B_hat = F.softmax(B + M, dim=1) * mask_H
        z_h = F.elu(torch.mm(B_hat, self.node_attn.W_e(e_mix)))

        G_hat = F.softmax(G + torch.mm(B, B.t()) + torch.mm(A, A.t()), dim=1)
        z_g = F.elu(torch.mm(G_hat, self.graph_attn.W(x)))

        # Residual + LayerNorm
        z_h = self.norm_h(x + z_h)
        z_g = self.norm_g(x + z_g)
        return z_h, z_g


# ── Multi-View Encoder ───────────────────────────────────────────────

class MultiViewEncoder(nn.Module):
    def __init__(self, d_txn: int, d_ether: int, d_erc20: int, cfg: Config):
        super().__init__()
        hid = cfg.hidden_dim
        emb = cfg.embed_dim
        proj = cfg.proj_dim

        self.proj_txn = nn.Linear(d_txn, hid)
        self.proj_ether = nn.Linear(d_ether, hid)
        self.proj_erc20 = nn.Linear(d_erc20, hid)
        self.combine = nn.Linear(hid * 3, hid)
        self.combine_norm = nn.LayerNorm(hid)

        self.layer1 = MixedAttentionLayer(hid)
        self.layer2 = MixedAttentionLayer(hid)
        self.gate = nn.Linear(hid * 2, hid)

        self.out_proj = nn.Linear(hid, emb)
        self.proj_h = nn.Sequential(nn.Linear(emb, hid), nn.ELU(),
                                    nn.Linear(hid, proj))
        self.proj_g = nn.Sequential(nn.Linear(emb, hid), nn.ELU(),
                                    nn.Linear(hid, proj))
        self.fraud_head = nn.Linear(emb, 2)

    def _unified(self, xt, xe, xr):
        h = torch.cat([
            F.elu(self.proj_txn(xt)),
            F.elu(self.proj_ether(xe)),
            F.elu(self.proj_erc20(xr)),
        ], dim=-1)
        return self.combine_norm(F.elu(self.combine(h)))

    def forward(self, xt, xe, xr, H, adj):
        x = self._unified(xt, xe, xr)
        zh1, zg1 = self.layer1(x, H, adj)
        # Gated fusion before layer 2
        g = torch.sigmoid(self.gate(torch.cat([zh1, zg1], dim=-1)))
        x2 = g * zh1 + (1 - g) * zg1
        zh2, zg2 = self.layer2(x2, H, adj)
        return self.out_proj(zh2), self.out_proj(zg2)

    def get_projections(self, zh, zg):
        return self.proj_h(zh), self.proj_g(zg)

    @torch.no_grad()
    def get_embeddings(self, xt, xe, xr, H, adj):
        return self.forward(xt, xe, xr, H, adj)

    @torch.no_grad()
    def project_features(self, xt, xe, xr):
        return self.out_proj(self._unified(xt, xe, xr))


# ── Loss functions ───────────────────────────────────────────────────

def info_nce_multiview(ph: torch.Tensor, pg: torch.Tensor, tau: float = 0.5):
    phn = F.normalize(ph, dim=1)
    pgn = F.normalize(pg, dim=1)
    logits = torch.mm(phn, pgn.t()) / tau
    labels = torch.arange(ph.size(0), device=ph.device)
    return (F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)) / 2


def sensitivity_loss(z: torch.Tensor, risk_scores: torch.Tensor,
                     prototypes: torch.Tensor, threshold: float = 0.25):
    """Vectorised sensitivity regularisation (no Python loop)."""
    if prototypes.size(0) == 0:
        return torch.tensor(0.0, device=z.device)
    mask = risk_scores > threshold
    if not mask.any():
        return torch.tensor(0.0, device=z.device)
    z_risky = z[mask]
    r_risky = risk_scores[mask]
    dists = torch.cdist(z_risky, prototypes)
    nearest = prototypes[dists.argmin(dim=1)]
    per_node = (z_risky - nearest).pow(2).mean(dim=1)
    return (r_risky * per_node).sum() / mask.sum()


# ── Risk scoring ─────────────────────────────────────────────────────

def compute_risk_scores(df, num_features: list) -> dict:
    X = df[num_features].values.astype(np.float64)
    sc = MinMaxScaler().fit_transform(X)
    k = min(20, len(X) - 1)
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit_predict(sc)
    scores = -lof.negative_outlier_factor_
    lof_n = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    means, stds = sc.mean(0), sc.std(0) + 1e-8
    rarity = np.clip(np.abs(sc - means).mean(1) / stds.mean() / 3, 0, 1)
    return {
        df.iloc[i]["addr_id"]: float(np.clip(0.5 * lof_n[i] + 0.5 * rarity[i], 0, 1))
        for i in range(len(df))
    }


# ── Training loop ────────────────────────────────────────────────────

def train_encoder(train_df, cfg: Config):
    """Train the multi-view encoder and return (embeddings, risk_scores)."""
    device = cfg.device
    N = len(train_df)

    # Subsample if dataset is too large for full graph
    if N > cfg.max_graph_nodes:
        sub_idx, _ = train_test_split(
            np.arange(N), train_size=cfg.max_graph_nodes,
            stratify=train_df[cfg.label_col].values, random_state=cfg.seed)
        sub_idx = sorted(sub_idx)
        sub_df = train_df.iloc[sub_idx].reset_index(drop=True)
        print(f"  Subsampled {cfg.max_graph_nodes}/{N} for encoder")
    else:
        sub_idx = list(range(N))
        sub_df = train_df.reset_index(drop=True)
    Ns = len(sub_df)

    # Prepare per-view features
    scalers, feats_np, feats_t = {}, {}, {}
    for vn, vc in cfg.all_views.items():
        s = MinMaxScaler()
        f = s.fit_transform(sub_df[vc].values.astype(np.float64))
        scalers[vn] = s
        feats_np[vn] = f
        feats_t[vn] = torch.FloatTensor(f).to(device)

    # Build hypergraph
    gb = HypergraphBuilder(feats_np, Ns, k=cfg.encoder_k)
    H, _, info = gb.build_hypergraph(device)
    adj = gb.build_clique_adjacency(H)
    print(f"  Hypergraph: {Ns} nodes x {info['n_he']} hyperedges  "
          f"adj_edges={int(adj.sum())}")

    # Risk scores (on sub_df for training)
    risk_sub = compute_risk_scores(sub_df, cfg.all_num_features)
    risk_tensor = torch.FloatTensor(
        [risk_sub[sub_df.iloc[i]["addr_id"]] for i in range(Ns)]
    ).to(device)

    # Model
    model = MultiViewEncoder(
        len(cfg.view_txn), len(cfg.view_ether), len(cfg.view_erc20), cfg
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.encoder_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.encoder_epochs, eta_min=cfg.encoder_lr * 0.01)

    fy = torch.LongTensor(sub_df[cfg.label_col].values).to(device)
    fraud_idx = (fy == 1).nonzero(as_tuple=True)[0]
    normal_idx = (fy == 0).nonzero(as_tuple=True)[0]

    print(f"  Training encoder ({Ns} nodes, {cfg.encoder_epochs} ep) ...")
    for ep in range(1, cfg.encoder_epochs + 1):
        model.train()
        opt.zero_grad()

        zh, zg = model(feats_t["txn"], feats_t["ether"], feats_t["erc20"], H, adj)
        ph, pg = model.get_projections(zh, zg)

        l_cl = info_nce_multiview(ph, pg, cfg.contrastive_tau)

        # Prototypes
        protos = []
        if len(fraud_idx) > 0:
            protos.append(zh[fraud_idx].mean(0, keepdim=True))
        if len(normal_idx) > 0:
            protos.append(zh[normal_idx].mean(0, keepdim=True))
        proto = torch.cat(protos, 0) if protos else zh[:1]

        l_se = sensitivity_loss(zh, risk_tensor, proto)
        l_ta = F.cross_entropy(model.fraud_head(zh), fy)
        loss = l_cl + cfg.lambda_sensitivity * l_se + cfg.lambda_task * l_ta

        loss.backward()
        opt.step()
        scheduler.step()

        if ep % 25 == 0:
            print(f"    ep {ep:>4d} | L={loss.item():.4f}  "
                  f"CL={l_cl.item():.4f}  SE={l_se.item():.4f}  "
                  f"TA={l_ta.item():.4f}  lr={scheduler.get_last_lr()[0]:.1e}")

    # Produce embeddings for ALL training rows
    model.eval()
    emb = torch.zeros(N, cfg.embed_dim, device=device)
    zh_sub, _ = model.get_embeddings(
        feats_t["txn"], feats_t["ether"], feats_t["erc20"], H, adj)
    for li, gi in enumerate(sub_idx):
        emb[gi] = zh_sub[li]

    other = sorted(set(range(N)) - set(sub_idx))
    if other:
        od = train_df.iloc[other]
        pe = model.project_features(
            torch.FloatTensor(scalers["txn"].transform(od[cfg.view_txn].values)).to(device),
            torch.FloatTensor(scalers["ether"].transform(od[cfg.view_ether].values)).to(device),
            torch.FloatTensor(scalers["erc20"].transform(od[cfg.view_erc20].values)).to(device),
        )
        for j, gi in enumerate(other):
            emb[gi] = pe[j]

    risk_all = compute_risk_scores(train_df, cfg.all_num_features)
    return emb.cpu(), risk_all
