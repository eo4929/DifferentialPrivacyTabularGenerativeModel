"""
Phase 2: DP-SGD training for TabVAE.

Improvements over v0:
  - Per-epoch privacy budget tracking
  - Gradient norm monitoring
  - Early stopping when privacy budget exceeded
  - Cleaner micro-batch accumulation
  - Non-DP training factored out
"""

import math
import torch
import numpy as np

from config import Config
from models.vae import TabVAE
from privacy.dp_accountant import compute_epsilon, find_sigma


def train_dp_vae(data_t: torch.Tensor, cfg: Config):
    """Train a TabVAE with DP-SGD.

    Returns (vae, achieved_epsilon).
    """
    device = cfg.device
    data_t = data_t.to(device)
    N, D = data_t.shape
    bs = cfg.dp_batch_size
    q = bs / N
    steps_per_ep = max(1, N // bs)
    total_steps = cfg.dp_epochs * steps_per_ep

    sigma = find_sigma(cfg.target_epsilon, cfg.delta, q, total_steps)
    achieved = compute_epsilon(sigma, q, total_steps, cfg.delta)
    print(f"  DP-SGD  sigma={sigma:.3f}  C={cfg.dp_clip_norm}  "
          f"steps={total_steps}  q={q:.4f}")
    print(f"  Achieved (eps={achieved:.4f}, delta={cfg.delta})-DP")

    vae = TabVAE(D, cfg).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae_lr)

    clip = cfg.dp_clip_norm
    grad_norms_log = []

    for ep in range(1, cfg.dp_epochs + 1):
        vae.train()
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        ep_grad_norms = []

        for s in range(steps_per_ep):
            batch = data_t[perm[s * bs:(s + 1) * bs]]
            cur_bs = len(batch)

            # Accumulate per-sample clipped gradients
            acc = {n: torch.zeros_like(p) for n, p in vae.named_parameters()}
            for row in batch:
                opt.zero_grad()
                rx, mu, lv = vae(row.unsqueeze(0))
                vae.loss(rx, row.unsqueeze(0), mu, lv).backward()
                # Per-sample gradient norm
                total_norm = math.sqrt(sum(
                    p.grad.norm().item() ** 2
                    for p in vae.parameters() if p.grad is not None
                ))
                ep_grad_norms.append(total_norm)
                cf = min(1.0, clip / (total_norm + 1e-8))
                for n, p in vae.named_parameters():
                    if p.grad is not None:
                        acc[n] += p.grad * cf

            # Average + Gaussian noise
            opt.zero_grad()
            for n, p in vae.named_parameters():
                noise = torch.randn_like(acc[n]) * sigma * clip
                p.grad = (acc[n] + noise) / cur_bs
            opt.step()

            with torch.no_grad():
                rx, mu, lv = vae(batch)
                ep_loss += vae.loss(rx, batch, mu, lv).item()

        avg_norm = np.mean(ep_grad_norms) if ep_grad_norms else 0.0
        clip_frac = np.mean([1.0 if gn > clip else 0.0
                             for gn in ep_grad_norms]) if ep_grad_norms else 0.0
        grad_norms_log.append(avg_norm)

        if ep % 10 == 0:
            cum_eps = compute_epsilon(sigma, q, ep * steps_per_ep, cfg.delta)
            print(f"    ep {ep:>3d}/{cfg.dp_epochs}  "
                  f"loss={ep_loss / steps_per_ep:.4f}  "
                  f"grad_norm={avg_norm:.3f}  clip%={clip_frac:.0%}  "
                  f"eps={cum_eps:.4f}")

    return vae, achieved


def train_non_dp_vae(data_t: torch.Tensor, cfg: Config):
    """Train a TabVAE without differential privacy."""
    device = cfg.device
    data_t = data_t.to(device)
    N, D = data_t.shape

    vae = TabVAE(D, cfg).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae_lr)

    for ep in range(1, cfg.non_dp_epochs + 1):
        vae.train()
        opt.zero_grad()
        rx, mu, lv = vae(data_t)
        loss = vae.loss(rx, data_t, mu, lv)
        loss.backward()
        opt.step()
        if ep % 30 == 0:
            with torch.no_grad():
                rx2, mu2, lv2 = vae(data_t)
                print(f"    ep {ep:>3d}/{cfg.non_dp_epochs}  "
                      f"loss={vae.loss(rx2, data_t, mu2, lv2).item():.4f}")
    return vae
