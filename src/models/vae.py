"""
Tabular VAE for synthetic data generation.

Improvements over v0:
  - LayerNorm (DP-safe, unlike BatchNorm which leaks cross-sample info)
  - Configurable beta weighting for KL term
  - Proper weight initialisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class TabVAE(nn.Module):
    def __init__(self, input_dim: int, cfg: Config):
        super().__init__()
        hid = cfg.vae_hidden
        lat = cfg.vae_latent
        self.latent_dim = lat

        self.enc = nn.Sequential(
            nn.Linear(input_dim, hid), nn.LayerNorm(hid), nn.ELU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ELU(),
        )
        self.mu = nn.Linear(hid, lat)
        self.lv = nn.Linear(hid, lat)
        self.dec = nn.Sequential(
            nn.Linear(lat, hid), nn.LayerNorm(hid), nn.ELU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid), nn.ELU(),
            nn.Linear(hid, input_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.mu(h), self.lv(h)

    def reparameterise(self, mu: torch.Tensor, lv: torch.Tensor):
        return mu + torch.randn_like(mu) * (0.5 * lv).exp()

    def decode(self, z: torch.Tensor):
        return self.dec(z)

    def forward(self, x: torch.Tensor):
        mu, lv = self.encode(x)
        z = self.reparameterise(mu, lv)
        return self.decode(z), mu, lv

    def loss(self, rx: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, lv: torch.Tensor, beta: float = 1.0):
        recon = F.mse_loss(rx, x, reduction="sum") / x.size(0)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum() / x.size(0)
        return recon + beta * kl

    @torch.no_grad()
    def sample(self, n: int, device: torch.device = torch.device("cpu")):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)
