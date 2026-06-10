from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def smoke_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    freqs = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SmokeDenoiser(nn.Module):
    """Tiny smoke-run denoiser matching conditional_ddpm_smoke_best.pt."""

    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int = 24, time_dim: int = 24):
        super().__init__()
        self.time_dim = time_dim
        self.input_proj = nn.Linear(input_dim + condition_dim + time_dim, hidden_dim)
        self.hidden = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = smoke_timestep_embedding(t, self.time_dim).unsqueeze(1).repeat(1, x.shape[1], 1)
        h = torch.cat([x, c, t_emb], dim=-1)
        return self.output(self.hidden(self.input_proj(h)))


class SmokeScheduler:
    def __init__(
        self,
        num_steps: int = 20,
        beta_start: float = 1e-4,
        beta_end: float = 1.5e-2,
        device: str = "cpu",
    ):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_omb = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    def sample_reverse(self, model: nn.Module, cond: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        x = torch.randn(shape, device=self.device, dtype=torch.float32)
        model.eval()
        for step in reversed(range(self.num_steps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            with torch.no_grad():
                eps_theta = model(x, cond, t)
            beta_t = self.betas[step]
            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)
            z = torch.randn_like(x) if step > 0 else torch.zeros_like(x)
            x = coef1 * (x - coef2 * eps_theta) + torch.sqrt(beta_t) * z
        return x
