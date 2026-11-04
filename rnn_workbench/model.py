"""Continuous-time rate RNN with separate excitatory and inhibitory populations.

Dynamics (Euler-discretized, alpha = dt / tau):

    x[t+1] = (1 - alpha) * x[t] + alpha * (W_rec @ r[t] + W_in @ u[t] + b + I_pert[t]) + noise
    r[t]   = relu(x[t])        (rates are clamped to 0 for silenced units)
    z[t]   = W_out @ r_E[t]    (readout from excitatory units only)

Dale's law: every presynaptic unit is either excitatory (all outgoing weights >= 0)
or inhibitory (all outgoing weights <= 0). This is enforced by construction:
W_rec = relu(W_raw) * sign, where sign[:, j] = +1 for E units j and -1 for I units j.
Following Song, Yang & Wang (2016), inputs are non-negative and only E units project
to the readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    n_units: int = 100
    frac_exc: float = 0.8
    n_inputs: int = 3
    n_outputs: int = 3
    dt: float = 20.0  # ms
    tau: float = 100.0  # ms
    noise_std: float = 0.05  # recurrent noise, in units of x, per sqrt(step)
    dale: bool = True
    # Optional areas (e.g. insula = 0, a downstream decision area = 1). Cross-area
    # connections come only from E units: area a feeds area a + 1, and with `feedback`
    # every later area also projects back. I units stay local.
    area_of_unit: list | None = None   # one area id per unit; None = one area
    input_areas: list | None = None    # per input, the areas it reaches; None = all
    readout_area: int | None = None    # only this area's E units drive the outputs
    feedback: bool = True


class EIRNN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        n = cfg.n_units
        self.n_exc = int(round(cfg.frac_exc * n)) if cfg.dale else n
        self.alpha = cfg.dt / cfg.tau

        is_exc = torch.zeros(n, dtype=torch.bool)
        is_exc[: self.n_exc] = True
        self.register_buffer("is_exc", is_exc)
        sign = torch.where(is_exc, 1.0, -1.0) if cfg.dale else torch.ones(n)
        # Column j holds unit j's outgoing weights.
        self.register_buffer("sign", sign.unsqueeze(0).expand(n, n).clone())
        self.register_buffer("no_self", 1.0 - torch.eye(n))

        # Balanced initialization: inhibitory weights scaled up so that the
        # expected E and I input to each unit roughly cancel.
        w = torch.rand(n, n) / n**0.5
        if cfg.dale:
            n_inh = n - self.n_exc
            if n_inh > 0:
                w[:, self.n_exc :] *= self.n_exc / n_inh
            w *= 0.5
        else:
            w = torch.randn(n, n) * (1.5 / n**0.5)
        self.w_raw = nn.Parameter(w)
        self.w_in_raw = nn.Parameter(torch.rand(n, cfg.n_inputs) * 0.5)
        self.b = nn.Parameter(torch.zeros(n))
        rec_mask, in_mask, out_mask = torch.ones(n, n), torch.ones(n, cfg.n_inputs), torch.ones(self.n_exc)
        if cfg.area_of_unit is not None:
            area = torch.tensor(cfg.area_of_unit)
            same = area[:, None] == area[None, :]
            fwd = area[:, None] == area[None, :] + 1           # post is in the next area
            back = area[:, None] < area[None, :] if cfg.feedback else torch.zeros(n, n, dtype=torch.bool)
            rec_mask = (same | ((fwd | back) & is_exc[None, :])).float()
            if cfg.input_areas is not None:
                for k, allowed in enumerate(cfg.input_areas):
                    in_mask[:, k] = torch.isin(area, torch.tensor(allowed)).float()
            if cfg.readout_area is not None:
                out_mask = (area[: self.n_exc] == cfg.readout_area).float()
        self.register_buffer("rec_mask", rec_mask, persistent=False)  # rebuilt from cfg
        self.register_buffer("in_mask", in_mask, persistent=False)
        self.register_buffer("out_mask", out_mask, persistent=False)
        n_read = self.n_exc
        self.w_out = nn.Parameter(torch.randn(cfg.n_outputs, n_read) / n_read**0.5)
        self.b_out = nn.Parameter(torch.zeros(cfg.n_outputs))

    # Effective (constrained) weights -----------------------------------------
    @property
    def w_rec(self) -> torch.Tensor:
        if self.cfg.dale:
            return torch.relu(self.w_raw) * self.sign * self.no_self * self.rec_mask
        return self.w_raw * self.no_self * self.rec_mask

    @property
    def w_in(self) -> torch.Tensor:
        return (torch.relu(self.w_in_raw) if self.cfg.dale else self.w_in_raw) * self.in_mask

    def forward(
        self,
        u: torch.Tensor,
        silence: torch.Tensor | None = None,
        current: torch.Tensor | None = None,
        noise: bool = True,
    ):
        """Run the network.

        u:        (T, B, n_inputs) inputs
        silence:  (T, B, N) bool, True where a unit's rate is clamped to 0 and its state held at or below 0
        current:  (T, B, N) extra input current added to the units (optogenetic drive)
        Returns outputs (T, B, n_outputs) and rates (T, B, N).
        """
        T, B, _ = u.shape
        n = self.cfg.n_units
        w_rec, w_in = self.w_rec, self.w_in
        x = torch.zeros(B, n, device=u.device)
        rates, outs = [], []
        noise_scale = self.cfg.noise_std * (2 * self.alpha) ** 0.5
        for t in range(T):
            r = torch.relu(x)
            if silence is not None:
                r = r.masked_fill(silence[t], 0.0)
            inp = r @ w_rec.T + u[t] @ w_in.T + self.b
            if current is not None:
                inp = inp + current[t]
            x = (1 - self.alpha) * x + self.alpha * inp
            if noise and noise_scale > 0:
                x = x + noise_scale * torch.randn_like(x)
            if silence is not None:
                # Hold silenced units down (like optogenetic hyperpolarisation): their
                # state cannot climb above threshold, so they do not build up input
                # while silenced and come back without a memory of it.
                x = torch.where(silence[t], x.clamp(max=0.0), x)
            r = torch.relu(x)
            if silence is not None:
                r = r.masked_fill(silence[t], 0.0)
            rates.append(r)
            outs.append(r[:, : self.n_exc] @ (self.w_out * self.out_mask).T + self.b_out)
        return torch.stack(outs), torch.stack(rates)
