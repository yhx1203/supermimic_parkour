"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from supermimic_parkour.mdp import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    "term": ((), torch.bool),
    "cut": ((), torch.bool),
    "valid": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str, context_dim: int = XI_DIM):
        self.T, self.N, self.device = T, N, device
        self.data = {}
        self.fields = dict(FIELDS, xi=(context_dim, torch.float32))
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr] = tr.get("valid", True) if k == "valid" else tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def _sample_valid_indices(self, B: int, stop: int):
        if stop <= 0:
            raise ValueError("Not enough replay steps to sample")
        age = torch.randint(0, stop, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        for _ in range(8):
            bad = ~self.data["valid"][self._row(age), env]
            if not bool(bad.any()):
                return age, env
            count = int(bad.sum())
            age[bad] = torch.randint(0, stop, (count,), device=self.device)
            env[bad] = torch.randint(0, self.N, (count,), device=self.device)
        # Rare fallback when much of replay is invalid. Never spin forever or
        # feed an all-zero probability vector to CUDA multinomial.
        candidates = self.data["valid"][
            self._row(torch.arange(stop, device=self.device))
        ].nonzero(as_tuple=False)
        if not len(candidates):
            raise RuntimeError("No valid replay transitions available")
        selected = candidates[torch.randint(len(candidates), (B,), device=self.device)]
        return selected[:, 0], selected[:, 1]

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0, env = self._sample_valid_indices(B, self.size - k + 1)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {
            key: self.data[key][rows, e]
            for key in ("s", "s1", "a", "xi", "done", "term", "f", "cut", "valid")
        }
        # transition j is valid if no episode ended at transitions 0..j-1
        d = (out["done"] | out["cut"] | ~out["valid"]).float()
        mask = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1
        )
        # Numerical failures are not terminal-state supervision. Exclude the
        # failed transition itself as well as everything following it.
        out["mask"] = mask * out["valid"]
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        age, env = self._sample_valid_indices(B, self.size)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(
            back.unsqueeze(0), ep_step.unsqueeze(-1)
        )  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
