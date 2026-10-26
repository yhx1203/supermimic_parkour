"""Revised SuperMimic networks with PHP teacher observations and terrain context.

Algorithm primitives come from core/ (SuperMimic-Reproduced). PHP keeps its
matrix/world-angular-velocity interface for FK; the learned dynamics use the
upstream quaternion/body-angular-velocity state and fixed scales internally.
"""

from __future__ import annotations

import torch
from torch import nn

from .core.dynamics import MLP, DynamicsMLP
from .core.state_spec import StateSpec, StateSpecCfg
from .mdp import NUM_J, XI_DIM, terrain_features
from .rotations import mat_to_quat, quat_to_mat

FORMAT = "php_supermimic_teacher_v4"
SUPPORTED_FORMATS = (FORMAT, "php_supermimic_teacher_v3", "php_supermimic_teacher_v2")
GOAL_FRAME_DIM = 68
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net, x):
    if AMP_DTYPE is None or x.device.type != "cuda":
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


def to_model_state(S):
    p, R, v, w, q, qd = S
    w_body = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
    return torch.cat((p, mat_to_quat(R), q, v, w_body, qd), -1)


def from_model_state(s):
    p, quat, q, v, w_body, qd = StateSpec.split(s)
    R = quat_to_mat(quat)
    return p, R, v, (R @ w_body.unsqueeze(-1)).squeeze(-1), q, qd


class WorldModel(nn.Module):
    def __init__(
        self,
        terrain_dim=0,
        hidden=(2048, 2048, 2048),
        termination_thresholds=(0.5, 0.8, 0.5),
    ):
        super().__init__()
        self.spec = StateSpec(StateSpecCfg(), "cpu")
        self.dynamics = DynamicsMLP(xi_dim=XI_DIM + terrain_dim, hidden=hidden)
        # PHP terminates on anchor position, anchor tilt and end-effector error.
        # Extend upstream's 2-feature head to these 3 teacher criteria.
        self.term_head = MLP(6, (64, 64), 1)
        self.register_buffer("term_scale", 1 / torch.tensor(termination_thresholds))

    @property
    def net(self):
        return self.dynamics.mlp.net

    def inputs(self, S, a, xi_n):
        return torch.cat(
            (
                self.spec.canonical(to_model_state(S)),
                a,
                xi_n[..., :XI_DIM],
                terrain_features(S, xi_n),
            ),
            -1,
        )

    def step(self, S, a, xi_n):
        s = to_model_state(S)
        delta = run_mlp(self.dynamics.mlp, self.inputs(S, a, xi_n))
        return from_model_state(self.spec.integrate(s, delta))

    def termination_logits(self, errors):
        x = torch.stack(errors, -1) * self.term_scale
        return run_mlp(self.term_head, torch.cat((x, x.abs()), -1)).squeeze(-1)

    def state_error(self, S_hat, S):
        return self.spec.error_vector(to_model_state(S_hat), to_model_state(S))


def observation_scale(obs_dim, *, critic=False):
    """Fixed scales for PHP terms; scan scales only for legacy playback."""
    base_dim = 286 if critic else 160
    if obs_dim not in (base_dim, base_dim + 289):
        raise ValueError(f"Unexpected PHP observation width: {obs_dim}")
    parts = [torch.ones(29), torch.full((29,), 0.2), torch.ones(9)]
    if critic:
        parts.append(torch.ones(126))  # tracked body positions / 6D rotations
    parts += [
        torch.ones(3),
        torch.full((3,), 0.5),
        torch.ones(29),
        torch.full((29,), 0.2),
        torch.ones(29),
    ]
    if obs_dim == base_dim + 289:
        parts.append(torch.ones(289))
    return torch.cat(parts)


def goal_scale(goal_dim):
    if goal_dim % GOAL_FRAME_DIM:
        raise ValueError("Goals must contain complete 68-dimensional frames")
    return torch.cat(
        (
            torch.ones(29),
            torch.full((29,), 0.2),
            torch.ones(3),
            torch.full((3,), 0.5),
            torch.ones(4),
        )
    ).repeat(goal_dim // 68)


class Policy(nn.Module):
    """SiLU motion encoder + shared decoder, fixed scales, tanh 32D latent."""

    def __init__(
        self, obs_dim, goal_dim, hist=10, z_dim=32, hidden=(2048, 1024, 512, 256)
    ):
        super().__init__()
        self.hist = hist
        self.enc = MLP(goal_dim, tuple(hidden), z_dim)
        self.dec = MLP(z_dim + obs_dim * hist, tuple(hidden), NUM_J, last_gain=0.01)
        self.register_buffer("obs_scale", observation_scale(obs_dim))
        self.register_buffer("goal_scale", goal_scale(goal_dim))

    def forward(self, obs_hist, goal):
        z = torch.tanh(run_mlp(self.enc, goal * self.goal_scale))
        # Keep the PHP history layout (oldest frame first) in collect and imagine.
        return run_mlp(
            self.dec, torch.cat((z, (obs_hist * self.obs_scale).flatten(-2)), -1)
        )

    def act(self, obs_hist, goal, noise_std=0.0):
        action = self(obs_hist, goal)
        if noise_std:
            action = action + noise_std * torch.randn_like(action)
        return action.clamp(-20.0, 20.0)


class Critic(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128)):
        super().__init__()
        self.net = MLP(in_dim, tuple(hidden), 1)
        self.register_buffer(
            "scale",
            torch.cat((observation_scale(286, critic=True), goal_scale(in_dim - 286))),
        )

    def forward(self, x):
        return run_mlp(self.net, x * self.scale).squeeze(-1)


def load_policy(config, state_dict, device):
    """Playback supports v2/v3 policies; new training writes v4 without scans."""
    version = config.get("format")
    if version in (FORMAT, "php_supermimic_teacher_v3"):
        expected_dim = 160 if version == FORMAT else 449
        if config["obs_dim"] != expected_dim:
            raise ValueError(f"Checkpoint {version} requires obs_dim={expected_dim}")
        policy = Policy(
            config["obs_dim"],
            68 * config["goal_frames"],
            hist=config["hist"],
            hidden=tuple(config["policy_hidden"]),
        )
    elif version == "php_supermimic_teacher_v2":
        from .legacy_networks import Policy as LegacyPolicy

        policy = LegacyPolicy(
            config["obs_dim"], 73 * config["goal_frames"], hist=config["hist"]
        )
    else:
        raise ValueError(f"Unsupported PHP SuperMimic checkpoint format: {version}")
    policy.to(device).load_state_dict(state_dict)
    return policy.eval()
