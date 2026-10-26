"""PHP observations/terminations with the original SuperMimic tracking reward.

Only the world model receives raw geometry/context. Actor and critic reproduce
PHP's single-frame observation terms in their original order.
"""

from __future__ import annotations

import torch
from mjlab.sensor import GridPatternCfg
from .motion import CONTROL_DT
from .rotations import mat_to_6d, quat_to_mat, mat_to_quat, rot_angle_sq, yaw_mat
from .dataset import BODY_NAMES
from .teacher import php_teacher_env_cfg

STATE_DIM = 71
XI_DIM = 33
NUM_J = 29
REWARD_DEFINITION = "supermimic_tracking_v1"
# Match unitree/supermimic_mjlab/envs/mdp/rewards.py, independently of PHP teachers.
TRACKING_REWARDS = {
    "motion_global_root_pos": (0.5, 0.3),
    "motion_global_root_ori": (0.5, 0.4),
    "motion_body_pos": (1.0, 0.3),
    "motion_body_ori": (1.0, 0.4),
    "motion_body_lin_vel": (1.0, 1.0),
    "motion_body_ang_vel": (1.0, 3.14),
}
MAX_STEP_REWARD = sum(weight for weight, _ in TRACKING_REWARDS.values()) * CONTROL_DT
ACTOR_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)
CRITIC_TERMS = (
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "body_pos",
    "body_ori",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
)


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return (
        x[..., 0:3],
        quat_to_mat(x[..., 3:7]),
        x[..., 7:10],
        x[..., 10:13],
        x[..., 13:42],
        x[..., 42:71],
    )


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (Rt @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + 29 + 29


class MDP:
    def __init__(self, kin, motion, default_q, *, legacy_height_scan=False):
        self.kin, self.motion, self.default_q = kin, motion, default_q
        # Legacy scans are used only to play checkpoints trained before v4.
        self.legacy_height_scan = legacy_height_scan
        self.actor_terms = ACTOR_TERMS + (
            ("height_scan",) if legacy_height_scan else ()
        )
        self.critic_terms = CRITIC_TERMS + (
            ("height_scan",) if legacy_height_scan else ()
        )
        self.teacher_cfg = cfg = php_teacher_env_cfg(motion.skill_names[0])
        # This is a fresh config: change only SuperMimic, not the PHP PPO teachers.
        cfg.rewards = {
            name: cfg.rewards[name]
            for name in (*TRACKING_REWARDS, "action_rate_l2", "joint_limit")
        }
        for name, (weight, std) in TRACKING_REWARDS.items():
            cfg.rewards[name].weight = weight
            cfg.rewards[name].params["std"] = std
        cfg.rewards["action_rate_l2"].weight = -0.1
        cfg.rewards["joint_limit"].weight = -2.0
        cfg.scene.sensors = tuple(
            sensor for sensor in cfg.scene.sensors if sensor.name != "undesired_contact"
        )
        if not legacy_height_scan:
            for group in ("actor", "critic"):
                cfg.observations[group].terms.pop("height_scan", None)
            cfg.scene.sensors = tuple(
                sensor for sensor in cfg.scene.sensors if sensor.name != "terrain_scan"
            )
        if tuple(cfg.observations["actor"].terms) != self.actor_terms:
            raise ValueError(
                "PHP teacher actor terms changed; update the differentiable adapter"
            )
        if tuple(cfg.observations["critic"].terms) != self.critic_terms:
            raise ValueError(
                "PHP teacher critic terms changed; update the differentiable adapter"
            )
        self.anchor = BODY_NAMES.index(cfg.commands["motion"].anchor_body_name)
        self.ee = [
            BODY_NAMES.index(n)
            for n in cfg.terminations["ee_body_pos"].params["body_names"]
        ]
        widths = (58, 3, 6, 3, 3, 29, 29, 29)
        if legacy_height_scan:
            scan = next(s for s in cfg.scene.sensors if s.name == "terrain_scan")
            if (
                not isinstance(scan.pattern, GridPatternCfg)
                or scan.ray_alignment != "yaw"
            ):
                raise ValueError("Expected the PHP yaw-aligned grid height scan")
            self.grid, _ = scan.pattern.generate_rays(None, str(default_q.device))
            self.scan_max = scan.max_distance
            self.scan_offset = (
                cfg.observations["actor"].terms["height_scan"].params["offset"]
            )
            widths += (len(self.grid),)
        self.obs_dim = sum(widths)
        self.critic_dim = self.obs_dim + len(BODY_NAMES) * 9
        self.obs_slices = {}
        noise = []
        start = 0
        for name, width in zip(self.actor_terms, widths):
            self.obs_slices[name] = slice(start, start + width)
            start += width
            term_noise = cfg.observations["actor"].terms[name].noise
            extent = 0.0 if term_noise is None else term_noise.n_max
            if term_noise is not None and term_noise.n_min != -extent:
                raise ValueError("Expected symmetric uniform teacher observation noise")
            noise.append(torch.full((width,), extent, device=default_q.device))
        self.obs_noise = torch.cat(noise)
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        factor = cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor
        self.soft_lo, self.soft_hi = mid - factor * half, mid + factor * half

    def _features(self, S, a_prev, xi, f):
        p, R, v, w, q, qd = S
        bp, bR, _, _ = self.kin.forward(*S)
        ap, aR = bp[..., self.anchor, :], bR[..., self.anchor, :, :]
        At = aR.transpose(-1, -2)
        # MuJoCo velocimeter is at imu_in_pelvis, not at the link origin.
        lever = (R @ self.kin.imu_pos.unsqueeze(-1)).squeeze(-1)
        sensor_Rt = (R @ self.kin.imu_R).transpose(-1, -2)
        lin_vel = (
            sensor_Rt @ (v + torch.cross(w, lever, dim=-1)).unsqueeze(-1)
        ).squeeze(-1)
        ang_vel = (sensor_Rt @ w.unsqueeze(-1)).squeeze(-1)
        features = {
            "command": torch.cat((self.motion.q[f], self.motion.qd[f]), -1),
            "motion_anchor_pos_b": (
                At @ (self.motion.body_p[f, self.anchor] - ap).unsqueeze(-1)
            ).squeeze(-1),
            "motion_anchor_ori_b": mat_to_6d(At @ self.motion.body_R[f, self.anchor]),
            "body_pos": (At.unsqueeze(-3) @ (bp - ap.unsqueeze(-2)).unsqueeze(-1))
            .squeeze(-1)
            .flatten(-2),
            "body_ori": mat_to_6d(At.unsqueeze(-3) @ bR).flatten(-2),
            "base_lin_vel": lin_vel,
            "base_ang_vel": ang_vel,
            "joint_pos": q - self.default_q,
            "joint_vel": qd,
            "actions": a_prev,
        }
        if self.legacy_height_scan:
            features["height_scan"] = self.height_scan(ap, aR, xi)
        return features

    def obs(self, S, a_prev, xi, f):
        features = self._features(S, a_prev, xi, f)
        # Teacher actor observes encoder-biased positions; critic uses true positions.
        features["joint_pos"] = features["joint_pos"] + xi[..., 1:30]
        return torch.cat([features[name] for name in self.actor_terms], -1)

    def critic_features(self, S, a_prev, xi, f):
        features = self._features(S, a_prev, xi, f)
        return torch.cat([features[name] for name in self.critic_terms], -1)

    def add_noise(self, obs):
        return obs + (2 * torch.rand_like(obs) - 1) * self.obs_noise

    def height_scan(self, anchor_pos, anchor_R, context):
        """Exact downward ray intersections with PHP's plane and oriented boxes.

        Same 17x17 xy ordering, torso frame, yaw alignment, range and miss
        convention as the teacher. Piecewise derivatives reach the imagined state.
        """
        origins = anchor_pos.unsqueeze(-2) + torch.einsum(
            "...ij,rj->...ri", yaw_mat(anchor_R), self.grid
        )
        boxes = context[..., XI_DIM:].reshape(*context.shape[:-1], -1, 10)
        # The plane is at z=0; negative distances are misses.
        distances = torch.where(
            origins[..., 2] >= 0, origins[..., 2], self.scan_max + 1
        )
        for i in range(boxes.shape[-2]):
            box = boxes[..., i, :]
            Rt = quat_to_mat(box[..., 3:7]).transpose(-1, -2)
            local_o = torch.einsum(
                "...ij,...rj->...ri", Rt, origins - box[..., None, :3]
            )
            local_d = -Rt[..., :, 2].unsqueeze(-2)
            parallel = local_d.abs() < 1e-8
            safe_d = torch.where(parallel, torch.ones_like(local_d), local_d)
            half = box[..., None, 7:] / 2
            t0, t1 = (-half - local_o) / safe_d, (half - local_o) / safe_d
            # Finite sentinels avoid NaN gradients in masked parallel slabs.
            near = torch.where(parallel, -1e6, torch.minimum(t0, t1)).amax(-1)
            far = torch.where(parallel, 1e6, torch.maximum(t0, t1)).amin(-1)
            outside = (parallel & (local_o.abs() > half)).any(-1)
            hit = (far >= near) & (far >= 0) & ~outside & (box[..., 2:3] > -40)
            distance = torch.where(near >= 0, near, far)
            distances = torch.minimum(
                distances, torch.where(hit, distance, self.scan_max + 1)
            )
        return torch.where(
            distances <= self.scan_max, distances - self.scan_offset, self.scan_max
        )

    def _tracking(self, S, f):
        bp, bR, bv, bw = self.kin.forward(*S)
        ap, aR = bp[..., self.anchor, :], bR[..., self.anchor, :, :]
        rp, rR = self.motion.body_p[f, self.anchor], self.motion.body_R[f, self.anchor]
        dR = yaw_mat(aR @ rR.transpose(-1, -2))
        anchor = torch.cat((ap[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + (
            dR.unsqueeze(-3) @ (self.motion.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)
        ).squeeze(-1)
        ref_bR = dR.unsqueeze(-3) @ self.motion.body_R[f]
        return bp, bR, bv, bw, ap, aR, rp, rR, ref_bp, ref_bR

    def reward(self, S, f, a, a_prev, terms=False):
        """Original SuperMimic reward; actions exclude execution exploration noise."""
        bp, bR, bv, bw, ap, aR, rp, rR, ref_bp, ref_bR = self._tracking(S, f)
        errors = {
            "motion_global_root_pos": (rp - ap).square().sum(-1),
            "motion_global_root_ori": rot_angle_sq(rR, aR),
            "motion_body_pos": (ref_bp - bp).square().sum(-1).mean(-1),
            "motion_body_ori": rot_angle_sq(ref_bR, bR).mean(-1),
            "motion_body_lin_vel": (self.motion.body_v[f] - bv)
            .square()
            .sum(-1)
            .mean(-1),
            "motion_body_ang_vel": (self.motion.body_w[f] - bw)
            .square()
            .sum(-1)
            .mean(-1),
        }
        components = {}
        for name, error in errors.items():
            cfg = self.teacher_cfg.rewards[name]
            components[name] = cfg.weight * torch.exp(-error / cfg.params["std"] ** 2)
        q = S[4]
        components["action_rate_l2"] = self.teacher_cfg.rewards[
            "action_rate_l2"
        ].weight * (a - a_prev).square().sum(-1)
        components["joint_limit"] = self.teacher_cfg.rewards["joint_limit"].weight * (
            torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)
        ).sum(-1)
        reward = sum(components.values()) * CONTROL_DT
        return (reward, components) if terms else reward

    def termination_errors(self, S, f):
        bp, _, _, _, ap, aR, rp, rR, ref_bp, _ = self._tracking(S, f)
        return (
            torch.linalg.vector_norm(rp - ap, dim=-1),
            (rR[..., 2, 2] - aR[..., 2, 2]).abs(),
            torch.linalg.vector_norm(
                ref_bp[..., self.ee, :] - bp[..., self.ee, :], dim=-1
            ).amax(-1),
        )

    def termination(self, S, f):
        errors = self.termination_errors(S, f)
        thresholds = [
            self.teacher_cfg.terminations[n].params["threshold"]
            for n in ("anchor_pos", "anchor_ori", "ee_body_pos")
        ]
        failed = torch.stack(
            [error > threshold for error, threshold in zip(errors, thresholds)]
        ).any(0)
        return failed | ~torch.isfinite(flatten_state(S)).all(-1)

    def soft_termination(self, S, f):
        errors = self.termination_errors(S, f)
        thresholds = [
            self.teacher_cfg.terminations[n].params["threshold"]
            for n in ("anchor_pos", "anchor_ori", "ee_body_pos")
        ]
        survival = torch.ones_like(errors[0])
        for error, threshold, temperature in zip(
            errors, thresholds, (0.02, 0.04, 0.02)
        ):
            survival = survival * torch.sigmoid((threshold - error) / temperature)
        return 1 - survival


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            (xi[..., :1] - 0.75) / 0.45,
            xi[..., 1:30] / 0.01,
            xi[..., 30:33] / 0.05,
            xi[..., 33:],
        ),
        -1,
    )


def terrain_features(S, context):
    """Yaw-local oriented boxes, recomputed at every imagined state.

    Context stores world poses and full dimensions, never a frozen height scan.
    Padded slots are zero with a validity bit, keeping the feature dimension fixed.
    """
    p, R = S[:2]
    boxes = context[..., XI_DIM:].reshape(*context.shape[:-1], -1, 10)
    valid = (boxes[..., 2:3] > -40).to(context.dtype)
    Ht = yaw_mat(R).transpose(-1, -2).unsqueeze(-3)
    delta = (Ht @ (boxes[..., :3] - p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
    orientation = mat_to_6d(Ht @ quat_to_mat(boxes[..., 3:7]))
    return torch.cat(
        (delta * valid, orientation * valid, boxes[..., 7:] * valid, valid), -1
    ).flatten(-2)
