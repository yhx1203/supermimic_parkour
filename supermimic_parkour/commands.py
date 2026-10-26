from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .dataset import BODY_NAMES, MotionLibrary, motion_paths, read_terrain
from .terrain import box_spec, place_boxes


class PhpMotionCommand(MotionCommand):
    """Reuse MJLab's tracking observations/rewards/ghost, with a named PHP library."""

    def __init__(self, cfg, env):
        # The upstream single-clip loader assumes model body/joint ordering.
        # Initialize the manager directly, then supply our named multi-clip buffer.
        CommandTerm.__init__(self, cfg, env)
        self.robot = env.scene[cfg.entity_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
            device=self.device,
        )
        self.motion = MotionLibrary(
            cfg.get_motion_paths(),
            self.robot.joint_names,
            cfg.body_names,
            self.device,
            require_velocity=cfg.require_velocity,
        )
        if not np.isclose(self.motion.fps * env.step_dt, 1.0):
            raise ValueError(
                f"PHP {self.motion.fps:g} FPS requires step_dt={1 / self.motion.fps:g}, got {env.step_dt}"
            )
        if cfg.fixed_motion_id is not None and not 0 <= cfg.fixed_motion_id < len(
            self.motion.paths
        ):
            raise ValueError(
                f"fixed_motion_id must be in [0, {len(self.motion.paths) - 1}]"
            )
        if cfg.fixed_variant_id is not None:
            counts = self.motion.variant_counts
            if cfg.fixed_motion_id is not None:
                counts = counts[cfg.fixed_motion_id : cfg.fixed_motion_id + 1]
            if cfg.fixed_variant_id < 0 or torch.any(counts <= cfg.fixed_variant_id):
                raise ValueError(
                    "fixed_variant_id is outside the selected clips' variant range"
                )
        if (
            cfg.adaptive_uniform_ratio <= 0
            or not 0 < cfg.adaptive_alpha <= 1
            or cfg.adaptive_kernel_size < 1
        ):
            raise ValueError("Invalid adaptive sampling parameters")
        self.terrain_entities = [
            env.scene[f"php_box_{i}"] for i in range(self.motion.max_boxes)
        ]
        self.motion_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.variant_ids = torch.zeros_like(self.motion_ids)
        self.time_steps = torch.zeros_like(self.motion_ids)
        self._next_motion_ids = torch.arange(self.num_envs, device=self.device) % len(
            self.motion.paths
        )
        self._episode_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.bin_count = len(self.motion.bin_starts)
        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)
        self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 3, device=self.device
        )
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(cfg.body_names), 4, device=self.device
        )
        self.body_quat_relative_w[..., 0] = 1
        for name in (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_anchor_lin_vel",
            "error_anchor_ang_vel",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "sampling_entropy",
            "sampling_top1_prob",
            "sampling_top1_bin",
            "motion_id",
            "terrain_variant",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._ghost_model = None
        self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    def _sample(self, env_ids):
        if self.cfg.sampling_mode == "adaptive":
            failed = (
                self._episode_active[env_ids]
                & self._env.termination_manager.terminated[env_ids]
            )
            failed_steps = self.time_steps[env_ids[failed]]
            bins = (
                torch.searchsorted(self.motion.bin_starts, failed_steps, right=True) - 1
            )
            self._current_bin_failed.scatter_add_(
                0, bins, torch.ones_like(bins, dtype=torch.float)
            )
        self._episode_active[env_ids] = True
        if self.cfg.sampling_mode in ("start", "sequential"):
            ids = (
                self._next_motion_ids[env_ids]
                if self.cfg.sampling_mode == "sequential"
                else torch.randint(
                    len(self.motion.paths), (len(env_ids),), device=self.device
                )
            )
            if self.cfg.fixed_motion_id is not None:
                ids = torch.full_like(ids, self.cfg.fixed_motion_id)
            self.motion_ids[env_ids] = ids
            self._next_motion_ids[env_ids] = (ids + 1) % len(self.motion.paths)
            self.time_steps[env_ids] = self.motion.motion_start_idx[ids]
        else:
            if self.cfg.sampling_mode == "uniform":
                probabilities = (
                    self.motion.bin_sizes.float() / self.motion.bin_sizes.sum()
                )
            else:
                probabilities = self.motion.sampling_probabilities(
                    self.bin_failed_count,
                    self.cfg.adaptive_uniform_ratio,
                    self.cfg.adaptive_kernel_size,
                    self.cfg.adaptive_lambda,
                )
            if self.cfg.fixed_motion_id is not None:
                probabilities = probabilities * (
                    self.motion.bin_clips == self.cfg.fixed_motion_id
                )
                probabilities /= probabilities.sum()
            bins = torch.multinomial(probabilities, len(env_ids), replacement=True)
            self.motion_ids[env_ids] = self.motion.bin_clips[bins]
            offsets = (
                torch.rand(len(env_ids), device=self.device)
                * self.motion.bin_sizes[bins]
            ).long()
            self.time_steps[env_ids] = self.motion.bin_starts[bins] + offsets
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            self.metrics["sampling_entropy"][:] = entropy / max(
                np.log(self.bin_count), 1.0
            )
            self.metrics["sampling_top1_prob"][:] = probabilities.max()
            self.metrics["sampling_top1_bin"][:] = (
                probabilities.argmax() / self.bin_count
            )
        counts = self.motion.variant_counts[self.motion_ids[env_ids]]
        self.variant_ids[env_ids] = (
            torch.rand(len(env_ids), device=self.device) * counts
        ).long()
        if self.cfg.fixed_variant_id is not None:
            self.variant_ids[env_ids] = self.cfg.fixed_variant_id

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        self._sample(env_ids)
        place_boxes(self, env_ids)
        root = torch.cat(
            (
                self.body_pos_w[env_ids, 0],
                self.body_quat_w[env_ids, 0],
                self.body_lin_vel_w[env_ids, 0],
                self.body_ang_vel_w[env_ids, 0],
            ),
            dim=-1,
        )

        def noise(ranges):
            limits = torch.tensor(
                [
                    ranges.get(k, (0.0, 0.0))
                    for k in ("x", "y", "z", "roll", "pitch", "yaw")
                ],
                device=self.device,
            )
            return sample_uniform(
                limits[:, 0], limits[:, 1], (len(env_ids), 6), self.device
            )

        delta = noise(self.cfg.pose_range)
        root[:, :3] += delta[:, :3]
        root[:, 3:7] = quat_mul(
            quat_from_euler_xyz(*delta[:, 3:].unbind(-1)), root[:, 3:7]
        )
        root[:, 7:] += noise(self.cfg.velocity_range)
        joints = self.joint_pos[env_ids].clone()
        joints += sample_uniform(
            *self.cfg.joint_position_range, joints.shape, self.device
        )
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joints = joints.clamp(limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(
            joints, self.joint_vel[env_ids], env_ids=env_ids
        )
        self.robot.write_root_state_to_sim(root, env_ids=env_ids)
        self.robot.clear_state(env_ids=env_ids)
        # Derived body transforms are stale until the environment's next
        # forward pass. Seed these rows from the reference, not stale anchors;
        # otherwise the first termination check rejects every reset.
        self.body_pos_relative_w[env_ids] = self.body_pos_w[env_ids]
        self.body_quat_relative_w[env_ids] = self.body_quat_w[env_ids]

    def _update_relative_poses(self):
        delta_pos = (
            self.robot_anchor_pos_w[:, None, :]
            .expand(-1, len(self.cfg.body_names), -1)
            .clone()
        )
        delta_pos[..., 2] = self.anchor_pos_w[:, None, 2]
        delta_ori = yaw_quat(
            quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w))
        )[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
        self.body_pos_relative_w = delta_pos + quat_apply(
            delta_ori, self.body_pos_w - self.anchor_pos_w[:, None, :]
        )

    def _update_command(self):
        ends = self.motion.motion_end_idx[self.motion_ids]
        # Normally motion_end produces a true episode reset. With terminations
        # disabled for ghost inspection, continue by explicitly resampling here.
        ended = torch.where(self.time_steps + 1 >= ends)[0]
        self.time_steps = torch.minimum(self.time_steps + 1, ends - 1)
        if (
            len(ended)
            and "motion_end" not in self._env.termination_manager.active_terms
        ):
            self._resample_command(ended)
        self._update_relative_poses()
        self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
        self._current_bin_failed.zero_()
        self.metrics["motion_id"][:] = self.motion_ids
        self.metrics["terrain_variant"][:] = self.variant_ids

    def _debug_vis_impl(self, visualizer):
        if self._ghost_model is None:
            import copy

            self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
            self._ghost_model.geom_rgba[:, 3] = 0
            self._ghost_model.geom_rgba[self.robot.indexing.geom_ids.cpu().numpy()] = (
                self._ghost_color
            )
        super()._debug_vis_impl(visualizer)


@dataclass(kw_only=True)
class PhpMotionCommandCfg(CommandTermCfg):
    dataset_dir: str
    require_velocity: bool = False
    entity_name: str = "robot"
    anchor_body_name: str = "torso_link"
    body_names: tuple[str, ...] = BODY_NAMES
    sampling_mode: Literal["adaptive", "uniform", "start", "sequential"] = "adaptive"
    fixed_motion_id: int | None = None
    fixed_variant_id: int | None = None
    adaptive_kernel_size: int = 3
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 1e-6
    adaptive_alpha: float = 0.001
    pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    joint_position_range: tuple[float, float] = (0.0, 0.0)
    viz: MotionCommandCfg.VizCfg = field(default_factory=MotionCommandCfg.VizCfg)

    def bind_scene_entities(self, scene_cfg):
        """Apply dataset CLI overrides before compiling the simulation scene."""
        paths = self.get_motion_paths()
        count = max(1, max(read_terrain(path).shape[0] for path in paths))
        for name in tuple(scene_cfg.entities):
            if name.startswith("php_box_"):
                del scene_cfg.entities[name]
        for i in range(count):
            scene_cfg.entities[f"php_box_{i}"] = EntityCfg(
                spec_fn=box_spec,
                # Entity's initial-state editor overrides the spec body pose.
                # Keep placeholders away from the robot before the first reset.
                init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -50.0)),
            )

    def build(self, env):
        return PhpMotionCommand(self, env)

    def get_motion_paths(self):
        return motion_paths(self.dataset_dir)
