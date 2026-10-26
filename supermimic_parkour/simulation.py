"""SuperMimic's batched motion tracker using mjlab Scene/Simulation directly.

State and actions use the PHP sphere-hand MJCF joint order, wxyz quaternions,
and link-origin world velocities. No PPO manager or Isaac Lab dependency.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import torch
import re
import warnings
from types import SimpleNamespace
from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.envs.mdp.dr.body import body_com_offset

from .core.sampling import update_bin_returns
from supermimic_parkour.mdp import MAX_STEP_REWARD, MDP, flatten_state
from supermimic_parkour.motion import CONTROL_DT, MotionLib
from supermimic_parkour.robot import G1_JOINT_NAMES, action_scale
from .commands import PhpMotionCommand
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from .terrain import box_spec, place_boxes
from supermimic_parkour.rotations import mat_to_quat, quat_to_mat


def make_scene_cfg(num_envs: int, max_boxes: int, teacher_cfg):
    cfg = deepcopy(teacher_cfg.scene)
    cfg.num_envs = num_envs
    cfg.env_spacing = 0.0
    cfg.terrain.env_spacing = 0.0
    for key in tuple(cfg.entities):
        if key.startswith("php_box_"):
            del cfg.entities[key]
    for i in range(max_boxes):
        cfg.entities[f"php_box_{i}"] = EntityCfg(
            spec_fn=box_spec,
            init_state=EntityCfg.InitialStateCfg(pos=(0, 0, -50)),
        )
    return cfg


class TeacherMotionView(PhpMotionCommand):
    """Expose our shared library through the original teacher observation terms."""

    def __init__(self, env):
        self._env = env
        self.cfg = env.mdp.teacher_cfg.commands["motion"]
        self.robot = env.robot
        self.motion = env.motion.library
        self.robot_anchor_body_index = self.robot.body_names.index(
            self.cfg.anchor_body_name
        )
        self.motion_anchor_body_index = self.cfg.body_names.index(
            self.cfg.anchor_body_name
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            device=env.device,
        )

    @property
    def time_steps(self):
        return self._env.f


class TrackingSim:
    def __init__(
        self,
        num_envs: int,
        motion: MotionLib,
        mdp: MDP,
        device="cuda:0",
        gamma: float = 0.99,
        eval_mode: bool = False,
        max_episode_steps: int = 500,
        reset_noise: bool = True,
        no_termination: bool = False,
        ccd_iterations: int = 200,
        strict_numerics: bool = False,
        numerical_error_dir: str | Path | None = None,
    ):
        if ccd_iterations < 1:
            raise ValueError("ccd_iterations must be positive")
        if no_termination and not eval_mode:
            raise ValueError(
                "no_termination is only supported in evaluation/playback mode"
            )
        self.no_termination = no_termination
        self.device, self.N = device, num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.strict_numerics = strict_numerics
        self.numerical_error_dir = numerical_error_dir
        self.numerical_failures = 0
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.scene = Scene(
            make_scene_cfg(num_envs, motion.max_boxes, mdp.teacher_cfg), device=device
        )
        # EPA scratch space scales with this limit AND the total contact
        # capacity. Do not force 500: at 4096 worlds one vertex buffer alone
        # would need 14.8 GiB (4096 * 320 * (10 + 2 * 500) * 12 bytes).
        teacher_sim = mdp.teacher_cfg.sim
        cfg = replace(
            teacher_sim,
            mujoco=replace(
                teacher_sim.mujoco,
                ccd_iterations=ccd_iterations,
            ),
        )
        print(
            f"[sim] device={device}, envs={num_envs}, ccd_iterations={ccd_iterations}",
            flush=True,
        )
        self.sim = Simulation(
            num_envs=num_envs, cfg=cfg, model=self.scene.compile(), device=device
        )
        # Include the derived constants required by mjlab's COM randomization.
        fields = ("geom_friction", "geom_size", "geom_rbound", "geom_aabb") + tuple(
            body_com_offset.model_fields
        )
        self.sim.expand_model_fields(fields)
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)
        self.robot = self.scene["robot"]
        if list(self.robot.joint_names) != G1_JOINT_NAMES:
            raise ValueError("G1 model joint order differs from the named PHP adapter")
        self.default_q = self.robot.data.default_joint_pos[0].clone()
        self.alpha = torch.tensor(
            [action_scale(n) for n in G1_JOINT_NAMES], device=device
        )
        self.xi = torch.zeros(num_envs, motion.context_dim, device=device)
        self.terrain_entities = [
            self.scene[f"php_box_{i}"] for i in range(motion.max_boxes)
        ]
        # Match the existing PHP terrain placement interface.
        self._env = self
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.variant_ids = torch.zeros_like(self.motion_ids)
        self.xi[:, 0] = 0.6
        if not eval_mode:
            self.xi[:, 0].uniform_(0.4, 1.3)
            bias_extent = torch.tensor(
                [0.06 if "ankle" in name else 0.01 for name in G1_JOINT_NAMES],
                device=device,
            )
            self.xi[:, 1:30] = (
                2 * torch.rand(num_envs, 29, device=device) - 1
            ) * bias_extent
            extent = torch.tensor([0.025, 0.05, 0.05], device=device)
            self.xi[:, 30:33] = (
                2 * torch.rand(num_envs, 3, device=device) - 1
            ) * extent
        torso_id = self.sim.mj_model.body("robot/torso_link").id
        base_com = self.sim.get_default_field("body_ipos")[torso_id]
        self.sim.model.body_ipos[:, torso_id] = base_com + self.xi[:, 30:33]
        # Match the teacher: randomize foot friction only.
        feet = [
            i
            for i in self.robot.indexing.geom_ids.tolist()
            if re.fullmatch(
                r"robot/(left|right)_foot[1-7]_collision",
                self.sim.mj_model.geom(i).name,
            )
        ]
        self.sim.model.geom_friction[:, feet, 0] = self.xi[:, :1]
        self.robot.data.encoder_bias[:] = self.xi[:, 1:30]
        self.sim.recompute_constants(RecomputeLevel.set_const)
        self.f = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_step = torch.zeros_like(self.f)
        self.replay_step = torch.zeros_like(self.f)
        self.ep_planned_steps = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(num_envs, 29, device=device)  # previous clean action
        self.start_bin = torch.zeros_like(self.f)
        self.ep_disc_ret = torch.zeros(num_envs, device=device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.full((motion.num_bins,), self.G_max, device=device)
        self.ep_rewards = torch.zeros(num_envs, max_episode_steps, device=device)
        self.ep_bins = torch.zeros(
            num_envs, max_episode_steps, dtype=torch.long, device=device
        )
        self.bin_w_avg = torch.full(
            (motion.num_bins,), 1.0 / motion.num_bins, device=device
        )
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self.push_timer = torch.empty(num_envs, device=device).uniform_(0.5, 1.0)
        self.num_envs = num_envs
        self.command = TeacherMotionView(self)
        self.command_manager = SimpleNamespace(
            get_term=lambda name: self.command,
            get_command=lambda name: self.command.command,
        )
        self.action_manager = SimpleNamespace(
            action=self.a_prev, prev_action=self.a_prev
        )
        self.reset(torch.arange(num_envs, device=device))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def state(self):
        d = self.robot.data
        pose, vel = d.root_link_pose_w.clone(), d.root_link_vel_w.clone()
        return (
            pose[:, :3] - self.scene.env_origins,
            quat_to_mat(pose[:, 3:7]),
            vel[:, :3],
            vel[:, 3:],
            d.joint_pos.clone(),
            d.joint_vel.clone(),
        )

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        probs = m.sampling_probabilities(self.bin_G, self.bin_w_avg, self.G_max)
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = start + m.bin_size[bins]
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def reset(
        self,
        env_ids: torch.Tensor,
        frames: torch.Tensor | None = None,
        variants: torch.Tensor | None = None,
    ):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        m = self.motion
        self.motion_ids[env_ids] = m.clip_id[frames]
        counts = m.variant_counts[self.motion_ids[env_ids]]
        if variants is None:
            variants = (torch.rand(n, device=dev) * counts).long()
        if torch.any((variants < 0) | (variants >= counts)):
            raise ValueError("Terrain variant is out of range for the selected motion")
        self.variant_ids[env_ids] = variants
        place_boxes(self, env_ids)
        self.xi[env_ids, 33:] = m.terrain[self.motion_ids[env_ids], variants].flatten(1)
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = quat_to_mat(quat_from_euler_xyz(*rv.unbind(-1))) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.05, 0.05)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.1, 0.1)), -1)  # fmt: skip
        q = torch.clamp(q, self.mdp.soft_lo, self.mdp.soft_hi)
        quat = mat_to_quat(R)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat), -1)
        self.robot.write_root_link_pose_to_sim(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(
            root_velocity=torch.cat((v, w), -1), env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(velocity=qd, env_ids=env_ids)
        self.robot.set_joint_position_target(position=q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_planned_steps[env_ids] = (m.clip_end[m.clip_id[frames]] - frames).clamp(
            min=1, max=self.max_episode_steps
        )
        self.ep_step[env_ids] = 0
        self.replay_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        self.done_eval[env_ids] = False
        self.failed_eval[env_ids] = False
        self.scene.write_data_to_sim()
        self.refresh()

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward()
        # Refresh derived state without advancing physics.
        for entity in self.scene.entities.values():
            entity.update(0.0)
        for sensor in self.scene.sensors.values():
            sensor.update(0.0)
        self.sim.sense()
        self.command._update_relative_poses()

    def observation(self, group="actor"):
        self.command._update_relative_poses()
        self.action_manager.action = self.a_prev
        return torch.cat(
            [
                term.func(self, **term.params)
                for term in self.mdp.teacher_cfg.observations[group].terms.values()
            ],
            -1,
        )

    def teacher_reward(self):
        terms = {
            name: cfg.weight * cfg.func(self, **cfg.params)
            for name, cfg in self.mdp.teacher_cfg.rewards.items()
        }
        return sum(terms.values()) * CONTROL_DT, terms

    def _check_finite(self, **fields):
        """Stop before invalid physics/rewards reach replay or sampling stats."""
        invalid = torch.stack(
            [
                ~torch.isfinite(value).reshape(self.N, -1).all(-1)
                for value in fields.values()
            ]
        )
        if not bool(invalid.any()):
            return
        ids = invalid.any(0).nonzero(as_tuple=False).flatten()[:8]
        names = [name for name, bad in zip(fields, invalid.any(1).tolist()) if bad]
        frames = self.f[ids]
        clips = self.motion.clip_id[frames].tolist()
        raise FloatingPointError(
            f"Non-finite {', '.join(names)} in TrackingSim.step: "
            f"env_ids={ids.tolist()}, frames={frames.tolist()}, "
            f"clips={[str(self.motion.paths[i]) for i in clips]}. "
            "Stopped before updating replay or motion sampling statistics."
        )

    def _record_numerical_failure(self, invalid, frames, action, **fields):
        ids = invalid.nonzero(as_tuple=False).flatten()
        clips = self.motion.clip_id[frames[ids]].tolist()
        paths = [str(self.motion.paths[i]) for i in clips]
        self.numerical_failures += len(ids)
        names = [name for name, value in fields.items() if not bool(value[ids].isfinite().all())]
        dump_path = None
        if self.numerical_error_dir is not None:
            directory = Path(self.numerical_error_dir)
            directory.mkdir(parents=True, exist_ok=True)
            dump_path = directory / "latest.pt"
            torch.save(
                {
                    "env_ids": ids.cpu(), "frames": frames[ids].cpu(),
                    "next_frames": self.f[ids].cpu(), "clips": paths,
                    "action": action[ids].cpu(), "a_prev": self.a_prev[ids].cpu(),
                    "xi": self.xi[ids].cpu(), "ep_step": self.ep_step[ids].cpu(),
                    "variant_ids": self.variant_ids[ids].cpu(),
                    **{name: value[ids].cpu() for name, value in fields.items()},
                },
                dump_path,
            )
        warnings.warn(
            f"Resetting {len(ids)} numerical-failure environment(s), "
            f"total={self.numerical_failures}: non-finite {', '.join(names)}; "
            f"env_ids={ids[:8].tolist()}, frames={frames[ids[:8]].tolist()}, "
            f"clips={paths[:8]}. Invalid transitions excluded from training."
            + (f" Snapshot: {dump_path}" if dump_path else ""),
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor, *, clean_action: torch.Tensor | None = None) -> dict:
        """Execute ``a``; reward/history use the action before exploration noise."""
        clean_action = a if clean_action is None else clean_action
        self._check_finite(action=a, clean_action=clean_action)
        S = self.state()
        s = flatten_state(S)
        self._check_finite(state=s)
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        replay_step = self.replay_step.clone()
        target = self.default_q - xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target(position=target)
        for _ in range(4):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(CONTROL_DT / 4)
        # mj_step leaves derived poses at the preceding substep. Refresh before
        # labeling replay transitions, so q and link poses describe one instant.
        self.refresh()
        S1 = self.state()
        f1 = torch.minimum(f + 1, self.motion.clip_end[self.motion.clip_id[f]])
        # Evaluate the original teacher reward functions at the next reference.
        self.f = f1
        self.command._update_relative_poses()
        self.action_manager.action = clean_action
        self.action_manager.prev_action = a_prev
        r, terms = self.teacher_reward()
        s1 = flatten_state(S1)
        fields = dict(
            next_state=s1, reward=r,
            **{f"reward/{name}": value for name, value in terms.items()},
        )
        valid = torch.stack([
            value.isfinite().reshape(self.N, -1).all(-1) for value in fields.values()
        ]).all(0)
        invalid = ~valid
        if bool(invalid.any()):
            # Playback/evaluation must expose failures, and a batch-wide blowup
            # is not an isolated simulator fault we can safely recover from.
            if self.eval_mode or self.strict_numerics or bool(invalid.all()):
                self._check_finite(**fields)
            self._record_numerical_failure(invalid, f, a, state=s, **fields)
            # Finite placeholders keep masked batched computations safe. They
            # are never used as model targets or imagination starting states.
            S1 = tuple(
                torch.where(invalid.reshape(-1, *([1] * (after.ndim - 1))), before, after)
                for before, after in zip(S, S1)
            )
            s1 = flatten_state(S1)
            r = torch.where(valid, r, 0.0)
            terms = {name: torch.where(valid, value, 0.0) for name, value in terms.items()}
        term = self.mdp.termination(S1, f1)
        if self.no_termination:
            term = torch.zeros_like(term)
        term |= invalid
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": s, "a": a.clone(), "a_prev": a_prev, "f": f, "s1": s1, "r": r,
            "term": term, "done": done, "ep_step": replay_step, "xi": xi, "terms": terms,
            "cut": invalid.clone(), "valid": valid,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.replay_step = replay_step + 1
        self.a_prev = clean_action.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        rows = torch.arange(self.num_envs, device=self.device)
        self.ep_rewards[rows, ep_step] = r
        self.ep_bins[rows, ep_step] = self.motion.frame_bin[f]
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            completed = ids[valid[ids]]
            # Censor the whole numerically aborted episode; its unknown tail
            # must not be interpreted as low task return by adaptive sampling.
            if len(completed):
                update_bin_returns(
                    self.bin_G,
                    self.ep_rewards[completed],
                    self.ep_bins[completed],
                    self.ep_step[completed],
                    term[completed],
                    f1[completed] >= clip_end[completed],
                    self.gamma,
                )
            T = self.ep_step[ids].float()
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            out["ep_return"] = self.ep_ret[ids] / (
                self.ep_planned_steps[ids] * MAX_STEP_REWARD
            )
            out["ep_skill"] = self.motion.clip_skill[self.motion_ids[ids]].clone()
            out["ep_failed"] = term[ids].clone()
            self.reset(ids)
        # Match the teacher's interval pushes. They happen between transitions;
        # cut replay windows/history at the discontinuity, not the RL episode.
        self.push_timer -= CONTROL_DT
        push_ids = torch.where((self.push_timer <= 0) & ~done)[0]
        if len(push_ids):
            velocity = self.robot.data.root_link_vel_w[push_ids].clone()
            limits = torch.tensor([0.1, 0.1, 0.05, 0.1, 0.1, 0.1], device=self.device)
            velocity += (2 * torch.rand_like(velocity) - 1) * limits
            self.robot.write_root_link_velocity_to_sim(velocity, env_ids=push_ids)
            self.push_timer[push_ids] = torch.empty(
                len(push_ids), device=self.device
            ).uniform_(0.5, 1.0)
            self.replay_step[push_ids] = 0
            out["cut"][push_ids] = True
            self.refresh()
        return out
