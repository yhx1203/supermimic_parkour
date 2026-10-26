"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import time
from pathlib import Path

from .paths import default_dataset_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_envs",
        type=int,
        default=4096,
        help="parallel environments per GPU/worker",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=320_000,
        help="total vectorized simulator steps, including initial warmup",
    )
    parser.add_argument("--buffer_steps", type=int, default=1000)
    parser.add_argument(
        "--utd", type=float, default=0.4, help="updates per vectorized simulator step"
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--goal_frames", type=int, default=10)
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--wm_window", type=int, default=8)
    parser.add_argument("--wm_batches", type=int, default=10)
    parser.add_argument(
        "--wm_batch", type=int, default=2048, help="world-model windows per GPU/worker"
    )
    parser.add_argument(
        "--pi_batch",
        type=int,
        default=4096,
        help="imagined trajectories per GPU/worker",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--critic_epochs", type=int, default=5)
    parser.add_argument("--critic_minibatches", type=int, default=4)
    parser.add_argument("--no_critic", action="store_true")
    parser.add_argument(
        "--warmup_steps", type=int, default=40, help="sim steps before any update"
    )
    parser.add_argument(
        "--wm_only_iters",
        type=int,
        default=0,
        help="optional world-model-only update groups (upstream default: 0)",
    )
    parser.add_argument("--lr_decay_iters", type=int, default=250_000)
    parser.add_argument("--ckpt_every", type=int, default=4000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--no_tensorboard",
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard events (CSV logging remains enabled)",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(default_dataset_dir()),
    )
    parser.add_argument(
        "--max-clips-per-skill",
        type=int,
        default=0,
        help="0 loads all clips; positive values are for smoke tests",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/supermimic/php_five_skills",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--amp",
        type=str,
        default="bf16",
        choices=["none", "bf16"],
        help="autocast dtype for MLP matmuls",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ccd-iterations",
        type=int,
        default=200,
        help="convex collision iteration limit (default: %(default)s); larger values increase physics scratch memory",
    )
    parser.add_argument(
        "--no_compile", action="store_true", help="Disable torch.compile for debugging"
    )
    parser.add_argument(
        "--strict-numerics", action="store_true",
        help="Stop on isolated non-finite simulator transitions instead of resetting them",
    )
    parser.add_argument("--wm-hidden", type=int, nargs="+", default=[2048, 2048, 2048])
    parser.add_argument(
        "--policy-hidden", type=int, nargs="+", default=[2048, 1024, 512, 256]
    )
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument(
        "--skills",
        nargs="+",
        choices=("Low-Step", "High-Step", "Low-Climb", "High-Climb", "Ultra-Climb"),
        default=["Low-Step", "High-Step", "Low-Climb", "High-Climb", "Ultra-Climb"],
        help="PHP bundles to train; configured skill weights are normalized over this selection",
    )
    args = parser.parse_args()
    if len(args.skills) != len(set(args.skills)):
        parser.error("--skills must not contain duplicates")

    import torch
    import torch.nn as nn

    from supermimic_parkour.buffer import ReplayBuffer
    from supermimic_parkour.distributed import DistributedContext
    from supermimic_parkour.simulation import TrackingSim
    from supermimic_parkour.kinematics import G1Kinematics
    from supermimic_parkour.logging import (
        LOG_FIELDS,
        create_writer,
        open_csv_log,
        write_scalars,
    )
    from supermimic_parkour.mdp import (
        MAX_STEP_REWARD,
        REWARD_DEFINITION,
        MDP,
        normalize_xi,
        unflatten_state,
    )
    from supermimic_parkour.motion import MotionLib
    import supermimic_parkour.networks as networks
    from supermimic_parkour.networks import (
        FORMAT,
        GOAL_FRAME_DIM,
        Critic,
        Policy,
        WorldModel,
    )
    from supermimic_parkour.core.objective import (
        ImagRollout,
        gae_targets,
        policy_objective,
    )
    from supermimic_parkour.core.schedule import cosine_lr
    from supermimic_parkour.robot import default_joint_pos_list

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
    dev = args.device
    if dev.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; fix the NVIDIA driver or use --device cpu --amp none --no_compile for a small smoke test"
        )
    if not dev.startswith("cuda"):
        networks.AMP_DTYPE = None
        args.amp = "none"
    if (
        min(
            args.num_envs,
            args.iterations,
            args.wm_window,
            args.wm_batches,
            args.wm_batch,
            args.pi_batch,
            args.horizon,
            args.hist,
            args.goal_frames,
            args.critic_epochs,
            args.critic_minibatches,
            args.ckpt_every,
            args.log_every,
            args.lr_decay_iters,
            args.ccd_iterations,
            *args.wm_hidden,
            *args.policy_hidden,
            *args.critic_hidden,
        )
        < 1
    ):
        parser.error(
            "Batch sizes, horizons, iteration counts and logging intervals must be positive"
        )
    if args.buffer_steps < args.wm_window or args.warmup_steps < args.wm_window:
        parser.error("buffer_steps and warmup_steps must be >= wm_window")
    if (
        args.wm_only_iters < 0
        or args.action_noise < 0
        or not 0 <= args.lam <= 1
        or args.lr <= 0
    ):
        parser.error("Invalid warmup, action noise, lambda or learning rate")
    if args.utd <= 0 or not 0 < args.gamma < 1:
        parser.error("utd must be positive and gamma must be in (0, 1)")
    if args.pi_batch * args.horizon < args.critic_minibatches:
        parser.error("pi_batch * horizon must be >= critic_minibatches")
    if not args.resume and any(
        (Path(args.log_dir) / name).exists() for name in ("last.pt", "train_log.csv")
    ):
        parser.error(
            "Log directory contains a previous run; use --resume or a new --log_dir"
        )
    distributed = DistributedContext.from_environment(dev)
    atexit.register(distributed.close)
    dev = args.device = distributed.device
    args.world_size = distributed.world_size
    args.global_num_envs = args.num_envs * distributed.world_size
    args.global_wm_batch = args.wm_batch * distributed.world_size
    args.global_pi_batch = args.pi_batch * distributed.world_size
    if distributed.is_main:
        os.makedirs(args.log_dir, exist_ok=True)
        print(
            f"[train] {distributed.world_size} worker(s) | envs {args.num_envs}/worker, {args.global_num_envs} total "
            f"| world-model batch {args.global_wm_batch} total | policy batch {args.global_pi_batch} total",
            flush=True,
        )
    print(
        f"[train] rank {distributed.rank}: {dev}, sampling seed {args.seed + distributed.rank}",
        flush=True,
    )
    distributed.barrier()

    kin = G1Kinematics(dev)
    motion = MotionLib(
        args.dataset_dir, dev, args.max_clips_per_skill, skills=args.skills
    )
    if not bool((motion.clip_len > args.horizon).any()):
        parser.error("At least one motion must contain more frames than horizon")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
    K, Hh, HZ = args.goal_frames, args.hist, args.horizon
    GOAL_DIM = GOAL_FRAME_DIM * K
    OBS_DIM = mdp.obs_dim
    CRITIC_IN = mdp.critic_dim + GOAL_DIM
    args.max_boxes = motion.max_boxes
    args.obs_dim = OBS_DIM
    args.format = FORMAT
    args.reward_definition = REWARD_DEFINITION
    args.observation_terms = list(mdp.obs_slices)
    args.skills = list(motion.skill_names)
    args.critic_dim = mdp.critic_dim
    args.motion_manifest = [
        str(p.relative_to(Path(args.dataset_dir).expanduser().resolve()))
        for p in motion.paths
    ]

    args.goal_dim = GOAL_DIM
    args.iteration_unit = "vectorized_simulator_step"
    args.effective_lr = args.lr * distributed.world_size**0.5
    thresholds = tuple(
        mdp.teacher_cfg.terminations[name].params["threshold"]
        for name in ("anchor_pos", "anchor_ori", "ee_body_pos")
    )
    wm = WorldModel(
        terrain_dim=motion.terrain_dim,
        hidden=tuple(args.wm_hidden),
        termination_thresholds=thresholds,
    ).to(dev)
    pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh, hidden=tuple(args.policy_hidden)).to(dev)
    critic = Critic(CRITIC_IN, hidden=tuple(args.critic_hidden)).to(dev)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
    start_it, wall_offset, sim_steps = 0, 0.0, 0
    gpu_offset = 0.0
    num_updates, update_acc = 0, 0.0
    sampling_state = None
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        if not all(k in ck for k in ("critic", "opt_wm", "opt_pi", "opt_v")):
            parser.error("Resume requires last.pt, not an evaluation-only ckpt_*.pt")
        for key in (
            "format",
            "reward_definition",
            "goal_dim",
            "wm_hidden",
            "policy_hidden",
            "critic_hidden",
            "iteration_unit",
            "hist",
            "goal_frames",
            "max_boxes",
            "obs_dim",
            "skills",
            "motion_manifest",
        ):
            if key not in ck["args"]:
                parser.error(f"Checkpoint is missing PHP metadata: {key}")
            if ck["args"][key] != getattr(args, key):
                parser.error(
                    f"--{key} must match the resumed checkpoint ({ck['args'][key]})"
                )
        (
            wm.load_state_dict(ck["wm"]),
            pi.load_state_dict(ck["pi"]),
            critic.load_state_dict(ck["critic"]),
        )
        (
            opt_wm.load_state_dict(ck["opt_wm"]),
            opt_pi.load_state_dict(ck["opt_pi"]),
            opt_v.load_state_dict(ck["opt_v"]),
        )
        start_it, wall_offset, sim_steps = (
            ck["iteration"],
            ck["wall_time"],
            ck["sim_steps"],
        )
        num_updates = ck["num_updates"]
        update_acc = ck["update_acc"]
        sampling_state = ck["sampling"]
        old_gpu_count = (
            ck["args"].get("world_size", 1)
            if ck["args"].get("device", "cuda").startswith("cuda")
            else 0
        )
        gpu_offset = ck.get("gpu_time", wall_offset * old_gpu_count)

    if args.iterations <= start_it:
        parser.error("--iterations must exceed the resumed checkpoint iteration")
    if distributed.is_main:
        with (Path(args.log_dir) / "args.json").open("w") as stream:
            json.dump(vars(args), stream, indent=2)

    for model in (wm, pi, critic):
        distributed.broadcast_module(model)
    # Networks start identically; environment randomization, replay sampling and
    # exploration use a distinct stream on every worker.
    torch.manual_seed(args.seed + distributed.rank)

    def _wm_rollout(s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(args.wm_window):
            S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple(
            torch.stack([P[i] for P in preds], 1) for i in range(6)
        )

    def _wm_err(S_hat, s1, mask, frames, term):
        sq = wm.state_error(S_hat, unflatten_state(s1)).square()
        next_frames = torch.minimum(frames + 1, motion.clip_end[motion.clip_id[frames]])
        logits = wm.termination_logits(mdp.termination_errors(S_hat, next_frames))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, term.float(), reduction="none"
        )
        weight = mask / mask.sum().clamp_min(1)
        loss = ((sq.sum(-1) + bce) * weight).sum()
        return loss, sq.mean(-1), (bce * mask).sum()

    def _imag_step(S, o_hist, a_prev, f, xi, xi_n, g):
        a = pi(o_hist, g)
        S1 = wm.step(S, a.clamp(-20.0, 20.0), xi_n)
        f1 = f + 1
        o1 = mdp.add_noise(mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return S1, a, o_hist, f1, motion.goal(f1, K)

    def _reward_term(S, f, a, a_prev):
        return mdp.reward(S, f, a, a_prev), torch.sigmoid(
            wm.termination_logits(mdp.termination_errors(S, f))
        )

    def _critic_input(S, f, xi_n, g, a_prev):
        return torch.cat((mdp.critic_features(S, a_prev, xi_n, f), g), -1)

    compile_fn = (lambda f: f) if args.no_compile else torch.compile
    wm_rollout = compile_fn(_wm_rollout)
    wm_err = compile_fn(_wm_err)
    imag_step = compile_fn(_imag_step)
    reward_term = compile_fn(_reward_term)
    critic_input_c = compile_fn(_critic_input)

    def lr_at(it: int) -> float:
        return cosine_lr(it, args.lr_decay_iters, args.effective_lr, decay_fraction=1.0)

    def set_requires_grad(m: nn.Module, flag: bool):
        for p in m.parameters():
            p.requires_grad_(flag)

    # ------------------------------------------------------------------------------------------------
    def wm_update():
        w = buffer.sample_windows(args.wm_batch, args.wm_window)
        S_hat = wm_rollout(w["s"], w["a"], w["xi"])
        loss, errors, term_sum = wm_err(
            S_hat, w["s1"], w["mask"], w["f"], w["term"]
        )
        opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        distributed.sync_gradients(wm)
        gn = nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
        if torch.isfinite(gn):
            opt_wm.step()
        m = w["mask"][:, -1]
        # Log fixed-scale state NSE over valid
        # transitions/windows, so shorter episodes do not artificially reduce loss.
        return (
            (errors.detach() * w["mask"]).sum(),
            w["mask"].sum(),
            (errors[:, -1].detach() * m).sum(),
            m.sum(),
            term_sum.detach(),
        )

    def sample_valid_starts(B: int):
        starts, hists, n = [], [], 0
        attempts = 0
        while n < B:
            attempts += 1
            if attempts > 100:
                raise RuntimeError(
                    "Replay has too few starts with horizon frames remaining; collect more data or shorten --horizon"
                )
            s, h = buffer.sample_starts(2 * B, Hh)
            f = s["f"]
            ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            (
                starts.append({k: v[idx] for k, v in s.items()}),
                hists.append({k: v[idx] for k, v in h.items()}),
            )
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
        return cat(starts), cat(hists)

    PROFILE = (
        bool(int(os.environ.get("SM_PROFILE", "0")))
        and dev.startswith("cuda")
        and distributed.is_main
    )

    def policy_update():
        if PROFILE:
            torch.cuda.synchronize()
            tq0 = time.time()
        s, h = sample_valid_starts(args.pi_batch)
        S = unflatten_state(s["s"])
        xi, f, a_prev = s["xi"], s["f"], s["a_prev"]
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = mdp.add_noise(o_hist)
        set_requires_grad(wm, False)
        set_requires_grad(critic, False)
        B = args.pi_batch
        g0 = motion.goal(f, K)
        g = g0
        S0 = S
        Ss, As, Aps, Fs, Gs = [], [], [], [], []
        for t in range(HZ):
            Aps.append(a_prev)
            S, a_prev, o_hist, f, g = imag_step(
                S, o_hist, a_prev, f, xi, xi_n, g
            )
            Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g)
        flat = lambda xs: torch.stack(xs).flatten(0, 1)  # noqa: E731  (H*B, ...)
        S_all = tuple(flat([St[i] for St in Ss]) for i in range(6))
        f_all = flat(Fs)
        r, d = reward_term(S_all, f_all, flat(As), flat(Aps))
        r, d = r.view(HZ, B), d.view(HZ, B)
        alive = torch.cumprod(1 - d, 0)  # c_{t+1}
        c_t = torch.cat((torch.ones(1, B, device=dev), alive[:-1]), 0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        rews, dones = r.detach(), d.detach()
        terminal_values = (
            None if args.no_critic else critic(_critic_input(S, f, xi_n, g, a_prev))
        )
        ro = ImagRollout(r, torch.cat((torch.ones(1, B, device=dev), alive), 0))
        loss = -policy_objective(ro, terminal_values, args.gamma) / HZ
        if PROFILE:
            torch.cuda.synchronize()
            tq1 = time.time()
        opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if PROFILE:
            torch.cuda.synchronize()
            tq2 = time.time()
        distributed.sync_gradients(pi)
        gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(pi.parameters(), 0.1)
        if torch.isfinite(gn):
            opt_pi.step()
        set_requires_grad(wm, True)
        set_requires_grad(critic, True)
        # ---------------- critic: GAE(lambda) targets over the imagined transitions ----------------
        v_loss_sum = torch.zeros((), device=dev)
        v_count = 0
        if not args.no_critic:
            with torch.no_grad():
                S_c = tuple(
                    torch.cat(
                        (
                            S0[i].unsqueeze(0),
                            torch.stack([St[i].detach() for St in Ss]),
                        ),
                        0,
                    ).flatten(0, 1)
                    for i in range(6)
                )
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
                ap_c = torch.cat(
                    (s["a_prev"].unsqueeze(0), torch.stack(As).detach()), 0
                ).flatten(0, 1)
                X = critic_input_c(S_c, f_c, xi_c, g_c, ap_c).view(HZ + 1, B, -1)
            with torch.no_grad():
                V = critic(X)
                targets = gae_targets(rews, dones, V, args.gamma, args.lam)
            Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
            n = Xf.shape[0]
            for _ in range(args.critic_epochs):
                perm = torch.randperm(n, device=dev)
                for i in range(args.critic_minibatches):
                    idx = torch.tensor_split(perm, args.critic_minibatches)[i]
                    v_loss = ((critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    v_loss_sum += v_loss.detach() * idx.numel()
                    v_count += idx.numel()
                    opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    distributed.sync_gradients(critic)
                    gn_v = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    if torch.isfinite(gn_v):
                        opt_v.step()
        if PROFILE:
            torch.cuda.synchronize()
            print(
                f"[prof] imagine fwd {1e3 * (tq1 - tq0):.0f} ms | bwd {1e3 * (tq2 - tq1):.0f} ms | critic {1e3 * (time.time() - tq2):.0f} ms",
                flush=True,
            )
        v_loss_mean = (
            v_loss_sum / v_count
            if v_count
            else torch.full((), float("nan"), device=dev)
        )
        return (
            loss.detach(),
            v_loss_mean,
            (r_sum / HZ / MAX_STEP_REWARD).mean(),
            gn.detach(),
            c.mean().detach(),
        )

    # ------------------------------------------------------------------------------------------------
    with TrackingSim(
        args.num_envs,
        motion,
        mdp,
        device=dev,
        gamma=args.gamma,
        ccd_iterations=args.ccd_iterations,
        strict_numerics=args.strict_numerics,
        numerical_error_dir=Path(args.log_dir) / "numerical_errors" / f"rank{distributed.rank}",
    ) as env:
        if sampling_state is not None:
            env.bin_G.copy_(sampling_state["bin_G"])
            env.bin_w_avg.copy_(sampling_state["bin_w_avg"])
        buffer = ReplayBuffer(
            args.buffer_steps, args.num_envs, dev, context_dim=motion.context_dim
        )
        obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)

        @torch.no_grad()
        def collect(n_steps: int, stats: dict):
            nonlocal obs_hist, sim_steps
            for _ in range(n_steps):
                o = mdp.add_noise(env.observation())
                fresh = env.replay_step == 0
                obs_hist = torch.where(
                    fresh[:, None, None],
                    o.unsqueeze(1).expand_as(obs_hist),
                    torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1),
                )
                g = motion.goal(env.f, K)
                clean_action = pi(obs_hist, g)
                a = clean_action
                if args.action_noise:
                    a = a + args.action_noise * torch.randn_like(a)
                out = env.step(a.clamp(-20.0, 20.0), clean_action=clean_action)
                buffer.add(out)
                sim_steps += args.global_num_envs
                stats["r"].append(out["r"].mean() / MAX_STEP_REWARD)
                if "ep_len" in out:
                    stats["ep_len"].append(out["ep_len"].sum())
                    stats["ep_ret"].append(out["ep_ret_norm"].sum())
                    stats["ep_return"].append(out["ep_return"].sum())
                    stats["term"].append(out["term"].float().sum())
                    stats["done"].append(out["done"].float().sum())
                if "ep_skill" in out:
                    stats["skill"].append(out["ep_skill"])
                    stats["skill_return"].append(out["ep_return"])
                    stats["skill_failed"].append(out["ep_failed"])

        stats = {
            "r": [],
            "ep_len": [],
            "ep_ret": [],
            "ep_return": [],
            "term": [],
            "done": [],
            "skill": [],
            "skill_return": [],
            "skill_failed": [],
        }
        # Replay is intentionally not checkpointed; refill it on resume too.
        if args.resume:
            collect(args.warmup_steps, stats)
        log_f = (
            open_csv_log(
                Path(args.log_dir) / "train_log.csv", LOG_FIELDS, missing="nan"
            )
            if distributed.is_main
            else None
        )
        tb_writer = None
        if distributed.is_main and not args.no_tensorboard:
            try:
                tb_writer = create_writer(
                    Path(args.log_dir) / "tensorboard",
                    purge_step=start_it + 1 if args.resume else None,
                )
            except ImportError:
                print(
                    "[train] TensorBoard unavailable; install with: python -m pip install tensorboard. CSV logging continues.",
                    flush=True,
                )
            if tb_writer is not None:
                atexit.register(tb_writer.close)
                tb_writer.add_text(
                    "Configuration/args",
                    "```json\n" + json.dumps(vars(args), indent=2) + "\n```",
                    start_it,
                )
                print(
                    f"[train] TensorBoard: {Path(args.log_dir) / 'tensorboard'}",
                    flush=True,
                )
        if distributed.is_main:
            skills_path = Path(args.log_dir) / "skills.csv"
            if not skills_path.exists():
                skills_path.write_text(
                    "iteration,skill,episodes,normalized_return,failure_rate\n"
                )
        t_start = time.time() - wall_offset
        t_log = time.time()
        collection_steps = args.warmup_steps if args.resume else 0
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        term_metrics = []
        for it in range(start_it, args.iterations):
            lr = lr_at(it + 1)
            for o in (opt_wm, opt_pi, opt_v):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            collect(1, stats)
            collection_steps += 1
            if collection_steps >= args.warmup_steps:
                update_acc += args.utd
                while update_acc + 1e-9 >= 1.0:
                    update_acc = max(0.0, update_acc - 1.0)
                    for _ in range(args.wm_batches):
                        error_sum, valid_steps, error8_sum, valid_windows, term_sum = (
                            wm_update()
                        )
                        agg["wm"].append((error_sum, valid_steps))
                        agg["wm8"].append((error8_sum, valid_windows))
                        term_metrics.append((term_sum, valid_steps))
                    if num_updates >= args.wm_only_iters:
                        pl, vl, ir, gn, surv = policy_update()
                        for key, value in zip(
                            ("pi", "v", "ir", "gn", "surv"), (pl, vl, ir, gn, surv)
                        ):
                            agg[key].append((value, 1))
                    num_updates += 1
            if (it + 1) % args.log_every == 0:
                total = lambda x: sum(float(v) for v in x)  # noqa: E731
                # Reduce sums/counts so workers with few completed episodes are not
                # weighted as heavily as those with many completed episodes.
                keys = list(agg)
                packed = [
                    value
                    for key in keys
                    for value in (
                        total(v for v, _ in agg[key]),
                        total(n for _, n in agg[key]),
                    )
                ]
                packed += [
                    total(stats["r"]),
                    len(stats["r"]),
                    total(stats["ep_len"]),
                    total(stats["ep_ret"]),
                    total(stats["term"]),
                    total(stats["done"]),
                    total(stats["ep_return"]),
                    env.numerical_failures,
                ]
                reduced = distributed.sum_values(packed)

                def ratio(a, b):
                    return a / b if b else float("nan")

                averages = [
                    ratio(reduced[2 * i], reduced[2 * i + 1]) for i in range(len(keys))
                ]
                (
                    reward_sum,
                    reward_count,
                    ep_len_sum,
                    ep_ret_sum,
                    term_sum,
                    done_sum,
                    ep_return_sum,
                    numerical_failures,
                ) = reduced[2 * len(keys) :]
                ips = args.log_every / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                term_frac = ratio(term_sum, done_sum)
                row = [
                    it + 1,
                    wall_h,
                    sim_steps,
                    *averages,
                    ratio(reward_sum, reward_count),
                    ratio(ep_len_sum, done_sum),
                    ratio(ep_ret_sum, done_sum),
                    term_frac,
                    ips,
                    lr,
                    ratio(ep_return_sum, done_sum),
                    3,
                ]
                term_sum, term_count = distributed.sum_values(
                    [
                        total(v for v, _ in term_metrics),
                        total(n for _, n in term_metrics),
                    ]
                )
                row += [ratio(term_sum, term_count), num_updates]
                if log_f is not None:
                    log_f.write(
                        ",".join(
                            f"{x:.6g}" if isinstance(x, float) else str(x) for x in row
                        )
                        + "\n"
                    )
                    log_f.flush()
                    if tb_writer is not None:
                        write_scalars(tb_writer, dict(zip(LOG_FIELDS, row)))
                        tb_writer.add_scalar("sim_numerical_failures", numerical_failures, it + 1)
                        tb_writer.flush()
                    print(
                        f"it {it + 1} | {wall_h:.3f} h | return {row[16]:.3f} fail {term_frac:.3f} "
                        f"| wm {row[3]:.4f} value {row[6]:.4f} grad {row[8]:.3g} "
                        f"| numerr {int(numerical_failures)} | {ips:.2f} it/s",
                        flush=True,
                    )
                # Aggregate completed episodes per skill across workers.
                skill_stats = []
                ids = (
                    torch.cat(stats["skill"])
                    if stats["skill"]
                    else torch.empty(0, device=dev)
                )
                returns = (
                    torch.cat(stats["skill_return"])
                    if stats["skill_return"]
                    else torch.empty(0, device=dev)
                )
                failures = (
                    torch.cat(stats["skill_failed"])
                    if stats["skill_failed"]
                    else torch.empty(0, device=dev)
                )
                for skill in range(len(motion.skill_names)):
                    mask = ids == skill
                    skill_stats.extend(
                        (
                            mask.sum().item(),
                            returns[mask].sum().item(),
                            failures[mask].sum().item(),
                        )
                    )
                skill_stats = distributed.sum_values(skill_stats)
                if distributed.is_main:
                    with (Path(args.log_dir) / "skills.csv").open("a") as stream:
                        for i, name in enumerate(motion.skill_names):
                            count, total_return, failed = skill_stats[3 * i : 3 * i + 3]
                            ret, failure = (
                                ratio(total_return, count),
                                ratio(failed, count),
                            )
                            stream.write(
                                f"{it + 1},{name},{int(count)},{ret},{failure}\n"
                            )
                            if tb_writer is not None and count:
                                tb_writer.add_scalar(
                                    f"Skills/{name}/return", ret, it + 1
                                )
                                tb_writer.add_scalar(
                                    f"Skills/{name}/failure_rate", failure, it + 1
                                )
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
                term_metrics = []
            if distributed.is_main and (
                (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations
            ):
                ck = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "opt_wm": opt_wm.state_dict(),
                      "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": it + 1,
                      "wall_time": time.time() - t_start, "sim_steps": sim_steps, "args": vars(args),
                      "gpu_time": gpu_offset + (time.time() - t_start - wall_offset) * (args.world_size if dev.startswith("cuda") else 0),
                      "num_updates": num_updates, "update_acc": update_acc,
                      "sampling": {"bin_G": env.bin_G, "bin_w_avg": env.bin_w_avg},
                      "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
                last = os.path.join(args.log_dir, "last.pt")
                torch.save(ck, last + ".tmp")
                os.replace(last + ".tmp", last)
                small = {
                    k: ck[k]
                    for k in (
                        "pi",
                        "wm",
                        "iteration",
                        "wall_time",
                        "gpu_time",
                        "sim_steps",
                        "args",
                        "saved_at",
                    )
                }
                path = os.path.join(args.log_dir, f"ckpt_{it + 1:06d}.pt")
                torch.save(small, path + ".tmp")
                os.replace(path + ".tmp", path)

        if log_f is not None:
            log_f.close()
        if tb_writer is not None:
            tb_writer.close()
            atexit.unregister(tb_writer.close)
    distributed.barrier()
    distributed.close()


if __name__ == "__main__":
    main()
