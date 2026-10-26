"""Evaluate a shared PHP SuperMimic policy on complete clips, grouped by skill."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument(
        "--skill",
        choices=("Locomotion", "Low-Step", "High-Step", "Low-Climb", "High-Climb", "Ultra-Climb"),
    )
    parser.add_argument(
        "--motion-id",
        type=int,
        help="Global clip ID across the checkpoint skill bundles",
    )
    parser.add_argument("--terrain-variant", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 evaluates full clips; positive values truncate smoke runs",
    )
    parser.add_argument("--output", default="logs/supermimic/php_evaluation.json")
    args = parser.parse_args()
    if args.num_envs < 1 or args.terrain_variant < 0 or args.max_steps < 0:
        parser.error(
            "num-envs must be positive; terrain-variant and max-steps nonnegative"
        )

    import torch
    from supermimic_parkour.kinematics import G1Kinematics
    from supermimic_parkour.mdp import MDP, MAX_STEP_REWARD, REWARD_DEFINITION
    from supermimic_parkour.motion import MotionLib
    from supermimic_parkour.networks import FORMAT, SUPPORTED_FORMATS, load_policy
    from supermimic_parkour.robot import default_joint_pos_list
    from supermimic_parkour.simulation import TrackingSim

    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = ck["args"]
    if config.get("format") not in SUPPORTED_FORMATS:
        parser.error("Expected a PHP SuperMimic v2/v3/v4 teacher checkpoint")
    motion = MotionLib(
        args.dataset_dir,
        args.device,
        config["max_clips_per_skill"],
        skills=config["skills"],
    )
    mdp = MDP(
        G1Kinematics(args.device),
        motion,
        torch.tensor(default_joint_pos_list(), device=args.device),
        legacy_height_scan=config["format"] != FORMAT,
    )
    manifest = [
        str(p.relative_to(Path(args.dataset_dir).expanduser().resolve()))
        for p in motion.paths
    ]
    if (
        mdp.obs_dim != config["obs_dim"]
        or motion.max_boxes != config["max_boxes"]
        or manifest != config["motion_manifest"]
    ):
        parser.error(
            "Dataset manifest or observation dimensions differ from checkpoint"
        )
    policy = load_policy(config, ck["pi"], args.device)
    goal_fn = (
        motion.legacy_goal
        if config["format"] == "php_supermimic_teacher_v2"
        else motion.goal
    )
    selected = torch.arange(len(motion.paths), device=args.device)
    if args.motion_id is not None:
        if not 0 <= args.motion_id < len(selected):
            parser.error("motion-id is outside the loaded library")
        selected = selected[selected == args.motion_id]
    if args.skill is not None:
        if args.skill not in motion.skill_names:
            parser.error(f"Skill {args.skill} was not included in this checkpoint")
        selected = selected[
            motion.clip_skill[selected] == motion.skill_names.index(args.skill)
        ]
    if not len(selected):
        parser.error("No motions match the selected skill and motion-id")
    if torch.any(motion.variant_counts[selected] <= args.terrain_variant):
        parser.error("terrain-variant is unavailable for some selected clips")

    results = []
    count = min(args.num_envs, len(selected))
    with (
        torch.inference_mode(),
        TrackingSim(
            count,
            motion,
            mdp,
            device=args.device,
            eval_mode=True,
            ccd_iterations=config.get("ccd_iterations", 200),
        ) as env,
    ):
        for offset in range(0, len(selected), count):
            clips = selected[offset : offset + count]
            size = len(clips)
            padded = torch.cat((clips, clips[-1:].expand(count - size)))
            env.reset(
                torch.arange(count, device=args.device),
                motion.clip_start[padded],
                torch.full((count,), args.terrain_variant, device=args.device),
            )
            hist = env.observation()[:, None].repeat(1, config["hist"], 1)
            returns = torch.zeros(count, device=args.device)
            steps = torch.zeros(count, device=args.device, dtype=torch.long)
            limit = int((motion.clip_len[clips] - 1).max())
            if args.max_steps:
                limit = min(limit, args.max_steps)
            for _ in range(limit):
                active = ~env.done_eval
                action = policy.act(hist, goal_fn(env.f, config["goal_frames"]))
                out = env.step(action)
                returns += torch.where(active, out["r"], 0.0)
                steps += active.long()
                if bool(env.done_eval[:size].all()):
                    break
                obs = env.observation()
                hist = torch.cat((hist[:, 1:], obs[:, None]), 1)
            for i, clip in enumerate(clips.tolist()):
                results.append(
                    dict(
                        motion_id=clip,
                        path=str(motion.paths[clip]),
                        skill=motion.skill_names[int(motion.clip_skill[clip])],
                        terrain_variant=args.terrain_variant,
                        steps=int(steps[i]),
                        failed=bool(env.failed_eval[i]),
                        truncated=not bool(env.done_eval[i]),
                        normalized_return=float(
                            returns[i] / ((motion.clip_len[clip] - 1) * MAX_STEP_REWARD)
                        ),
                    )
                )
    summary = {}
    for skill in motion.skill_names:
        rows = [r for r in results if r["skill"] == skill]
        if rows:
            summary[skill] = dict(
                clips=len(rows),
                failed=sum(r["failed"] for r in rows),
                truncated=sum(r["truncated"] for r in rows),
                normalized_return=sum(r["normalized_return"] for r in rows) / len(rows),
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                checkpoint=args.checkpoint,
                reward_definition=REWARD_DEFINITION,
                iteration=ck["iteration"],
                summary=summary,
                motions=results,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
