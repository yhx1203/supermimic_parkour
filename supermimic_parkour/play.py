"""Play PHP SuperMimic with reference comparison and selectable motion/terrain."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-dir", required=True,
        help="PHP bundle root; load all available clips regardless of training dataset size",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skill",
        choices=("Locomotion", "Low-Step", "High-Step", "Low-Climb", "High-Climb", "Ultra-Climb"),
    )
    parser.add_argument(
        "--motion-id", type=int, help="Global clip ID in the loaded library"
    )
    parser.add_argument("--terrain-variant", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
    parser.add_argument(
        "--max-steps", type=int, default=0, help="0 runs until interrupted"
    )
    args = parser.parse_args()
    if args.terrain_variant < 0 or args.max_steps < 0:
        parser.error("terrain-variant and max-steps must be nonnegative")
    if args.viewer == "none" and not args.max_steps:
        parser.error("--viewer none requires --max-steps")

    import torch
    from supermimic_parkour.kinematics import G1Kinematics
    from supermimic_parkour.mdp import MDP
    from supermimic_parkour.motion import CONTROL_DT, MotionLib
    from supermimic_parkour.networks import FORMAT, SUPPORTED_FORMATS, load_policy
    from supermimic_parkour.robot import default_joint_pos_list
    from supermimic_parkour.simulation import TrackingSim

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    config = checkpoint["args"]
    if config.get("format") not in SUPPORTED_FORMATS:
        parser.error("Expected a PHP SuperMimic v2/v3/v4 teacher checkpoint")
    try:
        motion = MotionLib.for_playback(args.dataset_dir, args.device)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    mdp = MDP(
        G1Kinematics(args.device),
        motion,
        torch.tensor(default_joint_pos_list(), device=args.device),
        legacy_height_scan=config["format"] != FORMAT,
    )
    if mdp.obs_dim != config["obs_dim"]:
        parser.error(
            f"Observation dimension mismatch: "
            f"current={mdp.obs_dim}, checkpoint={config['obs_dim']}"
        )

    # Playback loads only the policy. Dataset-dependent geometry/context sizes
    # belong to the training world model; the actor's input width stays fixed.
    policy = load_policy(config, checkpoint["pi"], args.device)
    goal_fn = (
        motion.legacy_goal
        if config["format"] == "php_supermimic_teacher_v2"
        else motion.goal
    )
    selected = [
        i
        for i in range(len(motion.paths))
        if (
            args.skill is None
            or motion.skill_names[int(motion.clip_skill[i])] == args.skill
        )
        and (args.motion_id is None or i == args.motion_id)
    ]
    if not selected:
        parser.error("No motions match skill/motion-id")
    if int(motion.variant_counts[selected[0]]) <= args.terrain_variant:
        parser.error("terrain-variant is unavailable for the initial clip")

    server = None
    restart = threading.Event()
    with (
        torch.inference_mode(),
        TrackingSim(
            1,
            motion,
            mdp,
            device=args.device,
            eval_mode=True,
            ccd_iterations=config.get("ccd_iterations", 200),
        ) as env,
    ):
        ids = torch.zeros(1, device=args.device, dtype=torch.long)
        variant = torch.full_like(ids, args.terrain_variant)
        active = selected[0]
        active_variant = args.terrain_variant
        labels = {
            f"{i}: {motion.paths[i].parent.name}/{motion.paths[i].stem}": i
            for i in selected
        }

        def reset(clip):
            env.reset(ids, motion.clip_start[clip : clip + 1], variant)
            return env.observation()[:, None].repeat(1, config["hist"], 1)

        history = reset(active)
        try:
            if args.viewer == "viser":
                import viser
                from supermimic_parkour.viewer import ComparisonScene

                server = viser.ViserServer(
                    host=args.host, port=args.port, label="PHP SuperMimic"
                )
                comparison = ComparisonScene(server, env)
                server.gui.add_markdown(
                    "**PHP SuperMimic**\n\n实体机器人：策略执行 · 半透明绿色：参考动作"
                )
                tabs = server.gui.add_tab_group()
                with tabs.add_tab("播放与数据", icon=viser.Icon.PLAYER_PLAY):
                    with server.gui.add_folder("数据选择"):
                        skill_names = tuple(dict.fromkeys(
                            motion.skill_names[int(motion.clip_skill[i])] for i in selected
                        ))
                        skill_selection = server.gui.add_dropdown(
                            "技能", options=("全部",) + skill_names, initial_value="全部"
                        )
                        selection = server.gui.add_dropdown(
                            "动作文件", options=tuple(labels), initial_value=next(iter(labels))
                        )
                        terrain_selection = server.gui.add_dropdown(
                            "地形变体",
                            options=tuple(
                                str(i) for i in range(int(motion.variant_counts[active]))
                            ),
                            initial_value=str(active_variant),
                        )
                    with server.gui.add_folder("播放"):
                        playing = server.gui.add_checkbox("播放", initial_value=True)
                        loop = server.gui.add_checkbox("结束后重播", initial_value=False)
                        speed = server.gui.add_slider(
                            "速度", min=0.1, max=2.0, step=0.1, initial_value=1.0
                        )
                        button = server.gui.add_button("重新开始")
                        status = server.gui.add_markdown("")
                    with server.gui.add_folder("动作对照"):
                        show_reference = server.gui.add_checkbox("显示参考动作", initial_value=True)
                        opacity = server.gui.add_slider(
                            "参考不透明度", min=0.1, max=1.0, step=0.05, initial_value=0.5
                        )
                with tabs.add_tab("视角与显示", icon=viser.Icon.SETTINGS):
                    comparison.scene.create_visualization_gui(
                        camera_distance=4.0, camera_azimuth=135.0,
                        camera_elevation=-25.0, show_debug_viz_control=False,
                    )

                @skill_selection.on_update
                def on_skill(_):
                    skill = skill_selection.value
                    selection.options = tuple(
                        label for label, i in labels.items()
                        if skill == "全部" or motion.skill_names[int(motion.clip_skill[i])] == skill
                    )

                @button.on_click
                def on_restart(_):
                    restart.set()

                print(f"Open http://{args.host}:{server.get_port()}", flush=True)

            def sync_view():
                if server is None:
                    return
                comparison.sync(
                    show_reference=show_reference.value, opacity=float(opacity.value)
                )
                frame = int(env.f[0] - motion.clip_start[active])
                done = bool(env.done_eval[0])
                result = "失败" if bool(env.failed_eval[0]) else "完成"
                status.content = (
                    f"**{motion.skill_names[int(motion.clip_skill[active])]}** · 动作 {active}"
                    f" · 地形 {active_variant}\n\n"
                    f"帧 {frame}/{int(motion.clip_len[active]) - 1}"
                    f" · {frame * CONTROL_DT:.2f} 秒"
                    + (f" · **{result}**" if done else "")
                )

            sync_view()
            steps = 0
            while not args.max_steps or steps < args.max_steps:
                start = time.monotonic()
                chosen = labels[selection.value] if server else active
                if server and chosen != active:
                    terrain_selection.options = tuple(
                        str(i) for i in range(int(motion.variant_counts[chosen]))
                    )
                chosen_variant = int(terrain_selection.value) if server else active_variant
                if chosen != active or chosen_variant != active_variant or restart.is_set():
                    active = chosen
                    active_variant = chosen_variant
                    variant.fill_(active_variant)
                    history = reset(active)
                    restart.clear()
                    sync_view()
                    if server:
                        playing.value = True
                if server and not playing.value:
                    sync_view()
                    time.sleep(0.02)
                    continue
                action = policy.act(history, goal_fn(env.f, config["goal_frames"]))
                env.step(action)
                steps += 1
                history = torch.cat((history[:, 1:], env.observation()[:, None]), 1)
                sync_view()
                done = bool(env.done_eval[0])
                if done:
                    print(
                        f"motion={active} steps={steps} failed={bool(env.failed_eval[0])}",
                        flush=True,
                    )
                    if server and not loop.value:
                        playing.value = False
                    else:
                        restart.set()
                if server:
                    time.sleep(
                        max(
                            0,
                            CONTROL_DT / float(speed.value)
                            - (time.monotonic() - start),
                        )
                    )
            print(f"Playback finished after {steps} physics control steps", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            if server is not None:
                server.stop()


if __name__ == "__main__":
    main()
