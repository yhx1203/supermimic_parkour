from dataclasses import replace

from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    GridPatternCfg,
    ObjRef,
    RayCastSensorCfg,
)
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from supermimic_parkour.robot_config import (
    get_g1_sphere_hand_robot_cfg,
)
from . import terms as mdp
from .dataset import SUPERMIMIC_BUNDLES as BUNDLES, DATA_ROOT


def php_teacher_env_cfg(skill="High-Climb", play=False):
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_g1_sphere_hand_robot_cfg()}
    robot = cfg.scene.entities["robot"]
    # Entity applies these overrides after spec_fn, so set the masks here.
    # Bit 1 preserves ground/self contact; bit 2 enables PHP terrain contact.
    # Copy each preset to avoid modifying the shared robot configuration.
    robot.collisions = tuple(
        replace(collision, conaffinity=3) for collision in robot.collisions
    )
    cfg.scene.num_envs = 1024
    velocity_range = {
        k: (-v, v)
        for k, v in zip(
            ("x", "y", "z", "roll", "pitch", "yaw"), (0.1, 0.1, 0.05, 0.1, 0.1, 0.1)
        )
    }
    cmd = mdp.PhpMotionCommandCfg(
        dataset_dir=str(DATA_ROOT / BUNDLES[skill]),
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=velocity_range,
    )
    cfg.commands["motion"] = cmd
    # A bundle directory may contain only its README. train/play rebind after
    # CLI overrides; direct callers still receive a bound scene when installed.
    if any((DATA_ROOT / BUNDLES[skill]).glob("*_motion.npz")):
        cmd.bind_scene_entities(cfg.scene)
    scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="torso_link", entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(1.6, 1.6), resolution=0.1),
        max_distance=4.0,
        include_geom_groups=(0,),
        exclude_parent_body=True,
    )
    contact = ContactSensorCfg(
        name="undesired_contact",
        primary=ContactMatch(
            mode="body",
            entity="robot",
            pattern=r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (scan, contact)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["height_scan"] = ObservationTermCfg(
            func=env_mdp.height_scan,
            params={"sensor_name": "terrain_scan", "offset": 0.5},
        )
    cfg.actions["joint_pos"].scale = 1.0
    cfg.episode_length_s = 10.0
    cfg.events["php_box_fields"] = EventTermCfg(
        func=mdp.prepare_box_fields, mode="startup"
    )
    cfg.events["push_robot"].interval_range_s = (0.5, 1.0)
    cfg.events["push_robot"].params["velocity_range"] = velocity_range
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["foot_friction"].params["ranges"] = (0.4, 1.3)
    cfg.events["encoder_bias"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!.*ankle).*$",)
    )
    cfg.events["ankle_bias"] = EventTermCfg(
        func=dr.encoder_bias,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*ankle.*",)),
            "bias_range": (-0.06, 0.06),
        },
    )
    cfg.rewards["motion_global_root_pos"].weight = 1.0
    cfg.rewards["motion_global_root_ori"].weight = 1.0
    cfg.rewards.pop("self_collisions")
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg.rewards["undesired_contacts"] = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"sensor_name": "undesired_contact", "threshold": 1.0},
    )
    cfg.terminations["anchor_pos"].func = tracking_mdp.bad_anchor_pos
    cfg.terminations["anchor_pos"].params["threshold"] = 0.5
    cfg.terminations["ee_body_pos"].func = tracking_mdp.bad_motion_body_pos
    cfg.terminations["ee_body_pos"].params.update(
        threshold=0.5,
        body_names=(
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
    )
    cfg.terminations["motion_end"] = TerminationTermCfg(
        func=mdp.motion_end, time_out=True
    )
    cfg.terminations["nan"] = TerminationTermCfg(func=env_mdp.nan_detection)
    cfg.viewer.body_name = "torso_link"
    cfg.sim.nconmax, cfg.sim.njmax = 320, 600
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = 100, 50
    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = 60.0
        cmd.sampling_mode, cmd.fixed_variant_id = "sequential", 0
        cmd.pose_range, cmd.velocity_range = {}, {}
        cfg.observations["actor"].enable_corruption = False
        cfg.events = {"php_box_fields": cfg.events["php_box_fields"]}
    return cfg
