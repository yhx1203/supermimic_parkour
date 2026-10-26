"""Use the PHP sphere-hand robot, joint order, drives and unit action scale."""

from dataclasses import replace
import re

import mujoco

from supermimic_parkour.robot_config import (
    G1_SPHERE_HAND_XML,
    get_g1_sphere_hand_robot_cfg,
)
from .dataset import BODY_NAMES

MJCF_PATH = str(G1_SPHERE_HAND_XML)
TRACKED_BODIES = list(BODY_NAMES)
_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
G1_JOINT_NAMES = [
    _model.joint(i).name
    for i in range(_model.njnt)
    if _model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
]
del _model


def action_scale(joint_name):
    return 1.0


def get_robot_cfg():
    cfg = get_g1_sphere_hand_robot_cfg()
    cfg.collisions = tuple(replace(c, conaffinity=3) for c in cfg.collisions)
    return cfg


def default_joint_pos_list():
    positions = get_robot_cfg().init_state.joint_pos
    return [
        next(
            (
                value
                for pattern, value in positions.items()
                if re.fullmatch(pattern, name)
            ),
            0.0,
        )
        for name in G1_JOINT_NAMES
    ]
