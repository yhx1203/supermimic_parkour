"""Unitree G1 configuration backed by the standalone sphere-hand MJCF."""

from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

G1_SPHERE_HAND_XML = Path(__file__).resolve().parent / "assets/g1/xmls/g1_29dof_spherehand.xml"
assert G1_SPHERE_HAND_XML.exists()


def get_g1_sphere_hand_spec() -> mujoco.MjSpec:
  """Load the standalone 29-DoF G1 model with half-sphere hands."""
  spec = mujoco.MjSpec.from_file(str(G1_SPHERE_HAND_XML))
  assets: dict[str, bytes] = {}
  update_assets(assets, G1_SPHERE_HAND_XML.parent / "assets", spec.meshdir)
  spec.assets = assets
  return spec


def get_g1_sphere_hand_robot_cfg() -> EntityCfg:
  """Return the standard G1 control configuration using the sphere-hand XML."""
  cfg = get_g1_robot_cfg()
  cfg.spec_fn = get_g1_sphere_hand_spec
  return cfg
