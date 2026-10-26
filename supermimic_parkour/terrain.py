"""A small pool of mocap boxes, resized and placed independently per environment."""

import mujoco
import torch

from mjlab.managers.event_manager import requires_model_fields


def box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box", mocap=True, pos=(0, 0, -50))
    body.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(0.5, 0.5, 0.5),
        group=0,
        rgba=(0.35, 0.59, 0.78, 1),
        contype=2,
        conaffinity=0,
        condim=3,
        friction=(0.9, 0.005, 0.0001),
    )
    return spec


@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def prepare_box_fields(env, env_ids):
    """Declare resized fields before CUDA graphs and ray sensors are initialized."""


def place_boxes(command, env_ids: torch.Tensor) -> None:
    env = command._env
    boxes = command.motion.terrain[
        command.motion_ids[env_ids], command.variant_ids[env_ids]
    ]
    for slot, entity in enumerate(command.terrain_entities):
        box = boxes[:, slot]
        half_size = box[:, 7:] * 0.5
        gid = entity.indexing.geom_ids[0]
        env.sim.model.geom_size[env_ids, gid] = half_size
        env.sim.model.geom_rbound[env_ids, gid] = torch.linalg.vector_norm(
            half_size, dim=-1
        )
        env.sim.model.geom_aabb[env_ids, gid, 0] = 0
        env.sim.model.geom_aabb[env_ids, gid, 1] = half_size
        pose = box[:, :7].clone()
        pose[:, :3] += env.scene.env_origins[env_ids]
        entity.write_mocap_pose_to_sim(pose, env_ids=env_ids)
