"""Ordinary MJLab rendering with a synchronized PHP reference ghost."""

from copy import deepcopy

import torch
from mjlab.viewer.viser.scene import ViserMujocoScene


class ComparisonScene:
    def __init__(self, server, env):
        self.server, self.env = server, env
        model = env.sim.mj_model
        server.scene.set_up_direction("+z")
        self.scene = ViserMujocoScene.create(server, model, num_envs=1)
        self.scene.debug_visualization_enabled = True

        # Match PhpViserPlayViewer: compile-time meshes do not reflect resized
        # terrain. Parent the replacement boxes to the camera-tracked world.
        terrain_bodies = {
            int(body)
            for entity in env.terrain_entities
            for body in entity.indexing.body_ids
        }
        for key in tuple(self.scene.mesh_handles_by_group):
            if key[0] in terrain_bodies:
                self.scene.mesh_handles_by_group.pop(key).remove()
        world = self.scene.fixed_bodies_frame.name
        self.boxes = [
            server.scene.add_box(f"{world}/php/box_{i}", color=(90, 150, 200))
            for i in range(env.motion.max_boxes)
        ]

        self.ghost_model = deepcopy(model)
        self.ghost_model.geom_rgba[:] = env.command.cfg.viz.ghost_color
        self.free_q = env.robot.indexing.free_joint_q_adr.cpu().numpy()
        self.joint_q = env.robot.indexing.joint_q_adr.cpu().numpy()

    def reference_qpos(self):
        """Use the exact current reference frame, including its world position."""
        env, motion = self.env, self.env.motion
        frame = int(env.f[0])
        pose = torch.cat(
            (
                motion.root_p[frame] + env.scene.env_origins[0],
                motion.library.body_quat_w[frame, 0],
                motion.q[frame],
            )
        ).cpu().numpy()
        qpos = self.ghost_model.qpos0.copy()
        qpos[self.free_q] = pose[:7]
        qpos[self.joint_q] = pose[7:]
        return qpos

    def sync(self, *, show_reference=True, opacity=0.5):
        env = self.env
        self.scene.clear()
        if show_reference:
            self.scene.add_ghost_mesh(
                self.reference_qpos(), model=self.ghost_model,
                alpha=opacity, label="PHP reference",
            )
        else:
            self.scene.clear_debug_all()
        boxes = env.xi[0, 33:].reshape(-1, 10).cpu().numpy()
        origin = env.scene.env_origins[0].cpu().numpy()
        count = int(env.motion.box_counts[int(env.motion_ids[0])])
        with self.server.atomic():
            self.scene.update(env.sim.data)
            for i, (box, handle) in enumerate(zip(boxes, self.boxes)):
                handle.visible = i < count and self.scene.geom_groups_visible[0]
                if handle.visible:
                    handle.position = box[:3] + origin
                    handle.wxyz = box[3:7]
                    handle.dimensions = tuple(box[7:])
