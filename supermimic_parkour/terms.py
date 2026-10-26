"""PHP-specific motion and terrain terms; standard tracking terms come from MJLab."""

from .commands import PhpMotionCommandCfg as PhpMotionCommandCfg
from .terrain import prepare_box_fields as prepare_box_fields


def undesired_contacts(env, sensor_name="undesired_contact", threshold=1.0):
    import torch

    data = env.scene[sensor_name].data
    # Count offending bodies, allowing hands and feet on obstacles by construction.
    if data.force_history is not None:
        forces = torch.linalg.vector_norm(data.force_history, dim=-1).amax(dim=-1)
    else:
        forces = torch.linalg.vector_norm(data.force, dim=-1)
    return (forces > threshold).float().sum(dim=-1)


def motion_end(env, command_name="motion"):
    cmd = env.command_manager.get_term(command_name)
    return cmd.time_steps >= cmd.motion.motion_end_idx[cmd.motion_ids] - 1
