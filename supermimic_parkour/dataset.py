from pathlib import Path

import numpy as np
import torch

from .paths import default_dataset_dir

DATA_ROOT = default_dataset_dir()
BUNDLES = {
    "Locomotion": "php-motions-locomotion-v1",
    "Low-Step": "php-motions-low-step-v1",
    "High-Step": "php-motions-high-step-v1",
    "Low-Climb": "php-motions-low-climb-76-v1",
    "High-Climb": "php-motions-high-climb-76-v1",
}
# SuperMimic additionally supports the ultra-climb bundle.
SUPERMIMIC_BUNDLES = {**BUNDLES, "Ultra-Climb": "php-motions-ultra-climb-130-v1"}
# Source NPZ files retain their original 15-column schema. The student uses
# only commands observed in these bundles; indices 11..14 are unused.
VELOCITY_COMMAND_NAMES = (
    "stand",
    "forward",
    "left_45",
    "left_90",
    "right_45",
    "right_90",
    "run_forward",
    "run_left_45",
    "run_left_90",
    "run_right_45",
    "run_right_90",
)
VELOCITY_COMMAND_DIM = len(VELOCITY_COMMAND_NAMES)
BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def motion_paths(directory: str) -> tuple[Path, ...]:
    root = Path(directory).expanduser().resolve()
    paths = tuple(sorted(root.glob("*_motion.npz")))
    if not paths:
        raise FileNotFoundError(f"No *_motion.npz files in PHP dataset: {root}")
    return paths


def read_terrain(motion_path: Path) -> np.ndarray:
    path = motion_path.with_name(
        motion_path.name.removesuffix("_motion.npz") + "_terrain.npy"
    )
    boxes = np.load(path, allow_pickle=False)
    if boxes.ndim != 3 or boxes.shape[-1] != 10 or boxes.shape[1] < 1:
        raise ValueError(f"Expected (boxes, variants, 10) in {path}, got {boxes.shape}")
    if not np.isfinite(boxes).all() or np.any(boxes[..., 7:] <= 0):
        raise ValueError(f"Nonfinite values or nonpositive box dimensions in {path}")
    if boxes.size and not np.allclose(
        np.linalg.norm(boxes[..., 3:7], axis=-1), 1, atol=1e-5
    ):
        raise ValueError(f"Terrain quaternions must be normalized wxyz in {path}")
    return boxes.astype(np.float32)


def compact_velocity_commands(velocity: np.ndarray, source) -> np.ndarray:
    """Remove unused source columns without silently losing a future command."""
    if np.any(velocity[:, VELOCITY_COMMAND_DIM:] != 0):
        raise ValueError(
            f"Unsupported vel_cmd index >= {VELOCITY_COMMAND_DIM} in {source}; "
            "the PHP student supports only the 11 observed commands"
        )
    return velocity[:, :VELOCITY_COMMAND_DIM].copy()


class MotionLibrary:
    """One immutable motion buffer and a padded [clip, variant, box, 10] table."""

    def __init__(
        self,
        paths,
        joint_names,
        body_names=BODY_NAMES,
        device="cpu",
        *,
        require_velocity=False,
    ):
        self.paths = tuple(paths)
        if not paths:
            raise ValueError("PHP needs at least one motion/terrain pair")
        keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        )
        parts = {key: [] for key in keys}
        velocity_parts = []
        lengths, terrains, fps_values = [], [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as src:
                required = set(keys) | {"fps", "joint_names", "body_names"}
                if required - set(src.files):
                    raise ValueError(
                        f"Missing fields in {path}: {sorted(required - set(src.files))}"
                    )
                fps = float(np.asarray(src["fps"]).item())
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Invalid FPS in {path}: {fps}")
                fps_values.append(fps)
                jnames, bnames = src["joint_names"].tolist(), src["body_names"].tolist()
                if len(set(jnames)) != len(jnames) or len(set(bnames)) != len(bnames):
                    raise ValueError(f"Duplicate joint/body names in {path}")
                missing = (set(joint_names) - set(jnames)) | (
                    set(body_names) - set(bnames)
                )
                if missing:
                    raise ValueError(
                        f"Missing robot names in {path}: {sorted(missing)}"
                    )
                ji = np.array([jnames.index(name) for name in joint_names])
                bi = np.array([bnames.index(name) for name in body_names])
                frames = len(src["joint_pos"])
                if frames < 2:
                    raise ValueError(f"Need at least two frames in {path}")
                lengths.append(frames)
                if require_velocity:
                    if "vel_cmd" not in src:
                        raise ValueError(f"Student requires vel_cmd in {path}")
                    velocity = src["vel_cmd"].astype(np.float32)
                    if (
                        velocity.shape != (frames, 15)
                        or not np.isfinite(velocity).all()
                    ):
                        raise ValueError(
                            f"Expected finite (frames, 15) vel_cmd in {path}"
                        )
                    velocity_parts.append(compact_velocity_commands(velocity, path))
                for key in keys:
                    value = src[key]
                    if len(value) != frames or not np.isfinite(value).all():
                        raise ValueError(
                            f"Invalid frame count or nonfinite {key} in {path}"
                        )
                    if key.startswith("joint_"):
                        if value.ndim != 2:
                            raise ValueError(f"Invalid {key} shape in {path}")
                        offset = value.shape[1] - len(jnames)
                        allowed = (0, 7) if key == "joint_pos" else (0, 6)
                        if offset not in allowed:
                            raise ValueError(
                                f"Invalid floating-base prefix in {path}: {key}"
                            )
                        value = value[:, ji + offset]
                    else:
                        width = 4 if key == "body_quat_w" else 3
                        if value.shape != (frames, len(bnames), width):
                            raise ValueError(
                                f"Invalid {key} shape in {path}: {value.shape}"
                            )
                        value = value[:, bi]
                    if key == "body_quat_w":
                        norms = np.linalg.norm(value, axis=-1, keepdims=True)
                        if np.any(norms < 1e-8):
                            raise ValueError(f"Zero quaternion in {path}")
                        value = value / norms
                    parts[key].append(value.astype(np.float32))
            terrains.append(read_terrain(path))
        if not np.allclose(fps_values, fps_values[0]):
            raise ValueError("All PHP clips in a teacher must have the same FPS")
        self.fps = fps_values[0]
        for key, values in parts.items():
            setattr(self, key, torch.as_tensor(np.concatenate(values), device=device))
        self.time_step_total = sum(lengths)
        if require_velocity:
            self.vel_cmd = torch.as_tensor(
                np.concatenate(velocity_parts), device=device
            )
        self.motion_lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        self.motion_end_idx = self.motion_lengths.cumsum(0)
        self.motion_start_idx = self.motion_end_idx - self.motion_lengths
        self.variant_counts = torch.tensor(
            [b.shape[1] for b in terrains], device=device
        )
        self.box_counts = torch.tensor([b.shape[0] for b in terrains], device=device)
        # A locomotion-only teacher still has one hidden slot, for the same scene contract.
        self.max_boxes = max(1, max(b.shape[0] for b in terrains))
        padded = np.zeros(
            (len(paths), max(b.shape[1] for b in terrains), self.max_boxes, 10),
            np.float32,
        )
        padded[..., 2], padded[..., 3], padded[..., 7:] = -50, 1, 0.02
        for i, boxes in enumerate(terrains):
            padded[i, : boxes.shape[1], : boxes.shape[0]] = boxes.transpose(1, 0, 2)
        self.terrain = torch.as_tensor(padded, device=device)
        # One-second bins stop at clip boundaries, including the final partial bin.
        bin_starts, bin_sizes, bin_clips = [], [], []
        start = 0
        for clip_id, length in enumerate(lengths):
            for local in range(0, length, round(self.fps)):
                bin_starts.append(start + local)
                bin_sizes.append(min(round(self.fps), length - local))
                bin_clips.append(clip_id)
            start += length
        self.bin_starts = torch.tensor(bin_starts, device=device)
        self.bin_sizes = torch.tensor(bin_sizes, device=device)
        self.bin_clips = torch.tensor(bin_clips, device=device)

    def sampling_probabilities(self, failures, uniform_ratio, kernel_size, decay):
        """Smooth failures backward in time without spilling into another clip."""
        score = failures.clone()
        scale = torch.ones_like(score)
        for shift in range(1, min(kernel_size, len(score))):
            same_clip = self.bin_clips[:-shift] == self.bin_clips[shift:]
            weight = decay**shift * same_clip
            score[:-shift] += failures[shift:] * weight
            scale[:-shift] += weight
        score = score / scale + uniform_ratio * self.bin_sizes / self.bin_sizes.sum()
        return score / score.sum()
