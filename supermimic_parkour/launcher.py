"""Run SuperMimic train; arguments are documented by --help."""

import sys
import argparse
import os
import subprocess
from pathlib import Path


def main():
    launcher = argparse.ArgumentParser(add_help=False)
    launcher.add_argument("--gpu-ids", "--gpu_ids", nargs="+", metavar="ID")
    options, remaining = launcher.parse_known_args()
    if options.gpu_ids is not None:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            launcher.error("Use either --gpu-ids or torchrun, not both")
        if any(arg == "--device" or arg.startswith("--device=") for arg in remaining):
            launcher.error("--gpu-ids selects CUDA devices; omit --device")
        from mjlab.utils.gpu import select_gpus

        try:
            ids = (
                "all"
                if options.gpu_ids == ["all"]
                else [int(value) for value in options.gpu_ids]
            )
            if ids != "all" and (not ids or len(set(ids)) != len(ids) or min(ids) < 0):
                raise ValueError("GPU IDs must be unique nonnegative integers")
            selected, count = select_gpus(ids)
            if not selected or count < 1:
                raise ValueError("No CUDA devices selected")
        except (ValueError, RuntimeError, IndexError) as error:
            launcher.error(str(error))
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, selected)))
        # Source-checkout wrappers also work when invoked outside the checkout.
        package_root = str(Path(__file__).resolve().parents[1])
        child_env["PYTHONPATH"] = os.pathsep.join(
            part for part in (package_root, child_env.get("PYTHONPATH", "")) if part
        )
        command = [sys.executable]
        if count > 1:
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={count}",
            ]
        command += ["-m", "supermimic_parkour", *remaining]
        print(
            f"[SuperMimic] Launching {count} worker(s) on GPUs {selected}", flush=True
        )
        raise SystemExit(subprocess.call(command, env=child_env))
    if "--help" in remaining or "-h" in remaining:
        print(
            "Launcher: --gpu-ids 0 1 (or --gpu-ids all); environment and batch sizes are per GPU.\n"
        )
    from supermimic_parkour.train import main as train

    train()


if __name__ == "__main__":
    main()
