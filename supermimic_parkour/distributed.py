"""Synchronous data parallelism for SuperMimic's alternating BPTT updates.

Gradient collectives run after backward, outside compiled rollouts. This keeps
world-model input gradients available while its parameters are frozen during
policy updates, and does not require wrapping WorldModel.step in a DDP forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    device: str
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    @classmethod
    def from_environment(cls, device: str):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device.startswith("cuda"):
            device = f"cuda:{local_rank}" if world_size > 1 else device
            torch.cuda.set_device(device)
        ctx = cls(device, rank, world_size, local_rank)
        if ctx.enabled:
            dist.init_process_group(
                backend="nccl" if device.startswith("cuda") else "gloo",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(minutes=30),
            )
        return ctx

    @torch.no_grad()
    def broadcast_module(self, module):
        if self.enabled:
            for tensor in module.state_dict().values():
                dist.broadcast(tensor, src=0)

    @torch.no_grad()
    def sync_gradients(self, module):
        """Average gradients before clipping; missing local gradients count as zero.

        All ranks use the same trainable parameter list. Thus an unused local gradient
        cannot change collective ordering, and a nonfinite gradient reaches all ranks
        so they make the same optimizer-step decision.
        """
        if not self.enabled:
            return
        params = [p for p in module.parameters() if p.requires_grad]
        if not params:
            return
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in params
            ]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        offset = 0
        for p in params:
            value = flat[offset : offset + p.numel()].view_as(p)
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.copy_(value)
            offset += p.numel()

    @torch.no_grad()
    def update_normalizer(self, normalizer, x):
        if not self.enabled:
            normalizer.update(x)
            return
        if normalizer.count >= normalizer.max_count:
            return  # Counts are global and identical, including after resume.
        x = x.reshape(-1, x.shape[-1]).float()
        n, dim = x.shape
        if n:
            var, mean = torch.var_mean(x, dim=0, unbiased=False)
        else:
            mean, var = x.new_zeros(dim), x.new_zeros(dim)
        # Gather batch moments, not history or samples. Float64 merging avoids
        # subtracting large raw second moments for nearly constant features.
        packed = torch.cat((x.new_tensor([n]), mean, var)).double()
        gathered = [torch.empty_like(packed) for _ in range(self.world_size)]
        dist.all_gather(gathered, packed)
        moments = torch.stack(gathered)
        count = moments[:, 0].sum()
        if count == 0:
            return
        weights = moments[:, :1] / count
        means, variances = moments[:, 1 : dim + 1], moments[:, dim + 1 :]
        mean = (weights * means).sum(0)
        var = (weights * (variances + (means - mean).square())).sum(0)
        normalizer.update_from_moments(
            count.to(normalizer.count), mean.to(normalizer.mean), var.to(normalizer.var)
        )

    def sum_values(self, values):
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
