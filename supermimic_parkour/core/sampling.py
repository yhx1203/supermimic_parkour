"""Per-frame return-to-go bin updates adapted from upstream commands.py (D14)."""

from __future__ import annotations

import torch


@torch.no_grad()
def update_bin_returns(
    bin_return,
    rewards,
    bins,
    lengths,
    terminated,
    clip_ended,
    gamma,
    *,
    ema_alpha=0.05,
    min_remaining_steps=100,
):
    """Update every visited bin; censor tails of timeouts without a value bootstrap.

    Failure and motion completion are true finite ends. A time limit alone must
    not assign an artificially low return to the final reference frames.
    """
    T = int(lengths.max().item())
    times = torch.arange(T, device=rewards.device)
    valid = times[None] < lengths[:, None]
    # Episode buffers retain old tails after reset. Multiplication by zero does
    # not mask NaN/Inf there, and the reverse cumsum would poison valid returns.
    r = torch.where(valid, rewards[:, :T].double(), 0.0)
    if not bool(torch.isfinite(r).all()):
        raise ValueError("Non-finite active episode reward in update_bin_returns")
    powers = gamma ** times.double()
    discounted = r * powers
    rtg = discounted.flip(1).cumsum(1).flip(1) / powers
    truncated = ~terminated & ~clip_ended
    remaining = lengths[:, None] - times[None]
    valid &= ~(truncated[:, None] & (remaining < min_remaining_steps))
    b = bins[:, :T][valid]
    g = rtg[valid].to(bin_return.dtype)
    if not bool(torch.isfinite(g).all()):
        raise ValueError("Non-finite return-to-go; check rewards and discount gamma")
    sums = torch.zeros_like(bin_return).scatter_add_(0, b, g)
    counts = torch.zeros_like(bin_return).scatter_add_(0, b, torch.ones_like(g))
    touched = counts > 0
    bin_return[touched] = (1 - ema_alpha) * bin_return[touched] + ema_alpha * sums[
        touched
    ] / counts[touched]
