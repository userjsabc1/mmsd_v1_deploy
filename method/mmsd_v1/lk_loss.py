"""
LK Loss: Direct Acceptance Rate Optimization for Speculative Decoding.

Reference: Samarin et al. "LK Losses" (2026)

L_LK^λ = λ·KL(p||q) + (1-λ)·TV(p,q)
λ = exp(-η · sg[α]), α = Σ min(p, q) ≈ acceptance rate

When α is low (draft is poor): λ≈1 → KL dominates (stable gradients)
When α is high (draft is good): λ≈0 → TV dominates (directly optimizes acceptance rate)
"""

import torch
import torch.nn.functional as F


def lk_loss(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    eta: float = 3.0,
    loss_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute LK Loss between draft and target logit distributions.

    Args:
        draft_logits: [B, T, V] draft model output logits
        target_logits: [B, T, V] target model output logits (detached)
        eta: temperature for adaptive blending (default 3.0)
        loss_mask: optional [B, T, 1] or [B, T] mask for valid positions

    Returns:
        scalar loss
    """
    p = F.softmax(target_logits.detach(), dim=-1)  # target distribution
    q = F.softmax(draft_logits, dim=-1)             # draft distribution

    # Approximate acceptance rate: α = Σ min(p, q)
    alpha = torch.sum(torch.min(p, q), dim=-1)  # [B, T]

    if loss_mask is not None:
        if loss_mask.dim() == 3:
            loss_mask = loss_mask.squeeze(-1)  # [B, T]
        alpha_mean = (alpha * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    else:
        alpha_mean = alpha.mean()

    # Adaptive blend weight
    lam = torch.exp(-eta * alpha_mean.detach())

    # KL divergence: KL(p || q)
    log_q = torch.log(q + 1e-10)
    kl = (p * (torch.log(p + 1e-10) - log_q)).sum(dim=-1)  # [B, T]

    # Total variation: TV(p, q) = 0.5 * Σ|p - q|
    tv = 0.5 * torch.sum(torch.abs(p - q), dim=-1)  # [B, T]

    # Combined loss per position
    loss_per_pos = lam * kl + (1 - lam) * tv  # [B, T]

    if loss_mask is not None:
        loss = (loss_per_pos * loss_mask).sum() / (loss_mask.sum() + 1e-8)
    else:
        loss = loss_per_pos.mean()

    return loss


def lk_loss_with_step_decay(
    draft_logits_list: list,
    target_logits_list: list,
    decay: float = 0.8,
    eta: float = 3.0,
    loss_mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute total LK Loss across multiple autoregressive steps with exponential decay.

    L_total = Σ_{step=0}^{N-1} decay^step × LK_Loss(draft_step, target_step)

    Args:
        draft_logits_list: list of [B, T, V] logits per step
        target_logits_list: list of [B, T, V] target logits per step
        decay: exponential decay factor per step (default 0.8)
        eta: LK Loss temperature
        loss_mask: optional mask

    Returns:
        scalar total loss
    """
    total_loss = 0.0
    for step, (draft_logits, target_logits) in enumerate(
        zip(draft_logits_list, target_logits_list)
    ):
        weight = decay ** step
        step_loss = lk_loss(draft_logits, target_logits, eta=eta, loss_mask=loss_mask)
        total_loss = total_loss + weight * step_loss
    return total_loss
