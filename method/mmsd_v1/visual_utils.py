"""
Visual token delta computation and sequence filtering utilities for MMSD v1.

Core operations:
1. compute_visual_delta: Cross-layer cosine distance for each visual token
2. select_visual_tokens: Top-64 by delta -> top-32 (or random 32 during training)
3. filter_sequence: Keep only selected visual tokens + all text tokens
4. collect_hidden_states: Extract hidden states from specific layers and concat
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def compute_visual_delta(
    hidden_states_list: List[torch.Tensor],
    visual_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-layer cosine distance delta for each visual token.

    Args:
        hidden_states_list: List of (L+1) tensors, each [B, T+V, D]
            (output of model with output_hidden_states=True)
        visual_positions: [V] indices of visual token positions in the sequence

    Returns:
        delta: [V] accumulated cross-layer cosine distance per visual token
               (higher = more information flow = more important)
    """
    num_layers = len(hidden_states_list) - 1
    delta = torch.zeros(len(visual_positions), device=hidden_states_list[0].device)

    for l in range(num_layers):
        h_l = hidden_states_list[l][:, visual_positions]      # [B, V, D]
        h_next = hidden_states_list[l + 1][:, visual_positions]  # [B, V, D]
        # cosine distance: 1 - cos_sim, averaged over batch
        cos_dist = 1 - F.cosine_similarity(h_l, h_next, dim=-1)  # [B, V]
        delta += cos_dist.mean(0)  # [V]

    return delta


def select_top_visual_tokens(
    delta: torch.Tensor,
    top_k: int = 32,
    pre_filter_k: int = 64,
) -> torch.Tensor:
    """
    Select top-k visual tokens by delta.
    If pre_filter_k > top_k, first select top pre_filter_k, then take top top_k from those.

    Args:
        delta: [V] importance scores
        top_k: number of visual tokens to keep (default 32)
        pre_filter_k: pre-filter pool size (default 64)

    Returns:
        selected_indices: [top_k] indices into the original visual_positions
    """
    if len(delta) <= top_k:
        return torch.arange(len(delta), device=delta.device)

    if pre_filter_k > top_k and len(delta) > pre_filter_k:
        # Two-stage: top-64 -> top-32
        top64_indices = delta.topk(pre_filter_k).indices
        top64_deltas = delta[top64_indices]
        top32_in_64 = top64_deltas.topk(top_k).indices
        return top64_indices[top32_in_64]
    else:
        return delta.topk(top_k).indices


def select_random_visual_tokens(
    delta: torch.Tensor,
    top_k: int = 32,
    pre_filter_k: int = 64,
) -> torch.Tensor:
    """
    Training-time random selection: top pre_filter_k by delta, then random top_k from those.

    Args:
        delta: [V] importance scores
        top_k: number to randomly select
        pre_filter_k: pre-filter pool size

    Returns:
        selected_indices: [top_k] indices into original visual_positions
    """
    if len(delta) <= top_k:
        return torch.arange(len(delta), device=delta.device)

    if len(delta) > pre_filter_k:
        top64_indices = delta.topk(pre_filter_k).indices
    else:
        top64_indices = torch.arange(len(delta), device=delta.device)

    perm = torch.randperm(len(top64_indices), device=delta.device)[:top_k]
    return top64_indices[perm]


def get_visual_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """
    Get positions of visual tokens in the input sequence.

    Args:
        input_ids: [B, T+V] or [1, T+V]
        image_token_id: the token ID used for image placeholders

    Returns:
        visual_positions: [V] positions (indices into seq dimension)
    """
    mask = (input_ids[0] == image_token_id)
    return mask.nonzero(as_tuple=True)[0]


def filter_sequence(
    hidden_states: torch.Tensor,
    visual_positions: torch.Tensor,
    selected_visual_indices: torch.Tensor,
    total_seq_len: int,
    image_token_id: Optional[int] = None,
    input_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter a sequence tensor to keep only selected visual tokens + all text tokens.

    Args:
        hidden_states: [B, T+V, D] or [B, T+V, 3D]
        visual_positions: [V] all visual token positions
        selected_visual_indices: [K] indices into visual_positions to keep
        total_seq_len: T+V
        image_token_id: optional, for building text positions from input_ids
        input_ids: optional [B, T+V]

    Returns:
        filtered_hs: [B, T+K, D]
        keep_positions: [T+K] sorted positions kept
    """
    device = hidden_states.device

    # Build text positions = all positions NOT in visual_positions
    all_positions = torch.arange(total_seq_len, device=device)
    is_visual = torch.zeros(total_seq_len, dtype=torch.bool, device=device)
    is_visual[visual_positions] = True
    text_positions = all_positions[~is_visual]

    # Selected visual positions (absolute)
    selected_visual_positions = visual_positions[selected_visual_indices]

    # Combine and sort
    keep_positions = torch.cat([text_positions, selected_visual_positions])
    keep_positions = keep_positions.sort().values

    # Filter
    filtered_hs = hidden_states[:, keep_positions]

    return filtered_hs, keep_positions


def collect_layer_hidden_states(
    all_hidden_states: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
) -> torch.Tensor:
    """
    Collect hidden states from specific layers and concatenate along feature dim.

    Args:
        all_hidden_states: tuple of (L+1) tensors, each [B, seq_len, D]
            Index 0 = embedding output, index i = layer i output
        layer_indices: list of 3 layer indices to collect

    Returns:
        fused: [B, seq_len, 3D] concatenated hidden states
    """
    selected = [all_hidden_states[i] for i in layer_indices]
    return torch.cat(selected, dim=-1)


def get_default_layer_indices(num_layers: int) -> List[int]:
    """
    Get default 3 layer indices following EAGLE-3 convention: {2, N//2, N-3}.
    Note: index 0 = embedding, index i = output of layer i.

    Args:
        num_layers: total number of decoder layers

    Returns:
        list of 3 indices into hidden_states tuple
    """
    return [2, num_layers // 2, num_layers - 3]


def get_random_layer_indices(num_layers: int, num_select: int = 3) -> List[int]:
    """
    Randomly select layer indices for training-time augmentation.

    Args:
        num_layers: total decoder layers
        num_select: how many to select

    Returns:
        sorted list of layer indices (1-indexed, since 0 is embedding)
    """
    indices = torch.randperm(num_layers)[:num_select] + 1  # +1 to skip embedding
    return sorted(indices.tolist())
