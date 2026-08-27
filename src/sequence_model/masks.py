"""The chunked-forward attention mask: causal AND sliding-window AND
same-episode.

The sliding-window term is what makes a burn-in-prefixed training chunk
match the rollout exactly. Without it, a position early in the chunk
attends over the whole prefix -- MORE context than it had when the action
was actually chosen -- and PPO's importance ratio is then comparing two
different functions.

The same-episode term matters roughly once per 160 chunks (episodes run
163,840 steps, chunks are 1024), which is exactly the frequency at which a
bug here would never be noticed."""

from __future__ import annotations

import torch


def build_chunk_mask(
    abs_pos: torch.Tensor, episode_id: torch.Tensor, context_len: int
) -> torch.Tensor:
    """`abs_pos` and `episode_id` are (B, L) int64. Returns a (B, 1, L, L)
    boolean mask where True means "may attend", shaped to broadcast over
    the head dimension.

    abs_pos resets to 0 at an episode boundary, so `q_pos >= kv_pos` alone
    is not a valid causal test across one -- the same-episode term is what
    makes the conjunction correct, not merely stricter."""
    query = abs_pos.unsqueeze(-1)
    key = abs_pos.unsqueeze(-2)
    causal = query >= key
    window = (query - key) < context_len
    same_episode = episode_id.unsqueeze(-1) == episode_id.unsqueeze(-2)
    return (causal & window & same_episode).unsqueeze(1)
