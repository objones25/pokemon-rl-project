"""Per-env ring-buffer KV cache for the rollout path.

Capacity is context_len, so memory is bounded regardless of episode length
-- episodes run 163,840 steps against a 1024-wide window, so an
append-only cache is not an option.

Slots are NOT in temporal order after the first wraparound, and nothing
downstream needs them to be: each key carries its own RoPE phase from
write time, and the attention mask keys off slot validity rather than slot
index. This is also why keys are never re-rotated on the way out -- doing
so restarts positions at 0 every decode step, which is the classic
cache bug (prefill perfect, rollout garbage).

write() and advance() are separate because every layer writes at the SAME
slot for a given timestep; the position advances once per env step, not
once per layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sequence_model.config import PolicyConfig


@dataclass
class RolloutCache:
    k: torch.Tensor  # (n_layers, N, n_kv_heads, context_len, head_dim)
    v: torch.Tensor
    slot_valid: torch.Tensor  # (N, context_len) bool
    write_pos: torch.Tensor  # (N,) int64
    abs_pos: torch.Tensor  # (N,) int64
    capacity: int

    @classmethod
    def empty(
        cls,
        config: PolicyConfig,
        n_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RolloutCache:
        shape = (config.n_layers, n_envs, config.n_kv_heads, config.context_len, config.head_dim)
        return cls(
            k=torch.zeros(shape, device=device, dtype=dtype),
            v=torch.zeros(shape, device=device, dtype=dtype),
            slot_valid=torch.zeros(n_envs, config.context_len, dtype=torch.bool, device=device),
            write_pos=torch.zeros(n_envs, dtype=torch.long, device=device),
            abs_pos=torch.zeros(n_envs, dtype=torch.long, device=device),
            capacity=config.context_len,
        )

    def write(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`k_new`/`v_new` are (N, n_kv_heads, 1, head_dim), already
        RoPE-rotated at their own absolute position. Returns the full
        (N, n_kv_heads, capacity, head_dim) buffers for this layer."""
        env = torch.arange(self.write_pos.shape[0], device=self.write_pos.device)
        self.k[layer][env, :, self.write_pos] = k_new[:, :, 0, :].to(self.k.dtype)
        self.v[layer][env, :, self.write_pos] = v_new[:, :, 0, :].to(self.v.dtype)
        self.slot_valid[env, self.write_pos] = True
        return self.k[layer], self.v[layer]

    def advance(self) -> None:
        """Called once per env step, after every layer has written."""
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.abs_pos = self.abs_pos + 1

    def attention_mask(self) -> torch.Tensor:
        """(N, 1, 1, capacity) bool -- one query position against every
        cached slot. Broadcasts over heads."""
        return self.slot_valid[:, None, None, :]

    def reset(self, done: torch.Tensor) -> None:
        """`done` is (N,) bool. Resetting abs_pos to 0 keeps RoPE
        positions small within an episode and makes episode masking
        automatic on the rollout path -- a reset cache holds nothing from
        the previous episode, so only a training chunk can straddle a
        boundary."""
        self.slot_valid[done] = False
        self.write_pos = torch.where(done, torch.zeros_like(self.write_pos), self.write_pos)
        self.abs_pos = torch.where(done, torch.zeros_like(self.abs_pos), self.abs_pos)
