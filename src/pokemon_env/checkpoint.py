"""Environment-side checkpoint schema.

Owns what a resumable env needs; the file I/O underneath (atomic write,
discovery, retention) is checkpointing.io, shared with contrastive_pretrain
and sequence_model.

The filename pattern is distinct so a PPO run, its policy checkpoints and the
pretraining run can share one network volume without pruning each other's
resume points -- which is why checkpointing.io takes the glob as a parameter."""

from __future__ import annotations

from pokemon_env.aux_state import AUX_STATE_VERSION
from pokemon_env.vec_env import VecPokemonEnv

ENV_CHECKPOINT_PATTERN = "env_update*.pt"


def build_env_checkpoint_state(
    update: int, vec_env: VecPokemonEnv, init_state_hash: str
) -> dict:
    return {
        "update": update,
        "aux_state_version": AUX_STATE_VERSION,
        "init_state_hash": init_state_hash,
        "env": vec_env.state_dict(),
    }


def restore_env_checkpoint(
    vec_env: VecPokemonEnv, state: dict, init_state_hash: str
) -> None:
    """`vec_env.load_state_dict` rejects a schema/AUX_STATE_VERSION or
    env-count mismatch on the NESTED copy. The things only this layer can see
    are the starting state and this envelope's own version stamp.

    A changed init.state invalidates every max_historical baseline in the
    checkpoint, because those describe progress from a game position the new
    state does not share."""
    # The envelope carries its own aux_state_version alongside the nested one
    # inside state["env"]. Both are written from the same constant, so they can
    # only diverge through a corrupted or hand-edited file -- but an unchecked
    # copy is exactly the "looks like protection, isn't" pattern this module
    # already fixed for schema_version, so it gets checked too.
    if state["aux_state_version"] != AUX_STATE_VERSION:
        raise ValueError(
            f"checkpoint envelope has aux_state_version={state['aux_state_version']}, "
            f"this build is {AUX_STATE_VERSION}. The aux vector's slot layout changed, "
            "so the policy would read different signals from the same indices."
        )
    if state["init_state_hash"] != init_state_hash:
        raise ValueError(
            f"init.state changed since this checkpoint was written "
            f"({state['init_state_hash'][:16]} -> {init_state_hash[:16]}). Every reward "
            "baseline in it describes progress from a starting position this run does "
            "not share; regenerate the checkpoint or restore the original init.state."
        )
    vec_env.load_state_dict(state["env"])
