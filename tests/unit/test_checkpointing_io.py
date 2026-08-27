"""The generic save/load/find/prune helpers, shared by contrastive_pretrain
and sequence_model.

The behaviours the two sub-projects have in common are covered by
tests/unit/test_contrastive_pretrain_checkpoint.py, which exercises them
through contrastive_pretrain.checkpoint's re-export. What is tested here is
what only the shared module has: the filename pattern is a parameter, so two
different runs can keep their checkpoints in one directory without pruning
each other's.
"""

import pytest

from checkpointing.io import find_latest_checkpoint, prune_checkpoints


def test_find_latest_checkpoint_honours_a_custom_pattern(tmp_path) -> None:
    (tmp_path / "policy_update00000100.pt").write_bytes(b"")
    (tmp_path / "policy_update00000900.pt").write_bytes(b"")

    result = find_latest_checkpoint(tmp_path, pattern="policy_update*.pt")

    assert result == tmp_path / "policy_update00000900.pt"


def test_find_latest_checkpoint_ignores_files_outside_its_pattern(tmp_path) -> None:
    """A PPO run and the pretraining run may share the network volume. If the
    pattern were ignored, `zzz_` sorting last would make it the resume point."""
    (tmp_path / "policy_update00000100.pt").write_bytes(b"")
    (tmp_path / "zzz_checkpoint_step00000900.pt").write_bytes(b"")

    result = find_latest_checkpoint(tmp_path, pattern="policy_update*.pt")

    assert result == tmp_path / "policy_update00000100.pt"


def test_prune_checkpoints_honours_a_custom_pattern(tmp_path) -> None:
    (tmp_path / "policy_update00000100.pt").write_bytes(b"")
    (tmp_path / "policy_update00000500.pt").write_bytes(b"")
    (tmp_path / "policy_update00000900.pt").write_bytes(b"")

    deleted = prune_checkpoints(tmp_path, keep_last_n=1, pattern="policy_update*.pt")

    assert deleted == [
        tmp_path / "policy_update00000100.pt",
        tmp_path / "policy_update00000500.pt",
    ]


def test_prune_checkpoints_never_deletes_files_outside_its_pattern(tmp_path) -> None:
    """The destructive counterpart of the find test above: a PPO prune that
    globbed everything would delete the pretraining run's resume point."""
    (tmp_path / "policy_update00000100.pt").write_bytes(b"")
    (tmp_path / "policy_update00000900.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")

    prune_checkpoints(tmp_path, keep_last_n=1, pattern="policy_update*.pt")

    assert (tmp_path / "checkpoint_step00000100.pt").exists()


def test_prune_checkpoints_rejects_keep_last_n_below_one(tmp_path) -> None:
    with pytest.raises(ValueError, match="keep_last_n must be at least 1"):
        prune_checkpoints(tmp_path, keep_last_n=0)
