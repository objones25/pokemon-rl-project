"""The generic save/load/find/prune helpers, shared by contrastive_pretrain
and sequence_model.

The behaviours the two sub-projects have in common are covered by
tests/unit/test_contrastive_pretrain_checkpoint.py, which exercises them
through contrastive_pretrain.checkpoint's re-export. What is tested here is
what only the shared module has: the filename pattern is a parameter, so two
different runs can keep their checkpoints in one directory without pruning
each other's.
"""

from pathlib import Path

import pytest

from checkpointing.io import find_latest_checkpoint, prune_checkpoints, save_text_atomic


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


def test_save_text_atomic_writes_the_exact_given_text(tmp_path) -> None:
    target = tmp_path / "manifest.json"

    save_text_atomic(target, '{"update": 1}')

    assert target.read_text() == '{"update": 1}'


def test_save_text_atomic_creates_missing_parent_directories(tmp_path) -> None:
    target = tmp_path / "nested" / "dir" / "manifest.json"

    save_text_atomic(target, "hello")

    assert target.read_text() == "hello"


def test_save_text_atomic_leaves_no_temp_file_behind_on_success(tmp_path) -> None:
    target = tmp_path / "manifest.json"

    save_text_atomic(target, "hello")

    assert list(tmp_path.iterdir()) == [target]


def test_save_text_atomic_writes_through_a_temp_file_then_replace(tmp_path, monkeypatch) -> None:
    """A direct `target.write_text(...)` would also pass a naive
    'file exists with the right content' check. This proves the write goes
    through a temp file and `Path.replace`, not a direct write, by making
    `replace` raise and confirming the real target was never created --
    a regression to a direct write would make `Path.replace` never get
    called at all, so `pytest.raises` below would fail to see the raise."""
    target = tmp_path / "manifest.json"
    calls: list[tuple[Path, Path]] = []

    def spy_replace(self: Path, other: Path):
        calls.append((self, Path(other)))
        raise RuntimeError("boom")

    monkeypatch.setattr(Path, "replace", spy_replace)

    with pytest.raises(RuntimeError, match="boom"):
        save_text_atomic(target, "hello")

    assert not target.exists()
    assert calls == [(target.with_suffix(target.suffix + ".tmp"), target)]
