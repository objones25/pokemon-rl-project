import pytest
import torch
from torch import nn

from contrastive_pretrain.checkpoint import (
    build_checkpoint_state,
    find_latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    restore_optimizer_and_scheduler,
    save_checkpoint,
)


@pytest.fixture
def make_training_components():
    def _make():
        model = nn.Linear(2, 2)
        projector = nn.Linear(2, 4)
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(projector.parameters()), lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return model, projector, optimizer, scheduler

    return _make


def test_build_checkpoint_state_captures_all_expected_keys(make_training_components) -> None:
    model, projector, optimizer, scheduler = make_training_components()

    state = build_checkpoint_state(
        epoch=3,
        global_step=150,
        model=model,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state={"fake": "state"},
        best_val_loss=1.23,
    )

    assert state["epoch"] == 3
    assert state["global_step"] == 150
    assert state["best_val_loss"] == pytest.approx(1.23)
    assert state["dataloader"] == {"fake": "state"}
    assert set(state["model"].keys()) == set(model.state_dict().keys())
    # The optimizer spans model + projector parameters, so a checkpoint that
    # omitted the projector would restore its Adam moments onto a freshly
    # random projection head on every resume.
    assert set(state["projector"].keys()) == set(projector.state_dict().keys())
    assert torch.equal(state["projector"]["weight"], projector.weight)
    assert "optimizer" in state
    assert "scheduler" in state
    assert "augmentation_rng" not in state  # per-row seeding needs no RNG state


def test_build_checkpoint_state_accepts_none_dataloader_state(make_training_components) -> None:
    """Epoch-boundary checkpoints deliberately store no dataloader state --
    the iterator that just finished the epoch is exhausted, and restoring it
    would make the resumed epoch yield nothing."""
    model, projector, optimizer, scheduler = make_training_components()

    state = build_checkpoint_state(
        epoch=3,
        global_step=150,
        model=model,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state=None,
        best_val_loss=1.23,
    )

    assert state["dataloader"] is None


def test_build_checkpoint_state_records_local_cache_dir(make_training_components) -> None:
    """The resume path compares this against the live config to decide
    whether the checkpointed dataloader state was built over the same
    pipeline structure -- see build_checkpoint_state's docstring."""
    model, projector, optimizer, scheduler = make_training_components()

    state = build_checkpoint_state(
        epoch=3,
        global_step=150,
        model=model,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state={"fake": "state"},
        best_val_loss=1.23,
        local_cache_dir="/workspace/foo",
    )

    assert state["local_cache_dir"] == "/workspace/foo"


def test_build_checkpoint_state_local_cache_dir_defaults_to_none(make_training_components) -> None:
    """Omitting it must mean "streaming", the pre-local-cache default --
    otherwise every existing call site would record a bogus data source."""
    model, projector, optimizer, scheduler = make_training_components()

    state = build_checkpoint_state(
        epoch=3,
        global_step=150,
        model=model,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state={"fake": "state"},
        best_val_loss=1.23,
    )

    assert state["local_cache_dir"] is None


def test_save_and_load_checkpoint_round_trip(tmp_path, make_training_components) -> None:
    model, projector, optimizer, scheduler = make_training_components()
    state = build_checkpoint_state(
        epoch=3,
        global_step=150,
        model=model,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state={"fake": "state"},
        best_val_loss=1.23,
    )
    path = tmp_path / "checkpoint_step00000150.pt"

    save_checkpoint(path, state)
    loaded = load_checkpoint(path)

    assert loaded["epoch"] == 3
    assert loaded["global_step"] == 150
    assert loaded["dataloader"] == {"fake": "state"}
    assert torch.equal(loaded["projector"]["weight"], projector.weight)


def test_save_checkpoint_creates_parent_dirs(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "checkpoint.pt"
    save_checkpoint(path, {"a": 1})
    assert path.exists()


def test_find_latest_checkpoint_picks_highest_step(tmp_path) -> None:
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000500.pt").write_bytes(b"")

    result = find_latest_checkpoint(tmp_path)

    assert result == tmp_path / "checkpoint_step00000900.pt"


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path) -> None:
    assert find_latest_checkpoint(tmp_path) is None


def test_find_latest_checkpoint_returns_none_when_directory_does_not_exist(tmp_path) -> None:
    assert find_latest_checkpoint(tmp_path / "nonexistent") is None


def test_load_checkpoint_raises_on_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_prune_checkpoints_keeps_only_the_newest_n(tmp_path) -> None:
    """Each checkpoint is ~336MB (measured: ResNet-50 encoder + projector +
    AdamW moments). Unpruned, a 100-epoch run writes ~138 of them -- ~46GB,
    which alone nearly fills the 50GB RunPod network volume that also has to
    hold the resize cache."""
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000500.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00001300.pt").write_bytes(b"")

    prune_checkpoints(tmp_path, keep_last_n=2)

    remaining = sorted(p.name for p in tmp_path.glob("checkpoint_step*.pt"))
    assert remaining == ["checkpoint_step00000900.pt", "checkpoint_step00001300.pt"]


def test_prune_checkpoints_returns_the_paths_it_deleted(tmp_path) -> None:
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000500.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")

    deleted = prune_checkpoints(tmp_path, keep_last_n=1)

    assert deleted == [
        tmp_path / "checkpoint_step00000100.pt",
        tmp_path / "checkpoint_step00000500.pt",
    ]


def test_prune_checkpoints_deletes_nothing_when_fewer_than_n_exist(tmp_path) -> None:
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")

    deleted = prune_checkpoints(tmp_path, keep_last_n=5)

    assert deleted == []
    assert (tmp_path / "checkpoint_step00000100.pt").exists()


def test_prune_checkpoints_leaves_unrelated_files_alone(tmp_path) -> None:
    """find_latest_checkpoint globs checkpoint_step*.pt specifically, and so
    must this -- the volume also holds the resize cache."""
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    prune_checkpoints(tmp_path, keep_last_n=1)

    assert (tmp_path / "notes.txt").exists()


def test_prune_checkpoints_returns_empty_when_directory_does_not_exist(tmp_path) -> None:
    assert prune_checkpoints(tmp_path / "nonexistent", keep_last_n=3) == []


def test_prune_checkpoints_rejects_keep_last_n_below_one(tmp_path) -> None:
    """keep_last_n=0 would delete the checkpoint just written, making the run
    unresumable -- refuse rather than silently destroy it."""
    with pytest.raises(ValueError, match="keep_last_n must be at least 1"):
        prune_checkpoints(tmp_path, keep_last_n=0)


def test_prune_checkpoints_leaves_the_checkpoint_find_latest_would_pick(tmp_path) -> None:
    """The retained set must always include the resume point, or a pruned run
    resumes from an older step than the one it just saved."""
    (tmp_path / "checkpoint_step00000100.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000500.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00000900.pt").write_bytes(b"")
    (tmp_path / "checkpoint_step00001300.pt").write_bytes(b"")

    prune_checkpoints(tmp_path, keep_last_n=2)

    assert find_latest_checkpoint(tmp_path) == tmp_path / "checkpoint_step00001300.pt"


def test_restore_optimizer_and_scheduler_restores_correct_lr(make_training_components) -> None:
    """Test that restore_optimizer_and_scheduler correctly restores the optimizer's
    learning rate when called in the documented order (scheduler constructed
    before restore). Implicitly verifies that optimizer.load_state_dict() is
    actually called — fully omitting it would cause this test to fail."""
    model, projector, optimizer, scheduler = make_training_components()
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    state = build_checkpoint_state(
        epoch=0,
        global_step=10,
        model=model,
        projector=projector,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state={},
        best_val_loss=1.0,
    )
    restored_lr_from_original = optimizer.param_groups[0]["lr"]

    _, _, optimizer2, scheduler2 = make_training_components()
    restore_optimizer_and_scheduler(optimizer2, scheduler2, state)

    assert optimizer2.param_groups[0]["lr"] == pytest.approx(restored_lr_from_original)
