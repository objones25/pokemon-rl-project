import pytest
import torch
from torch import nn

from contrastive_pretrain.checkpoint import (
    build_checkpoint_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_optimizer_and_scheduler,
    save_checkpoint,
)


def test_build_checkpoint_state_captures_all_expected_keys() -> None:
    model = nn.Linear(2, 2)
    projector = nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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


def test_build_checkpoint_state_accepts_none_dataloader_state() -> None:
    """Epoch-boundary checkpoints deliberately store no dataloader state --
    the iterator that just finished the epoch is exhausted, and restoring it
    would make the resumed epoch yield nothing."""
    model = nn.Linear(2, 2)
    projector = nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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


def test_build_checkpoint_state_records_local_cache_dir() -> None:
    """The resume path compares this against the live config to decide
    whether the checkpointed dataloader state was built over the same
    pipeline structure -- see build_checkpoint_state's docstring."""
    model = nn.Linear(2, 2)
    projector = nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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


def test_build_checkpoint_state_local_cache_dir_defaults_to_none() -> None:
    """Omitting it must mean "streaming", the pre-local-cache default --
    otherwise every existing call site would record a bogus data source."""
    model = nn.Linear(2, 2)
    projector = nn.Linear(2, 4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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


def test_save_and_load_checkpoint_round_trip(tmp_path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    projector = nn.Linear(2, 4)
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


def test_restore_optimizer_and_scheduler_restores_correct_lr() -> None:
    """Test that restore_optimizer_and_scheduler correctly restores the optimizer's
    learning rate when called in the documented order (scheduler constructed
    before restore). Implicitly verifies that optimizer.load_state_dict() is
    actually called — fully omitting it would cause this test to fail."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for _ in range(10):
        scheduler.step()
    state = build_checkpoint_state(
        epoch=0,
        global_step=10,
        model=model,
        projector=nn.Linear(2, 4),
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_state={},
        best_val_loss=1.0,
    )
    restored_lr_from_original = optimizer.param_groups[0]["lr"]

    model2 = nn.Linear(2, 2)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=100)
    restore_optimizer_and_scheduler(optimizer2, scheduler2, state)

    assert optimizer2.param_groups[0]["lr"] == pytest.approx(restored_lr_from_original)
