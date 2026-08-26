import pytest
import torch
from torch import nn

from contrastive_pretrain.model import SimCLRProjector, build_encoder, build_projector
from contrastive_pretrain.train import (
    check_finite_loss,
    compute_val_loss,
    run_memory_probe,
)


def test_run_memory_probe_raises_actionable_error_on_oom() -> None:
    def _oom_step() -> None:
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="batch_size=1024"):
        run_memory_probe(_oom_step, batch_size=1024)


def test_run_memory_probe_passes_through_on_success() -> None:
    calls = []
    run_memory_probe(lambda: calls.append(1), batch_size=32)
    assert calls == [1]


def test_run_memory_probe_does_not_swallow_other_errors() -> None:
    def _other_error() -> None:
        raise ValueError("something else")

    with pytest.raises(ValueError, match="something else"):
        run_memory_probe(_other_error, batch_size=32)


def test_check_finite_loss_raises_on_nan() -> None:
    with pytest.raises(RuntimeError, match="step 42"):
        check_finite_loss(torch.tensor(float("nan")), global_step=42)


def test_check_finite_loss_raises_on_inf() -> None:
    with pytest.raises(RuntimeError, match="step 1"):
        check_finite_loss(torch.tensor(float("inf")), global_step=1)


def test_check_finite_loss_passes_for_finite_value() -> None:
    check_finite_loss(torch.tensor(0.5), global_step=1)  # must not raise


def test_compute_val_loss_averages_over_batches() -> None:
    encoder, _ = build_encoder(pretrained=False)
    projector = build_projector()

    def fake_batches():
        for _ in range(3):
            yield {
                "view_a": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
                "view_b": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
            }

    loss = compute_val_loss(
        encoder,
        projector,
        fake_batches(),
        temperature=0.5,
        device=torch.device("cpu"),
        max_batches=3,
    )

    assert isinstance(loss, float)
    assert loss > 0


def test_compute_val_loss_restores_train_mode() -> None:
    encoder, _ = build_encoder(pretrained=False)
    projector = build_projector()
    encoder.train()
    projector.train()

    def fake_batches():
        yield {
            "view_a": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
            "view_b": torch.randint(0, 256, (2, 1, 144, 160), dtype=torch.uint8),
        }

    compute_val_loss(
        encoder,
        projector,
        fake_batches(),
        temperature=0.5,
        device=torch.device("cpu"),
        max_batches=1,
    )

    assert encoder.training is True
    assert projector.training is True


def test_compute_val_loss_raises_when_no_batches_produced() -> None:
    encoder, _ = build_encoder(pretrained=False)
    projector = build_projector()

    with pytest.raises(RuntimeError, match="no batches"):
        compute_val_loss(
            encoder,
            projector,
            iter([]),
            temperature=0.5,
            device=torch.device("cpu"),
            max_batches=3,
        )


import contrastive_pretrain.train
from contrastive_pretrain import checkpoint
from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.train import TrainingDeps, run_training
from tests.conftest import FakeHfClient as _FakeHfClient


class _FakeStreamingDataset(torch.utils.data.IterableDataset):
    """Stands in for contrastive_pretrain.dataset's HF streaming datasets --
    same per-row shape (a dict with "original"/"view_a"/"view_b" (1, H, W)
    uint8 tensors) and the same set_epoch(epoch) hook run_training calls,
    but entirely in-memory so this test needs no network or HF credentials."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        for _ in range(self.n):
            yield {
                "original": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8),
                "view_a": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8),
                "view_b": torch.randint(0, 256, (1, 144, 160), dtype=torch.uint8),
            }


@pytest.mark.slow
def test_run_training_completes_and_checkpoints_at_epoch_boundary(
    tmp_path, monkeypatch
) -> None:
    """Fast, network/credential-free regression coverage for run_training --
    the plan's highest-fan-in function otherwise had zero automated
    coverage protecting its checkpoint-restore ordering (model state loaded
    before torch.compile wraps it), its raw-vs-compiled module passing
    (build_checkpoint_state/push_frozen_encoder must get the raw `encoder`,
    never `compiled_encoder`), and its two logging cadences. Monkeypatches
    the streaming-dataset builders with a tiny in-memory fake and uses
    pretrained=False so the whole run needs no network or HF credentials.

    checkpoint_interval_steps is set far above the total step count so the
    only checkpoint written is the epoch-boundary one -- this also
    regression-tests the epoch-boundary checkpoint fix (a crash between a
    val-loss improvement and the next periodic save must not be able to
    resume with a stale best_val_loss)."""
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_train_dataset",
        lambda config: _FakeStreamingDataset(n=8),
    )
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_val_dataset",
        lambda config: _FakeStreamingDataset(n=8),
    )

    config = TrainingConfig(
        pretrained=False,
        batch_size=4,
        num_workers=0,
        max_epochs=1,
        checkpoint_interval_steps=1000,
        network_volume_checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    deps = TrainingDeps(
        config=config, frozen_encoder_client=_FakeHfClient(), device=torch.device("cpu")
    )

    run_training(deps)

    checkpoints = list((tmp_path / "checkpoints").glob("checkpoint_step*.pt"))
    assert len(checkpoints) == 1


class _EpochRecordingFakeDataset(_FakeStreamingDataset):
    """Same fake stream, but records every epoch run_training announces --
    the only externally observable signal for "which epochs did this run
    actually enter?", which is what the resume off-by-one corrupts."""

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.epochs_seen: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self.epochs_seen.append(epoch)


@pytest.mark.slow
def test_run_training_resumes_projector_and_makes_progress(
    tmp_path, monkeypatch
) -> None:
    """Regression guard for the two resume bugs the `state is not None`
    branch shipped with, neither of which any earlier test entered:

    1. The projector was never checkpointed, so every resume built a fresh
       random projection head and then loaded the OLD projector's Adam
       moments onto it via restore_optimizer_and_scheduler.
    2. The epoch-boundary checkpoint stored the just-completed epoch (so
       resume re-entered an epoch already finished) AND the just-exhausted
       dataloader state.

    The stored-epoch half of (2) is asserted behaviorally below (which
    epochs the resumed run actually enters). The dataloader half is
    asserted structurally -- "an epoch-boundary checkpoint carries no
    dataloader state" -- because this fake is a plain torch
    IterableDataset, and StatefulDataLoader only falls back to a naive
    fast-forward for those: measured, restoring an exhausted state to a
    plain fake still re-yields the whole epoch, so the fake CANNOT
    reproduce the starvation. The production stream is a
    datasets.IterableDataset, which implements state_dict/load_state_dict;
    there, measured in this training loop's actual shape (a new
    set_epoch(epoch) call every epoch), restoring an exhausted state
    starves only the resumed epoch of batches -- the epoch after it
    recovers fully. Not saving stale dataloader state at the epoch
    boundary eliminates that starved epoch entirely either way.

    8 rows at batch_size=4 is 2 steps per epoch, and
    checkpoint_interval_steps=1 makes step 1 a genuine mid-epoch periodic
    checkpoint -- so this covers the periodic call site too, and asserts it
    still stores real dataloader state (that path is correct and must stay).
    """
    checkpoint_dir = tmp_path / "checkpoints"

    def _config(max_epochs: int) -> TrainingConfig:
        return TrainingConfig(
            pretrained=False,
            batch_size=4,
            num_workers=0,
            max_epochs=max_epochs,
            checkpoint_interval_steps=1,
            network_volume_checkpoint_dir=str(checkpoint_dir),
        )

    first_train_dataset = _EpochRecordingFakeDataset(n=8)
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_train_dataset",
        lambda config: first_train_dataset,
    )
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_val_dataset",
        lambda config: _FakeStreamingDataset(n=8),
    )

    run_training(
        TrainingDeps(
            config=_config(1),
            frozen_encoder_client=_FakeHfClient(),
            device=torch.device("cpu"),
        )
    )

    assert first_train_dataset.epochs_seen == [0]
    mid_epoch_state = checkpoint.load_checkpoint(
        checkpoint_dir / "checkpoint_step00000001.pt"
    )
    assert mid_epoch_state["epoch"] == 0
    assert mid_epoch_state["dataloader"]  # mid-epoch: a real, resumable stream position

    first_run_path = checkpoint.find_latest_checkpoint(checkpoint_dir)
    assert (
        first_run_path is not None
    )  # run 1 above must have saved at least one checkpoint
    first_state = checkpoint.load_checkpoint(first_run_path)
    assert first_state["global_step"] == 2
    assert (
        first_state["epoch"] == 1
    )  # epoch 0 is DONE -- resume must start at 1, not re-run 0
    assert (
        first_state["dataloader"] is None
    )  # the epoch's iterator is exhausted; nothing to restore
    saved_projector_weight = first_state["projector"]["net.0.weight"].clone()

    # Snapshot the live projector at the exact moment the resume path hands
    # its (encoder + projector) optimizer state back: if the projector were
    # not checkpointed/restored, this would be freshly random instead.
    built_projectors: list[SimCLRProjector] = []
    real_build_projector = contrastive_pretrain.train.build_projector
    real_restore = contrastive_pretrain.train.restore_optimizer_and_scheduler
    projector_at_restore: list[torch.Tensor] = []

    def _spy_build_projector(*args, **kwargs):
        projector = real_build_projector(*args, **kwargs)
        assert isinstance(projector, SimCLRProjector)  # narrows for .net[0] below
        built_projectors.append(projector)
        return projector

    def _spy_restore(optimizer, scheduler, state):
        first_linear = built_projectors[-1].net[0]
        assert isinstance(
            first_linear, nn.Linear
        )  # Sequential.__getitem__ returns Sequential | Module
        projector_at_restore.append(first_linear.weight.detach().clone())
        return real_restore(optimizer, scheduler, state)

    monkeypatch.setattr(
        "contrastive_pretrain.train.build_projector", _spy_build_projector
    )
    monkeypatch.setattr(
        "contrastive_pretrain.train.restore_optimizer_and_scheduler", _spy_restore
    )

    second_train_dataset = _EpochRecordingFakeDataset(n=8)
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_train_dataset",
        lambda config: second_train_dataset,
    )

    run_training(
        TrainingDeps(
            config=_config(2),
            frozen_encoder_client=_FakeHfClient(),
            device=torch.device("cpu"),
        )
    )

    # The resume branch really ran, and it did not re-enter the completed epoch 0.
    assert len(projector_at_restore) == 1
    assert second_train_dataset.epochs_seen == [1]

    # The resumed epoch trained real steps -- a restored-exhausted dataloader
    # would leave global_step pinned at the first run's 2.
    second_run_path = checkpoint.find_latest_checkpoint(checkpoint_dir)
    assert (
        second_run_path is not None
    )  # run 2 above must have saved at least one checkpoint
    second_state = checkpoint.load_checkpoint(second_run_path)
    assert second_state["global_step"] == 4
    assert second_state["epoch"] == 2

    # Continuity: the projection head resumed from the checkpointed weights,
    # not from a fresh random init.
    assert torch.equal(projector_at_restore[0], saved_projector_weight)


def _config_for_resume(
    checkpoint_dir, local_cache_dir: str | None = None
) -> TrainingConfig:
    return TrainingConfig(
        pretrained=False,
        batch_size=4,
        num_workers=0,
        max_epochs=1,
        checkpoint_interval_steps=1,
        network_volume_checkpoint_dir=str(checkpoint_dir),
        local_cache_dir=local_cache_dir,
    )


def _write_mid_epoch_checkpoint(
    tmp_path, monkeypatch, checkpoint_local_cache_dir: str | None
):
    """Runs one short training run to produce a REAL mid-epoch checkpoint
    (checkpoint_interval_steps=1 makes step 1 a genuine periodic save that
    carries live dataloader state), drops the epoch-boundary checkpoint that
    follows it (that one deliberately stores dataloader=None, so
    find_latest_checkpoint would otherwise pick a checkpoint that never
    reaches the load_state_dict call at all), and rewrites the surviving
    checkpoint's recorded data source. Returns the checkpoint dir."""
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_train_dataset",
        lambda config: _FakeStreamingDataset(n=8),
    )
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_val_dataset",
        lambda config: _FakeStreamingDataset(n=8),
    )
    run_training(
        TrainingDeps(
            config=_config_for_resume(checkpoint_dir),
            frozen_encoder_client=_FakeHfClient(),
            device=torch.device("cpu"),
        )
    )

    mid_epoch_path = checkpoint_dir / "checkpoint_step00000001.pt"
    for path in checkpoint_dir.glob("checkpoint_step*.pt"):
        if path != mid_epoch_path:
            path.unlink()

    state = checkpoint.load_checkpoint(mid_epoch_path)
    assert state["dataloader"]  # precondition: there IS state to restore
    state["local_cache_dir"] = checkpoint_local_cache_dir
    checkpoint.save_checkpoint(mid_epoch_path, state)
    return checkpoint_dir


def _spy_on_dataloader_load_state_dict(monkeypatch) -> list[dict]:
    """Wraps train's build_dataloader so the returned StatefulDataLoader
    records every load_state_dict call -- same spy-around-the-real-callable
    pattern the projector-resume test uses for build_projector."""
    calls: list[dict] = []
    real_build_dataloader = contrastive_pretrain.train.build_dataloader

    def _spy(*args, **kwargs):
        loader = real_build_dataloader(*args, **kwargs)
        real_load_state_dict = loader.load_state_dict

        def _recording_load_state_dict(state_dict):
            calls.append(state_dict)
            return real_load_state_dict(state_dict)

        loader.load_state_dict = _recording_load_state_dict  # type: ignore[method-assign]
        return loader

    monkeypatch.setattr("contrastive_pretrain.train.build_dataloader", _spy)
    return calls


@pytest.mark.slow
def test_run_training_skips_dataloader_state_when_local_cache_dir_changed(
    tmp_path, monkeypatch
) -> None:
    """Flipping local_cache_dir on/off adds or drops build_train_dataset's
    pre-shuffle resize-map stage, which changes the nesting of the underlying
    datasets.IterableDataset state dict. StatefulDataLoader.load_state_dict
    accepts the mismatched shape without complaint and then dies with
    KeyError: 'examples_iterable' on the FIRST batch -- i.e. hours into the
    run, after a full cache build. The resume must therefore refuse state
    saved under a different data source rather than restoring it."""
    checkpoint_dir = _write_mid_epoch_checkpoint(
        tmp_path, monkeypatch, checkpoint_local_cache_dir="/workspace/old-cache"
    )
    load_state_dict_calls = _spy_on_dataloader_load_state_dict(monkeypatch)

    run_training(
        TrainingDeps(
            config=_config_for_resume(checkpoint_dir, local_cache_dir=None),
            frozen_encoder_client=_FakeHfClient(),
            device=torch.device("cpu"),
        )
    )

    assert load_state_dict_calls == []


@pytest.mark.slow
def test_run_training_restores_dataloader_state_when_local_cache_dir_matches(
    tmp_path, monkeypatch
) -> None:
    """Mirror of the test above: the guard must not be so eager that it
    throws away legitimately resumable state on a normal same-config
    resume."""
    checkpoint_dir = _write_mid_epoch_checkpoint(
        tmp_path, monkeypatch, checkpoint_local_cache_dir="/workspace/cache"
    )
    load_state_dict_calls = _spy_on_dataloader_load_state_dict(monkeypatch)

    run_training(
        TrainingDeps(
            config=_config_for_resume(
                checkpoint_dir, local_cache_dir="/workspace/cache"
            ),
            frozen_encoder_client=_FakeHfClient(),
            device=torch.device("cpu"),
        )
    )

    assert len(load_state_dict_calls) == 1


@pytest.mark.slow
def test_run_training_skips_publish_when_val_loss_does_not_improve(
    tmp_path, monkeypatch
) -> None:
    """Regression test for the `val_loss < best_val_loss` gate: prior
    coverage always started from best_val_loss=inf and ran <=2 epochs, so
    the gate was always true and never exercised the skip path. A flipped
    comparison would publish a worse model as "best" on every epoch,
    undetected. compute_val_loss is monkeypatched to a controlled
    improve-then-regress sequence -- independent of the fake dataset's
    random per-epoch content -- so the outcome is deterministic rather than
    incidental."""
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_train_dataset",
        lambda config: _FakeStreamingDataset(n=4),
    )
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_val_dataset",
        lambda config: _FakeStreamingDataset(n=4),
    )
    val_losses = iter([1.0, 2.0])  # epoch 0: inf -> 1.0 improves; epoch 1: 1.0 -> 2.0 regresses
    monkeypatch.setattr(
        "contrastive_pretrain.train.compute_val_loss",
        lambda *args, **kwargs: next(val_losses),
    )

    client = _FakeHfClient()
    config = TrainingConfig(
        pretrained=False,
        batch_size=2,
        num_workers=0,
        max_epochs=2,
        checkpoint_interval_steps=1000,
        network_volume_checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    run_training(
        TrainingDeps(config=config, frozen_encoder_client=client, device=torch.device("cpu"))
    )

    # Exactly one publish happened (epoch 0's improvement) -- epoch 1's
    # regression must not trigger a second push.
    assert client.upload_calls.count("model.safetensors") == 1


class _SpyWandbRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def finish(self) -> None:
        pass


@pytest.mark.slow
def test_run_training_logs_contact_sheet_exactly_once_per_epoch(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_train_dataset",
        lambda config: _FakeStreamingDataset(n=8),
    )
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_val_dataset",
        lambda config: _FakeStreamingDataset(n=4),
    )

    wandb_run = _SpyWandbRun()
    config = TrainingConfig(
        pretrained=False,
        batch_size=2,
        num_workers=0,
        max_epochs=2,
        checkpoint_interval_steps=1000,
        network_volume_checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    run_training(
        TrainingDeps(
            config=config,
            frozen_encoder_client=_FakeHfClient(),
            wandb_run=wandb_run,
            device=torch.device("cpu"),
        )
    )

    contact_sheet_logs = [m for m in wandb_run.logged if "augmentation_contact_sheet" in m]
    assert len(contact_sheet_logs) == config.max_epochs  # once per epoch, not once per step


@pytest.mark.slow
def test_data_wait_metric_excludes_epoch_boundary_overhead(tmp_path, monkeypatch) -> None:
    """Regression test for the prev_step_end reset at the epoch boundary:
    without it, the next epoch's first data_wait_s would include
    validation + Hub-push + checkpoint-save time, misreporting it as a
    streaming stall. Freezes time.monotonic() except for one deliberate
    500s jump injected inside compute_val_loss (standing in for slow
    epoch-boundary work), so the next epoch's first data_wait_s is
    unambiguous: near-zero if the reset fired between the jump and that
    step, ~500s if it didn't."""
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_train_dataset",
        lambda config: _FakeStreamingDataset(n=4),
    )
    monkeypatch.setattr(
        "contrastive_pretrain.train.build_val_dataset",
        lambda config: _FakeStreamingDataset(n=4),
    )

    clock = [0.0]
    monkeypatch.setattr(contrastive_pretrain.train.time, "monotonic", lambda: clock[0])

    real_compute_val_loss = contrastive_pretrain.train.compute_val_loss

    def _slow_compute_val_loss(*args, **kwargs):
        clock[0] += 500.0
        return real_compute_val_loss(*args, **kwargs)

    monkeypatch.setattr("contrastive_pretrain.train.compute_val_loss", _slow_compute_val_loss)

    logged_data_wait: list[dict] = []
    real_info = contrastive_pretrain.train.logger.info

    def _spy_info(event, *args, **kwargs):
        # rule 4 exception: this filters which of the spy's captured calls to
        # keep, not test-case branching.
        if event == "data_wait":
            logged_data_wait.append(kwargs.get("extra", {}))
        return real_info(event, *args, **kwargs)

    monkeypatch.setattr(contrastive_pretrain.train.logger, "info", _spy_info)

    config = TrainingConfig(
        pretrained=False,
        batch_size=2,
        num_workers=0,
        max_epochs=2,
        checkpoint_interval_steps=1000,
        network_volume_checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    run_training(
        TrainingDeps(
            config=config, frozen_encoder_client=_FakeHfClient(), device=torch.device("cpu")
        )
    )

    # n=4 rows / batch_size=2 is 2 steps/epoch -- index 2 is epoch 1's first
    # step, the one immediately after the injected 500s validation jump.
    assert len(logged_data_wait) == 4
    assert logged_data_wait[2]["data_wait_s"] < 1.0


@pytest.mark.slow
def test_run_training_completes_a_few_steps_without_nan(tmp_path) -> None:
    """A real, short smoke run (real streaming data, real model, a
    handful of steps) -- verifies the whole pipeline wires together and
    produces a finite, non-exploding loss before trusting it with a real
    paid A100 run. CPU-capable but slow; run on the target GPU when
    validating a full-scale config."""
    config = TrainingConfig(
        batch_size=4,
        num_workers=0,
        max_epochs=1,
        checkpoint_interval_steps=2,
        network_volume_checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    deps = TrainingDeps(
        config=config, frozen_encoder_client=_FakeHfClient(), device=torch.device("cpu")
    )

    # A 1-epoch run over the full streaming dataset is too slow for a
    # smoke test; monkeypatch max_epochs's effective loop bound isn't
    # exposed, so this test is intended to be run with a config small
    # enough to complete in seconds -- see the note above about running
    # it on the target GPU for full validation instead of relying on
    # this to prove end-to-end throughput.
    run_training(deps)

    checkpoints = list((tmp_path / "checkpoints").glob("checkpoint_step*.pt"))
    assert checkpoints
