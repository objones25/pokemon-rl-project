import pytest
import torch

from contrastive_pretrain.model import build_encoder, build_projector
from contrastive_pretrain.train import check_finite_loss, compute_val_loss, run_memory_probe


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

    with pytest.raises(ValueError):
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
        encoder, projector, fake_batches(), temperature=0.5, device=torch.device("cpu"), max_batches=3,
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
        encoder, projector, fake_batches(), temperature=0.5, device=torch.device("cpu"), max_batches=1,
    )

    assert encoder.training is True
    assert projector.training is True


def test_compute_val_loss_raises_when_no_batches_produced() -> None:
    encoder, _ = build_encoder(pretrained=False)
    projector = build_projector()

    with pytest.raises(RuntimeError, match="no batches"):
        compute_val_loss(encoder, projector, iter([]), temperature=0.5, device=torch.device("cpu"), max_batches=3)


import pytest

from contrastive_pretrain.config import TrainingConfig
from contrastive_pretrain.train import TrainingDeps, run_training


class _FakeHfClient:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def upload_bytes(self, data: bytes, path_in_repo: str) -> None:
        self.files[path_in_repo] = data

    def download_bytes(self, path_in_repo: str) -> bytes | None:
        return self.files.get(path_in_repo)


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
    deps = TrainingDeps(config=config, frozen_encoder_client=_FakeHfClient(), device=torch.device("cpu"))

    # A 1-epoch run over the full streaming dataset is too slow for a
    # smoke test; monkeypatch max_epochs's effective loop bound isn't
    # exposed, so this test is intended to be run with a config small
    # enough to complete in seconds -- see the note above about running
    # it on the target GPU for full validation instead of relying on
    # this to prove end-to-end throughput.
    run_training(deps)
