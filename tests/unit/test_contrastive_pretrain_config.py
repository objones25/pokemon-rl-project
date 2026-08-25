from contrastive_pretrain.config import TrainingConfig, load_config


def test_training_config_defaults_match_spec() -> None:
    config = TrainingConfig()

    assert config.dataset_repo_id == "objones25/pokemon-frames"
    assert config.frozen_encoder_repo_id == "objones25/pokemon-contrastive-encoder"
    assert config.val_video_ids == ("D1SrSFZrV7A", "YW29l3jJXr4")
    assert config.batch_size == 512
    assert config.learning_rate == 3e-4
    assert config.warmup_steps == 1000
    assert config.weight_decay == 1e-6
    assert config.temperature == 0.1
    assert config.max_epochs == 100
    assert config.checkpoint_interval_steps == 1000
    assert config.shuffle_buffer_size == 10_000


def test_load_config_applies_yaml_overrides(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("batch_size: 8\nlearning_rate: 0.001\n")

    config = load_config(path)

    assert config.batch_size == 8
    assert config.learning_rate == 0.001
    assert config.max_epochs == 100  # untouched fields keep their default


def test_load_config_converts_val_video_ids_to_tuple(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("val_video_ids: ['a', 'b', 'c']\n")

    config = load_config(path)

    assert config.val_video_ids == ("a", "b", "c")


def test_load_config_with_empty_file_returns_defaults(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("")

    config = load_config(path)

    assert config == TrainingConfig()


def test_real_config_file_loads_without_error() -> None:
    config = load_config("configs/contrastive_pretrain.yaml")
    assert config.batch_size == 512
