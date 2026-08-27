import pytest
import torch

from sequence_model.telemetry import (
    attention_distance_mass,
    attention_logit_max,
    residual_norm,
)


def test_distance_mass_puts_all_weight_in_the_adjacent_bucket() -> None:
    """Weight only on distance 1 (each query attends to the token right
    before it) must land entirely in the "1" bucket."""
    weights = torch.zeros(1, 1, 4, 4)
    weights[0, 0, 1, 0] = 1.0
    weights[0, 0, 2, 1] = 1.0
    weights[0, 0, 3, 2] = 1.0

    mass = attention_distance_mass(weights)

    assert mass["1"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("distance", "expected_bucket"),
    [(1, "1"), (3, "2-8"), (20, "9-64"), (100, "65-256")],
)
def test_distance_mass_assigns_each_distance_to_its_bucket(
    distance: int, expected_bucket: str
) -> None:
    weights = torch.zeros(1, 1, 300, 300)
    weights[0, 0, 299, 299 - distance] = 1.0

    mass = attention_distance_mass(weights)

    assert mass[expected_bucket] == pytest.approx(1.0, abs=1e-6)


def test_distance_mass_buckets_sum_to_one_for_normalized_weights() -> None:
    torch.manual_seed(0)
    weights = torch.softmax(torch.randn(2, 2, 16, 16), dim=-1)

    mass = attention_distance_mass(weights)

    assert sum(mass.values()) == pytest.approx(1.0, abs=1e-5)


def test_attention_logit_max_ignores_masked_positions() -> None:
    """A huge logit at a masked position must not be reported -- it is
    never attended to, so counting it would raise false alarms every
    step."""
    q = torch.zeros(1, 1, 2, 4)
    k = torch.zeros(1, 1, 2, 4)
    q[0, 0, 0] = torch.tensor([100.0, 0.0, 0.0, 0.0])
    k[0, 0, 1] = torch.tensor([100.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([[[[True, False], [True, True]]]])

    assert attention_logit_max(q, k, mask) == pytest.approx(0.0, abs=1e-4)


def test_attention_logit_max_reports_the_scaled_score() -> None:
    """head_dim=4, so the scale is 1/sqrt(4) = 0.5. A raw dot product of
    8 is reported as 4."""
    q = torch.zeros(1, 1, 1, 4)
    k = torch.zeros(1, 1, 1, 4)
    q[0, 0, 0, 0] = 4.0
    k[0, 0, 0, 0] = 2.0
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)

    assert attention_logit_max(q, k, mask) == pytest.approx(4.0, abs=1e-4)


def test_residual_norm_is_the_mean_per_token_l2_norm() -> None:
    """Two tokens of norm 3 and 4 average to 3.5."""
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])

    assert residual_norm(hidden) == pytest.approx(3.5, abs=1e-6)
