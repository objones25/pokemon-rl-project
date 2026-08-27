import torch

from sequence_model.masks import build_chunk_mask


def test_chunk_mask_shape_is_broadcastable_over_heads() -> None:
    abs_pos = torch.arange(6).unsqueeze(0)
    episode_id = torch.zeros(1, 6, dtype=torch.long)

    mask = build_chunk_mask(abs_pos, episode_id, context_len=4)

    assert tuple(mask.shape) == (1, 1, 6, 6)


def test_chunk_mask_is_causal() -> None:
    abs_pos = torch.arange(4).unsqueeze(0)
    episode_id = torch.zeros(1, 4, dtype=torch.long)

    mask = build_chunk_mask(abs_pos, episode_id, context_len=8)

    assert mask[0, 0].tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_chunk_mask_window_limits_each_row_to_context_len_positions() -> None:
    abs_pos = torch.arange(10).unsqueeze(0)
    episode_id = torch.zeros(1, 10, dtype=torch.long)

    mask = build_chunk_mask(abs_pos, episode_id, context_len=4)

    assert mask[0, 0, 9].tolist() == [False] * 6 + [True] * 4


def test_chunk_mask_blocks_attention_across_an_episode_boundary() -> None:
    abs_pos = torch.tensor([[0, 1, 2, 0, 1, 2]])
    episode_id = torch.tensor([[0, 0, 0, 1, 1, 1]])

    mask = build_chunk_mask(abs_pos, episode_id, context_len=8)

    assert mask[0, 0, 3].tolist() == [False, False, False, True, False, False]


def test_chunk_mask_diagonal_is_always_unmasked() -> None:
    """A fully-masked row makes softmax return NaN. The diagonal is the
    guarantee that never happens."""
    abs_pos = torch.tensor([[0, 1, 0, 1]])
    episode_id = torch.tensor([[0, 0, 1, 1]])

    mask = build_chunk_mask(abs_pos, episode_id, context_len=1)

    assert torch.diagonal(mask[0, 0]).tolist() == [True, True, True, True]
