import pytest

from hf_storage.retry import (
    is_rate_limited,
    rate_limit_aware_backoff,
    retry_with_backoff,
)


def test_retry_with_backoff_succeeds_after_transient_failures() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky() -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")

    retry_with_backoff(flaky, max_retries=3, base_delay=1.0, sleep_func=sleeps.append)

    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_retry_with_backoff_raises_after_exhausting_retries() -> None:
    def always_fails() -> None:
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        retry_with_backoff(always_fails, max_retries=2, base_delay=1.0, sleep_func=lambda _: None)


def test_retry_with_backoff_uses_injected_backoff_function() -> None:
    """backoff_seconds overrides the default exponential schedule entirely --
    this is what lets a caller (e.g. HfUploader) special-case rate-limit
    errors without retry_with_backoff itself knowing about HTTP status codes."""
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky() -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient")

    retry_with_backoff(
        flaky,
        max_retries=3,
        base_delay=1.0,
        sleep_func=sleeps.append,
        backoff_seconds=lambda attempt, exc: 42.0,
    )

    assert sleeps == [42.0, 42.0]


def test_is_rate_limited_matches_the_observed_hf_429_message() -> None:
    exc = RuntimeError(
        "429 Too Many Requests for url: "
        "https://huggingface.co/api/datasets/me/repo/commit/main.\n"
        "You have exceeded the rate limit for repository commits (256 per hour)."
    )
    assert is_rate_limited(exc) is True


def test_is_rate_limited_ignores_unrelated_errors() -> None:
    assert is_rate_limited(RuntimeError("connection reset by peer")) is False


def test_rate_limit_aware_backoff_uses_long_delay_for_rate_limit_errors() -> None:
    backoff = rate_limit_aware_backoff(base_delay=1.0, rate_limit_delay=120.0)

    delay = backoff(1, RuntimeError("429 Too Many Requests"))

    assert delay == pytest.approx(120.0)


def test_rate_limit_aware_backoff_falls_back_to_exponential_for_other_errors() -> None:
    backoff = rate_limit_aware_backoff(base_delay=1.0, rate_limit_delay=120.0)

    assert backoff(1, RuntimeError("connection reset")) == pytest.approx(1.0)
    assert backoff(2, RuntimeError("connection reset")) == pytest.approx(2.0)


def test_retry_with_backoff_returns_the_wrapped_callables_result() -> None:
    def make_greeting() -> str:
        return "hello"

    result = retry_with_backoff(make_greeting, max_retries=1, base_delay=1.0, sleep_func=lambda _: None)

    assert result == "hello"
