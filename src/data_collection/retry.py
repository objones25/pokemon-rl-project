"""Generic retry-with-backoff, shared by pipeline.py (video-level retry) and
hf_uploader.py (per-call retry on individual uploads/manifest saves) -- kept
dependency-free (no huggingface_hub imports) so it stays testable with plain
fakes, matching the rest of this codebase's DI conventions.
"""

from __future__ import annotations

from collections.abc import Callable

BackoffFunc = Callable[[int, Exception], float]


def exponential_backoff(base_delay: float) -> BackoffFunc:
    def _backoff(attempt: int, exc: Exception) -> float:
        return base_delay * (2 ** (attempt - 1))

    return _backoff


def is_rate_limited(exc: Exception) -> bool:
    """String-match on the exception text rather than a specific HTTP
    client's error type -- keeps this module free of any dependency on
    huggingface_hub, and matches the actual observed error text ('429 Too
    Many Requests... exceeded the rate limit for repository commits')."""
    text = str(exc).lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def rate_limit_aware_backoff(base_delay: float, rate_limit_delay: float) -> BackoffFunc:
    """A rate-limit error (e.g. HF Hub's commit quota) needs a wait measured
    in minutes, not the seconds-scale exponential schedule used for
    ordinary transient failures -- retrying sooner just re-hits the same
    quota wall."""
    fallback = exponential_backoff(base_delay)

    def _backoff(attempt: int, exc: Exception) -> float:
        if is_rate_limited(exc):
            return rate_limit_delay
        return fallback(attempt, exc)

    return _backoff


def retry_with_backoff(
    func: Callable[[], None],
    max_retries: int,
    base_delay: float,
    sleep_func: Callable[[float], None],
    backoff_seconds: BackoffFunc | None = None,
) -> None:
    backoff = backoff_seconds or exponential_backoff(base_delay)
    attempt = 0
    while True:
        try:
            func()
            return
        except Exception as exc:
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_func(backoff(attempt, exc))
