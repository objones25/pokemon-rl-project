"""Pytest configuration and fixtures."""

import os


def pytest_configure(config):
    """Set environment variables before test collection and module imports.

    Disables torch.compile caching to work around a known issue where
    torch.compile's inductor backend tries to serialize the compiled graph,
    which triggers yt_dlp's compatibility module loading, which fails on
    missing 'no_Cryptodome' module. This is a torch/yt_dlp compatibility
    issue, not a code bug. Disabling the compile counter means torch.compile
    still works (no caching), just doesn't try to serialize/cache the
    compiled functions.
    """
    os.environ["TORCH_COMPILE_DISABLE_BY_COUNTER"] = "1"
