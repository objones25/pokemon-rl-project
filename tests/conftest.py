"""Pytest configuration, loaded before any test module.

Disables torch.compile's on-disk FX graph cache (TORCHINDUCTOR_FX_GRAPH_CACHE
-- a real, torch-documented env var, see torch._inductor.config.fx_graph_cache)
for the test suite only. Compilation itself is unaffected; only the pickling
step that hashes/caches the compiled graph to disk is skipped.

This exists because that pickling step walks global interpreter state and, in
this test suite specifically, ends up probing yt_dlp's lazy `Cryptodome`
compat-shim module (imported transitively by data_collection's tests, which
run in the same pytest session as contrastive_pretrain's torch.compile-using
tests) with an attribute name it doesn't handle, raising
`ModuleNotFoundError: No module named 'no_Cryptodome'`. Set as an environment
variable (not a torch._inductor.config.fx_graph_cache = False assignment) so
it takes effect before torch._inductor.config is first imported by any test
module, regardless of import order.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "0")
