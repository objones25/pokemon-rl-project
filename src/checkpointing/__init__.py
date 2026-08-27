"""Checkpoint file I/O shared across sub-projects.

Deliberately knows nothing about *what* a checkpoint contains -- it moves
dicts to and from disk, finds the newest one, and enforces a retention
window. Each sub-project owns its own state schema (see
contrastive_pretrain.checkpoint and sequence_model.checkpoint).
"""
