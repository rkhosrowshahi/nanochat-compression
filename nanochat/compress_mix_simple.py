"""
The mix of compress_mix.py split in two (scripts/compress_search_mix_simple.py, then scripts/compress_storage.py), after what
its run showed: sharing changes the stored size and the loss, never the time or the GPU memory, and its front left pruning,
clipping and the merge off in 97-98% of its solutions.

    search        one format per matrix group among the deployed ones (bf16, int4), and the MLP width, for delta_loss, the time
                  of a turn and the deployed memory: MixCompressor(sharing=False), which needs nothing from this file
    storage step  optional, on a finished front: every matrix that stays in bf16 gets one shared grid of k levels. Only k varies,
                  so a solution has as many storage options as there are k. It runs as before, so only the loss and the
                  stored size change, and each solution keeps its time and memory.

The storage step only shares ALL of a solution's bf16 matrices at once: it does not search which ones.
"""

from dataclasses import replace

from nanochat.compress import Candidate
from nanochat.compress_mix import SHARE_K, SHARED

STORAGE_K = SHARE_K


def has_bf16(cfg):
    return cfg.lm_head == "bf16" or any(f == "bf16" for band in cfg.fmt for f in band)


def shared_storage(cfg, k):
    """cfg with its bf16 matrices stored on a shared grid of k levels: no pruning, clipping or merge."""
    share = lambda f: SHARED if f == "bf16" else f
    return replace(cfg, fmt=tuple(tuple(map(share, band)) for band in cfg.fmt), lm_head=share(cfg.lm_head), embed=share(cfg.embed),
                   share=Candidate(k, 1.0, 0.0, 0.0, 1.0))


def storage_front(rows):
    """The rows no other row beats on delta_loss and size_mb, by increasing delta_loss."""
    front = []
    for r in sorted(rows, key=lambda r: (r["delta_loss"], r["size_mb"])):
        if not front or r["size_mb"] < front[-1]["size_mb"]:
            front.append(r)
    return front
