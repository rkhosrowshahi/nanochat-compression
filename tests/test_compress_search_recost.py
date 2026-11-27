"""Cached evaluations get their costs from the current latency table (scripts/compress_search_common.recost)."""

from scripts.compress_search_common import recost


class FakeCompressor:
    """cost = per-format time from a table, as the real compressors' cost is: a pure function of the config and the table."""

    def __init__(self, us):
        self.us = us

    def config(self, genes):
        return tuple(genes)

    def cost(self, cfg):
        t = sum(self.us[g] for g in cfg) / 1000
        return {"latency_ms": t, "tok_s": 1000 / t, "memory_mb": 10.0 * len(cfg), "base": "not a statistic"}


def test_costs_follow_the_current_table():
    slow, fast = FakeCompressor([100.0, 50.0]), FakeCompressor([40.0, 20.0])  # two GPUs
    genes = [0, 1, 1]
    cached = {"kld": 0.25, **{k: v for k, v in slow.cost(slow.config(genes)).items() if k != "base"}}
    r = recost(cached, genes, fast)
    assert r["kld"] == 0.25  # quality stays
    assert r["latency_ms"] == fast.cost(genes)["latency_ms"] and r["tok_s"] == fast.cost(genes)["tok_s"]
    assert r["memory_mb"] == cached["memory_mb"]
    assert "base" not in r  # only the statistics the row has


def test_same_table_changes_nothing_and_kept_stats_stay():
    c = FakeCompressor([100.0, 50.0])
    genes = [1, 0]
    cached = {"kld": 0.1, "latency_ms": c.cost(genes)["latency_ms"], "tok_s": c.cost(genes)["tok_s"], "memory_mb": 7.0}
    assert recost(dict(cached, memory_mb=20.0), genes, c) == dict(cached, memory_mb=20.0)
    assert recost(cached, genes, c, kept={"memory_mb"})["memory_mb"] == 7.0  # measured by apply(): not overwritten
