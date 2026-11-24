"""
Measure the decode speed of a model as deployed on this GPU: TTFT, TPOT and tok/s (nanochat/compress_tpot.py), the
deployed weight memory, and the TPOT the latency table predicts (nanochat/compress_hw.py) for the same configuration.

Configurations: the uncompressed model (bf16) and uniform weight-only int4 at each group size, built with
HWCompressor.materialize on PyTorch's tinygemm kernel. As in GPTQ and AWQ, the int4 baselines quantize every matrix
of the transformer blocks and keep lm_head and the embeddings in bf16; --quantize-head puts lm_head in int4 as well.
The int4 rounding uses plain squared-error clipping (no calibration data): the time depends on the format and the
shapes only, not on how the weights were rounded.

The prediction is the latency table's decode step at the mean context of the measured generation
(prompt + gen/2), measured once per GPU, batch and context and cached in $COMPRESS_CACHE.

    python -m scripts.compress_tpot_measure --model Qwen/Qwen2.5-Coder-1.5B-Instruct
    python -m scripts.compress_tpot_measure --model Qwen/Qwen2.5-0.5B-Instruct --batch 1,8

Decode attention: --attention grouped (default) reads the KV cache in place, as serving runtimes and the latency table
do; --attention model keeps transformers' sdpa path, which copies the cache to every query head on each step.

Outputs in --out-dir (default compress_runs/tpot/<model>): tpot_<attention>.json (settings and every row) and .csv.
"""

import argparse
import collections
import csv
import gc
import json
import os

import torch

from nanochat import compress_eval, compress_hw, compress_models, compress_tpot
from nanochat.compress_hw import FORMATS, ROLES, Config, HWCompressor, Layout

parser = argparse.ArgumentParser(description="TTFT, TPOT and tok/s of uniform configurations, measured and predicted")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16"], help="the int4 kernel takes bf16 activations")
parser.add_argument("--formats", type=str, default=",".join(FORMATS), help=f"uniform configurations, from {','.join(FORMATS)}")
parser.add_argument("--quantize-head", action="store_true", help="lm_head in the configuration's format too (default: bf16)")
parser.add_argument("--batch", type=str, default="1", help="comma-separated batch sizes (sequences decoded together)")
parser.add_argument("--prompt-tokens", type=int, default=512)
parser.add_argument("--gen-tokens", type=int, default=256)
parser.add_argument("--repeats", type=int, default=5, help="timed generations per configuration (median)")
parser.add_argument("--check-tokens", type=int, default=32, help="decode steps compared between the CUDA graph and eager")
parser.add_argument("--attention", type=str, default="grouped", choices=["grouped", "model"],
                    help="decode attention: grouped reads the KV cache in place (nanochat/compress_tpot.py); model: the model's own")
parser.add_argument("--no-predict", action="store_true", help="skip the latency table's prediction")
parser.add_argument("--out-dir", type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda")
gpu = torch.cuda.get_device_name(device)
formats = [f for f in args.formats.split(",") if f]
assert set(formats) <= set(FORMATS), f"--formats must be among {FORMATS}"
batches = [int(b) for b in args.batch.split(",")]
context = args.prompt_tokens + args.gen_tokens // 2  # mean context of the decode steps
out_dir = args.out_dir or os.path.join("compress_runs", "tpot", args.model.replace("/", "_").replace(":", "_"))
os.makedirs(out_dir, exist_ok=True)
GiB = 1 << 30


def load():
    model, _, _ = compress_models.load(args.model, args.dtype, device)
    assert not hasattr(model, "gpt"), "nanochat checkpoints have no KV-cache interface here: use a HuggingFace model"
    return model, Layout(model)


def compressor(model, layout, table):
    """An HWCompressor that only builds and costs uniform configurations: no statistics, ranking or pruning."""
    act_sq = {n: torch.ones(layout.module(n).in_features) for n in layout.linear_names()}  # plain clipping error
    order = {k: [(i, 0.0) for i in range(layout.n_layer)] for k in ("attn", "mlp")}
    if table is None:  # --no-predict: no kernel availability known, so every shape takes every format (Int4Linear asserts)
        table = {"gemm_us": collections.defaultdict(lambda: dict.fromkeys(FORMATS, 0.0))}
    return HWCompressor(model, layout, table, act_sq, order, formats=FORMATS, bands=1, mlp_keep=(1.0,), max_skip=0,
                        compress_embeddings=False, offload="yes")


def uniform(fmt):
    head = fmt if args.quantize_head else "bf16"
    return Config(tuple((fmt,) for _ in ROLES), head, "bf16", (1.0,), 0, 0)


def latency_table(model, layout, batch):
    """The decode latency table for this GPU, batch and context: cached, or measured now."""
    settings = {"version": compress_hw.TABLE_VERSION, "kind": "decode", "gpu": gpu, "torch": str(torch.__version__),
                "model": args.model, "batch": batch, "context": context, "formats": list(FORMATS), "mlp_keep": [1.0]}
    path = compress_eval.cache_path("latency", settings)
    if os.path.exists(path):
        return torch.load(path)
    print(f"Measuring the latency table on {gpu}: batch {batch}, context {context}", flush=True)
    table = {"settings": settings, **compress_hw.build_latency_table(model, layout, list(FORMATS), [1.0], batch, context,
                                                                     log=lambda m: print(m, flush=True))}
    compress_eval.save_atomic(table, path)
    return table


tables = {}
if not args.no_predict:
    model, layout = load()
    tables = {b: latency_table(model, layout, b) for b in batches}
    del model, layout
    gc.collect()
    torch.cuda.empty_cache()

rows = []
compress_hw.warm_up(device)
for fmt in formats:
    model, layout = load()
    cfg = uniform(fmt)
    comp = compressor(model, layout, tables.get(batches[0]))
    predicted = {}
    for b, table in tables.items():
        comp.table = table
        predicted[b] = comp.cost(cfg)["latency_ms"]
    label = comp.describe(cfg).split(" keep:")[0]
    if fmt != "bf16" or cfg.lm_head != "bf16":
        comp.materialize(cfg)
    del comp
    gc.collect()
    torch.cuda.empty_cache()
    size = compress_tpot.deployed_bytes(model)
    for b in batches:
        r = compress_tpot.decode_timing(model, b, args.prompt_tokens, args.gen_tokens, args.repeats, args.check_tokens,
                                        attention=args.attention)
        rows.append({"configuration": fmt if cfg.lm_head == "bf16" or fmt == "bf16" else f"{fmt} + head",
                     "layout": label, "weights_gib": size["weights_bytes"] / GiB,
                     "embedding_gib": size["embedding_bytes"] / GiB, **r, "peak_gib": r.pop("peak_bytes") / GiB,
                     "predicted_tpot_ms": predicted.get(b)})
        p = predicted.get(b)
        print(f"{rows[-1]['configuration']:>16} batch {b:>3}: weights {rows[-1]['weights_gib']:.3f} GiB, "
              f"TTFT {r['ttft_ms']:7.2f} ms, TPOT {r['tpot_ms']:6.3f} ms ({r['tok_s']:6.1f} tok/s)"
              + (f", predicted {p:6.3f} ms ({100 * (p - r['tpot_ms']) / r['tpot_ms']:+.1f}%)" if p else "")
              + f", graph = eager on {r['graph_matches_eager']} tokens", flush=True)
    del model, layout
    gc.collect()
    torch.cuda.empty_cache()

base = {r["batch"]: r["tpot_ms"] for r in rows if r["configuration"] == "bf16"}
for r in rows:
    r["speedup"] = base[r["batch"]] / r["tpot_ms"] if r["batch"] in base else None

json.dump({"settings": {**vars(args), "gpu": gpu, "torch": str(torch.__version__), "predict_context": context},
           "rows": rows}, open(os.path.join(out_dir, f"tpot_{args.attention}.json"), "w"), indent=2)
fields = ["configuration", "batch", "weights_gib", "embedding_gib", "ttft_ms", "tpot_ms", "tok_s", "tok_s_total",
          "speedup", "predicted_tpot_ms", "e2e_ms", "peak_gib", "graph_matches_eager", "attention", "layout"]
with open(os.path.join(out_dir, f"tpot_{args.attention}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"\n{gpu}, prompt {args.prompt_tokens} tokens, {args.gen_tokens} generated, median of {args.repeats} runs, "
      f"{args.attention} attention")
print(f"| Configuration | Batch | Weights (GiB) | TTFT (ms) | TPOT (ms) | Decode speed (tok/s) | Speedup | "
      f"Predicted TPOT (ms) | Peak GPU memory (GiB) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    sp = f"{r['speedup']:.2f}x" if r["speedup"] else "-"
    pt = f"{r['predicted_tpot_ms']:.2f}" if r["predicted_tpot_ms"] else "-"
    print(f"| {r['configuration']} | {r['batch']} | {r['weights_gib']:.3f} | {r['ttft_ms']:.1f} | {r['tpot_ms']:.2f} | "
          f"{r['tok_s']:.1f} | {sp} | {pt} | {r['peak_gib']:.3f} |")
print(f"Results in {out_dir}")
