"""
Which weight formats are worth searching on this GPU? For one model's matrix shapes (only the config is
downloaded, no weights), every format is checked against the dense result and timed at several batch
sizes, then combined into the cost of a whole decode step, of prompt processing and of an agent's turn,
and into how many agents of a given context fit in memory.

Formats (each one is a uniform choice for the whole model; the search later mixes them):
    bf16                  dense baseline
    int4_g32/g64/g128     weight-only int4, per-group scale and zero (tinygemm)
    int8_weight           weight-only int8, per-row scale (tinygemm)
    w8a8, fp8             8-bit weights and activations, the activations converted in eager PyTorch
    w8a8_gemm, fp8_gemm   the same matrix product alone: what a fused kernel could reach at best
    sparse24              bf16 with 2 of every 4 weights removed (cuSPARSELt, Linux wheels only)

A format this PyTorch build or GPU cannot run is reported with the reason, it does not stop the run.
Results are saved after every measurement, so an interrupted run resumes where it stopped. The GPU should
otherwise be idle: another job slows some measurements and not others.

Outputs in <out-dir>/<gpu>__<model>:
    probe.json            every measurement
    probe.md              the tables

Example (run from the repo root; about 15-30 minutes):

    python -m scripts.compress_kernel_probe --model Qwen/Qwen2.5-32B-Instruct
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from nanochat import compress_hw as hw

sys.stdout.reconfigure(encoding="utf-8")

FORMATS = ("bf16", "int4_g32", "int4_g64", "int4_g128", "int8_weight", "w8a8", "w8a8_gemm", "fp8", "fp8_gemm", "sparse24")

parser = argparse.ArgumentParser(description="time the candidate weight formats on this GPU for one model's shapes")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-32B-Instruct", help="HuggingFace model (Llama/Qwen2-style); only its config is used")
parser.add_argument("--formats", type=str, default=",".join(FORMATS))
parser.add_argument("--batches", type=str, default="1,4,16,64", help="sequences decoding together")
parser.add_argument("--prefill-batch", type=int, default=2048, help="tokens processed together during prompt processing")
parser.add_argument("--attention-chunk", type=int, default=256, help="prompt tokens attending together when timing prompt attention (memory grows with it)")
parser.add_argument("--context", type=int, default=32000, help="tokens each agent keeps in the KV cache")
parser.add_argument("--new-tokens", type=int, default=2000, help="tokens processed per turn (the rest of the prompt is cached)")
parser.add_argument("--gen-tokens", type=int, default=300, help="tokens generated per turn")
parser.add_argument("--vram-gb", type=str, default="24,40,80", help="GPU sizes for the agents table")
parser.add_argument("--reserve", type=float, default=0.1, help="fraction of the memory kept free for activations and the runtime")
parser.add_argument("--out-dir", type=str, default="compress_runs/kernel_probe")
args = parser.parse_args()

assert torch.cuda.is_available(), "this script measures a CUDA GPU"
device = torch.device("cuda")
gpu = torch.cuda.get_device_properties(device)
formats = args.formats.split(",")
batches = [int(b) for b in args.batches.split(",")]
vrams = [float(v) for v in args.vram_gb.split(",")]
slug = lambda s: re.sub(r"[^A-Za-z0-9.-]+", "-", s).strip("-")
out_dir = os.path.join(args.out_dir, f"{slug(gpu.name)}__{slug(args.model)}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "probe.json")

config = AutoConfig.from_pretrained(args.model)
with torch.device("meta"):
    layout = hw.Layout(AutoModelForCausalLM.from_config(config))
shapes = Counter()  # (rows, cols, bias) -> how many matrices of the model have it
for name in layout.linear_names():
    m = layout.module(name)
    shapes[(m.out_features, m.in_features, m.bias is not None)] += 1
head = layout.module(layout.lm_head)
head_shape = (head.out_features, head.in_features, head.bias is not None)
embed_weights = sum(layout.module(n).weight.numel() for n in layout.embeds)  # untied input embeddings: memory only
kv_gb = layout.n_layer * layout.kv_bytes() * args.context / 1e9

info = {"gpu": gpu.name, "gpu_memory_gb": gpu.total_memory / 1e9, "torch": torch.__version__, "model": args.model,
        "n_layer": layout.n_layer, "n_head": layout.n_head, "n_kv_head": layout.n_kv_head, "head_dim": layout.head_dim,
        "shapes": [[*s, n] for s, n in sorted(shapes.items())], "context": args.context}
results = json.load(open(out_path)) if os.path.exists(out_path) else {"info": info, "verify": {}, "cells": {}, "attention": {}}
results["info"] = info


def save():
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, out_path)


def nbytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def build(fmt, rows, cols, bias, x):
    """(call, expected_output, bytes of the weights): one random matrix of this shape in this format.
    Results are checked with bf16 rounding in mind, so the operands are realistic in scale, not in values."""
    b = torch.randn(rows, device=device, dtype=torch.bfloat16) * 0.01 if bias else None
    add = (lambda y: y) if b is None else (lambda y: y + b)

    def dense(xd, wd):  # the matrix product with exactly what the kernel holds
        y = xd.float() @ wd.float().T
        return y if b is None else y + b.float()

    if fmt == "bf16":
        w = torch.randn(rows, cols, device=device, dtype=torch.bfloat16) * 0.02
        return lambda: F.linear(x, w, b), lambda: dense(x, w), nbytes(w)
    if fmt.startswith("int4"):
        g = hw.group_size(fmt)
        q = torch.randint(0, 16, (rows, cols), device=device, dtype=torch.uint8)
        scale = torch.rand(rows, cols // g, device=device).bfloat16() * 0.01
        zero = torch.zeros_like(scale)
        packed, scale_zero = hw.int4_pack(q, scale, zero)
        mm = torch.ops.aten._weight_int4pack_mm
        return (lambda: add(mm(x, packed, g, scale_zero)), lambda: dense(x, hw.int4_dequantize(q, scale, zero, g)),
                nbytes(packed, scale_zero))
    if fmt == "int8_weight":
        q = torch.randint(-127, 128, (rows, cols), device=device, dtype=torch.int8)
        s = torch.rand(rows, device=device).bfloat16() * 0.001
        mm = torch.ops.aten._weight_int8pack_mm
        return lambda: add(mm(x, q, s)), lambda: dense(x, q.float() * s.float()[:, None]), nbytes(q, s)
    if fmt.startswith("w8a8"):
        q = torch.randint(-127, 128, (cols, rows), device=device, dtype=torch.int8)  # [K, N]: the kernel takes x @ W
        s = torch.rand(rows, device=device).bfloat16() * 0.001

        def to_int8(xf):
            sx = xf.abs().amax() / 127
            return torch.round(xf / sx).clamp(-127, 127).to(torch.int8), sx

        xq, sx0 = to_int8(x)

        def from_int(y, sx):
            return add(y.to(torch.bfloat16) * (sx * s))

        expected = lambda: dense(xq.float() * sx0.float(), (q.float().T * s.float()[:, None]))
        if fmt == "w8a8_gemm":
            return lambda: torch._int_mm(xq, q), expected, nbytes(q, s)

        def call():
            xi, sx = to_int8(x)
            return from_int(torch._int_mm(xi, q), sx)

        return call, expected, nbytes(q, s)
    if fmt.startswith("fp8"):
        wf = torch.randn(cols, rows, device=device, dtype=torch.bfloat16) * 0.02
        w8 = wf.to(torch.float8_e4m3fn)  # [K, N]: the kernel takes x @ W
        one = torch.tensor(1.0, device=device)

        def to_fp8(xf):
            sx = (xf.abs().amax() / 448).float()  # 448: the largest FP8 value
            return (xf.float() / sx).to(torch.float8_e4m3fn), sx

        x8, sx0 = to_fp8(x)
        mm = lambda xi, sx: torch._scaled_mm(xi, w8, sx, one, out_dtype=torch.bfloat16, use_fast_accum=True)
        expected = lambda: dense(x8.float() * sx0, w8.float().T)
        if fmt == "fp8_gemm":
            return lambda: mm(x8, sx0), expected, nbytes(w8)

        def call():
            xi, sx = to_fp8(x)
            return add(mm(xi, sx))

        return call, expected, nbytes(w8)
    if fmt == "sparse24":
        assert cols % 4 == 0, f"{cols} columns are not a multiple of 4"
        w = torch.randn(rows, cols, device=device, dtype=torch.bfloat16) * 0.02
        keep = torch.rand(rows, cols // 4, 4, device=device).argsort(-1)[..., :2]  # 2 of every 4 in each row
        mask = torch.zeros(rows, cols // 4, 4, device=device, dtype=torch.bool).scatter_(-1, keep, True).view(rows, cols)
        w = w * mask
        compressed, meta = torch._to_sparse_semi_structured(w)
        return (lambda: torch._sparse_semi_structured_linear(x, compressed, meta, bias=b), lambda: dense(x, w),
                nbytes(compressed, meta))
    raise ValueError(fmt)


@torch.no_grad()
def verify(fmt, rows, cols, bias):
    """Relative error of the kernel against the dense product. The _gemm variants return raw kernel output,
    so the full format is checked. The batch is above 16: the int8 product refuses smaller ones."""
    x = torch.randn(64, cols, device=device, dtype=torch.bfloat16)
    call, expected, _ = build(fmt.removesuffix("_gemm"), rows, cols, bias, x)
    y, e = call().float(), expected()
    assert y.shape == e.shape, f"output shape {tuple(y.shape)}, expected {tuple(e.shape)}"
    return ((y - e).abs().max() / e.abs().max()).item()


@torch.no_grad()
def time_cell(fmt, rows, cols, bias, batch):
    """(GPU time in us of one product of `batch` rows, bytes of the weights)."""
    x = torch.randn(batch, cols, device=device, dtype=torch.bfloat16)
    fns, size = [], 0
    for _ in range(hw.n_copies(rows * cols, device)):
        call, _, size = build(fmt, rows, cols, bias, x)
        fns.append(call)
    for _ in range(2):  # the second call has no first-call set-up
        t0 = time.time()
        fns[0]()
        torch.cuda.synchronize()
        once = (time.time() - t0) * 1e6
    if once > 2e5:  # a graph replays the product hundreds of times: too slow for that, launch overhead is negligible here
        return once, size
    return hw.graph_time_us(fns), size


@torch.no_grad()
def time_prompt_attention(chunk):
    """GPU time in us per token, one layer, of attention during prompt processing. All tokens of a chunk read
    the same cache together, unlike decode, where every sequence reads its own cache for its one token."""
    d, n_kv = layout.head_dim, layout.n_kv_head
    q = torch.randn(1, n_kv, layout.n_head // n_kv * chunk, d, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, n_kv, args.context, d, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, n_kv, args.context, d, device=device, dtype=torch.bfloat16)

    def call():
        a = torch.softmax((q @ k.transpose(-1, -2)).float() * d ** -0.5, dim=-1).to(q.dtype)
        return a @ v

    return hw.graph_time_us([call]) / chunk


def report_error(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:140]}" if str(e) else type(e).__name__


# -----------------------------------------------------------------------------
# measurements

free, total = torch.cuda.mem_get_info(device)
print(f"{gpu.name}, {total / 1e9:.0f} GB ({(total - free) / 1e9:.1f} GB in use), torch {torch.__version__}", flush=True)
if total - free > 2e9:
    print("WARNING: another process is using this GPU: the timings will be unreliable", flush=True)
print(f"{args.model}: {layout.n_layer} layers, {dict(shapes)}", flush=True)

smallest = min(shapes, key=lambda s: s[0] * s[1])
usable = []
for fmt in formats:
    if fmt not in results["verify"]:
        try:
            results["verify"][fmt] = {"ok": True, "error": verify(fmt, *smallest)}
        except Exception as e:
            results["verify"][fmt] = {"ok": False, "reason": report_error(e)}
        torch.cuda.empty_cache()
        save()
    v = results["verify"][fmt]
    if v["ok"] and v["error"] > 0.1:
        v.update(ok=False, reason=f"output differs from the dense product by {v['error']:.1%}")
    print(f"  {fmt:>12}: " + (f"works, error {v['error']:.2e}" if v["ok"] else f"NOT AVAILABLE ({v['reason']})"), flush=True)
    if v["ok"]:
        usable.append(fmt)

hw.warm_up(device)
n_cells = len(usable) * len(shapes) * (len(batches) + 1)
done = 0
for fmt in usable:
    for shape in sorted(shapes):
        for batch in sorted({*batches, args.prefill_batch}):
            key = f"{fmt}|{shape[0]}x{shape[1]}{'+bias' if shape[2] else ''}|{batch}"
            done += 1
            if key in results["cells"]:
                continue
            try:
                us, size = time_cell(fmt, *shape, batch)
                results["cells"][key] = {"us": us, "bytes": size}
                print(f"[{done}/{n_cells}] {key}: {us:.1f} us, {us / batch:.2f} us per token", flush=True)
            except Exception as e:
                results["cells"][key] = {"error": report_error(e)}
                print(f"[{done}/{n_cells}] {key}: {report_error(e)}", flush=True)
            torch.cuda.empty_cache()
            save()

for batch in sorted({*batches}):
    key = str(batch)
    if key not in results["attention"]:
        entry = {}
        for impl in hw.ATTENTION:
            try:
                entry[impl] = hw.time_attention(batch, layout.n_head, layout.n_kv_head, layout.head_dim, args.context, device, impl)
            except Exception as e:
                entry[impl] = None
                print(f"  attention {impl} at batch {batch}: {report_error(e)}", flush=True)
            torch.cuda.empty_cache()
        results["attention"][key] = entry
        save()
    print(f"  attention at batch {batch}, context {args.context}: {results['attention'][key]}", flush=True)

if "prompt_attention" not in results:
    results["prompt_attention"] = time_prompt_attention(args.attention_chunk)
    torch.cuda.empty_cache()
    save()
print(f"  attention of a prompt chunk of {args.attention_chunk} over one cache: {results['prompt_attention']:.1f} us per token", flush=True)


# -----------------------------------------------------------------------------
# tables

def cell(fmt, shape, batch):
    return results["cells"].get(f"{fmt}|{shape[0]}x{shape[1]}{'+bias' if shape[2] else ''}|{batch}", {})


def gemm_us(fmt, batch, with_head=True):
    """Matrix products of one step of the whole model, us per token."""
    total = 0.0
    for shape, n in shapes.items():
        if shape == head_shape and not with_head:  # prompt processing needs the logits of the last token only
            continue
        total += n * cell(fmt, shape, batch).get("us", math.nan)
    return total / batch


def attention_us(batch):
    """Attention over the cache, one layer, per sequence: the faster implementation."""
    entry = [t for t in results["attention"].get(str(batch), {}).values() if t is not None]
    return min(entry) / batch if entry else math.nan


def decode_us(fmt, batch):
    return gemm_us(fmt, batch) + layout.n_layer * attention_us(batch)


def prefill_us(fmt):
    return gemm_us(fmt, args.prefill_batch, with_head=False) + layout.n_layer * results["prompt_attention"]


def weights_gb(fmt):
    size, bytes_per_weight = 0, math.nan
    for shape, n in shapes.items():
        got = next((c["bytes"] for c in (cell(fmt, shape, b) for b in [args.prefill_batch, *batches]) if "bytes" in c), math.nan)
        size += n * got
        if shape == head_shape:
            bytes_per_weight = got / (shape[0] * shape[1])
    return (size + embed_weights * bytes_per_weight) / 1e9


def fmt_number(v, spec=".0f"):
    return "-" if math.isnan(v) else format(v, spec)


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *["| " + " | ".join(r) + " |" for r in rows]]


lines = [f"# Formats on {gpu.name}, shapes of {args.model}", "",
         f"torch {torch.__version__}. {layout.n_layer} layers. KV cache: {kv_gb:.2f} GB per agent at {args.context} tokens. "
         "Times are GPU kernel times of the matrix products and of attention over the cache; norms, activations and "
         f"launch overhead are not included. Attention during prompt processing is measured on chunks of {args.attention_chunk} tokens "
         "reading one cache. A batch larger than the agents that fit in memory cannot be served at this context. "
         "A dash means the kernel failed for at least one matrix shape at that batch (see Failures).", "", "## Availability", ""]
lines += table(["Format", "Works", "Check against the dense product"],
               [[f, "yes" if results["verify"][f]["ok"] else "no",
                 f"error {results['verify'][f]['error']:.1e}" if results["verify"][f]["ok"] else results["verify"][f]["reason"]]
                for f in formats])

lines += ["", "## Decode, us per token (matrix products + attention)", ""]
lines += table(["Format", *[f"batch {b}" for b in batches], f"prefill (batch {args.prefill_batch})"],
               [[f, *[fmt_number(decode_us(f, b)) for b in batches], fmt_number(prefill_us(f))] for f in usable])

lines += ["", "## Matrix products only, us per token: how much of the time a format can change", ""]
lines += table(["Format", *[f"batch {b}" for b in batches], f"prefill (batch {args.prefill_batch})"],
               [[f, *[fmt_number(gemm_us(f, b)) for b in batches], fmt_number(gemm_us(f, args.prefill_batch, with_head=False))]
                for f in usable])

lines += ["", f"## GPU time of one agent turn, ms ({args.new_tokens} new prompt tokens + {args.gen_tokens} generated; "
          "times bf16's in brackets, below 1 is faster)", ""]
base = {b: (args.new_tokens * prefill_us("bf16") + args.gen_tokens * decode_us("bf16", b)) / 1e3 for b in batches} if "bf16" in usable else {}
rows = []
for f in usable:
    row = [f]
    for b in batches:
        t = (args.new_tokens * prefill_us(f) + args.gen_tokens * decode_us(f, b)) / 1e3
        row.append("-" if math.isnan(t) else f"{t:.0f}" + (f" ({t / base[b]:.2f})" if b in base and base[b] == base[b] else ""))
    rows.append(row)
lines += table(["Format", *[f"batch {b}" for b in batches]], rows)

lines += ["", f"## Agents that fit at {args.context} tokens of context ({args.reserve:.0%} of the memory kept free)", ""]
rows = []
for f in usable:
    w = weights_gb(f)
    agents = [math.floor(max(0.0, v * (1 - args.reserve) - w) / kv_gb) if not math.isnan(w) else math.nan for v in vrams]
    rows.append([f, fmt_number(w, ".1f"), *[fmt_number(a, ".0f") for a in agents]])
lines += table(["Format", "Weights (GB)", *[f"{v:.0f} GB GPU" for v in vrams]], rows)

failures = [[k.replace("|", ", "), c["error"]] for k, c in results["cells"].items() if "error" in c and k.split("|")[0] in usable]
if failures:
    lines += ["", "## Failures", ""]
    lines += table(["Format, shape, batch", "Reason"], failures)

md ="\n".join(lines) + "\n"
with open(os.path.join(out_dir, "probe.md"), "w", encoding="utf-8") as f:
    f.write(md)
print("\n" + md, flush=True)
print(f"Results in {out_dir}", flush=True)
