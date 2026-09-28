"""
Score the final Pareto front of a compress_search run on the test sets, without
resuming the search. Works for runs made by any version of compress_search.

For every solution on the last logged generation's front with delta_loss (on that
generation's calibration batch) at most --max-delta-loss:
    - WikiText-2 test (whole split) and C4 validation (GPTQ protocol, 256 windows)
      loss and perplexity, with --seq-len token windows (2048 as in GPTQ/AWQ)
    - the size with the CURRENT size accounting (used levels + lookup tables), so
      runs made before that change are measured the same way; the size the search
      saw is kept as search_size_mb
    - the Huffman-coded storage size (reported only)
Results go to <run-dir>/test_scores.json, updated after every solution, so an
interrupted scoring resumes where it stopped.

    python -m scripts.compress_score --run-dir compress_runs/local_100
"""

import argparse
import json
import math
import os
import time

import pandas as pd
import torch
from nanochat import compress_eval, compress_models
from nanochat.compress import Candidate, GlobalCompressor

parser = argparse.ArgumentParser(description="test-set perplexity of the final Pareto front of a compress_search run")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--max-delta-loss", type=float, default=1.0, help="skip solutions with a larger calibration delta_loss")
parser.add_argument("--seq-len", type=int, default=2048)
parser.add_argument("--wikitext-seqs", type=int, default=-1, help="-1 = the whole test split, 0 = skip")
parser.add_argument("--c4-test-seqs", type=int, default=256, help="0 = skip")
parser.add_argument("--batch-size", type=int, default=1)
args = parser.parse_args()

config = json.load(open(os.path.join(args.run_dir, "config.json")))
search = config["args"]
out_path = os.path.join(args.run_dir, "test_scores.json")

# the last generation's front; older runs named the columns k and loss_increase
front = pd.read_csv(os.path.join(args.run_dir, "front.csv")).rename(columns={"k": "k_init", "loss_increase": "delta_loss"})
gen = int(front["gen"].max())
front = front[front["gen"] == gen].sort_values("size_mb")
selected = front[front["delta_loss"] <= args.max_delta_loss]
print(f"Generation {gen}: scoring {len(selected)} of {len(front)} front solutions (delta_loss <= {args.max_delta_loss}) "
      f"with {args.seq_len}-token windows", flush=True)

settings = {"generation": gen, "max_delta_loss": args.max_delta_loss, "seq_len": args.seq_len,
            "wikitext_seqs": args.wikitext_seqs, "c4_test_seqs": args.c4_test_seqs}
results = json.load(open(out_path)) if os.path.exists(out_path) else {"settings": settings, "baseline": None, "pareto": []}
assert results["settings"] == settings, f"{out_path} was made with different settings: {results['settings']}"
done = {(p["k_init"], p["c"], p["alpha"], p["beta"]) for p in results["pareto"]}


def save():
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, out_path)


device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer, exclude = compress_models.load(search["model"], search["dtype"], device)
compressor = GlobalCompressor(model, search["include_lm_head"], search["size_mode"], search["offload_originals"],
                              search.get("reconstruction", "grid"), formats=search.get("formats", "dense,bitmap").split(","),
                              prune=not search.get("no_pruning", False), exclude=exclude,
                              include_embeddings=search.get("include_embeddings", False))
test_sets = compress_eval.load_test_sets(tokenizer, args.seq_len, args.wikitext_seqs, args.c4_test_seqs)
for name, tokens in test_sets.items():
    print(f"Test set {name}: {tokens.size(0)} windows, {tokens.numel()} tokens", flush=True)

if results["baseline"] is None:
    results["baseline"] = {"size_mb": compressor.baseline_bits / 8 / 1e6}
    for name, tokens in test_sets.items():
        loss = compress_eval.eval_loss(model, tokens, args.batch_size)
        results["baseline"].update({f"{name}_loss": loss, f"{name}_ppl": math.exp(loss)})
    save()
    print("uncompressed: " + ", ".join(f"{n} ppl {results['baseline'][f'{n}_ppl']:.3f}" for n in test_sets), flush=True)

for _, row in selected.iterrows():
    cand = Candidate(int(row["k_init"]), float(row["c"]), float(row["alpha"]), float(row["beta"]))
    if (cand.k, cand.c, cand.alpha, cand.beta) in done:
        continue
    t0 = time.time()
    entry = {"k_init": cand.k, "c": cand.c, "alpha": cand.alpha, "beta": cand.beta,
             "calib_delta_loss": float(row["delta_loss"]), "search_size_mb": float(row["size_mb"])}
    try:
        stats = compressor.apply(cand, huffman=True)
        entry.update({key: stats[key] for key in ("size_mb", "size_ratio", "target_bits_per_weight", "huffman_size_mb",
                                                  "huffman_bits_per_weight", "sparsity_pct", "pruned_pct", "k_used_mean")})
        for name, tokens in test_sets.items():
            loss = compress_eval.eval_loss(model, tokens, args.batch_size)
            entry.update({f"{name}_loss": loss, f"{name}_ppl": math.exp(min(loss, 700.0))})
    finally:
        compressor.restore()
    entry["seconds"] = round(time.time() - t0)
    results["pareto"].append(entry)
    save()
    print(f"k={cand.k} c={cand.c:.3f} {entry['size_mb']:.1f} MB | "
          + ", ".join(f"{n} ppl {entry[f'{n}_ppl']:.3f}" for n in test_sets) + f" | {entry['seconds']}s", flush=True)
print(f"Done. {len(results['pareto'])} solutions in {out_path}", flush=True)
