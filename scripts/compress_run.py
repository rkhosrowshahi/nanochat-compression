"""
Run one compression experiment from a YAML config: the NSGA-II search, then the test-set scores
of the final front, then every plot. Rerunning the same command continues where it stopped
(the search resumes from its checkpoint, scoring from test_scores.json).

    python -m scripts.compress_run configs/compress/d34_emb.yaml
    python -m scripts.compress_run configs/compress/d34_emb.yaml --set search.n_gen=20 eval_batch_size=8
    python -m scripts.compress_run configs/compress/d34_emb.yaml --smoke     # 2 tiny generations first
    python -m scripts.compress_run configs/compress/d34_emb.yaml --steps score plot

Config keys (anything else is an error, so a typo cannot be silently ignored):
    model               HuggingFace id or nanochat:<base|sft|rl>:<tag>[:<step>]
    out_dir             default compress_runs/<config file name>
    download            HF repo of a nanochat checkpoint, fetched once if missing (e.g. karpathy/nanochat-d34)
    nanochat_base_dir   where nanochat checkpoints and the tokenizer live (default ~/.cache/nanochat)
    include             extra groups to compress: [embeddings, lm_head]
    dtype, offload_originals, eval_batch_size
    search:             compress_search options by name: pop_size, n_gen, seed, k_space, k_min, k_max,
                        auto_k_bounds, warmup_*, c_min, c_max, alpha_min, beta_max, no_pruning,
                        max_delta_loss, size_mode, formats, reconstruction
    calibration:        seqs, seq_len, every (0 = one fixed batch), from (reuse another run's batch),
                        dataset, config, split, data_seed
    test:               seq_len, wikitext_seqs, c4_seqs, max_delta_loss (default search.max_delta_loss)
The resolved config is saved to <out_dir>/run_config.yaml.
"""

import argparse
import copy
import os
import subprocess
import sys

import yaml

# also runnable as a file (python scripts/compress_run.py ...): put the repo root on the path for nanochat,
# and run the steps from it so the scripts.* modules and relative paths resolve
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

TOP = {"model", "out_dir", "download", "nanochat_base_dir", "include", "dtype", "offload_originals",
       "eval_batch_size", "search", "calibration", "test"}
SEARCH = {"pop_size", "n_gen", "seed", "k_space", "k_min", "k_max", "auto_k_bounds", "warmup_min_delta",
          "warmup_max_delta", "warmup_log_k_range", "warmup_log_k_resolution", "c_min", "c_max", "alpha_min",
          "beta_max", "no_pruning", "max_delta_loss", "size_mode", "formats", "reconstruction"}
CALIBRATION = {"seqs": "calib_seqs", "seq_len": "seq_len", "every": "calib_every", "from": "calib_from",
               "dataset": "calib_dataset", "config": "calib_config", "split": "calib_split", "data_seed": "data_seed"}
TEST = {"seq_len": "seq_len", "wikitext_seqs": "wikitext_seqs", "c4_seqs": "c4_test_seqs",
        "max_delta_loss": "max_delta_loss"}

parser = argparse.ArgumentParser(description="search + test scores + plots of one compression experiment")
parser.add_argument("config", help="YAML file, e.g. configs/compress/d34_emb.yaml")
parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config values, dotted for sections: search.n_gen=20 eval_batch_size=8")
parser.add_argument("--steps", nargs="+", default=["search", "score", "plot"], choices=["search", "score", "plot"])
parser.add_argument("--smoke", action="store_true",
                    help="quick check: population 4, 2 generations, 8 calibration windows, 4 test windows, into <out_dir>_smoke")
args = parser.parse_args()


def check_keys(section, allowed, where):
    unknown = set(section) - set(allowed)
    if unknown:
        sys.exit(f"{args.config}: unknown key(s) {sorted(unknown)} in {where}; allowed: {sorted(allowed)}")


def flags(options):
    """{name: value} -> command-line flags: --name value, a bare --name for true, nothing for false/None."""
    out = []
    for name, value in options.items():
        flag = "--" + name.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag)
        elif isinstance(value, (list, tuple)):
            out += [flag, *map(str, value)]
        else:
            out += [flag, str(value)]
    return out


cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
for item in args.set:
    key, _, value = item.partition("=")
    *path, last = key.split(".")
    node = cfg
    for p in path:
        node = node.setdefault(p, {})
    node[last] = yaml.safe_load(value)  # so 20 is an int, true a bool, [a, b] a list
check_keys(cfg, TOP, "the top level")
search, calib, test = (dict(cfg.get(s) or {}) for s in ("search", "calibration", "test"))
check_keys(search, SEARCH, "search")
check_keys(calib, CALIBRATION, "calibration")
check_keys(test, TEST, "test")
assert "model" in cfg, f"{args.config}: 'model' is required"
out_dir = cfg.get("out_dir") or os.path.join("compress_runs", os.path.splitext(os.path.basename(args.config))[0])
if args.smoke:
    out_dir += "_smoke"
    search.update(pop_size=4, n_gen=2)
    calib["seqs"] = 8
    test.update(wikitext_seqs=4, c4_seqs=4)
os.makedirs(out_dir, exist_ok=True)
resolved = copy.deepcopy(cfg)
resolved.update(out_dir=out_dir, search=search, calibration=calib, test=test)
with open(os.path.join(out_dir, "run_config.yaml"), "w", encoding="utf-8") as f:
    yaml.safe_dump(resolved, f, sort_keys=False)
print(f"Experiment {args.config} -> {out_dir}", flush=True)

# nanochat checkpoints: where they live, and a one-time download
env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join(p for p in (REPO, env.get("PYTHONPATH")) if p)  # the steps import scripts.*, nanochat
if cfg["model"].startswith("nanochat:"):
    if cfg.get("nanochat_base_dir"):
        env["NANOCHAT_BASE_DIR"] = os.path.expanduser(cfg["nanochat_base_dir"])
    base = env.get("NANOCHAT_BASE_DIR") or os.path.expanduser("~/.cache/nanochat")
    if cfg.get("download") and {"search", "score"} & set(args.steps):  # the plots do not need the model
        from nanochat.compress_models import NANOCHAT_DIRS
        source, tag = cfg["model"].split(":")[1:3]
        ckpt_dir = os.path.join(base, NANOCHAT_DIRS[source], tag)
        if not (os.path.isdir(ckpt_dir) and os.path.exists(os.path.join(base, "tokenizer", "tokenizer.pkl"))):
            from huggingface_hub import hf_hub_download, list_repo_files
            print(f"Downloading {cfg['download']} into {base}", flush=True)
            for name in list_repo_files(cfg["download"]):
                if name in ("tokenizer.pkl", "token_bytes.pt"):
                    hf_hub_download(cfg["download"], name, local_dir=os.path.join(base, "tokenizer"))
                elif name.endswith(".json") and name.startswith("meta_") or name.endswith(".pt") and name.startswith("model_"):
                    hf_hub_download(cfg["download"], name, local_dir=ckpt_dir)

common = {"model": cfg["model"], "dtype": cfg.get("dtype"), "offload_originals": cfg.get("offload_originals"),
          "include": cfg.get("include") or None}
batch = cfg.get("eval_batch_size", 1)
max_delta = test.pop("max_delta_loss", search.get("max_delta_loss", 1.0))


def run(step, cmd, log=True):
    """Run one step; its output goes to the terminal and, unless the step logs itself, to run.log."""
    print(f"\n### {step}: {' '.join(cmd)}", flush=True)
    with open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            if log:
                f.write(line)
        if proc.wait():
            sys.exit(f"{step} failed (exit code {proc.returncode}); rerun the same command to continue")


py = [sys.executable, "-u", "-m"]
if "search" in args.steps:  # its own end-of-run test scoring is off (0 windows): the score step does it resumably
    run("search", py + ["scripts.compress_search"] + flags(common) + flags(search)
        + flags({CALIBRATION[k]: v for k, v in calib.items()})
        + ["--eval-batch-size", str(batch), "--wikitext-seqs", "0", "--c4-test-seqs", "0", "--out-dir", out_dir],
        log=False)  # compress_search writes run.log itself
if "score" in args.steps:
    run("score", py + ["scripts.compress_score", "--run-dir", out_dir, "--max-delta-loss", str(max_delta),
                       "--batch-size", str(batch)] + flags({TEST[k]: v for k, v in test.items()}))
if "plot" in args.steps:
    run("plot", py + ["scripts.compress_plot", "--run-dir", out_dir, "--all"])
print(f"\nDone: {out_dir} (test scores in test_scores.json, plots next to them)", flush=True)
