"""Make the slides' figures with larger text: the same plots as the report's (scripts/compress_plot.py and
scripts/compress_compare.py, unchanged), drawn with this folder's matplotlibrc (matplotlibrc_compare for
compress_compare, whose legend sits below the axes). Each run's data files are copied to a scratch folder first, so the
runs keep their own plots. Writes figures_slides/<name>.pdf, which make_slides.py uses in place of the report's figure
of the same name. Run from anywhere:

    python compress_runs/reports/slides/make_figures.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "figures_slides")


def plot(run, *flags, made):
    """compress_plot on one run; made: the file it writes in the run folder."""
    return [run], ["scripts.compress_plot", "--run-dir", "{r0}", *flags], "{r0}/" + made


def compare(runs, labels, *flags):
    """compress_compare on runs, written to {out}.pdf."""
    return list(runs), ["scripts.compress_compare", "--run-dirs", *(f"{{r{i}}}" for i in range(len(runs))), "--labels", *labels,
                        *flags, "--out", "{out}"], "{out}.pdf"


RTX = ("qwen_hw_2obj", "qwen_hw_3obj")
A100 = ("qwen_hw_2obj_p50_g100", "qwen_hw_3obj_p50_g100")
OBJ, OBJ_HV = ("2 objectives", "3 objectives"), ("2-objectives", "3-objectives")  # the labels of the report's plots
FRONT = ("Pareto front",)
# figure name in the report -> how it is made
FIGURES = {
    # RTX 3060: two-objective search, and the two searches compared
    "front_2obj_with_init_pop_log": plot("qwen_hw_2obj", "--x", "latency", "--with-init-pop", "--last-gen",
                                         made="pareto_front_latency_with_init_pop.pdf"),
    "front_2obj_with_init_pop_linear": plot("qwen_hw_2obj", "--x", "latency", "--y", "linear", "--with-init-pop", "--last-gen",
                                            made="pareto_front_latency_ylinear_with_init_pop.pdf"),
    "fronts_latency_log": compare(RTX, OBJ, "--x", "latency"),
    "fronts_latency_linear": compare(RTX, OBJ, "--x", "latency", "--y", "linear"),
    "fronts_memory_log": compare(RTX, OBJ, "--x", "memory"),
    "fronts_memory_linear": compare(RTX, OBJ, "--x", "memory", "--y", "linear"),
    "test_wikitext2_latency": compare(RTX, OBJ, "--metric", "wikitext2", "--x", "latency"),
    "hv_latency": compare(RTX, OBJ_HV, "--hv", "--x", "latency"),
    "hv_memory": compare(RTX, OBJ_HV, "--hv", "--x", "memory"),
    "hv_3d": compare(RTX, OBJ_HV, "--hv", "--3d"),
    # fronts_3d is kept as the report has it: alone on its slide, and its 3-D axis labels crowd with larger text
    # A100: each search, and the two compared
    **{f"a100_{n}obj_front_{x}": plot(run, "--x", x, "--last-gen", made=f"pareto_front_{x}.pdf")
       for n, run in zip("23", A100) for x in ("latency", "memory")},
    **{f"a100_{n}obj_test_{m}": compare([run], FRONT, "--metric", m, "--x", "latency") for n, run in zip("23", A100) for m in ("wikitext2", "c4")},
    "a100_fronts_latency": compare(A100, OBJ, "--x", "latency"),
    "a100_fronts_memory": compare(A100, OBJ, "--x", "memory"),
    "a100_hv_3d": compare(A100, OBJ, "--hv", "--3d"),
    # mixed search
    "mix_front_size": plot("qwen_mix", "--x", "size", "--last-gen", made="pareto_front.pdf"),
    "mix_front_time": plot("qwen_mix", "--x", "latency", "--last-gen", made="pareto_front_latency.pdf"),
    "mix_front_memory": plot("qwen_mix", "--x", "memory", "--last-gen", made="pareto_front_memory.pdf"),
    "mix_hv": plot("qwen_mix", "--hv", made="hypervolume.pdf"),
    **{f"mix_test_{m}": compare(["qwen_mix"], FRONT, "--metric", m, "--x", "size") for m in ("wikitext2", "c4")},
    # second KV search
    "kv2_front_time": plot("qwen_kv2", "--x", "latency", "--last-gen", made="pareto_front_latency.pdf"),
    "kv2_front_memory": plot("qwen_kv2", "--x", "memory", "--last-gen", made="pareto_front_memory.pdf"),
    "kv2_hv": plot("qwen_kv2", "--hv", made="hypervolume.pdf"),
    "kv2_test_wikitext2": compare(["qwen_kv2"], FRONT, "--metric", "wikitext2", "--x", "memory"),
}

os.makedirs(OUT, exist_ok=True)
only = set(sys.argv[1:])  # names to make (default: all)
with tempfile.TemporaryDirectory() as tmp:
    copies = {}
    os.makedirs(os.path.join(tmp, "out"))
    for name, (runs, cmd, made) in FIGURES.items():
        if only and name not in only:
            continue
        for run in runs:
            if run not in copies:  # the run's data, not its plots, checkpoints or tensors
                copies[run] = dst = os.path.join(tmp, run)
                os.makedirs(dst)
                for f in glob.glob(os.path.join(REPO, "compress_runs", run, "*")):
                    if os.path.isfile(f) and f.endswith((".csv", ".json", ".yaml")):
                        shutil.copy(f, dst)
        fill = {f"r{i}": copies[run] for i, run in enumerate(runs)} | {"out": os.path.join(tmp, "out", name)}
        rc = "matplotlibrc" if cmd[0] == "scripts.compress_plot" else "matplotlibrc_compare"
        subprocess.run([sys.executable, "-m", *(a.format(**fill) for a in cmd)], cwd=REPO, check=True, stdout=subprocess.DEVNULL,
                       env={**os.environ, "PYTHONPATH": REPO, "MATPLOTLIBRC": os.path.join(HERE, rc)})
        shutil.copy(made.format(**fill), os.path.join(OUT, f"{name}.pdf"))
        print(f"{name}.pdf  <- {' '.join(runs)}: {cmd[0]} {' '.join(a for a in cmd[1:] if '{' not in a)}")
