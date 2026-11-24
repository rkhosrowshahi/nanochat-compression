# Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization

arXiv 2607.22583. Muhammad Junaid Ali, Smail Niar, El Ghazali Talbi (Inria / Univ. Lille / UPHF). IEEE conference format.
Source read in full (`conference_101719.tex`).

## What it does

**Method:** two-stage structured **pruning**. It does no quantization and no format search.

- **Stage 1, NSGA-II (pymoo):**
  - Genes: 2L binary genes, whether to keep or drop each layer's attention block and MLP block (dropping means identity through the residual).
  - Objectives: (1) KL divergence from the original model, (2) measured latency.
  - Constraint: removed-parameter fraction ≤ target sparsity S.
  - Settings: population 40, 40 generations (1,600 evaluations), SBX crossover with mask mutation, 1,024 WikiText-2 calibration samples.
  - Returns about 8–10 Pareto candidates.
- **Stage 2, parallel Bayesian optimization (BoTorch, GP with RBF kernel, log-EI):**
  - On each Stage-1 candidate, searches continuous per-layer pruning ratios for attention heads and MLP neurons that use up the remaining sparsity budget S − p.
  - Heads and neurons are ranked by an importance score; Wanda-SP was best.
  - Selects one final model: lowest WikiText-2 perplexity, latency as tie-break. So the output is one model per sparsity level, not a front.
- **Fixed sparsity budgets:** 37.5% and 50%, searched separately.

**Models:** LLaMA-2-7B, Mistral-v0.3-7B, Qwen-2.5-7B, Phi-3-14B. No coding models.

**Evaluation:**
- Zero-shot MMLU, HellaSwag, ARC-e/c, PIQA, WinoGrande (average).
- Perplexity on WikiText-2, C4 and FineWeb-Edu.
- Latency: end-to-end time for a 100-token sequence, mean of 10 runs, on an A100 and a Jetson Nano. The Jetson models are 4-bit through bitsandbytes *after* pruning.
- No LoRA recovery.

**Cost:** about 4 h per sparsity level for Stage 1 and about 4 h for Stage 2, about 28 GPU-hours in total. They quote EvoPress at about 48 h.

## Key results

- Best average zero-shot accuracy of the structured-pruning baselines (ShortGPT, SliceGPT, BlockPruner, EvoPress, 2SSP, CFSP) at both sparsities. For example, Qwen-2.5-7B at 37.5%: 48.9 (51.8 with mixed calibration), against dense 68.7. All pruned models lose a lot.
- Latency for 100 tokens: 0.716 s on the A100, 4.9% faster than the fastest baseline (CFSP, 0.753 s).
- **KL vs perplexity as the search's quality objective:**
  - At 37.5% sparsity, KL correlates strongly with downstream accuracy (r = −0.967).
  - At 50% the correlation flips sign (r = +0.417), and perplexity is the better predictor (r = −0.828).
- KL + latency as objectives gave the lowest time and perplexity. KL + parameter count gave the best accuracy.

## Weak points, relevant if we cite or contrast it

- **"Latency" is not defined precisely enough to reproduce.** It is the end-to-end time of a 100-token sequence, and the text itself says prefill vs decode, batch size and prompt length must be matched to compare numbers. There is no TTFT/TPOT split and no tokens per second at batch 1. Ablations use 512-token sequences with times around 4 s, a different protocol. During the search, the commented-out text suggests 10-token inputs.
- **The multi-objective result is collapsed to one model:** lexicographic choice, perplexity first, then latency. Users don't get a front to choose from.
- **Size is a constraint (fixed sparsity), not an objective.** Memory is never measured.
- **Speedups are small:** about 5% over CFSP, because structured pruning at a fixed sparsity leaves little room to differ in time.
- Accuracy loss is large at 37.5–50% structured sparsity, matching our own finding that pruning MLP neurons costs 15–29 GSM8K points.

## Comparison with our plan

| | This paper | Our plan |
|---|---|---|
| Compression type | Structured pruning (blocks, heads, neurons) | Weight-only quantization format per matrix: {bf16, int4 g32/g64/g128} |
| Search | NSGA-II (binary keep/drop) + Bayesian optimization (continuous ratios) | NSGA-II over integer format genes per layer or per band |
| Quality objective | KL (Stage 1), WikiText-2 perplexity (Stage 2) | Exact KL over the full vocabulary on chat-formatted code |
| Speed objective | Measured end-to-end latency of 100 tokens (prefill + decode mixed) | TPOT at batch 1 (decode only): CUDA-graph decode loop, plus a latency table checked to within 2–5% |
| Memory | Constraint (sparsity budget) | Objective (deployed weight GiB) |
| Output | One model per sparsity | A Pareto front (KLD, TPOT, GiB) |
| Models / domain | General 7–14B; commonsense tasks | Coding models (Qwen2.5-Coder); code KLD, HumanEval/MBPP |

**Overlap with our novelty claims:**
- **Already done here:** NSGA-II plus KL-divergence fitness plus measured latency, no backpropagation, for LLM compression. We can't claim "KL + latency multi-objective evolutionary LLM compression" as new, and should cite this paper next to EvoPress.
- **Still open, as far as this paper shows:**
  1. **Quantization format and group size** as the decision variables. This paper prunes, and quantizes only afterwards, uniformly, for the Jetson.
  2. **A properly defined decode TPOT** (batch 1, prompt excluded) instead of mixed end-to-end time.
  3. **Memory as a third objective**, with a three-objective front given to the user.
  4. **A coding model with code-specific KL.**
- **Useful for our setup:**
  - Their KL-vs-perplexity correlation analysis is a model for checking our own proxy: correlate code KLD on the front with HumanEval/MBPP pass@1.
  - Their observation that KL stops predicting accuracy under heavy compression tells us to check whether code KLD still predicts benchmark accuracy at the aggressive end of our front.
- **Related to our earlier finding:** in our runs MLP pruning cost 15–29 GSM8K points, and this paper's pruned models also lose about 20 points on average. That supports keeping pruning off by default and searching formats instead.

## Ideas to take

1. Add a **proxy-validation section**: Pearson and Spearman correlation between search KLD and downstream pass@1 across front points, at mild and aggressive compression.
2. Their two-stage idea maps onto ours: coarse per-band NSGA-II first, then per-layer refinement starting from the band front. This is the middle option for the band/layer choice.
3. Report run cost in GPU-hours, as they do, so the search's expense is comparable with EvoPress and others.
