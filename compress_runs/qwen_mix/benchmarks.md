# Qwen/Qwen2.5-0.5B-Instruct: benchmarks of the mix search (compress_runs/qwen_mix)

| Configuration | Calib loss (%) | Turn (ms) | Memory (MB) | Stored (MB) | Bits/weight | PIQA (0-shot) | ARC-e (0-shot) | ARC-c (0-shot) | HellaSwag (0-shot) | WinoGrande (0-shot) | BoolQ (0-shot) | MMLU (5-shot) | GSM8K (5-shot) | Average | Accuracy retained (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uncompressed (bfloat16) | 0.00 | 1018 (1.00x) | 2561 (1.00x) | 988 | 16.00 | 70.6 | 59.2 | 33.0 | 52.4 | 55.2 | 67.7 | 47.0 | 30.3 | 51.9 | 100.0 |
| qkv:3SSB o:6S3S up:8333 down:3363 head:S keep:1/1/1/1 -attn 0 -mlp 0 share:k257 c1 p0 rho1 | 1.90 | 1070 (0.95x) | 2101 (1.22x) | 346 | 8.55 | 69.7 | 59.3 | 32.5 | 50.8 | 57.5 | 66.5 | 45.5 | 27.3 | 51.1 | 98.5 |
| qkv:3SS8 o:6S8S up:6366 down:3SSS head:8 keep:1/1/1/1 -attn 0 -mlp 0 share:k65 c1 p0 rho1 | 3.43 | 1049 (0.97x) | 1996 (1.28x) | 272 | 6.85 | 70.0 | 61.2 | 32.5 | 50.3 | 55.5 | 65.6 | 45.5 | 25.9 | 50.8 | 97.9 |
| qkv:3B88 o:6638 up:8SS3 down:6S33 head:S keep:1/0.875/0.875/1 -attn 0 -mlp 0 share:k17 c1 p0 rho1 | 9.80 | 1040 (0.98x) | 2212 (1.16x) | 232 | 10.35 | 67.2 | 52.7 | 30.5 | 45.7 | 56.5 | 50.6 | 36.3 | 2.7 | 42.8 | 82.4 |
| qkv:BBSB o:BBBB up:BBBB down:BSBB head:B keep:1/1/1/1 -attn 0 -mlp 0 share:k129 c1 p0 rho1 | -0.00 | 1018 (1.00x) | 2561 (1.00x) | 947 | 16.00 | 70.2 | 59.3 | 33.8 | 52.5 | 56.6 | 67.7 | 46.9 | 29.9 | 52.1 | 100.4 |
| qkv:883B o:38BS up:BSBB down:BSSB head:S keep:1/1/1/1 -attn 0 -mlp 0 share:k257 c1 p0 rho1 | 0.91 | 1012 (1.01x) | 2521 (1.02x) | 673 | 15.35 | 69.6 | 59.9 | 33.2 | 51.3 | 56.0 | 65.4 | 45.7 | 24.0 | 50.6 | 97.5 |

Scores are accuracy in percent. Accuracy retained: the average as a percentage of the uncompressed model's. Calib loss: how much higher (%) the loss on the search's calibration batch is. Turn: predicted GPU time of one agent's turn (2000 prompt tokens, 300 generated, 4 agents, 32000 tokens of context), with the speedup over the uncompressed model; memory: deployed weights plus the KV cache, with the compression ratio; stored: the stored size of the weights.
Metrics: PIQA acc_norm (±1.1 standard error); ARC-e acc_norm (±1.0 standard error); ARC-c acc_norm (±1.4 standard error); HellaSwag acc_norm (±0.5 standard error); WinoGrande acc (±1.4 standard error); BoolQ acc (±0.8 standard error); MMLU acc (±0.4 standard error); GSM8K exact match (±1.3 standard error).
