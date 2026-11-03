# Qwen/Qwen2.5-0.5B-Instruct: benchmarks of compressed models

| Model | Calib. loss vs. uncompressed | Decode (ms) | Faster by | Memory (MB) | Smaller by | Bits/weight | PIQA (0-shot) | ARC-e (0-shot) | ARC-c (0-shot) | HellaSwag (0-shot) | WinoGrande (0-shot) | BoolQ (0-shot) | MMLU (5-shot) | GSM8K (5-shot) | Average | Accuracy kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Uncompressed (bfloat16) | - | 3.95 | 1.00x | 1001 | 1.00x | 16.00 | 70.6 | 59.2 | 33.0 | 52.4 | 55.2 | 67.7 | 47.0 | 30.3 | 51.9 | 100.0% |
| #48 | +0.17% | 3.89 | 1.02x | 922 | 1.09x | 14.73 | 70.5 | 60.9 | 33.2 | 52.3 | 55.4 | 68.5 | 46.6 | 30.5 | 52.2 | 100.6% |
| #44 | +0.89% | 3.70 | 1.07x | 714 | 1.40x | 11.36 | 70.9 | 58.8 | 32.4 | 51.4 | 55.6 | 67.1 | 46.6 | 28.1 | 51.4 | 99.0% |
| #42 | +1.47% | 3.63 | 1.09x | 628 | 1.59x | 9.97 | 70.0 | 59.8 | 32.7 | 50.7 | 56.7 | 66.2 | 46.0 | 27.1 | 51.1 | 98.5% |
| #35 | +4.44% | 3.38 | 1.17x | 305 | 3.28x | 4.74 | 68.2 | 55.4 | 30.5 | 49.2 | 54.6 | 67.2 | 45.6 | 19.6 | 48.8 | 94.0% |
| #27 | +9.72% | 3.19 | 1.24x | 303 | 3.30x | 4.74 | 67.2 | 58.1 | 30.5 | 46.9 | 55.7 | 60.2 | 41.6 | 5.8 | 45.8 | 88.1% |

Scores are accuracy in percent (higher is better); 'Accuracy kept' is the average relative to the uncompressed model's. 'Calib. loss vs. uncompressed' is how much higher (in percent) each solution's loss on the search's calibration data is. Decode: predicted GPU time of one decode step (1 sequence, 1024 tokens of context); memory: weights plus that KV cache.
Metrics: PIQA acc_norm (±1.1 standard error); ARC-e acc_norm (±1.0 standard error); ARC-c acc_norm (±1.4 standard error); HellaSwag acc_norm (±0.5 standard error); WinoGrande acc (±1.4 standard error); BoolQ acc (±0.8 standard error); MMLU acc (±0.4 standard error); GSM8K exact match (±1.3 standard error).
