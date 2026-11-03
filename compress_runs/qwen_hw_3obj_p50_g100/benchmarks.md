# Qwen/Qwen2.5-0.5B-Instruct: benchmarks of compressed models

| Model | Calib. loss vs. uncompressed | Decode (ms) | Faster by | Memory (MB) | Smaller by | Bits/weight | PIQA (0-shot) | ARC-e (0-shot) | ARC-c (0-shot) | HellaSwag (0-shot) | WinoGrande (0-shot) | BoolQ (0-shot) | MMLU (5-shot) | GSM8K (5-shot) | Average | Accuracy kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Uncompressed (bfloat16) | - | 3.95 | 1.00x | 1001 | 1.00x | 16.00 | 70.6 | 59.2 | 33.0 | 52.4 | 55.2 | 67.7 | 47.0 | 30.3 | 51.9 | 100.0% |
| #48 | +0.06% | 3.92 | 1.01x | 965 | 1.04x | 15.42 | 70.3 | 58.8 | 33.4 | 52.2 | 56.1 | 67.1 | 47.1 | 30.6 | 52.0 | 100.1% |
| #40 | +0.93% | 3.75 | 1.05x | 672 | 1.49x | 10.68 | 69.9 | 59.3 | 32.5 | 51.5 | 57.3 | 67.4 | 46.2 | 29.9 | 51.8 | 99.7% |
| #32 | +1.83% | 3.60 | 1.10x | 485 | 2.06x | 7.65 | 69.6 | 58.0 | 31.7 | 51.0 | 55.8 | 65.7 | 46.1 | 27.9 | 50.7 | 97.7% |
| #26 | +4.73% | 3.39 | 1.16x | 306 | 3.27x | 4.75 | 67.9 | 55.0 | 31.1 | 49.1 | 55.4 | 66.5 | 45.1 | 18.3 | 48.5 | 93.5% |
| #22 | +9.10% | 3.28 | 1.20x | 273 | 3.66x | 4.24 | 67.4 | 56.1 | 29.4 | 45.4 | 54.0 | 62.0 | 33.3 | 2.6 | 43.8 | 84.3% |

Scores are accuracy in percent (higher is better); 'Accuracy kept' is the average relative to the uncompressed model's. 'Calib. loss vs. uncompressed' is how much higher (in percent) each solution's loss on the search's calibration data is. Decode: predicted GPU time of one decode step (1 sequence, 1024 tokens of context); memory: weights plus that KV cache.
Metrics: PIQA acc_norm (±1.1 standard error); ARC-e acc_norm (±1.0 standard error); ARC-c acc_norm (±1.4 standard error); HellaSwag acc_norm (±0.5 standard error); WinoGrande acc (±1.4 standard error); BoolQ acc (±0.8 standard error); MMLU acc (±0.4 standard error); GSM8K exact match (±1.3 standard error).
