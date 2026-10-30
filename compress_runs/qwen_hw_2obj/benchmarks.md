# Qwen/Qwen2.5-0.5B-Instruct: benchmarks of compressed models

| Model | Calib. loss vs. uncompressed | Decode (ms) | Faster by | Memory (MB) | Smaller by | Bits/weight | PIQA (0-shot) | ARC-e (0-shot) | ARC-c (0-shot) | HellaSwag (0-shot) | WinoGrande (0-shot) | BoolQ (0-shot) | MMLU (5-shot) | GSM8K (5-shot) | Average | Accuracy kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Uncompressed (bfloat16) | - | 6.78 | 1.00x | 1001 | 1.00x | 16.00 | 70.6 | 59.2 | 33.0 | 52.4 | 55.2 | 67.7 | 47.0 | 30.3 | 51.9 | 100.0% |
| #38 | +0.16% | 6.74 | 1.01x | 992 | 1.01x | 15.86 | 70.7 | 59.6 | 33.4 | 52.0 | 55.6 | 66.9 | 46.5 | 31.1 | 52.0 | 100.1% |
| #30 | +0.75% | 5.61 | 1.21x | 761 | 1.31x | 12.13 | 70.1 | 60.0 | 32.7 | 51.4 | 56.0 | 67.6 | 46.7 | 27.9 | 51.5 | 99.3% |
| #21 | +1.99% | 4.35 | 1.56x | 479 | 2.09x | 7.55 | 69.7 | 59.7 | 32.8 | 51.0 | 56.7 | 66.1 | 45.9 | 28.5 | 51.3 | 98.8% |
| #14 | +4.84% | 3.43 | 1.98x | 291 | 3.43x | 4.51 | 69.0 | 58.0 | 31.5 | 49.4 | 55.0 | 66.5 | 44.8 | 20.5 | 49.4 | 95.1% |
| #11 | +8.90% | 3.28 | 2.07x | 271 | 3.70x | 4.19 | 67.3 | 49.7 | 29.0 | 46.4 | 52.4 | 63.8 | 37.6 | 4.9 | 43.9 | 84.6% |

Scores are accuracy in percent (higher is better); 'Accuracy kept' is the average relative to the uncompressed model's. 'Calib. loss vs. uncompressed' is how much higher (in percent) each solution's loss on the search's calibration data is. Decode: predicted GPU time of one decode step (1 sequence, 1024 tokens of context); memory: weights plus that KV cache.
Metrics: PIQA acc_norm (±1.1 standard error); ARC-e acc_norm (±1.0 standard error); ARC-c acc_norm (±1.4 standard error); HellaSwag acc_norm (±0.5 standard error); WinoGrande acc (±1.4 standard error); BoolQ acc (±0.8 standard error); MMLU acc (±0.4 standard error); GSM8K exact match (±1.3 standard error).
