# Formats on NVIDIA A100 80GB PCIe, shapes of Qwen/Qwen2.5-32B-Instruct

torch 2.13.0+cu126. 64 layers. KV cache: 8.39 GB per agent at 32000 tokens. Times are GPU kernel times of the matrix products and of attention over the cache; norms, activations and launch overhead are not included. Attention during prompt processing is estimated from batch 64. A dash means the kernel cannot run at that batch or shape.

## Availability

| Format | Works | Check against the dense product |
|---|---|---|
| bf16 | yes | error 4.7e-03 |
| int4_g32 | yes | error 4.5e-03 |
| int4_g64 | yes | error 4.3e-03 |
| int4_g128 | yes | error 4.1e-03 |
| int8_weight | yes | error 2.4e-03 |
| w8a8 | yes | error 7.4e-03 |
| w8a8_gemm | yes | error 5.1e-03 |
| fp8 | no | RuntimeError: torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+ |
| fp8_gemm | no | RuntimeError: torch._scaled_mm is only supported on CUDA devices with compute capability >= 9.0 or 8.9, or ROCm MI300+ |
| sparse24 | yes | error 2.0e-03 |

## Decode, us per token (matrix products + attention)

| Format | batch 1 | batch 4 | batch 16 | batch 64 | prefill (batch 2048) |
|---|---|---|---|---|---|
| bf16 | 47660 | 15510 | 7500 | 7835 | 7411 |
| int4_g32 | 25401 | 11849 | 9591 | 11543 | 11045 |
| int4_g64 | 23718 | 11610 | 9267 | 11261 | 10843 |
| int4_g128 | 23168 | 11562 | 9124 | 11160 | 10761 |
| int8_weight | 451678 | 156530 | 132429 | 132675 | 126877 |
| w8a8 | - | - | - | 10713 | 8128 |
| w8a8_gemm | - | - | - | 10429 | 8033 |
| sparse24 | - | - | - | - | - |

## Matrix products only, us per token: how much of the time a format can change

| Format | batch 1 | batch 4 | batch 16 | batch 64 | prefill (batch 2048) |
|---|---|---|---|---|---|
| bf16 | 41172 | 10151 | 2593 | 701 | 278 |
| int4_g32 | 18913 | 6489 | 4684 | 4409 | 3911 |
| int4_g64 | 17230 | 6251 | 4360 | 4127 | 3709 |
| int4_g128 | 16680 | 6203 | 4216 | 4026 | 3627 |
| int8_weight | 445189 | 151171 | 127521 | 125542 | 119743 |
| w8a8 | - | - | - | 3579 | 994 |
| w8a8_gemm | - | - | - | 3295 | 899 |
| sparse24 | - | - | - | - | - |

## GPU time of one agent turn, ms (2000 new prompt tokens + 300 generated; times bf16's in brackets, below 1 is faster)

| Format | batch 1 | batch 4 | batch 16 | batch 64 |
|---|---|---|---|---|
| bf16 | 29121 (1.00) | 19476 (1.00) | 17073 (1.00) | 17173 (1.00) |
| int4_g32 | 29710 (1.02) | 25644 (1.32) | 24967 (1.46) | 25552 (1.49) |
| int4_g64 | 28802 (0.99) | 25169 (1.29) | 24466 (1.43) | 25064 (1.46) |
| int4_g128 | 28472 (0.98) | 24990 (1.28) | 24258 (1.42) | 24869 (1.45) |
| int8_weight | 389256 (13.37) | 300712 (15.44) | 293482 (17.19) | 293556 (17.09) |
| w8a8 | - | - | - | 19469 (1.13) |
| w8a8_gemm | - | - | - | 19195 (1.12) |
| sparse24 | - | - | - | - |

## Agents that fit at 32000 tokens of context (10% of the memory kept free)

| Format | Weights (GB) | 24 GB GPU | 40 GB GPU | 80 GB GPU |
|---|---|---|---|---|
| bf16 | 65.5 | 0 | 0 | 0 |
| int4_g32 | 20.5 | 0 | 1 | 6 |
| int4_g64 | 18.4 | 0 | 2 | 6 |
| int4_g128 | 17.4 | 0 | 2 | 6 |
| int8_weight | 32.8 | 0 | 0 | 4 |
| w8a8 | 32.8 | 0 | 0 | 4 |
| w8a8_gemm | 32.8 | 0 | 0 | 4 |
| sparse24 | - | - | - | - |
