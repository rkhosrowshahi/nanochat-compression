"""
Decode attention that reads a quantized KV cache in place (Triton), for the time table of nanochat/compress_kv2.py.
compress_kv.py timed a quantized cache as expanded to bf16 before each read, which reads it twice and writes it once: on the
A100 an int8 cache made the turn 53% slower than bf16. Serving engines read the codes inside the attention kernel instead,
so a smaller cache is also a faster read. This is such a kernel (flash decoding: each program reads one KV head over a span
of the cache, with all the query heads of its group, and the spans are combined after), with the cache in the layout of
compress_kv2.quantize_keys / quantize_values:
    keys    bf16 [B, Hkv, T, D]; int8c codes int8 [B, Hkv, T, D] or int4c codes uint8 [B, Hkv, T, D/2], with scales
            bf16 [B, Hkv, ceil(T / group), D]: one per channel and group of `group` tokens
    values  bf16 [B, Hkv, T, D]; int8 codes int8 [B, Hkv, T, D] or int4 codes uint8 [B, Hkv, T, D/2], with scales
            bf16 [B, Hkv, T]: one per token and head
int4 bytes hold channel j in the low nibble and channel j + D/2 in the high one (code + 8), so the kernel works on the two
halves of each head and never interleaves. The keys are read as stored (after the rotary embedding). A second kernel combines
the spans: the first version did that with about 8 torch operations, a floor of ~37 us per layer on the A100 whatever the cache.
"""

import torch
import triton
import triton.language as tl

KIND_ID = {"bf16": 0, "int8c": 1, "int4c": 2, "int8": 1, "int4": 2}


@triton.jit
def _load_half(ptr, rows, cols, mask, KIND: tl.constexpr, HALF: tl.constexpr, D: tl.constexpr):
    """Rows of one tensor of the cache as float codes (or bf16 values), both halves of the channels."""
    if KIND == 2:
        x = tl.load(ptr + rows[:, None] * HALF + cols[None, :], mask=mask[:, None], other=0)
        lo = (x & 15).to(tl.float32) - 8.0
        hi = (x >> 4).to(tl.float32) - 8.0
    else:
        lo = tl.load(ptr + rows[:, None] * D + cols[None, :], mask=mask[:, None], other=0).to(tl.float32)
        hi = tl.load(ptr + rows[:, None] * D + HALF + cols[None, :], mask=mask[:, None], other=0).to(tl.float32)
    return lo, hi


@triton.jit
def _decode_kernel(Q, K, KS, V, VS, O, ML, T, NG, chunk, n_rows, qk_scale,
                   H: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr,
                   GROUP: tl.constexpr, BLOCK_T: tl.constexpr, KEY: tl.constexpr, VAL: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)  # sequence * HKV + KV head
    split = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    g = tl.arange(0, G_PAD)
    c = tl.arange(0, HALF)
    t = tl.arange(0, BLOCK_T)
    rows = (bh // HKV) * H + (bh % HKV) * G + g  # the query heads of this KV head
    qmask = g < G
    q_lo = tl.load(Q + rows[:, None] * D + c[None, :], mask=qmask[:, None], other=0.0).to(tl.bfloat16)
    q_hi = tl.load(Q + rows[:, None] * D + HALF + c[None, :], mask=qmask[:, None], other=0.0).to(tl.bfloat16)
    m = tl.full([G_PAD], -1e30, tl.float32)
    l = tl.zeros([G_PAD], tl.float32)
    acc_lo = tl.zeros([G_PAD, HALF], tl.float32)
    acc_hi = tl.zeros([G_PAD, HALF], tl.float32)
    start = split * chunk
    end = tl.minimum(start + chunk, T)
    for t0 in range(start, end, BLOCK_T):
        tok = t0 + t
        tmask = tok < end
        row = bh * T + tok
        k_lo, k_hi = _load_half(K, row, c, tmask, KEY, HALF, D)
        if KEY != 0:  # each token's scales: the row of its group (repeated loads, from the L1 cache)
            group = bh * NG + tok // GROUP
            k_lo = k_lo * tl.load(KS + group[:, None] * D + c[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
            k_hi = k_hi * tl.load(KS + group[:, None] * D + HALF + c[None, :], mask=tmask[:, None], other=0.0).to(tl.float32)
        qk = tl.dot(q_lo, tl.trans(k_lo.to(tl.bfloat16))) + tl.dot(q_hi, tl.trans(k_hi.to(tl.bfloat16)))
        qk = tl.where(tmask[None, :], qk * qk_scale, float("-inf"))  # log2 units
        m_new = tl.maximum(m, tl.max(qk, 1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l = l * alpha + tl.sum(p, 1)
        v_lo, v_hi = _load_half(V, row, c, tmask, VAL, HALF, D)
        if VAL != 0:
            s = tl.load(VS + row, mask=tmask, other=0.0).to(tl.float32)[:, None]
            v_lo, v_hi = v_lo * s, v_hi * s
        pb = p.to(tl.bfloat16)
        acc_lo = acc_lo * alpha[:, None] + tl.dot(pb, v_lo.to(tl.bfloat16))
        acc_hi = acc_hi * alpha[:, None] + tl.dot(pb, v_hi.to(tl.bfloat16))
        m = m_new
    out = split * n_rows + rows
    tl.store(O + out[:, None] * D + c[None, :], acc_lo, mask=qmask[:, None])
    tl.store(O + out[:, None] * D + HALF + c[None, :], acc_hi, mask=qmask[:, None])
    tl.store(ML + out * 2, m, mask=qmask)
    tl.store(ML + out * 2 + 1, l, mask=qmask)


@triton.jit
def _combine_kernel(O, ML, OUT, S, n_rows, D: tl.constexpr, S_PAD: tl.constexpr):
    """One query head of one sequence: its S spans, weighted by their softmax sums."""
    r = tl.program_id(0).to(tl.int64)
    s = tl.arange(0, S_PAD)
    d = tl.arange(0, D)
    smask = s < S
    at = s * n_rows + r
    m = tl.load(ML + at * 2, mask=smask, other=-1e30)
    l = tl.load(ML + at * 2 + 1, mask=smask, other=0.0)
    w = tl.exp2(m - tl.max(m, 0))
    o = tl.load(O + at[:, None] * D + d[None, :], mask=smask[:, None], other=0.0)
    tl.store(OUT + r * D + d, (tl.sum(o * w[:, None], 0) / tl.sum(l * w, 0)).to(tl.bfloat16))


class DecodeAttention:
    """One decode step of attention, q [B, H, D] (one token per sequence) over a cache of T tokens per sequence, with
    the work split in about split_factor * (number of SMs) programs reading block_t tokens per step. Its buffers are
    allocated once, so calls can be captured in a CUDA graph."""

    def __init__(self, B, H, HKV, D, T, key, value, group=64, split_factor=2, block_t=64, num_warps=4, num_stages=2, device="cuda"):
        assert D % 32 == 0, "tl.dot needs half heads of at least 16 channels"
        assert key in ("bf16", "int8c", "int4c") and value in ("bf16", "int8", "int4")
        self.B, self.H, self.HKV, self.D, self.T, self.group = B, H, HKV, D, T, group
        self.key, self.value, self.num_warps, self.num_stages = KIND_ID[key], KIND_ID[value], num_warps, num_stages
        self.G, self.block_t = H // HKV, block_t
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        blocks = triton.cdiv(T, block_t)
        per_split = triton.cdiv(blocks, max(1, min(blocks, triton.cdiv(split_factor * sms, B * HKV))))
        self.splits = triton.cdiv(blocks, per_split)
        self.chunk = per_split * block_t
        self.o = torch.empty(self.splits, B * H, D, device=device, dtype=torch.float32)
        self.ml = torch.empty(self.splits, B * H, 2, device=device, dtype=torch.float32)
        self.out = torch.empty(B, H, D, device=device, dtype=torch.bfloat16)

    def __call__(self, q, k, ks, v, vs):
        _decode_kernel[(self.B * self.HKV, self.splits)](
            q, k, k if ks is None else ks, v, v if vs is None else vs, self.o, self.ml,
            self.T, triton.cdiv(self.T, self.group), self.chunk, self.B * self.H, self.D ** -0.5 * 1.4426950408889634,
            H=self.H, HKV=self.HKV, G=self.G, G_PAD=max(16, triton.next_power_of_2(self.G)), D=self.D, GROUP=self.group,
            BLOCK_T=self.block_t, KEY=self.key, VAL=self.value, num_warps=self.num_warps, num_stages=self.num_stages)
        _combine_kernel[(self.B * self.H,)](self.o, self.ml, self.out, self.splits, self.B * self.H, D=self.D,
                                            S_PAD=max(2, triton.next_power_of_2(self.splits)), num_warps=1)
        return self.out
