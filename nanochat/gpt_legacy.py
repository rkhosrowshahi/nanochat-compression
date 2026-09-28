"""
The GPT architecture of nanochat before 2026-01-11 (as of commit e1770a3), inference only, to load
checkpoints trained with it, e.g. the published base model karpathy/nanochat-d34 (Nov 2025,
vocab 65536). Later versions added residual/x0 lambdas, sliding windows, value embeddings, smear
and backout, so those checkpoints no longer load into nanochat.gpt.GPT.

Architecture: token embedding -> rmsnorm -> n_layer x [x + attn(norm(x)), x + mlp(norm(x))] ->
rmsnorm -> lm_head -> softcap 15. Attention: rotary embeddings, then QK rmsnorm, causal SDPA.
MLP: relu^2. No biases, no learnable norm parameters, untied embedding and lm_head.
It was trained and evaluated under torch.amp.autocast(bfloat16), so run it the same way.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.n_head != self.n_kv_head)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab_size = -(-config.vocab_size // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 10

    def init_rotary(self, base=10000):
        """The rotary tables (not in the checkpoint), on the model's device, in bfloat16 as trained."""
        head_dim = self.config.n_embd // self.config.n_head
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        freqs = torch.outer(torch.arange(self.rotary_seq_len, dtype=torch.float32, device=device), inv_freq)
        self.cos = freqs.cos().bfloat16()[None, :, None, :]
        self.sin = freqs.sin().bfloat16()[None, :, None, :]

    def forward(self, idx):
        T = idx.size(1)
        assert T <= self.cos.size(1), f"sequence length {T} beyond the rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        return softcap * torch.tanh(logits / softcap)
