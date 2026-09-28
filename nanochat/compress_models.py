"""
Load the model to compress, from HuggingFace or from a nanochat checkpoint, behind the one
interface the compression scripts use:
    model(input_ids=x, use_cache=False).logits    and    tokenizer(text, add_special_tokens=False)["input_ids"]

    --model Qwen/Qwen2.5-0.5B-Instruct    a HuggingFace model id or local path
    --model nanochat:base:d34             a nanochat checkpoint, <source>:<model tag>[:<step>]
                                          with source base | sft | rl, read from
                                          $NANOCHAT_BASE_DIR/{base,chatsft,chatrl}_checkpoints/<tag>
                                          (default base dir ~/.cache/nanochat; latest step if omitted)

A nanochat GPT keeps its fp32 master weights; its Linear layers cast them to the compute dtype
(bf16 on Ampere and newer) in every forward, so --dtype does not apply to it. Its two kinds of
tiny gates (ve_gate, smear_gate: 18 matrices, ~0.04M weights) act like the per-layer scalars and
are never compressed.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

NANOCHAT_PREFIX = "nanochat:"
NANOCHAT_EXCLUDE = ("ve_gate", "smear_gate")


class NanochatLM(nn.Module):
    """A nanochat GPT behind the HuggingFace call signature."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt = gpt

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.gpt(input_ids))  # softcapped fp32 logits over the real vocab

    def get_output_embeddings(self):
        return self.gpt.lm_head


class NanochatTokenizer:
    """nanochat's tokenizer behind the HuggingFace call signature (it adds no special tokens)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens, "the compression scripts tokenize raw text only"
        return {"input_ids": self.tokenizer.encode(text)}


def load(spec, dtype, device):
    """Returns (model in eval mode on device, tokenizer, names of modules never to compress)."""
    if not spec.startswith(NANOCHAT_PREFIX):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(spec)
        model = AutoModelForCausalLM.from_pretrained(spec, dtype=getattr(torch, dtype)).to(device).eval()
        return model, tokenizer, ()
    from nanochat.checkpoint_manager import load_model
    parts = spec[len(NANOCHAT_PREFIX):].split(":")
    assert len(parts) in (2, 3) and parts[0] in ("base", "sft", "rl"), \
        f"--model {spec}: expected nanochat:<base|sft|rl>:<model tag>[:<step>], e.g. nanochat:base:d34"
    step = int(parts[2]) if len(parts) == 3 else None
    gpt, tokenizer, _ = load_model(parts[0], torch.device(device), phase="eval", model_tag=parts[1], step=step)
    return NanochatLM(gpt).eval(), NanochatTokenizer(tokenizer), NANOCHAT_EXCLUDE
