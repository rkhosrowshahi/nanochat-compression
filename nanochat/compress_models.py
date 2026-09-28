"""
Load the model to compress, from HuggingFace or from a nanochat checkpoint, behind the one
interface the compression scripts use:
    model(input_ids=x, use_cache=False).logits    and    tokenizer(text, add_special_tokens=False)["input_ids"]

    --model Qwen/Qwen2.5-0.5B-Instruct    a HuggingFace model id or local path
    --model nanochat:base:d34             a nanochat checkpoint, <source>:<model tag>[:<step>]
                                          with source base | sft | rl, read from
                                          $NANOCHAT_BASE_DIR/{base,chatsft,chatrl}_checkpoints/<tag>
                                          and $NANOCHAT_BASE_DIR/tokenizer (default base dir
                                          ~/.cache/nanochat; latest step if omitted)

nanochat checkpoints come in two architectures. Those without a window_pattern in their config
(before 2026-01-11, e.g. the published karpathy/nanochat-d34) are built with nanochat.gpt_legacy
and run under bf16 autocast as they were trained; newer ones with nanochat.gpt.GPT, which manages
its dtypes itself. Either way the weights are stored in --dtype (bf16 by default, as for
HuggingFace models): the matmuls ran in bf16 during training anyway, so this changes nothing but
the memory. The two kinds of tiny gates of the new architecture (ve_gate, smear_gate) act like
per-layer scalars and are never compressed.
"""

import contextlib
import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn

NANOCHAT_PREFIX = "nanochat:"
NANOCHAT_EXCLUDE = ("ve_gate", "smear_gate")
NANOCHAT_DIRS = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}


class NanochatLM(nn.Module):
    """A nanochat GPT behind the HuggingFace call signature."""

    def __init__(self, gpt, autocast=False):
        super().__init__()
        self.gpt = gpt
        self.autocast = autocast

    def forward(self, input_ids, use_cache=False):
        ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
               if self.autocast and input_ids.device.type == "cuda" else contextlib.nullcontext())
        with ctx:
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

    from nanochat.checkpoint_manager import find_last_step, load_model
    from nanochat.common import get_base_dir
    from nanochat.tokenizer import get_tokenizer
    parts = spec[len(NANOCHAT_PREFIX):].split(":")
    assert len(parts) in (2, 3) and parts[0] in NANOCHAT_DIRS, \
        f"--model {spec}: expected nanochat:<base|sft|rl>:<model tag>[:<step>], e.g. nanochat:base:d34"
    ckpt_dir = os.path.join(get_base_dir(), NANOCHAT_DIRS[parts[0]], parts[1])
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"no checkpoint directory {ckpt_dir}: put model_<step>.pt and meta_<step>.json there "
                                f"(and the tokenizer in {os.path.join(get_base_dir(), 'tokenizer')}), or set NANOCHAT_BASE_DIR")
    step = int(parts[2]) if len(parts) == 3 else find_last_step(ckpt_dir)
    meta = json.load(open(os.path.join(ckpt_dir, f"meta_{step:06d}.json"), encoding="utf-8"))

    if "window_pattern" in meta["model_config"]:  # current architecture: nanochat's own loader
        gpt, tokenizer, _ = load_model(parts[0], torch.device(device), phase="eval", model_tag=parts[1], step=step)
        return NanochatLM(gpt.to(dtype=getattr(torch, dtype))).eval(), NanochatTokenizer(tokenizer), NANOCHAT_EXCLUDE

    from nanochat import gpt_legacy
    state = torch.load(os.path.join(ckpt_dir, f"model_{step:06d}.pt"), map_location="cpu", mmap=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    with torch.device("meta"):
        gpt = gpt_legacy.GPT(gpt_legacy.GPTConfig(**meta["model_config"]))
    gpt.load_state_dict(state, strict=True, assign=True)
    gpt = gpt.to(device=device, dtype=getattr(torch, dtype)).eval()
    gpt.init_rotary()
    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == gpt.config.vocab_size, \
        f"tokenizer vocab {tokenizer.get_vocab_size()} != model vocab {gpt.config.vocab_size}: wrong tokenizer.pkl?"
    print(f"Loaded {spec} step {step} with the pre-2026 nanochat architecture (nanochat/gpt_legacy.py)", flush=True)
    return NanochatLM(gpt, autocast=True).eval(), NanochatTokenizer(tokenizer), ()
