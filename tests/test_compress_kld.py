"""
Test the KL divergence quality measure (compress_eval.reference_hidden, eval_kld) on tiny random models (CPU).

python -m pytest tests/test_compress_kld.py -v
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from nanochat import compress_eval
from tests.test_compress_hw import tiny_nanochat, tiny_qwen


def perturbed(model, scale=0.5):
    other = copy.deepcopy(model)
    torch.manual_seed(1)
    with torch.no_grad():
        for p in other.parameters():
            p.add_(scale * p.std() * torch.randn_like(p))
    return other


def exact_kld(ref, model, tokens):
    with torch.no_grad():
        logp = F.log_softmax(ref(input_ids=tokens, use_cache=False).logits[:, :-1].float(), -1)
        logq = F.log_softmax(model(input_ids=tokens, use_cache=False).logits[:, :-1].float(), -1)
    return (logp.exp() * (logp - logq)).sum(-1)


@pytest.fixture
def setup():
    model = tiny_qwen()
    tokens = torch.randint(0, 512, (3, 40), generator=torch.Generator().manual_seed(0))
    return model, tokens


def test_the_model_against_itself(setup):
    model, tokens = setup
    r = compress_eval.eval_kld(model, tokens, compress_eval.reference_hidden(model, tokens))
    assert r["kld"] == pytest.approx(0, abs=1e-6) and r["top1_agree"] == 1.0
    assert r["loss"] == pytest.approx(compress_eval.eval_loss(model, tokens), rel=1e-5)


def test_equals_the_kld_of_the_two_full_distributions(setup):
    model, tokens = setup
    reference = compress_eval.reference_hidden(model, tokens)
    other = perturbed(model)  # the reference keeps the original head: the perturbed model has another one
    r = compress_eval.eval_kld(other, tokens, reference)
    exact = exact_kld(model, other, tokens)
    assert exact.mean().item() > 0.01
    assert r["kld"] == pytest.approx(exact.mean().item(), rel=1e-4)
    assert r["kld_median"] == pytest.approx(exact.flatten().median().item(), rel=1e-4)
    assert r["kld_p99"] == pytest.approx(exact.flatten().quantile(0.99).item(), rel=1e-4)


def test_batch_and_chunk_do_not_change_the_result(setup):
    model, tokens = setup
    other = perturbed(model)
    ref = compress_eval.reference_hidden(model, tokens, batch_size=2)
    assert torch.allclose(ref["hidden"], compress_eval.reference_hidden(model, tokens)["hidden"], atol=1e-5)
    a = compress_eval.eval_kld(other, tokens, ref, batch_size=2, chunk=7)
    b = compress_eval.eval_kld(other, tokens, ref)
    assert a == pytest.approx(b, rel=1e-5)


def test_logits_changed_after_the_head_are_refused():
    model = tiny_nanochat()  # softcaps its logits
    tokens = torch.randint(0, 512, (1, 16), generator=torch.Generator().manual_seed(0))
    with pytest.raises(AssertionError, match="output head"):
        compress_eval.reference_hidden(model, tokens)


class WordTokenizer:
    """One token per word, the document number as the conversation's first token."""

    def apply_chat_template(self, messages, tokenize=False):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(w) for w in text.split()]}


def test_code_windows_keep_test_documents_out_of_calibration(monkeypatch):
    from datasets import Dataset
    docs = Dataset.from_dict({"instruction": [str(d) for d in range(40)],
                              "response": [" ".join([str(d)] * 9) for d in range(40)]})  # 10 tokens per document
    monkeypatch.setattr(compress_eval, "load_dataset", lambda *a, **k: docs)
    monkeypatch.setattr(compress_eval, "CODE_TEST_DOCS", 10)
    tok = WordTokenizer()
    test = compress_eval.code_instruct_windows(tok, 10, -1, "test")
    assert test.shape == (10, 10) and all(len(set(w.tolist())) == 1 for w in test)  # one document per window here
    test_docs = set(test[:, 0].tolist())
    for seed in (0, 1):
        calib = compress_eval.code_instruct_windows(tok, 10, 30, "calib", seed=seed)
        assert calib.shape == (30, 10) and not test_docs & set(calib.flatten().tolist())
    a, b = (compress_eval.code_instruct_windows(tok, 10, 30, "calib", seed=s)[:, 0].tolist() for s in (0, 1))
    assert a != b and sorted(a) == sorted(b)  # the seed changes the order, not the documents
    assert torch.equal(test, compress_eval.code_instruct_windows(tok, 10, -1, "test"))  # the test set is fixed
