import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nllm import genlm
from nllm.data import LABELS


@pytest.fixture(scope="module")
def tiny_qwen3_lm(tmp_path_factory):
    """A randomly initialised 2-layer Qwen3 causal LM saved with the real Qwen3 tokenizer (skipped if not cached)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    config = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, eos_token_id=tok.eos_token_id,
    )
    path = tmp_path_factory.mktemp("tiny_qwen3_lm")
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_batch_scores_only_passage_tokens():
    ids, mask, target = genlm._batch([([5, 6], [7, 8, 9]), ([5], [7])], pad_id=0, device="cpu")
    assert ids.tolist() == [[5, 6, 7, 8, 9], [5, 7, 0, 0, 0]]
    assert target.tolist() == [[-100, -100, 7, 8, 9], [-100, 7, -100, -100, -100]]
    assert mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]


def test_chunked_logprobs_match_full_logits(tiny_qwen3_lm):
    cfg = genlm.LMConfig(lora_rank=4, gradient_checkpointing=False)
    tok, model = genlm.load_lm(tiny_qwen3_lm, cfg, "cpu")
    ids, mask, target = genlm._batch([([5, 6], [7, 8, 9, 10]), ([5], [7, 11])], tok.eos_token_id, "cpu")
    got = genlm._token_logprobs(model, ids, mask, target, chunk=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    t = target[:, 1:]
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), t.clamp(min=0), reduction="none")
    want = -(nll * (t != -100)).sum(dim=1)
    assert torch.allclose(got, want, rtol=1e-2, atol=1e-2)
    got.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def test_fit_and_score_shapes(tiny_qwen3_lm):
    cfg = genlm.LMConfig(lr=1e-2, epochs=2, batch_size=8, micro_batch=4, lora_rank=4, gradient_checkpointing=False)
    texts = [f"passage {i} written in style {i % 3}" for i in range(2 * len(LABELS))]
    tok, model, history = genlm.fit(tiny_qwen3_lm, texts, LABELS * 2, cfg, seed=0)
    assert len(history) == 2 and np.isfinite(history[-1]["train_nll_per_token"])
    scores = genlm.score(tok, model, texts[:5], cfg, batch_size=7)
    assert scores.shape == (5, len(LABELS)) and np.isfinite(scores).all() and (scores < 0).all()


def test_balancing_offsets_spread_predictions():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(1800, len(LABELS)))
    scores[:, 0] += 3.0  # one label the model finds much easier everywhere
    offsets = genlm.balancing_offsets(scores)
    share = np.bincount((scores + offsets).argmax(axis=1), minlength=len(LABELS)) / len(scores)
    assert share.max() < 2.5 / len(LABELS)
