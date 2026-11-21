from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from nllm import encoder
from nllm.data import LABELS


def test_length_grouped_batches_cover_every_row_once():
    lengths = np.random.default_rng(0).integers(10, 400, size=1003)
    batches = encoder.length_grouped_batches(lengths, 32, np.random.default_rng(1))
    assert sorted(np.concatenate(batches).tolist()) == list(range(1003))
    assert max(len(b) for b in batches) == 32


def test_pad_masks_padding():
    ids, mask = encoder.pad([[5, 6, 7], [8]], pad_id=1, device="cpu")
    assert ids.tolist() == [[5, 6, 7], [8, 1, 1]]
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]


class LengthModel(torch.nn.Module):
    """Scores each row with its unpadded length in column 0, to check order restoration."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        logits = torch.zeros(len(input_ids), len(LABELS))
        logits[:, 0] = attention_mask.sum(dim=1).float()
        return SimpleNamespace(logits=logits)


def test_predict_logits_restores_input_order():
    ids = [[1] * n for n in (7, 2, 9, 4, 4, 1)]
    out = encoder.predict_logits(LengthModel(), ids, pad_id=0, batch_size=4)
    assert out[:, 0].tolist() == [7, 2, 9, 4, 4, 1]


@pytest.fixture(scope="module")
def tiny_qwen3(tmp_path_factory):
    """A randomly initialised 2-layer Qwen3 saved with the real Qwen3 tokenizer (skipped if not cached)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["qwen3-8b-base"], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    config = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, eos_token_id=tok.eos_token_id,
    )
    path = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_decoder_encode_appends_eos_and_reads_special_tokens_as_text(tiny_qwen3):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_qwen3)
    cfg = encoder.TrainConfig("qwen3-4b-base", 1e-4)
    ids = encoder.encode(tok, ["Plain text — with a dash.", "a <|endoftext|> b"], cfg)
    assert all(i[-1] == tok.eos_token_id for i in ids)
    assert tok.eos_token_id not in ids[1][:-1]
    assert tok.decode(ids[0][:-1]) == "Plain text — with a dash."


def test_decoder_pooling_ignores_padding(tiny_qwen3):
    from transformers import AutoTokenizer

    cfg = encoder.TrainConfig("qwen3-4b-base", 1e-4)
    tok, model, pad_id = encoder.load_classifier(tiny_qwen3, cfg, torch.device("cpu"))
    assert pad_id not in (tok.eos_token_id, tok.pad_token_id)
    ids = encoder.encode(AutoTokenizer.from_pretrained(tiny_qwen3), ["short", "a much longer passage " * 8], cfg)
    alone = encoder.predict_logits(model, ids[:1], pad_id)
    padded = encoder.predict_logits(model, ids, pad_id, batch_size=2)[:1]
    assert np.abs(alone - padded).max() <= 0.02 * np.abs(alone).max() + 1e-6


def test_decoder_fit_trains_only_adapters_and_head(tiny_qwen3, tmp_path):
    cfg = encoder.TrainConfig("qwen3-4b-base", 1e-3, epochs=2, batch_size=8, micro_batch=4, lora_rank=4)
    _, model, _ = encoder.load_classifier(tiny_qwen3, cfg, torch.device("cpu"))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or "score" in n for n in trainable)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)

    texts = [f"passage number {i} — with some words" for i in range(2 * len(LABELS))]
    labels = LABELS * 2
    logits, history = encoder.fit_predict(tiny_qwen3, texts, labels, texts[:5], cfg, seed=0, save_dir=tmp_path)
    assert sorted(logits) == [1, 2]
    assert logits[2].shape == (5, len(LABELS)) and np.isfinite(logits[2]).all()
    assert len(history) == 2
    assert (tmp_path / "adapter_config.json").exists()


def test_select_breaks_ties_toward_fewer_epochs_then_smaller_lr():
    y = np.array(LABELS[:3])
    perfect = np.eye(len(LABELS))[:3]
    wrong = np.roll(perfect, 1, axis=1)
    logits = {(3e-5, 2): wrong, (3e-5, 5): perfect, (1e-5, 5): perfect, (3e-5, 4): perfect}
    (lr, epoch), table = encoder.select(logits, y)
    assert (lr, epoch) == (3e-5, 4)
    assert len(table) == 4
