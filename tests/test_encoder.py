from dataclasses import replace
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


def test_decoder_continues_from_saved_adapters(tiny_qwen3, tmp_path):
    # A pool stage 1 saves adapters and head; stage 2 must load exactly that model and keep training it.
    cfg = encoder.TrainConfig("qwen3-4b-base", 1e-3, epochs=1, batch_size=8, micro_batch=4, lora_rank=4)
    texts = [f"passage number {i} with some words" for i in range(2 * len(LABELS))]
    logits, _ = encoder.fit_predict(tiny_qwen3, texts, LABELS * 2, texts[:4], cfg, seed=0, save_dir=tmp_path)
    tok, model, pad_id = encoder.load_classifier(tiny_qwen3, cfg, torch.device("cpu"), init_from=str(tmp_path))
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or "score" in n for n in trainable)
    reloaded = encoder.predict_logits(model, encoder.encode(tok, texts[:4], cfg), pad_id)
    assert np.abs(reloaded - logits[1]).max() <= 0.02 * np.abs(logits[1]).max() + 1e-5


def test_adafactor_option_trains(tiny_qwen3):
    cfg = encoder.TrainConfig("qwen3-4b-base", 1e-3, epochs=1, batch_size=8, micro_batch=4, lora_rank=4,
                              optimizer="adafactor")
    texts = [f"passage number {i} with some words" for i in range(2 * len(LABELS))]
    logits, history = encoder.fit_predict(tiny_qwen3, texts, LABELS * 2, texts[:3], cfg, seed=0)
    assert np.isfinite(logits[1]).all() and np.isfinite(history[0]["train_loss"])


def test_unit_batches_keep_groups_together_and_cover_every_row_once():
    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 7 + ["b"] * 3 + ["c"] + ["d"] * 18)
    batches = encoder.unit_batches(groups, batch_size=18, rng=rng)
    units = [u for b in batches for u in b]
    rows = np.sort(np.concatenate(units))
    assert rows.tolist() == list(range(len(groups)))
    assert all(len(u) <= 3 and len(set(groups[u])) == 1 for u in units)
    assert all(len(b) <= 6 for b in batches)
    assert len(units) == 3 + 1 + 1 + 6  # ceil(7/3) + 1 + 1 + 18/3


def test_unit_batches_with_affinity_put_confusable_labels_in_one_unit():
    labels = np.tile(np.arange(18), 4)  # 4 prompts with one row per label
    groups = np.repeat(["a", "b", "c", "d"], 18)
    affinity = np.zeros((18, 18))
    for block in range(6):  # six blocks of three labels that are only ever confused with each other
        affinity[3 * block : 3 * block + 3, 3 * block : 3 * block + 3] = 1.0
    np.fill_diagonal(affinity, 0.0)
    kw = {"batch_size": 18, "labels": labels, "affinity": affinity}
    units = [u for b in encoder.unit_batches(groups, rng=np.random.default_rng(0), share=1.0, **kw) for u in b]
    assert np.sort(np.concatenate(units)).tolist() == list(range(len(groups)))
    assert len(units) == 24 and all(len(set(groups[u])) == 1 for u in units)
    assert all(len({int(x) // 3 for x in labels[u]}) == 1 for u in units)  # every unit is one block
    # share 0 ignores the affinity: the same random cut as without it.
    plain = encoder.unit_batches(groups, 18, np.random.default_rng(1))
    same = encoder.unit_batches(groups, rng=np.random.default_rng(1), share=0.0, **kw)
    assert all((u == v).all() for b, c in zip(plain, same, strict=True) for u, v in zip(b, c, strict=True))
    # In between, block mates share a unit more often than in random cuts but not always.
    mixed = [u for b in encoder.unit_batches(groups, rng=np.random.default_rng(2), share=0.5, **kw) for u in b]
    whole = sum(len({int(x) // 3 for x in labels[u]}) == 1 for u in mixed)
    assert 0 < whole < 24
    # Uneven groups and labels with no affinity to anything still give every row one unit.
    odd = [u for b in encoder.unit_batches(np.array(["a"] * 7 + ["b"]), 18, np.random.default_rng(3),
                                           labels=np.arange(8), affinity=np.zeros((18, 18)), share=1.0) for u in b]
    assert sorted(len(u) for u in odd) == [1, 1, 3, 3] and np.sort(np.concatenate(odd)).tolist() == list(range(8))


def test_confusion_affinity_reads_mean_probabilities_symmetrically():
    logits = np.log(np.array([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1], [0.3, 0.6, 0.1], [0.0001, 0.0001, 0.9998]]))
    affinity = encoder.confusion_affinity(logits, np.array([0, 0, 1, 2]))
    assert np.allclose(affinity, affinity.T) and np.allclose(np.diag(affinity), 0)
    assert np.isclose(affinity[0, 1], (0.3 + 0.3) / 2)  # label 0 rows give label 1 0.3 on average; label 1 gives 0 0.3
    assert np.isclose(affinity[0, 2], (0.1 + 0.0001) / 2, atol=1e-6)
    # A label with no rows is confused only as far as the other labels' rows say.
    assert np.isclose(encoder.confusion_affinity(logits[:3], np.array([0, 0, 1]))[0, 2], 0.1 / 2)


def test_same_family_pairs_counts_pairs_inside_units():
    index = {m: i for i, m in enumerate(LABELS)}
    labels = np.array([index[m] for m in ("gemini-3.6-flash", "gemini-3.7-flash", "kimi-k3", "glm-5.3", "gpt-5.5")])
    batches = [[np.array([0, 1, 2])], [np.array([3, 4])]]  # pairs: one Gemini pair of three, then one mixed pair
    assert np.isclose(encoder.same_family_pairs(batches, labels), 1 / 4)
    assert encoder.same_family_pairs([[np.array([2])]], labels) == 0.0


@pytest.fixture(scope="module")
def tiny_qwen3_deep(tmp_path_factory):
    """Like tiny_qwen3 with 4 layers, for truncation and multi-depth pooling."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["qwen3-8b-base"], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    config = transformers.Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, eos_token_id=tok.eos_token_id,
    )
    path = tmp_path_factory.mktemp("tiny_qwen3_deep")
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def pooled_cfg(**kw):
    base = {"epochs": 2, "batch_size": 9, "micro_batch": 6, "lora_rank": 4, "pool_layers": (1, 2, 3),
            "head_lr_mult": 10.0, "centred_weight": 1.0, "class_weighted": True, "schedule": "cosine"}
    return encoder.TrainConfig("qwen3-4b-base", 1e-3, **(base | kw))


def test_pooled_decoder_truncates_and_trains_only_adapters_and_heads(tiny_qwen3_deep):
    from transformers import AutoTokenizer

    cfg = pooled_cfg()
    _, model, pad_id = encoder.load_classifier(tiny_qwen3_deep, cfg, torch.device("cpu"))
    assert isinstance(model, encoder.PooledDecoder)
    assert model.base.config.num_hidden_layers == 3
    assert isinstance(model.base.base_model.model.norm, torch.nn.Identity)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n or not n.startswith("base.") for n in trainable)
    assert {"head.weight", "centred_head.weight", "norms.0.weight"} <= trainable
    # Mean pooling ignores padding: a row scores the same alone and padded next to a longer one.
    ids = encoder.encode(AutoTokenizer.from_pretrained(tiny_qwen3_deep), ["short", "a much longer passage " * 8], cfg)
    alone = encoder.predict_logits(model, ids[:1], pad_id)
    padded = encoder.predict_logits(model, ids, pad_id, batch_size=2)[:1]
    assert np.abs(alone - padded).max() <= 0.02 * np.abs(alone).max() + 1e-5


def test_pooled_decoder_fits_with_centred_units_saves_and_reloads(tiny_qwen3_deep, tmp_path):
    cfg = pooled_cfg()
    texts = [f"passage number {i} — with some words" for i in range(3 * len(LABELS))]
    labels = LABELS * 3
    groups = [f"p{i // 6}" for i in range(len(texts))]  # 9 prompts of 6 answers, plus uneven units
    logits, history = encoder.fit_predict(tiny_qwen3_deep, texts, labels, texts[:5], cfg, seed=0,
                                          save_dir=tmp_path, train_groups=groups)
    assert sorted(logits) == [1, 2] and np.isfinite(logits[2]).all()
    assert all(np.isfinite(h["train_loss"]) for h in history)
    assert (tmp_path / "adapter_config.json").exists() and (tmp_path / encoder.POOLED_HEADS).exists()
    tok, model, pad_id = encoder.load_classifier(tiny_qwen3_deep, cfg, torch.device("cpu"), init_from=str(tmp_path))
    reloaded = encoder.predict_logits(model, encoder.encode(tok, texts[:5], cfg), pad_id)
    assert np.abs(reloaded - logits[2]).max() <= 0.02 * np.abs(logits[2]).max() + 1e-5
    with pytest.raises(ValueError):
        encoder.load_classifier(tiny_qwen3_deep, pooled_cfg(pool_layers=(1, 2)), torch.device("cpu"),
                                init_from=str(tmp_path))


def test_sibling_units_start_from_a_saved_model_and_centred_outputs_give_the_twin_logits(tiny_qwen3_deep, tmp_path):
    texts = [f"passage number {i} — with some words" for i in range(2 * len(LABELS))]
    labels = LABELS * 2
    groups = [f"p{i // 18}" for i in range(len(texts))]  # 2 prompts, one answer per label
    encoder.fit_predict(tiny_qwen3_deep, texts, labels, texts[:2], pooled_cfg(epochs=1), seed=0, save_dir=tmp_path,
                        train_groups=groups)
    cfg, centred = pooled_cfg(epochs=1, sibling_units=1.0), {}
    with pytest.raises(ValueError, match="starting model"):
        encoder.fit_predict(tiny_qwen3_deep, texts, labels, texts[:6], cfg, seed=0, train_groups=groups)
    out = tmp_path / "stage2"
    logits, history = encoder.fit_predict(tiny_qwen3_deep, texts, labels, texts[:6], cfg, seed=0, save_dir=out,
                                          init_from=str(tmp_path), train_groups=groups, twin_out=centred)
    assert 0 <= history[0]["same_family_pairs"] <= 1 and np.isfinite(logits[1]).all()
    outputs, bias = centred[1]
    assert outputs.shape == (6, len(LABELS)) and bias.shape == (len(LABELS),)
    # The centred head on features centred within a group = its saved outputs - their group mean + its bias.
    tok, model, pad_id = encoder.load_classifier(tiny_qwen3_deep, cfg, torch.device("cpu"), init_from=str(out))
    input_ids, mask = encoder.pad(encoder.encode(tok, texts[:6], cfg), pad_id, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        feats = model.features(input_ids, mask).float()
        direct = model.centred_head(feats - feats.mean(0, keepdim=True)).numpy()
    assert np.abs(direct - (outputs - outputs.mean(0) + bias)).max() <= 0.02 * np.abs(direct).max() + 1e-4


def test_centred_weight_needs_groups_and_a_pooled_decoder(tiny_qwen3_deep):
    with pytest.raises(ValueError):
        encoder.fit_predict(tiny_qwen3_deep, ["a"], [LABELS[0]], ["a"], pooled_cfg(), seed=0)
    with pytest.raises(ValueError):
        encoder.fit_predict(tiny_qwen3_deep, ["a"], [LABELS[0]], ["a"],
                            encoder.TrainConfig("qwen3-4b-base", 1e-3, centred_weight=1.0), seed=0, train_groups=["g"])


def test_pooled_qwen2_trains_and_hand_merged_loader_matches(tmp_path):
    """Qwen2.5 (qwen2 architecture, biased q/k/v projections) through the LoRA24 path: truncation, a fit, the
    saved-model scorer, and nllm.inference's peft-free hand merge, which must reproduce those scores."""
    transformers = pytest.importorskip("transformers")
    from nllm import inference

    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["qwen3-8b-base"], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer (stands in for Qwen2.5's, same vocabulary family) not in the local cache")
    config = transformers.Qwen2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, eos_token_id=tok.eos_token_id,
    )
    base = tmp_path / "base"
    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(config).save_pretrained(base)
    tok.save_pretrained(base)
    cfg = replace(pooled_cfg(), backbone="qwen2.5-14b", epochs=1)
    texts = [f"passage {i} " + "word " * (i % 7) for i in range(18)]
    labels, groups = list(LABELS), np.repeat(np.arange(6), 3)
    save = tmp_path / "fit"
    logits, _ = encoder.fit_predict(str(base), texts, labels, texts, cfg, seed=0, save_dir=save, train_groups=groups)
    saved = encoder.predict_saved(str(base), cfg, str(save), texts)
    assert np.abs(saved - logits[1]).max() < 1e-4
    model = inference.load_pooled_decoder(str(base), save, dtype=torch.float32)
    assert len(model.base.layers) == 3
    ids = inference.tokenize_decoder(save, texts, cfg.max_length)
    merged = inference.predict_logits(model, ids, tok.convert_tokens_to_ids(encoder.DECODER_PAD_TOKEN), dtype=None)
    assert (merged.argmax(axis=1) == saved.argmax(axis=1)).mean() >= 0.95 and np.abs(merged - saved).max() < 0.1
