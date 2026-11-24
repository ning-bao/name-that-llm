from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from nllm import encoder, inference
from nllm.data import LABELS, load_train


def test_tokenize_matches_training_tokenizer(tmp_path):
    """The notebook's tokenizers-only path gives the training ids, after the save_pretrained round trip."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["ettin-400m"], local_files_only=True)
        texts = list(load_train().text)
    except (OSError, FileNotFoundError):
        pytest.skip("Ettin tokenizer or train.csv not available locally")
    texts += ["“Curly” — dashes…  double  spaces\n\nand tabs\t" * 400, "", " "]
    tok.save_pretrained(tmp_path)
    cfg = encoder.TrainConfig("ettin-400m", 1e-5)
    assert inference.tokenize(tmp_path, texts, cfg.max_length) == encoder.encode(tok, texts, cfg)


def test_tokenize_decoder_matches_training_tokenizer(tmp_path):
    """Decoder ids (special-token strings read as text, EOS appended) match encoder.encode after save_pretrained."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["qwen3-8b-base"], local_files_only=True)
        texts = list(load_train().text)
    except (OSError, FileNotFoundError):
        pytest.skip("Qwen3 tokenizer or train.csv not available locally")
    texts += ["“Curly” — dashes…  <|endoftext|> and <|fim_pad|> as text\n\n\t" * 300, "", " "]
    tok.save_pretrained(tmp_path)
    cfg = encoder.TrainConfig("qwen3-8b-base", 1e-4)
    assert inference.tokenize_decoder(tmp_path, texts, cfg.max_length) == encoder.encode(tok, texts, cfg)


class LengthModel(torch.nn.Module):
    """Scores each row with its unpadded length in column 0, and records the dtype it ran under."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.dtypes = set()

    def forward(self, input_ids, attention_mask):
        self.dtypes.add(torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else torch.float32)
        logits = torch.zeros(len(input_ids), len(LABELS))
        logits[:, 0] = attention_mask.sum(dim=1).float()
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("dtype", [None, torch.bfloat16])
def test_predict_logits_dtype_and_order(dtype):
    model = LengthModel()
    out = inference.predict_logits(model, [[1] * n for n in (7, 2, 9, 4)], pad_id=0, batch_size=3, dtype=dtype)
    assert out[:, 0].tolist() == [7, 2, 9, 4]
    assert model.dtypes == {dtype or torch.float32}
