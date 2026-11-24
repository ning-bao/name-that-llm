"""The layer-at-a-time scorer for the open-weight candidates, on small random models and small GGUF files."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")
pytest.importorskip("transformers.models.glm5_next")
from nllm import candidate_lm, gguf_segments


def tiny_flash():
    from transformers.models.glm5_next import modeling_glm5_next as m
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    types = ["linear_attention", "linear_attention", "linear_attention", "indexed_attention", "linear_attention",
             "indexed_attention"]
    cfg = Glm5NextTextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=4, n_routed_experts=12, num_experts_per_tok=3, n_shared_experts=1,
        first_k_dense_replace=3, q_lora_rank=24, kv_lora_rank=16, qk_head_dim=8, qk_nope_head_dim=8, qk_rope_head_dim=0,
        v_head_dim=8, index_head_dim=8, index_n_heads=2, index_topk=64, index_kpool=4, hc_mult=4, mla_use_nope=True,
        layer_types=types, mlp_layer_types=["dense"] * 3 + ["sparse"] * 3, indexer_types=["full"] * 6, pad_token_id=0,
        eos_token_id=1, linear_attn_config={"num_heads": 4, "head_dim": 8, "short_conv_kernel_size": 4,
                                            "gate_lower_bound": -5.0},
    )
    return cfg, m.Glm5NextTextModel(cfg).eval()


def tiny_glm53():
    from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as m
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=7,
        num_attention_heads=4, num_key_value_heads=4, n_routed_experts=12, num_experts_per_tok=3, n_shared_experts=1,
        first_k_dense_replace=3, q_lora_rank=24, kv_lora_rank=16, qk_head_dim=12, qk_nope_head_dim=8, qk_rope_head_dim=4,
        v_head_dim=8, head_dim=4, index_head_dim=8, index_n_heads=2, index_topk=64, pad_token_id=0, eos_token_id=1,
        mlp_layer_types=["dense"] * 3 + ["sparse"] * 4,
        indexer_types=["full", "full", "full", "shared", "shared", "full", "shared"],
    )
    cfg._attn_implementation = "eager"
    return cfg, m.GlmMoeDsaModel(cfg).eval()


def randomise(model) -> None:
    """Default initialisations leave routers and gates nearly symmetric; spread them."""
    torch.manual_seed(0)
    for p in model.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.2)


@pytest.mark.parametrize(("build", "arch_class"), [(tiny_flash, candidate_lm.Flash), (tiny_glm53, candidate_lm.Glm53)])
def test_layer_at_a_time_matches_the_library_forward(build, arch_class, tmp_path):
    """Padded batches, several token groups and expert chunks, and a crash after layer 3 resumed from its
    checkpoint: the same log-probabilities as the library's forward on each sequence alone."""
    cfg, model = build()
    randomise(model)
    ids = [torch.randint(2, 97, (n,)).tolist() for n in (9, 14, 14, 5, 11)]
    head = torch.randn(97, 64)
    with torch.no_grad():
        want = []
        for row in ids:
            hidden = model(input_ids=torch.tensor([row])).last_hidden_state
            want.append(candidate_lm.token_stats(hidden[:, :-1], head, torch.tensor([row[1:]]))["lp"][0].numpy())

    def layer_at(i):
        layer = model.layers[i]
        if cfg.mlp_layer_types[i] != "sparse":
            return layer, None
        return layer, lambda e: (layer.mlp.experts.gate_up_proj[e], layer.mlp.experts.down_proj[e])

    done, pooled, checkpoint = [], {}, tmp_path / "state.checkpoint"

    class Crash(Exception):
        pass

    def stop_after_layer_3(i, save):  # as a run that dies after it checkpointed and deleted a segment
        done.append(i)
        if i == 3:
            save()
            raise Crash

    arch = arch_class(cfg)
    arch.kda_row_chunks = 2  # Flash: the linear attention then runs on two rows of a three-row batch at a time
    options = {"dtype": torch.float32, "max_tokens": 40, "max_rows": 3, "log": lambda *a, **k: None,
               "stat_positions": 20, "chunk": 5, "group_tokens": 30, "pool_layers": (2, cfg.num_hidden_layers),
               "pool_from": 3, "checkpoint": checkpoint}
    with pytest.raises(Crash):
        candidate_lm.score(arch, ids, torch.device("cpu"), model.embed_tokens, layer_at, model.norm, head,
                           after_layer=stop_after_layer_3, pooled={}, **options)
    assert done == [0, 1, 2, 3] and checkpoint.exists()

    def no_embedding(_):
        raise AssertionError("a resumed run must not embed again")

    def layers_after_3(i):
        assert i > 3
        return layer_at(i)

    done.clear()
    got = candidate_lm.score(arch_class(cfg), ids, torch.device("cpu"), no_embedding, layers_after_3, model.norm, head,
                             after_layer=lambda i, save: done.append(i), pooled=pooled, **options)
    with pytest.raises(ValueError, match="other sequences"):
        candidate_lm.score(arch_class(cfg), ids[:2], torch.device("cpu"), no_embedding, layers_after_3, model.norm,
                           head, **options)
    assert done == list(range(4, cfg.num_hidden_layers))
    assert [len(g["lp"]) for g in got] == [len(row) - 1 for row in ids]
    worst = max(float(np.abs(g["lp"] - w).max()) for g, w in zip(got, want, strict=True))
    assert worst < 1e-3
    # Pooled hidden states: in input order, the mean over tokens 3.. of the library's final hidden state (which the
    # output norm then reads), so the norm of the pooled last layer matches the library's on one sequence.
    assert sorted(pooled) == [2, cfg.num_hidden_layers] and pooled[2].shape == (len(ids), cfg.hidden_size)
    with torch.no_grad():
        hooked = {}
        handle = model.norm.register_forward_hook(lambda mod, args, out: hooked.update(x=args[0]))
        model(input_ids=torch.tensor([ids[1]]))
        handle.remove()
    assert np.abs(pooled[cfg.num_hidden_layers][1] - hooked["x"][0, 3:].mean(0).numpy()).max() < 1e-3


def test_torch_decoders_equal_the_gguf_reference():
    rng = np.random.default_rng(0)
    for kind, (decode, reference) in gguf_segments.torch_decoders().items():
        size = gguf.constants.GGML_QUANT_SIZES[kind][1]
        blocks = rng.integers(0, 256, size=(50, size), dtype=np.uint8)
        blocks[:, :2] = rng.normal(size=(50, 1)).astype(np.float16).view(np.uint8)  # a finite block scale
        reference.init_grid()
        want = reference.dequantize_blocks(blocks)
        grid = torch.from_numpy(reference.grid.reshape(reference.grid_shape).copy())
        got = decode(torch.from_numpy(blocks), grid).numpy()
        assert got.shape == (50, 256)
        np.testing.assert_array_equal(got, want)


def write_gguf(path, tensors: dict) -> None:
    writer = gguf.GGUFWriter(str(path), "glm5-next")
    writer.add_uint32("ntl.layer_start", 0)
    writer.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(5000)])  # long arrays are skipped on read
    for name, value in tensors.items():
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_segments_undo_the_converter_and_fill_a_layer(tmp_path):
    """Two layers written the way llama.cpp's converter writes them (experts stacked, kv_b split per head, A_log
    as -exp, the grouped convolution as three, renamed gates) load back equal to the transformers layers."""
    cfg, model = tiny_flash()
    randomise(model)
    names = {v: k for k, v in gguf_segments.LAYER_NAMES.items()}
    tensors = {"token_embd.weight": model.embed_tokens.weight.detach().numpy()}
    for i in (3, 4):  # sparse attention + MoE, linear attention + MoE
        state = {k: v.detach().numpy() for k, v in model.layers[i].state_dict().items()}
        b = f"blk.{i}."
        for hf_name, value in state.items():
            if hf_name == "mlp.experts.gate_up_proj":
                gate, up = np.split(value, 2, axis=1)
                tensors[b + "ffn_gate_exps.weight"], tensors[b + "ffn_up_exps.weight"] = gate, up
            elif hf_name == "mlp.experts.down_proj":
                tensors[b + "ffn_down_exps.weight"] = value
            elif hf_name == "self_attn.kv_b_proj.weight":
                kv = value.reshape(cfg.num_attention_heads, cfg.qk_nope_head_dim + cfg.v_head_dim, -1)
                tensors[b + "attn_k_b.weight"] = kv[:, : cfg.qk_nope_head_dim].transpose(0, 2, 1)
                tensors[b + "attn_v_b.weight"] = kv[:, cfg.qk_nope_head_dim:]
            elif hf_name == "self_attn.forget_gate.A_log":
                tensors[b + "ssm_a"] = -np.exp(value)
            elif hf_name == "self_attn.conv1d.weight":
                for x, part in zip("qkv", np.split(value, 3, axis=0), strict=True):
                    tensors[f"{b}ssm_conv1d_{x}.weight"] = part
            else:
                tensors[b + gguf_segments.LAYER_NAMES[hf_name]] = value
    assert len(names) == len(gguf_segments.LAYER_NAMES)  # no two parameters share a GGUF name
    write_gguf(tmp_path / "seg00-L3-5.gguf", tensors)

    meta, table = gguf_segments.read_header(tmp_path / "seg00-L3-5.gguf")
    assert meta["general.architecture"] == "glm5-next" and meta["tokenizer.ggml.tokens"] == "<array of 5000>"
    assert table["blk.3.ffn_gate_exps.weight"][0] == (12, 32, 64) and table["token_embd.weight"][0] == (97, 64)

    segments = gguf_segments.Segments(tmp_path, dtype=torch.float32)
    np.testing.assert_array_equal(segments.numpy("token_embd.weight"), tensors["token_embd.weight"])
    for i in (3, 4):
        layer = gguf_segments.load_layer(type(model.layers[i]), cfg, segments, i, torch.device("cpu"))
        want = model.layers[i].state_dict()
        for name, value in {**dict(layer.named_parameters()), **dict(layer.named_buffers())}.items():
            if not name.startswith("mlp.experts."):
                assert torch.allclose(value.float(), want[name].float(), atol=1e-6), name
        gate_up, down = segments.experts(i, torch.device("cpu"))([7, 2])
        assert torch.equal(gate_up, want["mlp.experts.gate_up_proj"][[7, 2]])
        assert torch.equal(down, want["mlp.experts.down_proj"][[7, 2]])
    with pytest.raises(ValueError, match="expected"):
        wrong = type(cfg)(**{**cfg.to_dict(), "kv_lora_rank": 8})
        gguf_segments.load_layer(type(model.layers[3]), wrong, segments, 3, torch.device("cpu"))


def test_batches_are_length_sorted_and_cover_every_row_once():
    ids = [[1] * n for n in (5, 30, 7, 6, 29, 31)]
    batches = candidate_lm.batches_of(ids, pad_id=0, max_tokens=64, max_rows=3)
    rows = np.concatenate([r for r, _, _ in batches])
    assert sorted(rows.tolist()) == list(range(len(ids)))
    for r, padded, mask in batches:
        assert padded.shape[0] * padded.shape[1] <= 64 or len(r) == 1
        assert [int(m.sum()) for m in mask] == [len(ids[i]) for i in r]
        assert (padded[~mask] == 0).all()


def test_passage_features_summarise_token_statistics():
    stats = {"lp": np.array([-1.0, -3.0, -2.0, -0.5]), "rank": np.array([0, 4, 1, 0]),
             "entropy": np.array([1.0, 2.0, 1.5, 0.5]), "var": np.array([1.0, 1.0, 1.0, 1.0]),
             "top": np.array([-1.0, -0.2, -0.4, -0.5])}
    f = candidate_lm.passage_features(stats)
    assert f["n_tokens"] == 4 and np.isclose(f["lp_mean"], -1.625) and np.isclose(f["top1"], 0.5)
    assert np.isclose(f["curvature"], (stats["lp"] + stats["entropy"]).sum() / 2.0)
    assert np.isclose(f["gap_top"], 1.1) and f["lp_q50"] == np.median(stats["lp"])
