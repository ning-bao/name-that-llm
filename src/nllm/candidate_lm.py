"""Token log-probabilities of passages under an open-weight candidate model, one decoder layer at a time.

GLM-5.3-Flash (about 320 B parameters) and GLM-5.3 (about 740 B) are candidates whose weights are public
(nllm.gguf_segments). Neither fits in memory, so every passage's hidden state is carried through one layer before
the next layer's weights are read, and each weight is read once per pass:

    A. per batch, on the GPU: attention, and for a mixture-of-experts layer the router's choice of 8 experts per
       token and the shared expert;
    B. for a mixture-of-experts layer: the routed experts are unpacked a few at a time and each runs on the tokens
       routed to it, over all batches at once (`run_experts`);
    C. per batch: the residual (for GLM-5.3-Flash, its four hyper-connection streams) is put back together.

The attention, router, norm and hyper-connection modules are transformers' own (`models/glm5_next`,
`models/glm_moe_dsa`). Only the layer's control flow and the expert loop are written here, and tests/ check both
against the library's forward pass on small random models. `passage_features` turns the per-token statistics into
the per-passage features a stacker reads; they use no label.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nllm.gguf_segments import Segments, load_layer


@torch.no_grad()
def run_experts(x_all, idx_all, w_all, experts, n_experts: int, device, chunk: int = 8, swiglu_limit: float = 1e9,
                group_tokens: int = 300_000) -> torch.Tensor:
    """For every token, the sum over its routed experts of weight x expert(x). CPU tensors in, CPU tensor out.

    x_all is [T, D]; idx_all and w_all are [T, k] (expert ids and float32 weights). Tokens go to `device` in groups
    of at most `group_tokens`, so it holds one group's inputs and outputs; each group sweeps the experts it uses,
    `chunk` at a time (`experts(ids)` -> (gate_up [n, 2I, D], down [n, D, I])). Unused experts are never read."""
    out_all = torch.empty_like(x_all)
    k = idx_all.shape[1]
    for g in range(0, len(x_all), group_tokens):
        x, idx, w = (t[g: g + group_tokens].to(device) for t in (x_all, idx_all, w_all))
        out = torch.zeros_like(x)
        order = torch.argsort(idx.reshape(-1), stable=True)  # (token, slot) pairs grouped by expert
        counts = torch.bincount(idx.reshape(-1), minlength=n_experts).cumsum(0).tolist()
        starts = [0] + counts[:-1]
        used = [e for e in range(n_experts) if counts[e] > starts[e]]
        for lo in range(0, len(used), chunk):
            ids = used[lo: lo + chunk]
            gate_up, down = experts(ids)
            for j, e in enumerate(ids):
                pairs = order[starts[e]: counts[e]]
                token, slot = pairs // k, pairs % k
                gate, up = F.linear(x[token], gate_up[j]).chunk(2, dim=-1)
                act = F.silu(gate.clamp(max=swiglu_limit)) * up.clamp(min=-swiglu_limit, max=swiglu_limit)
                out.index_add_(0, token, (F.linear(act, down[j]).float() * w[token, slot, None]).to(out.dtype))
            del gate_up, down
        out_all[g: g + group_tokens] = out.cpu()
        del x, idx, w, out, order
    return out_all


class _Routing:
    """One layer's mixture-of-experts inputs and routing for every batch, gathered on the CPU."""

    def __init__(self, total: int):
        self.total, self.start, self.x = total, 0, None

    def add(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> None:
        n = x.shape[0] * x.shape[1]
        if self.x is None:
            self.x = torch.empty(self.total, x.shape[-1], dtype=x.dtype)
            self.idx = torch.empty(self.total, idx.shape[-1], dtype=torch.long)
            self.w = torch.empty(self.total, idx.shape[-1], dtype=torch.float32)
        s = slice(self.start, self.start + n)
        self.x[s], self.idx[s], self.w[s] = x.reshape(n, -1).cpu(), idx.cpu(), w.float().cpu()
        self.start += n


class Flash:
    """GLM-5.3-Flash (`glm5_next`): linear (KDA) and sparse attention layers, four hyper-connection streams."""

    name = "flash"

    def __init__(self, cfg, kda_row_chunks: int = 12):
        from transformers.models.glm5_next import modeling_glm5_next as m

        self.cfg, self.m, self.layer_class = cfg, m, m.Glm5NextTextDecoderLayer
        cfg._attn_implementation = "eager"
        # The library's reference linear attention builds a [rows, heads, chunks, 64, 64, head_dim] float32 tensor
        # and a few temporaries of that size: about 0.6 GB per row and 64-token chunk at this model's width. So
        # it is called on a few rows at a time (rows x chunks <= kda_row_chunks); rows do not interact.
        self.kda_row_chunks = kda_row_chunks

    @staticmethod
    def config(model_dir):
        from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

        raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        return Glm5NextTextConfig(**raw["text_config"])

    def start(self, embeddings: list, device) -> list:
        """The per-batch states to carry, from the embeddings (all None when a checkpoint supplies the states)."""
        return [None if e is None else e.unsqueeze(2).expand(-1, -1, self.cfg.hc_mult, -1).contiguous()
                for e in embeddings]

    def final_norm(self):
        return self.m.Glm5NextTextRMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)

    def last_hidden(self, state: torch.Tensor) -> torch.Tensor:
        return state.mean(dim=2)  # the library's Glm5NextTextHyperHead

    def extra_state(self):
        return None

    def restore(self, saved) -> None:
        pass

    @torch.no_grad()
    def run_layer(self, i: int, layer, states: list, masks: list, device, experts=None, **options) -> None:
        sparse, kept = self.cfg.mlp_layer_types[i] == "sparse", []
        routing = _Routing(sum(m.numel() for m in masks))
        for b, (h, mask) in enumerate(zip(states, masks, strict=True)):
            h, mask = h.to(device), mask.to(device)
            dtype = h.dtype
            post, comb, x = layer.attn_hc(h)
            x = layer.input_layernorm(x)
            if self.cfg.layer_types[i] == "linear_attention":
                rows = max(1, self.kda_row_chunks // math.ceil(x.shape[1] / 64))
                x = torch.cat([layer.self_attn(hidden_states=x[r: r + rows], cache_params=None,
                                               attention_mask=mask[r: r + rows]) for r in range(0, len(x), rows)])
            else:
                x, _, _ = layer.self_attn(hidden_states=x, attention_mask=mask, past_key_values=None, use_cache=False,
                                          position_embeddings=None, prev_topk_indices=None)
            h = post.to(dtype).unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.to(dtype).transpose(-1, -2), h)
            post, comb, x = layer.ffn_hc(h)
            x = layer.post_attention_layernorm(x)
            base, post = torch.matmul(comb.to(dtype).transpose(-1, -2), h), post.to(dtype)
            if not sparse:
                states[b] = (post.unsqueeze(-1) * layer.mlp(x).unsqueeze(-2) + base).cpu()
                continue
            _, w, idx = layer.mlp.gate(x)
            routing.add(x, idx, w)
            kept.append((post.cpu(), base.cpu(), layer.mlp.shared_experts(x).cpu(), x.shape))
            states[b] = None
        if not sparse:
            return
        out = run_experts(routing.x, routing.idx, routing.w, experts, layer.mlp.gate.num_experts, device,
                          swiglu_limit=self.cfg.swiglu_limit, **options)
        del routing
        start = 0
        for b, (post, base, shared, shape) in enumerate(kept):
            n = shape[0] * shape[1]
            mlp = (out[start: start + n].reshape(shape) + shared).to(device)
            states[b] = (post.to(device).unsqueeze(-1) * mlp.unsqueeze(-2) + base.to(device)).cpu()
            kept[b] = None
            start += n


class Glm53:
    """GLM-5.3 (`glm_moe_dsa`): plain residuals; sparse attention whose top-k indices pass from a layer with its
    own indexer ("full") to the "shared" layers after it."""

    name = "glm53"

    def __init__(self, cfg):
        from transformers.models.glm_moe_dsa import modeling_glm_moe_dsa as m

        self.cfg, self.m, self.layer_class = cfg, m, m.GlmMoeDsaDecoderLayer
        cfg._attn_implementation = "eager"
        self.topk, self.rotary = None, None

    @staticmethod
    def config(model_dir):
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

        raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        skip = ("architectures", "quantization_config", "transformers_version", "model_type")
        return GlmMoeDsaConfig(**{k: v for k, v in raw.items() if k not in skip})

    def start(self, embeddings: list, device) -> list:
        self.topk = [None] * len(embeddings)
        self.rotary = self.m.GlmMoeDsaRotaryEmbedding(config=self.cfg).to(device)
        return embeddings

    def final_norm(self):
        return self.m.GlmMoeDsaRMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)

    def last_hidden(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def extra_state(self):
        return [None if t is None else t.to(torch.int32) for t in self.topk]

    def restore(self, saved) -> None:
        self.topk = [None if t is None else t.long() for t in saved]

    @torch.no_grad()
    def run_layer(self, i: int, layer, states: list, masks: list, device, experts=None, **options) -> None:
        from transformers.masking_utils import create_causal_mask

        sparse, kept = self.cfg.mlp_layer_types[i] == "sparse", []
        routing = _Routing(sum(m.numel() for m in masks))
        for b, (h, mask) in enumerate(zip(states, masks, strict=True)):
            h, mask = h.to(device), mask.to(device)
            positions = torch.arange(h.shape[1], device=device).unsqueeze(0)
            causal = create_causal_mask(config=self.cfg, inputs_embeds=h, attention_mask=mask, past_key_values=None,
                                        position_ids=positions, allow_is_causal_skip=False)
            x, _, topk = layer.self_attn(
                hidden_states=layer.input_layernorm(h), attention_mask=causal, position_ids=positions,
                past_key_values=None, use_cache=False, position_embeddings=self.rotary(h, position_ids=positions),
                prev_topk_indices=None if self.topk[b] is None else self.topk[b].to(device))
            self.topk[b] = None if topk is None else topk.cpu()
            h = h + x
            x = layer.post_attention_layernorm(h)
            if not sparse:
                states[b] = (h + layer.mlp(x)).cpu()
                continue
            _, w, idx = layer.mlp.gate(x)
            routing.add(x, idx, w)
            kept.append(((h + layer.mlp.shared_experts(x)).cpu(), x.shape))
            states[b] = None
        if not sparse:
            return
        limit = float(getattr(self.cfg, "swiglu_limit", None) or 1e9)  # this architecture's experts are not clamped
        out = run_experts(routing.x, routing.idx, routing.w, experts, layer.mlp.gate.num_experts, device,
                          swiglu_limit=limit, **options)
        del routing
        start = 0
        for b, (partial, shape) in enumerate(kept):
            n = shape[0] * shape[1]
            states[b] = partial + out[start: start + n].reshape(shape)
            kept[b] = None
            start += n


ARCHITECTURES = {"flash": Flash, "glm53": Glm53}


def batches_of(ids: list[list[int]], pad_id: int, max_tokens: int, max_rows: int) -> list:
    """Length-sorted batches: (row positions, right-padded ids [B, S], mask [B, S])."""
    order = np.argsort([len(i) for i in ids], kind="stable")
    out, start = [], 0
    while start < len(order):
        longest, n = 0, 0
        while start + n < len(order) and n < max_rows:
            longest_next = max(longest, len(ids[order[start + n]]))
            if n and longest_next * (n + 1) > max_tokens:
                break
            longest, n = longest_next, n + 1
        rows = order[start: start + n]
        padded = torch.full((n, longest), pad_id, dtype=torch.long)
        mask = torch.zeros((n, longest), dtype=torch.bool)
        for r, row in enumerate(rows):
            padded[r, : len(ids[row])] = torch.tensor(ids[row])
            mask[r, : len(ids[row])] = True
        out.append((rows, padded, mask))
        start += n
    return out


@torch.no_grad()
def token_stats(hidden: torch.Tensor, head: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per position: the log-probability and rank of the target token, the best token's log-probability, and the
    entropy and variance of the log-probability under the model's own next-token distribution (for a
    Fast-DetectGPT style curvature)."""
    logp = F.log_softmax(F.linear(hidden, head).float(), dim=-1)
    p = logp.exp()
    lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mean = (p * logp).sum(-1)
    return {"lp": lp, "rank": (logp > lp.unsqueeze(-1)).sum(-1), "entropy": -mean,
            "var": (p * logp.square()).sum(-1) - mean.square(), "top": logp.max(-1).values}


@torch.no_grad()
def score(arch, ids: list[list[int]], device, embed, layer_at, final_norm, head, dtype=torch.bfloat16,
          max_tokens: int = 12000, max_rows: int = 64, log=print, after_layer=None, stat_positions: int = 1024,
          pooled: dict | None = None, pool_layers=(), pool_from: int = 0, checkpoint=None, **options) -> list[dict]:
    """Per sequence, statistics (token_stats) for predicting token t + 1 from tokens 0..t: arrays of length len - 1.

    `arch` is a Flash or Glm53. `embed(ids)` gives embeddings (its optional `.release()` is called once every batch
    is embedded); `layer_at(i)` gives (layer module, experts reader or None); `final_norm` is the output norm
    module and `head` the output projection weight, or functions returning them (called after the last layer, so
    a download can still be in progress). The output projection runs on about `stat_positions` token positions at
    a time. `options` go to run_experts.

    `after_layer(i, save)` is called when layer i is done; calling `save()` writes everything a later call needs
    to continue after layer i to the file `checkpoint` (the hidden states: gigabytes). A call that finds that
    file, written for the same token ids, resumes after its layer instead of starting again. Segments are deleted
    as they are used when a model is larger than the disk, so a crash without a checkpoint costs the download.

    With `pool_layers` (1-based layer counts) the dict `pooled` receives, per such depth, a float32 array
    [sequences, hidden]: each sequence's hidden state after that layer, averaged over its tokens from position
    `pool_from` on (the passage, without the chat prefix)."""
    cfg = arch.cfg
    batches = batches_of(ids, cfg.pad_token_id, max_tokens, max_rows)
    masks = [mask for _, _, mask in batches]
    back = np.argsort(np.concatenate([rows for rows, _, _ in batches]))  # batch order -> input order
    key = hashlib.sha256(repr((ids, max_tokens, max_rows, str(dtype))).encode()).hexdigest()
    saved = torch.load(checkpoint, weights_only=False) if checkpoint is not None and Path(checkpoint).exists() else None
    if saved is not None and saved["key"] != key:
        raise ValueError(f"{checkpoint} was written for other sequences or batch settings")
    if saved is None:
        states = arch.start([embed(padded.to(device)).to(dtype).cpu() for _, padded, _ in batches], device)
        first = 0
    else:
        arch.start([None] * len(batches), device)
        states, first = saved["states"], saved["layer"] + 1
        arch.restore(saved["arch"])
        if pooled is not None:
            pooled.update(saved["pooled"])
        log(f"resumed after layer {first} from {checkpoint}", flush=True)
    getattr(embed, "release", lambda: None)()
    started = time.time()

    def save_after(i: int) -> None:
        tmp = Path(str(checkpoint) + ".tmp")
        torch.save({"key": key, "layer": i, "states": states, "arch": arch.extra_state(), "pooled": dict(pooled or {})}, tmp)
        os.replace(tmp, checkpoint)

    for i in range(first, cfg.num_hidden_layers):
        layer, experts = layer_at(i)
        arch.run_layer(i, layer, states, masks, device, experts, **options)
        del layer, experts
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if pooled is not None and i + 1 in pool_layers:
            means = []
            for state, mask in zip(states, masks, strict=True):
                keep = (mask & (torch.arange(mask.shape[1]) >= pool_from)).unsqueeze(-1).float()
                means.append((arch.last_hidden(state).float() * keep).sum(1) / keep.sum(1).clamp(min=1))
            pooled[i + 1] = torch.cat(means).numpy()[back]
        log(f"layer {i + 1}/{cfg.num_hidden_layers} ({cfg.mlp_layer_types[i]}): {time.time() - started:.0f}s", flush=True)
        if after_layer is not None:
            after_layer(i, (lambda i=i: save_after(i)) if checkpoint is not None else (lambda: None))
    final_norm = final_norm if isinstance(final_norm, torch.nn.Module) else final_norm()
    head = head() if callable(head) else head
    out = [None] * len(ids)
    for (rows, padded, mask), state in zip(batches, states, strict=True):
        hidden, targets = final_norm(arch.last_hidden(state.to(device))), padded[:, 1:].to(device)
        step = max(1, stat_positions // hidden.shape[1])  # the vocabulary is 155k wide: a few rows at a time
        parts = [token_stats(hidden[r: r + step, :-1], head, targets[r: r + step]) for r in range(0, len(rows), step)]
        stats = {k: torch.cat([part[k] for part in parts]) for k in parts[0]}
        for r, row in enumerate(rows):
            n = int(mask[r].sum()) - 1
            out[row] = {k: v[r, :n].float().cpu().numpy() for k, v in stats.items()}
    return out


def segment_model(arch, segments_for, device, dtype=torch.bfloat16):
    """(embed, layer_at, final_norm, head) for `score`, reading from GGUF segments. `segments_for(tensor name)`
    returns a Segments whose files hold that tensor (it may wait for a download)."""
    table = []

    def embed(ids):
        if not table:  # read on first use: a resumed run never needs it (its segment may be gone)
            table.append(torch.from_numpy(segments_for("token_embd.weight").numpy("token_embd.weight")).to(device, dtype))
        return F.embedding(ids, table[0])

    embed.release = table.clear  # over a gigabyte, not needed once every batch is embedded

    def layer_at(i: int):
        segments = segments_for(f"blk.{i}.ffn_norm.weight")
        layer = load_layer(arch.layer_class, arch.cfg, segments, i, device)
        return layer, segments.experts(i, device) if arch.cfg.mlp_layer_types[i] == "sparse" else None

    def final_norm():
        norm = arch.final_norm()
        weight = segments_for("output_norm.weight").numpy("output_norm.weight").copy()
        norm.weight = torch.nn.Parameter(torch.from_numpy(weight).to(device, dtype), requires_grad=False)
        return norm

    def head():
        return torch.from_numpy(segments_for("output.weight").numpy("output.weight")).to(device, dtype)

    return embed, layer_at, final_norm, head


def passage_features(stats: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-passage summaries of one passage's per-token statistics."""
    lp, rank, ent, var = stats["lp"], stats["rank"], stats["entropy"], stats["var"]
    tail = lp[8:] if len(lp) > 16 else lp  # the first tokens depend most on the unknown prompt
    out = {"n_tokens": float(len(lp)), "lp_mean": lp.mean(), "lp_tail_mean": tail.mean(), "lp_std": lp.std(),
           "lp_min": lp.min(), "top1": (rank == 0).mean(), "top5": (rank < 5).mean(), "log_rank": np.log1p(rank).mean(),
           "entropy": ent.mean(), "gap_top": (stats["top"] - lp).mean(),
           "curvature": (lp + ent).sum() / np.sqrt(max(var.sum(), 1e-6))}
    out |= {f"lp_q{q}": np.quantile(lp, q / 100) for q in (10, 25, 50, 75, 90)}
    return {k: float(v) for k, v in out.items()}


__all__ = ["ARCHITECTURES", "Flash", "Glm53", "Segments", "batches_of", "passage_features", "run_experts", "score",
           "segment_model", "token_stats"]
