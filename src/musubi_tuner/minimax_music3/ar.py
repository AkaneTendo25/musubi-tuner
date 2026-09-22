"""Native MiniMax Music 3 autoregressive conditioning stage."""

import re
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

AUDIO_END = 151670
AUDIO_CFG = 151654
AUDIO_OFFSET = 151675
SEMANTIC_VOCAB = 16384
AR_CFG_SCALE = 1.5
AR_TOP_K = 50
FRAMES_PER_SECOND = 25
SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def derive_seed(seed: int, *parts: str) -> int:
    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


class DepthBlock(nn.Module):
    def __init__(self, dim=4096, heads=16, intermediate=6144):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.input_layernorm = RMSNorm(dim, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(dim, eps=1e-6)
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(dim, dim, bias=False)
        self.attn.to_k = nn.Linear(dim, dim, bias=False)
        self.attn.to_v = nn.Linear(dim, dim, bias=False)
        self.attn.to_out = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x):
        norm = self.input_layernorm(x)
        shape = (*norm.shape[:2], self.heads, self.head_dim)
        q, k, v = (tensor.view(shape).transpose(1, 2) for tensor in self.attn.qkv_proj(norm).chunk(3, dim=-1))
        mask = torch.full((norm.shape[1], norm.shape[1]), torch.finfo(norm.dtype).min,
                          device=norm.device, dtype=norm.dtype).triu_(1)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).flatten(2)
        x = x + self.attn.to_out(attention)
        norm = self.post_attention_layernorm(x)
        gate, up = self.gate_up_proj(norm).chunk(2, dim=-1)
        return x + self.down_proj(F.silu(gate).mul_(up))


class RVQDepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_embeddings = nn.Embedding(1024 * 7, 4096)
        self.projection = nn.Linear(4096, 4096, bias=False)
        self.pos_embedding = nn.Embedding(16, 4096)
        self.layers = nn.ModuleList([DepthBlock() for _ in range(4)])
        self.norm = RMSNorm(4096, eps=1e-6)
        self.audio_heads = nn.ModuleList([nn.Linear(4096, 1024, bias=False) for _ in range(7)])

    def forward(self, x):
        x = x + self.pos_embedding(torch.arange(x.shape[1], device=x.device))[None]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class RMSNorm(nn.Module):
    """RMSNorm used by the Comfy-format AR checkpoint."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), self.weight, self.eps)


def _use_fused_qwen_rms_norm(model: nn.Module) -> None:
    def fused_forward(module, hidden_states):
        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        return F.rms_norm(hidden_states, (hidden_states.shape[-1],), module.weight, eps)

    for module in model.modules():
        if module.__class__.__name__ == "Qwen3RMSNorm":
            module.forward = fused_forward.__get__(module, module.__class__)


def _use_fused_qwen_projections(model: nn.Module) -> None:
    def rotary_forward(module, hidden_states, position_ids):
        head_dim = model.config.head_dim
        indices = torch.arange(0, head_dim, 2, device=hidden_states.device, dtype=torch.float32)
        rope_theta = getattr(model.config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = model.config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (float(rope_theta) ** (indices / head_dim))
        frequencies = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos(), embedding.sin()

    def attention_forward(module, hidden_states, position_embeddings, attention_mask,
                          past_key_values=None, cache_position=None, **kwargs):
        batch, length, _ = hidden_states.shape
        execution_dtype = hidden_states.dtype
        q_size = module.config.num_attention_heads * module.head_dim
        kv_size = module.config.num_key_value_heads * module.head_dim
        query, key, value = module.qkv_proj(hidden_states).split((q_size, kv_size, kv_size), dim=-1)
        query = module.q_norm(query.view(batch, length, -1, module.head_dim)).transpose(1, 2)
        key = module.k_norm(key.view(batch, length, -1, module.head_dim)).transpose(1, 2)
        value = value.view(batch, length, -1, module.head_dim).transpose(1, 2)
        cos, sin = (tensor.unsqueeze(1) for tensor in position_embeddings)
        query_rotated = query * cos
        key_rotated = key * cos
        half = module.head_dim // 2
        query_rotated[..., :half].addcmul_(query[..., half:], -sin[..., :half])
        query_rotated[..., half:].addcmul_(query[..., :half], sin[..., half:])
        key_rotated[..., :half].addcmul_(key[..., half:], -sin[..., :half])
        key_rotated[..., half:].addcmul_(key[..., :half], sin[..., half:])
        query, key = query_rotated.to(execution_dtype), key_rotated.to(execution_dtype)
        if past_key_values is not None:
            key, value = past_key_values.update(
                key, value, module.layer_idx,
                {"sin": position_embeddings[1], "cos": position_embeddings[0], "cache_position": cache_position},
            )
        mask = attention_mask
        if mask is not None:
            mask = mask[..., :query.shape[-2], :key.shape[-2]]
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0,
            is_causal=mask is None and query.shape[-2] > 1,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        output = module.o_proj(output.transpose(1, 2).reshape(batch, length, -1).contiguous())
        return output, None

    def mlp_forward(module, hidden_states):
        gate, up = module.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return module.down_proj(F.silu(gate).mul_(up))

    for layer in model.model.layers:
        attention = layer.self_attn
        qkv = torch.cat((attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight), dim=0)
        attention.qkv_proj = nn.Linear(qkv.shape[1], qkv.shape[0], bias=False,
                                       device=qkv.device, dtype=qkv.dtype)
        attention.qkv_proj.weight.data.copy_(qkv)
        del attention.q_proj, attention.k_proj, attention.v_proj
        attention.forward = attention_forward.__get__(attention, attention.__class__)

        mlp = layer.mlp
        gate_up = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0)
        mlp.gate_up_proj = nn.Linear(gate_up.shape[1], gate_up.shape[0], bias=False,
                                     device=gate_up.device, dtype=gate_up.dtype)
        mlp.gate_up_proj.weight.data.copy_(gate_up)
        del mlp.gate_proj, mlp.up_proj
        mlp.forward = mlp_forward.__get__(mlp, mlp.__class__)
    model.model.rotary_emb.forward = rotary_forward.__get__(model.model.rotary_emb,
                                                             model.model.rotary_emb.__class__)


def _use_fused_depth_projections(decoder: RVQDepthDecoder) -> None:
    for layer in decoder.layers:
        attention = layer.attn
        qkv = torch.cat((attention.to_q.weight, attention.to_k.weight, attention.to_v.weight), dim=0)
        attention.qkv_proj = nn.Linear(qkv.shape[1], qkv.shape[0], bias=False,
                                       device=qkv.device, dtype=qkv.dtype)
        attention.qkv_proj.weight.data.copy_(qkv)
        del attention.to_q, attention.to_k, attention.to_v
        gate_up = torch.cat((layer.gate_proj.weight, layer.up_proj.weight), dim=0)
        layer.gate_up_proj = nn.Linear(gate_up.shape[1], gate_up.shape[0], bias=False,
                                       device=gate_up.device, dtype=gate_up.dtype)
        layer.gate_up_proj.weight.data.copy_(gate_up)
        del layer.gate_proj, layer.up_proj


def _sample(logits, generator, top_k=50):
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, min(top_k, values.shape[-1]), dim=-1).values[..., -1, None]
    probabilities = torch.nan_to_num(torch.softmax(values.masked_fill(values < threshold, -torch.inf), dim=-1), nan=0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


def _clean_caption(caption):
    def rewrite(match):
        parts = match.group(1).strip().split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else parts[0]
    lines = []
    for line in SPECIAL_TAG_RE.sub(rewrite, caption).splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        lines.append(re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line).rstrip())
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", "\n".join(lines), flags=re.MULTILINE)
    return re.sub(r"\n{2,}", "\n", text.replace("• ", "").replace("    ", ""))


def _normalize_lyrics(lyrics):
    lines = []
    for line in lyrics.split("\n"):
        match = LEADING_TAGS_RE.match(line)
        lines.append(match.group(1).strip() if match else line)
    text = "\n".join(lines).replace("] ", "]\n").replace(" [", "\n[").replace(" ^ ", "\n")
    return "[start]\n" + re.sub(r"\[([^]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)


def build_prompt(caption, lyrics):
    return (f"<|im_start|><|caption_start|>{_clean_caption(caption)}<|caption_end|><|lyrics_start|>"
            f"{_normalize_lyrics(lyrics)}<|lyrics_end|><|im_end|><|audio_start|>")


def load_depth_decoder(repo_or_path, device="cuda", dtype=torch.bfloat16, fuse=True):
    decoder = RVQDepthDecoder()
    path = Path(repo_or_path) / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo_or_path, "rvq_depth_decoder/diffusion_pytorch_model.safetensors"))
    decoder.load_state_dict(load_file(path), strict=True)
    decoder = decoder.to(device=device, dtype=dtype).eval()
    if fuse:
        _use_fused_depth_projections(decoder)
    return decoder


def load_ar(repo_or_path, device="cuda", dtype=torch.bfloat16, lora_path=None):
    language_model = AutoModelForCausalLM.from_pretrained(
        repo_or_path, subfolder="language_model", torch_dtype=dtype, low_cpu_mem_usage=True
    )
    if lora_path:
        from musubi_tuner.minimax_music3.ar_lora import load_and_merge_lora

        load_and_merge_lora(language_model, lora_path)
    language_model = language_model.to(device).eval()
    _use_fused_qwen_rms_norm(language_model)
    _use_fused_qwen_projections(language_model)
    tokenizer = AutoTokenizer.from_pretrained(repo_or_path, subfolder="tokenizer")
    decoder = load_depth_decoder(repo_or_path, device=device, dtype=dtype)
    return language_model, decoder, tokenizer


@torch.inference_mode()
def generate_conditioning(language_model, decoder, tokenizer, caption, lyrics, frames, seed=0,
                          cfg_scale=AR_CFG_SCALE, top_k=AR_TOP_K,
                          generator: torch.Generator | None = None,
                          return_codes: bool = False):
    device = language_model.device
    ids = tokenizer(build_prompt(caption, lyrics), return_tensors="pt").input_ids.to(device)
    unconditional = ids.clone()
    unconditional[:, 1:-2] = AUDIO_CFG
    ids = torch.cat((ids, unconditional))
    prompt_embeds = language_model.model.embed_tokens(ids).to(language_model.dtype)
    prompt_length = ids.shape[1]
    cache = DynamicCache()
    causal_mask = torch.full(
        (prompt_length, prompt_length),
        torch.finfo(prompt_embeds.dtype).min / 4,
        device=device,
        dtype=prompt_embeds.dtype,
    ).triu_(1)
    causal_mask = causal_mask[None, None].expand(ids.shape[0], 1, prompt_length, prompt_length)
    output = language_model.model(
        inputs_embeds=prompt_embeds,
        attention_mask=causal_mask,
        past_key_values=cache,
        cache_position=torch.arange(prompt_length, device=device),
        use_cache=True,
    )
    hidden, cache = output.last_hidden_state[:, -1], output.past_key_values
    if generator is None:
        generator = torch.Generator(device=device).manual_seed(derive_seed(seed, "ar"))
    result = []
    generated_codes = []
    for frame_index in range(frames + 1):
        logits = language_model.lm_head(hidden).float()
        logits = torch.cat((logits[..., AUDIO_END:AUDIO_END + 1],
                            logits[..., AUDIO_OFFSET:AUDIO_OFFSET + SEMANTIC_VOCAB]), dim=-1)
        conditional, unconditional = logits[:1], logits[1:]
        guided = unconditional + cfg_scale * (conditional - unconditional)
        threshold = torch.topk(conditional, top_k, dim=-1).values[..., -1, None]
        guided = guided.masked_fill(conditional < threshold, -torch.inf)
        token = _sample(guided, generator, top_k)
        if token.item() == 0:
            break
        semantic = (token - 1).repeat(2)
        sequence = [decoder.projection(hidden).unsqueeze(1)]
        semantic_embed = language_model.model.embed_tokens(semantic + AUDIO_OFFSET)
        sequence.append(decoder.projection(semantic_embed).unsqueeze(1))
        codes, depth_hidden = [semantic], []
        for index in range(1, 8):
            local = decoder(torch.cat(sequence, dim=1))[:, -1]
            depth_hidden.append(local[:1])
            local_logits = decoder.audio_heads[index - 1](local).float()
            code = _sample(local_logits[1:] + cfg_scale * (local_logits[:1] - local_logits[1:]), generator, top_k).repeat(2)
            codes.append(code)
            if index < 7:
                sequence.append(decoder.projection(decoder.audio_embeddings(code + (index - 1) * 1024)).unsqueeze(1))
        frame_codes = torch.stack(codes, dim=1)
        if frame_index:
            result.append(torch.cat((hidden[:1], torch.cat(depth_hidden, dim=-1)), dim=-1)[0].cpu())
            generated_codes.append(frame_codes[0].cpu())
        offsets = torch.arange(7, device=device) * 1024
        feedback = language_model.model.embed_tokens(frame_codes[:, :1] + AUDIO_OFFSET)
        feedback += decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        output = language_model.model(
            inputs_embeds=feedback * (8 ** -0.5),
            past_key_values=cache,
            cache_position=torch.tensor([prompt_length + frame_index], device=device),
            use_cache=True,
        )
        hidden, cache = output.last_hidden_state[:, -1], output.past_key_values
        if len(result) >= frames:
            break
    if not result:
        raise RuntimeError("The AR stage produced no audio frames")
    conditioning = torch.stack(result)
    if return_codes:
        return conditioning, torch.stack(generated_codes)
    return conditioning


@torch.inference_mode()
def teacher_force_conditioning(language_model, decoder, tokenizer, caption, lyrics, codes,
                               skip_first: bool = True):
    """Rebuild DiT conditioning from a known RVQ sequence without sampling codes."""
    if codes.ndim != 2 or codes.shape[1] != 8:
        raise ValueError(f"Expected RVQ codes [frames,8], got {tuple(codes.shape)}")
    device = language_model.device
    codes = codes.to(device=device, dtype=torch.long)
    if codes[:, 0].min() < 0 or codes[:, 0].max() >= SEMANTIC_VOCAB:
        raise ValueError("Semantic RVQ codes are outside [0,16384)")
    if codes[:, 1:].min() < 0 or codes[:, 1:].max() >= 1024:
        raise ValueError("Depth RVQ codes are outside [0,1024)")

    ids = tokenizer(build_prompt(caption, lyrics), return_tensors="pt").input_ids.to(device)
    prompt_embeds = language_model.model.embed_tokens(ids).to(language_model.dtype)
    prompt_length = ids.shape[1]
    cache = DynamicCache()
    causal_mask = torch.full(
        (prompt_length, prompt_length),
        torch.finfo(prompt_embeds.dtype).min / 4,
        device=device,
        dtype=prompt_embeds.dtype,
    ).triu_(1)
    output = language_model.model(
        inputs_embeds=prompt_embeds,
        attention_mask=causal_mask[None, None],
        past_key_values=cache,
        cache_position=torch.arange(prompt_length, device=device),
        use_cache=True,
    )
    hidden, cache = output.last_hidden_state[:, -1], output.past_key_values
    result = []
    offsets = torch.arange(7, device=device) * 1024
    for frame_index, frame_codes in enumerate(codes):
        semantic = frame_codes[0].view(1)
        sequence = [decoder.projection(hidden).unsqueeze(1)]
        semantic_embed = language_model.model.embed_tokens(semantic + AUDIO_OFFSET)
        sequence.append(decoder.projection(semantic_embed).unsqueeze(1))
        depth_hidden = []
        for index in range(1, 8):
            local = decoder(torch.cat(sequence, dim=1))[:, -1]
            depth_hidden.append(local)
            if index < 7:
                code = frame_codes[index].view(1)
                embedding = decoder.audio_embeddings(code + (index - 1) * 1024)
                sequence.append(decoder.projection(embedding).unsqueeze(1))
        if not skip_first or frame_index > 0:
            result.append(torch.cat((hidden, torch.cat(depth_hidden, dim=-1)), dim=-1)[0].cpu())
        feedback = language_model.model.embed_tokens(semantic.view(1, 1) + AUDIO_OFFSET)
        feedback += decoder.audio_embeddings(frame_codes[1:].view(1, 7) + offsets).sum(dim=1, keepdim=True)
        output = language_model.model(
            inputs_embeds=feedback * (8 ** -0.5),
            past_key_values=cache,
            cache_position=torch.tensor([prompt_length + frame_index], device=device),
            use_cache=True,
        )
        hidden, cache = output.last_hidden_state[:, -1], output.past_key_values
    if not result:
        raise ValueError("At least two RVQ frames are required when skip_first=True")
    return torch.stack(result)


@torch.inference_mode()
def decode_codes_with_prior(
    language_model,
    decoder,
    tokenizer,
    caption: str,
    lyrics: str,
    semantic_logits: torch.Tensor,
    depth_logits: list[torch.Tensor],
    prior_weight: float,
    depth_prior_weight: float | None = None,
) -> torch.Tensor:
    """Decode encoder emissions under the released AR and depth-code priors."""
    if semantic_logits.ndim != 2 or semantic_logits.shape[1] != SEMANTIC_VOCAB:
        raise ValueError(f"Expected semantic logits [frames,{SEMANTIC_VOCAB}], got {tuple(semantic_logits.shape)}")
    if len(depth_logits) != 7 or any(logits.shape != (semantic_logits.shape[0], 1024) for logits in depth_logits):
        raise ValueError("Expected seven depth-logit tensors shaped [frames,1024]")
    depth_prior_weight = prior_weight if depth_prior_weight is None else depth_prior_weight
    if prior_weight < 0 or depth_prior_weight < 0:
        raise ValueError("prior weights must be non-negative")

    device = language_model.device
    ids = tokenizer(build_prompt(caption, lyrics), return_tensors="pt").input_ids.to(device)
    prompt_embeds = language_model.model.embed_tokens(ids).to(language_model.dtype)
    prompt_length = ids.shape[1]
    cache = DynamicCache()
    causal_mask = torch.full(
        (prompt_length, prompt_length),
        torch.finfo(prompt_embeds.dtype).min / 4,
        device=device,
        dtype=prompt_embeds.dtype,
    ).triu_(1)
    output = language_model.model(
        inputs_embeds=prompt_embeds,
        attention_mask=causal_mask[None, None],
        past_key_values=cache,
        cache_position=torch.arange(prompt_length, device=device),
        use_cache=True,
    )
    hidden, cache = output.last_hidden_state[:, -1], output.past_key_values
    offsets = torch.arange(7, device=device) * 1024
    result = []
    for frame_index in range(semantic_logits.shape[0]):
        semantic_prior = language_model.lm_head(hidden)[0, AUDIO_OFFSET : AUDIO_OFFSET + SEMANTIC_VOCAB].float()
        semantic_score = F.log_softmax(semantic_logits[frame_index].to(device).float(), dim=-1)
        semantic_score.add_(F.log_softmax(semantic_prior, dim=-1), alpha=prior_weight)
        semantic = semantic_score.argmax().view(1)

        sequence = [decoder.projection(hidden).unsqueeze(1)]
        semantic_embedding = language_model.model.embed_tokens(semantic + AUDIO_OFFSET)
        sequence.append(decoder.projection(semantic_embedding).unsqueeze(1))
        codes = [semantic]
        for index in range(7):
            local = decoder(torch.cat(sequence, dim=1))[:, -1]
            prior = decoder.audio_heads[index](local)[0].float()
            score = F.log_softmax(depth_logits[index][frame_index].to(device).float(), dim=-1)
            score.add_(F.log_softmax(prior, dim=-1), alpha=depth_prior_weight)
            code = score.argmax().view(1)
            codes.append(code)
            if index < 6:
                embedding = decoder.audio_embeddings(code + index * 1024)
                sequence.append(decoder.projection(embedding).unsqueeze(1))
        frame_codes = torch.stack(codes, dim=1)
        result.append(frame_codes[0].cpu())
        feedback = language_model.model.embed_tokens(frame_codes[:, :1] + AUDIO_OFFSET)
        feedback += decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
        output = language_model.model(
            inputs_embeds=feedback * (8**-0.5),
            past_key_values=cache,
            cache_position=torch.tensor([prompt_length + frame_index], device=device),
            use_cache=True,
        )
        hidden, cache = output.last_hidden_state[:, -1], output.past_key_values
    return torch.stack(result)
