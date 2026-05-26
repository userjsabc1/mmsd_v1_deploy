# coding=utf-8
"""
MMSD v1 训练用 Draft Model。

基于 cnets_eagle3_llava.py，核心改动：
1. dataprepare(): 随机 3 层 hidden states + visual delta 计算 + 序列过滤
2. forward() loss: LK Loss 替换 KL Loss

架构不变：LlamaDecoderLayeremb (1-layer, dual-input) + fc(3D→D) + lm_head
"""
import math
from typing import List, Optional, Tuple, Union
from collections import Counter
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.activations import ACT2FN
from transformers import AutoTokenizer, LlavaForConditionalGeneration
from train.model.configs import EConfig
from safetensors import safe_open
from datasets import load_dataset
import multiprocessing

# --- visual_utils imports (MMSD v1 核心) ---
from method.mmsd_v1.visual_utils import (
    compute_visual_delta,
    select_random_visual_tokens,
    get_visual_positions,
    filter_sequence,
    collect_layer_hidden_states,
    get_random_layer_indices,
)
from method.mmsd_v1.lk_loss import lk_loss


# ============================================================
# 以下组件与 cnets_eagle3_llava.py 完全相同，不再重复注释
# ============================================================

def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")
        # dual-input: Q/K/V 的输入维度是 2*hidden_size (input_emb concat hidden_states)
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def forward(self, hidden_states, cache_hidden=None, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        lck = len(cache_hidden[0])
        cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids + lck)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        local_cache_k = list(cache_hidden[0])
        local_cache_v = list(cache_hidden[1])
        local_cache_k.append(key_states)
        local_cache_v.append(value_states)

        k0, v0 = local_cache_k[0], local_cache_v[0]
        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask

        for i in range(1, len(local_cache_k)):
            attn_weightsi = (query_states * local_cache_k[i]).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights[..., :q_len], v0)
        for i in range(1, len(local_cache_k)):
            attn_output = attn_output + attn_weights[..., q_len + i - 1][..., None] * local_cache_v[i]

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, [local_cache_k, local_cache_v]


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayeremb(nn.Module):
    """1-layer draft decoder，dual-input: (token_emb, hidden_states) → concat → QKV"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_emb, hidden_states, cache_hidden=None, attention_mask=None,
                position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)  # [B, T, 2D]
        return_hidden = hidden_states

        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden, hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, return_hidden), latest_hidden_cache


# ============================================================
# 以下是 MMSD v1 独有逻辑
# ============================================================

@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def process_data(data_chunk):
    token_dict = Counter()
    input_ids = data_chunk["input_ids"]
    loss_mask = data_chunk["loss_mask"]
    for i in range(len(input_ids)):
        ids = input_ids[i][0]
        mask = loss_mask[i][0]
        for j in range(len(ids)):
            if mask[j] == 1:
                token_dict[ids[j]] += 1
    return token_dict


def merge_dicts(dicts):
    result = Counter()
    for d in dicts:
        result.update(d)
    return result


class Model(nn.Module):
    """
    MMSD v1 训练用 Draft Model。

    架构（与 EAGLE-3 相同）：
    ┌─────────────────────────────────────────────────────┐
    │  target_model (LlavaForConditionalGeneration, frozen)│
    │  → output_hidden_states=True → (L+1) × [B, T+V, D]  │
    └────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────▼────────────────────────────────┐
    │  dataprepare() — MMSD v1 改动在这里                   │
    │  1. 随机选 3 层 hidden states → concat [B, T+V, 3D]   │
    │  2. compute_visual_delta → [V] importance scores      │
    │  3. select_random_visual_tokens(top64→rand32)         │
    │  4. filter_sequence → [B, T+K, 3D] (K=32)            │
    └────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────▼───────┐
    │  fc: Linear(3D → D)        │  融合 3 层信息
    └────────────────────┬───────┘
                         │
    ┌────────────────────▼───────────────────────────────┐
    │  7 步自回归展开 (idx=0..6):                          │
    │    embed_tokens(input_ids) → input_emb [B, T+K, D]  │
    │    midlayer(input_emb, hidden_states) → h_out        │
    │    lm_head(norm(h_out)) → draft_logits [B, T+K, V'] │
    │    LK_Loss(draft_logits, target_logits)              │
    │    shift input_ids/target for next step              │
    └────────────────────────────────────────────────────┘

    输入参数:
        input_ids:      [B, T+V] token IDs（含 image_token placeholder）
        attention_mask:  [B, T+V]
        loss_mask:       [B, T+V] 仅 assistant response 位置为 1
        pixel_values:    [B, C, H, W] 图像 tensor

    返回值:
        plosses: list of 7 scalar tensors, 每步的 LK Loss
        vlosses: [] (保留接口，未使用)
        acces:   list of 7 floats, 每步的 token 预测准确率
    """

    def __init__(self, config, ds_config, training_config, load_head=False, load_emb=True, path=None):
        super().__init__()
        self.train_config = training_config
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)

        # --- Draft model 可训练参数 ---
        self.midlayer = LlamaDecoderLayeremb(config)       # 1-layer decoder
        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)  # 3D → D
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        self.gradient_checkpointing = training_config["gradient_checkpointing"]
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size
        self.length = 7  # 自回归展开步数

        # --- MMSD v1 超参 ---
        self.num_visual_tokens = training_config.get("num_visual_tokens", 32)
        self.visual_pre_filter_k = training_config.get("visual_pre_filter_k", 64)
        self.lk_eta = training_config.get("lk_eta", 3.0)

        # --- Target model (frozen) ---
        self.target_model = LlavaForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16, output_hidden_states=True
        )
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        # image_token_id: LLaVA 用来标记 visual token 位置的特殊 token
        self._image_token_id = getattr(
            self.target_model.config, "image_token_index",
            getattr(self.target_model.config, "image_token_id", None)
        )
        self._num_layers = self.target_model.config.text_config.num_hidden_layers

        # --- Embedding (frozen, 从 target model 复制) ---
        if not load_emb:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        else:
            try:
                lm = self.target_model.language_model
                tensor = lm.model.embed_tokens.weight.float()
            except:
                import json as json_module
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json_module.loads(f.read())
                    emb_path = index_json["weight_map"].get(
                        "language_model.model.embed_tokens.weight",
                        index_json["weight_map"].get("model.embed_tokens.weight")
                    )
                with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
                    tensor = f.get_slice("language_model.model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor.get_shape()
                    tensor = tensor[:, :hidden_dim].float()
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, _weight=tensor)
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def scandata(self, datapath, tokenizerpath):
        """扫描训练数据统计 token 频率，构建 draft vocab 映射 (t2d, d2t)。与 eagle3 版本相同。"""
        N = self.draft_vocab_size
        if not os.path.exists("cache.pt"):
            tokenizer = AutoTokenizer.from_pretrained(tokenizerpath)
            dataset = load_dataset('json', data_files=datapath)['train']
            original_columns = dataset.column_names

            def preprocess_function(examples):
                new_examples = {"input_ids": [], "loss_mask": []}
                for i in range(len(examples['conversations'])):
                    roles = {"human": "user", "gpt": "assistant"}
                    source = examples['conversations'][i]
                    if not source:
                        continue
                    if roles.get(source[0]["from"]) != "user":
                        source = source[1:]
                    if not source:
                        continue
                    # 构建对话文本
                    text_parts = []
                    for sentence in source:
                        role = roles.get(sentence["from"], sentence["from"])
                        content = sentence["value"].replace("<image>", "").strip()
                        if role == "user":
                            text_parts.append(f"USER: {content}")
                        else:
                            text_parts.append(f"ASSISTANT: {content}")
                    conversation = " ".join(text_parts)

                    if not tokenizer.pad_token_id:
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    input_ids = tokenizer(conversation, return_tensors="pt", add_special_tokens=False).input_ids[0]
                    if len(input_ids) > self.train_config["max_len"]:
                        continue
                    loss_mask = torch.ones_like(input_ids)
                    # Mask non-assistant turns
                    sep_assistant = "ASSISTANT:"
                    sep = "</s>"
                    loss_mask[:] = 0
                    parts = conversation.split(sep_assistant)
                    cur_pos = 0
                    for pi, part in enumerate(parts):
                        if pi == 0:
                            cur_pos += len(tokenizer(part + sep_assistant, add_special_tokens=False).input_ids)
                        else:
                            response = part.split(sep)[0] if sep in part else part
                            resp_len = len(tokenizer(response, add_special_tokens=False).input_ids)
                            end_pos = min(cur_pos + resp_len, len(loss_mask))
                            loss_mask[cur_pos:end_pos] = 1
                            cur_pos += len(tokenizer(
                                part + (sep_assistant if pi < len(parts) - 1 else ""),
                                add_special_tokens=False
                            ).input_ids)
                    new_examples["input_ids"].append(input_ids[None, :])
                    new_examples["loss_mask"].append(loss_mask[None, :])
                return new_examples

            dataset = dataset.map(preprocess_function, batched=True, num_proc=8,
                                  remove_columns=original_columns, load_from_cache_file=False)
            chunk_size = len(dataset) // 8 + (len(dataset) % 8 > 0)
            chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
            with multiprocessing.Pool(8) as pool:
                results = pool.map(process_data, chunks)
            token_dict = merge_dicts(results)

            total_frequency = sum(token_dict.values())
            top_N = token_dict.most_common(N)
            top_N_frequency_sum = sum(freq for _, freq in top_N)
            print(f"top {N} token frequency ratio: {top_N_frequency_sum / total_frequency:.2%}")
            used_tokens = sorted([key for key, _ in top_N])
            if len(used_tokens) < N:
                used_set = set(used_tokens)
                for tid in range(self.vocab_size):
                    if tid not in used_set:
                        used_tokens.append(tid)
                    if len(used_tokens) == N:
                        break
                used_tokens.sort()
            d2t = torch.tensor([used_tokens[i] - i for i in range(len(used_tokens))])
            t2d = torch.tensor([i in used_tokens for i in range(self.vocab_size)])
            torch.save({"d2t": d2t, "t2d": t2d}, "cache.pt")
        else:
            cache = torch.load("cache.pt")
            d2t, t2d = cache["d2t"], cache["t2d"]
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    # ================================================================
    # dataprepare: MMSD v1 核心改动
    # ================================================================
    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask, loss_mask, pixel_values=None):
        """
        运行 target model，提取 hidden states 并执行 MMSD v1 的 visual token 压缩。

        流程：
        1. target_model forward → all_hidden_states: (L+1) × [B, seq, D], logits: [B, seq, V_full]
        2. 随机选 3 层 hidden states → concat [B, seq, 3D]
        3. 如果有 visual tokens (pixel_values != None):
           a. compute_visual_delta → [V] importance scores
           b. select_random_visual_tokens(delta, top_k=32, pre_filter_k=64) → [32] indices
           c. filter_sequence → hidden_states [B, T+32, 3D], target [B, T+32, V_full]
        4. shift: target 和 input_ids 左移一位（对齐 next-token prediction）

        输入:
            input_ids:      [B, T+V]
            attention_mask:  [B, T+V]
            loss_mask:       [B, T+V]
            pixel_values:    [B, C, H, W] or None

        输出:
            hidden_states:  [B, S, 3D]  (S = T+K, K=32 visual tokens; 无图则 S=T+V)
            target:         [B, S, V_full]  shifted target logits
            loss_mask:      [B, S, 1]       shifted & filtered
            input_ids:      [B, S]          shifted & filtered
        """
        device = input_ids.device

        # 1. Target model forward
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values.to(device)
        outs = self.target_model(**kwargs)
        all_hs = outs.hidden_states  # tuple of (L+1) tensors, each [B, seq, D]
        target = outs.logits         # [B, seq, V_full]

        # 2. 随机 3 层 hidden states (训练时数据增强)
        layer_indices = get_random_layer_indices(self._num_layers, num_select=3)
        hidden_states = collect_layer_hidden_states(all_hs, layer_indices)  # [B, seq, 3D]

        # 3. Visual token 压缩 (仅当有 visual tokens 时)
        if pixel_values is not None and self._image_token_id is not None:
            visual_pos = get_visual_positions(input_ids, self._image_token_id)
            if len(visual_pos) > self.num_visual_tokens:
                # a. 跨层 cosine distance → 每个 visual token 的重要性分数
                delta = compute_visual_delta(list(all_hs), visual_pos)
                # b. 从 top-64 中随机选 32 个 (训练时数据增强)
                selected_idx = select_random_visual_tokens(
                    delta, top_k=self.num_visual_tokens, pre_filter_k=self.visual_pre_filter_k
                )
                # c. 过滤序列：保留全部 text tokens + 选中的 32 个 visual tokens
                total_seq_len = hidden_states.shape[1]
                hidden_states, keep_pos = filter_sequence(
                    hidden_states, visual_pos, selected_idx, total_seq_len
                )
                target = target[:, keep_pos]
                # 同步过滤 input_ids, attention_mask, loss_mask
                input_ids = input_ids[:, keep_pos]
                loss_mask = loss_mask[:, keep_pos]

        # 4. Shift: next-token prediction 对齐
        target = padding(target, left=False)   # 左移一位
        input_ids = padding(input_ids, left=False)

        target = target.to(device)
        loss_mask = loss_mask[..., None].to(device)  # [B, S, 1]

        return hidden_states, target, loss_mask, input_ids

    # ================================================================
    # forward: 7 步自回归展开 + LK Loss
    # ================================================================
    def forward(
            self,
            input_ids,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            loss_mask=None,
            pixel_values=None,
    ):
        """
        训练 forward pass。

        输入:
            input_ids:      [B, T+V] 含 image_token placeholder
            attention_mask:  [B, T+V]
            loss_mask:       [B, T+V] 仅 assistant response 位置为 1
            pixel_values:    [B, C, H, W] 图像

        输出:
            plosses: list[Tensor], 长度 7, 每步的 LK Loss (scalar)
            vlosses: list[], 空 (保留接口兼容)
            acces:   list[float], 长度 7, 每步 token 预测准确率
        """
        # --- dataprepare: target forward + 随机层采样 + visual 压缩 ---
        hidden_states, target, loss_mask, input_ids = self.dataprepare(
            input_ids, attention_mask, loss_mask, pixel_values=pixel_values
        )
        # hidden_states: [B, S, 3D],  S = T + K (K=32 visual or all)
        # target:        [B, S, V_full] (shifted)
        # loss_mask:     [B, S, 1]
        # input_ids:     [B, S] (shifted)

        batch_size, seq_length, _ = hidden_states.shape

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

        # --- fc: 3D → D ---
        hidden_states = self.fc(hidden_states)  # [B, S, D]

        # --- 构建 attention mask 和 position_ids ---
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        attn_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=hidden_states.device)
        attn_mask = self._prepare_decoder_attention_mask(
            attn_mask, (batch_size, seq_length), hidden_states, 0
        )

        # --- 7 步自回归展开 ---
        plosses = []
        vlosses = []
        acces = []
        cache_hidden = [[], []]

        for idx in range(self.length):
            last = idx == self.length - 1

            # token embedding (frozen)
            inputs_embeds = self.embed_tokens(input_ids).to(hidden_states.dtype)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True

            # 1-layer draft decoder: dual-input (input_emb, hidden_states) → h_out
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, None, output_attentions)
                    return custom_forward
                layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.midlayer),
                    inputs_embeds, hidden_states, cache_hidden, attn_mask, position_ids,
                )
            else:
                layer_outputs, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds, hidden_states=hidden_states, cache_hidden=cache_hidden,
                    attention_mask=attn_mask, position_ids=position_ids,
                    past_key_value=None, output_attentions=output_attentions, use_cache=True,
                )

            hidden_states_out = layer_outputs[0]  # [B, S, D]

            # --- Draft logits ---
            logits = self.lm_head(self.norm(hidden_states_out))  # [B, S, V_draft]

            # --- LK Loss (替换原版 KL Loss) ---
            with torch.no_grad():
                target_head = target
                target_max_token = target_head.argmax(-1)
                self.t2d = self.t2d.to(target_max_token.device)
                target_mask = self.t2d[target_max_token][..., None].int()
                position_mask = target_mask * loss_mask  # [B, S, 1]
                # 映射 target logits 到 draft vocab
                target_head_draft = target_head[..., self.t2d].float()

            logits_float = logits.float()

            # LK Loss: λ·KL(p||q) + (1-λ)·TV(p,q), λ = exp(-η·α)
            step_loss = lk_loss(
                draft_logits=logits_float,
                target_logits=target_head_draft,
                eta=self.lk_eta,
                loss_mask=position_mask.squeeze(-1),
            )
            plosses.append(step_loss)

            # --- Accuracy (monitoring) ---
            with torch.no_grad():
                target_p = nn.Softmax(dim=2)(target_head_draft)
                correct = (logits_float.argmax(-1) == target_p.argmax(-1))
                acc = (correct * position_mask.squeeze(-1)).sum().item() / (loss_mask.sum().item() + 1e-6)
                acces.append(acc)

            # --- 准备下一步 ---
            hidden_states = hidden_states_out
            if not last:
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)

        return plosses, vlosses, acces
