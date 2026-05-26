import copy
import json
import os
import time
import warnings
from typing import Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from .cnets import Model
from .configs import EConfig
from .kv_cache import initialize_past_key_values
from .utils import *
from .visual_utils import (
    compute_visual_delta,
    select_top_visual_tokens,
    get_visual_positions,
    filter_sequence,
    collect_layer_hidden_states,
    get_default_layer_indices,
)

# Lazy import for architecture-specific models
def _get_model_class(arch_type):
    """Lazy import model classes based on architecture type."""
    if arch_type == "LlamaForCausalLM":
        from .modeling_llama_kv import LlamaForCausalLM
        return LlamaForCausalLM
    elif arch_type == "Qwen2ForCausalLM":
        from .modeling_qwen2_kv import LlamaForCausalLM
        return LlamaForCausalLM
    elif arch_type == "MixtralForCausalLM":
        from .modeling_mixtral_kv import MixtralForCausalLM
        return MixtralForCausalLM
    elif arch_type == "LlavaNextForConditionalGeneration":
        from method.llava_adapter import CustomLlavaNextForConditionalGeneration
        return CustomLlavaNextForConditionalGeneration
    elif arch_type == "LlavaForConditionalGeneration":
        from method.llava_adapter import CustomLlavaForConditionalGeneration
        return CustomLlavaForConditionalGeneration
    elif arch_type == "Qwen2_5_VLForConditionalGeneration":
        from .modeling_qwen2_5_vl_kv import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    else:
        raise NotImplementedError(f"Model type {arch_type} is not supported.")


def _normalize_token_ids(token_ids):
    if token_ids is None:
        return []
    if isinstance(token_ids, (list, tuple, set)):
        raw_ids = token_ids
    else:
        raw_ids = [token_ids]
    normalized = []
    for token_id in raw_ids:
        if token_id is None:
            continue
        if isinstance(token_id, torch.Tensor):
            if token_id.numel() == 1:
                token_id = token_id.item()
            else:
                normalized.extend(int(x) for x in token_id.view(-1).tolist())
                continue
        try:
            normalized.append(int(token_id))
        except (TypeError, ValueError):
            continue
    return normalized


def _collect_stop_token_ids(tokenizer, model):
    stop_ids = set(_normalize_token_ids(getattr(tokenizer, "eos_token_id", None)))
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        stop_ids.update(
            _normalize_token_ids(getattr(generation_config, "eos_token_id", None))
        )
    config = getattr(model, "config", None)
    if config is not None:
        stop_ids.update(_normalize_token_ids(getattr(config, "eos_token_id", None)))
    stop_ids.update(_normalize_token_ids(getattr(tokenizer, "eod_id", None)))
    return stop_ids


def _has_stop_token(generated_ids, stop_ids):
    if not stop_ids:
        return False
    if isinstance(generated_ids, torch.Tensor):
        generated_set = set(int(x) for x in generated_ids.view(-1).tolist())
    else:
        generated_set = set(int(x) for x in generated_ids)
    return bool(generated_set & stop_ids)


def _normalize_eagle3_checkpoint_keys(state_dict):
    normalized = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith("model."):
            k = k[len("model.") :]
        if k.startswith("layers.0."):
            k = "midlayer." + k[len("layers.0.") :]
        normalized[k] = v
    return normalized


class SpecModel(nn.Module):

    def __init__(
        self,
        base_model,
        base_model_name_or_path,
        spec_model_path,
        total_token,
        depth,
        top_k,
        threshold,
        spec_layer_state_dict,
        num_visual_tokens=32,
        visual_pre_filter_k=64,
        layer_indices=None,
    ):

        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # --- MMSD v1 config ---
        self.num_visual_tokens = num_visual_tokens
        self.visual_pre_filter_k = visual_pre_filter_k
        # Determine number of decoder layers
        _lang = getattr(base_model, "language_model", base_model)
        _num_layers = _lang.config.num_hidden_layers
        self.layer_indices = layer_indices or get_default_layer_indices(_num_layers)
        self._image_token_id = getattr(
            base_model.config, "image_token_index",
            getattr(base_model.config, "image_token_id", None)
        )

        if hasattr(base_model, "language_model"):
            base_model = base_model.language_model

        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name_or_path, use_fast=False
        )
        config = EConfig.from_pretrained(spec_model_path)
        with open(spec_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="You are using a model of type")
            self.spec_layer = Model(
                config,
                load_emb=True,
                path=self.base_model_name_or_path,
                bias=bias,
                total_tokens=total_token,
                depth=depth,
                top_k=top_k,
                threshold=threshold,
            )

        low_memory = False

        device = base_model.model.layers[-1].self_attn.q_proj.weight.device
        if device != base_model.lm_head.weight.device:
            self.spec_layer.diff_device = True
            if not low_memory:
                self.spec_layer.headweight = base_model.lm_head.weight.clone().to(
                    device
                )
            else:
                self.spec_layer.layer_device = device

        else:
            self.spec_layer.diff_device = False
        if spec_layer_state_dict is not None:
            # Remove bare 'weight' key (duplicate of embed_tokens.weight in some checkpoints)
            spec_layer_state_dict.pop("weight", None)
            spec_layer_state_dict = _normalize_eagle3_checkpoint_keys(
                spec_layer_state_dict
            )
            # Shape validation (防呆校验): every checkpoint key that exists in
            # the model must have exactly the same shape — no silent reshape.
            model_sd = self.spec_layer.state_dict()
            for k, v in spec_layer_state_dict.items():
                if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape):
                    raise RuntimeError(
                        f"[eagle3] Shape mismatch for '{k}': "
                        f"checkpoint={tuple(v.shape)}, model={tuple(model_sd[k].shape)}. "
                        f"Please check that the spec config matches the checkpoint."
                    )
            missing_keys, unexpected_keys = self.spec_layer.load_state_dict(
                spec_layer_state_dict, strict=False
            )
            # embed_tokens.weight is expected missing when loaded from base model
            missing_keys = [k for k in missing_keys if k != "embed_tokens.weight"]
            if len(missing_keys) > 0:
                print(f"missing_keys: {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"unexpected_keys: {unexpected_keys}")
        self.spec_layer.to(self.base_model.dtype).to(device)
        self.spec_layer.init_tree()

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        Type="LLaMA",
        base_model_path=None,
        spec_model_path=None,
        total_token=30,
        depth=3,
        top_k=8,
        threshold=1.0,
        num_visual_tokens=32,
        visual_pre_filter_k=64,
        layer_indices=None,
        **kwargs,
    ):
        # assert Type=="LLaMA" or "Mixtral"
        Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        model_class = _get_model_class(Type)
        base_model = model_class.from_pretrained(base_model_path, **kwargs)

        configpath = os.path.join(spec_model_path, "config.json")
        if not os.path.exists(configpath):
            if "Llava" in Type:
                fallback = "train/configs/llava1.5_7b_eagle3_config.json"
            else:
                fallback = "train/configs/qwen2.5vl_eagle3_config.json"
            if os.path.exists(fallback):
                configpath = fallback
            else:
                configpath = hf_hub_download(spec_model_path, "config.json")

        try:
            load_model_path = os.path.join(spec_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(spec_model_path, "pytorch_model.bin")
            spec_layer_state_dict = torch.load(
                load_model_path, map_location=base_model.device
            )
        except:
            from safetensors.torch import load, load_file

            load_model_path = os.path.join(spec_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                load_model_path = hf_hub_download(spec_model_path, "model.safetensors")
            with open(load_model_path, "rb") as f:
                spec_layer_state_dict = load(f.read())
        # spec_layer_state_dict = None
        model = cls(
            base_model,
            base_model_path,
            configpath,
            total_token,
            depth,
            top_k,
            threshold,
            spec_layer_state_dict,
            num_visual_tokens=num_visual_tokens,
            visual_pre_filter_k=visual_pre_filter_k,
            layer_indices=layer_indices,
        )

        if total_token == -1:
            device = model.base_model.model.layers[0].self_attn.q_proj.weight.device
            cans = [40, 48, 50, 56, 60]
            x = [1, 1.05, 1.07, 1.1, 1.13]
            times = []

            for i in range(len(cans)):
                length = cans[i]
                input_ids = torch.randint(
                    0, model.config.vocab_size - 200, (1, length)
                ).to(device)
                torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(20):
                    torch.cuda.synchronize()
                    with torch.no_grad():
                        outputs = model.base_model(input_ids)
                    torch.cuda.synchronize()
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) / x[i])
            total_token = cans[times.index(min(times))]
            model.spec_layer.total_tokens = total_token - 1

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        inputs_embeds=None,
        output_real_hidden=False,
        **kwargs,
    ):
        if (
            inputs_embeds is not None
            and self.base_model.config.architectures[0]
            == "LlavaNextForConditionalGeneration"
        ):
            input_ids = None
            kwargs = {}

        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True,
                output_hidden_states=True,
                **kwargs,
            )
            if output_orig:
                orig = outputs.logits
            all_hs = outputs.hidden_states  # (L+1) x [B, T+V, D]

            # --- MMSD v1: collect hidden states from configured layers ---
            hidden_states = collect_layer_hidden_states(all_hs, self.layer_indices)
            # hidden_states: [B, T+V, 3D]

            # --- MMSD v1: visual token delta computation & filtering ---
            if self._image_token_id is not None and input_ids is not None:
                visual_pos = get_visual_positions(input_ids, self._image_token_id)
                if len(visual_pos) > 0:
                    # Compute delta and select top visual tokens
                    delta = compute_visual_delta(list(all_hs), visual_pos)
                    selected_vis_idx = select_top_visual_tokens(
                        delta,
                        top_k=self.num_visual_tokens,
                        pre_filter_k=self.visual_pre_filter_k,
                    )
                    # Filter sequence: keep selected visual + all text
                    hidden_states, self._keep_positions = filter_sequence(
                        hidden_states,
                        visual_pos,
                        selected_vis_idx,
                        total_seq_len=hidden_states.shape[1],
                    )
                    # Also filter orig logits for consistency
                    if output_orig:
                        orig = orig[:, self._keep_positions]
                else:
                    self._keep_positions = None
            else:
                self._keep_positions = None

            if hasattr(self.spec_layer, "fc"):
                spec_device = self.spec_layer.fc.weight.device
                if hidden_states.device != spec_device:
                    hidden_states = hidden_states.to(spec_device)

        if output_real_hidden:
            return None, orig, hidden_states, outputs.hidden_states
        if output_orig:
            return None, orig, hidden_states
        else:
            return None, hidden_states

    @torch.no_grad()
    def specgenerate(
        self,
        input_ids,
        temperature=0.0,
        top_p=0.0,
        top_k=0.0,
        max_new_tokens=512,
        max_length=2048,
        log=False,
        is_llama3=False,
        inputs_embeds=None,
        return_acceptance_len=False,
        return_decode_time=False,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        stop_token_ids = _collect_stop_token_ids(self.tokenizer, self.base_model)
        if is_llama3:
            stop_token_ids.update(
                _normalize_token_ids(
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                )
            )
        max_length = max_length - self.spec_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.spec_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            try:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model)
            except:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(self.base_model.language_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data
        cache_max_len = past_key_values_data[0].shape[-2] if past_key_values_data else None

        embed_weights = None
        special_image_mask = None
        if (
            self.base_model.config.architectures[0]
            == "LlavaNextForConditionalGeneration"
        ):
            vision_feature_layer = kwargs.get("vision_feature_layer")
            vision_feature_select_strategy = kwargs.get(
                "vision_feature_select_strategy"
            )
            pixel_values = kwargs.get("pixel_values")
            image_sizes = kwargs.get("image_sizes")

            vision_feature_layer = (
                vision_feature_layer
                if vision_feature_layer is not None
                else self.base_model.config.vision_feature_layer
            )
            vision_feature_select_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else self.base_model.config.vision_feature_select_strategy
            )

            if pixel_values is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
                )

            if inputs_embeds is None:
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

            if pixel_values is not None and pixel_values.size(0) > 0:
                image_features = self.base_model.get_image_features(
                    pixel_values,
                    image_sizes,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

                # NOTE we only support multimodal_patch_merge_type == "spatial_unpad"
                image_features, feature_lens = self.base_model.pack_image_features(
                    image_features,
                    image_sizes,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                    image_newline=self.base_model.image_newline,
                )

                special_image_mask = (
                    input_ids == self.base_model.config.image_token_index
                ).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )
                if inputs_embeds[special_image_mask].numel() != image_features.numel():
                    n_image_tokens = (
                        input_ids == self.base_model.config.image_token_index
                    ).sum()
                    n_image_features = image_features.shape[0]
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_features = image_features.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_features
                )

                # special_image_mask = special_image_mask[..., 0]

        elif (
            self.base_model.config.architectures[0]
            == "Qwen2_5_VLForConditionalGeneration"
        ):
            pixel_values = kwargs.get("pixel_values")
            image_grid_thw = kwargs.get("image_grid_thw")
            # video_grid_thw = kwargs.get("video_grid_thw")

            if inputs_embeds is None:
                inputs_embeds = self.base_model.model.embed_tokens(input_ids)
                if pixel_values is not None:
                    pixel_values = pixel_values.type(self.base_model.visual.dtype)
                    image_embeds = self.base_model.visual(
                        pixel_values, grid_thw=image_grid_thw
                    )
                    n_image_tokens = (
                        (input_ids == self.base_model.config.image_token_id)
                        .sum()
                        .item()
                    )
                    n_image_features = image_embeds.shape[0]
                    if n_image_tokens != n_image_features:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                        )

                    mask = input_ids == self.base_model.config.image_token_id
                    mask_unsqueezed = mask.unsqueeze(-1)
                    mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                    image_mask = mask_expanded.to(inputs_embeds.device)

                    image_embeds = image_embeds.to(
                        inputs_embeds.device, inputs_embeds.dtype
                    )
                    inputs_embeds = inputs_embeds.masked_scatter(
                        image_mask, image_embeds
                    )

                    # special_image_mask = mask

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        (
            draft_tokens,
            retrieve_indices,
            tree_mask,
            tree_position_ids,
            logits,
            hidden_state,
            sample_token,
        ) = initialize_tree(
            input_ids,
            self,
            past_key_values,
            logits_processor,
            inputs_embeds,
            embed_weights,
            image_mask=special_image_mask,
            **kwargs,
        )
        new_token = 0

        if return_acceptance_len:
            acceptance_len = []
        if return_decode_time:
            torch.cuda.synchronize()
            start_time = time.time()

        for idx in range(max_length):
            # with Timer("all"):
            kv_cursor = input_ids.shape[1]
            if current_length_data is not None and current_length_data.numel() > 0:
                kv_cursor = max(kv_cursor, int(current_length_data.max().item()))
            if (
                cache_max_len is not None
                and kv_cursor + draft_tokens.shape[1] > cache_max_len
            ):
                break

            if not hasattr(self.base_model, "language_model"):
                self.base_model.model.tree_mask = tree_mask
            else:
                self.base_model.language_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)
            # with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
                current_length_data=current_length_data,
            )
            # retrieve_indices=tree_buffers["retrieve_indices"]
            # logits = logits[0, retrieve_indices]
            draft_tokens = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            accept_length = int(accept_length)
            remaining_tokens = max_new_tokens - new_token
            if remaining_tokens <= 0:
                break
            accept_length = min(accept_length, remaining_tokens - 1)
            if cache_max_len is not None:
                cache_remaining = cache_max_len - kv_cursor
                if cache_remaining <= 0:
                    break
                accept_length = min(accept_length, cache_remaining - 1)
            if accept_length < 0:
                break
            if return_acceptance_len:
                acceptance_len.append(int(accept_length))
            # with Timer("update_inference_inputs"):
            (
                input_ids,
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                new_token,
                hidden_state,
                sample_token,
                # inputs_embeds,
            ) = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p,
                cache_base_len=kv_cursor,
                # inputs_embeds,
                # embed_weights,
                # image_mask=special_image_mask,
            )

            if _has_stop_token(input_ids[0, input_len:], stop_token_ids):
                break
            if new_token >= max_new_tokens:
                break
        # if input_ids.shape[1] > max_length:
        #     break
        # if not log:
        #     return input_ids
        # else:
        #     return input_ids, new_token, idx

        outputs = (input_ids,)
        if log:
            outputs += (new_token, idx)
        if return_acceptance_len:
            outputs += (acceptance_len,)
        if return_decode_time:
            torch.cuda.synchronize()
            end_time = time.time()
            outputs += (end_time - start_time,)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
