import torch
from transformers import LlavaForConditionalGeneration, LlavaNextForConditionalGeneration
from transformers.cache_utils import DynamicCache
import types


def _is_custom_kv_cache(past_key_values):
    if not isinstance(past_key_values, (list, tuple)) or len(past_key_values) == 0:
        return False
    first = past_key_values[0]
    if not isinstance(first, (list, tuple)) or len(first) != 2:
        return False
    return hasattr(first[0], "data") and hasattr(first[0], "current_length")


def _custom_cache_to_legacy(past_key_values):
    legacy = []
    for layer in past_key_values:
        key_cache, value_cache = layer
        k_len = int(key_cache.current_length.item())
        v_len = int(value_cache.current_length.item())
        key = key_cache.data[..., :k_len, :]
        value = value_cache.data[..., :v_len, :]
        legacy.append((key, value))
    return tuple(legacy)


def _sync_legacy_to_custom_cache(past_key_values, legacy_cache):
    for i, (key, value) in enumerate(legacy_cache):
        key_cache, value_cache = past_key_values[i]
        k_len = key.shape[-2]
        v_len = value.shape[-2]
        key_cache.data[..., :k_len, :].copy_(key, non_blocking=True)
        value_cache.data[..., :v_len, :].copy_(value, non_blocking=True)
        key_cache.current_length.fill_(k_len)
        value_cache.current_length.fill_(v_len)


def _forward_with_custom_cache(super_forward, *args, **kwargs):
    custom_cache = None
    past_key_values = kwargs.get("past_key_values")
    if _is_custom_kv_cache(past_key_values):
        custom_cache = past_key_values
        legacy_cache = _custom_cache_to_legacy(custom_cache)
        kwargs["past_key_values"] = DynamicCache.from_legacy_cache(legacy_cache)

    outputs = super_forward(*args, **kwargs)

    if custom_cache is not None and hasattr(outputs, "past_key_values"):
        out_cache = outputs.past_key_values
        if out_cache is not None and hasattr(out_cache, "to_legacy_cache"):
            _sync_legacy_to_custom_cache(custom_cache, out_cache.to_legacy_cache())

    return outputs


def _patch_tree_mask_for_llama_model(model):
    """Inject tree_mask support into HF LLaMA causal mask update for speculative tree decoding."""
    language_model = getattr(model, "language_model", None)
    llama_model = getattr(language_model, "model", None)
    if llama_model is None or not hasattr(llama_model, "_update_causal_mask"):
        return
    if getattr(llama_model, "_mmspec_tree_mask_patched", False):
        return

    original_update = llama_model._update_causal_mask

    def _update_causal_mask_with_tree(
        self,
        attention_mask,
        input_tensor,
        cache_position,
        past_key_values,
        output_attentions,
    ):
        causal_mask = original_update(
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
            output_attentions,
        )
        tree_mask = getattr(self, "tree_mask", None)
        if tree_mask is None:
            return causal_mask

        if causal_mask is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            sequence_length = input_tensor.shape[1]
            target_length = past_seen_tokens + sequence_length + 1
            causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=target_length,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
                cache_position=cache_position,
                batch_size=input_tensor.shape[0],
            )

        if causal_mask is None or causal_mask.dim() != 4:
            return causal_mask

        tree_len = int(tree_mask.size(-1))
        if (
            tree_len <= 0
            or causal_mask.size(-1) < tree_len + 1
            or causal_mask.size(-2) < tree_len
        ):
            return causal_mask

        tree_mask = tree_mask.to(device=causal_mask.device)
        fill_value = torch.finfo(causal_mask.dtype).min
        # The causal mask has target_length = past_seen + seq_len + 1 (extra
        # +1 column).  The tree KV entries occupy columns
        # [-tree_len-1 : -1], NOT [-tree_len:].  Using [-tree_len:] shifts the
        # tree mask one column to the right, causing wrong attention patterns
        # during tree decoding and 0 % acceptance for LLaVA EAGLE.
        causal_mask[:, :, -tree_len:, -tree_len - 1 : -1] = causal_mask[
            :, :, -tree_len:, -tree_len - 1 : -1
        ].masked_fill(tree_mask == 0, fill_value)
        return causal_mask

    llama_model._update_causal_mask = types.MethodType(
        _update_causal_mask_with_tree, llama_model
    )
    llama_model._mmspec_tree_mask_patched = True


class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        _patch_tree_mask_for_llama_model(model)
        return model

    def forward(self, *args, **kwargs):
        return _forward_with_custom_cache(super().forward, *args, **kwargs)


class CustomLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        _patch_tree_mask_for_llama_model(model)
        return model

    def forward(self, *args, **kwargs):
        return _forward_with_custom_cache(super().forward, *args, **kwargs)
