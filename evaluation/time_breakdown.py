import time
from functools import wraps
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class TimeBreakdownTracker:
    """Track per-call forward time of target and draft modules."""

    def __init__(self, target_modules: List[nn.Module], draft_modules: List[nn.Module]):
        self.target_modules = target_modules
        self.draft_modules = draft_modules
        self.target_time = 0.0
        self.draft_time = 0.0
        self._handles = []
        self._stacks = {}
        self._wrapped_methods = []
        self._install_hooks()

    def _install_hooks(self):
        for module in self.target_modules:
            self._register_hooks(module, "target")
        for module in self.draft_modules:
            self._register_hooks(module, "draft")

    def install_phase_fallback_hooks(self, model):
        """Fallback when draft module hooks are unavailable.

        If a speculative method exists (e.g. specgenerate/msdgenerate), treat
        its non-target portion as draft time:
            draft += method_total_time - target_time_delta_within_method
        """
        if self.draft_modules:
            return
        for method_name in ("specgenerate", "msdgenerate"):
            self._wrap_phase_method(model, method_name)

    def _wrap_phase_method(self, obj, method_name: str):
        original = getattr(obj, method_name, None)
        if original is None or not callable(original):
            return
        if getattr(original, "__time_breakdown_wrapped__", False):
            return

        @wraps(original)
        def wrapped(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            target_before = self.target_time
            out = original(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            target_delta = max(0.0, self.target_time - target_before)
            # Count all non-target compute in spec phase as draft time.
            self.draft_time += max(0.0, elapsed - target_delta)
            return out

        setattr(wrapped, "__time_breakdown_wrapped__", True)
        setattr(obj, method_name, wrapped)
        self._wrapped_methods.append((obj, method_name, original))

    def _register_hooks(self, module: nn.Module, kind: str):
        key = id(module)
        self._stacks[key] = []

        def pre_hook(_module, _inputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._stacks[key].append(time.perf_counter())

        def post_hook(_module, _inputs, _output):
            if not self._stacks[key]:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = self._stacks[key].pop()
            dt = time.perf_counter() - start
            if kind == "target":
                self.target_time += dt
            else:
                self.draft_time += dt

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def reset(self):
        self.target_time = 0.0
        self.draft_time = 0.0
        for k in self._stacks:
            self._stacks[k].clear()

    def snapshot(self) -> Tuple[float, float]:
        return float(self.draft_time), float(self.target_time)

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        for obj, method_name, original in self._wrapped_methods:
            setattr(obj, method_name, original)
        self._wrapped_methods = []


def build_time_breakdown_tracker(model) -> TimeBreakdownTracker:
    target_modules: List[nn.Module] = []

    # MSD/EaModel special-case:
    # - non-HF path primarily runs self._get_language_model()
    # - HF-LLaVA path primarily runs self.base_model(...)
    # Pick one to avoid parent+child double counting.
    if hasattr(model, "_get_language_model") and hasattr(model, "_is_hf_llava"):
        if bool(getattr(model, "_is_hf_llava")):
            if hasattr(model, "base_model") and isinstance(model.base_model, nn.Module):
                target_modules.append(model.base_model)
        else:
            try:
                lang_model = model._get_language_model()
            except Exception:
                lang_model = None
            if isinstance(lang_model, nn.Module):
                target_modules.append(lang_model)

    if not target_modules:
        if hasattr(model, "base_model") and isinstance(model.base_model, nn.Module):
            target_modules.append(model.base_model)
        elif isinstance(model, nn.Module):
            target_modules.append(model)

    draft_modules = []
    for attr in ("spec_layer", "ea_layer", "draft_model"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            draft_modules.append(module)

    # Remove duplicates while preserving order.
    seen = set()
    uniq_targets = []
    for m in target_modules:
        if id(m) in seen:
            continue
        seen.add(id(m))
        uniq_targets.append(m)

    uniq_drafts = []
    for m in draft_modules:
        if id(m) in seen:
            continue
        seen.add(id(m))
        uniq_drafts.append(m)
    tracker = TimeBreakdownTracker(target_modules=uniq_targets, draft_modules=uniq_drafts)
    tracker.install_phase_fallback_hooks(model)
    return tracker
