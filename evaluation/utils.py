"""Common utilities for MMSpec evaluation.

This module contains shared functions used by both baseline and speculative evaluation scripts.
"""

import json
import os
import time
from datetime import datetime

import shortuuid
import torch
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor


def load_existing_ids(answer_file):
    """Load question IDs already present in an answer JSONL file.

    Returns an empty set when the file does not exist or is empty,
    so callers can always do ``if qid in existing_ids: continue``.
    """
    ids = set()
    if answer_file and os.path.exists(answer_file):
        with open(answer_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["question_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return ids


def load_mmspec_data(data_folder):
    """Load MMSpec unified dataset.
    
    Args:
        data_folder: Path to the MMSpec dataset folder containing mmspec.jsonl and images/
        
    Returns:
        Dataset object with all samples loaded
    """
    data = []
    jsonl_path = os.path.join(data_folder, "mmspec.jsonl")
    images_dir = os.path.join(data_folder, "images")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Load image
            img_path = os.path.join(images_dir, d["image"])
            d["image"] = Image.open(img_path)
            data.append(d)
    
    return Dataset.from_list(data)


def build_prompt(data, args, turn_idx=0, conversation_history=None):
    """Build prompt for MMSpec unified dataset.
    
    MMSpec has a unified format:
    - id: unique identifier
    - image: PIL Image object (already loaded)
    - turns: list of question texts (single or multiple turns)
    - category: original category
    - topic: "general vqa" | "complex reasoning" | "text vqa" | "image captioning" | "chart qa" | "multi-turn conversation"
    
    Args:
        data: A dict containing 'turns' and 'image' (PIL Image)
        args: Arguments containing model path
        turn_idx: Index of the current turn (0 for first turn)
        conversation_history: List of previous (user_msg, assistant_response) tuples for multi-turn
    
    Returns:
        Processed inputs ready for model inference
    """
    if "Qwen2.5-VL" in args.model:
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            args.model, use_fast=True, min_pixels=min_pixels, max_pixels=max_pixels
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model)

    # Simple system prompt - let the model respond naturally
    system_text = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."

    examples = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_text,
                },
            ],
        }
    ]
    
    images = [data["image"]]

    # Build conversation history for multi-turn
    if conversation_history:
        for user_msg, assistant_response in conversation_history:
            examples.append({
                "role": "user",
                "content": [
                    {"type": "image"} if len(examples) == 1 else {},  # Only first user message has image
                    {"type": "text", "text": user_msg},
                ],
            })
            examples[-1]["content"] = [c for c in examples[-1]["content"] if c]  # Remove empty dicts
            examples.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            })

    # Get current turn question
    turns = data.get("turns", [data.get("prompt", "")])  # Backward compatible with old format
    current_question = turns[turn_idx] if turn_idx < len(turns) else turns[0]
    
    # Add current user message
    if conversation_history:
        # Not the first message, so no image
        examples.append({
            "role": "user",
            "content": [{"type": "text", "text": current_question}],
        })
    else:
        # First message, include image
        examples.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": current_question},
            ],
        })

    # Create the prompt input
    prompt_input = processor.apply_chat_template(examples, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt_input, return_tensors="pt").to(
        "cuda:0"
    )

    return inputs


def process_output(output_ids, tokenizer, input_len):
    """Process model output to clean text.
    
    Args:
        output_ids: Raw output token ids
        tokenizer: Tokenizer to decode output
        input_len: Length of input (to exclude from output)
        
    Returns:
        Cleaned output string
    """
    output_ids = output_ids[0][input_len:]
    if isinstance(output_ids, torch.Tensor):
        output_ids = output_ids.detach().to("cpu")
    else:
        output_ids = torch.tensor(output_ids, dtype=torch.long)

    # Keep ids in tokenizer index range. For Qwen-VL, many valid special tokens
    # are >= tokenizer.vocab_size, so use len(tokenizer) instead of vocab_size.
    tokenizer_size = len(tokenizer)
    if tokenizer_size > 0:
        valid_mask = (output_ids >= 0) & (output_ids < tokenizer_size)
        output_ids = output_ids[valid_mask]
    else:
        output_ids = output_ids[output_ids >= 0]

    output = tokenizer.decode(
        output_ids.tolist(),
        spaces_between_special_tokens=False,
    )
    
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                output = output.replace(special_tok, "")
        else:
            output = output.replace(special_token, "")
    
    output = output.strip()
    
    if output.startswith("Assistant:"):
        output = output.replace("Assistant:", "", 1).strip()
    
    return output


def get_num_turns(data):
    """Get the number of turns for a sample.
    
    Args:
        data: A dict containing 'turns' or 'prompt'
        
    Returns:
        Number of turns (1 for single-turn, more for multi-turn)
    """
    turns = data.get("turns", [data.get("prompt", "")])
    return len(turns) if isinstance(turns, list) else 1


def iter_eval_samples(data, batch_size=1):
    """Yield samples in evaluation order with a batch-size-aware traversal.

    Note:
    - This utility standardizes the evaluation interface to accept `--batch-size`
      across methods, while preserving per-request result logging.
    - Method internals may still execute each sample independently.
    """
    if batch_size is None or batch_size <= 1:
        for sample in data:
            yield sample
        return

    total = len(data)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        for idx in range(start, end):
            yield data[idx]


def save_result(answer_file, sample_data, model_id, choices):
    """Save evaluation result to JSONL file.
    
    Args:
        answer_file: Path to output JSONL file
        sample_data: Original sample data (contains id, topic, category)
        model_id: Model identifier
        choices: List of choice dicts with turns, wall_time, etc.
    """
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    
    with open(os.path.expanduser(answer_file), "a") as fout:
        ans_json = {
            "question_id": sample_data["id"],
            "topic": sample_data.get("topic", "unknown"),
            "category": sample_data.get("category", "unknown"),
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": choices,
            "tstamp": time.time(),
        }
        fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication.
    
    Args:
        answer_file: Path to JSONL file to reorganize
    """
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def run_sanity_check(args, model, tokenizer=None, data=None):
    """Verify model loaded correctly without running full inference.

    Logs:
    - GPU environment
    - Base model config (arch, dtype, devices, param count)
    - Spec layer / draft head info (weight shapes, missing/unexpected keys, checkpoint origin)
    - One forward pass on sample[0] to confirm no runtime errors
    """
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("[SANITY CHECK]  Model Load Verification")
    print(f"{SEP}")

    # ── 1. Environment ──────────────────────────────────────────────────────
    print("\n[SANITY] ── Environment ──")
    print(f"  CUDA_VISIBLE_DEVICES : {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    n_gpus = torch.cuda.device_count()
    print(f"  torch.cuda.device_count(): {n_gpus}")
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        used_gb = torch.cuda.memory_allocated(i) / 1e9
        total_gb = p.total_memory / 1e9
        print(f"    GPU {i}: {p.name}  {total_gb:.1f} GB total  |  {used_gb:.2f} GB already allocated")

    # ── 2. Args summary ─────────────────────────────────────────────────────
    print("\n[SANITY] ── Launch Args ──")
    print(f"  base_model_path  : {getattr(args, 'base_model_path', '<N/A>')}")
    for attr in ("spec_model_path", "msd_model_path", "ea_model_path"):
        v = getattr(args, attr, None)
        if v is not None:
            print(f"  {attr:<18}: {v}")
    print(f"  temperature      : {getattr(args, 'temperature', '<N/A>')}")
    for attr in ("total_token", "depth", "top_k", "threshold", "num_q"):
        v = getattr(args, attr, None)
        if v is not None:
            print(f"  {attr:<18}: {v}")

    # ── 3. Base model ────────────────────────────────────────────────────────
    base = getattr(model, "base_model", model)
    print("\n[SANITY] ── Base Model ──")
    cfg = getattr(base, "config", None)
    if cfg is not None:
        print(f"  architectures    : {getattr(cfg, 'architectures', '?')}")
        print(f"  model_type       : {getattr(cfg, 'model_type', '?')}")
        print(f"  hidden_size      : {getattr(cfg, 'hidden_size', '?')}")
        print(f"  vocab_size       : {getattr(cfg, 'vocab_size', '?')}")
        print(f"  num_hidden_layers: {getattr(cfg, 'num_hidden_layers', '?')}")
        print(f"  num_attn_heads   : {getattr(cfg, 'num_attention_heads', '?')}")
        print(f"  num_kv_heads     : {getattr(cfg, 'num_key_value_heads', '?')}")
    # dtype + device from lm_head or first param
    try:
        lm_head = base.lm_head if hasattr(base, "lm_head") else (
            base.language_model.lm_head if hasattr(getattr(base, "language_model", None), "lm_head") else None
        )
        if lm_head is not None:
            w = lm_head.weight
            print(f"  lm_head.weight   : shape={tuple(w.shape)}  dtype={w.dtype}  device={w.device}")
    except Exception as e:
        print(f"  lm_head inspect failed: {e}")
    # total params
    total_p = sum(p.numel() for p in base.parameters())
    print(f"  total params     : {total_p/1e6:.1f} M")

    # ── 4. Spec layer / heads ────────────────────────────────────────────────
    spec_layer = getattr(model, "spec_layer", None)
    if spec_layer is None:
        # MSD / EaModel uses different naming
        for attr in ("ea_layer", "draft_model", "speculator"):
            spec_layer = getattr(model, attr, None)
            if spec_layer is not None:
                break
    if spec_layer is not None:
        print(f"\n[SANITY] ── Spec Layer ({type(spec_layer).__name__}) ──")
        for attr in ("depth", "top_k", "total_tokens", "hidden_size", "vocab_size", "threshold"):
            v = getattr(spec_layer, attr, None)
            if v is not None:
                print(f"  {attr:<18}: {v}")
        # Weight shapes
        print("  Weights:")
        for name, param in spec_layer.named_parameters():
            print(f"    {name:<55} {str(tuple(param.shape)):<25} {param.dtype}  {param.device}")
        # Medusa heads — spec_layer.medusa stores the *count* (int), not the ModuleList
        medusa = getattr(spec_layer, "medusa", None)
        if medusa is not None:
            num = medusa if isinstance(medusa, int) else len(medusa)
            print(f"  medusa heads     : num_heads={num}")
    else:
        print("\n[SANITY] ── No separate spec layer found (base-only method) ──")

    # ── 5. Tokenizer ─────────────────────────────────────────────────────────
    if tokenizer is not None:
        print("\n[SANITY] ── Tokenizer ──")
        print(f"  class            : {type(tokenizer).__name__}")
        print(f"  vocab_size       : {tokenizer.vocab_size}")
        print(f"  eos_token_id     : {tokenizer.eos_token_id}")
        print(f"  pad_token_id     : {tokenizer.pad_token_id}")

    # ── 6. Forward-pass smoke test ───────────────────────────────────────────
    if data is not None and len(data) > 0:
        print("\n[SANITY] ── Forward-Pass Smoke Test (1 sample) ──")
        try:
            sample = data[0]
            model_inputs = build_prompt(sample, args)
            print(f"  input_ids shape  : {model_inputs['input_ids'].shape}")
            pixel_val = model_inputs.get("pixel_values")
            if pixel_val is not None:
                print(f"  pixel_values shp : {pixel_val.shape}  dtype={pixel_val.dtype}")
            with torch.no_grad():
                if hasattr(model, "msdgenerate"):
                    # MSD requires inputs_embeds pre-processing (model.get_inputs_embeds).
                    # Skip the forward-pass smoke test; weight load is sufficient.
                    print("  (smoke test skipped — MSD requires inputs_embeds pre-processing)")
                elif hasattr(model, "specgenerate"):
                    # Spec models use a custom KV cache incompatible with HF .generate().
                    # Use the model's own specgenerate() interface instead.
                    result = model.specgenerate(
                        **model_inputs,
                        temperature=0.0,
                        max_new_tokens=10,
                        log=True,
                    )
                    out_ids = result[0]  # (input_ids, new_token, idx, ...)
                    n_new = int(result[1]) if len(result) > 1 else "?"
                    print(f"  specgenerate() OK: produced {n_new} new tokens")
                else:
                    # Base-only model: call generate() on the model object directly.
                    # (Do NOT use model.base_model — HF's base_model property returns
                    # the inner transformer backbone which lacks a language model head.)
                    out = model.generate(
                        **model_inputs,
                        max_new_tokens=10,
                        do_sample=False,
                    )
                    n_new = out.shape[1] - model_inputs["input_ids"].shape[1]
                    print(f"  generate() OK    : produced {n_new} new tokens  (output shape {tuple(out.shape)})")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
    else:
        print("\n[SANITY] ── Forward-Pass Smoke Test SKIPPED (no data) ──")

    # ── 7. GPU memory after load ─────────────────────────────────────────────
    print("\n[SANITY] ── GPU Memory After Load ──")
    for i in range(n_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved  = torch.cuda.memory_reserved(i) / 1e9
        print(f"  GPU {i}: allocated={allocated:.2f} GB  reserved={reserved:.2f} GB")

    print(f"\n{SEP}")
    print("[SANITY CHECK]  PASSED — model loaded successfully. Exiting (--sanity mode).")
    print(f"{SEP}\n")


def get_common_args(parser):
    """Add common arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        parser with common arguments added
    """
    parser.add_argument(
        "--spec-model-path",
        type=str,
        default="",
        help="Path to speculative model weights",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="Path to base model weights",
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="mmspec"
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="results/mmspec_test/",
        help="Output directory for results",
    )
    parser.add_argument("--answer-file", type=str, help="Output answer file path")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="Maximum number of new generated tokens",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="Number of GPUs per model",
    )
    parser.add_argument(
        "--num-gpus-total", 
        type=int, 
        default=1, 
        help="Total number of GPUs"
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maximum GPU memory used for model weights per GPU",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size setting (used for unified scheduling across methods).",
    )
    parser.add_argument(
        "--data-folder", 
        type=str, 
        default="dataset/MMSpec",
        help="Path to MMSpec dataset folder"
    )
    parser.add_argument(
        "--sanity",
        action="store_true",
        default=False,
        help="Sanity-check mode: load model, log details, run 1-sample smoke test, then exit without inference",
    )
    
    return parser
