"""Baseline evaluation for MMSpec unified dataset.

Usage:
python -m evaluation.eval_baseline_mmspec --base-model-path <path> --data-folder <path>
"""

import argparse
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from evaluation.utils import (
    load_mmspec_data,
    build_prompt,
    process_output,
    save_result,
    reorg_answer_file,
    load_existing_ids,
    get_common_args,
    get_num_turns,
    run_sanity_check,
    iter_eval_samples,
)
from evaluation.time_breakdown import build_time_breakdown_tracker


def baseline_forward(
    model,
    temperature=0.0,
    max_steps=2048,
    **kwargs,
):
    """Run baseline forward pass (autoregressive decoding)."""
    input_len = kwargs["input_ids"].shape[1]

    generate_kwargs = {
        "max_new_tokens": max_steps,
        "use_cache": True,
    }
    if temperature > 1e-5:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
    else:
        generate_kwargs["do_sample"] = False

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    output_ids = model.generate(**kwargs, **generate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()

    new_token = max(int(output_ids.shape[1] - input_len), 0)
    idx = max(new_token - 1, 0)
    return output_ids, new_token, idx, end_time - start_time


def load_baseline_model(base_model_path):
    """Load baseline model with architecture-aware class selection."""
    config = AutoConfig.from_pretrained(base_model_path)
    arch = config.architectures[0] if config.architectures else ""

    if arch == "Qwen2_5_VLForConditionalGeneration":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                device_map="auto",
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                device_map="auto",
            )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    return model, tokenizer, arch


@torch.inference_mode()
def evaluate(args):
    """Run baseline evaluation on MMSpec dataset."""
    # Load model
    model, tokenizer, arch = load_baseline_model(args.base_model_path)
    
    model.eval()
    time_tracker = build_time_breakdown_tracker(model)
    print("Loaded base architecture:", arch)
    print("Loaded model class:", model.__class__.__name__)
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    
    # Load data
    data = load_mmspec_data(args.data_folder)
    print(f"Loaded {len(data)} samples")
    if getattr(args, "sanity", False):
        run_sanity_check(args, model, tokenizer, data)
        return

    # Warmup
    print("Warming up...")
    for _ in range(3):
        torch.manual_seed(0)
        model_inputs = build_prompt(data[0], args)
        output_ids, _, _, _ = baseline_forward(
            **model_inputs,
            model=model,
            temperature=args.temperature,
            max_steps=min(args.max_new_token, 128),
        )
    print("Warmup done")
    
    # Evaluate
    existing_ids = load_existing_ids(args.answer_file)
    print(f"Skipping {len(existing_ids)} already-evaluated samples")
    for d in tqdm(iter_eval_samples(data, args.batch_size), total=len(data), desc=f"Evaluating (bs={args.batch_size})"):
        if d["id"] in existing_ids:
            continue
        choices = []
        num_turns = get_num_turns(d)
        
        for i in range(args.num_choices):
            torch.manual_seed(i)
            
            # For multi-turn, we need to collect results across all turns
            turn_outputs = []
            turn_idxs = []
            turn_new_tokens = []
            turn_wall_times = []
            turn_decode_times = []
            turn_draft_times = []
            turn_target_times = []
            conversation_history = []
            
            for turn_idx in range(num_turns):
                # Build prompt for this turn (with conversation history for multi-turn)
                model_inputs = build_prompt(d, args, turn_idx=turn_idx, conversation_history=conversation_history if turn_idx > 0 else None)
                input_len = model_inputs["input_ids"].shape[1]
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                time_tracker.reset()
                
                output_ids, new_token, idx, dec_time = baseline_forward(
                    **model_inputs,
                    model=model,
                    temperature=args.temperature,
                    max_steps=args.max_new_token,
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time = time.time() - start_time
                draft_time, target_time = time_tracker.snapshot()
                
                output = process_output(output_ids, tokenizer, input_len)
                
                # Store results for this turn
                turn_outputs.append(output)
                turn_idxs.append(int(idx))
                turn_new_tokens.append(int(new_token))
                turn_wall_times.append(total_time)
                turn_decode_times.append(dec_time)
                turn_draft_times.append(draft_time)
                turn_target_times.append(target_time)
                
                # Add to conversation history for next turn
                turns = d.get("turns", [d.get("prompt", "")])
                user_msg = turns[turn_idx] if turn_idx < len(turns) else turns[0]
                conversation_history.append((user_msg, output))
            
            choices.append({
                "index": i,
                "turns": turn_outputs,
                "idxs": turn_idxs,
                "new_tokens": turn_new_tokens,
                "wall_time": turn_wall_times,
                "decode_time": turn_decode_times,
                "draft_time": turn_draft_times,
                "target_time": turn_target_times,
            })
        
        save_result(args.answer_file, d, args.model_id, choices)
    
    reorg_answer_file(args.answer_file)
    print(f"Results saved to {args.answer_file}")


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for MMSpec")
    parser = get_common_args(parser)
    
    args = parser.parse_args()
    
    # Set model for prompt building
    args.model = args.base_model_path
    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    
    # Set answer file
    if not args.answer_file:
        args.answer_file = f"{args.bench_name}/{args.model_id}.jsonl"
    
    print(f"Output to {args.answer_file}")
    
    evaluate(args)


if __name__ == "__main__":
    main()
