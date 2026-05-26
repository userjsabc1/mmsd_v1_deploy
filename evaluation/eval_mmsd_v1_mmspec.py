"""MMSD v1 speculative decoding evaluation for MMSpec unified dataset.

Usage:
python -m evaluation.eval_mmsd_v1_mmspec --base-model-path <path> --spec-model-path <path> --data-folder <path>
"""

import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from method.mmsd_v1.kv_cache import initialize_past_key_values
from method.mmsd_v1.utils import *
from method.mmsd_v1.spec_model import SpecModel
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


@torch.inference_mode()
def evaluate(args):
    """Run MMSD v1 speculative decoding evaluation on MMSpec dataset."""
    # Load model
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        num_visual_tokens=args.num_visual_tokens,
        visual_pre_filter_k=args.visual_pre_filter_k,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature)
    else:
        logits_processor = None

    model.eval()
    time_tracker = build_time_breakdown_tracker(model)
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print(f"MMSD v1 config: num_visual_tokens={args.num_visual_tokens}, "
          f"visual_pre_filter_k={args.visual_pre_filter_k}")

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
        output_ids, new_token, idx = model.specgenerate(
            **model_inputs,
            temperature=args.temperature,
            max_new_tokens=min(args.max_new_token, 128),
            log=True,
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

            turn_outputs = []
            turn_idxs = []
            turn_new_tokens = []
            turn_wall_times = []
            turn_acceptance_lengths = []
            turn_draft_times = []
            turn_target_times = []
            conversation_history = []

            for turn_idx in range(num_turns):
                model_inputs = build_prompt(d, args, turn_idx=turn_idx, conversation_history=conversation_history if turn_idx > 0 else None)
                input_len = model_inputs["input_ids"].shape[1]

                torch.cuda.synchronize()
                start_time = time.time()
                time_tracker.reset()

                output_ids, new_token, idx, accp_len = model.specgenerate(
                    **model_inputs,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_token,
                    log=True,
                    return_acceptance_len=True,
                )

                torch.cuda.synchronize()
                total_time = time.time() - start_time
                draft_time, target_time = time_tracker.snapshot()

                output = process_output(output_ids, tokenizer, input_len)

                turn_outputs.append(output)
                turn_idxs.append(int(idx))
                turn_new_tokens.append(int(new_token))
                turn_wall_times.append(total_time)
                turn_acceptance_lengths.append(accp_len)
                turn_draft_times.append(draft_time)
                turn_target_times.append(target_time)

                turns = d.get("turns", [d.get("prompt", "")])
                user_msg = turns[turn_idx] if turn_idx < len(turns) else turns[0]
                conversation_history.append((user_msg, output))

            choices.append({
                "index": i,
                "turns": turn_outputs,
                "idxs": turn_idxs,
                "new_tokens": turn_new_tokens,
                "wall_time": turn_wall_times,
                "acceptance_length": turn_acceptance_lengths,
                "draft_time": turn_draft_times,
                "target_time": turn_target_times,
            })

        save_result(args.answer_file, d, args.model_id, choices)

    reorg_answer_file(args.answer_file)
    print(f"Results saved to {args.answer_file}")


def main():
    parser = argparse.ArgumentParser(description="MMSD v1 evaluation for MMSpec")
    parser = get_common_args(parser)

    # Speculative decoding args
    parser.add_argument(
        "--total-token",
        type=int,
        default=40,
        help="Total tokens for speculative decoding",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth for speculative decoding",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-k for speculative decoding",
    )

    # MMSD v1 specific args
    parser.add_argument(
        "--num-visual-tokens",
        type=int,
        default=32,
        help="Number of visual tokens to keep after delta filtering",
    )
    parser.add_argument(
        "--visual-pre-filter-k",
        type=int,
        default=64,
        help="Pre-filter top-k visual tokens before final selection",
    )

    args = parser.parse_args()

    # Set model for prompt building
    args.model = args.base_model_path
    args.model_id = args.model_id + "-mmsd_v1-temperature-" + str(args.temperature)

    # Set answer file
    if not args.answer_file:
        args.answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {args.answer_file}")
    print("Method: MMSD v1 (Multimodal Self-Speculative Decoding)")

    evaluate(args)


if __name__ == "__main__":
    main()
