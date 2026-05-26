"""
生成 MMSD v1 训练数据：用 target model (LLaVA-1.5-13B) 跑一遍 prompt，
把模型自己的回复作为训练数据。

原因：draft model 学的是 target model 的输出分布，训练数据必须是 target model
      自己生成的 response，不能用数据集里的人写 response（分布不匹配）。

参考：SpecForge (sgl-project/SpecForge) regenerate_train_data.py

两种模式：
  1. --mode hf     用 HuggingFace transformers 直接推理（简单，单 GPU）
  2. --mode sglang 用 SGLang server API 并发请求（快，需先启动 server）

用法：
  # HF 模式（单 GPU，适合小规模数据或无 SGLang 环境）
  python scripts/generate_training_data.py \
      --mode hf \
      --model-path /path/to/llava-1.5-13b-hf \
      --input-path /path/to/blip_laion_cc_sbu_558k.json \
      --image-dir /path/to/LLaVA-Pretrain \
      --output-path /path/to/mmsd_v1_train_data.jsonl \
      --max-new-tokens 512

  # SGLang 模式（需先启动 server: python -m sglang.launch_server --model ... --port 30000）
  python scripts/generate_training_data.py \
      --mode sglang \
      --server-address localhost:30000 \
      --input-path /path/to/blip_laion_cc_sbu_558k.json \
      --image-dir /path/to/LLaVA-Pretrain \
      --output-path /path/to/mmsd_v1_train_data.jsonl \
      --concurrency 64 --max-new-tokens 512
"""

import argparse
import json
import os
import sys
import time
import base64
from pathlib import Path

from tqdm import tqdm


def load_input_data(input_path):
    """加载 LLaVA-Pretrain 格式的 JSON 或 JSONL 数据。"""
    if input_path.endswith(".jsonl"):
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def extract_user_prompt(item):
    """从 LLaVA-Pretrain 格式中提取 user prompt（去掉 gpt 回复）。"""
    convs = item.get("conversations", [])
    if not convs:
        return None
    # 找第一个 human turn
    for c in convs:
        if c.get("from") == "human":
            return c["value"]
    return None


def load_existing_ids(output_path):
    """加载已处理的 sample IDs（用于断点续传）。"""
    ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        ids.add(obj.get("id", ""))
                    except json.JSONDecodeError:
                        pass
    return ids


# ================================================================
# HuggingFace 模式
# ================================================================
def generate_hf(args):
    """使用 HuggingFace transformers 直接推理。"""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    print(f"Loading model from {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    data = load_input_data(args.input_path)
    existing_ids = load_existing_ids(args.output_path)
    print(f"Total samples: {len(data)}, already processed: {len(existing_ids)}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    out_f = open(args.output_path, "a", encoding="utf-8")

    success, fail = 0, 0
    for i, item in enumerate(tqdm(data, desc="Generating")):
        item_id = item.get("id", str(i))
        if item_id in existing_ids:
            continue

        image_file = item.get("image")
        if not image_file:
            continue
        user_prompt = extract_user_prompt(item)
        if not user_prompt:
            continue

        image_path = os.path.join(args.image_dir, image_file)
        if not os.path.exists(image_path):
            fail += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            fail += 1
            continue

        # 构建 prompt：只有 user turn，让模型自己生成 response
        # LLaVA-1.5 的 chat template
        content_clean = user_prompt.replace("<image>", "").strip()
        if "<image>" in user_prompt:
            text = f"USER: <image>\n{content_clean} ASSISTANT:"
        else:
            text = f"USER: {content_clean} ASSISTANT:"

        try:
            inputs = processor(text=text, images=image, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    top_p=args.top_p if args.temperature > 0 else None,
                )

            # 截取生成的部分（去掉 input prompt）
            input_len = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, input_len:]
            response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if not response:
                fail += 1
                continue

            # 保存：与 LLaVA-Pretrain 格式兼容
            result = {
                "id": item_id,
                "image": image_file,
                "conversations": [
                    {"from": "human", "value": user_prompt},
                    {"from": "gpt", "value": response},
                ],
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            success += 1

        except Exception as e:
            if fail < 10:
                print(f"  Error on sample {item_id}: {e}")
            fail += 1
            continue

    out_f.close()
    print(f"Done. success={success}, fail={fail}")
    print(f"Output: {args.output_path}")


# ================================================================
# SGLang 模式（参考 SpecForge regenerate_train_data.py）
# ================================================================
def generate_sglang(args):
    """使用 SGLang server API 并发推理。需先启动 SGLang server。"""
    import concurrent.futures
    import threading

    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package required for sglang mode. pip install openai")
        sys.exit(1)

    data = load_input_data(args.input_path)
    existing_ids = load_existing_ids(args.output_path)
    print(f"Total samples: {len(data)}, already processed: {len(existing_ids)}")

    # 过滤已处理的
    pending = []
    for i, item in enumerate(data):
        item_id = item.get("id", str(i))
        if item_id not in existing_ids and item.get("image") and extract_user_prompt(item):
            item["_idx"] = i
            item["_id"] = item_id
            pending.append(item)
    print(f"Pending: {len(pending)} samples")

    servers = [s.strip() for s in args.server_address.split(",")]
    clients = [OpenAI(base_url=f"http://{s}/v1", api_key="None") for s in servers]

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    write_lock = threading.Lock()
    out_f = open(args.output_path, "a", encoding="utf-8")
    pbar = tqdm(total=len(pending), desc="Generating")

    def process_one(item, client_idx):
        client = clients[client_idx % len(clients)]
        item_id = item["_id"]
        image_file = item["image"]
        user_prompt = extract_user_prompt(item)
        content_clean = user_prompt.replace("<image>", "").strip()

        image_path = os.path.join(args.image_dir, image_file)
        if not os.path.exists(image_path):
            return False

        # 读取图片并 base64 编码
        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return False

        # 构建 OpenAI-compatible multimodal message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": content_clean},
                ],
            }
        ]

        try:
            resp = client.chat.completions.create(
                model="default",
                messages=messages,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            response = resp.choices[0].message.content.strip()
            if not response:
                return False

            result = {
                "id": item_id,
                "image": image_file,
                "conversations": [
                    {"from": "human", "value": user_prompt},
                    {"from": "gpt", "value": response},
                ],
            }
            with write_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                pbar.update(1)
            return True

        except Exception as e:
            return False

    success, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_one, item, i): item
            for i, item in enumerate(pending)
        }
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success += 1
            else:
                fail += 1

    out_f.close()
    pbar.close()
    print(f"Done. success={success}, fail={fail}")
    print(f"Output: {args.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MMSD v1 training data by running target model on prompts"
    )
    parser.add_argument("--mode", choices=["hf", "sglang"], default="hf",
                        help="hf: HuggingFace direct inference; sglang: SGLang server API")
    parser.add_argument("--model-path", type=str, default="",
                        help="Path to target model (required for hf mode)")
    parser.add_argument("--server-address", type=str, default="localhost:30000",
                        help="SGLang server address(es), comma-separated (for sglang mode)")
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path to input data (LLaVA-Pretrain JSON or JSONL)")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Directory containing images")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output JSONL path (regenerated data)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 for greedy, >0 for sampling")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Concurrent requests (sglang mode only)")

    args = parser.parse_args()

    if args.mode == "hf" and not args.model_path:
        parser.error("--model-path is required for hf mode")

    if args.mode == "hf":
        generate_hf(args)
    else:
        generate_sglang(args)


if __name__ == "__main__":
    main()
