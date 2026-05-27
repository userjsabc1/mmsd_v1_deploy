"""
准备混合数据集：从多个 HuggingFace 数据集下载、采样，
输出统一的 LLaVA-Pretrain 兼容格式 JSONL + 图片目录。

输出格式与 generate_training_data.py 输入兼容：
{
    "id": "gqa_00001",
    "image": "gqa/00001.jpg",
    "conversations": [{"from": "human", "value": "<image>\nWhat color is the car?"}]
}

用法：
    export HF_ENDPOINT=https://hf-mirror.com   # 国内镜像
    python scripts/prepare_mixed_dataset.py \
        --output-dir /mnt/workspace/matthew/mixed_data \
        --seed 42

数据集配比（可通过命令行调整）：
    TextVQA:     20000  (主力，图文理解)
    GQA:         10000  (视觉推理)
    COCO:         5000  (低占比，短文本描述)
    CharXiv:      2323  (全取，图表推理)
    MMMU_Pro:     1730  (全取，长文本多选)
    ConvBench:     240  (全取，多轮对话)
    MM-MT-Bench:    92  (全取，多轮多模态)
    ─────────────────
    合计:       ~39385
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm


def save_image(image, path):
    """保存 PIL Image 为 JPEG。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, "JPEG", quality=90)


def make_entry(entry_id, image_rel_path, prompt):
    """构建 LLaVA-Pretrain 兼容的 JSONL entry（只有 human turn）。"""
    return {
        "id": entry_id,
        "image": image_rel_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{prompt}"}
        ],
    }


# ================================================================
# 各数据集加载器
# ================================================================

def load_textvqa(count, output_dir, seed):
    """TextVQA: image + question，图文理解。"""
    from datasets import load_dataset
    print(f"\n[TextVQA] Loading (target: {count})...")
    ds = load_dataset("facebook/textvqa", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:count]

    entries = []
    img_dir = os.path.join(output_dir, "images", "textvqa")
    os.makedirs(img_dir, exist_ok=True)

    for i, idx in enumerate(tqdm(indices, desc="TextVQA")):
        sample = ds[idx]
        entry_id = f"textvqa_{i:06d}"
        img_path = f"textvqa/{entry_id}.jpg"
        full_img_path = os.path.join(output_dir, "images", img_path)
        try:
            save_image(sample["image"], full_img_path)
            entries.append(make_entry(entry_id, img_path, sample["question"]))
        except Exception as e:
            if i < 5:
                print(f"  Skip {entry_id}: {e}")
            continue

    print(f"  [TextVQA] Got {len(entries)} samples")
    return entries


def load_gqa(count, output_dir, seed):
    """GQA: image + question，视觉推理。需要 join images 和 instructions。"""
    from datasets import load_dataset
    print(f"\n[GQA] Loading (target: {count})...")

    # 加载 testdev balanced（~12.6k）
    try:
        instructions = load_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
        images_ds = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
    except Exception as e:
        print(f"  [GQA] Failed to load testdev_balanced, trying val_balanced: {e}")
        instructions = load_dataset("lmms-lab/GQA", "val_balanced_instructions", split="val")
        images_ds = load_dataset("lmms-lab/GQA", "val_balanced_images", split="val")

    # 建立 imageId -> image 的映射
    print("  Building image index...")
    img_index = {}
    for sample in tqdm(images_ds, desc="GQA images index"):
        img_index[sample["id"]] = sample["image"]

    # 采样 instructions
    indices = list(range(len(instructions)))
    random.shuffle(indices)
    indices = indices[:count]

    entries = []
    img_dir = os.path.join(output_dir, "images", "gqa")
    os.makedirs(img_dir, exist_ok=True)

    for i, idx in enumerate(tqdm(indices, desc="GQA")):
        sample = instructions[idx]
        image_id = sample["imageId"]
        if image_id not in img_index:
            continue

        entry_id = f"gqa_{i:06d}"
        img_path = f"gqa/{entry_id}.jpg"
        full_img_path = os.path.join(output_dir, "images", img_path)
        try:
            save_image(img_index[image_id], full_img_path)
            entries.append(make_entry(entry_id, img_path, sample["question"]))
        except Exception as e:
            if i < 5:
                print(f"  Skip {entry_id}: {e}")
            continue

    print(f"  [GQA] Got {len(entries)} samples")
    return entries


def load_coco(count, output_dir, seed):
    """COCO: image + caption → 转成 'Describe this image.' prompt。"""
    from datasets import load_dataset
    print(f"\n[COCO] Loading (target: {count})...")

    # 用 2014_captions config, 每张图只出现一次
    ds = load_dataset("HuggingFaceM4/COCO", "2014_captions", split="train",
                       trust_remote_code=True)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:count]

    # 多样化的 prompt 模板（避免所有 COCO 都是同一个 prompt）
    prompts = [
        "Describe this image in detail.",
        "What do you see in this image?",
        "Provide a detailed description of what is shown in this image.",
        "Please describe the contents of this image.",
        "What is happening in this image?",
    ]

    entries = []
    img_dir = os.path.join(output_dir, "images", "coco")
    os.makedirs(img_dir, exist_ok=True)

    for i, idx in enumerate(tqdm(indices, desc="COCO")):
        sample = ds[idx]
        entry_id = f"coco_{i:06d}"
        img_path = f"coco/{entry_id}.jpg"
        full_img_path = os.path.join(output_dir, "images", img_path)
        prompt = prompts[i % len(prompts)]
        try:
            save_image(sample["image"], full_img_path)
            entries.append(make_entry(entry_id, img_path, prompt))
        except Exception as e:
            if i < 5:
                print(f"  Skip {entry_id}: {e}")
            continue

    print(f"  [COCO] Got {len(entries)} samples")
    return entries


def load_charxiv(count, output_dir, seed):
    """CharXiv: image + reasoning_q，图表推理。"""
    from datasets import load_dataset
    print(f"\n[CharXiv] Loading (target: {count})...")

    entries = []
    img_dir = os.path.join(output_dir, "images", "charxiv")
    os.makedirs(img_dir, exist_ok=True)

    idx_counter = 0
    for split_name in ["validation", "test"]:
        ds = load_dataset("princeton-nlp/CharXiv", split=split_name)
        for sample in tqdm(ds, desc=f"CharXiv {split_name}"):
            if idx_counter >= count:
                break
            # 用 reasoning_q 作为 prompt（是完整的问题文本）
            question = sample.get("reasoning_q", "")
            if not question or not question.strip():
                continue

            entry_id = f"charxiv_{idx_counter:06d}"
            img_path = f"charxiv/{entry_id}.jpg"
            full_img_path = os.path.join(output_dir, "images", img_path)
            try:
                save_image(sample["image"], full_img_path)
                entries.append(make_entry(entry_id, img_path, question))
                idx_counter += 1
            except Exception as e:
                if idx_counter < 5:
                    print(f"  Skip {entry_id}: {e}")
                continue

    print(f"  [CharXiv] Got {len(entries)} samples")
    return entries


def load_mmmu_pro(count, output_dir, seed):
    """MMMU_Pro: 多选长文本问答。用 standard (10 options) config。"""
    from datasets import load_dataset
    print(f"\n[MMMU_Pro] Loading (target: {count})...")

    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")

    entries = []
    img_dir = os.path.join(output_dir, "images", "mmmu_pro")
    os.makedirs(img_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(ds, desc="MMMU_Pro")):
        if i >= count:
            break

        # 找第一个非空的 image
        image = None
        for img_key in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
            if sample.get(img_key) is not None:
                image = sample[img_key]
                break
        if image is None:
            continue

        # 构建长文本 prompt：问题 + 选项 + 要求解释
        question = sample["question"]
        try:
            options = json.loads(sample["options"])
        except (json.JSONDecodeError, TypeError):
            options = []

        options_text = ""
        for j, opt in enumerate(options):
            letter = chr(ord('A') + j)
            options_text += f"\n{letter}. {opt}"

        prompt = f"{question}\n\nOptions:{options_text}\n\nPlease select the correct answer and provide a detailed explanation."

        entry_id = f"mmmu_pro_{i:06d}"
        img_path = f"mmmu_pro/{entry_id}.jpg"
        full_img_path = os.path.join(output_dir, "images", img_path)
        try:
            save_image(image, full_img_path)
            entries.append(make_entry(entry_id, img_path, prompt))
        except Exception as e:
            if i < 5:
                print(f"  Skip {entry_id}: {e}")
            continue

    print(f"  [MMMU_Pro] Got {len(entries)} samples")
    return entries


def load_mm_mt_bench(count, output_dir, seed):
    """MM-MT-Bench: 多轮多模态对话。取第一轮 user prompt。"""
    from datasets import load_dataset
    print(f"\n[MM-MT-Bench] Loading (target: {count})...")

    ds = load_dataset("mistralai/MM-MT-Bench", split="eval")

    entries = []
    img_dir = os.path.join(output_dir, "images", "mm_mt_bench")
    os.makedirs(img_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(ds, desc="MM-MT-Bench")):
        if i >= count:
            break

        # 解析 conversation JSON，取第一轮 user prompt
        try:
            conv = json.loads(sample["conversation"])
        except (json.JSONDecodeError, TypeError):
            continue

        # 找第一个 user turn 的 text content
        prompt = ""
        for turn in conv:
            if turn.get("role") == "user":
                content = turn.get("content", [])
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        prompt = part.get("text", "")
                        break
                if prompt:
                    break
        if not prompt:
            continue

        entry_id = f"mm_mt_bench_{i:06d}"
        img_path = f"mm_mt_bench/{entry_id}.jpg"
        full_img_path = os.path.join(output_dir, "images", img_path)
        try:
            save_image(sample["image"], full_img_path)
            entries.append(make_entry(entry_id, img_path, prompt))
        except Exception as e:
            if i < 5:
                print(f"  Skip {entry_id}: {e}")
            continue

    print(f"  [MM-MT-Bench] Got {len(entries)} samples")
    return entries


def load_convbench(count, output_dir, seed):
    """ConvBench: 多轮对话。图片在 HF dataset，问答在 xlsx 里。"""
    from datasets import load_dataset
    print(f"\n[ConvBench] Loading (target: {count})...")

    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [ConvBench] SKIP: needs pandas + openpyxl. pip install pandas openpyxl")
        return []

    # 下载 xlsx 标注
    try:
        xlsx_path = hf_hub_download(
            repo_id="liushuo12345/ConvBench",
            filename="ConvBench.xlsx",
            repo_type="dataset"
        )
        df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"  [ConvBench] SKIP: Failed to load xlsx: {e}")
        return []

    # 加载图片
    ds_val = load_dataset("liushuo12345/ConvBench", split="validation")
    ds_test = load_dataset("liushuo12345/ConvBench", split="test")

    # 合并图片列表
    all_images = []
    for sample in ds_val:
        all_images.append(sample["image"])
    for sample in ds_test:
        all_images.append(sample["image"])

    entries = []
    img_dir = os.path.join(output_dir, "images", "convbench")
    os.makedirs(img_dir, exist_ok=True)

    # 尝试从 xlsx 提取问题，匹配图片
    # xlsx 列名不确定，尝试常见的列名
    question_cols = [c for c in df.columns if "question" in c.lower() or "perception" in c.lower() or "prompt" in c.lower()]
    if not question_cols:
        # 如果找不到 question 列，用所有文本列的第一个
        text_cols = [c for c in df.columns if df[c].dtype == object]
        question_cols = text_cols[:1]

    if not question_cols:
        print("  [ConvBench] SKIP: Cannot find question columns in xlsx")
        return []

    q_col = question_cols[0]
    n = min(count, len(df), len(all_images))

    for i in range(n):
        question = str(df.iloc[i][q_col]).strip()
        if not question or question == "nan":
            continue
        if i >= len(all_images):
            break

        entry_id = f"convbench_{i:06d}"
        img_path = f"convbench/{entry_id}.jpg"
        full_img_path = os.path.join(output_dir, "images", img_path)
        try:
            save_image(all_images[i], full_img_path)
            entries.append(make_entry(entry_id, img_path, question))
        except Exception as e:
            if i < 5:
                print(f"  Skip {entry_id}: {e}")
            continue

    print(f"  [ConvBench] Got {len(entries)} samples")
    return entries


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare mixed dataset for MMSD v1 training")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for images and JSONL")
    parser.add_argument("--seed", type=int, default=42)

    # 各数据集采样数（0=跳过）
    parser.add_argument("--textvqa-count", type=int, default=20000)
    parser.add_argument("--gqa-count", type=int, default=10000)
    parser.add_argument("--coco-count", type=int, default=5000)
    parser.add_argument("--charxiv-count", type=int, default=2323)
    parser.add_argument("--mmmu-pro-count", type=int, default=1730)
    parser.add_argument("--convbench-count", type=int, default=240)
    parser.add_argument("--mm-mt-bench-count", type=int, default=92)

    args = parser.parse_args()
    random.seed(args.seed)

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)

    all_entries = []
    loaders = [
        ("MMMU_Pro", load_mmmu_pro, args.mmmu_pro_count),
        ("MM-MT-Bench", load_mm_mt_bench, args.mm_mt_bench_count),
        ("CharXiv", load_charxiv, args.charxiv_count),
        ("ConvBench", load_convbench, args.convbench_count),
        ("GQA", load_gqa, args.gqa_count),
        ("TextVQA", load_textvqa, args.textvqa_count),
        ("COCO", load_coco, args.coco_count),
    ]

    for name, loader_fn, count in loaders:
        if count <= 0:
            print(f"\n[{name}] Skipped (count=0)")
            continue
        try:
            entries = loader_fn(count, args.output_dir, args.seed)
            all_entries.extend(entries)
        except Exception as e:
            print(f"\n[{name}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Skipping {name}, continuing with other datasets...")

    # 打乱顺序
    random.shuffle(all_entries)

    # 写入 JSONL
    output_jsonl = os.path.join(args.output_dir, "mixed_dataset.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Done! Total: {len(all_entries)} samples")
    print(f"JSONL: {output_jsonl}")
    print(f"Images: {os.path.join(args.output_dir, 'images')}/")
    print(f"{'='*60}")

    # 统计各数据集占比
    from collections import Counter
    dataset_counts = Counter()
    for e in all_entries:
        prefix = e["id"].rsplit("_", 1)[0]
        dataset_counts[prefix] += 1
    print("\nDataset breakdown:")
    for name, cnt in dataset_counts.most_common():
        pct = cnt / len(all_entries) * 100
        print(f"  {name:20s}: {cnt:6d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()