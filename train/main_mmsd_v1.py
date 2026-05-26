"""
MMSD v1 训练入口：用 target model 生成的数据训练 draft model。

训练数据必须先用 scripts/generate_training_data.py 生成：
  target model 跑一遍 prompt → 用模型自己的回复作为训练数据。
  原因：draft model 学的是 target model 的输出分布，不能用人写的 response。

支持两种数据格式：
  - JSONL: scripts/generate_training_data.py 生成的格式（推荐）
  - JSON:  LLaVA-Pretrain 原始格式（调试用）

用法：
  # 第 1 步：生成训练数据
  python scripts/generate_training_data.py \
      --mode hf --model-path /path/to/llava-1.5-13b-hf \
      --input-path /path/to/blip_laion_cc_sbu_558k.json \
      --image-dir /path/to/LLaVA-Pretrain \
      --output-path /path/to/mmsd_v1_train_data.jsonl

  # 第 2 步：训练
  deepspeed --num_gpus=1 train/main_mmsd_v1.py \
      --basepath /path/to/llava-1.5-13b-hf \
      --trainpath /path/to/mmsd_v1_train_data.jsonl \
      --imagedir /path/to/LLaVA-Pretrain \
      --savedir /path/to/checkpoints \
      --deepspeed_config train/ds_config_eagle3_stage2.json
"""
import argparse
import deepspeed

parser = argparse.ArgumentParser(description='MMSD v1: Multimodal Draft Model Training')
parser.add_argument('--basepath', type=str, required=True, help='Path to LLaVA-1.5-13B')
parser.add_argument('--trainpath', type=str, required=True, help='Path to training json (e.g. blip_laion_cc_sbu_558k.json)')
parser.add_argument('--imagedir', type=str, required=True, help='Directory containing training images')
parser.add_argument('--testpath', type=str, default='', help='Optional test json')
parser.add_argument('--savedir', type=str, default='checkpoints/mmsd_v1')
parser.add_argument("--local_rank", type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

import json
import re
import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoProcessor
from safetensors import safe_open

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed
set_seed(0)

from train.model.cnets_mmsd_v1 import Model, padding, process_data, merge_dicts
from train.model.configs import EConfig

# ============================================================
# Config
# ============================================================
deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)

train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 20,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "train/configs/llava1.5_13b_eagle3_config.json",
    "gradient_checkpointing": True,
    # MMSD v1 超参
    "num_visual_tokens": 32,
    "visual_pre_filter_k": 64,
    "lk_eta": 3.0,
}


# ============================================================
# Dataset
# ============================================================
class MultimodalDataset(Dataset):
    """
    MMSD v1 训练数据集。

    训练数据应由 scripts/generate_training_data.py 生成：
    target model 对 prompt 生成回复 → 用模型自己的 response 作为训练数据。
    这保证了训练分布与推理时一致（on-policy）。

    支持 JSONL（推荐，generate_training_data.py 输出）和 JSON（LLaVA-Pretrain 原始格式）。
    """

    def __init__(self, data_path, image_dir, processor, max_len=2048):
        if data_path.endswith(".jsonl"):
            self.data = []
            with open(data_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        else:
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_len = max_len
        self.valid_indices = []
        for i, item in enumerate(self.data):
            if item.get('conversations') and item.get('image'):
                self.valid_indices.append(i)
        print(f"Loaded {len(self.valid_indices)} valid samples from {len(self.data)} total")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        item = self.data[self.valid_indices[idx]]

        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            return None

        # Build text prompt
        roles = {"human": "user", "gpt": "assistant"}
        source = item['conversations']
        if roles.get(source[0]["from"]) != "user":
            source = source[1:]

        text_parts = []
        for sentence in source:
            role = roles.get(sentence["from"], sentence["from"])
            content = sentence["value"]
            if role == "user":
                content_clean = content.replace("<image>", "").strip()
                if "<image>" in sentence["value"]:
                    text_parts.append(f"USER: <image>\n{content_clean}")
                else:
                    text_parts.append(f"USER: {content_clean}")
            else:
                text_parts.append(f"ASSISTANT: {content}")
        text = " ".join(text_parts)

        # Process
        try:
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=False)
        except Exception:
            return None

        input_ids = inputs["input_ids"].squeeze(0)
        if len(input_ids) > self.max_len:
            return None

        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values", None)

        # Loss mask: only assistant responses
        loss_mask = torch.zeros_like(input_ids)
        sep_assistant = "ASSISTANT:"
        sep = "</s>"
        decoded = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        parts = decoded.split(sep_assistant)
        cur_pos = 0
        for pi, part in enumerate(parts):
            if pi == 0:
                cur_pos += len(self.processor.tokenizer(part + sep_assistant, add_special_tokens=False).input_ids)
            else:
                response = part.split(sep)[0] if sep in part else part
                resp_len = len(self.processor.tokenizer(response, add_special_tokens=False).input_ids)
                end_pos = min(cur_pos + resp_len, len(loss_mask))
                loss_mask[cur_pos:end_pos] = 1
                cur_pos += len(self.processor.tokenizer(
                    part + (sep_assistant if pi < len(parts) - 1 else ""),
                    add_special_tokens=False
                ).input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "pixel_values": pixel_values,
        }


def collate_fn(features):
    features = [f for f in features if f is not None]
    if len(features) == 0:
        return None

    max_length = max(f['input_ids'].shape[0] for f in features)
    batch_input_ids = []
    batch_attention_mask = []
    batch_loss_mask = []
    batch_pixel_values = []

    for f in features:
        pad_len = max_length - f['input_ids'].shape[0]
        if pad_len > 0:
            batch_input_ids.append(torch.cat([f['input_ids'], torch.zeros(pad_len, dtype=f['input_ids'].dtype)]))
            batch_attention_mask.append(torch.cat([f['attention_mask'], torch.zeros(pad_len, dtype=f['attention_mask'].dtype)]))
            batch_loss_mask.append(torch.cat([f['loss_mask'], torch.zeros(pad_len, dtype=f['loss_mask'].dtype)]))
        else:
            batch_input_ids.append(f['input_ids'])
            batch_attention_mask.append(f['attention_mask'])
            batch_loss_mask.append(f['loss_mask'])
        if f['pixel_values'] is not None:
            batch_pixel_values.append(f['pixel_values'])

    batch = {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "loss_mask": torch.stack(batch_loss_mask),
    }
    if batch_pixel_values:
        batch["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
    return batch


# ============================================================
# Training
# ============================================================
config = EConfig.from_pretrained(train_config["config_path"])

# cache.pt: draft vocab 映射（如已存在则复用）
if not os.path.exists("cache.pt"):
    print("cache.pt not found, will be generated by model.scandata()")

processor = AutoProcessor.from_pretrained(args.basepath)
traindataset = MultimodalDataset(args.trainpath, args.imagedir, processor, max_len=train_config["max_len"])

if args.testpath and os.path.exists(args.testpath):
    testdataset = MultimodalDataset(args.testpath, args.imagedir, processor, max_len=train_config["max_len"])
else:
    testdataset = None

model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=False)
model.scandata(args.trainpath, args.basepath)

num_epochs = train_config["num_epochs"]
model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args, model=model, model_parameters=model.parameters()
)

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    print(f"MMSD v1 Training | epochs={num_epochs} | bs={train_config['bs']} | "
          f"visual_tokens={train_config['num_visual_tokens']} | lk_eta={train_config['lk_eta']}")

os.makedirs(args.savedir, exist_ok=True)

train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], sampler=train_sampler,
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)

if testdataset is not None:
    test_sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    test_loader = DataLoader(testdataset, batch_size=train_config["bs"], sampler=test_sampler,
                             num_workers=4, pin_memory=True, collate_fn=collate_fn)
else:
    test_loader = None


def find_max_state(directory, filename="zero_to_fp32.py"):
    max_a = -1
    if not os.path.exists(directory):
        return None, 0
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state(args.savedir)
if checkpoint_path:
    print(f"Resuming from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)

# ============================================================
# Training loop
# ============================================================
# 一个 training step 的流程：
#
# 1. 从 DataLoader 取一个 batch:
#      input_ids [B, T+V], attention_mask [B, T+V], loss_mask [B, T+V], pixel_values [B, C, H, W]
#
# 2. model_engine(batch) 调用 Model.forward():
#    a. dataprepare():
#       - target_model forward → all_hs (41层), logits
#       - 随机选 3 层 → concat [B, T+V, 15360]
#       - visual delta → 选 32 个 visual tokens
#       - filter → [B, T+32, 15360], target/input_ids/loss_mask 同步过滤
#       - shift: target 和 input_ids 左移一位
#    b. fc(15360→5120) → [B, T+32, 5120]
#    c. 7 步自回归:
#       - embed_tokens(input_ids) → input_emb
#       - midlayer(input_emb, hidden_states) → h_out
#       - lm_head(norm(h_out)) → draft_logits
#       - LK_Loss(draft_logits, target_logits)
#       - shift input_ids/target
#    d. return plosses[7], [], acces[7]
#
# 3. 加权求和: loss = Σ 0.8^i × plosses[i]
# 4. backward + optimizer step

for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch + 1)
    if global_rank == 0:
        print(f"=== Epoch {epoch} ===")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(train_loader, disable=(global_rank != 0))):
        if data is None:
            continue

        model.zero_grad()

        kwargs = {
            "input_ids": data["input_ids"].to(rank),
            "attention_mask": data["attention_mask"].to(rank),
            "loss_mask": data["loss_mask"],
        }
        if "pixel_values" in data:
            kwargs["pixel_values"] = data["pixel_values"].to(rank)

        try:
            plosses, vlosses, acces = model_engine(**kwargs)
        except (ValueError, RuntimeError) as e:
            if global_rank == 0 and batch_idx % 1000 == 0:
                print(f"Skipping batch {batch_idx}: {e}")
            continue

        # 加权求和: 0.8^step decay
        ploss_weight = [0.8 ** i for i in range(len(plosses))]
        loss = sum(ploss_weight[i] * plosses[i] for i in range(len(plosses)))

        model_engine.backward(loss)
        model_engine.step()

        if global_rank == 0:
            for i in range(len(plosses)):
                epoch_plosses[i].append(plosses[i].item())
            for i in range(len(acces)):
                epoch_acces[i].append(acces[i])
            if batch_idx % 100 == 0:
                lr = optimizer.optimizer.param_groups[0]["lr"]
                print(f"  step {batch_idx}: lr={lr:.2e}, "
                      f"loss_0={plosses[0].item():.4f}, acc_0={acces[0]:.4f}")

        # Mid-epoch checkpoint
        if batch_idx > 0 and batch_idx % 5000 == 0:
            if global_rank == 0:
                print(f"  Saving mid-epoch checkpoint at step {batch_idx}...")
            model_engine.save_16bit_model(
                f"{args.savedir}/state_{epoch}_step{batch_idx}", exclude_frozen_parameters=True
            )

    # Epoch summary
    for i in range(len(epoch_acces)):
        if epoch_acces[i]:
            acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
            deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
            if global_rank == 0:
                print(f"  Train Epoch {epoch}, step {i}, Acc: {acc_i.item():.4f}")

    for i in range(len(epoch_plosses)):
        if epoch_plosses[i]:
            loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
            deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
            if global_rank == 0:
                print(f"  Train Epoch {epoch}, step {i}, Loss: {loss_i.item():.4f}")

    # Eval
    if test_loader is not None:
        eval_acces = [[] for _ in range(model.length)]
        for data in tqdm(test_loader, disable=(global_rank != 0)):
            if data is None:
                continue
            with torch.no_grad():
                kwargs = {"input_ids": data["input_ids"].to(rank),
                          "attention_mask": data["attention_mask"].to(rank),
                          "loss_mask": data["loss_mask"]}
                if "pixel_values" in data:
                    kwargs["pixel_values"] = data["pixel_values"].to(rank)
                _, _, acces = model_engine(**kwargs)
                for i in range(len(acces)):
                    eval_acces[i].append(acces[i])
        for i in range(len(eval_acces)):
            if eval_acces[i]:
                acc_i = torch.tensor(eval_acces[i]).cuda().mean()
                deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
                if global_rank == 0:
                    print(f"  Eval Epoch {epoch}, step {i}, Acc: {acc_i.item():.4f}")

    torch.cuda.empty_cache()
    model_engine.save_16bit_model(f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True)
    if epoch % 5 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f"{args.savedir}/state_{epoch}")
