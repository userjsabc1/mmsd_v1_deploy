# MMSD v1 实施计划：基于 EAGLE-3 的 VLM 自推测解码

## 核心思路

在 EAGLE-3 的 draft model 架构上**仅修改输入**：
1. Hidden states 来自**随机 3 层**（训练时随机，推理时固定），而非 EAGLE-3 的固定 {layer_2, layer_N//2, layer_N-3}
2. 序列中 visual token 从全量 V（576~1176）压缩到 **32 个**（从 top-64 by delta 中随机采样）

架构、训练流程、推理流程、tree verification 与 EAGLE-3 完全一致。

---

## 1. 与 EAGLE-3 的差异一览

| | EAGLE-3 | MMSD v1（ours） |
|---|---|---|
| 目标模型 | Text-only LLM（LLaMA 3.1 8B） | VLM（LLaVA-1.5-13B） |
| Hidden states 来源层 | 固定 3 层: {2, N//2, N-3} | 训练时**随机 3 层**；推理时固定 |
| 序列构成 | 全部 text tokens | **32 visual tokens** + 全部 text tokens |
| Visual token 选择 | N/A | 跨层 delta top-64 → 随机抽 32 |
| Fusion layer | Linear(3D → D) | Linear(3D → D)（**不变**） |
| Decoder layer | 1 层 LlamaDecoderLayeremb | **不变** |
| LM head | 独立 draft head, 32K vocab | **不变** |
| Loss | KL(draft ∥ target), 0.8^step 衰减 | **LK Loss**（λ·KL + (1-λ)·TV，直接优化 acceptance rate） |
| 自回归训练步数 | 7 步 + growing KV cache | **不变** |
| Tree verification | 25 candidates, depth 5 | **不变** |

**改动量极小**：仅 `dataprepare()`（输入构造 + 序列过滤）、`modeling_*_kv.py`（hidden states 收集）、loss 函数（KL → LK Loss）。

---

## 1.5 关键文献支撑

| 借鉴来源 | 借鉴内容 | 在本方案中的角色 |
|----------|----------|----------------|
| **EAGLE-3** [SafeAI'25] | 多层 hidden states fusion + 1-layer draft + tree verification | 核心架构（完全复用） |
| **ViSpec** [Kang'25, NeurIPS] | 证明 1-layer draft model + 冗余 visual token → attention 退化为均值 | 理论动机：必须压缩 visual token |
| **ShortV** [Yuan'25] | 跨层 hidden states delta = visual token 信息贡献的可靠度量 | delta top-64 预筛策略的信号来源 |
| **SWIFT** [Xia'24, ICLR'25] | 运行时随机层采样作为自投机解码策略 | 启发随机层采样训练 |
| **FastV** [Chen'24] | attention score top-k 剪枝 visual token | 对比基线（我们的方法更鲁棒） |
| **LK Losses** [Samarin'26] | λ·KL + (1-λ)·TV 直接优化 acceptance rate | 训练 loss（替代 KL） |
| **MSD** [Lin'25] | visual token 用原始 embedding、text token 用 displacement concatenation | 模态解耦原则（可选融入） |

---

## 2. 训练数据构造

### 2.1 离线预处理（一次性，无需训练）

对每个训练样本 `(image, text_prompt, text_response)`：

```python
# 1. Target VLM prefill（eager attention，收集所有层 hidden states）
with torch.no_grad():
    outputs = target_vlm(
        input_ids=input_ids,
        images=images,
        attn_implementation="eager",
        output_hidden_states=True,   # 收集所有层
    )

# 2. 收集
all_hidden_states = outputs.hidden_states          # (L+1) × [B, T+V, D]
target_logits = outputs.logits                      # [B, T+V, vocab_size]
visual_positions = get_visual_token_positions(input_ids)  # [V]

# 3. 计算 visual token 跨层 delta
delta = torch.zeros(V)
for l in range(L):
    h_l = all_hidden_states[l][:, visual_positions]      # [B, V, D]
    h_l1 = all_hidden_states[l + 1][:, visual_positions]  # [B, V, D]
    delta += (1 - F.cosine_similarity(h_l, h_l1, dim=-1)).mean(0)  # [V]

# 4. 选出 delta 最大的 64 个 visual token
top64_indices = delta.topk(64).indices  # [64]

# 5. 存储到磁盘
torch.save({
    'all_hidden_states': all_hidden_states,  # 或只存需要的层以省空间
    'target_logits': target_logits,
    'input_ids': input_ids,
    'visual_positions': visual_positions,
    'top64_visual_indices': top64_indices,
}, f'cache/{sample_id}.pt')
```

### 2.2 在线采样（每个 training step）

```python
def dataprepare(self, input_ids, attention_mask, loss_mask, cached_data):
    all_hidden_states = cached_data['all_hidden_states']
    target_logits = cached_data['target_logits']
    visual_positions = cached_data['visual_positions']
    top64 = cached_data['top64_visual_indices']
    text_positions = cached_data['text_positions']
    L = len(all_hidden_states) - 1  # 总层数

    # ===== 随机采样 =====
    # 1. 从 top-64 中随机抽 32 个 visual token
    perm = torch.randperm(64)[:32]
    selected_visual = top64[perm]  # [32]

    # 2. 随机抽 3 层（不放回）
    layer_indices = torch.randperm(L)[:3].sort().values  # [3], 排序保证一致性

    # ===== 构造 draft model 输入 =====
    # 3. 拼接 3 层 hidden states → [B, T+V, 3D]
    hs = torch.cat([
        all_hidden_states[layer_indices[0]],
        all_hidden_states[layer_indices[1]],
        all_hidden_states[layer_indices[2]],
    ], dim=-1)  # [B, T+V, 3D]

    # 4. 过滤序列：只保留 32 visual tokens + 全部 text tokens
    keep_positions = torch.cat([text_positions, selected_visual]).sort().values
    hs = hs[:, keep_positions, :]          # [B, T+32, 3D]
    target = target_logits[:, keep_positions, :]  # [B, T+32, vocab]
    input_ids_filtered = input_ids[:, keep_positions]  # [B, T+32]

    # 5. Shift（teacher forcing）
    target = padding(target, left=False)       # 左移一位
    input_ids_filtered = padding(input_ids_filtered, left=False)

    return hs, target, loss_mask, input_ids_filtered
```

### 2.3 随机采样 = 数据增强

同一个训练样本在不同 epoch 中：
- 看到不同的 3 层组合（如 epoch 1: {3, 15, 28}, epoch 2: {1, 20, 31}）
- 看到不同的 32 个 visual token（从 top-64 中随机）

**效果**：draft model 被迫在各种层组合 + 各种 visual token 子集下都能恢复 target 输出。
它隐式学会了"哪些信息是冗余的"——因为无论怎么抽，它都需要输出正确答案。

---

## 3. Draft Model 架构（= EAGLE-3，不改）

```python
class Model(nn.Module):
    """完全复用 EAGLE-3 的 draft model 结构"""
    def __init__(self, config):
        # Fusion: 3层 hidden states → 1层
        self.fc = nn.Linear(config.target_hidden_size * 3, self.hidden_size, bias=False)
        # Token embedding（冻结，从 target model 复制）
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # 核心：1 层 decoder，双输入（token_emb + hidden_states）
        self.midlayer = LlamaDecoderLayeremb(config)
        self.norm = LlamaRMSNorm(config.hidden_size)
        # 独立 LM head，reduced vocabulary
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
```

### LlamaDecoderLayeremb 的 forward（双输入）

```
input_emb  ──→ RMSNorm ──┐
                          ├─ concat → [B, seq, 2D] → Q/K/V → Self-Attention → MLP → output
hidden_states → RMSNorm ──┘
```

- Q/K/V 的 Linear 层输入维度是 `2 * hidden_size`（接收拼接后的向量）
- 这使得 draft model 能同时利用 token 语义（from embedding）和 target 内部表征（from hidden states）

### 训练时 7 步自回归模拟

```
Step 0: input = (target_hidden_states, token_embedding)     → draft 预测 token_1
Step 1: input = (draft_own_hidden_states, token_1_embedding) → draft 预测 token_2
Step 2: input = (draft_own_hidden_states, token_2_embedding) → draft 预测 token_3
...
Step 6: input = (draft_own_hidden_states, token_6_embedding) → draft 预测 token_7

每步计算 LK Loss（λ·KL + (1-λ)·TV），权重 0.8^step 指数衰减
Growing KV Cache 在步间累积，模拟真实推理行为
```

**关键**：只有 Step 0 使用 target model 的 hidden states；Step 1-6 使用 draft model 自身的 hidden states。
这与推理时的行为完全一致（ViSpec/EAGLE 的 training-time testing 思想）。

---

## 4. 监督信号与 LK Loss

| 要素 | 设计 |
|------|------|
| **Supervision** | Target model 的 output logits（soft distribution） |
| **Loss** | **LK Loss**：L_LK^λ = λ·KL(p∥q) + (1-λ)·TV(p,q)，直接优化 acceptance rate |
| **"哪些层该跳"** | 不显式教——随机 3 层训练隐式学会鲁棒性 |
| **"哪些 token 该留"** | 不显式教——delta top-64 预筛 + 随机 32 训练隐式学会 |
| **数据来源** | Target model 自身生成（sampling，非 greedy，同 ViSpec） |

### 4.1 LK Loss（替代 KL Loss）

传统 KL 散度是 acceptance rate 的松散上界，优化 KL 不等于直接优化推测解码性能。
LK Loss [Samarin'26] 通过自适应混合 KL 和 TV 距离，**直接逼近 acceptance rate 的优化目标**。

```python
def lk_loss(draft_logits, target_logits, eta=3.0):
    """
    L_LK^λ = λ·KL(p||q) + (1-λ)·TV(p,q)
    λ = exp(-η · sg[α])，α = Σ min(p, q) ≈ acceptance rate

    α 低时（draft 差）→ λ≈1 → KL 主导（稳定梯度，方向正确）
    α 高时（draft 好）→ λ≈0 → TV 主导（直接优化 acceptance rate）
    原论文在 EAGLE 上验证，提升 3.8-8.2% accepted length
    """
    p = F.softmax(target_logits, dim=-1)
    q = F.softmax(draft_logits, dim=-1)

    alpha = torch.sum(torch.min(p, q), dim=-1).mean()  # ≈ acceptance rate
    lam = torch.exp(-eta * alpha.detach())              # adaptive blending

    kl = F.kl_div(q.log(), p, reduction='batchmean')
    tv = 0.5 * torch.sum(torch.abs(p - q), dim=-1).mean()

    return lam * kl + (1 - lam) * tv
```

### 4.2 训练 Loss 完整公式

```
L_total = Σ_{step=0}^{6} 0.8^step × LK_Loss(draft_logits_step, target_logits_step)
```

7 步自回归，每步用 LK Loss，权重 0.8^step 指数衰减（远步误差累积大，权重低）。

### 4.3 核心 Insight

不需要 oracle label 告诉模型"应该跳第 5、12、27 层"——
通过随机训练，模型自己学会了从任意 3 层 + 32 tokens 中提取足够信息。
LK Loss 比 KL 更直接地优化 draft 质量，在信息不完整（32 tokens + 3 layers）的条件下尤为重要。

---

## 5. 推理流程

```
输入: image + text prompt
         │
         ▼
┌─ Target VLM Prefill（eager attention）───────────────────┐
│  1. Full model prefill → 提取 3 层 hidden states          │
│     推理时用固定层: {layer_2, layer_N//2, layer_N-3}      │
│     （与 EAGLE-3 默认一致，或通过 ablation 选最优 3 层）   │
│  2. 计算 visual token delta → 选 top-32                   │
│  3. 过滤序列: 32 visual + 全部 text                       │
│  4. concat 3 层 hidden states → fusion → draft input      │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Draft Phase（EAGLE-3 tree-based generation）────────────┐
│  Draft model 生成候选树:                                   │
│  - 第 1 个 token: target hidden states + token embedding  │
│  - 后续 token: draft 自身 hidden states + token embedding │
│  - 默认 25 candidates, depth 5                            │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Verify Phase ───────────────────────────────────────────┐
│  Full target model 验证全部候选（使用完整 KV Cache）:       │
│  - target model 的 KV Cache 包含全部 V 个 visual token    │
│  - 用 tree attention mask 做单次前向                       │
│  - 选最长匹配前缀，更新 KV Cache                          │
└──────────────────────────────────────────────────────────┘
         │
    重复直到 EOS
```

**KV Cache 不冲突**：
- Target model 的 KV Cache: 全部 V visual tokens + T text tokens（标准，不修改）
- Draft model 的 KV Cache: 32 visual tokens + T text tokens（独立 1 层 cache）
- 两者完全独立，不共享

---

## 6. 实现计划

### 基于 EAGLE repo 的修改清单

| 文件 | 修改 | 工作量 |
|------|------|--------|
| `eagle/traineagle3/modeling_*_kv.py` | hidden_states 收集: 固定 3 层 → 支持随机/可配置 | 小 |
| `eagle/traineagle3/cnets.py` → `dataprepare()` | 新增: visual token delta 计算 + 随机采样 + 序列过滤 | 中 |
| `eagle/traineagle3/cnets.py` → `forward()` | 适配过滤后的序列长度（attention mask, position_ids） | 小 |
| `eagle/model/modeling_*_kv.py` | 推理时的 hidden_states 收集（同训练端） | 小 |
| `eagle/model/cnets.py` → `topK_genrate()` | 适配过滤后的序列（visual token 位置管理） | 中 |
| `eagle/model/utils.py` | `initialize_tree()` 适配 VLM 输入 | 中 |
| **新增** `eagle/visual_utils.py` | visual token delta 计算、top-k 选择、序列过滤工具 | 小 |
| **新增** VLM 加载逻辑 | 加载 LLaVA-1.5-13B，处理 vision tower + multi_modal_projector（参考 method/swift 的做法） | 中 |

### 目标模型

**首选**: LLaVA-1.5-13B
- 基于 Vicuna-13B（LLaMA-2-13B），40 层 decoder，hidden_dim=5120
- 576 visual tokens（CLIP ViT-L/14@336px）
- MMSpec 已有完整评测 baseline（MSD 3.66x, ViSpec 3.00x）
- 13B 规模层间冗余更高（SWIFT 论文证明越大越稀疏），更适合验证 visual token 压缩 + 随机层采样的效果
- 7B 模型层数少（32 层），随机 3 层的信息量可能不足以支撑 draft 质量

**备选**: Qwen2.5-VL-7B-Instruct（MMSpec 另一条评测线）

### 训练数据

| 阶段 | 数据集 | 样本数 | 说明 |
|------|--------|--------|------|
| Stage 1: Text-only | ShareGPT | 68K | 建立文本基础（同 EAGLE-3/ViSpec） |
| Stage 2: Multimodal | LLaVA-Instruct + 合成长回复 | 68K+ | 用 target VLM sampling 生成长回复（同 ViSpec） |

Stage 2 的训练数据需要用 target VLM **sampling**（非 greedy）生成：
- 防止 draft model 学到 hidden state ↔ embedding 的一一对应（ViSpec 的发现）
- Prompt 追加 "请详细分析" 生成长回复（长回复场景加速收益更大）

### 里程碑

| 阶段 | 目标 | 预计 |
|------|------|------|
| M0 | fork EAGLE repo，跑通 EAGLE-3 text-only baseline | 1-2 天 |
| M1 | Target model 换成 LLaVA-1.5-13B，跑通 prefill + hidden states 提取 | 2-3 天 |
| M2 | 实现 visual token delta + 序列过滤 + 随机层采样 | 1-2 天 |
| M3 | 修改 dataprepare，Stage 1 text-only 训练 | 2-3 天 |
| M4 | Stage 2 multimodal 训练 | 3-5 天 |
| M5 | 推理 pipeline + speedup 测试 | 2-3 天 |
| M6 | Benchmark 实验 + ablation | 5-7 天 |

---

## 7. 实验设计

### 7.1 Baseline 对比

| 方法 | 类别 | 说明 |
|------|------|------|
| Autoregressive | 无加速 | Full LLaVA-1.5-13B |
| EAGLE-3 (原版) | Text-only SD | 不处理 visual token，用固定 3 层 |
| EAGLE-3 (全 visual) | 保留全部 visual token | 测试 visual 压缩的必要性 |
| ViSpec | VLM SD (separate draft) | Q-Former 压缩到 1 token（MMSpec 报告 3.00x） |
| MSD | VLM SD (modality decoupled) | 模态解耦（MMSpec 报告 3.66x） |
| **MMSD v1 (ours)** | VLM Self-SD | 随机层 + 32 visual tokens |

### 7.2 Ablation

| 实验 | 变量 | 目的 |
|------|------|------|
| A1: 层选择 | 固定 {2,N//2,N-3} vs 随机 3 层 (训练) | 随机训练是否优于固定层 |
| A2: visual token 数量 | 8 / 16 / 32 / 64 / all | 压缩比 vs 质量的最优点 |
| A3: 层数量 | 1 / 2 / 3 / 5 层 | 几层信息足够 |
| A4: visual 选择策略 | top-k by delta vs 随机 vs 均匀采样 | delta 信号的有效性 |
| A5: top-64 预筛 | 从 top-64 抽 vs 从全量抽 | 预筛是否有帮助 |
| A6: Hidden States 回归 | LK Loss only vs LK Loss + SmoothL1(draft_hs, target_hs) | EAGLE-2 用过但 EAGLE-3 去掉了；在信息压缩条件下是否有帮助 |
| A7: Loss 选择 | KL vs LK Loss vs CE | LK Loss 相比 KL 的增益 |

### 7.3 评测

| 维度 | 指标 |
|------|------|
| 加速 | wall-clock speedup, tokens/s, mean accepted length |
| 质量 | task accuracy (lossless check), acceptance rate |
| 数据集 | ScienceQA, TextVQA, MM-Vet, MMMU, COCO Caption |

---

## 8. Motivation & Innovation

### 8.1 问题：VLM 推测解码的核心瓶颈

EAGLE-3 是当前最强的 text-only 推测解码方法（LLaMA-3 70B: 4.4x 加速）。
但它**无法直接适用于 VLM**。原因：

1. **信噪比崩塌**：VLM 的 576-1176 个 visual token 中 ~95% 是冗余的（FastV/VisiPruner 已证明），
   但 EAGLE-3 的 1 层 draft model 无法区分有效和冗余 visual token。
   ViSpec [Kang'25] 从理论上证明：**浅层模型（1-layer attention）面对大量冗余 visual token 时退化为取均值**——
   R 个冗余 image embedding 占主导时，attention 权重趋向均匀 1/R，有效信息被淹没。
   这意味着直接将全量 visual token 喂给 EAGLE-3 draft model 会导致 draft 质量严重下降。

2. **现有方案的不足**：
   - ViSpec：引入 Q-Former 压缩 visual token 为 1 个 → 需要额外训练 vision adaptor，丢失空间信息
   - FastV：基于 attention score 排序（需要 attention map，与 eager/flash 实现耦合），且只做剪枝不做推测解码
   - MSD：模态解耦有效，但 draft 仍处理全量 visual token

### 8.2 核心观察：随机采样即最优信号

现有 visual token 压缩方法（FastV/ShortV/VisiPruner）都试图精确判定"哪些 token 重要"——
但**精确判定本身就是一种过拟合**：不同样本、不同层、不同任务下重要的 token 完全不同。

我们的关键观察：
- 跨层 hidden states delta（cosine distance）可以快速识别**变化最大的 64 个 visual token**（信息理论保证：变化大 = 承载更多信息流动）
- 但从这 64 个中选哪 32 个**并不重要**——因为 top-64 内部信息高度重叠
- **随机采 32 个 + 训练时持续随机 = 最强的数据增强**：
  draft model 被迫在任意子集下都能恢复 target 质量，隐式学会视觉冗余结构

这比 FastV 的"attention score top-k"更鲁棒：
- FastV 依赖特定层的 attention map（实验证明不同层排序不一致）
- 我们基于**全层 delta 累积**（稳定）+ **随机训练**（泛化），无需 attention map

### 8.3 方法总结

在 EAGLE-3 的训练过程中引入两个随机性维度：
1. **随机层采样**（借鉴 SWIFT）：每步随机选 3 层提取 hidden states → draft model 对层选择鲁棒
2. **随机 visual token 采样**：从 delta top-64 中随机选 32 → draft model 对 token 子集鲁棒

推理时只需 32 个 visual token + 3 层 hidden states，draft model 仍能生成高质量草稿。
配合 LK Loss（直接优化 acceptance rate），在信息大幅压缩的条件下最大化 draft 效率。

### 8.4 Contribution（2 个，紧密关联）

**贡献 1：Random Visual Token Subsampling 训练策略**

发现 VLM 推测解码中 visual token 的精确选择并不关键——
基于跨层 delta 的粗粒度预筛（top-64）+ 随机子采样（32）+训练时持续随机，
即可使 1 层 draft model 在仅 32/576 个 visual token 下恢复 target 质量。
- 这直接挑战 FastV 等方法的"精确排序"范式
- 同时解决了 ViSpec 指出的"浅层模型 + 冗余 visual token = 退化"问题——不是靠更深的模型，而是靠减少冗余输入

**贡献 2：将 EAGLE-3 扩展到 VLM 场景**

通过随机层采样 + visual token 压缩，以**最小改动**（仅修改 dataprepare 输入）将 EAGLE-3 适配到 VLM：
- 架构不变：fusion layer、LlamaDecoderLayeremb、draft LM head 完全复用
- 训练流程不变：7 步自回归模拟 + growing KV cache
- 推理流程不变：tree-based generation + verification
- 仅输入不同：3 层随机 → 固定；序列从全量 → 32 visual + 全部 text

### 8.5 vs 现有工作

| 方法 | 差异 |
|------|------|
| **EAGLE-3** | 仅支持 text-only LLM；全量 visual token 导致 VLM draft 质量崩塌 |
| **ViSpec** | 需训练独立 Q-Former + vision adaptor；我们零额外架构，复用 EAGLE-3 |
| **FastV** | 依赖单层 attention map 排序 visual token（不稳定、需 attention map）；我们用跨层 delta（稳定）+ 随机训练（泛化），无需 attention map |
| **LayerSkip** | 需重训 target model（layer dropout + early exit loss）；我们只训 draft head |
| **MSD** | 模态解耦有效但 draft 仍处理全量 visual token；我们压缩到 32 个 |
| **ShortV** | 只做推理时冻结/剪枝，不做推测解码；我们将 delta 信号用于 draft model 训练 |
