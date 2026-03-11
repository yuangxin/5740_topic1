# AIMS5740 Project 1 - Data Selection + RL for LLMs (Math/STEM)

## 项目概述

本项目使用数据筛选和强化学习（GRPO）来提升 LLM 在数学推理任务（GSM8K）上的表现。

### Pipeline
1. **数据筛选** - 筛选 GSM8K 数据集，移除低质量样本
2. **监督微调（SFT）** - 使用 llama-factory 对 Qwen2.5-3B 进行微调
3. **强化学习（RL）** - 使用 GRPO 算法进一步优化模型
4. **评估** - 对比 Base、SFT、RL 三个模型的性能

---

## 项目结构

```
5740 project/
├── data/                           # 数据目录
│   ├── filtered_gsm8k_train.jsonl  # 筛选后的训练数据
│   ├── filtered_gsm8k_test.jsonl   # 筛选后的测试数据
│   └── evaluation_results.json     # 评估结果
├── models/                         # 模型目录
│   ├── qwen2.5-3b-sft/             # SFT 模型
│   └── qwen2.5-3b-rl/              # RL 模型
├── scripts/                        # 脚本目录
│   ├── 01_filter_data.py           # 数据筛选脚本
│   ├── 03_rl_train.py              # RL 训练脚本
│   └── 04_evaluate.py              # 评估脚本
├── configs/
│   └── sft_config.yaml             # SFT 训练配置
├── run_all.bat                     # Windows 完整运行脚本
└── run_all.sh                      # Linux/Mac 完整运行脚本
```

---

## 环境配置

### 1. 创建虚拟环境

```bash
cd "C:\Users\14869\Desktop\5740 project"
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft trl
pip install llama-factory
pip install evaluate tqdm
```

### 3. 下载 Base 模型

模型会在首次使用时自动从 HuggingFace 下载。

---

## 快速开始

### 运行完整流程

**Windows:**
```bash
run_all.bat
```

**Linux/Mac:**
```bash
chmod +x run_all.sh
./run_all.sh
```

### 分步运行

#### 步骤 1: 数据筛选
```bash
python scripts/01_filter_data.py
```

#### 步骤 2: SFT 训练
```bash
llamafactory-cli train configs/sft_config.yaml
```

#### 步骤 3: RL 训练
```bash
python scripts/03_rl_train.py
```

#### 步骤 4: 评估
```bash
python scripts/04_evaluate.py
```

---

## 奖励函数设计

本项目使用了 **2 个组件** 的奖励函数：

### 1. 正确性奖励 (Correctness Reward)
```python
if abs(pred - gt) < 0.01:
    reward = 1.0      # 完全正确
elif abs(pred - gt) < 0.1:
    reward = 0.5      # 接近正确
else:
    reward = -0.5     # 错误
```

### 2. 格式/推理结构奖励 (Format Reward)
```python
has_reasoning = any(keyword in answer for keyword in ["first", "then", "because", "step"])
has_format = "####" in answer

if has_reasoning and has_format:
    reward = 0.5
elif has_reasoning or has_format:
    reward = 0.2
else:
    reward = -0.3
```

---

## 预期结果

| 模型 | 预期准确率 |
|------|-----------|
| Base (Qwen2.5-3B) | ~70-75% |
| SFT | ~80-85% |
| RL | ~82-88% |

---

## 训练时间估计（单张 GPU）

| 步骤 | 预计时间 |
|------|---------|
| 数据筛选 | < 5 分钟 |
| SFT 训练 | ~1-2 小时 |
| RL 训练 | ~2-3 小时 |
| 评估 | < 10 分钟 |

---

## GPU 要求

- 最低：8GB VRAM
- 推荐：12GB+ VRAM
- 架构：CUDA 11.8+ 支持

---

## 故障排查

### 问题 1: OOM (显存不足)
**解决方案：** 减少 `per_device_train_batch_size`

### 问题 2: RL 训练不稳定
**解决方案：** 在 `scripts/03_rl_train.py` 中降低 `learning_rate` 到 `5e-7`

### 问题 3: llama-factory 安装失败
**解决方案：**
```bash
pip install llama-factory -i https://pypi.org/simple
```

---

## 相关论文/工作

- [MetaMath / MetaMathQA](https://meta-math.github.io/)
- [GRPO (Group Relative Policy Optimization)](https://huggingface.co/learn/cookbook/en/fine_tuning_llm_grpo_trl)
- [Verifier-based RL for math reasoning](https://arxiv.org/abs/2110.14168)

---

## 数据集信息

- **训练数据**: GSM8K (筛选后约 7K 样本)
- **测试数据**: GSM8K (筛选后约 1K 样本)
- **下载**: https://huggingface.co/datasets/openai/gsm8k

---

## 模型信息

- **Base 模型**: Qwen/Qwen2.5-3B-Instruct
- **下载**: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- **参数量**: 3B

---

## 交付物清单

- [x] 训练脚本
- [x] 评估脚本
- [x] 可复现的 README (本文件)
- [ ] 最终报告 (6-8 页)
- [ ] 演示文稿
