# scripts/03_rl_train.py
"""
RL 训练脚本 (GRPO)
奖励函数包含两个组件：
1. 正确性奖励 (Correctness Reward)
2. 格式/推理结构奖励 (Format Reward)
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset
import re
import sys
import os

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_number(text):
    """从文本中提取最终答案数值"""
    # 首先尝试匹配 "#### number" 格式
    match = re.search(r'####\s*([-\d.]+)', text)
    if match:
        return float(match.group(1))

    # 尝试提取文本中的最后一个数字
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    if numbers:
        return float(numbers[-1])

    return None

def compute_reward(questions, generated_answers, ground_truths, **kwargs):
    """
    计算奖励，包含两个组件：
    1. 正确性奖励 (Correctness)
    2. 格式/推理结构奖励 (Format/Reasoning Structure)
    """
    rewards = []

    for q, gen_ans, gt in zip(questions, generated_answers, ground_truths):
        gen_num = extract_number(gen_ans)
        gt_num = extract_number(gt)

        # === 组件 1: 正确性奖励 ===
        if gen_num is not None and gt_num is not None:
            # 允许小的数值误差
            if abs(gen_num - gt_num) < 0.01:
                correctness = 1.0
            elif abs(gen_num - gt_num) < 0.1:
                correctness = 0.5
            else:
                correctness = -0.5
        else:
            correctness = -1.0

        # === 组件 2: 格式/推理结构奖励 ===
        # 检查是否有推理关键词
        reasoning_keywords = ["first", "then", "because", "step", "next", "so", "therefore", "we", "need"]
        has_reasoning = any(keyword in gen_ans.lower() for keyword in reasoning_keywords)

        # 检查是否有 "####" 格式
        has_format = "####" in gen_ans

        if has_reasoning and has_format:
            format_reward = 0.5
        elif has_reasoning or has_format:
            format_reward = 0.2
        else:
            format_reward = -0.3

        # 检查答案长度（避免过短或过长）
        gen_len = len(gen_ans.strip())
        if 50 <= gen_len <= 500:
            length_reward = 0.1
        else:
            length_reward = -0.1

        # === 总奖励 ===
        total_reward = correctness + format_reward + length_reward
        rewards.append(total_reward)

    return rewards

def create_prompt_dataset(dataset):
    """创建带 prompt 的数据集"""
    prompts = []
    completions = []

    for example in dataset:
        prompt = f"Solve this math problem step by step. Show your work and put the final answer in the format '#### number'.\n\nQuestion: {example['instruction']}"
        prompts.append(prompt)
        completions.append(example['output'])

    return {"prompt": prompts, "completion": completions}

def main():
    print("=" * 60)
    print("GRPO RL Training for GSM8K")
    print("=" * 60)

    # 模型路径
    model_id = "./models/qwen2.5-3b-sft"

    print(f"\n1. Loading model from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 加载数据
    print(f"\n2. Loading training data...")
    train_data = load_dataset("json", data_files="./data/filtered_gsm8k_train.jsonl", split="train")
    print(f"   Training samples: {len(train_data)}")

    # 使用较小的子集进行训练（加速训练）
    max_samples = 1000  # 可以根据需要调整
    if len(train_data) > max_samples:
        train_data = train_data.select(range(max_samples))
        print(f"   Using subset: {max_samples} samples")

    # GRPO 配置
    training_args = GRPOConfig(
        output_dir="./models/qwen2.5-3b-rl",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1.0e-6,
        logging_steps=10,
        save_steps=100,
        max_length=512,
        max_prompt_length=256,
        max_completion_length=256,
        num_generations=4,  # GRPO 生成的候选数量
        temperature=0.7,
        generation_kwargs={"max_new_tokens": 256, "do_sample": True},
    )

    print(f"\n3. Training configuration:")
    print(f"   Output dir: {training_args.output_dir}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Max length: {training_args.max_length}")

    # 构造训练数据
    print(f"\n4. Preparing training data...")

    class RewardWrapper:
        """包装 reward 函数以符合 GRPOTrainer 的要求"""
        def __init__(self, reward_func):
            self.reward_func = reward_func

        def __call__(self, prompts, completions, **kwargs):
            # 从 prompts 中提取问题
            questions = [p.split("Question: ")[-1] for p in prompts]
            return self.reward_func(questions, completions, completions, **kwargs)

    # 训练器
    print(f"\n5. Initializing trainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=RewardWrapper(compute_reward),
        args=training_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        prompt_key="instruction",
        completion_key="output",
    )

    # 开始训练
    print(f"\n6. Starting training...")
    print("=" * 60)
    trainer.train()

    # 保存模型
    print(f"\n7. Saving model...")
    trainer.save_model()
    print(f"   Model saved to: ./models/qwen2.5-3b-rl")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
