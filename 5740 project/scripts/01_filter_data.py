# scripts/01_filter_data.py
"""
GSM8K 数据筛选脚本
筛选规则：
1. 答案必须包含 "#### " 格式的最终答案
2. 问题长度在 50-500 字符
3. 答案不能为空
"""
from datasets import load_dataset
import json

def filter_gsm8k():
    """筛选 GSM8K 数据集"""
    # 加载 GSM8K
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")

    def is_valid(example):
        """验证样本是否有效"""
        answer = example["answer"]
        # 1. 答案必须包含 "#### " 格式的最终答案
        if "#### " not in answer:
            return False
        # 2. 问题长度在 50-500 字符
        if len(example["question"]) < 50 or len(example["question"]) > 500:
            return False
        # 3. 答案不能为空
        if not answer.strip():
            return False
        return True

    # 筛选数据
    print("Filtering training data...")
    train_filtered = dataset["train"].filter(is_valid)

    print("Filtering test data...")
    test_filtered = dataset["test"].filter(is_valid)

    # 转换为 llama-factory 格式
    def to_llama_factory_format(example):
        return {
            "instruction": example["question"],
            "input": "",
            "output": example["answer"]
        }

    train_data = train_filtered.map(to_llama_factory_format)
    test_data = test_filtered.map(to_llama_factory_format)

    # 保存为 JSONL
    train_data.to_json("../data/filtered_gsm8k_train.jsonl", orient="records", lines=True)
    test_data.to_json("../data/filtered_gsm8k_test.jsonl", orient="records", lines=True)

    print(f"\n=== Data Summary ===")
    print(f"Original train samples: {len(dataset['train'])}")
    print(f"Filtered train samples: {len(train_data)}")
    print(f"Original test samples: {len(dataset['test'])}")
    print(f"Filtered test samples: {len(test_data)}")
    print(f"\nData saved to:")
    print(f"  - ../data/filtered_gsm8k_train.jsonl")
    print(f"  - ../data/filtered_gsm8k_test.jsonl")

if __name__ == "__main__":
    filter_gsm8k()
