# scripts/04_evaluate.py
"""
评估脚本
对比 Base、SFT、RL 三个模型的性能
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import re
import json

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

def evaluate_model(model_path, test_data_path, model_name="Model"):
    """
    评估单个模型

    Returns:
        accuracy: 准确率 (百分比)
        results: 详细结果列表
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 60}")
    print(f"Model path: {model_path}")

    # 加载模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    # 加载测试数据
    print("Loading test data...")
    test_data = load_dataset("json", data_files=test_data_path, split="train")
    print(f"Test samples: {len(test_data)}")

    # 评估
    correct = 0
    total = len(test_data)
    results = []

    for idx, example in enumerate(tqdm(test_data, desc=f"{model_name}")):
        prompt = f"Solve this math problem step by step. Show your work and put the final answer in the format '#### number'.\n\nQuestion: {example['instruction']}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的答案（移除 prompt 部分）
        if prompt in generated:
            generated_answer = generated.split(prompt)[-1].strip()
        else:
            generated_answer = generated.strip()

        pred_num = extract_number(generated_answer)
        gt_num = extract_number(example['output'])

        is_correct = False
        if pred_num is not None and gt_num is not None:
            if abs(pred_num - gt_num) < 0.01:
                correct += 1
                is_correct = True

        results.append({
            "index": idx,
            "question": example['instruction'],
            "predicted": generated_answer,
            "ground_truth": example['output'],
            "pred_number": pred_num,
            "gt_number": gt_num,
            "correct": is_correct
        })

        # 打印前 3 个样本的详情
        if idx < 3:
            print(f"\n--- Sample {idx + 1} ---")
            print(f"Question: {example['instruction'][:100]}...")
            print(f"Predicted number: {pred_num}")
            print(f"Ground truth number: {gt_num}")
            print(f"Correct: {is_correct}")

    accuracy = correct / total * 100
    print(f"\n{'=' * 60}")
    print(f"Result for {model_name}:")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'=' * 60}")

    return accuracy, results

def main():
    print("\n" + "=" * 60)
    print("GSM8K Model Evaluation")
    print("=" * 60)

    test_data_path = "./data/filtered_gsm8k_test.jsonl"

    # 评估所有模型
    models = [
        ("Qwen/Qwen2.5-3B-Instruct", "Base Model"),
        ("./models/qwen2.5-3b-sft", "SFT Model"),
        ("./models/qwen2.5-3b-rl", "RL Model")
    ]

    results = {}
    detailed_results = {}

    for model_path, model_name in models:
        try:
            accuracy, detailed = evaluate_model(model_path, test_data_path, model_name)
            results[model_name] = accuracy
            detailed_results[model_name] = detailed
        except Exception as e:
            print(f"\nError evaluating {model_name}: {e}")
            results[model_name] = 0.0
            detailed_results[model_name] = []

    # 打印总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    base_acc = results.get("Base Model", 0)
    sft_acc = results.get("SFT Model", 0)
    rl_acc = results.get("RL Model", 0)

    print(f"{'Model':<20} {'Accuracy':<15} {'Improvement':<15}")
    print("-" * 50)
    print(f"{'Base':<20} {base_acc:<15.2f}% {'-':<15}")
    print(f"{'SFT':<20} {sft_acc:<15.2f}% {sft_acc - base_acc:+.2f}%")
    print(f"{'RL':<20} {rl_acc:<15.2f}% {rl_acc - base_acc:+.2f}%")

    # 保存详细结果
    output_file = "./data/evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": results,
            "detailed": detailed_results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    # 错误分析：找出 RL 模型预测错误的样本
    if "RL Model" in detailed_results and len(detailed_results["RL Model"]) > 0:
        wrong_predictions = [r for r in detailed_results["RL Model"] if not r["correct"]]
        print(f"\n=== Error Analysis (First 3 wrong predictions) ===")
        for wrong in wrong_predictions[:3]:
            print(f"\nQuestion: {wrong['question']}")
            print(f"Predicted: {wrong['predicted'][:200]}... (num: {wrong['pred_number']})")
            print(f"Ground truth: {wrong['gt_number']}")

if __name__ == "__main__":
    main()
