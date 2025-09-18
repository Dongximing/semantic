from collections import defaultdict
from paser import *  # 假设你自定义了 math_equal、find_box 等函数
import os
import json
from tqdm import tqdm
import transformers
import argparse

def check_math_correctness(ref, generation):
    if not find_box(generation):
        return False
    answer = strip_answer_string(ref)
    pred = extract_answer(generation)
    pred = strip_answer_string(pred)
    return math_equal(pred, answer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--end', type=int, default=500)
    parser.add_argument('--dataset', type=str, default='math-500')
    parser.add_argument('--eval_path', type=str, default='/data/semantic/speculative/spec_result_math-500_seed_456')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    number_correct = 0
    number_of_tokens = 0
    total_number = args.end - args.start
    wrong_list = []
    time = 0
    spe_step = 0
    target_step = 0
    small_tokens = 0
    big_tokens = 0
    whole_time = 0
    whole_length = 0

    # 用于统计每个 step 的时间总和和出现次数
    time_detail_total = defaultdict(float)
    time_detail_count = defaultdict(int)

    speculative_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        trust_remote_code=True
    )

    for idx, number in enumerate(tqdm(range(args.start, args.end))):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(args.eval_path, dirname)
        json_path = os.path.join(dir_path, "spec_generation.json")
        if not os.path.exists(json_path):
            wrong_list.append(number)
            print(f"[Warning] {json_path} does not exist, skipping...")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            gen = generations[0]

            # 基本字段提取
            time_detail = gen['time_detail']
            predict = gen['real_answer']
            standard = gen['standard_answer']
            length = gen.get('length_of_output', 0)
            whole_length += length
            details = gen['detail']
            total_time_str = gen.get('total_time', '0s')
            total_time = float(total_time_str.rstrip('s'))
            whole_time += total_time

            # 累加每个 step 的时间总和与次数
            for td in time_detail:
                for key, value in td.items():
                    time_detail_total[key] += value
                    time_detail_count[key] += 1

            # 评估 correctness
            result = check_math_correctness(standard, predict)
            spe_step += gen.get('correct_spe_number', 0)
            target_step += gen.get('try_correct_num', 0)

            if result:
                time += total_time
                number_of_tokens += length
                number_correct += 1

                for detail in details:
                    if 'spe_model' in detail:
                        small_tokens += speculative_tokenizer(detail['spe_model'], return_tensors="pt")["input_ids"].shape[1]
                    else:
                        big_tokens += speculative_tokenizer(detail['target_model'], return_tensors="pt")["input_ids"].shape[1]
            else:
                wrong_list.append(number)

    # ========== 输出统计信息 ==========
    print("\n====== Overall Evaluation ======")
    print(f"Total time (correct samples): {time:.2f} s")
    print(f'Accuracy: {number_correct / total_number:.4f} in {args.dataset}')
    print("Correct avg tokens per sample: ", number_of_tokens / number_correct if number_correct else 0)
    print("Avg speculative step ratio: ", spe_step / (spe_step + target_step) if (spe_step + target_step) else 0)
    print("Avg target step ratio: ", target_step / (spe_step + target_step) if (spe_step + target_step) else 0)
    print("Correct avg execution time: ", time / number_correct if number_correct else 0)
    print(f'Number_correct: {number_correct}')
    print(f'Total samples: {total_number}')
    print(f"Wrong list: {wrong_list}")
    print(f"Large token ratio: {1 - small_tokens / (small_tokens + big_tokens) if (small_tokens + big_tokens) else 0:.4f}")
    print(f"Avg total time per sample: {whole_time / total_number:.4f} s")
    print(f"Avg speed (tokens/sec): {number_of_tokens / time if time else 0:.2f}")
    print(f"Avg time per token (correct only): {time / number_of_tokens if number_of_tokens else 0:.6f} s")
    print(f"Avg time per token (all samples): {whole_time / whole_length if whole_length else 0:.6f} s")
    print(f"Avg output length: {whole_length / total_number:.2f} tokens")

    # ========== 输出时间细节（time_detail）平均值 ==========
    print("\n====== Time Detail Per-Step Average (based on actual count) ======")
    total_time_sum = 0
    for key in sorted(time_detail_total):
        total = time_detail_total[key]
        count = time_detail_count[key]
        avg = total / count if count else 0
        print(f"{key:<30}: {avg:.6f} s (count: {count})")
        if key != 'Completion_tokens' and key != 'average_token_ckecing':
            total_time_sum += total

    print(f"\nSum of all time_detail (avg per sample): {total_time_sum / total_number:.6f} s")
