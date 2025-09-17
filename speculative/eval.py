from collections import defaultdict
from paser import *
import os
import json
from tqdm import tqdm
import transformers
import argparse

def check_math_correctness(ref, generation):
    if not find_box(generation): return False
    answer = strip_answer_string(ref)
    pred = extract_answer(generation)
    pred = strip_answer_string(pred)
    return math_equal(pred, answer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='amc23')
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

    # 新增: 用于统计 time_detail 的 key -> sum
    time_detail_total = defaultdict(float)

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
            time_detail = generations[0]['time_detail']
            predict = generations[0]['real_answer']
            standard = generations[0]['standard_answer']
            length = generations[0].get('length_of_output')
            whole_length += length
            details = generations[0]['detail']
            whole_time += float(generations[0].get('total_time').rstrip('s'))

            # 新增：统计当前 sample 的 time_detail
            for td in time_detail:
                for key, value in td.items():
                    time_detail_total[key] += value

            result = check_math_correctness(standard, predict)
            spe_step += generations[0].get('correct_spe_number')
            target_step += generations[0].get('try_correct_num')

            if result:
                time += float(generations[0].get('total_time').rstrip('s'))
                number_of_tokens += length
                number_correct += 1
                for detail in details:
                    if 'spe_model' in detail:
                        small_tokens += speculative_tokenizer(detail['spe_model'], return_tensors="pt")["input_ids"].shape[1]
                    else:
                        big_tokens += speculative_tokenizer(detail['target_model'], return_tensors="pt")["input_ids"].shape[1]
            else:
                wrong_list.append(number)

    print(f"total time: {time}")
    print(f'Accuracy: {number_correct / (args.end - args.start)} in {args.dataset}')
    print("correct Number of tokens: ", number_of_tokens / number_correct)
    print("average spe step: ", spe_step / (spe_step + target_step))
    print("average target step: ", target_step / (spe_step + target_step))
    print("correct average execution time: ", time / number_correct)
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')
    print(f"wrong_list: {wrong_list}")
    print(f"large_tokens rate : ", 1 - small_tokens / (small_tokens + big_tokens))
    print(f"average time : {whole_time / (args.end - args.start)}")
    print(f'average speed: {number_of_tokens / time}')
    print(f'average whole execution time: {time / number_of_tokens}')
    print(f'average whole execution time....: {whole_time / whole_length}')
    print(f'average whole length: {whole_length / (args.end - args.start)}')

 
    print("\n====== Time Detail Total ======")
    total_time_sum = 0
    for key, total in time_detail_total.items():
        print(f"{key} avg per sample: {total / (args.end - args.start):.6f} s")
        total_time_sum += total
    print(f"Sum of all time_detail: {total_time_sum/(args.end - args.start):.6f} s")
