import json
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--end', type=int, default=500)
    parser.add_argument('--dataset', type=str, default='math-500')
    parser.add_argument('--eval_path', type=str, default='/data/semantic/speculative/spec_result_math-500_seed_456')
    args = parser.parse_args()
    length_target = 0
    length_spe = 0
    count = 0
    try_correct_num = 0
    correct_spe_number = 0
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
    for idx, number in enumerate(tqdm(range(args.start, args.end))):

        if args.dataset == 'math-500':
            dirname = f'spec_{args.dataset}_{number}'
        elif args.dataset == 'aime':
            dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(args.eval_path, dirname)
        json_path = os.path.join(dir_path, "spec_generation.json")
        if not os.path.exists(json_path):
            print(f"[Warning] {json_path} does not exist, skipping...")
            continue


        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            try_correct_num += generations[0]['try_correct_num']
            correct_spe_number += generations[0]['correct_spe_number']
            detail = generations[0]['detail']
            for d in detail:
                if  'why_is_not_good' in d:
                    length_target += tokenizer(d['target_model'], return_tensors="pt")['input_ids'].shape[1]
                    length_spe += tokenizer(d['why_is_not_good'], return_tensors="pt")['input_ids'].shape[1]
                    count += 1
    print('length_target',length_target/count)
    print('length_spe',length_spe/count)
    print('try_correct_num_rate',try_correct_num/(correct_spe_number+try_correct_num))
    print('correct_spe_number_rate', correct_spe_number / (correct_spe_number + try_correct_num))




