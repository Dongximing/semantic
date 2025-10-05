from paser import *
import os
import json
from tqdm import tqdm
import argparse
def check_math_correctness(ref, generation):
    if not find_box(generation): return False
    answer = strip_answer_string(ref)
    pred = extract_answer(generation)
    pred = strip_answer_string(pred)
    return math_equal(pred, answer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=100)
    parser.add_argument('--end', type=int, default=500)
    parser.add_argument('--dataset', type=str, default='math')
    parser.add_argument('--eval_path', type=str, default='/data/semantic/baseline/r1_1.5B_baseline_math_500_seed42')
    parser.add_argument('--seed', type=int, default=3210)
    args = parser.parse_args()

    time = 0
    number_correct = 0
    number_of_tokens = 0
    whole_time = 0
    whole_length = 0
    whole_number_of_tokens = 0 
    total_number = args.end - args.start
    wrong_list = []
    no = 0
    for idx, number in enumerate(tqdm(range(args.start, args.end))):

        if args.dataset == 'math':
            dirname = f'spec_{args.dataset}_{number}'
        elif args.dataset == 'aime':
            dirname = f'spec_{args.dataset}_{number}'
        elif args.dataset == 'amc23':
            dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(args.eval_path, dirname)
        json_path = os.path.join(dir_path, "result.json")
        if not os.path.exists(json_path):
            print(f"[Warning] {json_path} does not exist, skipping...")
            no+=1
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            predict = generations['reasoning']
            whole_time+=generations['execution_time']
            whole_number_of_tokens += generations['number_tokens']
            standard = generations['answer']
            whole_length += generations['number_tokens']
        result = check_math_correctness(standard,predict)
        if result:
            number_of_tokens += generations['number_tokens']
            time += generations['execution_time']
            number_correct += 1
        else:
            wrong_list.append(number)
    total_number = total_number - no
    print(f'Accuracy: {number_correct / total_number} in {args.dataset}')
    
    print(f'whole length: {whole_number_of_tokens / total_number} in {args.dataset}')
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')
    print(f'average whole execution time: {whole_time/total_number}')

    print(f'average whole execution time per token: {whole_time /whole_length }')
    print(f'wrong_list: {wrong_list}')
    print('no', no)
    print('\n\n\n\n')


