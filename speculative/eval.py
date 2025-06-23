
from paser import *
from utils import *
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
    parser.add_argument('--dataset', type=str, default='math-500')
    parser.add_argument('--eval_path', type=str, default='/data/semantic/speculative/spec_result_math-500_seed_456')
    #/home/cs/staff/shaowei/semantic
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    number_correct = 0
    number_of_tokens = 0
    total_number = args.end - args.start
    wrong_list =  []

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
            #print('generations',generations)
            predict = generations[0]['real_answer']
            #print(predict)
            standard = generations[0]['standard_answer']
            number_of_tokens += generations[0]['length_of_output']
        result = check_math_correctness(standard,predict)
        if result:
            number_correct += 1
        else:
            wrong_list.append(number)
            print(f'Error in {dirname}')
    print(f'Accuracy: {number_correct / total_number} in {args.dataset}')
    print("Number of tokens: ", number_of_tokens/total_number)
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')
    print(f"wrong_list: {wrong_list}")

