
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
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='math-500')
    parser.add_argument('--eval_path', type=str, default='/data/semantic/deepseek-32b_r1_awq_math')
    #/home/cs/staff/shaowei/semantic
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()


    number_correct = 0
    number_of_tokens = 0
    total_number = args.end - args.start
    wrong_list =  []
    time = 0
    spe_step = 0
    target_step = 0

    for idx, number in enumerate(tqdm(range(args.start, args.end))):

        if args.dataset == 'math-500':
            dirname = f'data-500-temp0_{number}'
        elif args.dataset == 'aime':
            dirname = f'data-500-temp0_{number}'
        dir_path = os.path.join(args.eval_path, dirname)
        json_path = os.path.join(dir_path, f"data-500_{number}.json")
        if not os.path.exists(json_path):
            print(f"[Warning] {json_path} does not exist, skipping...")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            #print('generations',generations)
            predict = generations['large_model_output']
            #print(predict)
            standard = generations['answer']

        result = check_math_correctness(standard,predict)
        if result:
            number_correct += 1
        else:
            wrong_list.append(number)
            # print(f'Error in {dirname}')
            # print("*" * 50)
            # print('standard',standard)
            # print("*"*50)
            # print('predict',predict)

    print(wrong_list)
