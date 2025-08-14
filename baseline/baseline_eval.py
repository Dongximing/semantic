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
    parser.add_argument('--dataset', type=str, default='math-500')
    parser.add_argument('--eval_path', type=str, default='/data/semantic/baseline/r1_1.5B_baseline_math_500_seed42')
    #//data/semantic/baseline
    #/home/cs/staff/shaowei/semantic/r1_1.5B_baseline_math_500_seed123
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    time = 0
    number_correct = 0
    number_of_tokens = 0
    whole_time = 0
    total_number = args.end - args.start

    for idx, number in enumerate(tqdm(range(args.start, args.end))):

        if args.dataset == 'math-500':
            dirname = f'seed_{args.seed}_baseline_{args.dataset}_{number}'
        elif args.dataset == 'aime':
            dirname = f'seed_{args.seed}_baseline_{args.dataset}_{number}'
        dir_path = os.path.join(args.eval_path, dirname)
        json_path = os.path.join(dir_path, "generation.json")
        if not os.path.exists(json_path):
            print(f"[Warning] {json_path} does not exist, skipping...")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            #print('generations',generations)
            predict = generations[0]['full_answer']
            #print(predict)
            whole_time += generations[0]['execution_time']

            standard = generations[0]['answer']
            # number_of_tokens += generations[0]['tokens_full_answer']
        result = check_math_correctness(standard,predict)
        if result:

            number_of_tokens += generations[0]['tokens_full_answer']
            time += generations[0]['execution_time']
            number_correct += 1
        else:
            print(f'Error in {dirname}')
    print(f'Accuracy: {number_correct / total_number} in {args.dataset}')
    print("Number of tokens: ", number_of_tokens/number_correct)
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')
    print(f'average execution time: {time/number_correct}')
    print(f'average speed: {number_of_tokens / time}')
    print(f'average whole execution time: {whole_time/args.end-args.start}')



