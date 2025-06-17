
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
    parser.add_argument('--end', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='math-500')
    parser.add_argument('--eval_path', type=str, default='/data/ximing/semantic')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    number_correct = 0
    total_number = args.end - args.start

    for idx, number in enumerate(tqdm(range(args.start, args.end))):

        if args.dataset == 'math-500':
            dirname = f'baseline_{args.dataset}_{number}'
        elif args.dataset == 'aime':
            dirname = f'baseline_{args.dataset}_{number}'
        dir_path = os.path.join(args.eval_path, dirname)
        json_path = os.path.join(dir_path, "generation.json")
        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            predict = generations[0]['real_answer']
            standard = generations[0]['standard_answer']
        result = check_math_correctness(standard,predict)
        if result:
            number_correct += 1
        else:
            print(f'Error in {dirname}')
    print(f'Accuracy: {number_correct / total_number} in {args.dataset}')
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')

