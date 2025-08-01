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
    parser.add_argument('--eval_path', type=str, default='/data/semantic/speculative/spec_result_math-500_seed_456')
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
            length = generations[0].get('length_of_output')

            time += float(generations[0].get('execution_time').rstrip('s'))
            result = check_math_correctness(standard, predict)

            spe_step += generations[0].get('correct_spe_number')
            target_step += generations[0].get('try_correct_num')
            if length is not None:
                number_of_tokens += length
                if result:
                    number_of_tokens += length
            else:
                if result:
                    number_of_tokens += generations[0].get('length_of_real_output')

        if result:
            number_correct += 1
        else:
            wrong_list.append(number)
            print(f'Error in {dirname}')
    print(f"total time: {time}")
    print(f'Accuracy: {number_correct / total_number} in {args.dataset}')
    print("Number of tokens: ", number_of_tokens/number_correct)
    print("average spe step: ", spe_step/(spe_step+target_step))
    print("average target step: ", target_step / (spe_step + target_step))
    print("average execution time: ", time/total_number)
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')
    print(f"wrong_list: {wrong_list}")

