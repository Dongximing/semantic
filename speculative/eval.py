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
    small_tokens =0
    big_tokens = 0
    whole_time = 0
    speculative_tokenizer = transformers.AutoTokenizer.from_pretrained(
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        trust_remote_code=True
    )

    #l = [102, 105, 106, 107, 111, 112, 113, 114, 115, 117, 121, 122, 124, 125, 127, 131, 132, 133, 134, 135, 137, 140, 141, 143, 144, 149, 157, 158, 159, 160, 161, 165, 167, 169, 171, 172, 174, 175, 176, 178, 179, 182, 185, 186, 187, 190, 191, 192, 195, 198, 199, 201, 206, 207, 208, 209, 212, 216, 218, 225, 226, 227, 229, 230, 231, 233, 237, 238, 241, 243, 244, 246, 247, 250, 251, 253, 254, 255, 257, 258, 259, 260, 262, 263, 265, 266, 268, 269, 270, 272, 273, 275, 277, 280, 283, 290, 293, 297, 300, 304, 305, 307, 310, 312, 313, 314, 318, 319, 321, 322, 325, 329, 330, 334, 336, 338, 339, 341, 343, 344, 345, 346, 347, 350, 354, 357, 358, 359, 362, 363, 364, 367, 368, 370, 373, 374, 376, 378, 383, 384, 385, 386, 388, 389, 393, 398, 404, 405, 406, 407, 410, 411, 413, 414, 417, 418, 424, 426, 427, 430, 433, 434, 435, 436, 437, 438, 440, 441, 442, 447, 449, 450, 451, 452, 453, 455, 458, 463, 464, 468, 469, 471, 472, 474, 476, 479, 487, 488, 492, 496]

    for idx, number in enumerate(tqdm(range(args.start, args.end))):

        if args.dataset == 'math-500':
            dirname = f'spec_{args.dataset}_{number}'
        elif args.dataset == 'aime':
            dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(args.eval_path, dirname)
        json_path = os.path.join(dir_path, "spec_generation.json")
        if not os.path.exists(json_path):
            wrong_list.append(number)
            print(f"[Warning] {json_path} does not exist, skipping...")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            generations = json.load(f)
            #print('generations',generations)
            predict = generations[0]['real_answer']
            #print(predict)
            standard = generations[0]['standard_answer']
            length = generations[0].get('length_of_output')
            details = generations[0]['detail']
            whole_time+=float(generations[0].get('execution_time').rstrip('s'))


            result = check_math_correctness(standard, predict)

            spe_step += generations[0].get('correct_spe_number')
            target_step += generations[0].get('try_correct_num')
            if length is not None:
                #number_of_tokens += length
                if result:
                    time += float(generations[0].get('execution_time').rstrip('s'))
                    number_of_tokens += length
                    details = generations[0]['detail']
                    for detail in details:
                        if 'spe_model' in detail:
                            small_tokens += speculative_tokenizer(detail['spe_model'], return_tensors="pt")["input_ids"].shape[1]
                        else:

                            big_tokens += speculative_tokenizer(detail['target_model'], return_tensors="pt")["input_ids"].shape[1]
            else:
                if result:
                    number_of_tokens += generations[0].get('length_of_real_output')

        if result:
            number_correct += 1
        else:
            wrong_list.append(number)
            print(f'Error in {dirname}')
    print(f"total time: {time}")
    print(f'Accuracy: {number_correct / (args.end-args.start)} in {args.dataset}')
    print("Number of tokens: ", number_of_tokens/number_correct)
    print("average spe step: ", spe_step/(spe_step+target_step))
    print("average target step: ", target_step / (spe_step + target_step))
    print("average execution time: ", time/number_correct)
    print(f'Number_correct: {number_correct}')
    print(f'Total: {total_number}')
    print(f"wrong_list: {wrong_list}")
    print(f"small_tokens rate : ", small_tokens/(small_tokens+big_tokens))

    print(f'average speed: {number_of_tokens / time}')
    print(f'average whole execution time: {whole_time/total_number}')


