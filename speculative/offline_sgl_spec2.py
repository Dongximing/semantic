import transformers
import random
import argparse
import json
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sglang as sgl
from utils import *
import math

BEGIN_TOKEN_NUM = 500
SPECULATIVE_OUTPUT_LENGTH = 500
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
TARGET_model = 0
SPEC_model = 1
TARGET_probe = 2
SPEC_probe = 3

# 优化1: 预定义常量，避免循环中重复创建字典
SAMPLING_PARAMS_BASE = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 500,
    "stop_token_ids": [4710, 382, 1447, 271, 692, 1939, 2533, 3593],
    "no_stop_trim": True
}

SAMPLING_PARAMS_END = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 2000,
}

SAMPLING_PARAMS_CHECK = {
    "temperature": 0.1,
    "max_new_tokens": 1
}

def speculative_accept(qi, pi, threshold_min=0.7):
    ratio = qi / pi if pi > 0 else 0
    if ratio < threshold_min:
        return False
    threshold = min(1.0, ratio)
    r = random.uniform(0, 1)
    return r < threshold

def extract_potential_ids(input_top_logprobs, input_token_logprobs, draft_len_output):
    """
    返回形如 [[top1_id, top2_id], ...] 的列表，长度为 draft_len_output。
    """
    last_top = input_top_logprobs[-draft_len_output:]
    last_tok = input_token_logprobs[-draft_len_output:]
    potential_ids = []

    def get_token_id_from_tuple(t):
        return t[1] if isinstance(t, (list, tuple)) and len(t) > 1 else None

    for top, tok in zip(last_top, last_tok):
        top1_id = top2_id = None

        if top and isinstance(top, (list, tuple)) and len(top) > 0:
            head1 = top[0]
            if isinstance(head1, (list, tuple)) and len(head1) > 1:
                top1_id = head1[1]

            if len(top) > 1:
                head2 = top[1]
                if isinstance(head2, (list, tuple)) and len(head2) > 1:
                    top2_id = head2[1]

        if top1_id is None:
            top1_id = get_token_id_from_tuple(tok)

        if top2_id is None:
            top2_id = top1_id if top1_id is not None else get_token_id_from_tuple(tok)

        if top1_id is None:
            top1_id = 0
        if top2_id is None:
            top2_id = top1_id

        potential_ids.append([top1_id, top2_id])

    return potential_ids


class SemanticEntropyProbTarget(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 256)  
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        h = F.relu(self.fc3(h))
        out = torch.sigmoid(self.fc4(h))
        return out.squeeze(-1)

class SemanticEntropyProbSpec(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 512)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(512, 256)   
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.dropout1(h)
        h = F.relu(self.fc2(h))
        h = self.dropout2(h)
        h = F.relu(self.fc3(h))
        out = torch.sigmoid(self.fc4(h))
        return out.squeeze(-1)


# 优化2: 提取隐藏状态转换为单独函数，减少重复代码
def convert_hidden_states_to_tensor(hidden_states):
    """批量转换隐藏状态为tensor并拼接"""
    for i in range(len(hidden_states)):
        if not isinstance(hidden_states[i], torch.Tensor):
            hidden_states[i] = torch.tensor(hidden_states[i], dtype=torch.bfloat16)
    
    return torch.cat([
        i.unsqueeze(0) if len(i.shape) == 1 else i
        for i in hidden_states
    ])


def speculative_decoding(llm_big, llm_small, target_tokenizer, speculative_tokenizer, 
                         problem, max_new_tokens, model_target_probe, model_spec_probe, probe_device):
    time_detial = []
    messages = [{"role": "user", "content": problem}]
    
    target_text = target_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    speculative_text = speculative_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    start_target_model_inputs = target_tokenizer(target_text, return_tensors="pt")
    original_target_prompt_len = start_target_model_inputs["input_ids"].shape[1]

    start_speculative_text_inputs = speculative_tokenizer(speculative_text, return_tensors="pt")
    original_speculative_text_len = start_speculative_text_inputs["input_ids"].shape[1]

    correct_tokens, try_correct_num, correct_spe_number = [], 0, 0
    detail = []
    begin = True
    use_target = True

    def checking_is_finish(generated_ids, max_new_tokens, use_target):
        return len(target_tokenizer.encode(generated_ids)) - original_target_prompt_len < max_new_tokens

    speculative_real_output_text = ''
    prob_target = 0
    prob_spec = 0
    target_real_output = ''
    generated_text = target_text
    break_target = False
    start_time = time.time()
    
    while checking_is_finish(generated_text, max_new_tokens, use_target):
        #print('-------------------------------------------------------\n')
        if break_target:
            break
        
        if begin:
            use_target = True
        
        if not begin:
            if use_target:
                detail.append({
                    'target_model': target_real_output, 
                    'why_is_not_gd': speculative_real_output_text,
                    "score_target": round(prob_target, 2), 
                    "score_spec": round(prob_spec, 2)
                })
                small_input = speculative_text + target_tokenizer.decode(
                    target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                    original_target_prompt_len:].tolist()
                )
            else:
                small_input = generated_text
            # #print('small_input',small_input)
     
            speculative_outputs_start = time.time()
            speculative_outputs = llm_small.generate(
                [small_input], 
                sampling_params=SAMPLING_PARAMS_BASE, 
                return_hidden_states=True,
                return_logprob=True
            )
            
            # 优化3: 使用列表推导式替代enumerate
            prob_small_result = [
                {"id": unvalid_id, "prob": math.exp(lp)}
                for lp, unvalid_id, _ in speculative_outputs[0]['meta_info']['output_token_logprobs']
            ]
     
            speculative_outputs_time = time.time() - speculative_outputs_start
            time_detial.append({'speculative_outputs_time': speculative_outputs_time})
            
            speculative_output = speculative_outputs[0]
            speculative_real_output_text = speculative_output['text']
            #print('speculative_real_output_text',speculative_real_output_text)
            
            if '</think>' in speculative_real_output_text:
                speculative_outputs_ending_start = time.time()
                speculative_outputs = llm_big.generate(
                    [generated_text],
                    sampling_params=SAMPLING_PARAMS_END,
                    return_hidden_states=False,     
                )
                speculative_outputs_ending = time.time() - speculative_outputs_ending_start
                time_detial.append({'speculative_outputs_ending': speculative_outputs_ending})
                detail.append({'spe_model': speculative_outputs[0]['text']})
                generated_text = generated_text + speculative_outputs[0]['text']
                break

            extract_time_small = time.time()
            # 优化4: 使用提取的函数转换tensor
            pooling_hidden_information = convert_hidden_states_to_tensor(
                speculative_output["meta_info"]["hidden_states"]
            )
            Completion_tokens = speculative_output['meta_info']['completion_tokens']
            pooling_hidden_information = pooling_hidden_information[-1, :]
            # print(pooling_hidden_information.shape)
            # pooling_hidden_information = pooling_hidden_information.mean(dim=0, keepdim=True)
            
            exetract_small = time.time() - extract_time_small
            time_detial.append({'exetract_small': exetract_small})
            
            if len(speculative_real_output_text) == 0:
                break

            checking_target_text = generated_text + speculative_real_output_text
            valid_checking_target_text_len = len(target_tokenizer.encode(generated_text))
            #print('checking_target_text',checking_target_text)
            
            checking_start = time.time()
            checking_outputs = llm_big.generate(
                [checking_target_text],
                sampling_params=SAMPLING_PARAMS_CHECK,
                return_hidden_states=True,
                return_logprob=True,
                logprob_start_len=valid_checking_target_text_len - 2,
                top_logprobs_num=2
            )

            potential_ids = extract_potential_ids(
                checking_outputs[0]['meta_info']['input_top_logprobs'],
                checking_outputs[0]['meta_info']['input_token_logprobs'], 
                Completion_tokens
            )
            potential_sets = [set(ids) for ids in potential_ids]
            result = all(small["id"] in potential_sets[i] for i, small in enumerate(prob_small_result))
            
            if result:
                #print('all accpet!\U0001F600')
                detail.append({'spe_model': speculative_real_output_text})
                correct_spe_number += 1
                use_target = False
                generated_text = small_input + speculative_output['text'] + checking_outputs[0]['text']
                
                if '</think>' in speculative_output['text']:
                    speculative_outputs_ending_start = time.time()
                    speculative_outputs = llm_big.generate(
                        [generated_text],
                        sampling_params=SAMPLING_PARAMS_END,
                        return_hidden_states=False,     
                    )
                    speculative_outputs_ending = time.time() - speculative_outputs_ending_start
                    time_detial.append({'speculative_outputs_ending': speculative_outputs_ending})
                    detail.append({'spe_model': speculative_outputs[0]['text']})
                    generated_text = generated_text + speculative_outputs[0]['text']
                    break_target = True
                    break
                
                continue
            
            checking_time = time.time() - checking_start
            time_detial.append({
                'checking_time': checking_time,
                'Completion_tokens': Completion_tokens,
                'average_token_ckecing': checking_time / Completion_tokens
            })
            
            checking_output = checking_outputs[0]
            extract_time_big = time.time()
            
            # 使用提取的函数转换tensor（保持原始完整的hidden_states）
            for i in range(len(checking_output["meta_info"]["hidden_states"])):
                checking_output["meta_info"]["hidden_states"][i] = torch.tensor(
                    checking_output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                )
            hidden_states = torch.cat([
                i.unsqueeze(0) if len(i.shape) == 1 else i
                for i in checking_output["meta_info"]["hidden_states"]
            ])
            
            computing_prob_start = time.time()
            target_pooling_hidden_information = hidden_states[-1, :]
            # print(target_pooling_hidden_information.shape )
            exetract_big = time.time() - extract_time_big
            time_detial.append({'exetract_big': exetract_big})
            
            if target_pooling_hidden_information.shape[0] == 0:
                break
            
            
            #target_pooling_hidden_information = target_pooling_hidden_information.mean(dim=0, keepdim=True)

            with torch.no_grad():
                prob_target = model_target_probe(target_pooling_hidden_information.float().to(probe_device))
                prob_spec = model_spec_probe(pooling_hidden_information.float().to(probe_device))

            prob_target = prob_target.item()
            prob_spec = prob_spec.item()
            
            computing_prob_time = time.time() - computing_prob_start
            time_detial.append({'computing_prob_time': computing_prob_time})
            
            if speculative_accept(prob_target, prob_spec):
                #print('\U0001F600\U0001F600 ----')
                detail.append({'spe_model': speculative_real_output_text})
                correct_spe_number += 1
                use_target = False
                generated_text = small_input + speculative_output['text'] + checking_outputs[0]['text']
                
                if '</think>' in speculative_output['text']:
                    speculative_outputs_ending_start = time.time()
                    speculative_outputs = llm_big.generate(
                        [generated_text],
                        sampling_params=SAMPLING_PARAMS_END,
                        return_hidden_states=False,     
                    )
                    speculative_outputs_ending = time.time() - speculative_outputs_ending_start
                    time_detial.append({'speculative_outputs_ending': speculative_outputs_ending})
                    detail.append({'spe_model': speculative_outputs[0]['text']})
                    generated_text = generated_text + speculative_outputs[0]['text']
                    break
            else:
                #print('❌ ❌ ❌ ')
                generated_text = target_text + speculative_tokenizer.decode(
    speculative_tokenizer(small_input, return_tensors="pt")['input_ids'][0,original_speculative_text_len :].tolist()
)
                use_target = True

        if use_target:
            begin = False
            try_correct_num = try_correct_num + 1
            #print(
            #     'generated_text',generated_text
            # )
            
            target_outputs_start = time.time()
            target_outputs = llm_big.generate(
                [generated_text],
                sampling_params=SAMPLING_PARAMS_BASE,
                return_hidden_states=False,
            )
            target_outputs_time = time.time() - target_outputs_start
            time_detial.append({'target_outputs_time': target_outputs_time})
            
            target_outputs = target_outputs
            target_real_output = target_outputs[0]['text']
            #print('target_real_output',target_real_output)
            
            if '</think>' in target_real_output:
                small_input = speculative_text + target_tokenizer.decode(
                    target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                    original_target_prompt_len:].tolist()
                )
                detail.append({'target_model': target_real_output})
         
                traget_ending_start = time.time()
                speculative_outputs = llm_big.generate(
                    [small_input + target_real_output],
                    sampling_params=SAMPLING_PARAMS_END,
                    return_hidden_states=False,
                )
                traget_ending_time = time.time() - traget_ending_start
                time_detial.append({'traget_ending_time': traget_ending_time})
                detail.append({'spe_model': speculative_outputs[0]['text']})
                generated_text = small_input + target_real_output + speculative_outputs[0]['text']
                break
            
            generated_text = generated_text + target_real_output

            if target_tokenizer.eos_token_id in target_tokenizer.encode(target_real_output):
                #print('target_tokenizer.eos_token_id 281', target_tokenizer.eos_token_id)
                break

    end_time = time.time()
    length_of_output = speculative_tokenizer.encode(generated_text[original_speculative_text_len:])

    # 优化6: 简化时间统计
    total_time = sum(
        t.get(key, 0) for t in time_detial 
        for key in ['target_outputs_time', 'checking_time', 'computing_prob_time',
                   'speculative_outputs_time', 'speculative_outputs_ending', 'traget_ending_time']
    )

    return (generated_text, try_correct_num, correct_spe_number, detail, 
            len(length_of_output), end_time - start_time, total_time, time_detial)


def process_file_to_json(
    dir_path, llm_big, llm_small, target_tokenizer, speculative_tokenizer,
    problem, answer, max_new_tokens, model_target_probe, model_spec_probe, idx, probe_device
):
    all_generations = []
    failed_list = []

    result = speculative_decoding(
        llm_big, llm_small, target_tokenizer, speculative_tokenizer,
        problem, max_new_tokens, model_target_probe, model_spec_probe, probe_device
    )

    generated_text, try_correct_num, correct_spe_number, detail, length_of_output, times, total_time, time_detial = result

    all_generations.append({
        "input_text": problem,
        "real_answer": generated_text,
        "try_correct_num": try_correct_num,
        "standard_answer": answer,
        "execution_time": f"{times:.2f}s",
        "correct_spe_number": correct_spe_number,
        "total_time": f"{total_time:.2f}s",
        "time_detail": time_detial,
        "detail": detail,
        "length_of_output": length_of_output,
        "index": idx
    })

    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, "spec_generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)

    return failed_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", default='gpqa')
    parser.add_argument("--target_model", type=str, help="target_model", default="/home/original_models/QwQ-32B")
    parser.add_argument("--speculative_model", type=str, help="speculative_model", default="/home/original_models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str, help="data_dir", default='../speculative/qwq32-r1lasthiden')
    parser.add_argument("--start_dataset", type=int, help="the beginning of the dataset", default=0)
    parser.add_argument("--end_dataset", type=int, help="the end of the dataset", default=198)
    parser.add_argument("--target_probe", type=str, help="speculative_probe", default="/home/ximing/semantic/training_limo_s1/s1_valid_h100_32b_gpqa_last_hidden_state_best_probe_mse")
    parser.add_argument("--speculative_probe", type=str, help="target_probe", default="/home/ximing/semantic/training_limo_s1/s1_valid_h100_r1.5b_gpqa_last_hidden_state_best_probe_mse")
    parser.add_argument("--target_temperature", type=float, help="target_temperature", default=0.1)
    parser.add_argument("--speculative_temperature", type=float, help="speculative_temperature", default=0.6)
    parser.add_argument("--max_new_tokens", type=int, help="max_new_tokens", default=14000)
    parser.add_argument("--top_p", type=float, help="top_p", default=0.9)
    parser.add_argument("--top_k", type=int, help="top_k", default=50)
    parser.add_argument("--seed", type=int, help="seed", default=6540)
    args = parser.parse_args()
    
    seed_everything(args.seed)

    probe_device = 'cuda:0'
    
    model_target_probe = SemanticEntropyProbTarget(5120, 2048)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.to(probe_device)
    model_target_probe.eval()

    model_spec_probe = SemanticEntropyProbSpec(1536, 1024)
    model_spec_probe.load_state_dict(torch.load(f'{args.speculative_probe}.pt'))
    model_spec_probe = model_spec_probe.to(probe_device)
    model_spec_probe.eval()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    llm_small = sgl.Engine(
        model_path=args.speculative_model,
        enable_return_hidden_states=True,
        mem_fraction_static=0.3,
        tp_size=1
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    llm_big = sgl.Engine(
        model_path=args.target_model,
        enable_return_hidden_states=True,
        mem_fraction_static=0.9,
        tp_size=1
    )

    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.target_model,
        trust_remote_code=True
    )

    speculative_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.speculative_model,
        trust_remote_code=True
    )

    if speculative_tokenizer.pad_token_id is None:
        speculative_tokenizer.pad_token_id = speculative_tokenizer.eos_token_id
    
    if args.dataset == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']
    elif args.dataset == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif args.dataset == "amc23":
        ds = load_dataset("zwhe99/amc23", split="test")
    elif args.dataset == "gpqa":
        loaded =load_dataset("/home/ximing/semantic/baseline/gpqa", "gpqa_diamond")
        subset = loaded["train"].select(range(args.start_dataset, args.end_dataset))
        train_data = subset.to_pandas()
        ds = [row.to_dict() for _, row in train_data.iterrows()]
        for problem in ds:
            multiple_choice_string, correct_answer_letter = (
                get_GPQA_multiple_choice_answers(problem)
            )

            problem["problem"] = (
                "Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response. "
                + problem["Question"]
                + "\n"
                + multiple_choice_string
            )
            problem["answer"] = correct_answer_letter
    else:
        raise ValueError(f"Unknown task: {args.dataset}")

   
    
    if args.dataset == "amc23":
        ds = ds.select(range(args.start_dataset, args.end_dataset))
        problems_and_answers = [{"problem": item["question"], "answer": item["answer"]} for item in ds]
    elif args.dataset == "gpqa": 
        problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]
    else:
        ds = ds.select(range(args.start_dataset, args.end_dataset))
        problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]

    # 注意: 原代码中 wrong_list 未定义，这里需要您提供
    # 暂时使用 range 作为示例
    if args.seed == 3210:

        wrong_list: [1, 3, 5, 7, 8, 9, 17, 18, 20, 22, 23, 24, 27, 28, 29, 30, 31, 33, 35, 36, 37, 39, 41, 42, 44, 45, 46, 47, 48, 51, 52, 53, 54, 57, 59, 60, 61, 62, 63, 64, 66, 68, 69, 70, 71, 73, 74, 76, 79, 85, 88, 89, 90, 91, 93, 94, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
    elif args.seed ==6540:
        wrong_list = [0, 1, 3, 5, 7, 8, 9, 10, 12, 13, 15, 17, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 35, 36, 39, 41, 42, 45, 46, 47, 48, 50, 52, 53, 54, 55, 56, 58, 60, 62, 63, 68, 73, 74, 75, 76, 78, 79, 81, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99, 101, 102, 103, 105, 106, 108, 109, 113, 116, 117, 118, 121, 125, 127, 128, 130, 131, 132, 134, 136, 138, 139, 140, 142, 143, 144, 145, 146, 147, 149, 152, 153, 155, 157, 158, 159, 160, 162, 164, 165, 166, 167, 170, 174, 175, 176, 178, 179, 180, 182, 183, 185, 186, 187, 188, 189, 190, 191, 193, 194, 196, 197]
    else:
        wrong_list =[0, 3, 6, 8, 9, 10, 11, 13, 15, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 36, 39, 42, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 60, 61, 63, 67, 68, 71, 73, 76, 78, 79, 80, 81, 82, 84, 85, 89, 90, 91, 92, 93, 94, 97, 98, 99, 101, 102, 105, 106, 108, 109, 113, 115, 116, 117, 118, 120, 121, 123, 125, 127, 128, 129, 130, 131, 133, 136, 137, 138, 139, 140, 142, 143, 144, 145, 147, 149, 152, 153, 155, 157, 159, 162, 163, 164, 165, 166, 167, 173, 174, 175, 178, 179, 180, 181, 182, 183, 185, 186, 187, 189, 190, 196, 197]
    for idx, number in enumerate(tqdm(wrong_list)):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.dataset}{args.data_dir}{args.seed}", dirname)
        problem = problems_and_answers[number]['problem']
        answer = problems_and_answers[number]['answer']
        failed = process_file_to_json(
            dir_path, llm_big, llm_small, target_tokenizer, speculative_tokenizer,
            problem, answer, args.max_new_tokens, model_target_probe, model_spec_probe, number, probe_device
        )