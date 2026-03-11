import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from tqdm import tqdm

from utils import get_GPQA_multiple_choice_answers, seed_everything

MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DEFAULT_STOP_TOKEN_IDS = [4710, 382, 1447, 271, 692, 1939, 2533, 3593]

# Shared sampling presets.
SAMPLING_PARAMS_BASE = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 500,
    "stop_token_ids": DEFAULT_STOP_TOKEN_IDS,
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

def speculative_accept(qi, pi, threshold_min=0.5):
    ratio = qi / pi if pi > 0 else 0
    threshold = min(1.0, ratio)
    r = random.uniform(0, 1)
    return r < threshold

def extract_potential_ids(input_top_logprobs, input_token_logprobs, draft_len_output):
    """
    Return a list like [[top1_id, top2_id], ...] for the final draft tokens.
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


class SemanticEntropyProbe(nn.Module):
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

SemanticEntropyProbTarget = SemanticEntropyProbe
SemanticEntropyProbSpec = SemanticEntropyProbe


def convert_hidden_states_to_tensor(hidden_states):
    """Convert hidden states to tensors and concatenate them."""
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
    messages = [{"role": "user", "content": problem }]
    
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
     
            speculative_outputs_start = time.time()
            speculative_outputs = llm_small.generate(
                [small_input], 
                sampling_params=SAMPLING_PARAMS_BASE, 
                return_hidden_states=True,
                return_logprob=True
            )
            
            prob_small_result = [
                {"id": unvalid_id, "prob": math.exp(lp)}
                for lp, unvalid_id, _ in speculative_outputs[0]['meta_info']['output_token_logprobs']
            ]
     
            speculative_outputs_time = time.time() - speculative_outputs_start
            time_detial.append({'speculative_outputs_time': speculative_outputs_time})
            
            speculative_output = speculative_outputs[0]
            speculative_real_output_text = speculative_output['text']
            
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
            pooling_hidden_information = convert_hidden_states_to_tensor(
                speculative_output["meta_info"]["hidden_states"]
            )
            Completion_tokens = speculative_output['meta_info']['completion_tokens']
            pooling_hidden_information = pooling_hidden_information[-Completion_tokens:, :]
            pooling_hidden_information = pooling_hidden_information.mean(dim=0, keepdim=True)
            
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
            
            for i in range(len(checking_output["meta_info"]["hidden_states"])):
                checking_output["meta_info"]["hidden_states"][i] = torch.tensor(
                    checking_output["meta_info"]["hidden_states"][i], dtype=torch.bfloat16
                )
            hidden_states = torch.cat([
                i.unsqueeze(0) if len(i.shape) == 1 else i
                for i in checking_output["meta_info"]["hidden_states"]
            ])
            
            computing_prob_start = time.time()
            target_pooling_hidden_information = hidden_states[-Completion_tokens-1:-1, :]
            exetract_big = time.time() - extract_time_big
            time_detial.append({'exetract_big': exetract_big})
            
            if target_pooling_hidden_information.shape[0] == 0:
                break
            
            target_pooling_hidden_information = target_pooling_hidden_information.mean(dim=0, keepdim=True)

            with torch.no_grad():
                prob_target = model_target_probe(target_pooling_hidden_information.float().to(probe_device))
                prob_spec = model_spec_probe(pooling_hidden_information.float().to(probe_device))

            prob_target = prob_target.item()
            prob_spec = prob_spec.item()
            
            computing_prob_time = time.time() - computing_prob_start
            time_detial.append({'computing_prob_time': computing_prob_time})
            
            if speculative_accept(prob_target, prob_spec):
        
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
                use_target = True

        if use_target:
            begin = False
            try_correct_num = try_correct_num + 1
            
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
                break

    end_time = time.time()
    length_of_output = speculative_tokenizer.encode(generated_text[original_speculative_text_len:])

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gpqa", choices=["math-500", "aime", "amc23", "gpqa"])
    parser.add_argument("--target_model", type=str, default="/home/original_models/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--speculative_model", type=str, default="/home/original_models/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str, default="../speculative/r132-r1cross_domain")
    parser.add_argument("--start_dataset", type=int, default=0)
    parser.add_argument("--end_dataset", type=int, default=198)
    parser.add_argument("--target_probe", type=str, default="/home/semantic/speculative/weight/s1_valid_h100_32r1b-200data_math_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--speculative_probe", type=str, default="/home/semantic/speculative/weight/s1_valid_h100_r1.5b_math_output_last_hidden_list_best_probe_mse")
    parser.add_argument("--max_new_tokens", type=int, default=14000)
    parser.add_argument("--seed", type=int, default=9870)
    parser.add_argument("--probe_device", type=str, default="cuda:4")
    parser.add_argument("--small_device", type=str, default="4")
    parser.add_argument("--big_device", type=str, default="5")
    parser.add_argument("--target_probe_dim", type=int, default=5120)
    parser.add_argument("--target_probe_hidden_dim", type=int, default=2048)
    parser.add_argument("--spec_probe_dim", type=int, default=1536)
    parser.add_argument("--spec_probe_hidden_dim", type=int, default=1024)
    return parser.parse_args()


def load_probe(model_cls, input_dim, hidden_dim, checkpoint_prefix, device):
    model = model_cls(input_dim, hidden_dim)
    model.load_state_dict(torch.load(f"{checkpoint_prefix}.pt"))
    model = model.to(device)
    model.eval()
    return model


def build_engines(args):
    import sglang as sgl

    print("loading the prediction done")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.small_device
    llm_small = sgl.Engine(
        model_path=args.speculative_model,
        enable_return_hidden_states=True,
        mem_fraction_static=0.3,
        tp_size=1,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.big_device
    llm_big = sgl.Engine(
        model_path=args.target_model,
        enable_return_hidden_states=True,
        mem_fraction_static=0.9,
        tp_size=1,
    )
    return llm_big, llm_small


def load_problems(args):
    if args.dataset == "math-500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        dataset = dataset.select(range(args.start_dataset, args.end_dataset))
        return [{"problem": item["problem"], "answer": item["answer"]} for item in dataset]

    if args.dataset == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        dataset = dataset.select(range(args.start_dataset, args.end_dataset))
        return [{"problem": item["problem"], "answer": item["answer"]} for item in dataset]

    if args.dataset == "amc23":
        dataset = load_dataset("zwhe99/amc23", split="test")
        dataset = dataset.select(range(args.start_dataset, args.end_dataset))
        return [{"problem": item["question"], "answer": item["answer"]} for item in dataset]

    loaded = load_dataset("/home/semantic/baseline/gpqa", "gpqa_diamond")
    subset = loaded["train"].select(range(args.start_dataset, args.end_dataset))
    rows = subset.to_pandas().to_dict("records")
    problems = []
    for row in rows:
        options, correct_answer = get_GPQA_multiple_choice_answers(row)
        problems.append(
            {
                "problem": (
                    "Return your final response within \\boxed{{}} and only include the letter choice "
                    "(A, B, C, or D) as your final response. "
                    f"{row['Question']}\n{options}"
                ),
                "answer": correct_answer,
            }
        )
    return problems


def main():
    args = parse_args()
    seed_everything(args.seed)
    probe_device = args.probe_device

    model_target_probe = load_probe(
        SemanticEntropyProbTarget,
        args.target_probe_dim,
        args.target_probe_hidden_dim,
        args.target_probe,
        probe_device,
    )
    model_spec_probe = load_probe(
        SemanticEntropyProbSpec,
        args.spec_probe_dim,
        args.spec_probe_hidden_dim,
        args.speculative_probe,
        probe_device,
    )
    llm_big, llm_small = build_engines(args)

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

    problems_and_answers = load_problems(args)

    for number, item in enumerate(tqdm(problems_and_answers), start=args.start_dataset):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.dataset}{args.data_dir}{args.seed}", dirname)
        problem = item["problem"]
        answer = item["answer"]
        process_file_to_json(
            dir_path, llm_big, llm_small, target_tokenizer, speculative_tokenizer,
            problem, answer, args.max_new_tokens, model_target_probe, model_spec_probe, number, probe_device
        )


if __name__ == "__main__":
    main()
