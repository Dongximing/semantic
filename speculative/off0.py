import transformers
import random
import argparse
import json
import os
import time
from tqdm import tqdm
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import sglang as sgl
from utils import seed_everything
from functools import lru_cache

BEGIN_TOKEN_NUM = 500
SPECULATIVE_OUTPUT_LENGTH = 500
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# 优化1: 预编译停止token集合，避免重复创建
STOP_TOKEN_IDS = {4710, 382, 1447, 271, 692, 1939, 2533, 3593}

def speculative_accept(qi, pi, threshold_min=0.5):
    """优化: 简化条件判断"""
    if pi <= 0:
        return False
    ratio = qi / pi
    threshold = min(1.0, ratio)
    return random.random() < threshold

def extract_potential_ids(input_top_logprobs, input_token_logprobs, draft_len_output):
    """优化: 使用列表推导式和预分配内存"""
    last_top = input_top_logprobs[-draft_len_output:]
    last_tok = input_token_logprobs[-draft_len_output:]
    potential_ids = []
    
    for top, tok in zip(last_top, last_tok):
        top1_id = top2_id = 0
        
        if top and len(top) > 0:
            if isinstance(top[0], (list, tuple)) and len(top[0]) > 1:
                top1_id = top[0][1]
            if len(top) > 1 and isinstance(top[1], (list, tuple)) and len(top[1]) > 1:
                top2_id = top[1][1]
        
        if top1_id == 0 and isinstance(tok, (list, tuple)) and len(tok) > 1:
            top1_id = tok[1]
        if top2_id == 0:
            top2_id = top1_id if top1_id != 0 else (tok[1] if isinstance(tok, (list, tuple)) and len(tok) > 1 else 0)
        
        potential_ids.append([top1_id, top2_id])
    
    return potential_ids


class SemanticEntropyProbTarget(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),  # 优化: inplace操作节省内存
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SemanticEntropyProbSpec(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def process_hidden_states(hidden_states, completion_tokens, device, dtype=torch.bfloat16):
    """优化: 批量处理隐藏状态，减少循环开销"""
    # 一次性转换所有张量
    tensors = [
        (torch.tensor(h, dtype=dtype, device=device) if not isinstance(h, torch.Tensor) 
         else h.to(dtype=dtype, device=device))
        for h in hidden_states
    ]
    
    # 批量cat操作
    pooling = torch.cat([
        t.unsqueeze(0) if t.dim() == 1 else t 
        for t in tensors
    ])
    
    # 切片和池化
    pooling = pooling[-completion_tokens:].mean(dim=0, keepdim=True)
    return pooling


def speculative_decoding(llm_big, llm_small, target_tokenizer, speculative_tokenizer,
                         problem, max_new_tokens, model_target_probe, model_spec_probe):
    """优化后的推测解码函数"""
    
    time_detail = []
    messages = [{"role": "user", "content": problem + MATH_PROMPT}]
    
    # 应用chat模板
    target_text = target_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    speculative_text = speculative_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 优化: 预定义采样参数，避免重复创建字典
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 500,
        "stop_token_ids": list(STOP_TOKEN_IDS),
        "no_stop_trim": True
    }
    
    sampling_params_end = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 2000,
    }
    
    # 优化: 缓存tokenize结果
    start_target_inputs = target_tokenizer(target_text, return_tensors="pt")
    original_target_prompt_len = start_target_inputs["input_ids"].shape[1]
    
    start_spec_inputs = speculative_tokenizer(speculative_text, return_tensors="pt")
    original_spec_prompt_len = start_spec_inputs["input_ids"].shape[1]
    
    # 初始化变量
    try_correct_num = correct_spe_number = 0
    detail = []
    use_target = begin = True
    generated_text = target_text
    break_target = False
    
    # 优化: 预定义设备
    probe_device = next(model_target_probe.parameters()).device
    
    start_time = time.time()
    
    # 优化: 使用更高效的长度检查
    @lru_cache(maxsize=128)
    def get_encoded_length(text):
        return len(target_tokenizer.encode(text))
    
    while get_encoded_length(generated_text) - original_target_prompt_len < max_new_tokens:
        if break_target:
            break
        
        if begin:
            use_target = True
        
        # 推测模型生成
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
            
            # 推测生成
            spec_start = time.time()
            spec_outputs = llm_small.generate(
                [small_input], 
                sampling_params=sampling_params,
                return_hidden_states=True, 
                return_logprob=True
            )
            time_detail.append({'speculative_outputs_time': time.time() - spec_start})
            
            spec_output = spec_outputs[0]
            speculative_real_output_text = spec_output['text']
            
            # 检查结束标记
            if '</think>' in speculative_real_output_text:
                spec_end_start = time.time()
                final_outputs = llm_small.generate(
                    [generated_text],
                    sampling_params=sampling_params_end,
                    return_hidden_states=False
                )
                time_detail.append({'speculative_outputs_ending': time.time() - spec_end_start})
                detail.append({'spe_model': final_outputs[0]['text']})
                generated_text += final_outputs[0]['text']
                break
            
            if not speculative_real_output_text:
                break
            
            # 优化: 使用新的隐藏状态处理函数
            extract_start = time.time()
            completion_tokens = spec_output['meta_info']['completion_tokens']
            pooling_hidden_spec = process_hidden_states(
                spec_output["meta_info"]["hidden_states"],
                completion_tokens,
                probe_device
            )
            time_detail.append({'extract_small': time.time() - extract_start})
            
            # 构建验证文本
            checking_target_text = generated_text + speculative_real_output_text
            valid_len = len(target_tokenizer.encode(generated_text))
            
            # 验证生成
            check_start = time.time()
            checking_outputs = llm_big.generate(
                [checking_target_text],
                sampling_params={"temperature": 0.1, "max_new_tokens": 1},
                return_hidden_states=True,
                return_logprob=True,
                logprob_start_len=valid_len - 2,
                top_logprobs_num=2
            )
            
            # 优化: 直接使用math.exp避免重复计算
            import math
            prob_small_result = [
                {"id": token_id, "prob": math.exp(lp)}
                for lp, token_id, _ in spec_output['meta_info']['output_token_logprobs']
            ]
            
            # 快速接受检查
            check_output = checking_outputs[0]
            potential_ids = extract_potential_ids(
                check_output['meta_info']['input_top_logprobs'],
                check_output['meta_info']['input_token_logprobs'],
                completion_tokens
            )
            potential_sets = [set(ids) for ids in potential_ids]
            
            if all(small["id"] in potential_sets[i] for i, small in enumerate(prob_small_result)):
                detail.append({'spe_model': speculative_real_output_text})
                correct_spe_number += 1
                use_target = False
                generated_text = small_input + spec_output['text'] + check_output['text']
                
                if '</think>' in spec_output['text']:
                    final_outputs = llm_small.generate(
                        [generated_text],
                        sampling_params=sampling_params_end,
                        return_hidden_states=False
                    )
                    detail.append({'spe_model': final_outputs[0]['text']})
                    generated_text += final_outputs[0]['text']
                    break_target = True
                    break
                continue
            
            check_time = time.time() - check_start
            time_detail.append({
                'checking_time': check_time,
                'Completion_tokens': completion_tokens,
                'average_token_checking': check_time / completion_tokens
            })
            
            # 处理大模型隐藏状态
            extract_big_start = time.time()
            target_pooling_hidden = process_hidden_states(
                check_output["meta_info"]["hidden_states"],
                completion_tokens + 1,
                probe_device
            )[:-1]  # 去掉最后一个
            time_detail.append({'extract_big': time.time() - extract_big_start})
            
            if target_pooling_hidden.shape[0] == 0:
                break
            
            target_pooling_hidden = target_pooling_hidden.mean(dim=0, keepdim=True)
            
            # 计算接受概率
            prob_start = time.time()
            with torch.no_grad():
                prob_target = model_target_probe(target_pooling_hidden.float()).item()
                prob_spec = model_spec_probe(pooling_hidden_spec.float()).item()
            time_detail.append({'computing_prob_time': time.time() - prob_start})
            
            if speculative_accept(prob_target, prob_spec):
                detail.append({'spe_model': speculative_real_output_text})
                correct_spe_number += 1
                use_target = False
                generated_text = small_input + spec_output['text']
                
                if '</think>' in spec_output['text']:
                    final_outputs = llm_small.generate(
                        [generated_text],
                        sampling_params=sampling_params_end,
                        return_hidden_states=False
                    )
                    detail.append({'spe_model': final_outputs[0]['text']})
                    generated_text += final_outputs[0]['text']
                    break
            else:
                use_target = True
        
        # 目标模型生成
        if use_target:
            begin = False
            try_correct_num += 1
            
            target_start = time.time()
            target_outputs = llm_big.generate(
                [generated_text],
                sampling_params=sampling_params,
                return_hidden_states=False
            )
            time_detail.append({'target_outputs_time': time.time() - target_start})
            
            target_real_output = target_outputs[0]['text']
            
            if '</think>' in target_real_output:
                small_input = speculative_text + target_tokenizer.decode(
                    target_tokenizer(generated_text, return_tensors="pt")['input_ids'][0,
                    original_target_prompt_len:].tolist()
                )
                detail.append({'target_model': target_real_output})
                
                end_start = time.time()
                final_outputs = llm_small.generate(
                    [small_input + target_real_output],
                    sampling_params=sampling_params_end,
                    return_hidden_states=False
                )
                time_detail.append({'target_ending_time': time.time() - end_start})
                detail.append({'spe_model': final_outputs[0]['text']})
                generated_text = small_input + target_real_output + final_outputs[0]['text']
                break
            
            generated_text += target_real_output
            
            if target_tokenizer.eos_token_id in target_tokenizer.encode(target_real_output):
                break
    
    end_time = time.time()
    length_of_output = len(speculative_tokenizer.encode(generated_text[original_spec_prompt_len:]))
    
    # 优化: 使用更高效的时间统计
    total_time = sum(
        t.get(key, 0)
        for t in time_detail
        for key in ['target_outputs_time', 'checking_time', 'computing_prob_time',
                   'speculative_outputs_time', 'speculative_outputs_ending', 'target_ending_time']
    )
    
    return (generated_text, try_correct_num, correct_spe_number, detail,
            length_of_output, end_time - start_time, total_time, time_detail)


def process_file_to_json(dir_path, llm_big, llm_small, target_tokenizer,
                         speculative_tokenizer, problem, answer, max_new_tokens,
                         model_target_probe, model_spec_probe, idx):
    """优化: 简化异常处理，直接返回结果"""
    result = speculative_decoding(
        llm_big, llm_small, target_tokenizer, speculative_tokenizer,
        problem, max_new_tokens, model_target_probe, model_spec_probe
    )
    
    (generated_text, try_correct_num, correct_spe_number, detail,
     length_of_output, times, total_time, time_detail) = result
    
    all_generations = [{
        "input_text": problem,
        "real_answer": generated_text,
        "try_correct_num": try_correct_num,
        "standard_answer": answer,
        "execution_time": f"{times:.2f}s",
        "correct_spe_number": correct_spe_number,
        "total_time": f"{total_time:.2f}s",
        "time_detail": time_detail,
        "detail": detail,
        "length_of_output": length_of_output,
        "index": idx
    }]
    
    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, "spec_generation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, ensure_ascii=False, indent=2)
    
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='aime')
    parser.add_argument("--target_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--speculative_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--data_dir", type=str, default='../speculative/offq')
    parser.add_argument("--start_dataset", type=int, default=0)
    parser.add_argument("--end_dataset", type=int, default=30)
    parser.add_argument("--target_probe", type=str, default="/path/to/target/probe")
    parser.add_argument("--speculative_probe", type=str, default="/path/to/spec/probe")
    parser.add_argument("--max_new_tokens", type=int, default=14000)
    parser.add_argument("--seed", type=int, default=9870)
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    # 加载模型
    model_target_probe = SemanticEntropyProbTarget(5120, 2048)
    model_target_probe.load_state_dict(torch.load(f'{args.target_probe}.pt'))
    model_target_probe = model_target_probe.cuda().eval()
    
    model_spec_probe = SemanticEntropyProbSpec(1536, 1024)
    model_spec_probe.load_state_dict(torch.load(f'{args.speculative_probe}.pt'))
    model_spec_probe = model_spec_probe.cuda().eval()
    
    # 初始化引擎
    llm_small = sgl.Engine(
        model_path=args.speculative_model,
        enable_return_hidden_states=True,
        mem_fraction_static=0.7,
        tp_size=1
    )
    
    llm_big = sgl.Engine(
        model_path=args.target_model,
        enable_return_hidden_states=True,
        mem_fraction_static=0.9,
        tp_size=1
    )
    
    # 加载tokenizer
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.target_model, trust_remote_code=True
    )
    speculative_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.speculative_model, trust_remote_code=True
    )
    
    if speculative_tokenizer.pad_token_id is None:
        speculative_tokenizer.pad_token_id = speculative_tokenizer.eos_token_id
    
    # 加载数据集
    if args.dataset == "math-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")['test']
    elif args.dataset == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    elif args.dataset == "amc23":
        ds = load_dataset("zwhe99/amc23", split="test")
    elif args.dataset == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", split="train")
    else:
        raise ValueError(f"Unknown task: {args.dataset}")
    
    ds = ds.select(range(args.start_dataset, args.end_dataset))
    
    if args.dataset in ["amc23", "gpqa"]:
        problems_and_answers = [{"problem": item["question"], "answer": item["answer"]} for item in ds]
    else:
        problems_and_answers = [{"problem": item["problem"], "answer": item["answer"]} for item in ds]
    
    # 选择测试样本
    wrong_list = {90: [13, 25, 27], 30: [1, 13, 25, 27]}.get(args.seed, [1, 27])
    
    for number in tqdm(wrong_list):
        dirname = f'spec_{args.dataset}_{number}'
        dir_path = os.path.join(f"{args.dataset}{args.data_dir}{args.seed}", dirname)
        problem = problems_and_answers[number]['problem']
        answer = problems_and_answers[number]['answer']
        
        process_file_to_json(
            dir_path, llm_big, llm_small, target_tokenizer,
            speculative_tokenizer, problem, answer, args.max_new_tokens,
            model_target_probe, model_spec_probe, number
        )