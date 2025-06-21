import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
import os
import datetime
from collections import Counter


# STOP_TOKENS = [' \n\n','.\n\n', ':\n\n','\n\n', ' Wait', 'Alternatively','Wait',' But', ')\n\n'
#                '?\n\n', ']\n\n',').\n\n'] #' Maybe' 'Hmm'' Hmm'
# STOP_TOKENS_ID = [4710,382,1447,271,13824,92014,14190,1988,692,1939,2533,3593] #10696 80022 88190

AIME_STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n',
]
AIME_STOP_TOKENS_ID = [4710,382,1447,271,692,1939,2533,3593]

def collect_stop_segments(token_ids, stop_ids):
    
    segments = []
    for idx, tid in enumerate(token_ids):
        if tid in stop_ids:
            seg_ids = token_ids[:idx+1]
            stop_token_id = tid
            segments.append((seg_ids, stop_token_id, idx))
    return segments

def process_file(json_path, out_json_path, tokenizer):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    output_token_ids = data.get("large_model_output_ids", [])
    input_token_ids = data.get("large_model_input_ids", [])
    segments = collect_stop_segments(output_token_ids, AIME_STOP_TOKENS_ID)
    out_segments = []
    for seg_ids, stop_token_id, stop_idx in segments:
        stop_token_text = tokenizer.decode([stop_token_id])
        text = tokenizer.decode(input_token_ids+seg_ids)
        out_segments.append({
            "token_ids":  ",".join(map(str, seg_ids)),
            "text": text,
            "stop_token_id": stop_token_id,
            "stop_token_text": stop_token_text,
            "stop_token_idx": stop_idx
        })
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(out_segments, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_json_path} with {len(out_segments)} segments.")

def inference_model(task_name: str, model, tokenizer):
    base_dir = '/home/cs/staff/shaowei/semantic/deepseek-32b_r1_awq_math'
    # base_dir = '/home/shaowei/hf/math-result_left'
    for dirname in os.listdir(base_dir):
        if dirname.startswith('data-500-temp0_'):
            number = dirname.split('_')[-1]
            dir_path = os.path.join(base_dir, dirname)
            if task_name == 'aime':
                json_path = os.path.join(dir_path, f'data-60_{number}.json')
            elif task_name == 'math-500':
                json_path = os.path.join(dir_path, f'data-500_{number}.json')
            out_json_path = os.path.join(dir_path, f"seg_by_stop_{number}.json")
            if not os.path.exists(json_path):
                print(f"{json_path} is not existing! skip!")
                continue
            process_file(json_path, out_json_path, tokenizer)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/QwQ-32B-AWQ",
        trust_remote_code=True
    )
    inference_model("math-500", None, tokenizer)
