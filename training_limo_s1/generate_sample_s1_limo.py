import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
import os
import datetime
from datasets import load_dataset
from tqdm import tqdm
import sys
MATH_PROMPT = "\nPlease reason step by step, and put your final answer within \\boxed{}."

AIME_STOP_TOKENS = [
    ' \n\n', '.\n\n', ':\n\n', '\n\n',
    ')\n\n', '?\n\n', ']\n\n', ').\n\n',
]
AIME_STOP_TOKENS_ID = [4710, 382, 1447, 271, 692, 1939, 2533, 3593]


def collect_stop_segments(token_ids, stop_ids):
    segments = []
    for idx, tid in enumerate(token_ids):
        if tid in stop_ids:
            seg_ids = token_ids[:idx + 1]
            stop_token_id = tid
            segments.append((seg_ids, stop_token_id, idx))
    return segments


def inference_model():
    # 加载并筛选数据
    dataset = load_dataset("simplescaling/s1K-1.1")["train"]
    filtered_dataset = dataset.filter(lambda x: x['cot_type'] == 'math')
    filtered_dataset = filtered_dataset.map(
        lambda x: {'prompt': x['question'], 'solution': x['deepseek_thinking_trajectory']}
    )

    tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit", skip_special_tokens=True)

    # 新建存放子文件的文件夹
    out_dir = './data_s1/'
    os.makedirs(out_dir, exist_ok=True)

    for idx, item in enumerate(tqdm(filtered_dataset)):
        messages = [
            {"role": "user", "content": item['prompt'] + MATH_PROMPT}
        ]
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        question_token_id = tokenizer(formatted_input, return_tensors="pt", split_special_tokens=True)['input_ids'][0]
        answer_token_id = tokenizer(item['solution'], return_tensors="pt")['input_ids'][0]

        out_segments = []
        segments = collect_stop_segments(answer_token_id[1:], AIME_STOP_TOKENS_ID)
        for seg_ids, stop_token_id, stop_idx in segments:
            stop_token_text = tokenizer.decode([stop_token_id])
            print('seg_ids',seg_ids)
            print('question_token_id[1:].unsqueeze(0)',question_token_id[1:].unsqueeze(0))
            text = tokenizer.decode(question_token_id[1:].tolist() + seg_ids.tolist())
            out_segments.append({
                "token_ids": ",".join(map(str, seg_ids)),
                "text": text,
                "stop_token_id": stop_token_id,
                "stop_token_text": stop_token_text,
                "stop_token_idx": stop_idx
            })

        # 每个样本一个子目录
        sample_dir = os.path.join(out_dir, f"data-877_{idx}")
        os.makedirs(sample_dir, exist_ok=True)
        out_json_path = os.path.join(sample_dir, "generation.json")
        with open(out_json_path, 'w', encoding='utf-8') as f:
            json.dump(out_segments, f, indent=2, ensure_ascii=False)
        print(f"Saved {out_json_path} with {len(out_segments)} segments.")










if __name__ == "__main__":

    inference_model()






















