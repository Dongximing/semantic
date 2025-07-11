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





def inference_model():
    dataset = load_dataset("simplescaling/s1K-1.1")
    dataset = dataset["train"]
    filtered_dataset = dataset.filter(lambda x: x['cot_type'] == 'math')
    filtered_dataset = filtered_dataset.map(
        lambda x: {'prompt': x['question'], 'solution': x['deepseek_thinking_trajectory']}
    )

    tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit")
    for dataset in tqdm(filtered_dataset):
        messages = [
            {"role": "user", "content": dataset['prompt'] + MATH_PROMPT}
        ]
        formatted_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        question_token_id = tokenizer(formatted_input, return_tensors="pt",split_special_tokens=True)['input_ids'][0]
        answer_token_id = tokenizer(dataset['solution'], return_tensors="pt",split_special_tokens=True)['input_ids'][0]
        total_id = torch.cat([question_token_id, answer_token_id],dim=-1)
        print("total answer:\n", tokenizer.decode(total_id,remove_special_tokens=True))
        sys.exit()










if __name__ == "__main__":

    inference_model()






















