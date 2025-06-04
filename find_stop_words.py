import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
import os
import datetime
from collections import Counter
def inference_model(base_dir, task, tokenizer):

    all_token_ids = []
    for dirname in os.listdir(base_dir):
        if dirname.startswith('data-500-temp0_'):
            number = dirname.split('_')[-1]
            dir_path = os.path.join(base_dir, dirname)
            if task == 'aime':
                json_path = os.path.join(dir_path, f'data-60_{number}.json')
            elif task == 'math-500':
                json_path = os.path.join(dir_path, f'data-500_{number}.json')

            if not os.path.exists(json_path):
                print(f"{json_path} is not existing! skip!")
                continue
          
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                token_ids = data.get("large_model_output_ids", [])
                all_token_ids.extend(token_ids)
    counter = Counter(all_token_ids)
    most_common_200 = counter.most_common(400)
    for tid, freq in most_common_200:
        try:
            token_text = tokenizer.decode([tid])
            print(f"{tid}\t{freq}\t{repr(token_text)}")
        except Exception as e:
            print(f"{tid}\t{freq}\tDecode error: {e}")


                

STOP_TOKENS = [' \n\n','.\n\n', ':\n\n','\n\n', ' Wait', 'Alternatively','Wait',' But',' Hmm', ')\n\n'
               '?\n\n', 'Hmm', ']\n\n',').\n\n',' Maybe']
           

            
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-model-path",
        type=str,
        default="Qwen/QwQ-32B-AWQ",
    )
    # Qwen/QwQ-32B-AWQ
    #/home/ximing/QwQ-32B-AWQ/
    parser.add_argument(
        "--task",
        type=str,
        default="aime",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.main_model_path,trust_remote_code=True)
    #/home/cs/staff/shaowei/semantic/aime
    inference_model(base_dir = '/home/cs/staff/shaowei/semantic/aime',task = args.task,tokenizer=tokenizer)

if __name__ == "__main__":
    main()
