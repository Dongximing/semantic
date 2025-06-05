import json
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import scipy
import torch
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
log_path = "filtering.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def checking(generations, group_size=21):
    all_same = True
    for i in range(0, len(generations), group_size):
        group = generations[i:i + group_size]

        input_texts = [group[0]['most_input_text']] + [g['input_text'] for g in group[1:]]


        if len(set(input_texts)) > 1:
            logger.warning(f"Group {i // group_size} contains different input_texts!")
            all_same = False
    return all_same

def process_file_to_pickle(pkl_path, out_pkl_path, tokenizer):
    number = 21
    val_data = []

    with open(pkl_path, 'rb') as f:
        generations = pickle.load(f)
    if checking(generations):
        for i in range(0, len(generations), number):
            g = generations[i]
            if g['most_likely_answer']['answer'] is None:
                logger.warning(f"Group {i} most_likely_answer i None! Skipping...")
                continue
            else:
                val_data.append(g)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(val_data, f)
    print(f"Processed {len(val_data)} items (saved to {out_pkl_path}).")







def data_preprocess(base_dir, task_name,tokenizer):
    end = 0
    start = 100
    for number in tqdm(range(start, end)):
        if task_name == 'math-500':
            dirname = f'data-500-temp0_{number}'
            dir_path = os.path.join(base_dir, dirname)
            pkl_path = os.path.join(dir_path, f'new_generations_with_entropy{number}.pkl')
            out_pkl_path = os.path.join(dir_path, f'valid_preprocess_{number}.pkl')
        elif task_name == 'aime':
            dirname = f'data-500-temp0_{number}'
            dir_path = os.path.join(base_dir, dirname)
            pkl_path = os.path.join(dir_path, f'new_generations_with_entropy{number}.pkl')
            out_pkl_path = os.path.join(dir_path, f'valid_preprocess_{number}.pkl')
        if not os.path.isfile(pkl_path):
            print(f"[Warning] {pkl_path} does not exist! Skipping...")
            continue
        process_file_to_pickle(pkl_path, out_pkl_path, tokenizer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/ximing/math-result_left",
    )
    parser.add_argument(
        "--main_model_path",
        type=str,
        default="Qwen/QwQ-32B-AWQ",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="math-500",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.main_model_path)
    data_preprocess(task_name=args.task, base_dir=args.data_path, tokenizer=tokenizer)

if __name__ == "__main__":
    main()











