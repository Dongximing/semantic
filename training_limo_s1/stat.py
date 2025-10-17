import openai
import logging
import torch
import torch.nn.functional as F
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math  # 用于log2

from collections import Counter
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 初始化logger（原代码缺少）
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

def checking(generations, group_size=21):
    all_same = True
    for i in range(0, len(generations), group_size):
        group = generations[i:i + group_size]
        input_texts = [group[0]['most_input_text']] + [g['input_text'] for g in group[1:]]
        if len(set(input_texts)) > 1:
            logger.warning(f"Group {i // group_size} contains different input_texts!")
            all_same = False
    return all_same

def process_file_to_pickle(input_pkl_path):
    group_size = 21
    with open(input_pkl_path, "rb") as f:  # 修复：用input_pkl_path
        generations = pickle.load(f)
    
    all_generations = []  # 未用，可删
    local_labels = []  # 收集本文件的labels

    if checking(generations):
        for i in range(0, len(generations), group_size):
            group = generations[i:i + group_size]
            answer_lists = [g.get('real_answer') for g in group[1:]]  # 未用，可删
            labels = []
            for local_idx, g in enumerate(group[1:]):
                label = g['clustering-gpt-prompty_deberta']
                if label is not None:
                    labels.append(label)
            print(labels)  # 原有print
            local_labels.extend(labels)  # 收集到本地list
            label_counts = Counter(labels)
            # 这里可以加本地处理，但我们移到全局

    
    return local_labels  # 返回本文件的labels，用于全局统计

def inference_model_pickle(base_dir='/home/cs/staff/shaowei/semantic/training_limo_s1/data_s1_100',
                           start=0, end=877):
    all_labels = []  # 收集所有文件的labels
    # wrong = [4, 5, 2, 6, 11, 12, 13, 18, 20, 21, 25, 26, 29, 30, 33, 35, 38, 44, 46, 47, 49, 50, 51, 56, 57, 59]
    
    for number in tqdm(range(start, end)):
        # if number in wrong: continue
        dirname = f'data-877_{number}'
        dir_path = os.path.join(base_dir, dirname)
        input_pkl_path = os.path.join(dir_path, f'new_generations_with_entropy_prob{number}.pkl')  # 假设输入pkl名，改成你的实际
 
        
        if not os.path.exists(input_pkl_path):  # 修复：检查input
            logger.warning(f"{input_pkl_path} does not exist, skipping.")
            continue
        local_labels = process_file_to_pickle(input_pkl_path)
        all_labels.extend(local_labels)  # 汇总
    
    # 现在统计整体分布
    if all_labels:
        total_count = len(all_labels)
        label_counts = Counter(all_labels)
        
        # 生成表（用pandas）
        df = pd.DataFrame({
            'Label': list(label_counts.keys()),
            'Count': list(label_counts.values()),
            'Proportion': [count / total_count for count in label_counts.values()]
        })
        df = df.sort_values(by='Count', ascending=False)
        print("\nLabel分布表：")
        print(df)
        
        # 绘制柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(df['Label'], df['Count'])
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Label Distribution Bar Chart')
        plt.xticks(rotation=45)
        plt.savefig('label_distributio_samll.png')
        print("柱状图已保存为 'label_distribution.png'")
        
        # 计算多样性
        proportions = np.array(list(label_counts.values())) / total_count
        shannon_entropy = -np.sum(proportions * np.log2(proportions + 1e-10))  # 避免log0
        gini_simpson = 1 - np.sum(proportions ** 2)
        
        print(f"\n多样性指标：")
        print(f"总类别数: {len(label_counts)}")
        print(f"Shannon Entropy: {shannon_entropy:.4f} (最大可能: {math.log2(len(label_counts)):.4f})")
        print(f"Gini-Simpson Index: {gini_simpson:.4f} (范围[0,1])")
        
        # 评估
        if shannon_entropy > 1.5 and gini_simpson > 0.5:
            print("多样性中等以上，分布较均匀。")
        elif shannon_entropy < 0.5:
            print("多样性低，可能有严重bias。")
        else:
            print("多样性一般，建议检查是否需要平衡数据。")
    else:
        print("无labels数据，无法统计。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 修复：parser而非argparse
    parser.add_argument('--base_dir', type=str, default='/shared_workspace_mfs/ximing/data_s1_200_segments_math_small')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=200)
    args = parser.parse_args()
    inference_model_pickle(base_dir=args.base_dir, start=args.start, end=args.end)