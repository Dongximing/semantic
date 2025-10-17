import os
import pickle
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse


def checking(generations, group_size=21):
    all_same = True
    for i in range(0, len(generations), group_size):
        group = generations[i:i + group_size]
        input_texts = [group[0]['most_input_text']] + [g['input_text'] for g in group[1:]]
        if len(set(input_texts)) > 1:
            print(f"Group {i // group_size} contains different input_texts!")
            all_same = False
    return all_same

def get_deberta_output(text1, text2, model, tokenizer):
    inputs = tokenizer(text1, text2, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits
    largest_index = torch.argmax(F.softmax(logits, dim=1))
    prediction = largest_index.cpu().item()
    return prediction

def get_semantic_ids(strings_list, model, prefix, strict_entailment=True, tokenizer=None, method='deberta'):
    def are_equivalent(text1, text2, prefix):
        implication_1 = get_deberta_output(text1, text2, model, tokenizer)
        implication_2 = get_deberta_output(text2, text1, model, tokenizer)
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])
        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        return semantically_equivalent

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j], prefix):
                    semantic_set_ids[j] = next_id
            next_id += 1
    assert -1 not in semantic_set_ids
    return semantic_set_ids

def process_file_to_pickle(json_path, out_pkl_path, small_json_path, small_out_pkl_path, gpu):
    tokenizer = AutoTokenizer.from_pretrained("./deberta-v2-xlarge-mnli")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./deberta-v2-xlarge-mnli").to(f"cuda:{gpu}")
    group_size = 21

    # 加载大模型和小模型文件
    large_generations = []
    small_generations = []
    if os.path.exists(json_path):
        with open(json_path, "rb") as f:
            large_generations = pickle.load(f)
    if os.path.exists(small_json_path):
        with open(small_json_path, "rb") as f:
            small_generations = pickle.load(f)

    # 检查输入一致性
    large_valid = checking(large_generations) if large_generations else False
    small_valid = checking(small_generations) if small_generations else False
    if not (large_valid and small_valid):
        print(f"Checking failed: large={large_valid}, small={small_valid}")
        return

    large_output = []
    small_output = []
    for i in range(0, max(len(large_generations), len(small_generations)), group_size):
        # 获取大模型和小模型的组
        large_group = large_generations[i:i + group_size] if i < len(large_generations) else []
        small_group = small_generations[i:i + group_size] if i < len(small_generations) else []

        # 提取答案，带索引跟踪来源
        large_answers = [g.get('real_answer') for g in large_group[1:]] if large_group else []
        small_answers = [g.get('real_answer') for g in small_group[1:]] if small_group else []
        
        # 合并答案，记录来源
        valid_answers = []
        answer_sources = []
        for idx, ans in enumerate(large_answers):
            if ans is not None:
                valid_answers.append(ans)
                answer_sources.append(('large', idx))
        for idx, ans in enumerate(small_answers):
            if ans is not None:
                valid_answers.append(ans)
                answer_sources.append(('small', idx))
        
        # 验证句子数量
        expected_count = min(len(large_answers), 20) + min(len(small_answers), 20)
        if len(valid_answers) > expected_count:
            print(f"Group {i // group_size}: Expected up to {expected_count} valid answers, got {len(valid_answers)}")
        if len(valid_answers) < expected_count // 2:  # 警告如果句子数远少于预期
            print(f"Group {i // group_size}: Expected ~{expected_count} valid answers, got only {len(valid_answers)}")
        
        # 获取prefix
        prefix = large_group[0]['most_input_text'] if large_group else small_group[0]['most_input_text'] if small_group else ""

        # 对合并的句子进行语义聚类
        if valid_answers:
            print("valid_answers",valid_answers)
            
            print(f"Group {i // group_size} merged answer_lists ({len(valid_answers)} sentences): \n{valid_answers}\n")
            cluster_ids = get_semantic_ids(strings_list=valid_answers, model=model, tokenizer=tokenizer, prefix=prefix, method='deberta')
            print(f"Group {i // group_size} merged cluster_ids: \n{cluster_ids}\n")
        else:
            print(f"Group {i // group_size}: No valid answers")
            cluster_ids = []

        # 合并labels用于统一的概率计算
        merged_labels = []

        # 为大模型分配cluster_ids
        if large_group:
            large_cluster_gpt = [None] * len(large_answers)
            for source, idx in answer_sources:
                if source == 'large' and idx < len(large_cluster_gpt):
                    large_cluster_gpt[idx] = cluster_ids[answer_sources.index((source, idx))] if answer_sources.index((source, idx)) < len(cluster_ids) else None
            for local_idx, g in enumerate(large_group[1:]):
                g['clustering-gpt-prompty_deberta'] = large_cluster_gpt[local_idx]
            large_labels = [g['clustering-gpt-prompty_deberta'] for g in large_group[1:] if g['clustering-gpt-prompty_deberta'] is not None]
            if large_labels:
                merged_labels.extend(large_labels)
                print(f"Group {i // group_size} large model labels (count={len(large_labels)}): {large_labels}")
            else:
                print(f"Group {i // group_size} large model labels: empty")
            large_output.extend(large_group)
        else:
            print(f"Group {i // group_size} large model: empty")

        # 为小模型分配cluster_ids
        if small_group:
            small_cluster_gpt = [None] * len(small_answers)
            for source, idx in answer_sources:
                if source == 'small' and idx < len(small_cluster_gpt):
                    small_cluster_gpt[idx] = cluster_ids[answer_sources.index((source, idx))] if answer_sources.index((source, idx)) < len(cluster_ids) else None
            for local_idx, g in enumerate(small_group[1:]):
                g['clustering-gpt-prompty_deberta'] = small_cluster_gpt[local_idx]
            small_labels = [g['clustering-gpt-prompty_deberta'] for g in small_group[1:] if g['clustering-gpt-prompty_deberta'] is not None]
            if small_labels:
                merged_labels.extend(small_labels)
                print(f"Group {i // group_size} small model labels (count={len(small_labels)}): {small_labels}")
            else:
                print(f"Group {i // group_size} small model labels: empty")
            small_output.extend(small_group)
        else:
            print(f"Group {i // group_size} small model: empty")

        # 计算统一的label_counts
        label_counts = Counter(merged_labels)
        total = len(merged_labels)
        print(f"Group {i // group_size} merged label_counts: {label_counts}")

        # 为大模型设置概率（基于合并的label_counts）
        if large_group:
            for g in large_group[1:]:
                label = g['clustering-gpt-prompty_deberta']
                g['probability_of_deberta'] = label_counts[label] / total if label is not None and total > 0 else None

        # 为小模型设置概率（基于合并的label_counts）
        if small_group:
            for g in small_group[1:]:
                label = g['clustering-gpt-prompty_deberta']
                g['probability_of_deberta'] = label_counts[label] / total if label is not None and total > 0 else None

    # 保存输出
    if large_output:
        with open(out_pkl_path, "wb") as f:
            pickle.dump(large_output, f)
            print(f"Saved large model output to {out_pkl_path}")
    if small_output:
        with open(small_out_pkl_path, "wb") as f:
            pickle.dump(small_output, f)
            print(f"Saved small model output to {small_out_pkl_path}")

def inference_model_pickle(
    base_dir='/home/cs/staff/shaowei/semantic/training_limo_s1/data_s1_100',
    base_dir_small=None,
    start=0,
    end=877,
    gpu=None
):
    """
    批量处理大模型和小型闭源模型的pickle文件，调用process_file_to_pickle。
    
    Args:
        base_dir (str): 大模型数据目录
        base_dir_small (str): 小模型数据目录
        start (int): 文件编号起始
        end (int): 文件编号结束
        gpu (str/int): GPU设备
    """
    if base_dir_small is None:
        print("base_dir_small is None, only processing base_dir.")

    for number in tqdm(range(start, end)):
        # 大模型路径
        dirname = f'data-877_{number}'
        dir_path = os.path.join(base_dir, dirname)
        json_path = os.path.join(dir_path, f'new_generations_{number}.pkl')
        out_pkl_path = os.path.join(dir_path, f'combine_new_generations_with_entropy_prob{number}.pkl')

        # 小模型路径
        small_dir_path = os.path.join(base_dir_small, dirname) if base_dir_small else None
        small_json_path = os.path.join(small_dir_path, f'new_generations_{number}.pkl') if base_dir_small else None
        small_out_pkl_path = os.path.join(small_dir_path, f'combine_new_generations_with_entropy_prob{number}.pkl') if base_dir_small else None

        # 检查大模型文件
        if not os.path.exists(json_path):
            print(f"{json_path} does not exist, skipping (base_dir).")
            continue
        if os.path.exists(out_pkl_path):
            print(f"{out_pkl_path} already exists, skipping (base_dir).")
            continue

        # 检查小模型文件
        if base_dir_small and not os.path.exists(small_json_path):
            print(f"{small_json_path} does not exist, skipping (base_dir_small).")
            continue
        if base_dir_small and os.path.exists(small_out_pkl_path):
            print(f"{small_out_pkl_path} already exists, skipping (base_dir_small).")
            continue

        # 处理文件
        try:
            process_file_to_pickle(json_path, out_pkl_path, small_json_path, small_out_pkl_path, gpu)
            print(f"{number}: {json_path} and {small_json_path}")
        except Exception as e:
            print(f"Error processing number {number}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pickle files for large and small models")
    parser.add_argument('--base_dir_big', type=str, default='/shared_workspace_mfs/ximing/data_s1_200_segments_math')
    parser.add_argument('--base_dir_small', type=str, default='/shared_workspace_mfs/ximing/data_s1_200_segments_math_small')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=25)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    inference_model_pickle(
        base_dir=args.base_dir_big,  # 修复：base_dir_big -> base_dir
        base_dir_small=args.base_dir_small,
        start=args.start,
        end=args.end,
        gpu=args.gpu
    )