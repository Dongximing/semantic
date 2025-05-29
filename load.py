import pickle
import transformers
import openai
import os
openai.api_key =os.environ["OPENAI_API_KEY"]
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import numpy as np
import hdbscan
from collections import Counter
from sklearn.metrics import silhouette_score
def checking(generations):
    group_size = 20

    all_same = True
    for i in range(0, len(generations), group_size):
        group = generations[i:i + group_size]
        input_texts = [g['input_text'] for g in group]
        if len(set(input_texts)) > 1:
            print(f"第 {i // group_size} 组有不同 input_text!")
            all_same = False
    return  all_same
def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=1):
    """
    texts: list of strings,
    model: "text-embedding-3-small"
    return: np.array, shape = (N, D)
    """
    import numpy as np
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(input=batch, model=model)

        batch_embeddings = [record.embedding for record in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

with open("/home/shaowei/hf/math-result_left/data-500-temp0_10/generations_10_with_real_output.pkl", "rb") as f:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/QwQ-32B-AWQ",
        trust_remote_code=True
    )
    generations = pickle.load(f)
    if checking(generations):
        group_size = 20

    for group_idx, group_start in enumerate(range(0, len(generations), group_size)):
        group = generations[group_start:group_start + group_size]
        valid_idxs = [i for i, g in enumerate(group) if g.get('real_answer_embedding') is not None]
        none_count = group_size - len(valid_idxs)  # None 的数量
        print(f"\n=== Group {group_idx} (from index {group_start}) ===")
        print(f"本组 embedding 为 None 的有 {none_count} 个")

        embeddings = [group[i]['real_answer_embedding'] for i in valid_idxs]

        if len(embeddings) >= 1:
            emb_arr = np.array(embeddings)
            clusterer = hdbscan.HDBSCAN(
                metric="cosine",
                min_cluster_size=2,
                prediction_data=True
            )
            labels = clusterer.fit_predict(emb_arr)
            # 统计 label 数量
            label_counts = Counter(labels)
            print("Cluster label counts:", dict(label_counts))
        else:
            labels = []
            print("本组没有有效 embedding，无法聚类。")
            label_counts = {}

        # 回写 cluster_id
        for idx, g in enumerate(group):
            if g.get('real_answer_embedding') is not None:
                idx_in_valid = valid_idxs.index(idx)
                g['real_answer_cluster_id'] = int(labels[idx_in_valid]) if len(labels) > 0 else None
            else:
                g['real_answer_cluster_id'] = None

    # print(len(generations))

    # for g in generations:
    #     pred = g.get('real_output')
    #     if pred is None:
    #         g['real_answer_embedding'] = None
    #
    #     else:
    #         g['real_answer_embedding'] = get_openai_embeddings(texts=[pred])[0]
    #         print(g['real_answer_embedding'])



#
#         input = tokenizer.encode(g['input_text'])
#         b = tokenizer.decode(input, skip_special_tokens=True)
#
#         pred = g.get('predicted_answer')
#         if pred is None:
#             print(b)
#             g['real_output'] = None  # 或者 continue 跳过不加
#         else:
#             g['real_output'] = pred[len(b):]
#
output_path = "/home/shaowei/hf/math-result_left/data-500-temp0_10/generations_10_with_real_output.pkl"
with open(output_path, "wb") as f:
    pickle.dump(generations, f)

print(f"Saved updated generations to {output_path}")
#
#

