import pickle
import transformers
import openai
import os
openai.api_key =os.environ["OPENAI_API_KEY"]
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
import hdbscan
import pickle
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

        all_stats = []

        for group_idx, group_start in enumerate(range(0, len(generations), group_size)):
            group = generations[group_start:group_start + group_size]
            valid_idxs = [i for i, g in enumerate(group) if g.get('real_answer_embedding') is not None]
            none_count = group_size - len(valid_idxs)
            embeddings = [group[i]['real_answer_embedding'] for i in valid_idxs]
            group_stats = {
                'group_idx': group_idx,
                'start': group_start,
                'none_count': none_count,
                'total': len(group),
            }

            print(f"\n=== Group {group_idx} (from index {group_start}) ===")
            print(f"本组 embedding 为 None 的有 {none_count} 个")

            # 聚类
            if len(embeddings) >= 2:
                emb_arr = np.array(embeddings)
                dist_mat = cosine_distances(emb_arr)
                clusterer = hdbscan.HDBSCAN(
                    metric="precomputed",
                    min_cluster_size=2,
                    prediction_data=True
                )
                labels = clusterer.fit_predict(dist_mat)
                label_counts = Counter(labels)
                print("Cluster label counts:", dict(label_counts))
                group_stats['label_counts'] = dict(label_counts)

                # silhouette_score 评估
                mask = labels != -1
                if mask.sum() > 1 and len(set(labels[mask])) > 1:
                    sil = silhouette_score(dist_mat[mask][:, mask], labels[mask], metric="precomputed")
                    print(f"Silhouette score (不含噪声): {sil:.3f}")
                    group_stats['silhouette'] = sil
                else:
                    print("Silhouette score: 无法评估（非噪声簇太少）")
                    group_stats['silhouette'] = None
            elif len(embeddings) == 1:
                labels = np.array([0])
                label_counts = Counter(labels)
                print("仅有1个有效 embedding，label=0")
                group_stats['label_counts'] = dict(label_counts)
                group_stats['silhouette'] = None
            else:
                labels = []
                label_counts = {}
                print("本组没有有效 embedding，无法聚类。")
                group_stats['label_counts'] = dict(label_counts)
                group_stats['silhouette'] = None

            # 回写 cluster_id
            for idx, g in enumerate(group):
                if g.get('real_answer_embedding') is not None:
                    idx_in_valid = valid_idxs.index(idx)
                    g['real_answer_cluster_id'] = int(labels[idx_in_valid]) if len(labels) > 0 else None
                else:
                    g['real_answer_cluster_id'] = None

            all_stats.append(group_stats)
            # ------- 可视化并保存 -------
            n_valid = len(embeddings)

            if n_valid >= 2:
                emb_arr = np.array(embeddings)

                if n_valid >= 5:
                    # t-SNE：自动调小 perplexity
                    perp = min(30, max(2, n_valid // 3))
                    tsne_proj = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(emb_arr)
                    proj = tsne_proj
                    viz_title = f"t-SNE (perplexity={perp})"
                else:
                    # n_valid 为 2–4 用 PCA
                    from sklearn.decomposition import PCA

                    proj = PCA(n_components=2, random_state=42).fit_transform(emb_arr)
                    viz_title = "PCA"

                plt.figure(figsize=(6, 5))
                scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=60, edgecolors='k')
                plt.title(f"Group {group_idx} Embedding Cluster Visualization\n{viz_title}")
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.colorbar(scatter, label='Cluster ID')
                plt.tight_layout()
                plt.savefig(f"group_{group_idx}_cluster_viz.png")
                plt.close()
                print(f"已保存 group_{group_idx}_cluster_viz.png")
            else:
                print("有效 embedding < 2，跳过可视化")

        # 保存聚类结果到 pickle
        with open("generations_with_cluster_id.pkl", "wb") as f:
            pickle.dump(generations, f)

        # 保存统计信息到 csv
        df = pd.DataFrame(all_stats)
        df.to_csv("group_cluster_stats.csv", index=False)
        print("\n分组统计信息已保存到 group_cluster_stats.csv")
        print("聚类ID已写回并保存为 generations_with_cluster_id.pkl")

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

