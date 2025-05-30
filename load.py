import openai
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
import hdbscan
import pickle
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging

# Logging configuration
log_path = "cluster_labeling.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]

def checking(generations, group_size=20):
    all_same = True
    for i in range(0, len(generations), group_size):
        group = generations[i:i + group_size]
        input_texts = [g['input_text'] for g in group]
        if len(set(input_texts)) > 1:
            logger.warning(f"Group {i // group_size} contains different input_texts!")
            all_same = False
    return all_same

def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=1):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        response = openai.embeddings.create(input=batch, model=model)
        batch_embeddings = [record.embedding for record in response.data]
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

def labeling_data(generations, output_dir):
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

        logger.info(f"\n=== Group {group_idx} (from index {group_start}) ===")
        logger.info(f"Number of None embeddings in this group: {none_count}")

        # Clustering
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
            logger.info(f"Cluster label counts: {dict(label_counts)}")
            group_stats['label_counts'] = dict(label_counts)

            mask = labels != -1
            if mask.sum() > 1 and len(set(labels[mask])) > 1:
                sil = silhouette_score(dist_mat[mask][:, mask], labels[mask], metric="precomputed")
                logger.info(f"Silhouette score (without noise): {sil:.3f}")
                group_stats['silhouette'] = sil
            else:
                logger.info("Silhouette score: Not enough non-noise clusters to evaluate.")
                group_stats['silhouette'] = None
        elif len(embeddings) == 1:
            labels = np.array([0])
            label_counts = Counter(labels)
            logger.info("Only 1 valid embedding, assigned to label=0.")
            group_stats['label_counts'] = dict(label_counts)
            group_stats['silhouette'] = None
        else:
            labels = []
            label_counts = {}
            logger.info("No valid embeddings in this group. Skipping clustering.")
            group_stats['label_counts'] = dict(label_counts)
            group_stats['silhouette'] = None

        # Write back cluster_id
        for idx, g in enumerate(group):
            if g.get('real_answer_embedding') is not None:
                idx_in_valid = valid_idxs.index(idx)
                g['real_answer_cluster_id'] = int(labels[idx_in_valid]) if len(labels) > 0 else None
            else:
                g['real_answer_cluster_id'] = None

        all_stats.append(group_stats)

        # Visualization and saving
        n_valid = len(embeddings)
        if n_valid >= 2:
            emb_arr = np.array(embeddings)
            if n_valid >= 5:
                perp = min(30, max(2, n_valid // 3))
                proj = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(emb_arr)
                viz_title = f"t-SNE (perplexity={perp})"
            else:
                from sklearn.decomposition import PCA
                proj = PCA(n_components=2, random_state=42).fit_transform(emb_arr)
                viz_title = "PCA"

            plt.figure(figsize=(6, 5))
            scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='tab10', s=60, edgecolors='k')
            for x, y, lbl in zip(proj[:, 0], proj[:, 1], labels):
                plt.text(x, y, str(lbl), fontsize=10, ha='center', va='center',
                         bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.15'))
            plt.title(f"Group {group_idx} Embedding Cluster Visualization\n{viz_title}")
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.colorbar(scatter, label='Cluster ID')
            plt.tight_layout()
            fig_path = os.path.join(output_dir, f"group_{group_idx}_cluster_viz.png")
            plt.savefig(fig_path)
            plt.close()
            logger.info(f"Saved plot to {fig_path}")
        else:
            logger.info("Fewer than 2 valid embeddings, skipping visualization.")

    # Save clustering results to pickle
    pickle_path = os.path.join(output_dir, "generations_with_cluster_id.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(generations, f)

    # Save statistics to CSV
    df = pd.DataFrame(all_stats)
    csv_path = os.path.join(output_dir, "group_cluster_stats.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\nGroup statistics saved to {csv_path}")
    logger.info(f"Cluster IDs written back and saved to {pickle_path}")

def process_file_to_pickle(json_path, out_pkl_path):
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/QwQ-32B-AWQ",
        trust_remote_code=True
    )
    with open(json_path, "rb") as f:
        generations = pickle.load(f)
    if checking(generations):
        texts_to_embed = []
        for g in generations:
            input_ids = tokenizer.encode(g['input_text'])
            b = tokenizer.decode(input_ids, skip_special_tokens=True)
            pred = g.get('predicted_answer')
            if pred is None:
                g['real_output'] = None
                g['real_answer_embedding'] = None
            else:
                real_output = pred[len(b):].strip()
                g['real_output'] = real_output
                texts_to_embed.append(real_output)

        # Batch embedding
        if texts_to_embed:
            embs = get_openai_embeddings(texts_to_embed, model="text-embedding-3-small", batch_size=1)
        else:
            embs = []

        idx = 0
        for g in generations:
            if g.get('real_output') is not None:
                g['real_answer_embedding'] = embs[idx]
                idx += 1
            else:
                g['real_answer_embedding'] = None

        # Infer output directory
        output_dir = os.path.dirname(os.path.abspath(out_pkl_path))
        labeling_data(generations, output_dir)

        # Save processed pickle
        with open(out_pkl_path, "wb") as f:
            pickle.dump(generations, f)

def inference_model_pickle(task_name: str = None, model=None, tokenizer=None,
                          base_dir='/home/shaowei/hf/math-result_left',
                          start=10, end=20, num_generations=20):
    for number in range(start, end):
        dirname = f'data-500-temp0_{number}'
        dir_path = os.path.join(base_dir, dirname)
        json_path = os.path.join(dir_path, f'generations_{number}.pkl')
        out_pkl_path = os.path.join(dir_path, f'label_semantic_{number}.pkl')
        if not os.path.exists(json_path):
            logger.warning(f"{json_path} does not exist, skipping.")
            continue
        process_file_to_pickle(json_path, out_pkl_path)

if __name__ == "__main__":
    inference_model_pickle()
