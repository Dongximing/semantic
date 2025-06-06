import openai
import logging
import torch
import torch.nn.functional as F
import os
import hdbscan
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_score
from collections import Counter
from sklearn.manifold import TSNE
from openai import OpenAI
from tqdm import tqdm
from semantic_entropy import cluster_assignment_entropy, predictive_entropy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)
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

def checking(generations, group_size=21):
    all_same = True
    for i in range(0, len(generations), group_size):
        group = generations[i:i + group_size]

        input_texts = [group[0]['most_input_text']] + [g['input_text'] for g in group[1:]]


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
def equivalence_prompt(text1, text2, prefix):

    prompt = f"""We are evaluating generation to the prefix \"{prefix}\"\n"""

    prompt += "Here are two possible generation:\n"
    prompt += f"Possible Generation 1: {text1}\nPossible Generation 2: {text2}\n"
    prompt += "Does Possible Generation 1 semantically entail Possible Generation 2? Only respond with entailment, contradiction, or neutral."""

    return prompt
def get_deberta_output(text1,text2,model,tokenizer):
    inputs = tokenizer(text1, text2, return_tensors="pt").to("cuda:3")
    outputs = model(**inputs)
    logits = outputs.logits
    # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
    largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
    prediction = largest_index.cpu().item()

    return prediction



def get_openai_output(text1,text2,prefix):
    prompt = equivalence_prompt(text1, text2, prefix)

    messages = [
            {"role": "user", "content": prompt},
        ]

    output = CLIENT.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        max_tokens=4,
        temperature=0.1,
    )
    response = output.choices[0].message.content
    # print("response",response)
    binary_response = response.lower()
    if 'entailment' in binary_response:
        return 2
    elif 'neutral' in binary_response:
        return 1
    elif 'contradiction' in binary_response:
        return 0
    else:
        logging.warning('MANUAL NEUTRAL!')
        return 1
    return binary_response

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
            cluster= hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=2,
                prediction_data=True
            )
            labels = cluster.fit_predict(dist_mat)
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
def get_semantic_ids(strings_list, model,prefix, strict_entailment=True, tokenizer=None, method = 'deberta'):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2, prefix):


        if method == 'embedding':
            implication_1 = get_openai_output(text1, text2, prefix=prefix)
            implication_2 = get_openai_output(text2, text1, prefix=prefix)
        elif method == 'openai':
            implication_1 = get_openai_output(text1, text2, prefix=prefix)
            implication_2 = get_openai_output(text2, text1, prefix=prefix)
        elif method == 'deberta':
            implication_1 = get_deberta_output(text1, text2,model,tokenizer)
            implication_2 = get_deberta_output(text1, text2,model,tokenizer)



            # pylint: disable=arguments-out-of-order

        # print("implication_1",implication_1)
        # print("implication_2", implication_2)

        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j],prefix):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids
def process_file_to_pickle(json_path, out_pkl_path):
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "Qwen/QwQ-32B-AWQ",
    #     trust_remote_code=True
    # )
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     "microsoft/deberta-v2-xlarge-mnli").to("cuda:3")

    group_size = 21
    with open(json_path, "rb") as f:
        generations = pickle.load(f)
    all_generations = []
    if checking(generations):
        for i in range(0, len(generations), group_size):


            group = generations[i:i + group_size]
            # answer_lists = [g.get('real_answer') for g in group[1:]]
            #
            # valid_answers = [ans for ans in answer_lists if ans is not None]
            #
            # if valid_answers:
            #
            #     cluster_ids = get_semantic_ids(strings_list=valid_answers, model=model,tokenizer=tokenizer,
            #                                    prefix=group[0]['most_input_text'],method='deberta')
            # else:
            #     cluster_ids = []
            #
            # cluster_gpt = []
            # cid = 0
            # for idx, ans in enumerate(answer_lists):
            #     if ans is None:
            #         cluster_gpt.append(None)
            #     else:
            #         cluster_gpt.append(cluster_ids[cid])
            #         cid += 1
            # if len(valid_answers)>0:
            #     group[0]['cluster_assignment_entropy_deberta'] = cluster_assignment_entropy([c for c in cluster_gpt if c is not None])
            # else:
            #     group[0]['cluster_assignment_entropy_deberta'] = None
            #
            #
            #
            # for local_idx, g in enumerate(group[1:]):
            #     g['clustering-gpt-prompty_deberta'] = cluster_gpt[local_id
            labels = []
            for local_idx, g in enumerate(group[1:]):
                label = g['clustering-gpt-prompty_deberta']

                if label is not None:
                    labels.append(label)
            label_counts = Counter(labels)
            total = len(labels)
            for g in group[1:]:
                label = g['clustering-gpt-prompty_deberta']
                print(label)
                if label is not None:
                    g['probability_of_deberta'] = label_counts[label] / total
                else:
                    g['probability_of_deberta'] = None


            all_generations.extend(group)

        with open(out_pkl_path, "wb") as f:
            pickle.dump(all_generations, f)





def inference_model_pickle(task_name: str = None, model=None, tokenizer=None,
                          base_dir='/data/ximing/math-result_left/',
                          start=0, end=100, num_generations=20):


    for number in tqdm(range(start, end)):
        dirname = f'data-500-temp0_{number}'
        dir_path = os.path.join(base_dir, dirname)
        json_path = os.path.join(dir_path, f'new_generations_with_entropy{number}.pkl')

        out_pkl_path = os.path.join(dir_path, f'new_generations_with_entropy_and_prob{number}.pkl')
        if not os.path.exists(json_path):
            logger.warning(f"{json_path} does not exist, skipping.")
            continue
        process_file_to_pickle(json_path, out_pkl_path)

if __name__ == "__main__":
    inference_model_pickle()
