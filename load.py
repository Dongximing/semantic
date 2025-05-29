import pickle
import transformers
import openai
import os
openai.api_key =os.environ["OPENAI_API_KEY"]
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
def get_openai_embeddings(texts, model="text-embedding-3-small", batch_size=20):
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
    for g in generations:
        print(g[1])
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
# output_path = "/home/shaowei/hf/math-result_left/data-500-temp0_10/generations_10_with_real_output.pkl"
# with open(output_path, "wb") as f:
#     pickle.dump(generations, f)
#
# print(f"Saved updated generations to {output_path}")
#
#

