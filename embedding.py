
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hdbscan
from umap import UMAP
import matplotlib.pyplot as plt
outputs = ['lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1=-2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1=-2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n', 'lower= -1 -1= -2\n\n']

# 1. 获取 embedding
# model = SentenceTransformer('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
# embeddings = model.encode(outputs, normalize_embeddings=True)\
import openai
# 2. 计算相似度（AP 需要输入 similarity matrix）
def get_openai_embeddings(texts, model="text-embedding-3-small"):
    # 官方接口支持批量，但有长度限制
    response = openai.embeddings.create(input=texts, model=model)
    embeddings = [item.embedding for item in response.data]
    return embeddings

import umap.umap_ as umap
embeddings = get_openai_embeddings(outputs)
plt.figure(figsize=(8, 6))
sim_matrix = cosine_similarity(embeddings)
im = plt.imshow(sim_matrix, cmap='hot', interpolation='nearest')
plt.colorbar(im, label='Cosine Similarity')

plt.title("Embedding Cosine Similarity Heatmap")
plt.xlabel("Output Index")
plt.ylabel("Output Index")

# 可选：加xy轴标签
plt.xticks(range(len(outputs)), range(len(outputs)))
plt.yticks(range(len(outputs)), range(len(outputs)))

plt.tight_layout()
plt.show()


