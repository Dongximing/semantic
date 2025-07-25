from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

outputs = ['333 squared is (300 + 33)^2 = 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889.\n\n',
"First, 333 squared: 333*333. Let's compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889. Wait",
"First, 333 squared: 333*333. Let's compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889.\n\n",
"333 squared: 333*333. Let's compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 = 109800; 109800 + 1089 = 110,889.\n\n",
'333 squared is (300 + 33)^2 = 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889.\n\n',
'333 squared: 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800 + 1089 is 110,889.\n\n',
 'First, 333 squared: 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 = 109800; 109800 + 1089 = 110,889.\n\n',
 '333 squared: 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 = 109800; 109800 + 1089 = 110,889.\n\n',
 "333² is 333*333. Let's compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889.\n\n",
 "333 squared: 333 * 333. Let's compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889.\n\n",
 'First, 333 squared: 333 * 333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889. Wait',
 'First, 333 squared: 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889.\n\n',
 '333 squared: 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 = 109800; 109800 + 1089 = 110,889. Wait',
  "333 squared: 333*333. Let's compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000+19800=109800; 109800 +1089=110,889. Wait",
  '333 squared is 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 = 109800; 109800 + 1089 = 110,889.\n\n',
  '333 squared: 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 = 109800; 109800 + 1089 = 110,889. Wait',
  '333 squared: 333*333. Let me compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889. Wait',
  "First, 333 squared: 333 * 333. Let's compute 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889. Wait",
  '333 squared is 333*333. Let me compute that:\n\n',
   '333² = (300 + 33)^2 = 300² + 2*300*33 + 33² = 90000 + 19800 + 1089 = 90000 + 19800 is 109800, plus 1089 is 110,889.\n\n']


# 1. 获取 embedding
model = SentenceTransformer('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
embeddings = model.encode(outputs, normalize_embeddings=True)

# 2. 计算相似度（AP 需要输入 similarity matrix）
similarity_matrix = cosine_similarity(embeddings)

# 3. 聚类（preference 默认是中位数，也可以调节使分组数量更多/更少）
ap = AffinityPropagation(affinity='precomputed', random_state=42)
labels = ap.fit_predict(similarity_matrix)

# 4. 查看分组
for idx, (output, label) in enumerate(zip(outputs, labels)):
    print(f"[{idx:2}] Cluster {label}: {output.strip()[:60]}...")

print("AP自动聚类标签：", list(labels))
