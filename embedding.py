from sentence_transformers import SentenceTransformer
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

outputs = [' but for s=4, which is 1 mod3, the count should be 9? Wait', " but s=4, which is s≡1 mod3, so R(s)=1, count(s)=9. So that's correct.\n\n", ' but earlier for s=4 (which is s≡1 mod3), count(s)=9, so correct.\n\n', " but when s=4, R(s)=1, count(s)=9, so that's okay.\n\n", ', but for s=4, which is s=4, R(s)=1, so count(s)=9. So that matches.\n\n', " but for s=4, which is s≡1 mod3, R(s)=1, so count(s)=9. So that's correct.\n\n", ' but for s=4, which is s≡1 mod3, R(s)=1, count was 9. Correct.\n\n', ' but s=4, which is 1 mod3 (since 4 mod3=1). Wait', ' but earlier for s=4 (which is s≡1 mod3), count(s)=9, so that matches.\n\n', " but earlier for s=4 which is 1 mod3, count(s)=9. So that's correct.\n\n", ' but s=4 is in first residue class (R=1), count(s)=9. Correct.\n\n', ' but for s=4, which is s≡1 mod3, count(s)=9. Correct.\n\n', ' but for s=4, R(s)=1, count(s)=9, so same.\n\n', ', but s=4 is in the first residue again (since s=4 mod3=1). The count was 9, which matches.\n\n', ' but s=4, which is in the first position of the cycle (since s=4≡1 mod3), so count(s)=9, which matches.\n\n', ' but for s=4, which is 1 mod3 (since s=4=3*1+1), so R(s)=1, so count(s)=9. So that matches.\n\n', ' but s=4, which is s≡1 mod3, so count(s)=9. Correct.\n\n', ' but earlier for s=4, which is 1 mod3, count(s)=9, so this is correct.\n\n', ' but s=4 is also s≡1 mod3, so count(s)=9, which matches.\n\n', ' but for s=4, which is s=4≡1 mod3, so R(s)=1. Then count(s)=9, which matches.\n\n']  # 你的模型输出

# 1. 获取 embedding
model = SentenceTransformer('intfloat/e5-mistral-7b-instruct')
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
