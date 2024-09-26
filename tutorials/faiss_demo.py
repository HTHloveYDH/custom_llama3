import faiss
from sentence_transformers import SentenceTransformer


# 读取数据
with open('data.txt', 'r', encoding='utf-8') as f:
    documents = f.readlines()

# 加载模型
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# 生成向量
embeddings = model.encode(documents, convert_to_numpy=True)

# 创建索引
dimension = embeddings.shape[1]  # 向量维度
index = faiss.IndexFlatL2(dimension)  # L2 距离索引

# 添加向量到索引
index.add(embeddings.astype('float32'))

faiss.write_index(index, 'vector_index.faiss')

# 假设 query 是一个字符串
query = "2026年世界杯将在哪些城市举行？"
query_embedding = model.encode([query], convert_to_numpy=True)

# 查询相似文本
k = 5  # 返回前 k 个最近邻
D, I = index.search(query_embedding.astype('float32'), k)

# 输出结果
for i in range(k):
    print(f"相似文本: {documents[I[0][i]]}, 距离: {D[0][i]}")