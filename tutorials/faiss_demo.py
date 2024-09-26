import faiss
from sentence_transformers import SentenceTransformer


# 读取数据
with open('../data/pt/txt/debug_input_val.txt', 'r', encoding='utf-8') as f:
    documents = f.readlines()
print('documents', type(documents), documents[0], len(documents), len(documents[0]))

# 加载模型
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
print('model', type(model), model, model.device)

# 生成向量
embeddings = model.encode(documents, convert_to_numpy=True)
print('embeddings', type(embeddings), embeddings.shape, embeddings[0].shape)

# 创建索引
dimension = embeddings.shape[1]  # 向量维度
index = faiss.IndexFlatL2(dimension)  # L2 距离索引
print('dimension', type(dimension), dimension)

# 添加向量到索引
index.add(embeddings.astype('float32'))

faiss.write_index(index, 'vector_index.faiss')

# 假设 query 是一个字符串
query = "2026年世界杯将在哪些城市举行？"
query_embedding = model.encode([query], convert_to_numpy=True)
print('query_embedding', type(query_embedding), query_embedding.shape)

# 查询相似文本
k = 5  # 返回前 k 个最近邻
D, I = index.search(query_embedding.astype('float32'), k)  # distances, ids
print('D, I', D, I)

# 输出结果
for i in range(k):
    print(f"相似文本: {documents[I[0][i]]}, 距离: {D[0][i]}")