from modelscope import snapshot_download
from src import DocumentChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_dir = r'D:\models\BAAI\bge-large-zh-v1.5'  # 向量模型
embeddings = HuggingFaceBgeEmbeddings(model_name=model_dir, model_kwargs={'device': 'cpu'})

# 测试这个embeddings
# text = "你好吗？"
# embedding = embeddings.embed_query(text)
# print(embedding)


loader = DocumentChunker('llama2.pdf')
documents = loader.load()  # 加载段落
print(documents)

# model_dir = snapshot_download('BAAI/bge-large-zh-v1.5', cache_dir='D:\\models')

# print(model_dir)