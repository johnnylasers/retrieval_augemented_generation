from sentence_transformers import SentenceTransformer
from typing import List

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
def embed_chunk(chunk: str) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()

embeddings = []
def embedding_doc(chunks: List[str]) -> List[List[float]]:
    for chunk in chunks:
        embeddings.append(embed_chunk(chunk))

    return embeddings

# or do: embeddings = [embed_chunk(chunk) for chunk in chunks]


# Testing:
# chunkList = []
# chunkList.append("chunk1")
# chunkList.append("chunk2")
# print(len(chunkList))
# print(len(embedding_doc(chunkList)))
# print(len(embeddings[0]))



# Testing:
# embedding = embed_chunk("测试内容")
# print(len(embedding))
# print(embedding)

# Have to pin sentence transformer to version 2.2.2 as PyTorch dropped support  for Intel Macs (x86_64) in newer versions. You need to pin torch to an older compatible version:
# uv add sentence-transformers "torch<2.3"


# have two compounding problems:

# sentence-transformers (latest) requires torch >= 2.4, but torch 2.4+ dropped Intel Mac support
# NumPy 2.x is incompatible with modules compiled against NumPy 1.x

# The fix is to pin all three to compatible older versions:
#uv add "sentence-transformers==2.7.0" "torch==2.2.2" "numpy<2"