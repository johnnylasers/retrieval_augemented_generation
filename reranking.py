from typing import List
from sentence_transformers import CrossEncoder
import chunking
import embedding
import vector_db
import retrieval



def reranking(user_prompt: str, retrieved_chunks:List[str], topK: int) -> List[str]:
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs = [(user_prompt, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = [(chunk, score) for chunk, score in zip(retrieved_chunks, scores)]
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:topK]


# Testing
# chunks = chunking.split_into_chunks("doc.md")
# embeddingList = embedding.embedding_doc(chunks)

# vector_db.save_embeddings(chunks, embeddingList)
# query = "哆啦A梦使用的3个秘密道具分别是什么？"
# retrieved_chunks = retrieval.retrieve(query, 5)

# reranked_chunks = reranking(query, retrieved_chunks, 3)

# for i, chunk in enumerate(reranked_chunks):
#     print(f"[{i}] {chunk}\n")