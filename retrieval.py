from typing import List
import embedding
import chromadb
import vector_db
import chunking

# encode user prompt into embedding
# query vector DB to return semantically matched chunks
chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")
def retrieve(user_prompt: str, topK: int) -> List[str]:
    prompt_embedding = embedding.embed_chunk(user_prompt)
    results = chromadb_collection.query(
        query_embeddings=[prompt_embedding],
        n_results=topK
    )
    return results['documents'][0]

# Testing
# chunks = chunking.split_into_chunks("doc.md")
# embeddingList = embedding.embedding_doc(chunks)

# vector_db.save_embeddings(chunks, embeddingList)
# query = "哆啦A梦使用的3个秘密道具分别是什么？"
# retrieved_chunks = retrieve(query, 5)

# for i, chunk in enumerate(retrieved_chunks):
#     print(f"[{i}] {chunk}\n")