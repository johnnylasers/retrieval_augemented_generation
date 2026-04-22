import chromadb
from typing import List
import chunking
import embedding

chromadb_client = chromadb.EphemeralClient()
chromadb_collection = chromadb_client.get_or_create_collection(name="default")

# save the embeddings, along with the chunks, into the Vector DB
# version 1
# def save_embeddings(chunks:List[str], embeddings:List[List[float]]) -> None:
#     for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#         chromadb_collection.add(
#             documents=[chunk],
#             embeddings=[embedding],
#             ids=[str(i)]
#         )

# version 2
def save_embeddings(chunks:List[str], embeddings:List[List[float]]) -> None:
    ids = [str(i) for i in range(len(chunks))]
    chromadb_collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


# testing:
# chunks = chunking.split_into_chunks("doc.md")
# embeddingList = embedding.embedding_doc(chunks)

# save_embeddings(chunks, embeddingList)

# uv add chromadb onnxruntime==1.19.2, or
# uv add chromadb --override "onnxruntime<1.20"