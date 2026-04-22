from dotenv import load_dotenv
from google import genai
from typing import List
import vector_db
import embedding
import chunking
import retrieval
import reranking

load_dotenv()

google_client = genai.Client()

def generate(query: str, chunks: List[str]) -> str:
    prompt = f"""You are a domain assistant. Please help generate precise answer based on User querty and the following chunks.PermissionError

    User prompt: {query}
    Relevant chunks: {"\n\n".join(chunks)}

    Please response based on the context provided above. Don't come up with anything new on your won. """

    print(f"{prompt}\n\n---\n")

    response = google_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

user_prompt = "哆啦A梦使用的3个秘密道具分别是什么？"
chunks = chunking.split_into_chunks("doc.md")
embeddingList = embedding.embedding_doc(chunks)

vector_db.save_embeddings(chunks, embeddingList)
retrieved_chunks = retrieval.retrieve(user_prompt, 5)
print(len(retrieved_chunks))
reranked_chunks = reranking.reranking(user_prompt, retrieved_chunks, 3)
print(len(reranked_chunks))
answer = generate(user_prompt, reranked_chunks)
print(answer)