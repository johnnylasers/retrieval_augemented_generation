from dotenv import load_dotenv
from typing import List
import vector_db
import embedding
import chunking
import retrieval
import reranking

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
_prompt = PromptTemplate.from_template(
    "You are a domain assistant. Answer based only on the context below.\n\n"
    "Context:\n{chunks}\n\n"
    "Question: {query}"
)
_chain = _prompt | _llm | StrOutputParser()

def generate(query: str, chunks: List[str]) -> str:
    return _chain.invoke({"query": query, "chunks": "\n\n".join(chunks)})


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
