try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully switched to pysqlite3-binary.")
except ImportError:
    print("pysqlite3-binary not found, using system sqlite3. THIS MAY CAUSE ISSUES.")
    pass # Fall back to system sqlite3 if pysqlite3-binary is not found, though this defeats the purpose for Chroma

import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import torch # For device check


### Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-V2'
VECTOR_DB_PATH = "./chroma_db_store"
COLLECTION_NAME = "research_papers_v1" # For versioning
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)


def answer_question_with_rag(user_question: str, top_k_chunks: int = 3) -> str:
    print(f"Processing Question: '{user_question}'")

    # Embed the user's question
    print("Embedding user question...")
    question_embedding = embedding_model.encode(user_question).tolist()

    # Query Vector DB for relevant chunks
    print(f"Querying ChromaDB for top {top_k_chunks} relevant chunks...")
    results = collection.query(
        query_embeddings=[question_embedding], 
        n_results= top_k_chunks,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks_texts = []
    if results and results["documents"] and results["documents"][0]:
        retrieved_chunks_texts = results["documents"][0]
        print(f"Retrieved {len(retrieved_chunks_texts)} chunks.")
        for i, chunk_text in enumerate(retrieved_chunks_texts):
            print(f"  Chunk {i+1}: {chunk_text[:150]}...") # Print snippet
    else:
        print("No relevant chunks found in ChromaDB.")
        return "I could not find relevant information in the loaded documents to answer your question."
    
    # Construct the prompt
    context_str = "\n\n".join(retrieved_chunks_texts)

    # TODO: replace prompt template with a JSON file of templates to choose from
    prompt_template = f"""Based on the following context from research papers, please answer the question.
    If the context does not provide enough information to answer the question, please state that you cannot answer based on the provided information.
    Do not make up information outside of the provided context.
    Context:
    {context_str}
    Question: {user_question}
    Answer:"""
    print(f"Constructed Prompt (first 300 chars):\n{prompt_template[:300]}...")

    # Send to Ollama
    print(f"Sending prompt to Ollama model: {OLLAMA_MODEL}")
    ollama_payload = {
        "model": OLLAMA_MODEL, 
        "prompt": prompt_template, 
        "stream": False # Get full response at once
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=ollama_payload, timeout=120)
        response.raise_for_status()
        response_data = response.json()

        llm_answer = response_data.get('response', response_data.get('message', {}).get('content', 'Error: No answer found in LLM response'))
        if 'error' in response_data: # Check for Ollama specific errors
            llm_answer = f"Ollama Error: {response_data['error']}"

        #print(f"\nLLM Answer: {llm_answer}")
        return llm_answer
    except requests.exceptions.RequestException as e:
        error_msg = f"Error communicating with Ollama: {e}"
        print(error_msg)
        return error_msg

if __name__ == "__main__":
    test_question = input("Query: ")
    while test_question.lower() not in ["q", "quit", "exit"]:
        answer = answer_question_with_rag(test_question, top_k_chunks = 3)
        print(answer)
        test_question = input("Query: ")