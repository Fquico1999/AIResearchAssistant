try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully switched to pysqlite3-binary.")
except ImportError:
    print("pysqlite3-binary not found, using system sqlite3. THIS MAY CAUSE ISSUES.")
    pass # Fall back to system sqlite3 if pysqlite3-binary is not found, though this defeats the purpose for Chroma


import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import torch # For device check

### Configuration
PDF_DIR = "./data/papers"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-V2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
VECTOR_DB_PATH = "./chroma_db_store"
COLLECTION_NAME = "research_papers_v1" # For versioning

### Helper Functions
def extract_text_from_pdf(pdf_path):

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text only
            page_text = page.extract_text()
            if page_text:
                # Newline between pages
                text+= page_text + "\n"
    return text

def chunk_text(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

### Ingestion Logic
if __name__ == "__main__":
    # Initialize Embedding model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for embedding model.")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded.")

    # Initialize Chroma DB
    chroma_client = chromadb.PersistentClient(path = VECTOR_DB_PATH)
    
    try:
        collection = chroma_client.get_or_create_collection(name = COLLECTION_NAME)
        print(f"Using/Created ChromaDB collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"Error with ChromaDB collection: {e}")
        exit()
    
    # Process PDFs
    num_processed_files = 0
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_filepath = os.path.join(PDF_DIR, filename)
            print(f"Processing: {filename}...")


            try:
                # Extract Test
                pdf_text = extract_text_from_pdf(pdf_filepath)
                if not pdf_text.strip():
                    print(f"No text extracted from {filename}. Skipping.")
                    continue
                
                # Chunk Text
                text_chunks = chunk_text(pdf_text, CHUNK_SIZE, CHUNK_OVERLAP)
                if not text_chunks:
                    print(f"No chunks created for {filename}. Skipping.")
                    continue
                
                print(f"Extracted text, created {len(text_chunks)} chunks.")

                # Embed and store chunks
                embeddings_to_add = []
                documents_to_add = []
                metadatas_to_add = []
                ids_to_add = []

                for i, chunk in enumerate(text_chunks):
                    # Simple duplicate checking by ID. Should be replaced by more robust option
                    # As it does not check for content match, just filename and chunk id
                    chunk_id = f"{filename}_chunk_{i}"

                    existing_chunk = collection.get(ids = [chunk_id])
                    if existing_chunk and existing_chunk['ids']:
                        print(f"Chunk {chunk_id} already exists. Skipping.")
                        continue

                    embedding = embedding_model.encode(chunk).tolist()

                    embeddings_to_add.append(embedding)
                    documents_to_add.append(chunk)
                    metadatas_to_add.append({"source_pdf": filename, "chunk_index": i})
                    ids_to_add.append(chunk_id)
                
                if ids_to_add:
                    collection.add(
                        embeddings = embeddings_to_add, 
                        documents = documents_to_add, 
                        metadatas = metadatas_to_add, 
                        ids = ids_to_add
                    )
                    print(f"Added {len(ids_to_add)} new chunks to ChromaDB for {filename}.")
                
                num_processed_files+=1

            except Exception as e:
                print(f"Error processiong {filename}: {e}")

    print(f"Ingestion Complete: Processed {num_processed_files} PDF files.")
    print(f"Total documents in ChromaDB collection `{COLLECTION_NAME}`: {collection.count()}")
