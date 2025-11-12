print("Ingestion started....")

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Initialize local embedding model (no API token required)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

def data_ingestion():
    """Load and split PDF documents."""
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    return docs


def get_vector_store(docs):
    """Generate embeddings and create FAISS vector store."""
    # Sanity check: Test embedding
    try:
        test_embedding = embedding_model.embed_query("test embedding check")
        print(f"Embedding test successful. Vector length: {len(test_embedding)}")
    except Exception as e:
        raise RuntimeError(f"Embedding model failed: {e}")

    # Build FAISS vector store
    vector_store_faiss = FAISS.from_documents(docs, embedding_model)
    vector_store_faiss.save_local("faiss_index")
    print("FAISS index saved successfully.")
    return vector_store_faiss


if __name__ == "__main__":
    docs = data_ingestion()
    get_vector_store(docs)
    print("Ingestion pipeline completed successfully!")

