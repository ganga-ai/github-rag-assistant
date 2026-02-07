from github_rag.utils.config import get_vector_store_config

def get_vector_store():
    """Factory function to get the appropriate vector store based on config."""
    config = get_vector_store_config()
    store_type = config.get("type", "chromadb").lower()
    
    if store_type == "pinecone":
        from github_rag.rag.pinecone_store import PineconeStore
        return PineconeStore()
    elif store_type == "chromadb":
        from github_rag.rag.chromadb_store import ChromaDBStore
        return ChromaDBStore()
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")