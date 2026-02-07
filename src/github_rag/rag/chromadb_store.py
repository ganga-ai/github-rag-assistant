import chromadb
from chromadb.config import Settings
from typing import List, Dict
from github_rag.utils.config import get_vector_store_config


class ChromaDBStore:
    """Manages ChromaDB vector store for storing and retrieving chunks."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        config = get_vector_store_config()
        self.collection_name = config.get("collection_name", "github_repo_chunks")
        self.persist_directory = config.get("persist_directory", "data/chroma_db")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """Add chunks with embeddings to the vector store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        ids = []
        documents = []
        metadatas = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{chunk['metadata']['file_path']}_chunk_{chunk['metadata']['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk['content'])
            metadata = {k: str(v) for k, v in chunk['metadata'].items()}
            metadatas.append(metadata)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """Search for similar chunks using a query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    
    def clear_collection(self) -> None:
        """Delete all items from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": self.persist_directory
        }