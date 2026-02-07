import os
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from github_rag.utils.config import get_vector_store_config

load_dotenv()


class PineconeStore:
    """Manages Pinecone vector store for storing and retrieving chunks."""
    
    def __init__(self):
        """Initialize Pinecone client and index."""
        config = get_vector_store_config()
        
        # Get API key
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=api_key)
        
        # Get index name and host
        self.index_name = config.get("pinecone_index_name", "github-rag-assistant")
        self.host = config.get("pinecone_host")
        
        # Connect to index
        self.index = self.pc.Index(name=self.index_name, host=self.host)
    
    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        """Add chunks with embeddings to Pinecone."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare vectors for upsert
        vectors = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{chunk['metadata']['file_path']}_chunk_{chunk['metadata']['chunk_index']}"
            
            # Pinecone metadata (all values must be strings, numbers, or booleans)
            metadata = {
                'file_path': chunk['metadata']['file_path'],
                'file_name': chunk['metadata']['file_name'],
                'file_extension': chunk['metadata']['file_extension'],
                'chunk_index': int(chunk['metadata']['chunk_index']),
                'start_line': int(chunk['metadata']['start_line']),
                'end_line': int(chunk['metadata']['end_line']),
                'token_count': int(chunk['metadata']['token_count']),
                'content': chunk['content']  # Store content in metadata for retrieval
            }
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> Dict:
        """Search for similar chunks using query embedding."""
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )
        
        # Format results to match ChromaDB structure
        ids = [[match['id'] for match in results['matches']]]
        documents = [[match['metadata']['content'] for match in results['matches']]]
        metadatas = [[{k: str(v) for k, v in match['metadata'].items() if k != 'content'} 
                      for match in results['matches']]]
        distances = [[match['score'] for match in results['matches']]]
        
        return {
            'ids': ids,
            'documents': documents,
            'metadatas': metadatas,
            'distances': distances
        }
    
    def clear_collection(self) -> None:
        """Delete all vectors from the index."""
        # Delete all vectors (Pinecone doesn't have a "clear all" - need to delete by namespace or recreate)
        try:
            self.index.delete(delete_all=True)
        except Exception:
            pass
    
    def get_collection_info(self) -> Dict:
        """Get information about the index."""
        stats = self.index.describe_index_stats()
        
        return {
            'name': self.index_name,
            'count': stats['total_vector_count'],
            'persist_directory': 'pinecone-cloud'
        }