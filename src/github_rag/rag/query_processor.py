from typing import List, Dict
from github_rag.rag.embeddings import EmbeddingGenerator


class QueryProcessor:
    """Processes user queries and retrieves relevant chunks."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store):
        """
        Initialize query processor.
        
        Args:
            embedding_generator: Instance for generating query embeddings
            vector_store: Instance for searching chunks
        """
        self.embedding_gen = embedding_generator
        self.vector_store = vector_store
    
    def process_query(self, query: str, n_results: int = 5) -> Dict:
        """
        Process a user query and retrieve relevant chunks.
        
        Args:
            query: User's question
            n_results: Number of relevant chunks to retrieve
        
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        # Generate embedding for the query
        query_embedding = self.embedding_gen.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, n_results=n_results)
        
        # Format results
        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            chunk = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
            }
            retrieved_chunks.append(chunk)
        
        return {
            'query': query,
            'chunks': retrieved_chunks,
            'n_results': len(retrieved_chunks)
        }
    
    def format_context_for_llm(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks):
            file_path = chunk['metadata']['file_path']
            lines = f"{chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}"
            content = chunk['content']
            
            context_parts.append(
                f"[Source {i+1}: {file_path} (lines {lines})]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)