from typing import Dict, Optional
from github_rag.rag.embeddings import EmbeddingGenerator
from github_rag.rag.vector_store import get_vector_store
from github_rag.rag.query_processor import QueryProcessor
from github_rag.rag.answer_generator import AnswerGenerator


class RAGEngine:
    """Main RAG engine that orchestrates the question-answering pipeline."""
    
    def __init__(
        self,
        embedding_gen = None,
        vector_store = None
    ):
        """
        Initialize all RAG components.
        
        Args:
            embedding_gen: Optional existing EmbeddingGenerator instance
            vector_store: Optional existing VectorStore instance
        """
        self.embedding_gen = embedding_gen or EmbeddingGenerator()
        self.vector_store = vector_store or get_vector_store()
        self.query_processor = QueryProcessor(self.embedding_gen, self.vector_store)
        self.answer_generator = AnswerGenerator()
    
    def answer_question(self, query: str, n_results: int = 5) -> Dict:
        """
        Complete RAG pipeline: retrieve relevant chunks and generate answer.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Step 1: Process query and retrieve relevant chunks
        retrieval_results = self.query_processor.process_query(query, n_results)
        retrieved_chunks = retrieval_results['chunks']
        
        if not retrieved_chunks:
            return {
                'query': query,
                'answer': "I couldn't find any relevant information in the repository to answer this question.",
                'sources': [],
                'n_chunks_retrieved': 0
            }
        
        # Step 2: Format context for LLM
        context = self.query_processor.format_context_for_llm(retrieved_chunks)
        
        # Step 3: Generate answer using LLM
        answer_result = self.answer_generator.generate_answer(
            query,
            context,
            retrieved_chunks
        )
        
        # Step 4: Return complete result
        return {
            'query': query,
            'answer': answer_result['answer'],
            'sources': answer_result['sources'],
            'model_used': answer_result['model_used'],
            'n_chunks_retrieved': len(retrieved_chunks)
        }
    
    def get_vector_store_status(self) -> Dict:
        """Get current status of the vector store."""
        return self.vector_store.get_collection_info()