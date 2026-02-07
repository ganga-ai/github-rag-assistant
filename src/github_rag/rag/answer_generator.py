import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from github_rag.utils.config import get_model_config

# Load environment variables
load_dotenv()


class AnswerGenerator:
    """Generates answers using LLM based on retrieved context."""
    
    def __init__(self):
        """Initialize OpenAI client and load model config."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        # Get LLM model from config
        model_config = get_model_config()
        self.llm_model = model_config.get("llm_model", "gpt-4o-mini")
    
    def generate_answer(self, query: str, context: str, retrieved_chunks: List[Dict]) -> Dict:
        """
        Generate answer using LLM based on query and retrieved context.
        
        Args:
            query: User's question
            context: Formatted context from retrieved chunks
            retrieved_chunks: List of retrieved chunk dictionaries
        
        Returns:
            Dictionary with answer and sources
        """
        # Create system prompt
        system_prompt = """You are a helpful assistant that answers questions about GitHub repositories based on the provided code and documentation.

Your task:
1. Carefully read the provided context from the repository
2. Answer the user's question based ONLY on the information in the context
3. If the context doesn't contain enough information to answer, say so clearly
4. Be specific and cite which files or sections you're referring to
5. Use markdown formatting for code snippets when appropriate

Important: Do not make assumptions or add information that isn't in the provided context."""

        # Create user prompt with context
        user_prompt = f"""Based on the following code and documentation from the repository:

{context}

---

Question: {query}

Please provide a clear, accurate answer based on the context above."""

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=1000
        )
        
        answer_text = response.choices[0].message.content
        
        # Format sources
        sources = []
        for i, chunk in enumerate(retrieved_chunks):
            sources.append({
                'source_number': i + 1,
                'file_path': chunk['metadata']['file_path'],
                'lines': f"{chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}",
                'relevance_score': chunk['relevance_score'],
                'file_url': chunk['metadata'].get('file_url', '')
            })
        
        return {
            'answer': answer_text,
            'sources': sources,
            'model_used': self.llm_model
        }