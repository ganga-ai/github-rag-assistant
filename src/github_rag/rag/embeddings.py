import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from github_rag.utils.config import get_model_config
from github_rag.utils.usage_tracker import UsageTracker

# Load environment variables
load_dotenv()


class EmbeddingGenerator:
    """Generates embeddings using OpenAI's API."""
    
    def __init__(self):
        """Initialize OpenAI client and load model config."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        # Get embedding model from config
        model_config = get_model_config()
        self.embedding_model = model_config.get("embedding_model", "text-embedding-3-small")
        self.tracker = UsageTracker()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )

        #Track usage
        total_tokens = response.usage.total_tokens
        self.tracker.log_embedding(total_tokens)
        
        # Sort by index to maintain order
        embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        return embeddings