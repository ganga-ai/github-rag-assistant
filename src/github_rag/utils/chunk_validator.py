import tiktoken
from typing import List, Dict, Tuple


class ChunkValidator:
    """Validate chunks before embedding to catch issues early."""
    
    def __init__(self, max_chunk_tokens: int = 500):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_chunk_tokens = max_chunk_tokens
    
    def validate_chunks(self, chunks: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Validate chunks and return valid ones + warnings.
        
        Returns:
            (valid_chunks, warnings)
        """
        valid_chunks = []
        warnings = []
        
        for i, chunk in enumerate(chunks):
            content = chunk['content']
            
            # Check if empty
            if not content.strip():
                warnings.append(f"Chunk {i}: Empty content, skipped")
                continue
            
            # Check token count
            tokens = len(self.encoder.encode(content))
            
            if tokens > self.max_chunk_tokens:
                warnings.append(
                    f"Chunk {i}: {tokens} tokens (max: {self.max_chunk_tokens}), "
                    f"from {chunk['metadata']['file_path']}"
                )
                # Don't skip, but warn
            
            valid_chunks.append(chunk)
        
        return valid_chunks, warnings