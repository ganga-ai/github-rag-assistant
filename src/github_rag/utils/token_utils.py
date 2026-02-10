import tiktoken
from typing import List, Dict


class TokenCounter:
    """Count and manage tokens to stay within limits."""
    
    def __init__(self, model="gpt-4o-mini", max_tokens=100):  # Set to 100 for testing
        self.encoder = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoder.encode(text))
    
    def truncate_chunks(self, chunks: List[Dict], system_prompt: str, user_query: str) -> List[Dict]:
        """Truncate chunks to fit within token limit."""
        
        # Count base tokens
        base_tokens = self.count_tokens(system_prompt) + self.count_tokens(user_query)
        base_tokens += 50  # Reserve for answer + formatting
        
        available_tokens = self.max_tokens - base_tokens
        
        if available_tokens < 20:
            return []  # Not enough room
        
        # Add chunks until limit
        truncated = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk['content'])
            
            if current_tokens + chunk_tokens <= available_tokens:
                truncated.append(chunk)
                current_tokens += chunk_tokens
            else:
                # Try to fit partial chunk
                remaining = available_tokens - current_tokens
                if remaining > 10:
                    # Truncate this chunk's content
                    words = chunk['content'].split()
                    truncated_content = ""
                    for word in words:
                        test = truncated_content + " " + word
                        if self.count_tokens(test) < remaining:
                            truncated_content = test
                        else:
                            break
                    
                    if truncated_content:
                        chunk_copy = chunk.copy()
                        chunk_copy['content'] = truncated_content + "..."
                        truncated.append(chunk_copy)
                break
        
        return truncated