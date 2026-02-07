from typing import List, Dict
import tiktoken
from github_rag.utils.config import get_chunking_config


class Chunker:
    """Splits document content into semantic chunks with metadata."""
    
    def __init__(self):
        """Initialize chunker with config from config.toml."""
        config = get_chunking_config()
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.encoder = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using OpenAI's tokenizer."""
        return len(self.encoder.encode(text))
    
    def split_by_lines(self, content: str, metadata: Dict[str, str]) -> List[Dict[str, any]]:
        """
        Split content into chunks by lines, respecting token limits.
        
        Args:
            content: Normalized file content
            metadata: File metadata
        
        Returns:
            List of chunk dictionaries with content and metadata
        """
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)
            
            # If single line exceeds chunk size, split it
            if line_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk),
                        metadata,
                        chunk_index,
                        i - len(current_chunk),
                        i - 1
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split the long line by tokens
                words = line.split()
                word_chunk = []
                word_tokens = 0
                
                for word in words:
                    word_token_count = self.count_tokens(word)
                    if word_tokens + word_token_count > self.chunk_size:
                        if word_chunk:
                            chunks.append(self._create_chunk(
                                ' '.join(word_chunk),
                                metadata,
                                chunk_index,
                                i,
                                i
                            ))
                            chunk_index += 1
                        word_chunk = [word]
                        word_tokens = word_token_count
                    else:
                        word_chunk.append(word)
                        word_tokens += word_token_count
                
                if word_chunk:
                    chunks.append(self._create_chunk(
                        ' '.join(word_chunk),
                        metadata,
                        chunk_index,
                        i,
                        i
                    ))
                    chunk_index += 1
                continue
            
            # Check if adding this line would exceed chunk size
            if current_tokens + line_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        '\n'.join(current_chunk),
                        metadata,
                        chunk_index,
                        i - len(current_chunk),
                        i - 1
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_lines = self._get_overlap_lines(current_chunk)
                current_chunk = overlap_lines + [line]
                current_tokens = self.count_tokens('\n'.join(current_chunk))
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(self._create_chunk(
                '\n'.join(current_chunk),
                metadata,
                chunk_index,
                len(lines) - len(current_chunk),
                len(lines) - 1
            ))
        
        return chunks
    
    def _get_overlap_lines(self, lines: List[str]) -> List[str]:
        """Get lines for overlap based on token count."""
        if not lines:
            return []
        
        overlap_lines = []
        overlap_tokens = 0
        
        # Take lines from the end until we reach overlap size
        for line in reversed(lines):
            line_tokens = self.count_tokens(line)
            if overlap_tokens + line_tokens > self.chunk_overlap:
                break
            overlap_lines.insert(0, line)
            overlap_tokens += line_tokens
        
        return overlap_lines
    
    def _create_chunk(
        self,
        content: str,
        file_metadata: Dict[str, str],
        chunk_index: int,
        start_line: int,
        end_line: int
    ) -> Dict[str, any]:
        """Create a chunk dictionary with content and metadata."""
        return {
            'content': content,
            'metadata': {
                **file_metadata,
                'chunk_index': chunk_index,
                'start_line': start_line,
                'end_line': end_line,
                'token_count': self.count_tokens(content)
            }
        }