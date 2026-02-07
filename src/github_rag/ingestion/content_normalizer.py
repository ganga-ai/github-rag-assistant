from typing import Optional, Dict
from github import ContentFile
from github_rag.ingestion.github_client import GitHubClient


class ContentNormalizer:
    """Extracts and normalizes content from repository files."""
    
    def __init__(self, github_client: GitHubClient):
        """
        Initialize the content normalizer.
        
        Args:
            github_client: GitHubClient instance for fetching file content
        """
        self.github_client = github_client
    
    def extract_content(self, content_file: ContentFile) -> Optional[str]:
        """
        Extract and decode content from a file.
        
        Args:
            content_file: GitHub ContentFile object
        
        Returns:
            Decoded content as string, or None if extraction fails
        """
        try:
            content = self.github_client.get_file_content(content_file)
            return content
        except Exception as e:
            print(f"Warning: Failed to extract content from {content_file.path}: {e}")
            return None
    
    def normalize_content(self, content: str, file_path: str) -> str:
        """
        Normalize content by cleaning whitespace and formatting.
        
        Args:
            content: Raw file content
            file_path: Path to the file (for context)
        
        Returns:
            Normalized content
        """
        if not content:
            return ""
        
        # Remove excessive blank lines (more than 2 consecutive)
        lines = content.split('\n')
        normalized_lines = []
        blank_count = 0
        
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    normalized_lines.append(line)
            else:
                blank_count = 0
                normalized_lines.append(line)
        
        normalized = '\n'.join(normalized_lines)
        
        # Remove trailing whitespace from each line
        normalized = '\n'.join(line.rstrip() for line in normalized.split('\n'))
        
        return normalized.strip()
    
    def create_file_metadata(self, content_file: ContentFile) -> Dict[str, str]:
        """
        Create metadata for a file.
        
        Args:
            content_file: GitHub ContentFile object
        
        Returns:
            Dictionary with file metadata
        """
        file_extension = content_file.name.split('.')[-1] if '.' in content_file.name else 'none'
        
        return {
            'file_path': content_file.path,
            'file_name': content_file.name,
            'file_extension': file_extension,
            'file_size': str(content_file.size),
            'file_url': content_file.html_url
        }
    
    def process_file(self, content_file: ContentFile) -> Optional[Dict[str, any]]:
        """
        Complete processing pipeline for a single file.
        
        Args:
            content_file: GitHub ContentFile object
        
        Returns:
            Dictionary with content and metadata, or None if processing fails
        """
        # Extract content
        content = self.extract_content(content_file)
        if content is None:
            return None
        
        # Normalize content
        normalized_content = self.normalize_content(content, content_file.path)
        if not normalized_content:
            return None
        
        # Create metadata
        metadata = self.create_file_metadata(content_file)
        
        return {
            'content': normalized_content,
            'metadata': metadata
        }