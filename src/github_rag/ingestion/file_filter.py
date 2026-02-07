from typing import List
from github import ContentFile
from github_rag.utils.config import get_filtering_config


class FileFilter:
    """Filters repository files based on config rules."""
    
    def __init__(self):
        """Load filtering rules from config.toml."""
        config = get_filtering_config()
        self.include_extensions = config.get("include_extensions", [])
        self.include_exact_names = config.get("include_exact_names", [])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.max_file_size_mb = config.get("max_file_size_mb", 1)
        self.max_file_size_bytes = self.max_file_size_mb * 1024 * 1024
    
    def is_valid_extension(self, filename: str) -> bool:
        """Check if file has an allowed extension."""
        return any(filename.endswith(ext) for ext in self.include_extensions)
    
    def is_exact_name_match(self, filename: str) -> bool:
        """Check if filename matches any exact name in the include list."""
        return filename in self.include_exact_names
    
    def is_excluded_path(self, file_path: str) -> bool:
        """Check if file path contains any excluded patterns."""
        return any(pattern in file_path for pattern in self.exclude_patterns)
    
    def is_within_size_limit(self, content_file: ContentFile) -> bool:
        """Check if file is within the max size limit."""
        return content_file.size <= self.max_file_size_bytes
    
    def should_include(self, content_file: ContentFile) -> bool:
        """
        Run all filter checks on a file.
        
        Args:
            content_file: GitHub ContentFile object
        
        Returns:
            True if file should be included, False otherwise
        """
        if content_file.type != "file":
            return False
        if not self.is_valid_extension(content_file.name) and not self.is_exact_name_match(content_file.name):
            return False
        if self.is_excluded_path(content_file.path):
            return False
        if not self.is_within_size_limit(content_file):
            return False
        return True