import os
from typing import List, Dict, Optional
from github import Github, Repository, ContentFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GitHubClient:
    """Client for interacting with GitHub API."""
    
    def __init__(self):
        """Initialize GitHub client with optional token."""
        github_token = os.getenv("GITHUB_TOKEN")
        self.client = Github(github_token) if github_token else Github()
    
    def parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        """
        Parse GitHub repository URL to extract owner and repo name.
        
        Args:
            repo_url: GitHub repository URL (e.g., 'https://github.com/owner/repo')
        
        Returns:
            Tuple of (owner, repo_name)
        """
        # Remove trailing slashes and .git
        repo_url = repo_url.rstrip('/').replace('.git', '')
        
        # Extract owner and repo from URL
        parts = repo_url.split('github.com/')[-1].split('/')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        owner = parts[0]
        repo_name = parts[1]
        
        return owner, repo_name
    
    def get_repository(self, repo_url: str) -> Repository:
        """
        Get repository object from URL.
        
        Args:
            repo_url: GitHub repository URL
        
        Returns:
            Repository object
        """
        owner, repo_name = self.parse_repo_url(repo_url)
        repo = self.client.get_repo(f"{owner}/{repo_name}")
        return repo
    
    def get_repo_contents(self, repo: Repository, path: str = "") -> List[ContentFile]:
        """
        Get contents of a repository at a given path.
        
        Args:
            repo: Repository object
            path: Path within the repository (default: root)
        
        Returns:
            List of ContentFile objects
        """
        contents = repo.get_contents(path)
        return contents if isinstance(contents, list) else [contents]
    
    def get_file_content(self, content_file: ContentFile) -> Optional[str]:
        """
        Get decoded content of a file.
        
        Args:
            content_file: ContentFile object
        
        Returns:
            Decoded file content as string, or None if binary
        """
        try:
            return content_file.decoded_content.decode('utf-8')
        except Exception:
            return None
        
    def get_all_files(self, repo: Repository) -> List[ContentFile]:
        """
        Recursively fetch all files from a repository.
        
        Args:
            repo: Repository object
        
        Returns:
            List of all ContentFile objects in the repo
        """
        all_files = []
        contents = self.get_repo_contents(repo)
        
        while contents:
            current = contents.pop(0)
            if current.type == "dir":
                # If it's a directory, fetch its contents and add to the list
                contents.extend(self.get_repo_contents(repo, current.path))
            else:
                # If it's a file, add to our results
                all_files.append(current)
        
        return all_files