from src.github_rag.ingestion.github_client import GitHubClient
from src.github_rag.ingestion.file_filter import FileFilter


def test_file_filtering():
    """Test file fetching and filtering with a public repo."""
    
    client = GitHubClient()
    file_filter = FileFilter()
    
    # Test with a slightly larger repo this time
    test_repo_url = "https://github.com/octocat/Hello-World"
    
    print(f"Testing file filtering with: {test_repo_url}")
    print("-" * 50)
    
    try:
        # Get repository
        repo = client.get_repository(test_repo_url)
        print(f"✅ Repository: {repo.full_name}")
        
        # Fetch all files recursively
        all_files = client.get_all_files(repo)
        print(f"✅ Total files found: {len(all_files)}")
        
        # Apply filters
        filtered_files = [f for f in all_files if file_filter.should_include(f)]
        print(f"✅ Files after filtering: {len(filtered_files)}")
        
        # Show what passed the filter
        print("\n📂 Filtered files:")
        for f in filtered_files:
            print(f"   - {f.path} ({f.size} bytes)")
        
        # Show what was excluded
        excluded_files = [f for f in all_files if not file_filter.should_include(f)]
        print(f"\n🚫 Excluded files:")
        for f in excluded_files:
            print(f"   - {f.path} ({f.type})")
        
        print("-" * 50)
        print("🎉 File filtering test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_file_filtering()