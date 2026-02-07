from github_rag.ingestion.github_client import GitHubClient
from github_rag.ingestion.file_filter import FileFilter
from github_rag.ingestion.content_normalizer import ContentNormalizer
from github_rag.ingestion.chunker import Chunker


def test_chunking_pipeline():
    """Test the complete content extraction and chunking pipeline."""
    
    client = GitHubClient()
    file_filter = FileFilter()
    normalizer = ContentNormalizer(client)
    chunker = Chunker()
    
    # Test with a simple repo
    test_repo_url = "https://github.com/octocat/Hello-World"
    
    print(f"Testing chunking pipeline with: {test_repo_url}")
    print("=" * 60)
    
    try:
        # Get repository and files
        repo = client.get_repository(test_repo_url)
        all_files = client.get_all_files(repo)
        filtered_files = [f for f in all_files if file_filter.should_include(f)]
        
        print(f"✅ Found {len(filtered_files)} files to process\n")
        
        all_chunks = []
        
        # Process each file
        for file in filtered_files:
            print(f"📄 Processing: {file.path}")
            
            # Extract and normalize content
            processed = normalizer.process_file(file)
            if not processed:
                print(f"   ⚠️ Skipped (no content)\n")
                continue
            
            content = processed['content']
            metadata = processed['metadata']
            
            print(f"   Content length: {len(content)} characters")
            
            # Create chunks
            chunks = chunker.split_by_lines(content, metadata)
            all_chunks.extend(chunks)
            
            print(f"   Created {len(chunks)} chunks")
            
            # Show first chunk details
            if chunks:
                first_chunk = chunks[0]
                print(f"   First chunk preview:")
                print(f"      Tokens: {first_chunk['metadata']['token_count']}")
                print(f"      Lines: {first_chunk['metadata']['start_line']}-{first_chunk['metadata']['end_line']}")
                print(f"      Content: {first_chunk['content'][:100]}...")
            
            print()
        
        print("=" * 60)
        print(f"🎉 Pipeline complete!")
        print(f"   Total chunks created: {len(all_chunks)}")
        
        # Show chunk statistics
        if all_chunks:
            avg_tokens = sum(c['metadata']['token_count'] for c in all_chunks) / len(all_chunks)
            print(f"   Average tokens per chunk: {avg_tokens:.0f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_chunking_pipeline()