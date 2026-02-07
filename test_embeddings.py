from github_rag.ingestion.github_client import GitHubClient
from github_rag.ingestion.file_filter import FileFilter
from github_rag.ingestion.content_normalizer import ContentNormalizer
from github_rag.ingestion.chunker import Chunker
from github_rag.rag.embeddings import EmbeddingGenerator
from github_rag.rag.vector_store import VectorStore


def test_embeddings_and_storage():
    """Test the complete embedding and storage pipeline."""
    
    print("Testing Embeddings & Vector Storage Pipeline")
    print("=" * 60)
    
    # Initialize all components
    client = GitHubClient()
    file_filter = FileFilter()
    normalizer = ContentNormalizer(client)
    chunker = Chunker()
    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    
    # Clear any existing data
    print("🧹 Clearing existing vector store...")
    vector_store.clear_collection()
    
    # Test with a simple repo
    test_repo_url = "https://github.com/octocat/Hello-World"
    
    try:
        print(f"\n📂 Processing repository: {test_repo_url}")
        
        # Get files and create chunks
        repo = client.get_repository(test_repo_url)
        all_files = client.get_all_files(repo)
        filtered_files = [f for f in all_files if file_filter.should_include(f)]
        
        print(f"✅ Found {len(filtered_files)} files to process")
        
        all_chunks = []
        for file in filtered_files:
            processed = normalizer.process_file(file)
            if processed:
                chunks = chunker.split_by_lines(
                    processed['content'],
                    processed['metadata']
                )
                all_chunks.extend(chunks)
        
        print(f"✅ Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        print(f"\n🔮 Generating embeddings...")
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        embeddings = embedding_gen.generate_embeddings_batch(chunk_texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        
        # Store in vector database
        print(f"\n💾 Storing in ChromaDB...")
        vector_store.add_chunks(all_chunks, embeddings)
        
        # Verify storage
        info = vector_store.get_collection_info()
        print(f"✅ Stored in collection: {info['name']}")
        print(f"   Total chunks in DB: {info['count']}")
        print(f"   Persist directory: {info['persist_directory']}")
        
        # Test retrieval
        print(f"\n🔍 Testing semantic search...")
        test_query = "Hello World"
        query_embedding = embedding_gen.generate_embedding(test_query)
        results = vector_store.search(query_embedding, n_results=3)
        
        print(f"✅ Search results for '{test_query}':")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n   Result {i+1}:")
            print(f"      File: {metadata['file_path']}")
            print(f"      Distance: {distance:.4f}")
            print(f"      Content preview: {doc[:100]}...")
        
        print("\n" + "=" * 60)
        print("🎉 Embeddings and storage test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_embeddings_and_storage()