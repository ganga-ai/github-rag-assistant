from github_rag.ingestion.github_client import GitHubClient
from github_rag.ingestion.file_filter import FileFilter
from github_rag.ingestion.content_normalizer import ContentNormalizer
from github_rag.ingestion.chunker import Chunker
from github_rag.rag.embeddings import EmbeddingGenerator
from github_rag.rag.vector_store import VectorStore
from github_rag.rag.rag_engine import RAGEngine


def test_complete_rag_pipeline():
    """Test the complete RAG pipeline from ingestion to question answering."""
    
    print("Testing Complete RAG Pipeline")
    print("=" * 60)
    
    # Initialize components
    client = GitHubClient()
    file_filter = FileFilter()
    normalizer = ContentNormalizer(client)
    chunker = Chunker()
    embedding_gen = EmbeddingGenerator()
    vector_store = VectorStore()
    rag_engine = RAGEngine(embedding_gen=embedding_gen, vector_store=vector_store)
    
    # Test repository
    test_repo_url = "https://github.com/octocat/Hello-World"
    
    try:
        print(f"\n📂 Step 1: Ingesting repository: {test_repo_url}")
        
        # Clear existing data
        vector_store.clear_collection()
        
        # Get files and create chunks
        repo = client.get_repository(test_repo_url)
        all_files = client.get_all_files(repo)
        filtered_files = [f for f in all_files if file_filter.should_include(f)]
        
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
        
        # Generate embeddings and store
        print(f"\n🔮 Step 2: Generating embeddings and storing...")
        chunk_texts = [chunk['content'] for chunk in all_chunks]
        embeddings = embedding_gen.generate_embeddings_batch(chunk_texts)
        vector_store.add_chunks(all_chunks, embeddings)
        
        info = vector_store.get_collection_info()
        print(f"✅ Stored {info['count']} chunks in vector database")
        
        # Test questions
        print(f"\n💬 Step 3: Testing question answering...")
        test_questions = [
            "What is this repository about?",
            "What files are in this repository?",
            "Is there any documentation?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'─' * 60}")
            print(f"Question {i}: {question}")
            print(f"{'─' * 60}")
            
            result = rag_engine.answer_question(question, n_results=3)
            
            print(f"\n📝 Answer:")
            print(result['answer'])
            
            print(f"\n📚 Sources ({len(result['sources'])} chunks):")
            for source in result['sources']:
                print(f"   [{source['source_number']}] {source['file_path']} (lines {source['lines']})")
                print(f"       Relevance: {source['relevance_score']:.3f}")
            
            print(f"\n🤖 Model: {result['model_used']}")
        
        print("\n" + "=" * 60)
        print("🎉 Complete RAG pipeline test passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_complete_rag_pipeline()