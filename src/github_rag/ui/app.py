import streamlit as st
from github_rag.ingestion.github_client import GitHubClient
from github_rag.ingestion.file_filter import FileFilter
from github_rag.ingestion.content_normalizer import ContentNormalizer
from github_rag.ingestion.chunker import Chunker
from github_rag.rag.embeddings import EmbeddingGenerator
from github_rag.utils.chunk_validator import ChunkValidator

st.set_page_config(page_title="GitHub RAG Assistant", page_icon="🤖")

st.title("🤖 GitHub RAG Assistant")
st.write("Ask questions about any public GitHub repository!")

# Debug shortcut
if st.button("🐛 Debug: Skip to Q&A (if data exists in Pinecone)"):
    st.session_state.ingestion_complete = True
    st.rerun()

# Sidebar with usage stats
with st.sidebar:
    st.header("📊 Usage Stats")
    
    try:
        from github_rag.utils.usage_tracker import UsageTracker
        tracker = UsageTracker()
        stats = tracker.get_session_stats()
        total_cost = tracker.get_total_cost()
        
        st.metric("Today's Tokens", f"{stats['total_tokens']:,}")
        st.metric("Today's Cost", f"${stats['total_cost']:.4f}")
        st.metric("Total Cost (All Time)", f"${total_cost:.4f}")
        
        with st.expander("Details"):
            st.write(f"Embedding calls: {stats['embedding_calls']}")
            st.write(f"LLM calls: {stats['llm_calls']}")
    except Exception as e:
        st.caption("Usage tracking unavailable")

st.markdown("---")

# Initialize components
@st.cache_resource
def get_github_client():
    return GitHubClient()

@st.cache_resource
def get_file_filter():
    return FileFilter()

@st.cache_resource
def get_content_normalizer():
    return ContentNormalizer(get_github_client())

@st.cache_resource
def get_chunker():
    return Chunker()

@st.cache_resource
def get_embedding_generator():
    return EmbeddingGenerator()

def get_vector_store():
    if 'vector_store' not in st.session_state:
        from github_rag.rag.vector_store import get_vector_store as create_vector_store
        st.session_state.vector_store = create_vector_store()
    return st.session_state.vector_store

client = get_github_client()
file_filter = get_file_filter()
normalizer = get_content_normalizer()
chunker = get_chunker()
embedding_gen = get_embedding_generator()
vector_store = get_vector_store()

# Repository input section
st.subheader("📂 Step 1: Enter Repository")

repo_url = st.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/owner/repository",
    help="Enter the URL of a public GitHub repository"
)

if st.button("🔍 Validate Repository", type="primary"):
    if not repo_url:
        st.error("Please enter a repository URL")
    else:
        with st.spinner("Validating repository..."):
            try:
                owner, repo_name = client.parse_repo_url(repo_url)
                repo = client.get_repository(repo_url)
                
                st.success(f"✅ Repository found: {repo.full_name}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stars", repo.stargazers_count)
                with col2:
                    st.metric("Forks", repo.forks_count)
                with col3:
                    st.metric("Language", repo.language or "N/A")
                
                st.info(f"📝 Description: {repo.description or 'No description'}")
                
                st.session_state.repo = repo
                st.session_state.repo_url = repo_url
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")

# File scanning section    
   # File scanning section - Step 2a: Scan Folders
if "repo" in st.session_state:
    st.subheader("📁 Step 2a: Scan Folder Structure")
    
    if st.button("🔎 Scan Folders", type="primary"):
        with st.spinner("Scanning folder structure..."):
            try:
                from github_rag.utils.folder_utils import scan_folder_structure
                
                repo = st.session_state.repo
                folder_structure = scan_folder_structure(repo)
                
                st.session_state.folder_structure = folder_structure
                
                st.success(f"✅ Found {len(folder_structure)} folders")
                
                # Display folders
                st.subheader("📂 Available Folders")
                folder_data = []
                for folder, info in folder_structure.items():
                    folder_data.append({
                        "Folder": folder,
                        "Files": info['count'],
                        "Extensions": ", ".join(info['extensions'][:5])  # Show first 5
                    })
                
                st.dataframe(folder_data, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")

# Step 2b: Select Folders and Fetch Files
if "folder_structure" in st.session_state:
    st.subheader("📁 Step 2b: Select Folders to Process")
    
    folder_structure = st.session_state.folder_structure
    
    st.info(f"💡 Select specific folders to reduce costs. Unselected folders will be skipped.")
    
    # NEW: Extension filter
    st.subheader("🔧 Filter by File Extensions")
    
    # Collect all unique extensions
    all_extensions = set()
    for folder, info in folder_structure.items():
        all_extensions.update(info['extensions'])
    all_extensions = sorted(all_extensions)
    
    selected_extensions = st.multiselect(
        "Select file extensions to include (leave empty for all):",
        options=all_extensions,
        default=all_extensions,  # All selected by default
        help="Only files with these extensions will be processed"
    )
    
    st.session_state.selected_extensions = selected_extensions if selected_extensions else all_extensions
    
    st.markdown("---")
    st.subheader("📂 Select Folders")
    
    # Display folders with checkboxes
    selected_folders = []
    
    col1, col2 = st.columns(2)
    folder_items = list(folder_structure.items())
    mid = len(folder_items) // 2
    
    with col1:
        for folder, info in folder_items[:mid]:
            label = f"{folder} ({info['count']} files, {', '.join(info['extensions'][:3])})"
            if st.checkbox(label, key=f"folder_{folder}"):
                selected_folders.append(folder)
    
    with col2:
        for folder, info in folder_items[mid:]:
            label = f"{folder} ({info['count']} files, {', '.join(info['extensions'][:3])})"
            if st.checkbox(label, key=f"folder_{folder}"):
                selected_folders.append(folder)
    
    # Fetch files from selected folders
    if st.button("📥 Fetch Files from Selected Folders", type="primary", disabled=len(selected_folders)==0):
        with st.spinner(f"Fetching files from {len(selected_folders)} folders..."):
            try:
                from github_rag.utils.folder_utils import get_files_from_folders
                
                repo = st.session_state.repo
                all_files = get_files_from_folders(repo, selected_folders)
                
                # Apply file filter
                filtered_files = [f for f in all_files  
                    if any(f.name.endswith(ext) for ext in st.session_state.selected_extensions) and
                    file_filter.is_within_size_limit(f) and
                    not file_filter.is_excluded_path(f.path)]
                excluded_files = [f for f in all_files if f not in filtered_files]
                
                st.session_state.filtered_files = filtered_files
                st.session_state.excluded_files = excluded_files
                st.session_state.selected_folders = selected_folders
                
                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("✅ Files to Process", len(filtered_files))
                with col2:
                    st.metric("🚫 Files Excluded", len(excluded_files))
                
                if filtered_files:
                    st.subheader("✅ Files to Process")
                    file_data = [
                        {"File Path": f.path, "Size (bytes)": f.size}
                        for f in filtered_files
                    ]
                    st.dataframe(file_data, use_container_width=True)
                
                if excluded_files:
                    with st.expander("🚫 Excluded Files"):
                        excluded_data = [
                            {
                                "File Path": f.path,
                                "Reason": "Binary/Unsupported" if f.type != "file" 
                                         else "Unsupported extension"
                            }
                            for f in excluded_files
                        ]
                        st.dataframe(excluded_data, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")             

# Chunking section
if "filtered_files" in st.session_state:
    st.subheader("✂️ Step 3: Extract and Chunk Content")
    
    if st.button("⚙️ Process Files", type="primary"):
        with st.spinner("Processing files and creating chunks..."):
            try:
                filtered_files = st.session_state.filtered_files
                all_chunks = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(filtered_files):
                    status_text.text(f"Processing: {file.path}")
                    
                    processed = normalizer.process_file(file)
                    if processed:
                        content = processed['content']
                        metadata = processed['metadata']
                        chunks = chunker.split_by_lines(content, metadata)
                        all_chunks.extend(chunks)
                    
                    progress_bar.progress((idx + 1) / len(filtered_files))
                
                status_text.text("✅ Processing complete!")
                
                st.session_state.chunks = all_chunks
                
                st.success(f"🎉 Created {len(all_chunks)} chunks from {len(filtered_files)} files!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", len(all_chunks))
                with col2:
                    avg_tokens = sum(c['metadata']['token_count'] for c in all_chunks) / len(all_chunks)
                    st.metric("Avg Tokens/Chunk", f"{avg_tokens:.0f}")
                with col3:
                    total_tokens = sum(c['metadata']['token_count'] for c in all_chunks)
                    st.metric("Total Tokens", total_tokens)
                
                with st.expander("📋 View Sample Chunks"):
                    for i, chunk in enumerate(all_chunks[:3]):
                        st.markdown(f"**Chunk {i+1}** from `{chunk['metadata']['file_path']}`")
                        st.code(chunk['content'][:200] + "...", language="text")
                        st.caption(f"Tokens: {chunk['metadata']['token_count']} | Lines: {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

# Embedding and storage section
if "chunks" in st.session_state:
    st.subheader("🔮 Step 4: Embed and Store in Vector Database")
    
    # Show current vector store status
    info = vector_store.get_collection_info()
    st.info(f"📊 Current vector store has {info['count']} chunks stored")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Embed & Store Chunks", type="primary"):
            with st.spinner("Generating embeddings and storing..."):
                try:
                    chunks = st.session_state.chunks

                    # Validate chunks before embedding
                    validator = ChunkValidator(max_chunk_tokens=500)
                    valid_chunks, warnings = validator.validate_chunks(chunks)
                    
                    if warnings:
                        with st.expander("⚠️ Validation Warnings", expanded=True):
                            for warning in warnings:
                                st.warning(warning)
                    
                    if not valid_chunks:
                        st.error("❌ No valid chunks to embed!")
                        st.stop()
                    
                    st.info(f"✅ Validated {len(valid_chunks)} chunks (from {len(chunks)} total)")
                                        
                    # Clear existing data
                    status_text = st.empty()
                    status_text.text("🧹 Clearing previous data...")
                    vector_store.clear_collection()
                    
                    # Generate embeddings
                    status_text.text("🔮 Generating embeddings...")
                    chunk_texts = [chunk['content'] for chunk in valid_chunks]
                    
                    # Batch process (OpenAI allows up to 2048 inputs per request)
                    batch_size = 100
                    all_embeddings = []
                    
                    progress_bar = st.progress(0)
                    for i in range(0, len(chunk_texts), batch_size):
                        batch = chunk_texts[i:i + batch_size]
                        embeddings = embedding_gen.generate_embeddings_batch(batch)
                        all_embeddings.extend(embeddings)
                        progress_bar.progress(min((i + batch_size) / len(chunk_texts), 1.0))
                    
                    status_text.text("💾 Storing in ChromaDB...")
                    vector_store.add_chunks(chunks, all_embeddings)
                    
                    # Verify storage
                    info = vector_store.get_collection_info()
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Successfully stored {info['count']} chunks in vector database!")
                    st.session_state.ingestion_complete = True
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunks Stored", info['count'])
                    with col2:
                        st.metric("Embedding Dimension", len(all_embeddings[0]))
                    with col3:
                        st.metric("Collection", info['name'])
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        if st.button("🗑️ Clear Vector Store"):
            vector_store.clear_collection()
            if 'ingestion_complete' in st.session_state:
                del st.session_state.ingestion_complete
            st.success("✅ Vector store cleared!")
            st.rerun()

    st.markdown("---")

# Question answering section
if "ingestion_complete" in st.session_state and st.session_state.ingestion_complete:
    st.subheader("💬 Step 5: Ask Questions About the Repository")
    st.success("✅ Repository is ready for questions!")
    
    # Initialize RAG engine (reuse existing instances)
    from github_rag.rag.rag_engine import RAGEngine
    
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine(
            embedding_gen=embedding_gen,
            vector_store=vector_store
        )
    
    rag_engine = st.session_state.rag_engine
    
    # Question input
    question = st.text_input(
        "Ask a question about the repository:",
        placeholder="e.g., What does this repository do? How is X implemented?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        n_results = st.number_input("Sources", min_value=1, max_value=10, value=5, help="Number of relevant code chunks to retrieve")
    
    if st.button("🔍 Get Answer", type="primary"):
        if not question:
            st.warning("Please enter a question")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get answer from RAG engine
                    result = rag_engine.answer_question(question, n_results=n_results)
                    
                    # Display answer
                    st.markdown("### 📝 Answer")
                    st.markdown(result['answer'])
                    
                    # Display sources
                    st.markdown("---")
                    st.markdown(f"### 📚 Sources ({result['n_chunks_retrieved']} relevant chunks)")
                    
                    for source in result['sources']:
                        with st.expander(
                            f"[{source['source_number']}] {source['file_path']} (lines {source['lines']}) - "
                            f"Relevance: {source['relevance_score']:.2%}"
                        ):
                            # Find the actual chunk content to display
                            if 'chunks' not in st.session_state:
                                st.warning("⚠️ Full source preview unavailable (data from previous session)")
                            else:
                                chunks = st.session_state.chunks
                                matching_chunk = None
                                for chunk in chunks:
                                    if (chunk['metadata']['file_path'] == source['file_path'] and
                                        str(chunk['metadata']['start_line']) == source['lines'].split('-')[0]):
                                        matching_chunk = chunk
                                        break
                                
                                if matching_chunk:
                                    st.code(matching_chunk['content'], language=source['file_path'].split('.')[-1])
                                
                                if source['file_url']:
                                    st.markdown(f"[View file on GitHub]({source['file_url']})")
                    
                    # Display metadata
                    st.caption(f"🤖 Model: {result['model_used']}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Chat history (optional enhancement)
    st.markdown("---")
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What is this repository about?",
        "What are the main files in this repository?",
        "How is the code structured?",
        "What dependencies does this project use?"
    ]
    
    cols = st.columns(2)
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % 2]:
            if st.button(suggestion, key=f"suggestion_{idx}"):
                st.session_state.question_input = suggestion
                st.rerun()