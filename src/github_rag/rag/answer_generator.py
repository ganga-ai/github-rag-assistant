import os
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
from github_rag.utils.config import get_model_config
from github_rag.utils.usage_tracker import UsageTracker
from github_rag.utils.prompt_templates import get_prompt_template
from github_rag.utils.token_utils import TokenCounter

# Load environment variables
load_dotenv()


class AnswerGenerator:
    """Generates answers using LLM based on retrieved context."""
    
    def __init__(self):
        """Initialize OpenAI client and load model config."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        # Get LLM model from config
        model_config = get_model_config()
        self.llm_model = model_config.get("llm_model", "gpt-4o-mini")
        self.tracker = UsageTracker()
        self.token_counter = TokenCounter(model=self.llm_model, max_tokens=1500)
    
    def generate_answer(self, query: str, context: str, retrieved_chunks: List[Dict]) -> Dict:
        """Generate answer using LLM with file-type-specific prompts."""
        
        # Determine predominant file type from chunks
        file_extensions = [chunk['metadata'].get('file_extension', '') for chunk in retrieved_chunks]
        most_common_ext = max(set(file_extensions), key=file_extensions.count) if file_extensions else ''
        ext = f".{most_common_ext}" if most_common_ext and most_common_ext != 'none' else 'default'
        
        # Get appropriate template
        template = get_prompt_template(ext)
        system_prompt = template['system']
        user_prompt_template = template['user']

        # Truncate chunks to fit token budget
        truncated_chunks = self.token_counter.truncate_chunks(
            retrieved_chunks, 
            system_prompt, 
            query
        )

        if not truncated_chunks:
            return {
                'answer': "❌ Question too long - no tokens left for context. Try shorter question.",
                'sources': [],
                'model_used': self.llm_model
            }

        # Format context from truncated chunks
        from github_rag.rag.query_processor import QueryProcessor
        temp_processor = QueryProcessor(None, None)
        context = temp_processor.format_context_for_llm(truncated_chunks)

        # Format user prompt
        user_prompt = user_prompt_template.format(context=context, query=query)

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Track usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        self.tracker.log_llm_call(input_tokens, output_tokens)
        
        answer_text = response.choices[0].message.content
        
        # Format sources
        sources = []
        for i, chunk in enumerate(retrieved_chunks):
            sources.append({
                'source_number': i + 1,
                'file_path': chunk['metadata']['file_path'],
                'lines': f"{chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}",
                'relevance_score': chunk['relevance_score'],
                'file_url': chunk['metadata'].get('file_url', '')
            })
        
        return {
            'answer': answer_text,
            'sources': sources,
            'model_used': self.llm_model,
            'prompt_type': ext  # Track which prompt was used
        }