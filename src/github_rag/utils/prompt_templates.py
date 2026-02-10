def get_prompt_template(file_extension: str) -> dict:
    """Get minimal prompt templates based on file type."""
    
    templates = {
        '.ipynb': {
            'system': "You are a concise code assistant. Answer in 2 sentences max.",
            'user': "Notebook section:\n{context}\n\nQuestion: {query}\n\nAnswer briefly (2 sentences):"
        },
        
        '.py': {
            'system': "You are a concise code assistant. Answer in 2 sentences max.",
            'user': "Python code:\n{context}\n\nQuestion: {query}\n\nAnswer briefly (2 sentences):"
        },
        
        '.md': {
            'system': "You are a concise documentation assistant. Answer in 2 sentences max.",
            'user': "Documentation:\n{context}\n\nQuestion: {query}\n\nAnswer briefly (2 sentences):"
        },
        
        'default': {
            'system': "You are a concise code assistant. Answer in 2 sentences max.",
            'user': "Code:\n{context}\n\nQuestion: {query}\n\nAnswer briefly (2 sentences):"
        }
    }
    
    return templates.get(file_extension, templates['default'])