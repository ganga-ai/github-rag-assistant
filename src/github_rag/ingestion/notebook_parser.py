import json
from typing import Optional


def parse_notebook_cells(content: str, max_cells: int = 5) -> Optional[str]:
    """
    Parse Jupyter notebook and extract only first N code cells.
    
    Args:
        content: Raw .ipynb file content (JSON string)
        max_cells: Maximum number of code cells to extract
    
    Returns:
        Extracted code as string, or None if parsing fails
    """
    try:
        notebook = json.loads(content)
        
        if 'cells' not in notebook:
            return None
        
        code_cells = []
        for cell in notebook['cells']:
            if cell.get('cell_type') == 'code':
                # Get source (can be string or list of strings)
                source = cell.get('source', '')
                if isinstance(source, list):
                    source = ''.join(source)
                
                # Skip empty cells
                if source.strip():
                    code_cells.append(source)
                
                # Stop after max_cells
                if len(code_cells) >= max_cells:
                    break
        
        if not code_cells:
            return None
        
        # Join with separators
        return '\n\n# --- Next Cell ---\n\n'.join(code_cells)
    
    except Exception:
        return None