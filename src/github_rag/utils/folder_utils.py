from typing import List, Dict
from github import ContentFile, Repository


def scan_folder_structure(repo):
    """Scan repository and return folder structure."""
    folders = {}
    
    def scan_directory(path=""):
        contents = repo.get_contents(path)
        contents = contents if isinstance(contents, list) else [contents]
        
        for item in contents:
            if item.type == "dir":
                scan_directory(item.path)
            else:
                if '/' in item.path:
                    folder = '/'.join(item.path.split('/')[:-1])
                else:
                    folder = '.'
                
                extension = item.name.split('.')[-1] if '.' in item.name else 'no-ext'
                
                if folder not in folders:
                    folders[folder] = {'count': 0, 'extensions': set()}
                
                folders[folder]['count'] += 1
                folders[folder]['extensions'].add(f".{extension}")
    
    scan_directory()
    
    for folder in folders:
        folders[folder]['extensions'] = sorted(folders[folder]['extensions'])
    
    return dict(sorted(folders.items()))


def get_files_from_folders(repo, selected_folders):
    """Fetch files from selected folders."""
    files = []
    
    for folder in selected_folders:
        path = "" if folder == "." else folder
        contents = repo.get_contents(path)
        contents = contents if isinstance(contents, list) else [contents]
        
        for item in contents:
            if item.type == "file":
                files.append(item)
            elif item.type == "dir":
                files.extend(get_files_recursively(repo, item.path))
    
    return files


def get_files_recursively(repo, path):
    """Recursively get all files."""
    files = []
    contents = repo.get_contents(path)
    contents = contents if isinstance(contents, list) else [contents]
    
    for item in contents:
        if item.type == "file":
            files.append(item)
        elif item.type == "dir":
            files.extend(get_files_recursively(repo, item.path))
    
    return files