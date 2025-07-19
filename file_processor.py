import os
import mimetypes
import logging
from typing import List, Dict, Any
from pathlib import Path

class FileProcessor:
    """Service for processing and validating files."""
    
    def __init__(self):
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "1048576"))  # 1MB default
        self.supported_extensions = {
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt',
            '.md', '.txt', '.rst', '.tex', '.adoc', '.wiki',
            '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
            '.sql', '.sh', '.bat', '.ps1', '.dockerfile', '.dockerignore',
            '.gitignore', '.gitattributes', '.editorconfig', '.eslintrc', '.prettierrc'
        }
        
        # Binary file extensions to exclude
        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
            '.zip', '.tar', '.gz', '.rar', '.7z',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.class', '.o', '.obj', '.a', '.lib'
        }
    
    def is_text_file(self, file_path: str) -> bool:
        """Check if a file is a text file based on extension and MIME type."""
        path = Path(file_path.lower())
        extension = path.suffix
        
        # Check if extension is explicitly supported
        if extension in self.supported_extensions:
            return True
            
        # Check if extension is explicitly binary
        if extension in self.binary_extensions:
            return False
            
        # Try to guess MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type.startswith('text/')
            
        # Default to True for unknown extensions (let content validation decide)
        return True
    
    def is_valid_text_file(self, content: str, file_size: int) -> bool:
        """Validate if a file content is valid text and within size limits."""
        # Check file size
        if file_size > self.max_file_size:
            logging.info(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            return False
            
        # Check if content is empty
        if not content or not content.strip():
            logging.info("File is empty")
            return False
            
        # Check for binary content (look for null bytes)
        if '\x00' in content:
            logging.info("File contains binary data (null bytes)")
            return False
            
        # Check for reasonable text content (at least some printable characters)
        printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
        if len(content) > 0 and printable_chars / len(content) < 0.8:
            logging.info("File contains too many non-printable characters")
            return False
            
        return True
    
    def filter_text_files(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter a list of files to only include valid text files."""
        valid_files = []
        
        for file_info in files:
            file_path = file_info.get('path', '')
            
            # Skip if not a text file
            if not self.is_text_file(file_path):
                continue
                
            # Skip if file is too large
            file_size = file_info.get('size', 0)
            if file_size > self.max_file_size:
                continue
                
            valid_files.append(file_info)
            
        logging.info(f"Filtered {len(files)} files to {len(valid_files)} valid text files")
        return valid_files
    
    def get_file_extension(self, file_path: str) -> str:
        """Get the file extension from a path."""
        return Path(file_path).suffix.lower()
    
    def get_file_language(self, file_path: str) -> str:
        """Guess the programming language based on file extension."""
        extension = self.get_file_extension(file_path)
        
        language_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++ Header',
            '.hpp': 'C++ Header',
            '.cs': 'C#',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.go': 'Go',
            '.rs': 'Rust',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.rst': 'reStructuredText',
            '.tex': 'LaTeX',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'Configuration',
            '.conf': 'Configuration',
            '.sql': 'SQL',
            '.sh': 'Shell Script',
            '.bat': 'Batch Script',
            '.ps1': 'PowerShell',
            '.dockerfile': 'Dockerfile',
            '.gitignore': 'Git Ignore',
            '.gitattributes': 'Git Attributes',
            '.editorconfig': 'EditorConfig',
            '.eslintrc': 'ESLint Config',
            '.prettierrc': 'Prettier Config'
        }
        
        return language_map.get(extension, 'Unknown') 