import re
import uuid
import logging
from typing import List, Dict, Any
from collections import Counter
import math

class ChunkingStrategies:
    """Service for creating different types of chunks for dense and sparse indexing."""
    
    def __init__(self):
        self.min_chunk_size = 50  # Minimum characters for a chunk
        self.max_chunk_size = 2000  # Maximum characters for a chunk
        self.overlap_size = 100  # Overlap between chunks
        
    def create_dense_chunks(self, content: str, file_info: Dict[str, Any], repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create dense chunks using paragraph and sentence-based chunking."""
        try:
            chunks = []
            
            # Strategy 1: Paragraph-based chunking
            paragraph_chunks = self._chunk_by_paragraphs(content, file_info, repo_info)
            chunks.extend(paragraph_chunks)
            
            # Strategy 2: Sentence-based chunking for smaller paragraphs
            sentence_chunks = self._chunk_by_sentences(content, file_info, repo_info)
            chunks.extend(sentence_chunks)
            
            # Remove duplicates and filter by size
            unique_chunks = self._deduplicate_chunks(chunks)
            filtered_chunks = [chunk for chunk in unique_chunks if self._is_valid_chunk_size(chunk['text'])]
            
            # If no chunks created, create a single chunk for the entire content
            if not filtered_chunks and len(content.strip()) >= self.min_chunk_size:
                chunk_id = str(uuid.uuid4())
                filtered_chunks.append({
                    'chunk_id': chunk_id,
                    'text': content.strip(),
                    'chunk_type': 'full_content',
                    'chunk_index': 0,
                    'metadata': self._create_metadata(file_info, repo_info, 'dense')
                })
            
            logging.info(f"Created {len(filtered_chunks)} dense chunks for {file_info['path']}")
            return filtered_chunks
            
        except Exception as e:
            logging.error(f"Error creating dense chunks for {file_info['path']}: {e}")
            return []
    
    def create_sparse_chunks(self, content: str, file_info: Dict[str, Any], repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create sparse chunks using file-based and section-based chunking."""
        try:
            chunks = []
            
            # Strategy 1: File-based chunking (entire file as one chunk)
            file_chunk = self._create_file_chunk(content, file_info, repo_info)
            chunks.append(file_chunk)
            
            # Strategy 2: Section-based chunking (large sections)
            section_chunks = self._chunk_by_sections(content, file_info, repo_info)
            chunks.extend(section_chunks)
            
            # Filter by size
            filtered_chunks = [chunk for chunk in chunks if self._is_valid_chunk_size(chunk['text'])]
            
            logging.info(f"Created {len(filtered_chunks)} sparse chunks for {file_info['path']}")
            return filtered_chunks
            
        except Exception as e:
            logging.error(f"Error creating sparse chunks for {file_info['path']}: {e}")
            return []
    
    def _chunk_by_paragraphs(self, content: str, file_info: Dict[str, Any], repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split content by paragraphs."""
        chunks = []
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', content.strip())
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) < self.min_chunk_size:
                continue
                
            chunk_id = str(uuid.uuid4())
            chunks.append({
                'chunk_id': chunk_id,
                'text': paragraph,
                'chunk_type': 'paragraph',
                'chunk_index': i,
                'metadata': self._create_metadata(file_info, repo_info, 'dense')
            })
        
        return chunks
    
    def _chunk_by_sentences(self, content: str, file_info: Dict[str, Any], repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split content by sentences."""
        chunks = []
        
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', content.strip())
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
            
            # If chunk is big enough, create it
            if len(current_chunk) >= self.min_chunk_size:
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk,
                    'chunk_type': 'sentence_group',
                    'chunk_index': chunk_index,
                    'metadata': self._create_metadata(file_info, repo_info, 'dense')
                })
                current_chunk = ""
                chunk_index += 1
        
        # Add remaining content as last chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_id = str(uuid.uuid4())
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk,
                'chunk_type': 'sentence_group',
                'chunk_index': chunk_index,
                'metadata': self._create_metadata(file_info, repo_info, 'dense')
            })
        
        return chunks
    
    def _create_file_chunk(self, content: str, file_info: Dict[str, Any], repo_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk containing the entire file."""
        chunk_id = str(uuid.uuid4())
        return {
            'chunk_id': chunk_id,
            'text': content,
            'chunk_type': 'file',
            'chunk_index': 0,
            'metadata': self._create_metadata(file_info, repo_info, 'sparse')
        }
    
    def _chunk_by_sections(self, content: str, file_info: Dict[str, Any], repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split content by large sections (headers, etc.)."""
        chunks = []
        
        # Look for markdown headers, code blocks, or other section markers
        lines = content.split('\n')
        current_section = []
        section_index = 0
        
        for line in lines:
            # Check for section markers
            is_header = re.match(r'^#{1,6}\s+', line)  # Markdown headers
            is_code_block = line.startswith('```') or line.startswith('    ')  # Code blocks
            is_separator = re.match(r'^[-=*_]{3,}$', line)  # Separators
            
            if (is_header or is_code_block or is_separator) and current_section:
                # Create chunk from current section
                section_text = '\n'.join(current_section).strip()
                if len(section_text) >= self.min_chunk_size:
                    chunk_id = str(uuid.uuid4())
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': section_text,
                        'chunk_type': 'section',
                        'chunk_index': section_index,
                        'metadata': self._create_metadata(file_info, repo_info, 'sparse')
                    })
                    section_index += 1
                current_section = []
            
            current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if len(section_text) >= self.min_chunk_size:
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': section_text,
                    'chunk_type': 'section',
                    'chunk_index': section_index,
                    'metadata': self._create_metadata(file_info, repo_info, 'sparse')
                })
        
        return chunks
    
    def _create_metadata(self, file_info: Dict[str, Any], repo_info: Dict[str, Any], chunking_type: str) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        return {
            'repo_name': repo_info.get('name', ''),
            'file_path': file_info.get('path', ''),
            'branch': repo_info.get('default_branch', 'main'),
            'commit_hash': file_info.get('sha', ''),
            'author': file_info.get('author', ''),
            'last_modified': file_info.get('last_modified', ''),
            'file_size': file_info.get('size', 0),
            'chunking_type': chunking_type,
            'file_language': self._get_file_language(file_info.get('path', ''))
        }
    
    def _get_file_language(self, file_path: str) -> str:
        """Get programming language from file path."""
        extension = file_path.lower().split('.')[-1] if '.' in file_path else ''
        
        language_map = {
            'py': 'Python',
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'java': 'Java',
            'cpp': 'C++',
            'c': 'C',
            'h': 'C/C++ Header',
            'cs': 'C#',
            'php': 'PHP',
            'rb': 'Ruby',
            'go': 'Go',
            'rs': 'Rust',
            'swift': 'Swift',
            'kt': 'Kotlin',
            'md': 'Markdown',
            'txt': 'Text',
            'json': 'JSON',
            'xml': 'XML',
            'yaml': 'YAML',
            'yml': 'YAML',
            'sql': 'SQL',
            'sh': 'Shell Script'
        }
        
        return language_map.get(extension, 'Unknown')
    
    def _is_valid_chunk_size(self, text: str) -> bool:
        """Check if chunk size is valid."""
        return self.min_chunk_size <= len(text) <= self.max_chunk_size
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on text content."""
        seen_texts = set()
        unique_chunks = []
        
        for chunk in chunks:
            text = chunk['text'].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_chunks.append(chunk)
        
        return unique_chunks 