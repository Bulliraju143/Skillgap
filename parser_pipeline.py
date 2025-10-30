import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from file_readers_txt import read_txt
from file_readers_docx import read_docx
from file_readers_pdf import read_pdf
from txt_cleaner import normalize_text
from remove_personal import remove_personal

class DocumentParser:
    """Unified document parser"""
    
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temp directory"""
        file_path = os.path.join(self.temp_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    def extract_text_auto(self, file_path: str) -> Tuple[str, str, str]:
        """Extract and clean text from file"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # Extract raw text
        if ext == '.txt':
            raw_text = read_txt(file_path)
        elif ext == '.docx':
            raw_text = read_docx(file_path)
        elif ext == '.pdf':
            raw_text = read_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Clean text
        cleaned_text = remove_personal(raw_text)
        cleaned_text = normalize_text(cleaned_text)
        
        return raw_text, cleaned_text, ext
    
    def process_text_input(self, text: str, doc_type: str) -> Tuple[str, str, str]:
        """Process manually entered text"""
        raw_text = text
        cleaned_text = remove_personal(text)
        cleaned_text = normalize_text(cleaned_text)
        
        return raw_text, cleaned_text, "text"

def get_parser():
    """Get parser instance"""
    return DocumentParser()