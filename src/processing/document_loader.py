import os
import logging
from pathlib import Path
from typing import List, Optional, Dict
import PyPDF2
from models.data_structures import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Robust document loader for OECD BEPS Pillar Two dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.supported_extensions = config.get('document_loader', {}).get('supported_extensions', ['.pdf'])
        self.max_file_size_mb = config.get('document_loader', {}).get('max_file_size_mb', 100)
        self.timeout_seconds = config.get('document_loader', {}).get('timeout_seconds', 30)
        
    def load_documents(self, directory: str) -> List[Document]:
        """Load all documents from a directory"""
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.error(f"Directory does not exist: {directory}")
            return documents
            
        # Find all supported files
        for extension in self.supported_extensions:
            pattern = f"*{extension}"
            files = directory_path.rglob(pattern)
            
            for file_path in files:
                try:
                    document = self._load_single_document(str(file_path))
                    if document:
                        documents.append(document)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {str(e)}")
                    
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def _load_single_document(self, file_path: str) -> Optional[Document]:
        """Load a single document with error handling"""
        try:
            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > self.max_file_size_mb:
                logger.warning(f"File too large: {file_path} ({file_size_mb:.1f}MB)")
                return None
                
            # Extract content based on file type
            content, metadata = self._extract_content(file_path)
            
            if not content.strip():
                logger.warning(f"Empty content in {file_path}")
                return None
                
            # Create document ID from file path
            doc_id = self._generate_document_id(file_path)
            
            return Document(
                id=doc_id,
                file_path=file_path,
                content=content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return Document(
                id=self._generate_document_id(file_path),
                file_path=file_path,
                content="",
                metadata={},
                error=str(e)
            )
    
    def _extract_content(self, file_path: str) -> tuple[str, Dict[str, str]]:
        """Extract content and metadata from file"""
        file_path_obj = Path(file_path)
        extension = file_path_obj.suffix.lower()
        
        if extension == '.pdf':
            return self._extract_pdf_content(file_path)
        elif extension in ['.txt', '.md']:
            return self._extract_text_content(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def _extract_pdf_content(self, file_path: str) -> tuple[str, Dict[str, str]]:
        """Extract text content from PDF"""
        content = ""
        metadata = {
            'file_type': 'pdf',
            'file_name': Path(file_path).name,
            'file_size': str(os.path.getsize(file_path))
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract PDF metadata
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                        'producer': pdf_reader.metadata.get('/Producer', ''),
                        'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')),
                        'modification_date': str(pdf_reader.metadata.get('/ModDate', ''))
                    })
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text:
                            content += f"\n\n[Page {page_num}]\n{text}"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            raise
            
        return content.strip(), metadata
    
    def _extract_text_content(self, file_path: str) -> tuple[str, Dict[str, str]]:
        """Extract text content from text files"""
        metadata = {
            'file_type': Path(file_path).suffix.lower(),
            'file_name': Path(file_path).name,
            'file_size': str(os.path.getsize(file_path))
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                
        return content, metadata
    
    def _generate_document_id(self, file_path: str) -> str:
        """Generate unique document ID from file path"""
        # Use relative path from documents directory
        path = Path(file_path)
        # Create ID from stem and parent directory
        parent = path.parent.name if path.parent.name != 'documents' else ''
        stem = path.stem
        if parent:
            return f"{parent}_{stem}"
        return stem