"""Processing configuration for hierarchical RAG system."""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class PathsConfig:
    """Configuration for file paths."""
    input_dir: str = "data/raw"
    output_dir: str = "data/processed"
    temp_dir: str = "data/temp"
    log_dir: str = "logs"


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    batch_size: int = 3
    max_workers: int = 2
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retries: int = 3
    timeout: int = 300


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    semantic_boundary_detection: bool = True
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True


@dataclass
class SummarizationConfig:
    """Configuration for summarization parameters."""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_summary_length: int = 500
    max_topics: int = 10
    max_keywords: int = 20
    temperature: float = 0.3
    top_p: float = 0.9


@dataclass
class DocumentLoaderConfig:
    """Configuration for document loading."""
    supported_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.txt'])
    max_file_size_mb: int = 100
    timeout_seconds: int = 30


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file: str = "processing.log"


@dataclass
class ProcessingConfig:
    """Main processing configuration class."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    document_loader: DocumentLoaderConfig = field(default_factory=DocumentLoaderConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> 'ProcessingConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            paths=PathsConfig(**config_dict.get('paths', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            chunking=ChunkingConfig(**config_dict.get('chunking', {})),
            summarization=SummarizationConfig(**config_dict.get('summarization', {})),
            document_loader=DocumentLoaderConfig(**config_dict.get('document_loader', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'paths': {
                'input_dir': self.paths.input_dir,
                'output_dir': self.paths.output_dir,
                'temp_dir': self.paths.temp_dir,
                'log_dir': self.paths.log_dir
            },
            'processing': {
                'batch_size': self.processing.batch_size,
                'max_workers': self.processing.max_workers,
                'chunk_size': self.processing.chunk_size,
                'chunk_overlap': self.processing.chunk_overlap,
                'max_retries': self.processing.max_retries,
                'timeout': self.processing.timeout
            },
            'chunking': {
                'min_chunk_size': self.chunking.min_chunk_size,
                'max_chunk_size': self.chunking.max_chunk_size,
                'semantic_boundary_detection': self.chunking.semantic_boundary_detection,
                'preserve_paragraphs': self.chunking.preserve_paragraphs,
                'preserve_sentences': self.chunking.preserve_sentences
            },
            'summarization': {
                'model_name': self.summarization.model_name,
                'max_summary_length': self.summarization.max_summary_length,
                'max_topics': self.summarization.max_topics,
                'max_keywords': self.summarization.max_keywords,
                'temperature': self.summarization.temperature,
                'top_p': self.summarization.top_p
            },
            'document_loader': {
                'supported_extensions': self.document_loader.supported_extensions,
                'max_file_size_mb': self.document_loader.max_file_size_mb,
                'timeout_seconds': self.document_loader.timeout_seconds
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file': self.logging.file
            }
        }

    @classmethod
    def default(cls) -> 'ProcessingConfig':
        """Create default configuration."""
        return cls()

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.processing.chunk_overlap >= self.processing.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.chunking.min_chunk_size >= self.chunking.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")
        
        if not self.document_loader.supported_extensions:
            raise ValueError("supported_extensions cannot be empty")