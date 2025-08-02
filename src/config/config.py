"""Configuration management for the hierarchical RAG system."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class ProcessingConfig:
    """Configuration class for the hierarchical RAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.config_path = config_path or "config/processing_config.yaml"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self._config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self._config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'processing': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'max_workers': 4,
                'timeout': 30
            },
            'chunking': {
                'min_chunk_size': 100,
                'max_chunk_size': 2000,
                'preserve_paragraphs': True,
                'preserve_sentences': False
            },
            'summarization': {
                'model_name': 'all-MiniLM-L6-v2',
                'max_length': 512,
                'temperature': 0.7
            },
            'vector_store': {
                'similarity_threshold': 0.7,
                'top_k': 5,
                'embedding_dimension': 384
            },
            'agent': {
                'confidence_threshold': 0.8,
                'max_retries': 3,
                'timeout': 30
            }
        }
    
    @property
    def processing(self):
        """Processing configuration."""
        return type('ProcessingConfig', (), self._config.get('processing', {}))
    
    @property
    def chunking(self):
        """Chunking configuration."""
        return type('ChunkingConfig', (), self._config.get('chunking', {}))
    
    @property
    def summarization(self):
        """Summarization configuration."""
        return type('SummarizationConfig', (), self._config.get('summarization', {}))
    
    @property
    def vector_store(self):
        """Vector store configuration."""
        return type('VectorStoreConfig', (), self._config.get('vector_store', {}))
    
    @property
    def agent(self):
        """Agent configuration."""
        return type('AgentConfig', (), self._config.get('agent', {}))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        
        try:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_file, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            raise
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        self._config.update(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProcessingConfig':
        """Create configuration from dictionary."""
        config = cls()
        config._config = config_dict
        return config
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ProcessingConfig(path={self.config_path})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ProcessingConfig(config_path='{self.config_path}', config={self._config})"