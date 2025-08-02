import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        # Look for config in standard locations
        possible_paths = [
            'config/processing_config.yaml',
            'src/config/processing_config.yaml',
            'processing_config.yaml'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            # Return default config if no file found
            return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        'paths': {
            'input_dir': 'data/raw',
            'output_dir': 'data/processed',
            'temp_dir': 'data/temp',
            'log_dir': 'logs'
        },
        'processing': {
            'batch_size': 5,
            'max_workers': 4,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'max_retries': 3,
            'timeout': 300
        },
        'chunking': {
            'min_chunk_size': 100,
            'max_chunk_size': 2000,
            'semantic_boundary_detection': True,
            'preserve_paragraphs': True,
            'preserve_sentences': True
        },
        'summarization': {
            'model_name': 'microsoft/Phi-3-mini-4k-instruct',
            'max_summary_length': 500,
            'max_topics': 10,
            'max_keywords': 20,
            'temperature': 0.3,
            'top_p': 0.9
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'processing.log'
        }
    }

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)