#!/usr/bin/env python3
"""
Setup script for the Hierarchical RAG Document Processing Pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = requirements_file.read_text().splitlines()

setup(
    name="hierarchical-rag-beps",
    version="1.0.0",
    description="Hierarchical RAG system for OECD BEPS Pillar Two document processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hierarchical RAG Team",
    author_email="team@hierarchical-rag.com",
    url="https://github.com/your-org/hierarchical-rag-beps",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hierarchical-rag=hierarchical_rag.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="rag oecd beps pillar-two document-processing nlp",
    project_urls={
        "Bug Reports": "https://github.com/your-org/hierarchical-rag-beps/issues",
        "Source": "https://github.com/your-org/hierarchical-rag-beps",
        "Documentation": "https://hierarchical-rag-beps.readthedocs.io/",
    },
)