"""Basic smoke tests to ensure repository structure is valid."""

import os
import sys
from pathlib import Path
import pytest


def test_repository_structure():
    """Test that all expected directories exist."""
    repo_root = Path(__file__).parent.parent
    
    expected_dirs = [
        "01_text_preprocessing",
        "02_embeddings", 
        "03_transformers",
        "04_language_modeling",
        "05_sequence_tasks",
        "06_retrieval_augmented_generation",
        "07_prompt_engineering",
        "08_evaluation",
        "09_ethics_safety",
    ]
    
    for dir_name in expected_dirs:
        dir_path = repo_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} not found"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"


def test_essential_files_exist():
    """Test that essential repository files exist."""
    repo_root = Path(__file__).parent.parent
    
    essential_files = [
        "README.md",
        "LICENSE", 
        "requirements.txt",
        ".gitignore",
        "CONTRIBUTING.md",
        "setup.py",
    ]
    
    for file_name in essential_files:
        file_path = repo_root / file_name
        assert file_path.exists(), f"Essential file {file_name} not found"
        assert file_path.is_file(), f"{file_name} is not a file"


def test_python_version():
    """Test that Python version meets requirements."""
    assert sys.version_info >= (3, 8), "Python 3.8 or higher required"


def test_imports():
    """Test that core dependencies can be imported."""
    try:
        import torch
        import transformers
        import numpy
        import sklearn
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
    except ImportError as e:
        pytest.fail(f"Failed to import core dependencies: {e}")


def test_readme_content():
    """Test that README contains expected sections."""
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    expected_sections = [
        "# NLP Syntax Exercises",
        "## Overview", 
        "## Repository Structure",
        "## Installation",
        "## Getting Started",
        "## Contributing",
        "## License",
    ]
    
    for section in expected_sections:
        assert section in content, f"README missing section: {section}"
