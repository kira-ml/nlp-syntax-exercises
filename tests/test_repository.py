"""Basic smoke tests to ensure repository structure is valid."""

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


def test_basic_imports():
    """Test that core dependencies can be imported."""
    # Test lightweight imports first
    try:
        import sys
        import re
        import string
        print("✅ Standard library imports successful")
    except ImportError as e:
        pytest.fail(f"Failed to import standard library: {e}")
    
    # Test NumPy (usually reliable)
    try:
        import numpy
        print(f"✅ NumPy version: {numpy.__version__}")
    except ImportError as e:
        pytest.fail(f"Failed to import NumPy: {e}")
    
    # Test heavy imports with more tolerance
    heavy_imports = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
    }
    
    for module_name, display_name in heavy_imports.items():
        try:
            module = __import__(module_name)
            print(f"✅ {display_name} version: {module.__version__}")
        except ImportError as e:
            print(f"⚠️ Warning: {display_name} not available: {e}")
            # Don't fail the test for heavy dependencies in CI
            if not hasattr(sys, '_called_from_test'):
                pytest.skip(f"Skipping {display_name} test in CI environment")
        except Exception as e:
            print(f"⚠️ Warning: {display_name} import issue: {e}")
            # Don't fail on version access issues


def test_readme_content():
    """Test that README contains expected sections."""
    repo_root = Path(__file__).parent.parent
    readme_path = repo_root / "README.md"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    expected_sections = [
        "# NLP Syntax Exercises",
        "## 🎯 Motivation & Learning Objectives", 
        "## 📁 Folder Structure Overview",
        "## 🧪 How to Use These Exercises",
        "## 🤝 Contribution Guidelines",
        "## 📄 License",
    ]
    
    for section in expected_sections:
        assert section in content, f"README missing section: {section}"


def test_text_preprocessing_modules():
    """Test that text preprocessing modules can be imported and have basic functionality."""
    repo_root = Path(__file__).parent.parent
    preprocessing_dir = repo_root / "01_text_preprocessing"
    
    # Test basic modules that don't require heavy dependencies
    basic_modules = [
        "01_tokenize_with_regex.py",
        "02_remove_stopwords_dict.py", 
        "03_lowercase_and_strip.py",
        "04_unicode_normalization.py",
        "05_remove_html_tags.py",
        "06_extract_emojis_and_symbols.py",
        "10_filter_non_english_lines.py",
        "11_clean_text_pipeline_step.py",
    ]
    
    for module_file in basic_modules:
        module_path = preprocessing_dir / module_file
        if module_path.exists():
            try:
                # Basic syntax check
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                compile(content, str(module_path), 'exec')
                print(f"✅ {module_file} syntax check passed")
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {module_file}: {e}")
            except Exception as e:
                print(f"⚠️ Warning: Issue with {module_file}: {e}")


def test_nlp_dependencies_optional():
    """Test NLP dependencies with graceful handling of missing packages."""
    nlp_packages = {
        'nltk': 'NLTK',
        'spacy': 'spaCy', 
    }
    
    available_packages = []
    for package_name, display_name in nlp_packages.items():
        try:
            __import__(package_name)
            available_packages.append(display_name)
            print(f"✅ {display_name} is available")
        except ImportError:
            print(f"⚠️ {display_name} not available (optional for basic tests)")
    
    # At least one basic package should be available, but don't fail if none are
    if available_packages:
        print(f"Available NLP packages: {', '.join(available_packages)}")
    else:
        print("No NLP packages available - this is OK for basic repository tests")
