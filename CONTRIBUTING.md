# Contributing to NLP Syntax Exercises

Thank you for your interest in contributing to NLP Syntax Exercises! This repository aims to provide high-quality, educational NLP code examples that help learners understand syntax and patterns used in production systems.

## How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **New exercises** that demonstrate important NLP concepts
- **Bug fixes** and code improvements
- **Documentation** enhancements and clarifications
- **Performance optimizations**
- **Test coverage** improvements
- **Translation** of comments and documentation

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/nlp-syntax-exercises.git
   cd nlp-syntax-exercises
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv nlp-env
   source nlp-env/bin/activate  # On Windows: nlp-env\Scripts\activate
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

### Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following our coding standards
3. **Test your changes** thoroughly
4. **Commit your changes** with descriptive messages:
   ```bash
   git commit -m "Add tokenization exercise for subword algorithms"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Submit a pull request** via GitHub

## Coding Standards

### Code Style

- **Follow PEP 8** style guidelines
- **Use type hints** for all function signatures
- **Include comprehensive docstrings** with examples
- **Format code** with Black formatter:
  ```bash
  black your_file.py
  ```
- **Sort imports** with isort:
  ```bash
  isort your_file.py
  ```

### Documentation Standards

- **Add clear docstrings** to all functions and classes
- **Include usage examples** in docstrings
- **Comment complex logic** thoroughly
- **Update README.md** files when adding new modules
- **Maintain educational focus** with explanatory comments

### Example Code Structure

```python
from typing import List, Tuple
import torch
from transformers import AutoTokenizer


def tokenize_text(text: str, model_name: str = "bert-base-uncased") -> Tuple[List[str], List[int]]:
    """
    Tokenize input text using a specified transformer model's tokenizer.
    
    This function demonstrates the basic tokenization workflow used in
    modern NLP pipelines, including subword tokenization and ID conversion.
    
    Args:
        text: Input text to tokenize
        model_name: Hugging Face model name for tokenizer
        
    Returns:
        Tuple of (tokens, token_ids)
        
    Example:
        >>> tokens, ids = tokenize_text("Hello world!")
        >>> print(f"Tokens: {tokens}")
        >>> print(f"IDs: {ids}")
    """
    # Initialize tokenizer - this loads the vocabulary and special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize text into subword units
    tokens = tokenizer.tokenize(text)
    
    # Convert tokens to numerical IDs for model input
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return tokens, token_ids
```

## Adding New Exercises

### Exercise Structure

Each exercise should follow this pattern:

1. **Clear learning objective** stated at the top
2. **Minimal imports** - only what's necessary
3. **Step-by-step implementation** with comments
4. **Example usage** demonstrating the concept
5. **Extension suggestions** for deeper exploration

### File Organization

- Place exercises in appropriate module directories
- Use descriptive filenames (e.g., `bert_attention_visualization.py`)
- Include module-level README.md with exercise descriptions
- Follow existing naming conventions

### Educational Guidelines

- **Focus on syntax and patterns** rather than end-to-end projects
- **Explain the "why"** behind code choices in comments
- **Use realistic examples** that reflect production usage
- **Keep exercises focused** - one concept per file
- **Provide context** about when/where techniques are used

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_tokenization.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Add unit tests for new functions
- Test edge cases and error conditions
- Include doctest examples where appropriate
- Ensure tests are fast and reliable

## Documentation

### README Updates

When adding new modules or exercises:

1. Update the main README.md repository structure
2. Add brief descriptions of new exercises
3. Update the recommended learning path if needed
4. Include new dependencies in the Dependencies section

### Module Documentation

Each module should have:

- **README.md** with learning objectives
- **Exercise descriptions** and prerequisites
- **Usage examples** and expected outputs
- **Further reading** suggestions

## Submitting Pull Requests

### Before Submitting

- [ ] Code follows style guidelines (Black, isort, PEP 8)
- [ ] All tests pass
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] Changes are focused and atomic

### Pull Request Description

Include in your PR description:

- **What**: Brief description of changes
- **Why**: Motivation for the changes
- **How**: Technical approach taken
- **Testing**: How you verified the changes work
- **Documentation**: What documentation was updated

### Review Process

1. Automated checks will run (linting, tests)
2. Maintainers will review code and provide feedback
3. Address review comments and update PR
4. Once approved, changes will be merged

## Code of Conduct

### Our Standards

- **Be respectful** and inclusive in all interactions
- **Focus on learning** and educational value
- **Provide constructive feedback** in reviews
- **Help others learn** through clear explanations
- **Maintain professional tone** in all communications

### Reporting Issues

If you encounter inappropriate behavior, please report it by:
- Opening a GitHub issue (for public discussion)
- Emailing maintainers directly (for sensitive matters)

## Questions and Support

- **Documentation**: Check module README files first
- **Issues**: Search existing GitHub issues before creating new ones
- **Discussions**: Use GitHub Discussions for general questions
- **Bugs**: Report bugs with minimal reproduction examples

## Recognition

Contributors will be:
- Listed in repository contributors
- Credited in relevant module documentation
- Recognized in release notes for significant contributions

Thank you for helping make NLP education more accessible! 🚀
