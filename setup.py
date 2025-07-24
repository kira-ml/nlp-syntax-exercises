"""Setup configuration for NLP Syntax Exercises."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nlp-syntax-exercises",
    version="0.1.0",
    author="NLP Syntax Exercises Contributors",
    author_email="",
    description="Syntax-focused Natural Language Processing exercises for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nlp-syntax-exercises",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nlp-syntax-exercises/issues",
        "Source": "https://github.com/yourusername/nlp-syntax-exercises",
        "Documentation": "https://github.com/yourusername/nlp-syntax-exercises#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipykernel>=6.0",
            "matplotlib>=3.5",
            "seaborn>=0.11",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            # Future CLI tools can be added here
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "nlp",
        "natural language processing",
        "machine learning",
        "education",
        "syntax",
        "transformers",
        "pytorch",
        "huggingface",
        "exercises",
        "tutorial",
    ],
    zip_safe=False,
)
