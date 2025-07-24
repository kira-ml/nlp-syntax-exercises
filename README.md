# NLP Syntax Exercises

A curated collection of standalone, syntax-focused Python exercises that build foundational fluency in Natural Language Processing (NLP). These scripts are designed to help learners understand the low-level patterns and coding techniques commonly used in modern NLP workflows—without the overhead of full project scaffolding or complex modeling pipelines.

---

## 🎯 Motivation & Learning Objectives

The goal of this repository is to help aspiring NLP engineers, Python learners, and ML practitioners:

- Develop practical fluency with the core Python constructs used in real-world NLP
- Understand syntax patterns found in academic papers, research prototypes, and production codebases
- Gain confidence manipulating text, embeddings, attention mechanisms, and evaluation metrics at the code level

Rather than building end-to-end systems, each exercise focuses on **minimal, explainable code examples** that reflect real NLP components—ideal for study, practice, or interview preparation.

---

## 📁 Folder Structure Overview

Exercises are organized into topical directories, each representing a key area in NLP pipelines:

```
nlp-syntax-exercises/
├── 01_text_preprocessing/             # Tokenization, normalization, text cleaning
├── 02_embeddings/                     # Word, contextual, and sentence embeddings
├── 03_transformers/                   # Attention and transformer mechanics
├── 04_language_modeling/              # Causal/masked language models
├── 05_sequence_tasks/                 # NER, QA, classification, summarization
├── 06_retrieval_augmented_generation/ # Dense retrieval, index building
├── 07_prompt_engineering/             # Prompt formats and few-shot examples
├── 08_evaluation/                     # Traditional and learned evaluation metrics
├── 09_ethics_safety/                  # Bias, redaction, hallucination
└── utils/                             # Shared helper scripts (minimal)
```

Each folder contains **independent Python files** (`*.py`), designed to be run directly and studied in isolation.

---

## 🧪 How to Use These Exercises

### Requirements

- Python 3.8 or higher
- A terminal or IDE to run `.py` files
- Recommended: install packages listed in `requirements.txt` using `pip`

### Run an Exercise

Navigate to any topic directory and execute a script:

```bash
cd 01_text_preprocessing/
python basic_tokenization.py
````

Each script:

* Is self-contained and directly executable
* Includes inline comments explaining syntax and NLP logic
* Emphasizes readability and reproducibility
* Avoids hidden dependencies or abstracted interfaces

There is **no need to install or import this repository as a library**—just open a script and run it.

---

## 📚 Recommended Learning Sequence

For learners progressing from fundamentals to advanced concepts:

| Phase                  | Modules                                     |
| ---------------------- | ------------------------------------------- |
| **1. Fundamentals**    | `01_text_preprocessing/`                    |
| **2. Representations** | `02_embeddings/`                            |
| **3. Architectures**   | `03_transformers/`, `04_language_modeling/` |
| **4. Applications**    | `05_sequence_tasks/`                        |
| **5. Extensions**      | `06_retrieval_augmented_generation/` onward |

---

## 🧠 Example Exercise Output

While all scripts are independent, here's a taste of what you’ll see:

```text
Original text: "NLP models tokenize text differently."
BPE Tokens: ['N', 'L', 'P', '▁models', '▁token', 'ize', '▁text', '▁different', 'ly', '.']
Token IDs: [45, 76, 21, 930, 1286, 619, 351, 2142, 614, 4]
```

Many exercises will print intermediate outputs, visualizations (when applicable), or explanations via comments. You’re encouraged to **modify the code directly** as a learning tool.

---

## 🤝 Contribution Guidelines

We welcome contributions that enhance the educational value of this repository.

### You Can Contribute By:

* Adding new exercises that highlight core NLP syntax
* Improving clarity, comments, or instructional explanations
* Fixing errors or inconsistencies
* Suggesting new topics relevant to foundational NLP work

### Contribution Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b add-new-exercise`
3. Follow Python best practices: PEP8, docstrings, clear variable names
4. Submit a pull request with a short explanation of your change

Please keep submissions **focused, educational, and syntax-oriented**.

---

## 📦 Dependencies

**Current core dependencies:**

* [`torch`](https://pytorch.org/) – Deep learning framework
* [`transformers`](https://huggingface.co/transformers) – Pretrained tokenizers & models
* [`nltk`](https://www.nltk.org/) – Basic text operations
* [`numpy`](https://numpy.org/) – Numerical operations

**Additional dependencies will be added as exercises are implemented:**

* `spacy` – For tokenization and NER exercises
* `sentence-transformers` – For embedding utilities
* `faiss-cpu` – For vector search exercises
* `pandas` – For data handling (used sparingly)

Run `pip install -r requirements.txt` to install the current dependency set.

---

## 📄 License

This repository is released under the **MIT License**. See the [LICENSE](./LICENSE) file for full terms.

---

## 🙏 Acknowledgments

* Hugging Face, spaCy, and the open NLP community for tooling and inspiration
* The many educators and researchers whose code and lectures informed these patterns
* Learners and contributors who share the mission of making NLP more accessible

---

**Stay curious, and happy coding!** ✍️
*Understanding NLP begins with understanding its syntax.*
