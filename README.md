# EU AI Act RAG Assistant
![EU AI Act Logo](data/img/streamlit_home.png)
A Retrieval-Augmented Generation (RAG) system designed to answer questions about the European Union AI Act using local LLM inference and semantic search capabilities.

## Overview

This project implements a comprehensive RAG pipeline that:
- Parses and processes the EU AI Act legal text from official sources
- Creates a vector database for efficient semantic search
- Uses local LLM inference (Llama 3.1)
- Provides a conversational interface with context-aware responses
- Includes evaluation framework for measuring RAG performance
- Features an interactive Streamlit web application

## Architecture

The system consists of several key components:

### 1. **Data Preprocessing**
- **HTML Parser** (`src/preprocessing/html_parser.py`): Fetches and parses the EU AI Act from EUR-Lex
- **Chunk Processor** (`src/preprocessing/chunk_processor.py`): Transforms hierarchical legal documents into optimized chunks with overlap and context preservation

### 2. **Retrieval System**
- **Embedding Pipeline** (`src/retrieval/embedding_pipeline.py`): Manages ChromaDB vector store and sentence transformers
- **Content Resolver** (`src/retrieval/content_resolver.py`): Resolves chunk references to full context
- Supports multiple embedding models:
  - BGE-M3
  - Snowflake Arctic Embed
  - MiniLM-L6-v2

### 3. **RAG Pipeline**
- **RAG Chatbot** (`src/rag/rag_pipeline.py`): Core conversational agent with:
  - Vector database retrieval
  - Cross-encoder reranking (FlashRank)
  - Conversation history management
  - Token-aware context window management
  - Local LLM inference using `llama-cpp-python`

### 4. **Evaluation Framework**
- **RAG Evaluator** (`src/rag/rag_evaluator.py`): Comprehensive evaluation using LLM-as-judge methodology
- Metrics include:
  - Retrieval hit rate
  - RAG vs baseline comparison
  - Source attribution accuracy

### 5. **User Interface**
- **Streamlit App** (`stream_lit/app.py`): Interactive web interface with:
  - Multi-turn conversations
  - Chat history management
  - Source citation display

## Installation

### Prerequisites

- Python 3.12 or higher
- 16GB+ RAM recommended for local LLM
- GPU support optional but recommended (CUDA or Metal)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Arthurreuss/RAG_AI_Act.git
cd RAG_AI_Act
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the LLM model**
```bash
huggingface-cli download Bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --local-dir data/llm \
  --local-dir-use-symlinks False
```


## Usage

### 1. Preprocess the EU AI Act

Download and chunk the legal text:

```bash
python scripts/preprocess.py
```

This will:
- Fetch the EU AI Act HTML from EUR-Lex
- Parse the hierarchical structure (Articles, Recitals, Annexes)
- Create optimized chunks with context
- Save processed data to `data/json/`

### 2. Create Vector Database

Generate embeddings and populate ChromaDB:

```bash
python scripts/create_db.py
```

This creates a persistent vector store in `data/chroma_db/` using the configured embedding model.

### 3. Run Interactive Chatbot

#### Command Line Interface

```bash
python scripts/run_rag.py
```


#### Streamlit Web Application

```bash
streamlit run stream_lit/app.py
```

Access the web interface at `http://localhost:8501`

Features:
- Multi-turn conversations
- Source citation viewing
- Chat history persistence
- New chat sessions

### 4. Evaluate RAG Performance

Run comprehensive evaluation on test sets:

```bash
python scripts/evaluate_rag.py
```

This will:
- Execute queries from golden test sets
- Compare RAG answers against baselines
- Generate metrics (hit rate, relevance scores)
- Save results to `results/` directory

## Configuration

All settings are managed in `config.yaml`:

### Preprocessing
```yaml
preprocessing:
  chunk_size: 150          # Words per chunk
  chunk_overlap: 30        # Overlapping words
  split_threshold: 250     # Max words before splitting
```

### RAG Pipeline
```yaml
rag_pipeline:
  embedding_model_name: snowflake-m
  llm_model: data/llm/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
  ranker_model: ms-marco-MiniLM-L-12-v2
  retrieval_top_k: 5
  rerank_candidate_k: 20
  generation_max_new_tokens: 512
  generation_temperature: 0.1
```

### Vector Store
```yaml
vector_store:
  collection_name: ai_act_legal
  path: data/chroma_db
  embedding_model_map:
    bge-m3: BAAI/bge-m3
    snowflake-m: Snowflake/snowflake-arctic-embed-m
    minilm: sentence-transformers/all-MiniLM-L6-v2
```

## Project Structure

```
RAG_AI_Act/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project metadata
│
├── data/                       # Data directory
│   ├── chroma_db/             # Vector database
│   ├── json/                  # Processed legal text
│   ├── llm/                   # LLM model files
│   └── test_set/              # Evaluation datasets
│
├── src/                        # Source code
│   ├── preprocessing/         # Text parsing and chunking
│   ├── retrieval/             # Embedding and search
│   ├── rag/                   # RAG pipeline implementation
│   ├── analysis/              # Error and data analysis
│   └── utils/                 # Helper functions
│
├── scripts/                    # Executable scripts
│   ├── preprocess.py          # Data preprocessing
│   ├── create_db.py           # Database creation
│   ├── run_rag.py             # Interactive chatbot
│   └── evaluate_rag.py        # Performance evaluation
│
├── stream_lit/                 # Web application
│   ├── app.py                 # Main Streamlit app
│   └── ui/                    # UI components
│
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── evaluation.ipynb       # Evaluation analysis
│   ├── error_analysis.ipynb   # Error diagnostics
│   └── comparison_embedding.ipynb
│
└── results/                    # Evaluation results
    ├── *.json                 # Detailed results
    └── *.csv                  # Metrics summaries
```


## Evaluation Metrics

The evaluation framework provides:

- **Retrieval Hit Rate**: Percentage of queries where relevant context is retrieved
- **RAG Score**: LLM-judged quality of RAG answers (0-10)
- **Baseline Score**: Quality of answers without retrieval (0-10)
- **Source Count**: Number of sources used per answer
- **Winner Analysis**: Head-to-head comparison between RAG and baseline

Results are saved in both CSV (metrics) and JSON (full details) formats.
