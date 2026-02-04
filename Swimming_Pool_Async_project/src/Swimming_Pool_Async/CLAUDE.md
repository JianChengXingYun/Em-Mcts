# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Swimming_Pool_Async is a Python package for LLM-based exploration and reasoning systems. The project implements advanced LLM interaction patterns including:

- **LLaMA-Berry Arena**: A Monte Carlo Tree Search (MCTS) based system for LLM response generation with multi-model evaluation
- **Socratic Dialogue System**: Iterative reasoning and exploration using multiple LLM models
- **RAG Integration**: Async FAISS-based retrieval augmented generation
- **Process Control**: Structured LLM interaction with retry logic, streaming, and structured output parsing

## Core Architecture

### Main Components

1. **LLM_Core** (`LLM_Core.py`)
   - Async OpenAI-compatible API client wrapper
   - Supports both async and sync modes
   - Handles streaming, structured outputs (Pydantic), embeddings, and tool calling
   - Base URL defaults to `http://0.0.0.0:6001/v1`

2. **Process_Controller** (`Process_Controller.py`)
   - Orchestrates LLM interactions with retry logic and error handling
   - Methods: `receive_data()`, `receive_data_thinking()`, `receive_data_structural()`
   - Handles sensitive word filtering via `Tools.sensitive_words`
   - Supports structured output parsing with Pydantic models

3. **LLMExplorer_Socrates** (`LLMExplorer_Socrates_re_berry_v4_arena.py`)
   - Main exploration engine using MCTS-like algorithms
   - Supports multi-model arena evaluation with ELO ratings
   - Configurable rollout iterations (`max_iter`)
   - Integrates with RAG for context retrieval
   - State tracking and visualization capabilities

4. **Prompter** (`Prompter.py`)
   - Large collection of prompt templates (1029KB file)
   - Includes safety prompts for sensitive content detection
   - Supports multiple tokenizer types (Qwen, Llama, Gemma)

5. **Tools** (`Tools.py`)
   - Utility functions for data processing
   - Document deduplication using MinHash LSH
   - Sensitive word filtering
   - Label extraction and context management

6. **AsyncFaissRAG** (`simple_rag.py`)
   - Async FAISS-based vector search
   - JSON storage for documents
   - Embedding generation via API
   - Thread-safe operations with asyncio locks

7. **llama_berry_arena_server** (`llama_berry_arena_server.py`)
   - FastAPI server providing OpenAI-compatible API
   - Wraps LLMExplorer_Socrates for inference
   - Model naming pattern: `fresh-vegetables-{method}-rollout{N}-{base_model}`
   - Methods: `EmMcts` (MCTS-based) or `bestofN` (best-of-N sampling)

## Development Commands

### Installation

```bash
# Install the package in development mode
cd /data/jcxy/haolu/workspace/skills-upload/Swimming_Pool_Async_project
pip install -e .
```

### Running the API Server

```bash
# Start the LLaMA-Berry Arena API server
cd /data/jcxy/haolu/workspace/skills-upload/Swimming_Pool_Async_project/src/Swimming_Pool_Async
python llama_berry_arena_server.py
```

The server provides an OpenAI-compatible API endpoint for the LLaMA-Berry system.

### Testing

```bash
# Run tests with pytest
cd /data/jcxy/haolu/workspace/skills-upload/Swimming_Pool_Async_project
python -m pytest

# Run specific test file
python -m pytest build/lib/Swimming_Pool_Async/test_api_server.py
```

### Package Structure

```
Swimming_Pool_Async/
├── LLM_Core.py                    # Core LLM client
├── Process_Controller.py          # LLM interaction orchestration
├── LLMExplorer_Socrates_re_berry_v4_arena.py  # Main exploration engine
├── Prompter.py                    # Prompt templates
├── Tools.py                       # Utility functions
├── simple_rag.py                  # RAG system
├── llama_berry_arena_server.py   # FastAPI server
└── __init__.py                    # Package exports
```

## Key Design Patterns

### Async-First Architecture

All I/O operations use `async/await`:
- LLM API calls via `AsyncOpenAI`
- File operations via `aiofiles` or `asyncio.to_thread()`
- RAG operations with async locks for thread safety

### Multi-Model Evaluation

LLMExplorer_Socrates supports:
- Multiple LLM instances (`llm`, `api_llm`, `api_llm2`)
- Model arena with configurable model pools (`model_configs`)
- ELO rating system for model performance tracking
- Diversity fusion for combining multiple model outputs

### Structured Output Parsing

Uses OpenAI's beta structured output API:
```python
completion = await client.beta.chat.completions.parse(
    model=model,
    messages=messages,
    response_format=PydanticModel,
    temperature=temperature
)
parsed_result = completion.choices[0].message.parsed
```

### Retry Logic with Exponential Backoff

All LLM calls include retry logic:
- Default: 3-6 retries with exponential backoff
- Returns `"<|_error_|>"` on failure
- Configurable via `max_retries` and `initial_delay` parameters

## Important Notes

### API Configuration

- Default base URL: `http://0.0.0.0:6001/v1`
- API key: `"EMPTY"` (for local/internal APIs)
- Models are specified via `api_model` parameter in LLM_Core

### Model Naming Convention

For the arena server, models follow this pattern:
```
fresh-vegetables-{method}-rollout{N}-{base_model}
```
Example: `fresh-vegetables-EmMcts-rollout2-gpt-oss-120b-low`

### Sensitive Content Handling

The system includes safety mechanisms:
- Sensitive word filtering via `Tools.sensitive_words`
- Safety prompts in Prompter for detecting suicide risk, self-harm, etc.
- Automatic retry if sensitive content detected

### Python Version

Requires Python >= 3.10 (specified in setup.py)

## Common Workflows

### Creating a New LLM Instance

```python
from Swimming_Pool_Async import LLM_Core

llm = LLM_Core(
    use_async=True,
    api_model="your-model-name",
    base_url="http://localhost:6001/v1",
    api_key="EMPTY"
)
```

### Using LLMExplorer_Socrates

```python
from Swimming_Pool_Async import LLMExplorer_Socrates_re_berry_v4_arena as LLMExplorer

explorer = LLMExplorer(
    llm=llm,
    max_iter=2,  # Number of MCTS rollouts
    use_diversity_fusion=False,
    enable_elo_weighting=True
)

# Run exploration
result = await explorer.explore(query="Your question here")
```

### Using RAG

```python
from Swimming_Pool_Async import AsyncFaissRAG

# Create RAG instance
rag = await AsyncFaissRAG.create(
    json_path="rag_data.json",
    faiss_index_path="rag_index.faiss",
    api_url="http://localhost:6007/v1",
    model_name="qwen3-emb"
)

# Add documents
await rag.add_document(key="doc1", value="content", metadata={})

# Search
results = await rag.search(query="search query", top_k=5)
```

## Git Workflow

Current branch: `flash-mctsr-zero2`

The repository has undergone significant cleanup with many experimental files deleted. Active development focuses on:
- `LLM_Core.py`
- `Process_Controller.py`
- `Prompter.py`
- `Tools.py`
- `LLMExplorer_Socrates_re_berry_v4_arena.py`
- `llama_berry_arena_server.py`
