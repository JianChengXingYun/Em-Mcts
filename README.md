# Em-Mcts - Empirical Monte Carlo Tree Search Framework

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.04248-b31b1b?style=flat-square)](https://arxiv.org/abs/2602.04248)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Empirical-MCTS**: A dual-loop inference-time scaling framework that transforms stateless Monte Carlo Tree Search into continuous empirical learning. By unifying real-time meta-prompt evolution (PE-EMP) with global memory optimization, it enables LLMs to accumulate and reuse reasoning wisdom across problems—significantly boosting performance on complex reasoning benchmarks like AIME25 and MathArena Apex.

<img width="1027" height="671" alt="Em-Mcts Architecture" src="https://github.com/user-attachments/assets/eaff7000-bb40-491d-9d1b-e8a358a8eb0f" />

</div>

<div align="center">

<img width="484" height="839" alt="Performance Comparison" src="https://github.com/user-attachments/assets/5232ff91-3b8f-482f-a4f0-2ee75824f549" />

<img width="1499" height="525" alt="Method Overview" src="https://github.com/user-attachments/assets/451c5052-f1f3-4330-bfee-3a5c5728a9e9" />

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Usage Examples](#usage-examples)
- [API Documentation](#api-documentation)
- [FAQ](#faq)
- [Citation](#citation)

---

## Project Overview

Em-Mcts is an LLM reasoning framework based on Monte Carlo Tree Search (MCTS), specifically designed for complex reasoning tasks. It significantly improves performance through the following innovative mechanisms:

- **Dual-Loop Architecture**: Combines inference-time search with training-time learning
- **Meta-Prompt Evolution**: Dynamically optimizes system prompts to adapt to different problems
- **Global Memory Optimization**: Accumulates reasoning experience across problems
- **ELO Rating System**: Dynamically evaluates and selects optimal models
- **Async RAG Integration**: Supports retrieval-augmented generation

Demonstrates excellent performance on complex reasoning benchmarks such as AIME25 and MathArena Apex.

---

## Core Features

### 🎯 Main Capabilities

| Feature | Description |
|---------|-------------|
| **MCTS Search** | Reasoning process based on Monte Carlo Tree Search |
| **Multi-Model Arena** | Parallel evaluation support for multiple LLM models |
| **ELO Rating** | Dynamic model performance tracking and optimal model selection |
| **Async RAG** | Asynchronous FAISS vector search and JSON storage |
| **State Tracking** | Complete search process recording and visualization |
| **Configuration Management** | Centralized configuration file for all model parameters |

### 🔧 Technology Stack

- **LLM Framework**: OpenAI-compatible API
- **Async Programming**: asyncio + aiohttp
- **Vector Search**: FAISS
- **Visualization**: Pyvis network graphs
- **Configuration Management**: JSON configuration files

---

## Project Structure

```
Em-Mcts/
├── README.md                              # Original README
├── README_EN.md                           # English README (this file)
├── README_CN.md                           # Chinese README
├── config.json                            # Model configuration file
├── config_loader.py                       # Configuration loader
├── requirements.txt                       # Dependencies list
├── LLMExplorer_Socrates_em_mcts.py       # Main program (129KB)
├── LICENSE                                # MIT License
├── rollout_data/                          # Search state save directory
├── rollout_data_continued/                # Continued search state directory
└── Swimming_Pool_Async_project/           # Core library project
    ├── setup.py                           # Installation configuration
    └── src/Swimming_Pool_Async/
        ├── __init__.py
        ├── LLM_Core.py                    # LLM client core
        ├── Process_Controller.py          # LLM interaction controller
        ├── LLMExplorer_Socrates_em_mcts.py # MCTS exploration engine
        ├── Prompter.py                    # Prompt template library
        ├── Tools.py                       # Utility functions
        ├── simple_rag.py                  # Async RAG system
        └── em_mcts_server.py              # FastAPI server
```

---

## Quick Start

### 1. Requirements

- Python 3.10+
- CUDA 11.0+ (if using GPU)
- Sufficient disk space for FAISS indices

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/JianChengXingYun/Em-Mcts
cd Em-Mcts

# Install Swimming_Pool package
cd Swimming_Pool_Async_project
pip install -e .
cd ..

# Install other dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

Edit the `config.json` file and fill in your API information:

```json
{
  "gen_models": {
    "gemini-3-pro-preview": {
      "api_base": "https://your-api-endpoint/v1",
      "api_key": "your-api-key-here"
    }
  },
  "judge_models": {
    "gemini-3-pro-preview": {
      "api_base": "https://your-api-endpoint/v1",
      "api_key": "your-api-key-here"
    }
  },
  "emb_models": {
    "qwen3-emb": {
      "api_base": "http://localhost:6007/v1",
      "api_key": "EMPTY"
    }
  }
}
```

### 4. Run the Program

```bash
python LLMExplorer_Socrates_em_mcts.py
```

The program will output:
- Search process logs
- Final reasoning results
- Visualization file path (HTML format)
- State save file location

---

## Configuration Guide

### config.json Structure

#### gen_models (Generation Models)
Model configuration for generating reasoning answers:

```json
"gen_models": {
  "model-name": {
    "model_name": "model-name",
    "api_base": "https://api-endpoint/v1",
    "api_key": "your-key",
    "sampling_params": {
      "extra_body": {
        "enable_thinking": true
      }
    },
    "sampling_weight": 1
  }
}
```

#### judge_models (Judge Models)
Model configuration for evaluating answer quality:

```json
"judge_models": {
  "model-name": {
    "model_name": "model-name",
    "api_base": "https://api-endpoint/v1",
    "api_key": "your-key",
    "temperature": 0.95,
    "top_p": 0.9
  }
}
```

#### mem_models (Memory Models)
Model configuration for global memory optimization:

```json
"mem_models": {
  "model-name": {
    "model_name": "model-name",
    "api_base": "https://api-endpoint/v1",
    "api_key": "your-key"
  }
}
```

#### emb_models (Embedding Models)
Model configuration for vector embeddings and RAG:

```json
"emb_models": {
  "model-name": {
    "model_name": "model-name",
    "api_base": "http://localhost:6007/v1",
    "api_key": "EMPTY"
  }
}
```

#### api_config (Global API Configuration)
Global API settings:

```json
"api_config": {
  "default_base_url": "https://api-endpoint/v1",
  "default_api_key": "your-key",
  "timeout": 30,
  "max_retries": 6
}
```

### Using ConfigLoader

```python
from config_loader import get_config_loader, init_config

# Initialize configuration loader
init_config("config.json")
config = get_config_loader()

# Get model configurations
gen_models = config.get_gen_models()
judge_models = config.get_judge_models()
emb_models = config.get_emb_models()

# Get specific model
model_config = config.get_gen_model("gemini-3-pro-preview")

# Get global API configuration
api_config = config.get_api_config()
```

---

## Usage Examples

### Basic Usage

```python
import asyncio
from Swimming_Pool_Async import LLM_Core, LLMExplorer_Socrates_re_berry_v4_arena
from Swimming_Pool_Async import AsyncFaissRAG
from config_loader import init_config, get_config_loader

async def main():
    # Initialize configuration
    init_config("config.json")
    config = get_config_loader()

    # Get model configurations
    gen_config = config.get_gen_model("gemini-3-pro-preview")
    judge_config = config.get_judge_model("gemini-3-pro-preview")
    emb_config = config.get_emb_model("qwen3-emb")

    # Create LLM instances
    llm = LLM_Core(
        use_async=True,
        api_model=gen_config["model_name"],
        base_url=gen_config["api_base"],
        api_key=gen_config["api_key"]
    )

    judge_llm = LLM_Core(
        use_async=True,
        api_model=judge_config["model_name"],
        base_url=judge_config["api_base"],
        api_key=judge_config["api_key"]
    )

    # Create RAG instance
    rag = await AsyncFaissRAG.create(
        model_name=emb_config["model_name"],
        base_url=emb_config["api_base"],
        api_key=emb_config["api_key"]
    )

    # Create explorer
    explorer = LLMExplorer_Socrates(
        llm=llm,
        api_llm=judge_llm,
        rag=rag,
        max_iter=2,
        enable_state_tracking=True,
        enable_visualization=True
    )

    # Execute search
    query = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Solve this problem: ..."}
        ]
    }

    results = await explorer.main_loop(query)
    print(f"Final Result: {results[0]}")

# Run
asyncio.run(main())
```

### Enable State Tracking and Visualization

```python
explorer = LLMExplorer_Socrates(
    llm=llm,
    api_llm=judge_llm,
    rag=rag,
    max_iter=8,
    enable_state_tracking=True,      # Enable state recording
    state_save_path="./rollout_data", # State save directory
    auto_save_interval=1,             # Auto-save every iteration
    enable_visualization=True         # Enable visualization
)

# After search completes
if hasattr(explorer, 'visualization_file'):
    print(f"Visualization file: {explorer.visualization_file}")
    print(f"State file: {explorer.state_file}")
```

### Resume from Saved State

```python
# Create new explorer
new_explorer = LLMExplorer_Socrates(
    llm=llm,
    api_llm=judge_llm,
    rag=rag,
    max_iter=10,
    enable_state_tracking=True
)

# Load previous state
if new_explorer.load_state("./rollout_data/state_file.json"):
    print("✅ State restored successfully")
    # Continue search
    results = await new_explorer.main_loop(query)
else:
    print("❌ State restoration failed")
```

---

## API Documentation

### LLMExplorer_Socrates

Main MCTS exploration engine class.

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | LLM_Core | Required | Main LLM instance |
| `api_llm` | LLM_Core | None | Judge LLM instance |
| `api_llm2` | LLM_Core | None | Second judge LLM instance |
| `max_iter` | int | 8 | Maximum iterations |
| `rag` | AsyncFaissRAG | None | RAG instance |
| `use_diversity_fusion` | bool | False | Whether to use diversity fusion |
| `enable_state_tracking` | bool | False | Whether to enable state tracking |
| `enable_visualization` | bool | True | Whether to enable visualization |
| `state_save_path` | str | "rollout_states" | State save directory |
| `auto_save_interval` | int | 1 | Auto-save interval |

#### Main Methods

```python
# Execute main search loop
results = await explorer.main_loop(query)

# Load saved state
success = explorer.load_state(state_file_path)

# Reset all data structures
explorer.reset()

# Get HTML visualization of search tree
html = explorer.get_tree_html()
```

### AsyncFaissRAG

Asynchronous FAISS vector search and RAG system.

#### Initialization

```python
rag = await AsyncFaissRAG.create(
    json_path="rag_data.json",
    faiss_index_path="rag_index.faiss",
    api_url="http://localhost:6007/v1",
    model_name="qwen3-emb",
    api_key=""
)
```

#### Main Methods

```python
# Add document
doc_id = await rag.add_document(
    key="document_key",
    value="document_content",
    metadata={"source": "example"}
)

# Search similar documents
results = await rag.search(query="search query", top_k=5)

# Calculate similarity
similarity = await rag.calculate_similarity("text1", "text2")

# Delete document
success = await rag.delete_document(doc_id)

# Get single document
doc = rag.get_document(doc_id)

# List all documents
docs = rag.list_documents(limit=100)
```

### LLM_Core

LLM client core class.

#### Initialization

```python
llm = LLM_Core(
    tokenizer=None,
    use_async=True,
    api_model="model-name",
    base_url="https://api-endpoint/v1",
    api_key="your-key"
)
```

#### Main Methods

```python
# Async model call
async for chunk in llm.async_model(data=messages):
    print(chunk)

# Sync model call
response = llm.sync_model(data=messages)

# Get embedding
embedding = await llm.get_embedding(text="text to embed")

# Structured output
result = await llm.get_structured_output(
    data=messages,
    response_format=PydanticModel
)
```

---

## FAQ

### Q: How do I modify model configuration?

A: Edit the `config.json` file and modify the `api_base` and `api_key` for the respective model, then re-run the program.

### Q: How do I add a new model?

A: Add a new model configuration in the appropriate section (gen_models, judge_models, etc.) in `config.json`:

```json
"new-model": {
  "model_name": "new-model",
  "api_base": "https://api-endpoint/v1",
  "api_key": "your-key",
  ...
}
```

### Q: How do I enable GPU acceleration?

A: Install `faiss-gpu` instead of `faiss-cpu`:

```bash
pip install faiss-gpu
```

### Q: What should I do if I encounter API errors during search?

A: Check the following:
1. Is the API key correct?
2. Is the API endpoint accessible?
3. Is the network connection working?
4. Is the API quota sufficient?

### Q: How do I resume an interrupted search?

A: Use the `load_state()` method to restore from a saved state file:

```python
explorer.load_state("./rollout_data/state_file.json")
results = await explorer.main_loop(query)
```

### Q: Where is the visualization file?

A: After search completes, the program outputs the visualization file path. Usually in the `./rollout_data/` directory with a `.html` filename.

---

## Performance Optimization Tips

1. **Adjust Iteration Count**: Adjust the `max_iter` parameter based on problem complexity
2. **Use Multiple Models**: Configure multiple judge models for better evaluation
3. **Enable RAG**: Enable RAG for problems requiring external knowledge
4. **Async Processing**: Fully utilize async features for handling multiple requests
5. **GPU Acceleration**: Use FAISS GPU version to accelerate vector search

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'Swimming_Pool_Async'`

**Solution**: Ensure the Swimming_Pool package is installed:
```bash
cd Swimming_Pool_Async_project
pip install -e .
```

### Issue: `FileNotFoundError: config.json not found`

**Solution**: Ensure `config.json` is in the current working directory, or specify the full path.

### Issue: `openai.RateLimitError: Error code: 429`

**Solution**: API quota exhausted. Please try again later or check your API configuration.

### Issue: FAISS index loading failed

**Solution**: Delete old index files and recreate:
```bash
rm rag_index.faiss rag_data.json
```

---

## Contributing

We welcome Issue submissions and Pull Requests!

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this framework in your research, please cite our paper:

```bibtex
@misc{lu2026empiricalmctscontinuousagentevolution,
      title={Empirical-MCTS: Continuous Agent Evolution via Dual-Experience Monte Carlo Tree Search},
      author={Hao Lu and Haoyuan Huang and Yulin Zhou and Chen Li and Ningxin Zhu},
      year={2026},
      eprint={2602.04248},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.04248}
}
```

---

## Contact

- **GitHub Issues**: [Submit an issue](https://github.com/JianChengXingYun/Em-Mcts/issues)
- **Paper**: [arXiv:2602.04248](https://arxiv.org/abs/2602.04248)

---

## Changelog

### v1.0.0 (2026-02-11)
- ✅ Initial release
- ✅ Configuration file management support
- ✅ Complete state tracking and visualization
- ✅ Async RAG integration
- ✅ ELO rating system

---

**Last Updated**: 2026-02-11
