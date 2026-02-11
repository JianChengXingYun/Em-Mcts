# Em-Mcts - 经验性蒙特卡洛树搜索框架

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.04248-b31b1b?style=flat-square)](https://arxiv.org/abs/2602.04248)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<img width="1027" height="671" alt="企业微信截图_20260206090312" src="https://github.com/user-attachments/assets/eaff7000-bb40-491d-9d1b-e8a358a8eb0f" />



**Empirical-MCTS**: 一个双循环推理时间缩放框架，将无状态蒙特卡洛树搜索转变为连续经验学习。通过统一实时元提示进化（PE-EMP）与全局记忆优化，使大语言模型能够跨问题积累和重用推理智慧。

</div>
---

<img width="484" height="839" alt="企业微信截图_20260206091122" src="https://github.com/user-attachments/assets/5232ff91-3b8f-482f-a4f0-2ee75824f549" />

---
<img width="1499" height="525" alt="企业微信截图_20260206091136" src="https://github.com/user-attachments/assets/451c5052-f1f3-4330-bfee-3a5c5728a9e9" />
---

## 📋 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [API 文档](#api-文档)
- [常见问题](#常见问题)
- [引用](#引用)

---

## 项目概述

Em-Mcts 是一个基于蒙特卡洛树搜索（MCTS）的 LLM 推理框架，专门设计用于复杂推理任务。它通过以下创新机制显著提升性能：

- **双循环架构**：结合推理时间搜索与训练时间学习
- **元提示进化**：动态优化系统提示以适应不同问题
- **全局记忆优化**：跨问题积累推理经验
- **ELO 评分系统**：动态评估和选择最优模型
- **异步 RAG 集成**：支持检索增强生成

在 AIME25 和 MathArena Apex 等复杂推理基准上表现出色。

---

## 核心特性

### 🎯 主要功能

| 功能 | 描述 |
|------|------|
| **MCTS 搜索** | 基于蒙特卡洛树搜索的推理过程 |
| **多模型竞技场** | 支持多个 LLM 模型的并行评估 |
| **ELO 评分** | 动态追踪模型性能和选择最优模型 |
| **异步 RAG** | 异步 FAISS 向量搜索和 JSON 存储 |
| **状态追踪** | 完整的搜索过程记录和可视化 |
| **配置管理** | 集中式配置文件管理所有模型参数 |
| **流式输出** | 支持流式响应和实时反馈 |

### 🔧 技术栈

- **LLM 框架**：OpenAI 兼容 API
- **异步编程**：asyncio + aiohttp
- **向量搜索**：FAISS
- **可视化**：Pyvis 网络图
- **配置管理**：JSON 配置文件

---

## 项目结构

```
Em-Mcts/
├── README.md                              # 原始 README
├── README_CN.md                           # 中文 README（本文件）
├── config.json                            # 模型配置文件
├── config_loader.py                       # 配置加载器
├── requirements.txt                       # 依赖列表
├── LLMExplorer_Socrates_em_mcts.py       # 主程序（129KB）
├── LICENSE                                # MIT 许可证
├── rollout_data/                          # 搜索状态保存目录
├── rollout_data_continued/                # 续搜索状态保存目录
└── Swimming_Pool_Async_project/           # 核心库项目
    ├── setup.py                           # 安装配置
    └── src/Swimming_Pool_Async/
        ├── __init__.py
        ├── LLM_Core.py                    # LLM 客户端核心
        ├── Process_Controller.py          # LLM 交互控制器
        ├── LLMExplorer_Socrates_em_mcts.py # MCTS 探索引擎
        ├── Prompter.py                    # 提示词模板库
        ├── Tools.py                       # 工具函数集
        ├── simple_rag.py                  # 异步 RAG 系统
        └── em_mcts_server.py              # FastAPI 服务器
```

---

## 快速开始

### 1. 环境要求

- Python 3.10+
- CUDA 11.0+ (如果使用 GPU)
- 足够的磁盘空间用于 FAISS 索引

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/JianChengXingYun/Em-Mcts
cd Em-Mcts

# 安装 Swimming_Pool 包
cd Swimming_Pool_Async_project
pip install -e .
cd ..

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 配置 API 密钥

编辑 `config.json` 文件，填入你的 API 信息：

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

### 4. 运行程序

```bash
python LLMExplorer_Socrates_em_mcts.py
```

程序会输出：
- 搜索过程日志
- 最终推理结果
- 可视化文件路径（HTML 格式）
- 状态保存文件位置

---

## 配置说明

### config.json 结构

#### gen_models（生成模型）
用于生成推理答案的模型配置：

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

#### judge_models（评判模型）
用于评估答案质量的模型配置：

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

#### mem_models（记忆模型）
用于全局记忆优化的模型配置：

```json
"mem_models": {
  "model-name": {
    "model_name": "model-name",
    "api_base": "https://api-endpoint/v1",
    "api_key": "your-key"
  }
}
```

#### emb_models（嵌入模型）
用于向量嵌入和 RAG 的模型配置：

```json
"emb_models": {
  "model-name": {
    "model_name": "model-name",
    "api_base": "http://localhost:6007/v1",
    "api_key": "EMPTY"
  }
}
```

#### api_config（通用 API 配置）
全局 API 设置：

```json
"api_config": {
  "default_base_url": "https://api-endpoint/v1",
  "default_api_key": "your-key",
  "timeout": 30,
  "max_retries": 6
}
```

### 使用 ConfigLoader

```python
from config_loader import get_config_loader, init_config

# 初始化配置加载器
init_config("config.json")
config = get_config_loader()

# 获取模型配置
gen_models = config.get_gen_models()
judge_models = config.get_judge_models()
emb_models = config.get_emb_models()

# 获取特定模型
model_config = config.get_gen_model("gemini-3-pro-preview")

# 获取通用 API 配置
api_config = config.get_api_config()
```

---

## 使用示例

### 基础使用

```python
import asyncio
from Swimming_Pool_Async import LLM_Core, LLMExplorer_Socrates_re_berry_v4_arena
from Swimming_Pool_Async import AsyncFaissRAG
from config_loader import init_config, get_config_loader

async def main():
    # 初始化配置
    init_config("config.json")
    config = get_config_loader()

    # 获取模型配置
    gen_config = config.get_gen_model("gemini-3-pro-preview")
    judge_config = config.get_judge_model("gemini-3-pro-preview")
    emb_config = config.get_emb_model("qwen3-emb")

    # 创建 LLM 实例
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

    # 创建 RAG 实例
    rag = await AsyncFaissRAG.create(
        model_name=emb_config["model_name"],
        base_url=emb_config["api_base"],
        api_key=emb_config["api_key"]
    )

    # 创建探索器
    explorer = LLMExplorer_Socrates(
        llm=llm,
        api_llm=judge_llm,
        rag=rag,
        max_iter=2,
        enable_state_tracking=True,
        enable_visualization=True
    )

    # 执行搜索
    query = {
        "prompt": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Solve this problem: ..."}
        ]
    }

    results = await explorer.main_loop(query)
    print(f"最终结果: {results[0]}")

# 运行
asyncio.run(main())
```

### 启用状态追踪和可视化

```python
explorer = LLMExplorer_Socrates(
    llm=llm,
    api_llm=judge_llm,
    rag=rag,
    max_iter=8,
    enable_state_tracking=True,      # 启用状态记录
    state_save_path="./rollout_data", # 状态保存目录
    auto_save_interval=1,             # 每次迭代自动保存
    enable_visualization=True         # 启用可视化
)

# 搜索完成后
if hasattr(explorer, 'visualization_file'):
    print(f"可视化文件: {explorer.visualization_file}")
    print(f"状态文件: {explorer.state_file}")
```

### 从保存的状态恢复

```python
# 创建新的探索器
new_explorer = LLMExplorer_Socrates(
    llm=llm,
    api_llm=judge_llm,
    rag=rag,
    max_iter=10,
    enable_state_tracking=True
)

# 加载之前的状态
if new_explorer.load_state("./rollout_data/state_file.json"):
    print("✅ 状态恢复成功")
    # 继续搜索
    results = await new_explorer.main_loop(query)
else:
    print("❌ 状态恢复失败")
```

---

## API 文档

### LLMExplorer_Socrates

主要的 MCTS 探索引擎类。

#### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm` | LLM_Core | 必需 | 主 LLM 实例 |
| `api_llm` | LLM_Core | None | 评判 LLM 实例 |
| `api_llm2` | LLM_Core | None | 记忆操作 LLM 实例 |
| `max_iter` | int | 8 | 最大迭代次数 |
| `rag` | AsyncFaissRAG | None | RAG 实例 |
| `use_diversity_fusion` | bool | False | 是否使用多样性融合 |
| `enable_state_tracking` | bool | False | 是否启用状态追踪 |
| `enable_visualization` | bool | True | 是否启用可视化 |
| `state_save_path` | str | "rollout_states" | 状态保存目录 |
| `auto_save_interval` | int | 1 | 自动保存间隔 |

#### 主要方法

```python
# 执行主搜索循环
results = await explorer.main_loop(query)

# 加载保存的状态
success = explorer.load_state(state_file_path)

# 重置所有数据结构
explorer.reset()

# 获取搜索树的 HTML 可视化
html = explorer.get_tree_html()
```

### AsyncFaissRAG

异步 FAISS 向量搜索和 RAG 系统。

#### 初始化

```python
rag = await AsyncFaissRAG.create(
    json_path="rag_data.json",
    faiss_index_path="rag_index.faiss",
    api_url="http://localhost:6007/v1",
    model_name="qwen3-emb",
    api_key=""
)
```

#### 主要方法

```python
# 添加文档
doc_id = await rag.add_document(
    key="document_key",
    value="document_content",
    metadata={"source": "example"}
)

# 搜索相似文档
results = await rag.search(query="search query", top_k=5)

# 计算相似度
similarity = await rag.calculate_similarity("text1", "text2")

# 删除文档
success = await rag.delete_document(doc_id)

# 获取单个文档
doc = rag.get_document(doc_id)

# 列出所有文档
docs = rag.list_documents(limit=100)
```

### LLM_Core

LLM 客户端核心类。

#### 初始化

```python
llm = LLM_Core(
    tokenizer=None,
    use_async=True,
    api_model="model-name",
    base_url="https://api-endpoint/v1",
    api_key="your-key"
)
```

#### 主要方法

```python
# 异步模型调用
async for chunk in llm.async_model(data=messages):
    print(chunk)

# 同步模型调用
response = llm.sync_model(data=messages)

# 获取嵌入
embedding = await llm.get_embedding(text="text to embed")

# 结构化输出
result = await llm.get_structured_output(
    data=messages,
    response_format=PydanticModel
)
```

---

## 常见问题

### Q: 如何修改模型配置？

A: 编辑 `config.json` 文件，修改相应模型的 `api_base` 和 `api_key`，然后重新运行程序。

### Q: 如何添加新的模型？

A: 在 `config.json` 中的相应部分（gen_models、judge_models 等）添加新的模型配置：

```json
"new-model": {
  "model_name": "new-model",
  "api_base": "https://api-endpoint/v1",
  "api_key": "your-key",
  ...
}
```

### Q: 如何启用 GPU 加速？

A: 安装 `faiss-gpu` 而不是 `faiss-cpu`：

```bash
pip install faiss-gpu
```

### Q: 搜索过程中出现 API 错误怎么办？

A: 检查以下几点：
1. API 密钥是否正确
2. API 端点是否可访问
3. 网络连接是否正常
4. API 配额是否充足

### Q: 如何恢复中断的搜索？

A: 使用 `load_state()` 方法从保存的状态文件恢复：

```python
explorer.load_state("./rollout_data/state_file.json")
results = await explorer.main_loop(query)
```

### Q: 可视化文件在哪里？

A: 搜索完成后，程序会输出可视化文件的路径。通常在 `./rollout_data/` 目录下，文件名为 `*.html`。

---

## 性能优化建议

1. **调整迭代次数**：根据问题复杂度调整 `max_iter` 参数
2. **使用多模型**：配置多个 judge 模型以获得更好的评估
3. **启用 RAG**：对于需要外部知识的问题，启用 RAG 功能
4. **异步处理**：充分利用异步特性处理多个请求
5. **GPU 加速**：使用 FAISS GPU 版本加速向量搜索

---

## 故障排除

### 问题：`ModuleNotFoundError: No module named 'Swimming_Pool_Async'`

**解决方案**：确保已安装 Swimming_Pool 包：
```bash
cd Swimming_Pool_Async_project
pip install -e .
```

### 问题：`FileNotFoundError: config.json not found`

**解决方案**：确保 `config.json` 在当前工作目录中，或指定完整路径。

### 问题：`openai.RateLimitError: Error code: 429`

**解决方案**：API 配额已用尽，请稍后重试或检查 API 配置。

### 问题：FAISS 索引加载失败

**解决方案**：删除旧的索引文件并重新创建：
```bash
rm rag_index.faiss rag_data.json
```

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 引用

如果你在研究中使用了本框架，请引用我们的论文：

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

## 联系方式

- **GitHub Issues**: [提交问题](https://github.com/JianChengXingYun/Em-Mcts/issues)
- **论文**: [arXiv:2602.04248](https://arxiv.org/abs/2602.04248)

---

## 更新日志

### v1.0.0 (2026-02-11)
- ✅ 初始版本发布
- ✅ 支持配置文件管理
- ✅ 完整的状态追踪和可视化
- ✅ 异步 RAG 集成
- ✅ ELO 评分系统

---

**最后更新**: 2026-02-11
