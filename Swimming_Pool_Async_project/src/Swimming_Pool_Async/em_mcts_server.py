#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLaMA-Berry Arena OpenAI Compatible API Server
将 LLMExplorer_Socrates_re_berry_v4_arena.py 包装为 OpenAI 兼容的 API 服务
"""

import asyncio
import json
import time
import uuid
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict
import uvicorn

# 导入我们的LLMExplorer
from Swimming_Pool_Async.LLMExplorer_Socrates_re_berry_v4_arena import LLMExplorer_Socrates
from Swimming_Pool_Async.LLM_Core import LLM_Core
from Swimming_Pool_Async.simple_rag import AsyncFaissRAG
from transformers import AutoTokenizer
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # Startup
    logger.info("Starting LLaMA-Berry Arena API Server...")
    await init_tokenizer()
    await init_rag()
    logger.info("Server startup completed")
    yield
    # Shutdown (可选)
    logger.info("Server shutting down...")

app = FastAPI(
    title="LLaMA-Berry Arena API",
    description="OpenAI Compatible API for LLaMA-Berry Arena Model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 认证
security = HTTPBearer()

# 配置
class ServerConfig:
    API_KEY = "EMPTY"
    # 支持的模型名称模式（使用正则表达式）
    MODEL_NAME_PATTERN = r"fresh-vegetables-(EmMcts|bestofN)-rollout(\d+)-(.+)"
    # 默认模型名称（用于兼容性）
    MODEL_NAME = "fresh-vegetables-EmMcts-rollout2-gpt-oss-120b-low"
    MAX_ITER = 2
    USE_DIVERSITY_FUSION = False
    TOKENIZER_PATH = "Llama-3.1-8B-Instruct"

config = ServerConfig()

def parse_model_name(model_name: str) -> Optional[Dict[str, Any]]:
    """
    解析模型名称并提取参数

    格式: fresh-vegetables-{method}-rollout{N}-{base_model}
    例如: fresh-vegetables-EmMcts-rollout2-gpt-oss-120b-low

    返回: {
        "method": "EmMcts" 或 "bestofN",
        "rollout": 2,
        "base_model": "gpt-oss-120b-low",
        "full_name": "原始模型名称"
    }
    """
    pattern = config.MODEL_NAME_PATTERN
    match = re.match(pattern, model_name)

    if match:
        method, rollout_str, base_model = match.groups()
        return {
            "method": method,
            "rollout": int(rollout_str),
            "base_model": base_model,
            "full_name": model_name
        }

    # 兼容默认模型名称
    if model_name == config.MODEL_NAME:
        # 从默认名称中解析
        match = re.match(pattern, config.MODEL_NAME)
        if match:
            method, rollout_str, base_model = match.groups()
            return {
                "method": method,
                "rollout": int(rollout_str),
                "base_model": base_model,
                "full_name": model_name
            }

    return None

def is_valid_model_name(model_name: str) -> bool:
    """验证模型名称是否有效"""
    return parse_model_name(model_name) is not None

# 全局变量
explorer_cache = {}
tokenizer = None
rag_cache = {}
class Message(BaseModel):
    role: str = Field(..., description="角色：system, user, assistant")
    content: str = Field(..., description="消息内容")

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="对话消息列表")
    temperature: Optional[float] = Field(0.9, description="温度参数")
    max_tokens: Optional[int] = Field(None, description="最大token数")
    stream: Optional[bool] = Field(False, description="是否流式返回")
    # 自定义参数
    max_iter: Optional[int] = Field(2, description="最大迭代次数（仅用于 Arena）")
    rollout: Optional[int] = Field(None, description="Rollout参数")
    use_diversity_fusion: Optional[bool] = Field(False, description="是否使用多样性融合")
    domain: Optional[str] = Field("通用", description="领域：通用 或 心理")
    save_dataset_path: Optional[str] = Field("judge_training_data_aime25.jsonl", description="ELO评分数据文件路径")
    elo_data_file: Optional[str] = Field("judge_training_data_aime25.jsonl", description="ELO评分数据文件路径")
    enable_elo_weighting: Optional[bool] = Field(True, description="是否启用ELO动态权重调整")
    use_best_of_n: Optional[bool] = Field(False, description="是否使用 Best-of-N 模式")
    best_of_n: Optional[int] = Field(8, description="Best-of-N 中的 N 值")

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: Optional[int] = 0  # [新增] reasoning tokens

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    # 额外的Arena信息
    arena_metadata: Optional[Dict[str, Any]] = None

# 全局变量
explorer_cache = {}
tokenizer = None
rag_cache = {}

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证API Key"""
    if credentials.credentials != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials

async def init_tokenizer():
    """初始化tokenizer"""
    global tokenizer
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_PATH, trust_remote_code=True)
            logger.info(f"Tokenizer loaded from {config.TOKENIZER_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using None")
            tokenizer = None

async def init_rag(model_name: Optional[int] = None):
    """初始化RAG

    Args:
        rollout: rollout次数，用于区分不同rollout的RAG库
    """
    # 根据rollout生成唯一的rag_key和文件名
    if model_name is not None:
        rag_key = f"{model_name}"
        json_path = f"rag_data_{model_name}.json"
        faiss_index_path = f"rag_index_{model_name}.faiss"
    else:
        rag_key = "default"
        json_path = "rag_data.json"
        faiss_index_path = "rag_index.faiss"

    if rag_key not in rag_cache:
        try:
            rag = await AsyncFaissRAG.create(
                json_path=json_path,
                faiss_index_path=faiss_index_path,
                api_url="http://localhost:60046/v1"
            )
            rag_cache[rag_key] = {
                "rag": rag,
            }
            logger.info(f"RAG initialized successfully with key: {rag_key}")
        except Exception as e:
            logger.warning(f"Failed to initialize RAG: {e}")
            rag_cache[rag_key] = None
async def create_llm_explorer(request: ChatCompletionRequest, model_name: None) -> LLMExplorer_Socrates:
    """创建LLMExplorer实例"""
    global tokenizer
    
    # 初始化tokenizer和RAG
    await init_tokenizer()
    # 传递rollout参数以使用对应的RAG库
    await init_rag(model_name)
   
    default_llm = LLM_Core(
        tokenizer=tokenizer,
        use_async=True,
        api_model="gpt-oss-120b",
        base_url="http://localhost:6014/v1",
        api_key="EMPTY"
    )
   
    default_llm2 = LLM_Core(
        tokenizer=tokenizer,
        use_async=True,
        api_model="gpt-4.1-2025-04-14",
        base_url='<your-base-url>',
        api_key='<your-api-key>'
    )
    # 获取RAG - 根据rollout参数使用对应的RAG库
    rag_key = model_name
    rag_data = rag_cache.get(rag_key)
    rag = rag_data["rag"] if rag_data else None
    
    # 使用rollout参数覆盖max_iter（如果提供）
    max_iter = request.rollout if request.rollout is not None else request.max_iter
    
    # 创建Explorer
    explorer = LLMExplorer_Socrates(
        llm=default_llm,
        api_llm=default_llm,
        api_llm2=default_llm2,
        max_iter=max_iter,
        use_diversity_fusion=request.use_diversity_fusion,
        rag=rag,
        save_dataset_path=request.save_dataset_path,
        elo_data_file=request.elo_data_file,
        state_save_path=f"rollout_states-arc-agi2-{model_name}",
        enable_state_tracking=True,
        use_expert_prompt=True,
        enable_elo_weighting=request.enable_elo_weighting
    )
    
    return explorer

def messages_to_explorer_input(messages: List[Message], domain: str = "通用") -> Dict[str, Any]:
    """将OpenAI格式的messages转换为Explorer输入格式"""
    # 提取system和user消息
    system_content = "You are a helpful assistant."
    user_content = ""
    
    for msg in messages:
        if msg.role == "system":
            system_content = msg.content
        elif msg.role == "user":
            user_content += msg.content + "\n"
    
    user_content = user_content.strip()
    
    # 构建Explorer输入格式
    explorer_input = {
        "prompt": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "domain": domain,
        "uid": str(uuid.uuid4())
    }
    
    return explorer_input

def count_tokens(text: str) -> int:
    """简单的token计数"""
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except:
            pass
    # 回退到字符计数的近似
    return len(text.split())

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LLaMA-Berry Arena API Server",
        "model": config.MODEL_NAME,
        "version": "1.0.0"
    }

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    """列出可用模型"""
    # 动态生成支持的模型列表
    methods = ["EmMcts", "bestofN"]
    rollouts = [1, 2, 4, 8]
    base_model = "gpt-oss-120b-low"

    models = []
    for method in methods:
        for rollout in rollouts:
            model_name = f"fresh-vegetables-{method}-rollout{rollout}-{base_model}"
            models.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llama-berry-arena",
                "permission": [],
                "root": model_name,
                "parent": None,
            })

    return {
        "object": "list",
        "data": models
    }

@app.get("/v1/tokenizer_info")
@app.get("/tokenizer_info")
async def tokenizer_info():
    """返回tokenizer信息"""
    await init_tokenizer()

    if tokenizer is None:
        return {
            "tokenizer_class": "unknown",
            "vocab_size": 32000,  # 默认值
            "eos_token_id": 2,
            "bos_token_id": 1,
            "pad_token_id": 0,
        }

    try:
        return {
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": tokenizer.vocab_size,
            "eos_token_id": tokenizer.eos_token_id if hasattr(tokenizer, "eos_token_id") else None,
            "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, "bos_token_id") else None,
            "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None,
            "eos_token": tokenizer.eos_token if hasattr(tokenizer, "eos_token") else None,
            "bos_token": tokenizer.bos_token if hasattr(tokenizer, "bos_token") else None,
            "pad_token": tokenizer.pad_token if hasattr(tokenizer, "pad_token") else None,
        }
    except Exception as e:
        logger.warning(f"Error getting tokenizer info: {e}")
        return {
            "tokenizer_class": "unknown",
            "vocab_size": 32000,
            "error": str(e)
        }

class TokenizeRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # 允许额外字段

    text: Union[str, List[str]] = Field(..., description="要tokenize的文本")
    add_special_tokens: Optional[bool] = Field(True, description="是否添加特殊token")

class TokenizeResponse(BaseModel):
    tokens: Union[List[int], List[List[int]]] = Field(..., description="Token IDs")
    count: Union[int, List[int]] = Field(..., description="Token数量")

@app.post("/v1/tokenize")
@app.post("/tokenize")
async def tokenize_text(request: Request):
    """Tokenize文本 - 灵活处理不同格式的请求"""
    await init_tokenizer()

    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not available")

    try:
        # 获取原始请求体
        body = await request.json()

        # 提取文本（支持多种字段名）
        text = body.get("text") or body.get("prompt") or body.get("input")
        if text is None:
            raise HTTPException(status_code=422, detail="Missing text/prompt/input field")

        add_special_tokens = body.get("add_special_tokens", True)

        if isinstance(text, str):
            # 单个文本
            token_ids = tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens
            )
            return {
                "tokens": token_ids,
                "count": len(token_ids)
            }
        else:
            # 批量文本
            all_token_ids = []
            counts = []
            for t in text:
                token_ids = tokenizer.encode(
                    t,
                    add_special_tokens=add_special_tokens
                )
                all_token_ids.append(token_ids)
                counts.append(len(token_ids))

            return {
                "tokens": all_token_ids,
                "count": counts
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tokenizing text: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Tokenization error: {str(e)}")
    
@app.post("/v1/completions")
@app.post("/completions")
async def completions(request: Request):
    """OpenAI兼容的文本补全接口（用于 lm-eval），支持 Best-of-N 开关"""
    start_time = time.time()
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    try:
        body = await request.json()
        logger.info(f"Received completions request: {body}")

        model = body.get("model")
        prompt = body.get("prompt") or body.get("messages", "")

        # 验证并解析模型名称
        if not is_valid_model_name(model):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model} not found. Expected format: fresh-vegetables-{{EmMcts|bestofN}}-rollout{{N}}-{{base_model}}"
            )

        model_info = parse_model_name(model)
        logger.info(f"Parsed model info: {model_info}")

        # 处理 prompt 格式
        if isinstance(prompt, list):
            if len(prompt) == 0:
                raise HTTPException(status_code=400, detail="Prompt cannot be empty")
            if isinstance(prompt[0], list):
                prompt_text = prompt[0][0] if prompt[0] else ""
            else:
                prompt_text = prompt[0] if isinstance(prompt[0], str) else str(prompt[0])
        else:
            prompt_text = str(prompt)

        if not prompt_text:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # 提取参数
        max_tokens = body.get("max_tokens", 32768)
        temperature = body.get("temperature", 0.0)
        stop = body.get("stop", [])
        use_best_of_n = body.get("use_best_of_n", False)
        best_of_n = body.get("best_of_n", 8)

        logger.info(f"Processing completions request {request_id} with mode={'Best-of-N' if use_best_of_n else 'Arena'}")

        if use_best_of_n:
            # ========== Best-of-N 模式 ==========
            logger.info(f"Using Best-of-N with n={best_of_n}")
            generator = await create_best_of_n_generator(ChatCompletionRequest(
                model=model,
                messages=[Message(role="user", content=prompt_text)],
                best_of_n=best_of_n
            ))
            result = await generator.generate_best_of_n("You are a helpful assistant.", prompt_text)

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            best_content = result["chosen"]["content"]
            token_usage = result.get("token_usage", {})
            prompt_tokens = token_usage.get("total_prompt_tokens", count_tokens(prompt_text))
            completion_tokens = token_usage.get("total_completion_tokens", count_tokens(best_content))
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
            total_api_calls = token_usage.get("total_api_calls", 0)

        else:
            # ========== Arena 模式 ==========
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
            messages_obj = [Message(role=msg["role"], content=msg["content"]) for msg in messages]

            # 使用从模型名称解析的 rollout 值
            max_iter = model_info["rollout"]

            internal_request = ChatCompletionRequest(
                model=model,
                messages=messages_obj,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                max_iter=max_iter,
                rollout=model_info["rollout"],
                use_diversity_fusion=False,
                domain="通用"
            )

            explorer_input = messages_to_explorer_input(messages_obj, "通用")

            MAX_RETRIES = 3
            results = None
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    # 传递完整的模型名称以便使用对应的 RAG 库
                    explorer = await create_llm_explorer(internal_request, model_info["full_name"])
                    results = await explorer.main_loop(explorer_input)

                    if results == "<|_error_|>" or not results or len(results) == 0:
                        last_error = "Invalid result"
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        else:
                            break

                    break

                except Exception as e:
                    last_error = str(e)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Inference failed after {MAX_RETRIES} attempts: {last_error}"
                        )

            if not results or results == "<|_error_|>" or len(results) == 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed after {MAX_RETRIES} attempts: {last_error}"
                )

            result = results[0]
            best_content = result["chosen"][0]["content"] if "chosen" in result and result["chosen"] else ""

            # Token usage
            token_usage = result.get("token_usage", {})
            prompt_tokens = token_usage.get("total_prompt_tokens", count_tokens(prompt_text))
            completion_tokens = token_usage.get("total_completion_tokens", count_tokens(best_content))
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
            total_api_calls = token_usage.get("total_api_calls", 0)

        # ========== 构建 Completions 响应 ==========
        response = {
            "id": request_id,
            "object": "text_completion",
            "created": int(start_time),
            "model": model,
            "choices": [
                {
                    "text": best_content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }

        logger.info(f"Completions request {request_id} completed in {time.time() - start_time:.2f}s")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in completions request {request_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI兼容的聊天补全接口，支持 Arena 模式和 Best-of-N 模式"""
    start_time = time.time()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    try:
        # 获取原始请求体
        body = await request.json()
        # 提取必需字段
        model = body.get("model")
        messages = body.get("messages", [])

        # 验证并解析模型名称
        if not is_valid_model_name(model):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model} not found. Expected format: fresh-vegetables-{{EmMcts|bestofN}}-rollout{{N}}-{{base_model}}"
            )

        model_info = parse_model_name(model)
        logger.info(f"Parsed model info: {model_info}")

        # 验证messages
        if not messages:
            raise HTTPException(status_code=400, detail="Messages cannot be empty")

        # 提取可选参数
        temperature = body.get("temperature", 0.9)
        max_tokens = body.get("max_tokens") or body.get("max_gen_toks")
        stream = body.get("stream", False)

        # 自定义参数 - 优先使用模型名称中的 rollout，其次使用请求参数
        max_iter = body.get("rollout") if body.get("rollout") is not None else model_info["rollout"]
        use_diversity_fusion = body.get("use_diversity_fusion", False)
        domain = body.get("domain", "通用")
        elo_data_file = body.get("elo_data_file")
        enable_elo_weighting = body.get("enable_elo_weighting", True)
        use_best_of_n = body.get("use_best_of_n", False)
        best_of_n = body.get("best_of_n", 8)

        # 转换为内部格式
        messages_obj = [Message(role=msg["role"], content=msg["content"]) for msg in messages]

        logger.info(f"Processing request {request_id} with mode={'Best-of-N' if use_best_of_n else 'Arena'}")

        # 提取 system 和 user 内容（用于 Best-of-N）
        system_content = "You are a helpful assistant."
        user_content = ""
        for msg in messages_obj:
            if msg.role == "system":
                system_content = msg.content
            elif msg.role == "user":
                user_content += msg.content + "\n"
        user_content = user_content.strip()

        if use_best_of_n:
            # ========== Best-of-N 模式 ==========
            logger.info(f"Using Best-of-N with n={best_of_n}")
            generator = await create_best_of_n_generator(ChatCompletionRequest(
                model=model,
                messages=messages_obj,
                best_of_n=best_of_n
            ))
            result = await generator.generate_best_of_n(system_content, user_content)

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            best_content = result["chosen"]["content"]
            token_usage = result.get("token_usage", {})
            actual_prompt_tokens = token_usage.get("total_prompt_tokens", 0)
            actual_completion_tokens = token_usage.get("total_completion_tokens", 0)
            actual_total_tokens = token_usage.get("total_tokens", 0)
            total_api_calls = token_usage.get("total_api_calls", 0)

            # Fallback to estimation if no token usage
            if actual_total_tokens == 0:
                prompt_text = system_content + "\n" + user_content
                actual_prompt_tokens = count_tokens(prompt_text)
                actual_completion_tokens = count_tokens(best_content)
                actual_total_tokens = actual_prompt_tokens + actual_completion_tokens

            reasoning_tokens = 0  # Best-of-N has no reasoning tokens

        else:
            # ========== LLaMA-Berry Arena 模式 ==========
            internal_request = ChatCompletionRequest(
                model=model,
                messages=messages_obj,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                max_iter=max_iter,
                rollout=model_info["rollout"],
                use_diversity_fusion=use_diversity_fusion,
                domain=domain,
                elo_data_file=elo_data_file,
                enable_elo_weighting=enable_elo_weighting
            )

            explorer_input = messages_to_explorer_input(messages_obj, domain)

            MAX_RETRIES = 3
            results = None
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    logger.info(f"Starting Arena inference for request {request_id} (attempt {attempt}/{MAX_RETRIES})")
                    # 使用完整的模型名称以便使用对应的 RAG 库
                    explorer = await create_llm_explorer(internal_request, model_info["full_name"])
                    results = await explorer.main_loop(explorer_input)

                    if results == "<|_error_|>" or not results or len(results) == 0:
                        last_error = "Invalid or empty result"
                        if attempt < MAX_RETRIES:
                            await asyncio.sleep(1)
                            continue
                        else:
                            break

                    logger.info(f"Request {request_id} succeeded on attempt {attempt}")
                    break

                except Exception as e:
                    last_error = str(e)
                    logger.error(f"Arena attempt {attempt} failed: {e}")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(1)
                        continue
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Inference failed after {MAX_RETRIES} attempts: {last_error}"
                        )

            if not results or results == "<|_error_|>" or len(results) == 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate valid response after {MAX_RETRIES} attempts: {last_error}"
                )

            result = results[0]
            best_content = result["chosen"][0]["content"] if "chosen" in result and result["chosen"] else "No valid response generated"

            # Extract token usage from Arena
            token_usage = result.get("token_usage", {})
            actual_prompt_tokens = token_usage.get("total_prompt_tokens", 0)
            actual_reasoning_tokens = token_usage.get("total_reasoning_tokens", 0)
            actual_completion_tokens = token_usage.get("total_completion_tokens", 0)
            actual_total_tokens = token_usage.get("total_tokens", 0)
            total_api_calls = token_usage.get("total_api_calls", 0)

            # Fallback estimation
            if actual_total_tokens == 0:
                prompt_text = "\n".join([msg.content for msg in messages_obj])
                actual_prompt_tokens = count_tokens(prompt_text)
                actual_completion_tokens = count_tokens(best_content)
                actual_total_tokens = actual_prompt_tokens + actual_completion_tokens
                actual_reasoning_tokens = 0

            reasoning_tokens = actual_reasoning_tokens

        # ========== 构建统一响应 ==========
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(start_time),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": best_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": actual_prompt_tokens,
                "completion_tokens": actual_completion_tokens,
                "total_tokens": actual_total_tokens,
                "reasoning_tokens": reasoning_tokens
            },
            "arena_metadata": {
                "mode": "best_of_n" if use_best_of_n else "arena",
                "inference_time_sec": round(time.time() - start_time, 2),
                "best_of_n": best_of_n if use_best_of_n else None,
                "max_iter": max_iter if not use_best_of_n else None,
                "use_diversity_fusion": use_diversity_fusion if not use_best_of_n else None,
                "domain": domain,
                "token_usage_details": {
                    "total_prompt_tokens": actual_prompt_tokens,
                    "total_completion_tokens": actual_completion_tokens,
                    "total_reasoning_tokens": reasoning_tokens,
                    "total_tokens": actual_total_tokens,
                    "total_api_calls": total_api_calls
                }
            }
        }

        logger.info(f"Request {request_id} completed in {time.time() - start_time:.2f}s ({'Best-of-N' if use_best_of_n else 'Arena'})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "llama_berry_arena_server:app",
        host="0.0.0.0",
        port=7866,
        log_level="info",
        reload=False
    )