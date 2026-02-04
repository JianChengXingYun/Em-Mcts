


import asyncio
import copy
from datetime import datetime
import json
import time
import aiofiles
import numpy as np
from openai import AsyncOpenAI, OpenAI
import traceback

class LLM_Core:
    def __init__(self, tokenizer=None, use_async=True, api_model="",base_url="http://0.0.0.0:6001/v1", api_key='EMPTY', task="chat"):
        self.use_async = use_async
        self.api_key = api_key
        self.base_url = base_url
        self.api_model = api_model
        self.tokenizer = tokenizer
        self.task = task
        self.timeout=360
        self.extra_body = {
            'repetition_penalty': 1.05,
                    }
        api_model = api_model.split("/")[-1]
        if use_async==True:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
    async def async_embedding(self, data):
        data_copy = copy.deepcopy(data)
        responses = await self.client.embeddings.create(
            input=data_copy["messages"],
            model=data_copy["model"],
        )
        return [data.embedding for data in responses.data]
    async def async_reward(self, data):
        data_copy = copy.deepcopy(data)
        responses = await self.client.embeddings.create(
            input=data_copy["messages"],
            model=data_copy["model"],
        )
        return f"{str(responses.data[-1].embedding[-1])}"
        
    async def async_model(self, data, tools=None, tool_choice=None):
        data_copy = copy.deepcopy(data)
        
        # 提取extra_headers
        extra_headers = data.get("extra_headers", None)
        # 1. 首先构造基础参数字典
        create_kwargs = {
            "model": data["model"],
            "messages": data["messages"],
            "temperature": data.get("temperature", 0.95),
            "top_p": data.get("top_p", 0.9),
            "extra_body": data.get("extra_body", {}),
            "stream": data.get("stream", False),
            "tools": tools,
            "tool_choice": tool_choice,
            "extra_headers": extra_headers,
        }
        extra_body = data.get("extra_body", {})
     
        if data["model"] != "gpt-5-2025-08-07":
            create_kwargs["top_p"] = data.get("top_p", 0.35)
        if data["model"] == "gpt-5.2":
            del create_kwargs["extra_body"]
        if data["model"] == "gpt-5.1-2025-11-13":
            del create_kwargs["extra_body"]
        # 3. 使用 ** 解包进行调用
        completion = await self.client.chat.completions.create(**create_kwargs)
        count = 0
        if data_copy.get("stream", False):
            async for chunk in completion:
                    yield chunk
            yield "data: [DONE]\n\n"
            
        else:
            chunk = completion
            
            tool_calls_data = None
            if chunk.choices[0].message.tool_calls:
                # 将 tool_calls 对象转换为字典列表
                tool_calls_data = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            # **关键**: 直接解析 arguments 字符串为字典
                            # 这样 json.dumps 就不会重复转义它
                            "arguments": json.loads(call.function.arguments) 
                        }
                    } 
                    for call in chunk.choices[0].message.tool_calls
                ]
            yield chunk
    # --- 步骤 2: 使用 parse 直接获取结构化对象 ---
    async def get_critique_object(self, data, struct):
        messages = data["messages"]
        temperature = data.get("temperature", 0)
        
        try:
            # 调用新版 parse 方法，直接指定 response_format 为 Pydantic 模型
            completion = await self.client.beta.chat.completions.parse(
                model=self.api_model,
                messages=messages,
                response_format=struct,
                temperature=temperature,
            )
            
            # 获取 Pydantic 对象
            parsed_result = completion.choices[0].message.parsed

            return parsed_result

        except Exception as e:
            print(f"请求或解析失败: {e}")
            raise e

    async def receive_data_structural(self, data, max_retries=3, initial_delay=1, struct=None):
        """异步接收GPT模型数据并在出错时重新请求。"""
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay
        
        while attempt < max_retries:
            try:
                critique_data = await self.get_critique_object(data_copy, struct)
                #critique_data.model_dump_json(indent=2)
                return critique_data
                # # 请求成功，返回输出
                # return output
            except Exception as e:
                error_message = traceback.format_exc()
                print(f"接收数据时出错:\n{error_message}")
                attempt += 1
                if attempt < max_retries:
                    print(f"重试 {attempt}/{max_retries} 次后请求...")
                    await asyncio.sleep(delay)  # 指数退避延迟
                    delay *= 1.1  # 增加延迟时间
                else:
                    print(f"超过最大重试次数，终止请求。{output}")
                    break
        return output if output else "<|_error_|>"
   
    async def receive_data(self, data):
        data_copy = copy.deepcopy(data)
        output = ""
        async for chunk in self.async_model(data=data_copy):
            try:
                if '[DONE]' in chunk:
                    break
                if data_copy.get('stream', False):
                    output += chunk.choices[0].delta.content
                else:
                    output += chunk.choices[0].message.content
                        
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"JSON解码或解析错误: {chunk}, 错误: {e}")
                continue
        return output
    
    async def receive_data_GPT(self, data, max_retries=6, initial_delay=1):
        """异步接收GPT模型数据并在出错时重新请求。"""
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay
        
        while attempt < max_retries:
            #print("发送消息:", data_copy["messages"])
            try:
                # 等待一段时间，模拟异步请求
                await asyncio.sleep(0.1)
                async for chunk in self.async_model(data=data_copy):
                    json_string = chunk
                    try:
                        chunk_data = json.loads(json_string)
                    except json.JSONDecodeError:
                        continue
                    if data_copy.get('stream', False):
                        output += chunk_data['choices'][0]['delta']['content']
                    else:
                        output += chunk_data['choices'][0]['message']['content']
                # 请求成功，返回输出
                return output
            except Exception as e:
                print(f"接收数据时出错: {e}")
                attempt += 1
                if attempt < max_retries:
                    print(f"重试 {attempt}/{max_retries} 次后请求...")
                    await asyncio.sleep(delay)  # 指数退避延迟
                    delay *= 1.1  # 增加延迟时间
                else:
                    print("超过最大重试次数，终止请求。")
                    break
        #print("最终输出:", output)
        # 如果达到最大重试次数仍然失败，返回错误消息或空输出
        return output if output else "请求失败，超过最大重试次数。"
   