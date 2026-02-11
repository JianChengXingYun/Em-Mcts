import asyncio
import copy
import json
import logging
import re
import traceback
from typing import Optional

from .LLM_Core import LLM_Core
from .Prompter import Prompter
from .Tools import Tools

class Process_Controller:
    def __init__(self, llm: LLM_Core, tools: Tools):
        """
        Inicializa el controlador de procesos con los componentes necesarios.
        Se han eliminado todos los atributos y componentes no utilizados por LLMExplorer_Socrates.
        """
        self.llm = llm
        self.tools = tools
        self.prompter = Prompter()
        self.logger = logging.getLogger(__name__)
        
        # Plantilla de datos base para las llamadas a la API del LLM.
        self.data_template = {
            "model": self.llm.api_model,
            "messages": [],
            "temperature": 0.95,
            "top_p": 0.9,
            "extra_body": {},
            "stream": False,
        }

    async def receive_data(self, llm: LLM_Core, data, max_retries=3, initial_delay=1, tools=None, tool_choice=None):
        """异步接收GPT模型数据并在出错时重新请求。"""
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay
        while attempt < max_retries:
            try:
                # 等待一段时间，模拟异步请求
                #await asyncio.sleep(0.2)
                async for chunk in llm.async_model(data=data_copy, tools=tools, tool_choice=tool_choice):
                    try:
                        if chunk.choices[0].message.tool_calls:
                            return chunk.choices[0].message.tool_calls
                        if data_copy.get('stream', False):
                            output += chunk.choices[0].delta.content
                        else:
                            output += chunk.choices[0].message.content
                    except:
                        continue
                # 请求成功，返回输出
                return output
            except Exception as e:
                error_message = traceback.format_exc()
                print(f"接收数据时出错{llm.api_model}:\n{error_message}")
                attempt += 1
                if attempt < max_retries:
                    print(f"重试 {attempt}/{max_retries} 次后请求...")
                    await asyncio.sleep(delay)  # 指数退避延迟
                    delay *= 1.1  # 增加延迟时间
                else:
                    print(f"超过最大重试次数，终止请求。{output}")
                    break
        # 如果达到最大重试次数仍然失败，返回错误消息或空输出
        return output if output else "<|_error_|>"
        
    # --- 步骤 2: 使用 parse 直接获取结构化对象 ---
    async def get_critique_object(self, data, llm, struct):
        messages = data["messages"]
        temperature = data["temperature"]
        #top_p = data["top_p"]
        # 调用新版 parse 方法，直接指定 response_format 为 Pydantic 模型
        completion = await llm.client.beta.chat.completions.parse(
            model=llm.api_model,
            messages=messages,
            response_format=struct,
            temperature=temperature,
            extra_body=data["extra_body"],
            timeout=data.get("timeout", 360000),
            #top_p=top_p,
        )

        # [修改] 同时返回completion对象和解析后的结构
        return completion.choices[0].message.parsed, completion
    def check_contains_sensitive(self, string_a: str) -> bool:
        """
        Comprueba si una cadena de texto contiene alguna de las palabras sensibles definidas en las herramientas.
        """
        string_list = self.tools.sensitive_words
        for item in string_list:
            if item in string_a:
                return True
        return False

    async def receive_data_thinking(self, llm: LLM_Core, data, max_retries=1, initial_delay=1, tools=None, tool_choice=None, thinking=False):
        """异步接收GPT模型数据并在出错时重新请求。"""
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay
        think = ""
        completion = None  # [新增] 保存completion对象
        while attempt < max_retries:
            try:
                # 等待一段时间，模拟异步请求
                #await asyncio.sleep(0.2)
                async for chunk in llm.async_model(data=data_copy, tools=tools, tool_choice=tool_choice):
                    # if chunk.choices[0].message.tool_calls:
                    #     print("use tools and tool call")
                    #     return chunk.choices[0].message.tool_calls
                    if data_copy.get('stream', False):
                        output += chunk.choices[0].delta.content
                    else:
                        output += chunk.choices[0].message.content
                        completion = chunk  # [新增] 在非流式模式下保存completion对象
                    if thinking == True:
                        think = getattr(chunk.choices[0].message, 'reasoning_content', None) or \
                            getattr(chunk.choices[0].message, 'thinking', None) or ""

                # 请求成功，返回输出和completion对象
                return output, think, completion
            except Exception as e:
                error_message = traceback.format_exc()
                print(f"接收数据时出错{llm.api_model}:\n{error_message}")
                attempt += 1
                if attempt < max_retries:
                    print(f"重试 {attempt}/{max_retries} 次后请求...")
                    await asyncio.sleep(delay)  # 指数退避延迟
                    delay *= 1.01  # 增加延迟时间
                else:
                    print(f"超过最大重试次数，终止请求。{output}")
                    break
        # 如果达到最大重试次数仍然失败，返回错误消息或空输出
        return "<|_error_|>", "", ""

    async def receive_data_rw(self, llm: LLM_Core, data, max_retries=1, initial_delay=1):
        """异步接收GPT模型数据并在出错时重新请求。"""
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay
        reward = 0
        while attempt < max_retries:
            try:
                # 等待一段时间，模拟异步请求
                #await asyncio.sleep(0.2)
                async for chunk in self.llm.async_logprob(data=data_copy):
                    json_string = chunk
                    chunk_data = json.loads(json_string)
                    output += chunk_data['response']
                    reward += float(chunk_data['reward'])
                # 请求成功，返回输出
                return output, reward
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
        # 如果达到最大重试次数仍然失败，返回错误消息或空输出
        return "<|_error_|>", 0


    async def receive_data_structural(self, llm: LLM_Core, data, max_retries=3, initial_delay=1, struct=None):
        """异步接收GPT模型数据并在出错时重新请求。"""
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay

        while attempt < max_retries:
            try:
                critique_data, completion = await self.get_critique_object(data_copy, llm, struct)
                #critique_data.model_dump_json(indent=2)
                return critique_data, completion  # [修改] 同时返回completion对象
                # # 请求成功，返回输出
                # return output
            except Exception as e:
                error_message = traceback.format_exc()
                print(f"接收数据时出错{llm.api_model}:\n{error_message}")
                attempt += 1
                if attempt < max_retries:
                    print(f"重试 {attempt}/{max_retries} 次后请求...")
                    await asyncio.sleep(delay)  # 指数退避延迟
                    delay *= 1.1  # 增加延迟时间
                else:
                    print(f"超过最大重试次数，终止请求。{output}")
                    break
        return (output if output else "<|_error_|>"), None

    async def Generate_Response(self, choose_llm, data_template, max_retries=6, Reward=False, pattern = None, think=False):
        completion = None  # [新增] 用于存储completion对象
        for attempt in range(int(max_retries)):
            data_template2 = copy.deepcopy(data_template)
            if Reward:
                Example_Response,reward = await self.receive_data_rw(choose_llm, data_template2)
            elif think:
                Example_Response, thinking, completion = await self.receive_data_thinking(choose_llm, data_template2, max_retries=max_retries, thinking=think)
            else:
                Example_Response = await self.receive_data(choose_llm, data_template2)

            if self.check_contains_sensitive(Example_Response):
                print(f"发现存在敏感词。尝试第 {attempt} 次")
                await asyncio.sleep(0.1)
                continue

            if pattern:
                match = re.search(pattern, Example_Response, re.DOTALL)
                if not match:
                    print(f"尝试第 {attempt + 1} 次：未找到回答部分")
                    continue
                Example_Response = match.group(1).strip()
            else:
                Example_Response = Example_Response
            if Reward:
                return Example_Response,reward
            elif think:
                return Example_Response,thinking,completion
            else:
                return Example_Response,"",None
        return "<|_error_|>",""
    