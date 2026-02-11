from collections import defaultdict
import json
import os
import random
import re
import uuid
class Tools:
    def __init__(self, filename, tokenizer):
        self.filename = filename
        self.tokenizer = tokenizer
        self.filename2 = ""
        self.filename3 = ""
        self.input_data = []
        self.sensitive_words = []
        
    def deduplicate_by_context(self, data):
        unique_contexts = {}
        deduplicated_data = []
        for item in data:
            line = item
            if not line:  # 跳过空行
                continue
            try:
                if not isinstance(item, dict):
                    input_data = json.loads(line)
                else:
                    input_data = item
            except json.JSONDecodeError as e:
                print(f"JSON解析失败，跳过该行：{line}\n错误：{e}")
                continue
            if not input_data["chosen"]:
                continue
            # 遍历可能的字段
            for field in ['context', 'instruction', 'prompt','chosen', 'output']:
                if field in input_data and input_data[field] and input_data[field]!=[]:
                    break
            else:
                # 如果没有找到任何指定的字段，则跳过该条数据
                continue

            # 如果字段是列表，则取列表中最后一个元素的 "content"
            if isinstance(input_data[field], list):
                # 确保列表不为空且每个元素是字典并含有 "content"
                if input_data[field] and isinstance(input_data[field][-1], dict) and "content" in input_data[field][-1]:
                    context_str = input_data[field][-1]["content"]
                else:
                    continue
            else:
                context_str = input_data[field]
            # 添加去重判断
            if context_str and context_str not in unique_contexts:
                unique_contexts[context_str] = True
                deduplicated_data.append(input_data)

        return deduplicated_data

    
    def process_context(self, data):
        out_list = []
        for item in data:
            input_data = json.loads(item.strip())
            processed_data = {}

            context_set = False
            for field in ['label','context', 'prompt', 'instruction', "Counseling_Report"]:
                if field in input_data and input_data[field]:
                    if isinstance(input_data[field], list):
                        input_data[field] = input_data[field][-1]["content"]
                    processed_data['context'] = input_data[field]
                    context_set = True
                    break

            if not context_set:
                try:
                    for key, value in input_data.items():
                        if isinstance(value, list) and value:
                            processed_data['context'] = self.taken_label(value)
                            context_set = True
                            break
                        elif isinstance(value, str) and value:
                            processed_data['context'] = value
                            context_set = True
                            break
                except:
                    print(input_data)
                    

            # 将剩余的字段添加到 processed_data 中
            for key, value in input_data.items():
                if key not in processed_data:
                    processed_data[key] = value

            out_list.append(processed_data)

        return out_list
    
    def Read_Document_01(self):
        self.input_data = []
        existing_contexts = []
        skipped_count = 0
        processed_count = 0
        # Check if file exists and read the data. If not, start from beginning.
        if os.path.exists(self.filename2):
            print("读文件：",self.filename2)
            with open(self.filename2, 'r', encoding='utf-8') as file:
                existing_data = file.readlines()
                processed_count = len(existing_data)
                for item in existing_data:
                    if not isinstance(item, dict):
                        input_data = json.loads(item.strip())
                    else:
                        input_data = item
                    if "uid" in input_data:
                        existing_contexts.append(input_data["uid"])
                    elif len(input_data["prompt"]) >= 2:
                        existing_contexts.append(input_data["prompt"][-1]["content"])
                    elif "chosen" in input_data:
                        concatenated_content = "\n".join([item["content"] for item in input_data["chosen"]])
                        existing_contexts.append(concatenated_content)
                    elif "instruction" in input_data:
                        existing_contexts.append(input_data["instruction"][-1]["content"])

        with open(self.filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
            random.shuffle(lines)
            total_lines = len(lines)
            for item in lines:
                if not isinstance(item, dict):
                    input_data = json.loads(item.strip())
                else:
                    input_data = item
                if "uid" not in input_data:
                    uid1 = uuid.uuid4()
                    uid_str = str(uid1)
                    input_data["uid"] = uid_str
                elif len(input_data["prompt"]) >= 2:
                    if input_data["prompt"][-1]["content"] in existing_contexts:
                        skipped_count += 1
                        continue
                elif "chosen" in input_data:
                    concatenated_content = "\n".join([item["content"] for item in input_data["chosen"]])
                    if concatenated_content in existing_contexts:
                        skipped_count += 1
                        continue
                elif "instruction" in input_data:
                    if input_data["instruction"][-1]["content"] in existing_contexts:
                        skipped_count += 1
                        continue
                self.input_data.append(input_data)
        processed_count = len(existing_contexts)
        print(f"total_lines count: {total_lines}")
        print(f"Skipped count: {skipped_count}")
        print(f"input_data length: {len(self.input_data)}")
        return self.input_data, total_lines, processed_count