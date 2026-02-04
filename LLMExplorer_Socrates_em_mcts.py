import ast
import asyncio
from collections import defaultdict
import copy
from datetime import datetime
import json
import math
import random
import re
import os
from typing import Dict, List, Literal
import numpy as np
from pyvis.network import Network
from Swimming_Pool_Async.LLM_Core import LLM_Core
from Swimming_Pool_Async.Prompter import Prompter
from Swimming_Pool_Async.Tools import Tools
from Swimming_Pool_Async.Process_Controller import Process_Controller
from transformers import AutoTokenizer
from Swimming_Pool_Async.simple_rag import AsyncFaissRAG
import uuid
import logging
from pydantic import BaseModel, Field

class LLMExplorer_Socrates:
    def __init__( 
                  self,
                  llm: LLM_Core,
                  api_llm: LLM_Core = None,
                  api_llm2: LLM_Core = None,
                  initial_threshold=0.3,
                  current_score=None, 
                  max_iter=8, 
                  rag: AsyncFaissRAG = None, 
                  rag_j: AsyncFaissRAG = None, 
                  use_diversity_fusion: bool = False, 
                  save_dataset_path: str = "judge_training_data.jsonl", 
                  model_configs: dict = None,
                  elo_data_file: str = "judge_training_data.jsonl",
                  enable_elo_weighting: bool = True,
                  # [新增] 状态记录和可视化功能
                  enable_state_tracking: bool = False,
                  use_expert_prompt: bool = False,
                  state_save_path: str = "rollout_states",
                  auto_save_interval: int = 1,  # 每N次迭代自动保存一次
                  enable_visualization: bool = True
                    ):
        # 初始化组件
        self.llm = llm
        self.api_llm = api_llm if api_llm else llm
        self.api_llm2 = api_llm2 if api_llm2 else llm
        self.current_score = current_score
        self.use_diversity_fusion = use_diversity_fusion
        self.prompter = Prompter()
        self.tools = Tools(filename="",tokenizer = "")
        self.rag = rag
        self.rag_j = rag_j
        self.process_controller = Process_Controller(llm = self.llm, tools = self.tools)
        self.logger = logging.getLogger(self.__class__.__name__)
        # [新增] 保存路径
        self.save_dataset_path = save_dataset_path
        # 确保目录存在
        if os.path.dirname(self.save_dataset_path):
            os.makedirs(os.path.dirname(self.save_dataset_path), exist_ok=True)
        
        # [新增] 模型配置列表
        self.model_configs = model_configs or self._get_default_model_configs()
        self.model_cache = {}  # 缓存已创建的LLM实例
        
        # [新增] ELO评分系统相关
        self.elo_data_file = elo_data_file
        self.enable_elo_weighting = enable_elo_weighting
        self.current_elo_ratings = {}  # 当前ELO分数
        self.model_battle_stats = {}  # 模型对战统计
        # 初始化参数
        self.threshold = initial_threshold
        self.max_iter = max_iter
        # 初始化数据结构
        self._initialize_data_structures(enable_state_tracking,
                                         state_save_path,
                                         auto_save_interval,
                                         enable_visualization,
                                         use_expert_prompt)
        
    def _initialize_data_structures(self, 
                                    enable_state_tracking,
                                    state_save_path,
                                    auto_save_interval,
                                    enable_visualization,
                                    use_expert_prompt):
        """初始化或重置所有数据结构。"""
        self.to_explore = []
        self.to_explore_reward = {} # 存储融合后的奖励 Q(s)
        self.to_explore_judgeA_reward = {}
        self.Meta_Prompt_bank = {}
        self.visit_counts = {}
        self.history_bank = {}
        self.thinks_bank = {}
        self.think = False
        self.ucb_bank = {}
        self.fathers = {}
        self.evaluations_bank = {}
        self.childs = {}
        self.reward_imp_bank = {}
        self.answers_list = []
        self.max_rejected_usage = 1
        self.use_expert_prompt = use_expert_prompt
        self.system = None
        self.default_system = None
        self.query = None
        self.use_meta_prompt = True
        self.use_enhancce = False
        self.context = ""
        self.standard_criteria = self.prompter.default_standard_criteriaV2
        self.uid = ""
        self.evolved_meta_prompt = ""
        self.evolved_judgeA_prompt = self.prompter.Self_Critique_Judge_exp_default_system
        self.node_meta_prompts = {}
        self.judgeA_meta_prompts = {}
        self.score_before_judgeA = 0
        self.early_stop = False
        
        self.node_model_mapping = {}
        
        self.model_usage_queue = []  # 模型使用队列：先遍历所有模型，再随机选择
        self.current_rollout_models = []  # 当前rollout周期中已使用的模型
        self.rollout_round = 0  # 当前rollout轮次
        
        self.pairwise_relations = [] 
        # 缓存 EBC 计算出的全局分数
        self.ebc_global_scores = {}
        # 缓存局部价值
        self.local_values = {}
        self.Q_values = {}
        self.ebc_alpha = 0.9
        self.gamma = 0.1
        # 杂交融合相关配置
        self.max_merges = 3 
        self.merge_trigger_iters = {5, 10, 15}
        self.merge_counter = 0
        self.max_reward = 10
        self.max_expand = 2
        self.class_tag = ""
        self.category = ""
        self.iter = 0

        self.total_prompt_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_api_calls = 0

        self.current_elo_ratings = {"GLOBAL": {}} 
        self.model_battle_stats = {"GLOBAL": {}}  

        self.enable_state_tracking = enable_state_tracking
        self.state_save_path = state_save_path
        self.auto_save_interval = auto_save_interval
        self.enable_visualization = enable_visualization
        
        if self.enable_state_tracking:
            self._initialize_state_tracking()

        # 初始化数据模板
        self.data_template = {
            "model": self.llm.api_model,
            "messages": [],
            "temperature": 0.95,
            "top_p": 0.9,
            "stream": False,
        }
        self.extra_body = {
        }

        self.judge_model_configs = {
            "deepseek/deepseek-v3.1-terminus": {
                "temperature": 0.95,
                "top_p": 0.9,
                "extra_body": {
                "reasoning": {"enabled": False},
                "provider" : {
                    "order": ["novita/fp8"],
                    "allow_fallbacks": False
                            }
                            }
            },
            "gpt-oss-120b": {
                "temperature": 0.95,
                "top_p": 0.9,
                "extra_body": {
                    "include_reasoning": True,  # 显式开启推理
                    "reasoning_effort": "low"   # 设置推理等级为 low
                            }
            },
            "default": {
                "temperature": 0.95,
                "top_p": 0.9,
                "extra_body": {}
            }
        }
    def reset(self):
        """重置所有储存的数据结构以便重新使用。"""
        print("正在重置 LLMExplorer_Socrates 实例的所有数据结构。")
        self._initialize_data_structures()
        if self.enable_state_tracking:
            self._initialize_state_tracking()

    def _record_token_usage(self, completion):
        """
        记录API调用的token使用情况

        Args:
            completion: OpenAI API返回的completion对象，包含usage信息
        """
        if completion and hasattr(completion, 'usage') and completion.usage:
            usage = completion.usage
            self.total_api_calls += 1
            self.total_prompt_tokens += getattr(usage, 'prompt_tokens', 0)
            self.total_completion_tokens += getattr(usage, 'completion_tokens', 0)
            self.total_tokens += getattr(usage, 'total_tokens', 0)

            # 处理reasoning tokens（如果存在）
            if hasattr(usage, 'completion_tokens_details'):
                details = usage.completion_tokens_details
                if details and hasattr(details, 'reasoning_tokens'):
                    self.total_reasoning_tokens += getattr(details, 'reasoning_tokens', 0)

    # ================= [新增] 状态追踪和可视化系统 =================
    
    def _initialize_state_tracking(self):
        """初始化状态追踪系统"""
        import os
        from datetime import datetime
        
        # 创建保存目录
        if not os.path.exists(self.state_save_path):
            os.makedirs(self.state_save_path)
        
        # 生成唯一的会话ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"rollout_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # 状态追踪数据结构
        self.state_history = []  # 存储每次迭代的完整状态
        self.operation_log = []  # 存储所有操作的详细日志
        self.visualization_data = []  # 存储可视化数据
        
        # 文件路径
        self.state_file = os.path.join(self.state_save_path, f"{self.session_id}_state.json")
        self.operation_file = os.path.join(self.state_save_path, f"{self.session_id}_operations.jsonl")
        self.visualization_file = os.path.join(self.state_save_path, f"{self.session_id}_visualization.html")
        
        print(f"[状态追踪] 已初始化，会话ID: {self.session_id}")
        print(f"[状态追踪] 状态文件: {self.state_file}")
        
    def _record_operation(self, operation_type: str, data: dict, iteration: int = None):
        """记录操作到日志"""
        if not self.enable_state_tracking:
            return
            
        from datetime import datetime
        
        operation = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration or self.iter,
            "operation_type": operation_type,
            "data": data
        }
        
        self.operation_log.append(operation)
        
        # 异步写入操作日志文件
        try:
            with open(self.operation_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(operation, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"[警告] 记录操作失败: {e}")
    
    def _capture_state_snapshot(self):
        """捕获当前完整状态快照"""
        if not self.enable_state_tracking:
            return
            
        # 构建状态快照（深拷贝重要数据）
        snapshot = {
            "iteration": self.iter,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            
            # 核心数据结构
            "answers_list": copy.deepcopy(self.answers_list),
            "to_explore": copy.deepcopy(self.to_explore),
            "to_explore_reward": copy.deepcopy(self.to_explore_reward),
            "visit_counts": copy.deepcopy(self.visit_counts),
            "ucb_bank": copy.deepcopy(self.ucb_bank),
            "fathers": copy.deepcopy(self.fathers),
            "childs": copy.deepcopy(self.childs),
            "evaluations_bank": copy.deepcopy(self.evaluations_bank),
            "node_meta_prompts": copy.deepcopy(self.node_meta_prompts),
            
            # EBC和价值计算
            "pairwise_relations": copy.deepcopy(self.pairwise_relations),
            "ebc_global_scores": copy.deepcopy(self.ebc_global_scores),
            "local_values": copy.deepcopy(self.local_values),
            "Q_values": copy.deepcopy(self.Q_values),
            
            # 模型相关
            "node_model_mapping": copy.deepcopy(self.node_model_mapping),
            "current_elo_ratings": copy.deepcopy(self.current_elo_ratings),
            "model_battle_stats": copy.deepcopy(self.model_battle_stats),
            "model_usage_queue": copy.deepcopy(self.model_usage_queue),
            "current_rollout_models": copy.deepcopy(self.current_rollout_models),
            "rollout_round": self.rollout_round,
            
            # 配置参数
            "max_iter": self.max_iter,
            "threshold": self.threshold,
            "use_diversity_fusion": self.use_diversity_fusion,
            "ebc_alpha": self.ebc_alpha,
            "gamma": self.gamma,
            
            # 输入信息
            "query": self.query,
            "system": self.system,
            "domain": getattr(self, 'domain', '通用'),
            "class_tag": self.class_tag
        }
        
        self.state_history.append(snapshot)
        return snapshot
    
    def save_state(self, filepath: str = None):
        """手动保存当前状态到文件"""
        if not self.enable_state_tracking:
            print("[警告] 状态追踪未启用，无法保存状态")
            return False
            
        filepath = filepath or self.state_file
        
        try:
            # 捕获当前状态
            current_state = self._capture_state_snapshot()
            
            # 构建完整保存数据
            save_data = {
                "session_info": {
                    "session_id": self.session_id,
                    "save_timestamp": datetime.now().isoformat(),
                    "total_iterations": self.iter,
                    "state_tracking_enabled": True
                },
                "current_state": current_state,
                "state_history": self.state_history,
                "operation_count": len(self.operation_log)
            }
            
            # 保存到文件
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"[状态追踪] 状态已保存到: {filepath}")
            return True
            
        except Exception as e:
            print(f"[错误] 保存状态失败: {e}")
            return False
    
    def load_state(self, filepath: str):
        """从文件加载状态并恢复"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                save_data = json.load(f)
            
            # 检查数据完整性
            if "current_state" not in save_data:
                raise ValueError("保存文件格式不正确：缺少current_state")
            
            current_state = save_data["current_state"]
            
            # 恢复核心数据结构
            self.answers_list = current_state.get("answers_list", [])
            self.to_explore = current_state.get("to_explore", [])
            self.to_explore_reward = current_state.get("to_explore_reward", {})
            self.visit_counts = current_state.get("visit_counts", {})
            self.ucb_bank = current_state.get("ucb_bank", {})
            self.fathers = current_state.get("fathers", {})
            self.childs = current_state.get("childs", {})
            self.evaluations_bank = current_state.get("evaluations_bank", {})
            self.node_meta_prompts = current_state.get("node_meta_prompts", {})
            
            # 恢复EBC和价值计算
            self.pairwise_relations = current_state.get("pairwise_relations", [])
            self.ebc_global_scores = current_state.get("ebc_global_scores", {})
            self.local_values = current_state.get("local_values", {})
            self.Q_values = current_state.get("Q_values", {})
            
            # 恢复模型相关
            self.node_model_mapping = current_state.get("node_model_mapping", {})
            self.current_elo_ratings = current_state.get("current_elo_ratings", {"GLOBAL": {}})
            self.model_battle_stats = current_state.get("model_battle_stats", {"GLOBAL": {}})
            self.model_usage_queue = current_state.get("model_usage_queue", [])
            self.current_rollout_models = current_state.get("current_rollout_models", [])
            self.rollout_round = current_state.get("rollout_round", 0)
            
            # 恢复配置和状态
            self.iter = current_state.get("iteration", 0)
            self.query = current_state.get("query", "")
            self.system = current_state.get("system", "")
            self.domain = current_state.get("domain", "通用")
            self.class_tag = current_state.get("class_tag", "")
            
            # 恢复状态追踪历史（如果启用）
            if self.enable_state_tracking:
                self.state_history = save_data.get("state_history", [])
                self.session_id = save_data.get("session_info", {}).get("session_id", self.session_id)
                
                # 重新设置文件路径
                self.state_file = os.path.join(self.state_save_path, f"{self.session_id}_continued_state.json")
                self.operation_file = os.path.join(self.state_save_path, f"{self.session_id}_continued_operations.jsonl")
                self.visualization_file = os.path.join(self.state_save_path, f"{self.session_id}_continued_visualization.html")
            
            print(f"[状态恢复] 成功加载状态，当前迭代: {self.iter}")
            print(f"[状态恢复] 节点数: {len(self.answers_list)}, 探索列表: {len(self.to_explore)}")
            return True
            
        except Exception as e:
            print(f"[错误] 加载状态失败: {e}")
            return False
    def create_interactive_visualization(self, filename: str = None):
        """创建交互式可视化HTML"""
        if not self.enable_state_tracking or not self.enable_visualization:
            print("[警告] 可视化功能未启用")
            return False
            
        filename = filename or self.visualization_file
        
        try:
            # 生成完整的交互式HTML
            html_content = self._generate_interactive_html()
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            print(f"[可视化] 交互式HTML已生成: {filename}")
            return True
            
        except Exception as e:
            print(f"[错误] 生成可视化失败: {e}")
            return False
    
    def _generate_interactive_html(self):
        """生成完整的交互式HTML内容"""
        # 准备数据
        viz_data = self._prepare_visualization_data()
        
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaMA-Berry Arena 树搜索可视化</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .controls {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .control-group {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #495057;
        }}
        .control-group input, .control-group select {{
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            background: white;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            margin-right: 10px;
            margin-bottom: 10px;
        }}
        .btn-primary {{
            background: #007bff;
            color: white;
        }}
        .btn-success {{
            background: #28a745;
            color: white;
        }}
        .btn-warning {{
            background: #ffc107;
            color: #212529;
        }}
        .btn:hover {{
            opacity: 0.8;
        }}
        .main-content {{
            display: flex;
            height: 600px;
        }}
        .network-container {{
            flex: 1;
            position: relative;
        }}
        #network {{
            width: 100%;
            height: 100%;
            border: 1px solid #dee2e6;
        }}
        .sidebar {{
            width: 400px;
            background: #f8f9fa;
            padding: 20px;
            overflow-y: auto;
            border-left: 1px solid #dee2e6;
        }}
        .info-panel {{
            background: white;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .info-title {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
        }}
        .info-content {{
            color: #6c757d;
            line-height: 1.4;
        }}
        .operation-log {{
            max-height: 300px;
            overflow-y: auto;
            background: #f1f3f4;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }}
        .log-entry {{
            margin-bottom: 8px;
            padding: 5px;
            background: white;
            border-radius: 3px;
            font-size: 12px;
        }}
        .timestamp {{
            color: #6c757d;
            font-weight: bold;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .progress-bar {{
            background: #e9ecef;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #28a745, #20c997);
            height: 100%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌲 LLaMA-Berry Arena 树搜索可视化</h1>
            <p>会话ID: {viz_data['session_id']} | 总迭代数: {viz_data['total_iterations']}</p>
        </div>
        
        <div class="controls">
            <div class="control-group">
                <label>选择迭代</label>
                <select id="iterationSelect" onchange="loadIteration()">
                    {self._generate_iteration_options()}
                </select>
            </div>
            <div class="control-group">
                <label>播放速度</label>
                <select id="playbackSpeed">
                    <option value="500">慢速 (0.5s)</option>
                    <option value="1000" selected>正常 (1s)</option>
                    <option value="2000">快速 (2s)</option>
                </select>
            </div>
            <button class="btn btn-primary" onclick="playAnimation()">🎬 播放动画</button>
            <button class="btn btn-warning" onclick="pauseAnimation()">⏸️ 暂停</button>
            <button class="btn btn-success" onclick="resetView()">🔄 重置视图</button>
            <button class="btn btn-primary" onclick="exportData()">💾 导出数据</button>
        </div>
        
        <div class="main-content">
            <div class="network-container">
                <div id="network"></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressBar" style="width: 0%">
                        迭代 0 / {viz_data['total_iterations']}
                    </div>
                </div>
            </div>
            
            <div class="sidebar">
                <div class="info-panel">
                    <div class="info-title">📊 当前统计</div>
                    <div class="info-content" id="statsContent">
                        节点总数: {len(self.answers_list)}<br>
                        探索节点: {len(self.to_explore)}<br>
                        当前迭代: {self.iter}
                    </div>
                </div>
                
                <div class="info-panel">
                    <div class="info-title">🎯 选中节点信息</div>
                    <div class="info-content" id="nodeInfo">
                        点击节点查看详细信息
                    </div>
                </div>
                
                <div class="info-panel">
                    <div class="info-title">📈 UCB排行榜</div>
                    <div class="info-content" id="ucbRanking">
                        {self._generate_ucb_ranking_html()}
                    </div>
                </div>
                
                <div class="info-panel">
                    <div class="info-title">📝 操作日志</div>
                    <div class="operation-log" id="operationLog">
                        {self._generate_operation_log_html()}
                    </div>
                </div>
                
                <div class="legend">
                    <div class="info-title">🎨 图例</div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #33FF57;"></div>
                        <span>根节点</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: linear-gradient(45deg, #ff6b6b, #feca57);"></div>
                        <span>评分节点 (颜色表示分数高低)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #74b9ff;"></div>
                        <span>当前最佳节点</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 全局变量
        let network;
        let nodes, edges;
        let animationTimer;
        let currentIteration = 0;
        let isPlaying = false;
        
        // 数据
        const visualizationData = {json.dumps(viz_data, ensure_ascii=False, default=str)};
        
        // 初始化
        document.addEventListener('DOMContentLoaded', function() {{
            initializeNetwork();
            loadIteration();
        }});
        
        // 网络初始化
        function initializeNetwork() {{
            const container = document.getElementById('network');
            
            const data = {{
                nodes: new vis.DataSet(),
                edges: new vis.DataSet()
            }};
            
            const options = {{
                nodes: {{
                    shape: 'dot',
                    size: 20,
                    font: {{
                        size: 14,
                        color: '#343a40'
                    }},
                    borderWidth: 2,
                    borderWidthSelected: 4
                }},
                edges: {{
                    width: 2,
                    color: {{
                        color: '#FF5733',
                        highlight: '#FF0000',
                        hover: '#FF0000'
                    }},
                    arrows: {{
                        to: {{ enabled: true, scaleFactor: 1 }}
                    }},
                    smooth: {{
                        type: 'continuous'
                    }}
                }},
                physics: {{
                    stabilization: {{ iterations: 100 }},
                    barnesHut: {{
                        gravitationalConstant: -30000,
                        centralGravity: 0.3,
                        springLength: 95,
                        springConstant: 0.04,
                        damping: 0.09,
                        avoidOverlap: 0.1
                    }}
                }},
                interaction: {{
                    hover: true,
                    selectConnectedEdges: true
                }},
                layout: {{
                    improvedLayout: true
                }}
            }};
            
            network = new vis.Network(container, data, options);
            
            // 节点选择事件
            network.on('selectNode', function(params) {{
                if (params.nodes.length > 0) {{
                    showNodeDetails(params.nodes[0]);
                }}
            }});
            
            nodes = network.body.data.nodes;
            edges = network.body.data.edges;
        }}
        
        // 加载特定迭代的数据
        function loadIteration() {{
            const select = document.getElementById('iterationSelect');
            currentIteration = parseInt(select.value);
            
            if (visualizationData.state_history && visualizationData.state_history[currentIteration]) {{
                const state = visualizationData.state_history[currentIteration];
                updateNetworkData(state);
                updateStats(state);
                updateProgressBar();
            }}
        }}
        
        // 更新网络数据
        function updateNetworkData(state) {{
            const nodeData = [];
            const edgeData = [];
            
            // 创建节点
            state.answers_list.forEach((nodeText, index) => {{
                const rewards = state.to_explore_reward[nodeText] || [];
                const avgReward = rewards.length > 0 ? rewards.reduce((a, b) => a + b, 0) / rewards.length : 0;
                const ucbValue = state.ucb_bank[nodeText] || 0;
                
                // 判断是否为根节点
                const isRoot = !state.fathers[nodeText];
                
                // 根据分数设置颜色
                let color = '#74b9ff'; // 默认蓝色
                if (isRoot) {{
                    color = '#33FF57'; // 根节点绿色
                }} else if (avgReward > 0) {{
                    // 根据分数设置渐变色
                    const intensity = Math.min(avgReward / 10, 1);
                    const r = Math.floor(255 * (1 - intensity) + 255 * intensity);
                    const g = Math.floor(255 * (1 - intensity) + 107 * intensity);
                    const b = Math.floor(255 * (1 - intensity) + 87 * intensity);
                    color = `rgb(${{r}}, ${{g}}, ${{b}})`;
                }}
                
                // 获取模型信息
                const modelInfo = state.node_model_mapping[nodeText];
                const modelName = modelInfo ? modelInfo.model_name : 'unknown';
                
                nodeData.push({{
                    id: nodeText,
                    label: `${{nodeText.substring(0, 20)}}...\\n[${{modelName}}]\\nUCB: ${{ucbValue.toFixed(2)}}`,
                    title: `完整内容: ${{nodeText}}\\n模型: ${{modelName}}\\n平均奖励: ${{avgReward.toFixed(2)}}\\nUCB值: ${{ucbValue.toFixed(2)}}\\n访问次数: ${{rewards.length}}`,
                    color: color,
                    size: Math.max(15, Math.min(40, 15 + rewards.length * 3))
                }});
            }});
            
            // 创建边
            Object.keys(state.fathers).forEach(child => {{
                const parent = state.fathers[child];
                if (parent) {{
                    edgeData.push({{
                        from: parent,
                        to: child,
                        id: `${{parent}}-${{child}}`
                    }});
                }}
            }});
            
            // 更新网络
            nodes.clear();
            edges.clear();
            nodes.add(nodeData);
            edges.add(edgeData);
        }}
        
        // 显示节点详细信息
        function showNodeDetails(nodeId) {{
            const state = visualizationData.state_history[currentIteration];
            const rewards = state.to_explore_reward[nodeId] || [];
            const avgReward = rewards.length > 0 ? rewards.reduce((a, b) => a + b, 0) / rewards.length : 0;
            const modelInfo = state.node_model_mapping[nodeId];
            const metaPrompts = state.node_meta_prompts[nodeId] || [];
            
            const info = `
                <strong>节点内容:</strong><br>
                <div style="max-height: 100px; overflow-y: auto; background: #f8f9fa; padding: 8px; border-radius: 4px; margin: 5px 0;">
                    ${{nodeId.substring(0, 200)}}${{nodeId.length > 200 ? '...' : ''}}
                </div>
                <strong>模型信息:</strong> ${{modelInfo ? modelInfo.model_name : 'unknown'}}<br>
                <strong>平均奖励:</strong> ${{avgReward.toFixed(4)}}<br>
                <strong>访问次数:</strong> ${{rewards.length}}<br>
                <strong>UCB值:</strong> ${{(state.ucb_bank[nodeId] || 0).toFixed(4)}}<br>
                <strong>Meta Prompts:</strong> ${{metaPrompts.length}} 个
            `;
            
            document.getElementById('nodeInfo').innerHTML = info;
        }}
        
        // 更新统计信息
        function updateStats(state) {{
            const stats = `
                节点总数: ${{state.answers_list.length}}<br>
                探索节点: ${{state.to_explore.length}}<br>
                当前迭代: ${{state.iteration}}<br>
                UCB均值: ${{Object.values(state.ucb_bank).length > 0 ? (Object.values(state.ucb_bank).reduce((a,b) => a+b, 0) / Object.values(state.ucb_bank).length).toFixed(2) : '0.00'}}<br>
                对战关系: ${{state.pairwise_relations.length}} 对
            `;
            document.getElementById('statsContent').innerHTML = stats;
        }}
        
        // 更新进度条
        function updateProgressBar() {{
            const progress = (currentIteration / Math.max(1, visualizationData.total_iterations)) * 100;
            const progressBar = document.getElementById('progressBar');
            progressBar.style.width = progress + '%';
            progressBar.textContent = `迭代 ${{currentIteration}} / ${{visualizationData.total_iterations}}`;
        }}
        
        // 播放动画
        function playAnimation() {{
            if (isPlaying) return;
            
            isPlaying = true;
            const speed = parseInt(document.getElementById('playbackSpeed').value);
            const totalIterations = visualizationData.total_iterations;
            
            animationTimer = setInterval(() => {{
                currentIteration++;
                if (currentIteration >= totalIterations) {{
                    currentIteration = 0;
                }}
                
                document.getElementById('iterationSelect').value = currentIteration;
                loadIteration();
            }}, speed);
        }}
        
        // 暂停动画
        function pauseAnimation() {{
            if (animationTimer) {{
                clearInterval(animationTimer);
                isPlaying = false;
            }}
        }}
        
        // 重置视图
        function resetView() {{
            currentIteration = 0;
            document.getElementById('iterationSelect').value = 0;
            loadIteration();
            pauseAnimation();
            if (network) {{
                network.fit();
            }}
        }}
        
        // 导出数据
        function exportData() {{
            const dataStr = JSON.stringify(visualizationData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `rollout_data_${{visualizationData.session_id}}.json`;
            link.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>
        """
        
        return html_template

    def _prepare_visualization_data(self):
        """准备可视化数据"""
        return {
            "session_id": getattr(self, 'session_id', 'unknown'),
            "total_iterations": self.iter,
            "state_history": getattr(self, 'state_history', []),
            "operation_log": getattr(self, 'operation_log', []),
            "config": {
                "max_iter": self.max_iter,
                "use_diversity_fusion": self.use_diversity_fusion,
                "ebc_alpha": self.ebc_alpha,
                "gamma": self.gamma
            }
        }
    def _generate_iteration_options(self):
        """生成迭代选择下拉框选项"""
        options = []
        if hasattr(self, 'state_history'):
            for i in range(len(self.state_history)):
                options.append(f'<option value="{i}">迭代 {i}</option>')
        else:
            options.append('<option value="0">迭代 0</option>')
        return '\n'.join(options)
    
    def _generate_ucb_ranking_html(self):
        """生成UCB排行榜HTML"""
        if not self.ucb_bank:
            return "暂无UCB数据"
        
        # 按UCB值排序
        sorted_nodes = sorted(self.ucb_bank.items(), key=lambda x: x[1], reverse=True)[:5]
        
        html = "<div style='font-size: 12px;'>"
        for i, (node, ucb_value) in enumerate(sorted_nodes, 1):
            node_preview = node[:30] + "..." if len(node) > 30 else node
            model_info = self.node_model_mapping.get(node, {})
            model_name = model_info.get("model_name", "unknown")
            
            html += f"""
            <div style='margin-bottom: 8px; padding: 5px; background: #f8f9fa; border-radius: 3px;'>
                <strong>#{i}</strong> UCB: {ucb_value:.3f}<br>
                <span style='color: #6c757d;'>{node_preview}</span><br>
                <span style='color: #007bff; font-size: 10px;'>[{model_name}]</span>
            </div>
            """
        html += "</div>"
        return html
    
    def _generate_operation_log_html(self):
        """生成操作日志HTML"""
        if not hasattr(self, 'operation_log') or not self.operation_log:
            return "暂无操作日志"
        
        html = ""
        # 显示最近的10个操作
        recent_operations = self.operation_log[-10:] if len(self.operation_log) > 10 else self.operation_log
        
        for op in reversed(recent_operations):  # 最新的在前
            timestamp = op.get('timestamp', '')
            if timestamp:
                timestamp = timestamp.split('T')[1].split('.')[0]  # 只显示时间部分
            
            html += f"""
            <div class='log-entry'>
                <span class='timestamp'>{timestamp}</span> - 
                <strong>{op.get('operation_type', 'Unknown')}</strong>
                <div style='margin-top: 3px; color: #6c757d; font-size: 11px;'>
                    迭代 {op.get('iteration', '?')}
                </div>
            </div>
            """
        
        return html
    def _get_default_model_configs(self):
        """获取默认的模型配置列表"""
        return {
            "gemini-3-pro-preview": {
                "model_name": "gemini-3-pro-preview",
                "api_base": '<your-base-url>',
                "api_type": "openai",
                "api_key": '<your-api-key>',
                "anony_only": False,
                "sampling_params": {
                    "extra_body": {"enable_thinking": True}
                },
                "model_link": "https://aistudio.google.com/app/prompts/new_chat?model=gemini-3-pro-preview",
                "description": "From Google",
                "organization": "Google",
                "license": "Proprietary",
                "sampling_weight": 1
            },
        }

    def _select_random_model(self):
        """
        [修改版] 模型选择策略：
        1. 初始阶段 / 数据不足阶段：强制循环遍历队列，确保每个模型都有机会运行。
        2. 成熟阶段 (所有模型对战数 > min_battles)：切换为基于 Softmax+UCB 的加权随机采样。
        """
        # 过滤掉权重为0的模型（被禁用的模型）
        available_models = {
            key: config for key, config in self.model_configs.items() 
            if config.get("sampling_weight", 1) > 0
        }
        
        if not available_models:
            print("警告：没有可用的模型配置，使用默认LLM")
            return self.llm, {"model_name": "default", "organization": "Unknown"}
        
        available_model_names = list(available_models.keys())
        
        if self.model_usage_queue:
            # 队列不为空，处于【强制遍历阶段】
            selected_model_name = self.model_usage_queue.pop(0)
            selection_type = "遍历(数据积累中)"
            # print(f"遍历阶段 - 选择模型: {selected_model_name} (队列剩余: {len(self.model_usage_queue)})")
            
        else:
            weights = [available_models[name]["sampling_weight"] for name in available_model_names]
            selected_model_name = random.choices(available_model_names, weights=weights)[0]
            # 找到权重最大值对应的索引，然后取名字
            # max_weight_index = weights.index(max(weights))
            # selected_model_name = available_model_names[max_weight_index]
            selection_type = "智能采样(ELO+UCB)"
            # print(f"智能阶段 - 基于权重选择: {selected_model_name} (权重: {available_models[selected_model_name]['sampling_weight']:.2f})")

        # --- 核心逻辑修改结束 ---

        selected_config = available_models[selected_model_name]
        
        # 以下保持原代码不变...
        
        # 根据模型配置设置thinking模式
        sampling_params = selected_config.get("sampling_params", {})
        extra_body = sampling_params.get("extra_body", {})
        
        # 检查是否启用thinking模式
        thinking_enabled = False
        if "enable_thinking" in extra_body and extra_body["enable_thinking"]:
            thinking_enabled = True
            self.extra_body = extra_body
        elif "thinking" in extra_body:
            thinking_config = extra_body["thinking"]
            if isinstance(thinking_config, dict) and thinking_config.get("type") == "enabled":
                thinking_enabled = True
                self.extra_body = extra_body
        # 3. [新增] 检查 reasoning: {"enabled": True} (DeepSeek R1/SGLang 风格)
        elif "reasoning" in extra_body:
            reasoning_config = extra_body["reasoning"]
            self.extra_body = extra_body
            # 确保 reasoning 是字典，且 enabled 为 True
            if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is True:
                thinking_enabled = True
        elif "include_reasoning" in extra_body:
            reasoning_config = extra_body["include_reasoning"]
            self.extra_body = extra_body
            # 确保 reasoning 是字典，且 enabled 为 True
            if reasoning_config == True:
                thinking_enabled = True
        # 设置self.think
        self.think = thinking_enabled
        print(f"[{selection_type}] 选择模型: {selected_model_name} | Thinking: {thinking_enabled}")
        
        # 构建模型信息
        model_info = {
            "model_key": selected_model_name,
            "model_name": selected_config["model_name"],
            "organization": selected_config.get("organization", "Unknown"),
            "api_base": selected_config["api_base"],
            "description": selected_config.get("description", ""),
            "selection_type": selection_type,
            "rollout_round": self.rollout_round,
            "thinking_enabled": thinking_enabled
        }
        
        # 记录当前rollout中的模型使用
        if selected_model_name not in self.current_rollout_models:
            self.current_rollout_models.append(selected_model_name)
        
        # 检查缓存
        if selected_model_name in self.model_cache:
            return self.model_cache[selected_model_name], model_info
        
        # 创建新的LLM实例
        try:
            llm_instance = LLM_Core(
                tokenizer=self.llm.tokenizer if hasattr(self.llm, 'tokenizer') else None,
                use_async=True,
                api_model=selected_config["model_name"],
                base_url=selected_config["api_base"],
                api_key=selected_config["api_key"]
            )
            
            self.model_cache[selected_model_name] = llm_instance
            return llm_instance, model_info
            
        except Exception as e:
            print(f"创建模型 {selected_model_name} 失败: {e}")
            print("使用默认LLM")
            return self.llm, {"model_name": "default_fallback", "organization": "Unknown"}

    def _initialize_model_queue(self, available_model_names):
        """初始化模型使用队列，可以选择随机打乱顺序"""
        # 复制可用模型列表
        queue = available_model_names.copy()
        
        # 可选：随机打乱遍历顺序，让每次运行的遍历顺序不同
        random.shuffle(queue)
        
        self.model_usage_queue = queue
        print(f"初始化模型遍历队列: {queue}")

    def start_new_rollout(self):
        """开始新的rollout，重置模型选择状态"""
        self.rollout_round += 1
        previous_models = self.current_rollout_models.copy()
        self.current_rollout_models = []
        
        print(f"\n=== 开始新的rollout (第{self.rollout_round}轮) ===")
        print(f"上一轮使用的模型: {previous_models}")
        print(f"队列状态: {len(self.model_usage_queue)} 个模型待遍历")
        
        # [新增] 在每次新rollout时更新ELO评分和权重
        self.update_elo_ratings()
        
        # 如果所有模型都已遍历过，可以选择重新初始化队列
        if not self.model_usage_queue and self.rollout_round > 1:
            available_models = {
                key: config for key, config in self.model_configs.items() 
                if config.get("sampling_weight", 1) > 0
            }
            print("所有模型已遍历完成，后续将使用随机选择策略")

    def calculate_elo(self, winner_rating: float, loser_rating: float, k: int = 32):
        """
        计算 Elo 分数更新
        :param winner_rating: 赢家的当前分数
        :param loser_rating: 输家的当前分数
        :param k: K-因子，决定分数变动的幅度
        :return: (更新后的赢家分数, 更新后的输家分数)
        """
        import math
        # 计算胜率期望
        expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        expected_loser = 1 - expected_winner
        
        # 更新分数
        new_winner_rating = winner_rating + k * (1 - expected_winner)
        new_loser_rating = loser_rating + k * (0 - expected_loser)
        
        return new_winner_rating, new_loser_rating

    def load_elo_ratings_from_file(self):
        """
        从数据文件中加载和计算ELO评分
        """
        if not self.elo_data_file or not os.path.exists(self.elo_data_file):
            print(f"ELO数据文件不存在或未指定: {self.elo_data_file}")
            return
        
        print(f"正在从文件计算ELO评分: {self.elo_data_file}")
        
        # 初始化所有模型的ELO分数为1200
        model_elo = {}
        battle_stats = {}
        
        # 初始化配置中的所有模型
        for model_key in self.model_configs.keys():
            model_elo[model_key] = 1200.0
            battle_stats[model_key] = {"win": 0, "loss": 0, "total": 0}
        
        # 读取数据并处理对战结果
        battles_processed = 0
        try:
            with open(self.elo_data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        model_config = data.get("model_config", {})
                        
                        # 提取赢家和输家的模型key
                        chosen_model = model_config.get("chosen_model", {})
                        rejected_model = model_config.get("rejected_model", {})
                        
                        chosen_key = chosen_model.get("model_key")
                        rejected_key = rejected_model.get("model_key")
                        
                        if not chosen_key or not rejected_key or chosen_key == rejected_key:
                            continue
                        
                        # 如果模型不在字典中，添加它们
                        if chosen_key not in model_elo:
                            model_elo[chosen_key] = 1200.0
                            battle_stats[chosen_key] = {"win": 0, "loss": 0, "total": 0}
                        if rejected_key not in model_elo:
                            model_elo[rejected_key] = 1200.0
                            battle_stats[rejected_key] = {"win": 0, "loss": 0, "total": 0}
                        
                        # 更新ELO分数
                        w_old = model_elo[chosen_key]
                        l_old = model_elo[rejected_key]
                        
                        w_new, l_new = self.calculate_elo(w_old, l_old)
                        
                        model_elo[chosen_key] = w_new
                        model_elo[rejected_key] = l_new
                        
                        # 统计对战记录
                        battle_stats[chosen_key]["win"] += 1
                        battle_stats[chosen_key]["total"] += 1
                        battle_stats[rejected_key]["loss"] += 1
                        battle_stats[rejected_key]["total"] += 1
                        
                        battles_processed += 1
                        
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            print(f"ELO数据文件未找到: {self.elo_data_file}")
            return
        
        # 保存计算结果
        self.current_elo_ratings = model_elo
        self.model_battle_stats = battle_stats
        
        print(f"ELO评分计算完成，处理了 {battles_processed} 场对战")
        
        # 打印排行榜（仅显示配置中的模型）
        config_models = [(k, v) for k, v in model_elo.items() if k in self.model_configs]
        sorted_leaderboard = sorted(config_models, key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*60)
        print(f"{'排名':<4} {'模型':<25} {'ELO分数':<10} {'胜率':<8} {'对战数'}")
        print("-"*60)
        
        for rank, (name, score) in enumerate(sorted_leaderboard, 1):
            stats = battle_stats.get(name, {"win": 0, "total": 0})
            win_rate = (stats["win"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"{rank:<4} {name:<25} {score:<10.1f} {win_rate:>6.1f}% {stats['total']:>8}")
        print("="*60)

    def update_model_weights_by_elo(self, temperature: float = 100.0, exploration_weight: float = 1.0):
        """
        [修改版] 根据ELO分数和使用次数动态调整模型采样权重
        应用策略：
        1. Softmax: 将ELO分数转化为概率分布，保留相对差距 (Exploitation)
        2. UCB (Upper Confidence Bound): 给予对战次数少的模型额外加成 (Exploration)
        
        :param temperature: 温度系数 (T)。
               T越小，权重越集中在最强模型（贪婪）；
               T越大，权重分布越平缓。建议 50-200 之间。
        :param exploration_weight: 探索权重系数 (C)。
               控制"尝试新模型"的意愿。建议 0.5-2.0 之间。
        """
        if not self.enable_elo_weighting or not self.current_elo_ratings:
            return
        
        print(f"\n正在根据ELO分数更新模型权重 (Softmax T={temperature}, UCB C={exploration_weight})...")
        
        # 1. 筛选在配置文件中存在的模型
        config_model_keys = [k for k in self.model_configs.keys() if k in self.current_elo_ratings]
        
        if len(config_model_keys) < 2:
            print("可用模型数量不足，跳过权重更新")
            return
            
        # 获取相关数据
        elos = {k: self.current_elo_ratings[k] for k in config_model_keys}
        battle_stats = self.model_battle_stats
        
        # 计算总对战数 (用于 UCB 分子)
        total_battles = sum(s.get("total", 0) for s in battle_stats.values())
        # 避免 log(0)
        log_total_battles = math.log(max(total_battles, 1))
        
        # 2. 计算 Softmax 概率 (Exploitation 部分)
        # Step A: 减去最大值防止 exp 溢出 (Shift Invariance)
        max_elo = max(elos.values())
        # Step B: 计算 exp( (elo - max) / T )
        exp_scores = {k: math.exp((v - max_elo) / temperature) for k, v in elos.items()}
        sum_exp = sum(exp_scores.values())
        # Step C: 归一化
        softmax_probs = {k: v / sum_exp for k, v in exp_scores.items()}
        
        print(f"{'模型':<35} | {'ELO':<8} | {'对战数':<6} | {'胜率项(Prob)':<10} | {'探索项(UCB)':<10} | {'新权重'}")
        print("-" * 100)
        
        # 3. 结合 UCB 计算最终权重并更新
        for k in config_model_keys:
            # A. 基础胜率分 (来自 Softmax)
            prob_score = softmax_probs[k]
            
            # B. 探索加成 (UCB Bonus)
            # 公式: C * sqrt( ln(Total) / (N_j + 1) )
            # 加 1 是为了防止除零，且确保 0 次对战的新模型获得最大加成
            n_j = battle_stats.get(k, {}).get("total", 0)
            ucb_bonus = exploration_weight * math.sqrt(log_total_battles / (n_j + 1))
            
            # C. 最终权重 (Prob + Bonus)
            # 乘以 10 是为了让权重数值在肉眼查看时更直观 (变成 1.0 ~ 5.0 的量级)，
            # 对 random.choices 来说，权重的绝对大小不影响，只看相对比例。
            final_weight = (prob_score + ucb_bonus) * 10
            
            # 更新到配置中
            old_weight = self.model_configs[k].get("sampling_weight", 1.0)
            self.model_configs[k]["sampling_weight"] = final_weight
            
            print(f"{k:<35} | {elos[k]:<8.1f} | {n_j:<6} | {prob_score:<10.4f} | {ucb_bonus:<10.4f} | {final_weight:.4f}")
            
        print("-" * 100)

    def update_elo_ratings(self):
        """
        在每次rollout时更新ELO评分和权重
        """
        if self.enable_elo_weighting and self.elo_data_file:
            self.load_elo_ratings_from_file()
            self.update_model_weights_by_elo()

    def get_model_usage_stats(self):
        """获取模型使用统计信息"""
        stats = {
            "rollout_round": self.rollout_round,
            "models_in_queue": len(self.model_usage_queue),
            "models_used_this_round": self.current_rollout_models.copy(),
            "total_nodes": len(self.node_model_mapping),
            "model_distribution": {}
        }
        
        # 统计每个模型的使用次数
        for node, model_info in self.node_model_mapping.items():
            model_name = model_info.get("model_name", "unknown")
            if model_name not in stats["model_distribution"]:
                stats["model_distribution"][model_name] = {
                    "count": 0,
                    "selection_types": {"遍历": 0, "随机": 0}
                }
            stats["model_distribution"][model_name]["count"] += 1
            selection_type = model_info.get("selection_type", "未知")
            if selection_type in stats["model_distribution"][model_name]["selection_types"]:
                stats["model_distribution"][model_name]["selection_types"][selection_type] += 1
        
        return stats

    def _save_judge_data(self, prompt: str, evaluation_object: BaseModel, chosen_model_info: dict = None, rejected_model_info: dict = None):
        """
        [新增] 将评测的 Prompt 和模型的结构化输出保存为 JSONL 格式。
        格式遵循常见的 SFT 数据格式：{"messages": [{"role": "user",...}, {"role": "assistant",...}]}
        [修改] 添加模型对战信息，使其可以作为ELO计算的数据源
        """
        try:
            # 获取 Pydantic 模型的 JSON 字符串表示 (作为 Assistant 的回复)
            # 兼容 Pydantic v1 (.json()) 和 v2 (.model_dump_json())
            if hasattr(evaluation_object, 'model_dump_json'):
                response_content = evaluation_object.model_dump_json(indent=2, ensure_ascii=False)
            else:
                response_content = evaluation_object.json(indent=2, ensure_ascii=False)

            # 构建训练数据条目
            entry = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response_content}
                ],
                "metadata": {
                    "source": "LLMExplorer_Socrates_Gen",
                    "timestamp": str(uuid.uuid4()) # 或者用时间戳
                }
            }
            
            # [新增] 添加模型对战信息，使其可以作为ELO数据源
            if chosen_model_info and rejected_model_info:
                entry["model_config"] = {
                    "chosen_model": chosen_model_info,
                    "rejected_model": rejected_model_info,
                    "strategy": " Em-Mcts (EBC + Bradley-Terry) + Arena Model Selection"
                }

            # 追加写入文件 (JSONL)
            with open(self.save_dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            print(f"[Warning] 保存训练数据失败: {e}")

    def calculate_ebc_scores(self) -> Dict[str, float]:
        """
        [修正版] 组件三：Enhanced Borda Count (EBC) 核心算法
        根据论文  Em-Mcts 实现：
        1. 构建偏好矩阵 M
        2. 计算传递闭包 C
        3. 计算 Borda Count (战胜的节点数)
        4. 计算全局分位数分数 Qg = 1 - (rank - 1) / (N - 1)
        """
        # 1. 获取所有参与过比较的唯一节点
        unique_nodes = list(set(self.answers_list))
        n = len(unique_nodes)
        if n == 0:
            return {}
        if n == 1:
            return {unique_nodes[0]: 1.0}
        
        node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
        
        # 2. 初始化邻接矩阵 M
        M = np.zeros((n, n), dtype=int)
        
        for winner, loser in self.pairwise_relations:
            if winner in node_to_idx and loser in node_to_idx:
                u, v = node_to_idx[winner], node_to_idx[loser]
                M[u][v] = 1
                M[v][u] = 0 

        # 3. 计算传递闭包 (Floyd-Warshall)
        Closure = M.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # 如果 i>k 且 k>j，则推断 i>j
                    Closure[i][j] = Closure[i][j] or (Closure[i][k] and Closure[k][j])
        
        # 4. 计算 Borda Count (出度)
        borda_counts = Closure.sum(axis=1) # Shape: (n,)
        
        # 5. 计算全局分位数分数 (Global Quantile Score)
        # 论文公式: Qg(v) = 1 - (Rank(v) - 1) / (N - 1)
        # Rank(v) 是 1-based (1代表最好), N 是节点总数
        
        # 将节点按 Borda Count 从大到小排序
        # 创建 (index, borda_count) 列表
        indexed_borda = list(enumerate(borda_counts))
        # 排序：分数高的排前面
        indexed_borda.sort(key=lambda x: x[1], reverse=True)
        
        ebc_scores = {}
        # 分配排名
        # i 是 0-based index, 对应 Rank = i + 1
        # 但我们需要处理 Borda Count 相同的情况（Tie-breaking）
        # 这里简化处理，直接使用排序后的索引
        
        for rank_idx, (original_idx, count) in enumerate(indexed_borda):
            node = unique_nodes[original_idx]
            rank = rank_idx + 1 # 1-based rank
            
            # 应用公式
            if n > 1:
                quantile_score = 1.0 - (rank - 1) / (n - 1)
            else:
                quantile_score = 1.0
                
            ebc_scores[node] = quantile_score

        self.ebc_global_scores = ebc_scores
        print(f"--- EBC 计算完成，覆盖 {n} 个节点 (Top Node Borda: {indexed_borda[0][1]}) ---")
        return ebc_scores

    # ... (省略 optimize_experience_library 及相关辅助函数，保持原样或按需保留) ...
    async def optimize_experience_library(self, rag: 'AsyncFaissRAG', now_experiences: List[str]):
        """
        根据 GRPO 论文的思想，智能地优化经验库 (最终版：手动构建 Schema 以确保 API 兼容性)。
        :param rag: 经验库RAG实例。
        :param now_experiences: 本轮迭代中新生成的一批经验。
        """
        if not now_experiences:
            print("没有新的经验，跳过优化。")
            return

        print(f"\n--- 开始经验库优化，收到 {len(now_experiences)} 条现有经验 ---")
        from typing import List, Literal, Dict, Any
        from pydantic import BaseModel, Field
        # --- 步骤 1: 【仍然需要】定义 Pydantic 模型，但仅用于【解析】LLM 的返回结果 ---
        # 我们不再使用它们来生成 Schema，所以它们的定义不会引发错误。
        class ExperienceOperation(BaseModel):
            option: Literal["add", "modify", "merge", "delete"]
            experience: str = None
            modified_from_id: int = None
            merged_from_ids: List[int] = None
            delete_id: int = None

        class ExperienceUpdatePlan(BaseModel):
            think: str
            plan: List[ExperienceOperation]

        # --- 步骤 2: 准备 Prompt 输入 (保持不变) ---
        existing_results = await rag.search(self.query, top_k=10)
        existing_experiences_str = "\n".join([f"ID: {res['id']}, Content: {res['value']}" for res in existing_results])
        if not existing_experiences_str:
            existing_experiences_str = "无相关过往经验。"
        now_experiences_str = "\n".join([f"- {s}" for s in now_experiences])
        prompt = self.prompter.OPTIMIZE_EXPERIENCE_PROMPT.format(
            existing_experiences=existing_experiences_str,
            now_experiences=now_experiences_str
        )

        # --- 步骤 3: 【核心修改】手动定义一个完全“扁平化”的 JSON Schema ---
        # 这个 schema 没有任何嵌套定义($defs, definitions)或引用($ref)，保证了最大的兼容性。
        MANUAL_TOOL_SCHEMA = {
            "type": "object",
            "properties": {
                "think": {
                    "type": "string",
                    "description": "优化决策的思考过程不少于500字。"
                },
                "plan": {
                    "type": "array",
                    "description": "包含一系列操作的修订计划。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "option": {
                                "type": "string",
                                "description": "要执行的操作类型。",
                                "enum": ["add", "modify", "merge", "delete"]
                            },
                            "experience": {
                                "type": "string",
                                "description": "对于'add', 'modify', 'merge'操作，这是新的经验内容。"
                            },
                            "modified_from_id": {
                                "type": "integer",
                                "description": "对于'modify'操作，这是被修改的旧经验的ID。"
                            },
                            "merged_from_ids": {
                                "type": "array",
                                "description": "对于'merge'操作，这是被合并的旧经验的ID列表。",
                                "items": {"type": "integer"}
                            },
                            "delete_id": {
                                "type": "integer",
                                "description": "对于'delete'操作，这是要删除的经验的ID。"
                            }
                        },
                        "required": ["option"]
                    }
                }
            },
            "required": ["think", "plan"]
        }
        
        # --- 步骤 4: 构建并发送 API 请求 ---
        update_plan = None
        MAX_RETRIES = 3
    
        for attempt in range(MAX_RETRIES):
            try:
                data = copy.deepcopy(self.data_template)
                data["model"] = self.api_llm2.api_model
                data["messages"] = [{"role": "user", "content": prompt}]
                
                # 根据您的API要求，可能需要使用 'function_declarations' 或 'tools'
                # 我们这里使用标准的 'tools' 格式
                data["tools"] = [{
                    "type": "function",
                    "function": {
                        "name": "ExperienceUpdatePlan",
                        "description": "用于优化经验库的思考过程和具体操作计划。",
                        "parameters": MANUAL_TOOL_SCHEMA  # 使用我们手动创建的、干净的 schema
                    }
                }]
                data["tool_choice"] = {"type": "function", "function": {"name": "ExperienceUpdatePlan"}}
                data["timeout"] = 36000
                print("--- 正在调用 LLM 生成优化计划 (使用手动 Schema)... ---")
                response = await self.api_llm2.client.chat.completions.create(**data)

                # [新增] 记录token使用
                self._record_token_usage(response)

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                if tool_calls:
                    function_args_str = tool_calls[0].function.arguments
                    function_args = json.loads(function_args_str)
                    # 使用 Pydantic 模型来验证和解析返回的 JSON
                    update_plan = ExperienceUpdatePlan(**function_args)
                    print("--- 成功解析优化计划 ---")
                else:
                    print("错误：LLM 未能按预期生成工具调用。")

            except Exception as e:
                print(f"调用LLM或解析优化计划时出错 (尝试 {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    print("将在1秒后重试...")
                    await asyncio.sleep(1)
                else:
                    print("已达到最大重试次数。")

        # --- 步骤 5: 如果所有重试都失败，则执行保守策略 ---
        if not update_plan:
            print("所有尝试均失败。采取保守策略：仅添加新经验。")
            for exp in now_experiences:
                await rag.add_document(self.query, exp)
            return

        if not update_plan or not update_plan.plan:
            print("LLM未能生成有效的优化计划或计划为空。")
            if update_plan and update_plan.think:
                print(f"LLM的思考过程: {update_plan.think}")
            return
        
        print(f"LLM 思考过程:\n{update_plan.think}\n")

        # --- 步骤 5: 执行优化计划 (您的逻辑保持不变) ---
        print(f"--- 准备执行 {len(update_plan.plan)} 个操作 ---")
        # ... 您的 for 循环执行逻辑 ...
            # --- 步骤 4: 执行优化计划 (保持不变) ---
        print("正在执行LLM生成的经验库优化计划...")
        for op in update_plan.plan:
            try:
                if op.option == "add":
                    print(f"  [ADD] 添加新经验: {op.experience[:50]}...")
                    await rag.add_document(self.query, op.experience)
                
                elif op.option == "delete":
                    print(f"  [DELETE] 删除经验 ID: {op.delete_id}")
                    await rag.delete_document(int(op.delete_id))

                elif op.option == "modify":
                    print(f"  [MODIFY] 修改经验 ID: {op.modified_from_id}")
                    print(f"     └─ 新内容: {op.experience[:50]}...")
                    if op.experience == "":
                        print("新内容经验为空...skip")
                        continue
                    await rag.delete_document(op.modified_from_id)
                    await rag.add_document(self.query, op.experience)

                elif op.option == "merge":
                    print(f"  [MERGE] 合并经验 IDs: {op.merged_from_ids}")
                    print(f"     └─ 生成新经验: {op.experience[:50]}...")
                    if op.experience == "":
                        print("新内容经验为空...skip")
                        continue
                    for id_to_delete in op.merged_from_ids:
                        await rag.delete_document(id_to_delete)
                    await rag.add_document(self.query, op.experience)
            except Exception as e:
                print(f"执行操作 {op.option} 时失败: {e}")
        
        print("--- 经验库优化完成 ---\n")

    async def get_prior_experience(self, rag: AsyncFaissRAG, top_k=10):
        results = await rag.search(self.query, top_k=top_k)
        if not results:
            return "无"
        new_experience = ""
        for result in results:
            if float(result['similarity']) < 0.9: # 稍微放宽一点
                continue
            new_experience += f"{str(result['id'])}. {result['value']}\n"
        return new_experience

    async def get_weak_answer(self, parent_node):
        """基于父节点进化Meta-Prompt并生成新答案"""
        print("正在执行进化步骤并生成新答案...")
        data_template2 = copy.deepcopy(self.data_template)
        parent_meta_prompt = self.node_meta_prompts.get(parent_node, [self.system])[-1]
        
        # [新增] 随机选择模型
        selected_llm, model_info = self._select_random_model()
        print(f"[DEBUG] 选择模型: {model_info['model_name']} ({model_info.get('model_key', 'unknown_key')})")
        
        if self.use_meta_prompt:
            prior_experience = await self.get_prior_experience(self.rag)
            if parent_node is None:
                print(f"正在从父节点 '{str(parent_node)[:50]}...' 的meta-prompt进化...")
                evolved_meta_prompt = self.system
            else:
                evolved_meta_prompt = parent_meta_prompt
                
        # 通用领域逻辑
        if self.use_meta_prompt:
            data_template2["messages"] = [
                {"role": "user", "content":  evolved_meta_prompt + "\n" + "##先验经验##\n" + prior_experience + "\n" + self.query}]
        else:
            evolved_meta_prompt = ""
            data_template2["messages"] = [{"role": "user", "content": self.query}]
        data_template2["model"] = selected_llm.api_model
        data_template2["extra_body"] = self.extra_body
        Example_Response, thinking, completion = await self.process_controller.Generate_Response(selected_llm, data_template2, think=self.think)
        # [新增] 记录token使用
        self._record_token_usage(completion)
        self.thinks_bank[Example_Response] = thinking
        result = Example_Response

        # [DEBUG] 检查生成的结果
        print(f"[DEBUG] 生成的结果长度: {len(result) if result else 0}")
        print(f"[DEBUG] 结果内容预览: '{result[:50] if result else 'EMPTY'}...'")
        print(f"[DEBUG] 结果是否为空: {not result}")
        print(f"[DEBUG] 结果是否为错误: {result == '<|_error_|>'}")

        # [新增] 记录节点与模型的映射关系
        if result and result != "<|_error_|>":
            self.node_model_mapping[result] = model_info
            print(f"[DEBUG] ✅ 成功记录节点模型映射: {result[:30]}... -> {model_info['model_name']}")
            print(f"[DEBUG] 当前映射字典大小: {len(self.node_model_mapping)}")
        else:
            print(f"[DEBUG] ❌ 未记录模型映射 - 结果为空或错误")
            print(f"[DEBUG] 但仍然记录模型信息用于追踪: {model_info['model_name']}")
            # 对于空结果，我们也要记录模型信息以便追踪
            empty_key = f"<EMPTY_RESULT_{len(self.node_model_mapping)}>"
            self.node_model_mapping[empty_key] = model_info
            print(f"[DEBUG] 使用临时键记录空结果的模型: {empty_key}")

        return result, evolved_meta_prompt
    
    async def step(self, parent_node):
        return await self.get_weak_answer(parent_node)

    async def _register_node_and_link_to_parent(self, child, father):
        """注册节点并建立父子链接"""
        if child not in self.answers_list:
            self.answers_list.append(child)
            self.to_explore.append(child)
            self.childs[child] = []
            self.fathers[child] = father

        if father is not None:
            if father not in self.childs:
                self.childs[father] = []
            if child not in self.childs[father]:
                self.childs[father].append(child)

    def _truncate_to_tokens(self, text: str, max_tokens: int = 10000) -> str:
        """
        将文本截取到指定的token数量
        """
        if not text:
            return text
            
        try:
            # 使用tokenizer进行编码
            if hasattr(self.llm, 'tokenizer') and self.llm.tokenizer:
                tokens = self.llm.tokenizer.encode(text, add_special_tokens=False)
                
                # 如果超过最大token数，进行截取
                if len(tokens) > max_tokens:
                    truncated_tokens = tokens[:max_tokens]
                    truncated_text = self.llm.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    print(f"文本被截取: {len(tokens)} -> {len(truncated_tokens)} tokens")
                    return truncated_text
                else:
                    return text
            else:
                # 如果没有tokenizer，使用字符长度的近似方法
                # 大概按照中文1个字符=2个token，英文1个单词=1.3个token来估算
                estimated_tokens = len(text.split()) * 1.3 + len([c for c in text if ord(c) > 127]) * 2
                if estimated_tokens > max_tokens:
                    # 粗略截取到指定比例
                    ratio = max_tokens / estimated_tokens
                    truncated_text = text[:int(len(text) * ratio)]
                    print(f"文本被近似截取: 估计{estimated_tokens:.0f} tokens -> 目标{max_tokens} tokens")
                    return truncated_text
                else:
                    return text
        except Exception as e:
            print(f"截取文本时发生错误: {e}, 返回原文本")
            return text

    def _check_all_models_ready(self, min_battles=1):
        """
        检查配置中的所有启用模型是否都已具备有效的数据基础。
        
        :param min_battles: 每个模型至少需要的对战次数，才被视为"Effective ELO"。
        :return: Boolean
        """
        # 1. 获取所有配置中权重 > 0 的模型Key
        active_models = [
            k for k, v in self.model_configs.items() 
            if v.get("sampling_weight", 1) > 0
        ]
        
        if not active_models:
            return False

        # 2. 检查这些模型是否有足够的对战数据
        for model_key in active_models:
            stats = self.model_battle_stats.get(model_key, {"total": 0})
            if stats["total"] < min_battles:
                # 只要有一个模型对战数不足，就认为未准备好，继续强制遍历
                return False
        
        return True
    async def cal_reward(self, answer):
        """
        计算奖励并进化 Meta Prompt (Pairwise Comparison)。
        [修正]：解析 final_score 并确定 Local Value 所需的原始分数。
        [新增]：对过长的回答内容进行token截取。
        """
        print("计算奖励与进化 Meta Prompt 中 (PPRM Logic)...")
        
        # 1. 获取父节点
        parent_answer = self.fathers.get(answer)
        if parent_answer is None:
            # 根节点，没有父节点进行比较，给予默认分或自我评估
            # 这里为了简单，返回一个默认高分，或者需要单独的Self-Evaluation逻辑
            return 8.0, 8.0, "Root Node Initialized", "", False
        
        # 2. 截取过长的回答内容到前10000个token
        truncated_answer = self._truncate_to_tokens(answer, max_tokens=12000)
        truncated_parent_answer = self._truncate_to_tokens(parent_answer, max_tokens=12000)
            
        # 3. 获取 Prompt
        parent_system = self.node_meta_prompts.get(parent_answer, [self.system])[-1]
        current_system = self.node_meta_prompts.get(answer, [self.system])[-1]
        
        # 4. 随机交换位置 (防止位置偏差)
        is_swapped = random.random() > 0.5
        if is_swapped:
            r1, r2 = truncated_answer, truncated_parent_answer
            s1, s2 = current_system, parent_system
        else:
            r1, r2 = truncated_parent_answer, truncated_answer
            s1, s2 = parent_system, current_system

        # 5. 构建 Prompt
        query = self.query
        prior_experience = await self.get_prior_experience(self.rag)
        
        class PairwiseEvaluation(BaseModel):
            """用于存储成对比较评估结果及元系统进化的混合模型"""
            specific_criteria: str = Field(description="1. 特有评估标准：针对当前用户问题和需求类别的特有评估标准。")
            critique: str = Field(description="2. 批评：一步一步思考，对所提供的指令以及两个助手的回复，对每一条特定标准和评估维度都进行具体、详细的批评，必须明确指出哪一方更好。")
            weight_allocation: str = Field(description="3. 权重分配：通用和特有标准的权重分配，总和100%。")
            scoring_process: str = Field(description="4. 打分：计算每个维度的得分及加权平均分的过程。对每个评估维度单独评分，评分范围为1 到 10 分。1 分表示完全不符合要求，10 分表示完全符合要求。评分后，结合每个维度的权重。")
            final_score: list[float,float] = Field(description="5. 输出最终得分：计算加权平均得分，得出每个回答的综合得分，综合得分在 1-10 之间，格式必须为 [分数1,分数2]，其中分数1对应助手1，分数2对应助手2，综合得分在 1-10 之间。")
            new_experiences: list[str] = Field(
                description="6. 基于上述分析生成多个独立、不重复的，凝练地总结出核心问题和关键的改进原则或经验教训。这是从具体问题到通用解决方案的提炼。"
            )
            new_system_prompt: str = Field(description="7. 系统提示词进化：分析胜出者的关键成功因子和落败者的教训。基于此分析，生成一个全新的、完整的、更强的可以直接使用的System Prompt。")


        prompt = self.prompter.PAIRWISE_COMPARE_PROMPT.format(
            prompt=query,
            prior_experience=prior_experience,
            system_1=s1, system_2=s2,
            response_1=r1, response_2=r2,
        )

        data_template3 = copy.deepcopy(self.data_template)
        data_template3["messages"] = [{"role": "user", "content": prompt}]
        data_template3["model"] = self.api_llm.api_model

        # [新增] 根据模型配置动态设置采样参数
        judge_model_name = self.api_llm.api_model
        model_config = self.judge_model_configs.get(judge_model_name, self.judge_model_configs["default"])

        # 应用配置
        data_template3["temperature"] = model_config.get("temperature", 0.9)
        data_template3["top_p"] = model_config.get("top_p", 0.9)
        data_template3["extra_body"] = model_config.get("extra_body", {})

        print(f"[Judge Model Config] 使用模型: {judge_model_name}")
        print(f"  - temperature: {data_template3['temperature']}")
        print(f"  - top_p: {data_template3['top_p']}")
        print(f"  - extra_body: {data_template3['extra_body']}")


        # [新增] 重试机制
        MAX_RETRIES = 3
        evaluation = None
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"[Judge Attempt {attempt}/{MAX_RETRIES}] 开始评测...")
                evaluation, completion = await self.process_controller.receive_data_structural(
                    self.api_llm, data_template3, struct=PairwiseEvaluation
                )
                # [新增] 记录token使用
                self._record_token_usage(completion)

                # ### 类型检查 ###
                # 如果返回的是字符串，说明结构化解析失败
                if isinstance(evaluation, str):
                    print(f"[Error] 评测返回了非结构化字符串，内容片段: {evaluation[:50]}...")
                    last_error = "返回非结构化字符串"
                    if attempt < MAX_RETRIES:
                        print(f"  ⟳ 将在 1 秒后重试...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        print(f"  ✗ 所有 {MAX_RETRIES} 次尝试均失败")
                        return 0.0, 0.0, "<|_error_|>", "", is_swapped

                # 检查是否是 None
                if evaluation is None:
                    print(f"[Error] 评测返回 None")
                    last_error = "返回None"
                    if attempt < MAX_RETRIES:
                        print(f"  ⟳ 将在 1 秒后重试...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        print(f"  ✗ 所有 {MAX_RETRIES} 次尝试均失败")
                        return 0.0, 0.0, "<|_error_|>", "", is_swapped

                # ---> 成功获取评测结果，保存数据并跳出循环 <---
                print(f"  ✓ 评测成功（第 {attempt} 次尝试）")

                # 获取对战双方的模型信息
                answer_model_info = self.node_model_mapping.get(answer, {"model_key": "unknown", "model_name": "unknown", "organization": "unknown"})
                parent_model_info = self.node_model_mapping.get(parent_answer, {"model_key": "unknown", "model_name": "unknown", "organization": "unknown"})

                # 根据is_swapped确定chosen和rejected
                if is_swapped:
                    # answer作为第一个，parent_answer作为第二个
                    # 在评测中，chosen是获胜方，需要根据分数确定
                    if len(evaluation.final_score) >= 2:
                        score1, score2 = evaluation.final_score[0], evaluation.final_score[1]
                        if score1 > score2:
                            chosen_model_info, rejected_model_info = answer_model_info, parent_model_info
                        else:
                            chosen_model_info, rejected_model_info = parent_model_info, answer_model_info
                    else:
                        # 默认answer为chosen（因为它是新生成的节点）
                        chosen_model_info, rejected_model_info = answer_model_info, parent_model_info
                else:
                    # parent_answer作为第一个，answer作为第二个
                    if len(evaluation.final_score) >= 2:
                        score1, score2 = evaluation.final_score[0], evaluation.final_score[1]
                        if score1 > score2:
                            chosen_model_info, rejected_model_info = parent_model_info, answer_model_info
                        else:
                            chosen_model_info, rejected_model_info = answer_model_info, parent_model_info
                    else:
                        # 默认answer为chosen
                        chosen_model_info, rejected_model_info = answer_model_info, parent_model_info

                self._save_judge_data(prompt, evaluation, chosen_model_info, rejected_model_info)
                print(f"  >> 已保存评测数据到 {self.save_dataset_path}")
                break  # 成功，跳出重试循环

            except Exception as e:
                print(f"[Judge Attempt {attempt}/{MAX_RETRIES}] 异常: {e}")
                import traceback
                traceback.print_exc()
                last_error = str(e)

                if attempt < MAX_RETRIES:
                    print(f"  ⟳ 将在 1 秒后重试...")
                    await asyncio.sleep(1)
                    continue
                else:
                    print(f"  ✗ 评测失败: 所有 {MAX_RETRIES} 次尝试都失败，最后错误: {last_error}")
                    return 0.0, 0.0, "<|_error_|>", "", is_swapped

        # 最终检查：确保 evaluation 有效
        if evaluation is None or isinstance(evaluation, str):
            print(f"[Fatal] 评测最终失败: {last_error}")
            return 0.0, 0.0, "<|_error_|>", "", is_swapped

        # 5. 解析分数
        scores = evaluation.final_score
        if len(scores) < 2:
            scores = [0.0, 0.0]
            
        score1, score2 = scores[0], scores[1]
        
        # 还原分数归属
        if is_swapped:
            child_score = score1
            parent_score = score2
        else:
            child_score = score2
            parent_score = score1
            
        await self.optimize_experience_library(self.rag, evaluation.new_experiences)
        return child_score, parent_score, evaluation.critique, evaluation.new_system_prompt, is_swapped


    async def sampling_reward(self, answer):
        """
        [核心修改]：融合 Local Value (PPRM Bradley-Terry) 和 Global Value (EBC)。
        """
        if answer not in self.to_explore_reward:
            self.to_explore_reward[answer] = []
        if answer not in self.evaluations_bank:
            self.evaluations_bank[answer] = []
        if answer not in self.node_meta_prompts:
            self.node_meta_prompts[answer] = []

        # 1. 调用 PPRM 获取原始分数
        child_score, parent_score, judge_text, new_prompt, is_swapped = await self.cal_reward(answer)
        
        if judge_text == "<|_error_|>":
            return "<|_error_|>", 0.0

        if judge_text == "Root Node Initialized":
            # [根节点特殊处理逻辑]
            q_global = 1.0 
            self.ebc_global_scores[answer] = q_global
            q_local = 0.5 
            self.local_values[answer] = q_local
            
            quality_score_norm = self.ebc_alpha * q_global + (1 - self.ebc_alpha) * q_local
            diversity_score_norm = 1.0 # 根节点多样性设为1
            
            final_reward = quality_score_norm * self.max_reward
            if self.use_diversity_fusion:
                 final_reward = final_reward * diversity_score_norm

            print(f"--> [Root Node] Init Reward: {final_reward:.4f}")
            
            # [关键修复点]：必须在这里存入 evaluations_bank，否则 process_results 会报错
            self.evaluations_bank[answer].append({
                "final_reward": final_reward,
                "quality_score": quality_score_norm,
                "diversity_score": diversity_score_norm,
                "judge": "Root Init",
                "is_fused": self.use_diversity_fusion
            })
            
            self.to_explore_reward[answer].append(final_reward)
            
            return "Root", final_reward

        # 2. 计算局部价值 (Local Value) - Bradley-Terry Proxy
        # 使用 Softmax 将分数差转换为概率
        # Q_l = exp(S_child) / (exp(S_child) + exp(S_parent))
        try:
            # 缩放分数以避免溢出，假设分数 1-10，可以直接用
            exp_child = math.exp(child_score)
            exp_parent = math.exp(parent_score)
            q_local = exp_child / (exp_child + exp_parent)
        except OverflowError:
            q_local = 1.0 if child_score > parent_score else 0.0

        self.local_values[answer] = q_local
        print(f"局部价值 (Bradley-Terry): {q_local:.4f} (Child: {child_score}, Parent: {parent_score})")

        # 3. 更新 EBC 关系 (用于计算 Global Value)
        parent_node = self.fathers.get(answer)
        margin = 0.0 # 设定胜负阈值
        if child_score > parent_score + margin:
            self.pairwise_relations.append((answer, parent_node))
            print("EBC Relation: Child > Parent")
        elif parent_score > child_score + margin:
            self.pairwise_relations.append((parent_node, answer))
            print("EBC Relation: Parent > Child")
        
        # 4. 重新计算所有节点的 Global Value (EBC)
        # 注意：在大型树中这可能耗时，但在 <100 节点的树中很快
        self.calculate_ebc_scores()
        q_global = self.ebc_global_scores.get(answer, 0.5) # 默认为中位数
        # [指标 A] 纯质量分数 (归一化 0~1)
        # Quality = alpha * Qg + (1-alpha) * Ql
        quality_score_norm = self.ebc_alpha * q_global + (1 - self.ebc_alpha) * q_local
        # --- [新增] Darling 论文核心：计算多样性系数 ---
        # 计算当前回答相对于历史回答的多样性 (0.0 ~ 1.0)
        diversity_score_norm = await self._calculate_semantic_diversity(answer)
        print(f"Diversity Score (semantic): {diversity_score_norm:.4f}")
        # [决策] 是否融合
        if self.use_diversity_fusion:
            # 融合模式：Darling 乘法公式
            # Final = Quality * Diversity * Max_Reward
            final_reward = quality_score_norm * diversity_score_norm * self.max_reward * 10
            print(f"--> [Fusion ON] Final Reward: {final_reward:.4f}")
        else:
            # 独立模式：仅使用质量分作为奖励
            # Final = Quality * Max_Reward
            final_reward = quality_score_norm * self.max_reward
            print(f"--> [Fusion OFF] Final Reward: {final_reward:.4f} (Diversity ignored in UCB)")
        
        print(f"Final Reward: {final_reward:.4f} (Qg: {q_global:.4f}, Ql: {q_local:.4f})")

        # 6. 存储与更新
        self.to_explore_reward[answer].append(final_reward)
        #self.evaluations_bank[answer].append({"reward": final_reward, "judge": judge_text})
        self.evaluations_bank[answer].append({
            "final_reward": final_reward,    # 实际用于 UCB 的分
            "quality_score": quality_score_norm, # 独立的质量指标
            "diversity_score": diversity_score_norm, # 独立的多样性指标
            "judge": judge_text,
            "is_fused": self.use_diversity_fusion
        })
        # 存储进化后的 Prompt (如果有)
        if new_prompt:
            self.node_meta_prompts[answer].append(new_prompt)
            self.evolved_meta_prompt = new_prompt
        else:
            # 如果没生成，沿用父节点的
             self.node_meta_prompts[answer].append(self.node_meta_prompts.get(answer, [""])[0])

        return judge_text, final_reward
    
    async def _calculate_semantic_diversity(self, current_node: str):
        if not self.answers_list:
            return 1.0
            
        total_similarity = 0.0
        count = 0
        
        for other_node in self.answers_list:
            if other_node == current_node: continue
            truncated_current_node = self._truncate_to_tokens(current_node, max_tokens=10000)
            truncated_other_node = self._truncate_to_tokens(other_node, max_tokens=10000)
            # 使用 RAG 的 calculate_similarity
            sim = await self.rag.calculate_similarity(truncated_current_node, truncated_other_node)
            total_similarity += sim
            count += 1
            
        if count == 0: return 1.0
        
        avg_similarity = total_similarity / count
        # 距离 = 1 - 相似度
        diversity = 1.0 - avg_similarity 
        return max(0.0, diversity)

    async def update_ucb(self, C: float = 1.4, leaf_bonus: float = 1e-6):
        """
        更新 UCB 值。
        [核心还原]: 实现了  Em-Mcts 论文 Section 2.2 的 Backpropagation phase。
        公式: Q(s_i) = (1 - gamma)Q(s_i) + gamma * Q(s')
        """
        # [新增] 记录UCB更新开始
        self._record_operation("ucb_update_start", {
            "iteration": self.iter,
            "node_count": len(self.to_explore),
            "C": C,
            "gamma": self.gamma
        })
        
        # 1. 重新计算一次全局 EBC，确保图结构是最新的
        self.calculate_ebc_scores()
        
        # 初始化 Q_values 存储结构 (如果还没有的话)
        if not hasattr(self, 'Q_values'):
            self.Q_values = {}

        # --- 阶段 1: 计算所有节点的基础评估值 (Evaluation Phase) ---
        # 这一步对应论文中的 Evaluation: Q(s') = alpha * Qg + (1-alpha) * Ql
        # 我们先计算出每个节点当下的"静态"价值
        
        # 预计算最大原始奖励用于归一化 (防止除零)
        all_rewards = [np.mean(r) for r in self.to_explore_reward.values() if r]
        max_raw = max(all_rewards) if all_rewards else 1.0
        if max_raw == 0: max_raw = 1.0

        current_iter_q_estimates = {}

        for node in self.to_explore:
            # 获取评分组件
            q_g = self.ebc_global_scores.get(node, 0.0) # Global Value (EBC)
            q_l = self.local_values.get(node, 0.5)      # Local Value (Bradley-Terry)
            
            # 动态混合分数 (归一化状态 [0,1])
            r_c_norm = self.ebc_alpha * q_g + (1 - self.ebc_alpha) * q_l
            
            # 还原到原始奖励量级 (假设 max_reward=10)
            r_c = r_c_norm * 10.0 
            
            # 更新/初始化该节点的 Q 值
            # 注意：如果是新节点，直接使用计算值；如果是老节点，这里先作为基准
            if node not in self.Q_values:
                self.Q_values[node] = r_c
            
            # 暂存一下当前的估值，用于调试或逻辑判断
            current_iter_q_estimates[node] = r_c

        # --- 阶段 2: 反向传播 (Backpropagation Phase) [核心还原] ---
        # 论文公式: Q(si) = (1 - gamma)Q(si) + gamma * Q(s')
        # 我们使用倒序遍历 (Reversed) 来模拟从子节点 (Latest/Leaf) 向父节点 (Root) 的传播
        # 这样确保深层的优异表现能传递回上层节点
        
        print(f"\n--- 执行反向传播 (Gamma: {self.gamma}) ---")
        # 假设 answers_list 是按生成顺序排列的，倒序即从最新的子节点开始
        backprop_operations = []
        for node in reversed(self.answers_list):
            parent = self.fathers.get(node)
            
            # 如果有父节点，且父节点也在我们的探索列表中
            if parent and parent in self.Q_values:
                q_child = self.Q_values[node]
                q_parent_old = self.Q_values[parent]
                
                # [核心公式应用]
                # 父节点的价值 = (1 - gamma) * 原价值 + gamma * 子节点价值
                q_parent_new = (1 - self.gamma) * q_parent_old + self.gamma * q_child
                
                self.Q_values[parent] = q_parent_new
                
                # 记录显著变化
                if abs(q_parent_new - q_parent_old) > 0.1:
                    backprop_operations.append({
                        "child": node[:20] + "...",
                        "parent": parent[:20] + "...",
                        "old_q": q_parent_old,
                        "new_q": q_parent_new,
                        "change": q_parent_new - q_parent_old
                    })
                    print(f"Propagated: {node[:10]}... -> {parent[:10]}... | Parent Q: {q_parent_old:.2f} -> {q_parent_new:.2f}")

        # --- 阶段 3: 计算 UCB (Selection Phase Preparation) ---
        # 使用更新后的 self.Q_values 计算 UCB
        
        debug_stats = []
        for node in self.to_explore:
            # 使用反向传播更新后的 Q 值
            Q_val = self.Q_values.get(node, 0.0)
            
            # 获取访问次数 N_c
            rewards_list = self.to_explore_reward.get(node, [])
            N_c = len(rewards_list)
            
            # 获取父节点访问次数 N_n
            parent = self.fathers.get(node)
            if parent:
                parent_rewards = self.to_explore_reward.get(parent, [])
                N_n = len(parent_rewards)
                if N_n == 0: N_n = 1
            else:
                N_n = self.iter 

            # UCB 公式计算
            if N_c == 0:
                 ucb_value = Q_val + 0.1 # 未探索加成
            else:
                 ucb_value = Q_val + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))
            
            self.ucb_bank[node] = ucb_value
            
            q_g = self.ebc_global_scores.get(node, 0.0)
            q_l = self.local_values.get(node, 0.5)
            debug_stats.append((node, ucb_value, Q_val, q_g, q_l, N_c))

        # [新增] 记录UCB更新完成
        self._record_operation("ucb_update_complete", {
            "iteration": self.iter,
            "total_ucb_nodes": len(self.ucb_bank),
            "backprop_changes": len(backprop_operations),
            "avg_ucb": sum(self.ucb_bank.values()) / len(self.ucb_bank) if self.ucb_bank else 0
        })

        # 打印调试信息
        debug_stats.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 UCB Nodes (Iter {self.iter}) [After Backprop]:")
        print(f"{'Node':<20} | {'UCB':<8} | {'Q(Prop)':<8} | {'Qg':<6} | {'Ql':<6} | {'Visits':<6} | {'Model'}")
        print("-" * 100)
        for d in debug_stats[:5]:
            # d[0]现在是完整的节点文本，需要截断显示
            node_str = d[0][:20].replace('\n', ' ')
            # 获取节点对应的模型信息，使用完整节点文本作为键
            model_info = self.node_model_mapping.get(d[0], {"model_name": "unknown"})
            model_name = model_info.get("model_name", "unknown")
            
            # [DEBUG] 如果模型是unknown，尝试查找空结果的模型映射
            if model_name == "unknown":
                print(f"[DEBUG] 未找到节点 '{d[0][:30] if d[0] else 'EMPTY_STRING'}...' 的模型映射")
                print(f"[DEBUG] 节点长度: {len(d[0])}")
                print(f"[DEBUG] 节点repr: {repr(d[0][:50])}")
                
                # 检查是否有空结果的临时键
                empty_keys = [k for k in self.node_model_mapping.keys() if k.startswith("<EMPTY_RESULT_")]
                if empty_keys:
                    print(f"[DEBUG] 发现空结果临时键: {empty_keys}")
                    # 使用最新的空结果键
                    latest_empty_key = max(empty_keys, key=lambda x: int(x.split('_')[-1].rstrip('>')))
                    empty_model_info = self.node_model_mapping[latest_empty_key]
                    model_name = f"{empty_model_info.get('model_name', 'unknown')} (EMPTY)"
                    print(f"[DEBUG] 空结果对应模型: {model_name}")
                else:
                    print(f"[DEBUG] 映射字典中的所有键: {list(self.node_model_mapping.keys())[:5]}")
                    print(f"[DEBUG] 节点是否在映射中: {d[0] in self.node_model_mapping}")
            
            print(f"{node_str:<20} | {d[1]:<8.2f} | {d[2]:<8.2f} | {d[3]:<6.2f} | {d[4]:<6.2f} | {d[5]:<6} | {model_name}")

        if self.enable_state_tracking:
            self._capture_state_snapshot()

    async def filter_mature_node(self, max_expand=2):
        filtered_to_explore = [
            node for node in self.to_explore
            if len(self.childs.get(node, [])) < max_expand
        ]
        return filtered_to_explore

    def get_best_explore_from_ucb(self, to_explore):
        best_node = None
        highest_ucb = float('-inf')
        for node in to_explore:
            ucb_value = self.ucb_bank.get(node, 0)
            if ucb_value > highest_ucb:
                highest_ucb = ucb_value
                best_node = node
        return best_node
    
    async def system_create_expert_algorithm(self,choose_llm: LLM_Core, question,roleplay=False):
        prompt = self.prompter.Create_Expert_Prompt.format(question=question)
        data_template2 = copy.deepcopy(self.data_template)
        data_template2["messages"] = [
                                {"role": "user", "content": prompt}]
        data_template2["model"] = choose_llm.api_model
        L_instruction = await self.process_controller.receive_data(choose_llm,data_template2)
        if "[角色描述]：" in L_instruction:
            L_instruction = L_instruction.split("[角色描述]：")[-1]
        return L_instruction
    
    def draw_tree_pyvis(self, filename="tree.html"):
        """使用pyvis绘制树结构并保存为HTML文件，并标记节点产生的迭代次数和平均分，颜色根据分值变化"""
        net = Network(directed=True, width="100%", height="600px", bgcolor="#222222", font_color="white")

        avg_reward = {node: np.mean(rewards) for node, rewards in self.to_explore_reward.items() if rewards}
        plottable_nodes = {node: score for node, score in avg_reward.items() if not np.isnan(score)}
        all_scores = list(plottable_nodes.values())
        min_score = min(all_scores) if all_scores else 0
        max_score = max(all_scores) if all_scores else 1

        node_iterations = {answer: i for i, answer in enumerate(self.answers_list)}

        # --- 1. 绘制节点 (保持原有逻辑和样式不变) ---
        for node, score in plottable_nodes.items():
            score_str = f"{score:.2f}"
            
            # 识别根节点
            is_root = self.fathers.get(node) is None

            if is_root:
                label = f"ROOT (Iter {node_iterations.get(node, 0)})"
                color = "#33FF57"
            else:
                iteration = node_iterations.get(node, "?")
                short_text = node[:30] + "..." if len(node) > 30 else node
                label = f"{short_text}\nIter {iteration}, Score: {score_str}"
                
                normalized_score = (score - min_score) / (max_score - min_score) if max_score != min_score else 0.5
                if normalized_score < 0.5:
                    r, g, b = 255, int(255 * (normalized_score * 2)), 0
                else:
                    r, g, b = int(255 * (1 - (normalized_score - 0.5) * 2)), 255, 0
                color = f"#{r:02x}{g:02x}{b:02x}"
            
            # 在标题中显示对应的meta-prompt
            node_meta = self.node_meta_prompts.get(node, "N/A")
            # 注意：这里取最后一条 meta prompt，如果是列表
            if isinstance(node_meta, list) and node_meta:
                 node_meta_str = node_meta[-1]
            else:
                 node_meta_str = str(node_meta)

            title = f"Full text: {node}\nAverage score: {score_str}\n---\nMeta-Prompt: {node_meta_str}"
            net.add_node(node, label=label, color=color, title=title)

        for node, father in self.fathers.items():
            if father and father in net.get_nodes() and node in net.get_nodes():
                net.add_edge(father, node, color="#FF5733")

        net.force_atlas_2based()
        net.show(filename, notebook=False)
        print(f"树结构已保存到 {filename}")

    async def main_loop(self, inputs):
        """主循环"""
        inputs_copy = copy.deepcopy(inputs)
        print("max iter:", self.max_iter)
        
        # [新增] 记录主循环开始
        self._record_operation("main_loop_start", {
            "inputs": inputs_copy,
            "max_iter": self.max_iter,
            "enable_state_tracking": self.enable_state_tracking
        })
        
        if "category" in inputs_copy: self.class_tag = inputs_copy.get("category", "")
        if "class_tag" in inputs_copy: self.class_tag = inputs_copy.get("class_tag", "")
        self.context = inputs_copy.get("context", "")
        self.system, self.query = inputs_copy["prompt"][0]["content"], inputs_copy["prompt"][1]["content"]
        if self.use_expert_prompt == True:
            choose_llm = self.api_llm2
            L_system = await self.system_create_expert_algorithm(choose_llm, question=self.query)
            print("L_system:",L_system)
            self.system = L_system
        self.default_system = self.system
        self.domain = inputs_copy.get("domain", "通用")
        self.uid = inputs_copy.get("uid","")

        # --- 1. 初始化 ---
        print("--- 初始化阶段 ---")
        
        # [新增] 在开始之前先更新一次ELO评分和权重
        self.update_elo_ratings()
        
        # [新增] 记录初始化完成
        self._record_operation("initialization_complete", {
            "domain": self.domain,
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "system": self.system[:100] + "..." if len(self.system) > 100 else self.system
        })
        
        first_answer, evolved_meta_for_first = await self.step(parent_node=None)
        if "<|_error_|>" in first_answer: return "<|_error_|>"
        
        await self._register_node_and_link_to_parent(first_answer, None)
        self.node_meta_prompts[first_answer] = [evolved_meta_for_first]
        
        # 初始评估
        judge, initial_reward = await self.sampling_reward(first_answer)
        if judge == "<|_error_|>": return "<|_error_|>"
        await self.update_ucb(C=1.4)
        MAX_ITER_RETRIES = 3
        # --- 2. 迭代搜索 ---
        for i in range(self.max_iter):
            print(f'\n--- 迭代 {i + 1} / {self.max_iter} ---')
            self.iter = i + 1
            
            # [新增] 记录迭代开始
            self._record_operation("iteration_start", {
                "iteration": self.iter,
                "total_nodes": len(self.answers_list),
                "explore_nodes": len(self.to_explore)
            })
            
            # [新增] 开始第一个rollout
            self.update_elo_ratings()
            filtered_to_explore = await self.filter_mature_node(max_expand=self.max_expand)
            node_to_expand = self.get_best_explore_from_ucb(filtered_to_explore)

            if not node_to_expand: 
                print("没有可探索的节点，结束循环。")
                # [新增] 记录提前结束
                self._record_operation("early_termination", {
                    "reason": "no_expandable_nodes",
                    "iteration": self.iter,
                    "total_iterations": self.max_iter
                })
                break
            
            print(f"选择节点进行扩展: {str(node_to_expand)[:50]}...")
            
            # [新增] 记录节点选择
            self._record_operation("node_selection", {
                "selected_node": str(node_to_expand)[:100] + "..." if len(str(node_to_expand)) > 100 else str(node_to_expand),
                "ucb_value": self.ucb_bank.get(node_to_expand, 0),
                "iteration": self.iter
            })
            
            new_answer, new_meta_prompt = await self.step(node_to_expand)

            if "<|_error_|>" in new_answer: 
                # [新增] 记录错误
                self._record_operation("generation_error", {
                    "parent_node": str(node_to_expand)[:100] + "..." if len(str(node_to_expand)) > 100 else str(node_to_expand),
                    "iteration": self.iter
                })
                continue
            if new_answer == node_to_expand: 
                self.max_expand += 1
                # [新增] 记录重复答案
                self._record_operation("duplicate_answer", {
                    "parent_node": str(node_to_expand)[:100] + "..." if len(str(node_to_expand)) > 100 else str(node_to_expand),
                    "iteration": self.iter,
                    "new_max_expand": self.max_expand
                })
                continue

            # 注册与链接
            await self._register_node_and_link_to_parent(new_answer, node_to_expand)
            if new_answer not in self.node_meta_prompts: self.node_meta_prompts[new_answer] = []
            self.node_meta_prompts[new_answer].append(new_meta_prompt)

            # 评估与反向传播
            judge, reward = await self.sampling_reward(new_answer)
            if judge == "<|_error_|>": return "<|_error_|>"
            
            # 更新 UCB
            await self.update_ucb(C=1.4)
            
            # [新增] 记录迭代完成
            self._record_operation("iteration_complete", {
                "iteration": self.iter,
                "new_node": str(new_answer)[:100] + "..." if len(str(new_answer)) > 100 else str(new_answer),
                "reward": reward,
                "total_nodes": len(self.answers_list)
            })
            
            # [新增] 自动保存检查
            if self.enable_state_tracking and self.auto_save_interval > 0 and self.iter % self.auto_save_interval == 0:
                print(f"[自动保存] 迭代 {self.iter} - 保存状态...")
                self.save_state()
                if self.enable_visualization:
                    self.create_interactive_visualization()
            
            # path = os.path.join(f"tree_iter_{i + 1}.html")
            # self.draw_tree_pyvis(path)
            
        # [新增] 记录主循环完成
        self._record_operation("main_loop_complete", {
            "total_iterations": self.iter,
            "total_nodes": len(self.answers_list),
            "final_ucb_values": len(self.ucb_bank)
        })
        
        # [新增] 最终保存状态和可视化
        if self.enable_state_tracking:
            print("[最终保存] 保存完整状态和可视化...")
            self.save_state()
            if self.enable_visualization:
                self.create_interactive_visualization()
        
        return await self.process_results(inputs_copy)
    
    def dialog_length(self, node):
        dialog_list = self.tools.parse_fields(node)
        return len(dialog_list)

    async def process_results(self, inputs, alpha: float = 0.9, rho: float = 0.1) -> List[Dict[str, any]]:
        """
        [完整版] 处理最终结果。
        1. 汇总所有节点的 Local Value 和 Global Value。
        2. 计算最终混合得分。
        3. 提取 Best (Chosen) 和 Worst (Rejected)。
        4. 组装包含 Chain-of-Thought 和 Meta Prompt 的完整数据字典。
        """
        try:
            input_dict = copy.deepcopy(inputs)
            
            # 1. 过滤无效节点 (必须有奖励记录)
            valid_nodes = [n for n in self.to_explore_reward if self.to_explore_reward[n]]
            if not valid_nodes:
                print("没有有效的奖励数据，无法选择结果。")
                return []

            # 2. 准备评分数据
            # 重新计算一次 EBC 以防万一
            q_g_scores_raw = self.calculate_ebc_scores()
            
            # 统计原始奖励均值 (用于记录日志，不参与最终排序)
            raw_rewards_mean = {n: np.mean(self.to_explore_reward[n]) for n in valid_nodes}
            
            final_scores = {}
            debug_info = []
                
            
            for node in valid_nodes:
                if self.domain == "心理" and rho > 0:
                    # 获取当前节点长度
                    length = self.dialog_length(node)
                    
                    # 获取所有 valid_nodes 的长度（用于归一化）
                    all_lengths = [self.dialog_length(n) for n in valid_nodes]
                    max_len = max(all_lengths)
                    min_len = min(all_lengths)
                # --- A. 获取 Local Score (Ql) ---
                # 优先使用 sampling_reward 中计算好的 Bradley-Terry 概率
                q_local = self.local_values.get(node)
                # 如果缺失 (理论上不应发生)，使用 Sigmoid 对原始分归一化作为兜底
                if q_local is None:
                    raw_val = raw_rewards_mean[node]
                    # 简单的 Sigmoid 映射: 1 / (1 + exp(-(x - 5))) 假设均分5分
                    q_local = 1 / (1 + np.exp(-(raw_val - 5.0)))

                # --- B. 获取 Global Score (Qg) ---
                q_global = q_g_scores_raw.get(node, 0.0)
                
                # --- C. 融合 Q(s) ---
                # alpha 控制 EBC 全局排名的权重
                final_score = alpha * q_global + (1 - alpha) * q_local

                if self.domain == "心理":
                    # 线性归一化到 [0, 1]
                    # 避免除零：如果所有长度相同，则长度分数为 0.5
                    if max_len == min_len:
                        norm_length_score = 0.5
                    else:
                        # 线性归一化到 [0, 1]
                        norm_length_score = (length - min_len) / (max_len - min_len)
                    #final_score = self.dialog_length(node) * final_score
                    final_score = (1 - rho) * final_score + rho * norm_length_score
                final_scores[node] = final_score
                
                debug_info.append({
                    "node": node,
                    "final": final_score,
                    "q_g": q_global,
                    "q_l": q_local,
                    "raw_mean": raw_rewards_mean[node],
                    #"length": self.dialog_length(node)  # store length for later use
                })

            # 3. 排序与选择
            # 按最终得分降序排列
            sorted_nodes = sorted(debug_info, key=lambda x: x['final'], reverse=True)
            
            # 打印 Top 3 和 Bottom 1 用于调试
            print(f"\n--- 最终结果排序 (Top 3) ---")
            for info in sorted_nodes[:3]:
                print(f"Node: {info['node'][:30]}... | Final: {info['final']:.4f} (Qg: {info['q_g']:.2f}, Ql: {info['q_l']:.2f})")
            
            # 选择最佳和最差
            best_node_info = sorted_nodes[0]
            worst_node_info = sorted_nodes[-1]
            
            best_node = best_node_info['node']
            worst_node = worst_node_info['node']
            
            # 防御性检查：如果只有一个节点，或最佳最差相同
            if best_node == worst_node and len(sorted_nodes) > 1:
                worst_node_info = sorted_nodes[-1] # 确保取列表最后一个
                worst_node = worst_node_info['node']

            # 4. 数据组装
            # 提取 Thinking (Chain of Thought)
            c_reasoning = self.thinks_bank.get(best_node, "")
            j_reasoning = self.thinks_bank.get(worst_node, "")
            
            # 提取生成该节点时使用的 System Prompt (Meta Prompt)
            # 列表取最后一个 [-1] 代表当前生效的
            chosen_meta_prompt = self.node_meta_prompts.get(best_node, [self.system])[-1]
            final_query = self.query

            # [新增] 获取节点对应的模型信息
            chosen_model_info = self.node_model_mapping.get(best_node, {"model_name": "unknown", "organization": "unknown"})
            rejected_model_info = self.node_model_mapping.get(worst_node, {"model_name": "unknown", "organization": "unknown"})

            # 构建 Chosen 内容块
            chosen_msg = {
                "role": "assistant", 
                "content": best_node, 
                "reasoning_content": c_reasoning,
                # [新增] 绑定模型信息
                "model_info": chosen_model_info
            }
            
            # 构建 Rejected 内容块
            rejected_msg = {
                "role": "assistant", 
                "content": worst_node, 
                "reasoning_content": j_reasoning,
                # [新增] 绑定模型信息
                "model_info": rejected_model_info
            }
            # 在组装 output_dict 之前，获取最佳节点的详细评估信息
            # 取 evaluations_bank 中最后一次评估记录
            best_eval_info = self.evaluations_bank.get(best_node, [{}])[-1]
            worst_eval_info = self.evaluations_bank.get(worst_node, [{}])[-1]
            output_dict = {
                "prompt": [
                    {"role": "system", "content": self.default_system + "\n" + chosen_meta_prompt},
                    {"role": "user", "content": final_query}
                ],
                "chosen": [chosen_msg],
                "rejected": [rejected_msg],
                # --- [新增] 独立指标输出 ---
                "metrics": {
                    "use_diversity_fusion": self.use_diversity_fusion,
                    "chosen_quality": best_eval_info.get("quality_score", 0),
                    "chosen_diversity": best_eval_info.get("diversity_score", 0),
                    "chosen_final_reward": best_eval_info.get("final_reward", 0),
                    "rejected_quality": worst_eval_info.get("quality_score", 0),
                    "rejected_diversity": worst_eval_info.get("diversity_score", 0),
                },
                # 详细分数记录
                "chosen_Q_score": best_node_info['final'],
                "rejected_Q_score": worst_node_info['final'],
                "chosen_raw_reward": best_node_info['raw_mean'],
                "rejected_raw_reward": worst_node_info['raw_mean'],
                "Q_score_diff": best_node_info['final'] - worst_node_info['final'],
                # 辅助信息
                "domain": self.domain,
                "category": self.class_tag,
                "uid": self.uid,
                "model_config": {
                    "alpha": alpha,
                    "strategy": " Em-Mcts (EBC + Bradley-Terry) + Arena Model Selection",
                    # [新增] 模型信息汇总
                    "chosen_model": chosen_model_info,
                    "rejected_model": rejected_model_info,
                    # [新增] 模型使用统计
                    "model_usage_stats": self.get_model_usage_stats()
                },
                # [新增] Token使用统计
                "token_usage": {
                    "total_prompt_tokens": self.total_prompt_tokens,
                    "total_reasoning_tokens": self.total_reasoning_tokens,
                    "total_completion_tokens": self.total_completion_tokens,
                    "total_tokens": self.total_tokens,
                    "total_api_calls": self.total_api_calls
                }
            }

            # 补全原 input_dict 中可能存在的其他额外字段
            for key, value in inputs.items():
                if key not in output_dict and key != "prompt":
                    output_dict[key] = value

            return [output_dict]

        except Exception as e:
            print(f"处理结果时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            return []

# --- 异步运行入口 ---
async def run_llm_query():
    # 模拟初始化逻辑
    tokenizer_path = "Qwen2.5-7B-Instruct-AWQ"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except:
        tokenizer = None # Mock
        
    query = {}
    query["prompt"] = [{"role":"system", "content": "You are a helpful assistant."},
                       {"role":"user", "content": """最近感到很焦虑，如何缓解焦虑？"""}]
    llm = LLM_Core(
        tokenizer,
        use_async=True,
        api_model="gpt-4.1-2025-04-14",
        base_url="<your-base-url>",
        api_key='<your-api-key>')
    rag = await AsyncFaissRAG.create(api_url="http://localhost:60046/emb/v1")
    
    # [新增] 启用状态追踪和可视化的示例
    explorer = LLMExplorer_Socrates(
        llm=llm, 
        rag=rag, 
        max_iter=2, 
        use_diversity_fusion=False,
        # [新功能] 启用状态追踪和可视化
        enable_state_tracking=True,           # 开启状态记录
        state_save_path="./rollout_data",     # 状态保存目录
        auto_save_interval=1,                 # 每1次迭代自动保存
        enable_visualization=True,              # 开启可视化
        use_expert_prompt=True
    )
    
    print("🚀 开始 Em-Mcts Arena树搜索 (已启用状态追踪)")
    dicts = await explorer.main_loop(query)
    print("✅ 搜索完成！")
    
    print(f"📊 最终结果: {dicts[0]}")
    
    # [新增] 演示状态恢复功能
    print("\n🔄 演示状态恢复功能...")
    if hasattr(explorer, 'state_file') and explorer.state_file:
        # 创建新的探索器实例
        new_explorer = LLMExplorer_Socrates(
            llm=llm, 
            rag=rag, 
            max_iter=8,  # 可以设置更多迭代
            use_diversity_fusion=True,
            enable_state_tracking=True,
            state_save_path="./rollout_data_continued",
            auto_save_interval=2,
            enable_visualization=True
        )
        
        # 加载之前的状态
        if new_explorer.load_state(explorer.state_file):
            print("✅ 状态恢复成功！可以继续rollout...")
            
            # 继续搜索更多迭代
            print("🔄 继续进行树搜索...")
            continued_dicts = await new_explorer.main_loop(query)
            print("✅ 续搜索完成！")
            print(f"📊 续搜索结果: {continued_dicts[0]}")
        else:
            print("❌ 状态恢复失败")
    
    # [新增] 输出文件位置信息
    if hasattr(explorer, 'visualization_file') and explorer.visualization_file:
        print(f"\n📈 可视化文件已生成: {explorer.visualization_file}")
        print(f"📝 操作日志文件: {explorer.operation_file}")
        print(f"💾 状态文件: {explorer.state_file}")
        print(f"\n🌐 打开 {explorer.visualization_file} 查看交互式可视化！")

# [新增] 独立的状态恢复示例函数
async def demo_state_recovery(state_file_path: str):
    """演示如何从保存的状态恢复并继续rollout"""
    print(f"🔄 从状态文件恢复: {state_file_path}")
    
    # 初始化LLM和RAG（实际项目中应该保持一致）
    llm = LLM_Core(
        None,  # tokenizer
        use_async=True,
        api_model="gpt-4.1-nano-2025-04-14",
        base_url='<your-base-url>',
        api_key='<your-api-key>'
    )
    rag = await AsyncFaissRAG.create() #api_url="http://172.21.30.231:60046/emb/v1"
    
    # 创建探索器并恢复状态
    explorer = LLMExplorer_Socrates(
        llm=llm, 
        rag=rag, 
        max_iter=10,  # 可以设置更多迭代继续搜索
        use_diversity_fusion=True,
        enable_state_tracking=True,
        enable_visualization=True
    )
    
    if explorer.load_state(state_file_path):
        print(f"✅ 状态恢复成功！当前迭代: {explorer.iter}")
        print(f"📊 当前节点数: {len(explorer.answers_list)}")
        
        # 构建继续搜索所需的inputs（从状态中恢复）
        inputs = {
            "prompt": [
                {"role": "system", "content": explorer.system},
                {"role": "user", "content": explorer.query}
            ],
            "domain": explorer.domain,
            "class_tag": explorer.class_tag
        }
        
        # 继续搜索
        print("🚀 继续搜索...")
        results = await explorer.main_loop(inputs)
        print("✅ 继续搜索完成！")
        return results
    else:
        print("❌ 状态恢复失败")
        return None

if __name__ == "__main__":
    try:
        asyncio.run(run_llm_query())
    except Exception as e:
        print(f"Error: {e}")