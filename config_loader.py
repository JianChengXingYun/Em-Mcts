"""
配置加载器 - 用于管理 gen、judge 和 emb 模型的配置
"""
import json
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """配置加载器类"""

    def __init__(self, config_path: str = "config.json"):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为 config.json
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_gen_models(self) -> Dict[str, Any]:
        """获取生成模型配置"""
        return self.config.get("gen_models", {})

    def get_judge_models(self) -> Dict[str, Any]:
        """获取评判模型配置"""
        return self.config.get("judge_models", {})
    
    def get_mem_models(self) -> Dict[str, Any]:
        """获取评判模型配置"""
        return self.config.get("mem_models", {})
    
    def get_emb_models(self) -> Dict[str, Any]:
        """获取嵌入模型配置"""
        return self.config.get("emb_models", {})

    def get_gen_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取指定的生成模型配置"""
        return self.get_gen_models().get(model_name)

    def get_judge_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取指定的评判模型配置"""
        return self.get_judge_models().get(model_name)

    def get_emb_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取指定的嵌入模型配置"""
        return self.get_emb_models().get(model_name)
    
    def get_mem_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取指定的记忆模型配置"""
        return self.get_mem_models().get(model_name)

    def get_api_config(self) -> Dict[str, Any]:
        """获取通用 API 配置"""
        return self.config.get("api_config", {})

    def get_default_base_url(self) -> str:
        """获取默认的 API 基础 URL"""
        return self.get_api_config().get("default_base_url", "http://localhost:6001/v1")

    def get_default_api_key(self) -> str:
        """获取默认的 API 密钥"""
        return self.get_api_config().get("default_api_key", "")

    def reload(self):
        """重新加载配置文件"""
        self.config = self._load_config()

    def save_config(self, config_path: Optional[str] = None):
        """保存配置到文件"""
        path = config_path or self.config_path
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)


# 全局配置加载器实例
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: str = "config.json") -> ConfigLoader:
    """获取全局配置加载器实例"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def init_config(config_path: str = "config.json"):
    """初始化全局配置加载器"""
    global _config_loader
    _config_loader = ConfigLoader(config_path)
