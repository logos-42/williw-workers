"""
配置管理模块
"""
import os
from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径（可选）
        """
        self.config = {}
        
        # 从环境变量读取配置
        self.config['williw_api_url'] = os.environ.get('WILLIW_API_URL')
        self.config['huggingface_cache_dir'] = os.environ.get('HUGGINGFACE_CACHE_DIR')
        self.config['model_repository_path'] = os.environ.get('MODEL_REPOSITORY_PATH', './model_repository')
        self.config['server_port'] = int(os.environ.get('PORT', 8080))
        self.config['server_host'] = os.environ.get('HOST', '0.0.0.0')
        
        # 从配置文件读取（如果提供）
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self.config.update(file_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value


# 全局配置实例
config = Config()
