"""配置中心：YAML 配置 + 环境变量替换（单例）"""
import os
import re
from pathlib import Path
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def _resolve_env_vars(value: Any) -> Any:
    """递归解析配置值中的 ${ENV_VAR} 引用"""
    if isinstance(value, str):
        pattern = re.compile(r'\$\{(\w+)\}')
        matches = pattern.findall(value)
        for var_name in matches:
            env_value = os.getenv(var_name, "")
            if not env_value:
                raise ValueError(
                    f"环境变量 {var_name} 未设置！"
                    f"请复制 .env.example 为 .env 并填写。"
                )
            value = value.replace(f"${{{var_name}}}", env_value)
        return value
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


@lru_cache()
def load_config() -> dict:
    """加载 YAML 配置，自动替换 ${ENV_VAR} 引用"""
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _resolve_env_vars(config)


settings = load_config()
