#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM工具模块 - FridayClient封装器
支持多API key并行请求
"""

import os
import sys
import json
import time
import logging
import threading
import random
import yaml
import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
from threading import Lock
import requests
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
DEFAULT_LLM_API_KEYS = [
    key.strip()
    for key in os.getenv("LLM_API_KEYS", os.getenv("LLM_API_KEY", "")).split(",")
    if key.strip()
]


# ============================================================================
# 配置管理
# ============================================================================

class ConfigManager:
    """配置管理器，负责从YAML文件加载配置"""

    def __init__(self, config_path: str = "model_rpm.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    return yaml.safe_load(file)
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'model_rpm': {'default': 20},
            'api_config': {
                'default_rpm': 20,
                'base_url': DEFAULT_LLM_BASE_URL,
                'api_keys': DEFAULT_LLM_API_KEYS
            }
        }

    def get_api_keys(self) -> List[str]:
        """获取API keys列表"""
        api_keys = self.config.get('api_config', {}).get('api_keys', DEFAULT_LLM_API_KEYS)
        if isinstance(api_keys, str):
            api_keys = [key.strip() for key in api_keys.split(",") if key.strip()]
        return api_keys

    def get_default_rpm(self) -> int:
        """获取默认RPM"""
        return self.config.get('api_config', {}).get('default_rpm', 20)

    def get_base_url(self) -> str:
        """获取API基础URL"""
        return self.config.get('api_config', {}).get('base_url', DEFAULT_LLM_BASE_URL)

    def get_model_rpm(self, model_name: str) -> int:
        """获取指定模型的RPM限制"""
        model_rpm_config = self.config.get('model_rpm', {})
        return model_rpm_config.get(model_name, model_rpm_config.get('default', 20))


# 全局配置管理器实例
_config_manager = None

def get_config_manager(config_path: str = "model_rpm.yaml") -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


# ============================================================================
# API Key分配管理
# ============================================================================

class ApiKeyManager:
    """管理API key分配和速率限制"""
    _api_key_assignments = {}  # 线程到API key的映射
    _rate_limiters = {}  # API key到速率限制器的映射
    _assignment_lock = Lock()  # 分配锁
    _rate_limiter_lock = Lock()  # 速率限制锁
    _next_index = 0  # 下一个分配的API key索引

    @classmethod
    def get_assigned_api_key(cls, api_keys: List[str]) -> str:
        """为当前线程获取固定分配的API key"""
        if not api_keys:
            logger.warning("API keys列表为空；请设置 LLM_API_KEY 或 LLM_API_KEYS")
            return ""

        # 使用线程ID和进程ID的组合作为标识符
        thread_id = threading.get_ident()
        process_id = os.getpid()
        worker_key = f"{process_id}_{thread_id}"

        # 如果当前worker还未分配API key，则分配一个
        if worker_key not in cls._api_key_assignments:
            with cls._assignment_lock:
                # 双重检查锁定
                if worker_key not in cls._api_key_assignments:
                    api_index = cls._next_index % len(api_keys)
                    assigned_api_key = api_keys[api_index]
                    cls._api_key_assignments[worker_key] = assigned_api_key
                    cls._next_index += 1

                    logger.info(f"🔗 Worker {worker_key} 分配到 API key ...{assigned_api_key[-6:]} (索引 {api_index})")

        return cls._api_key_assignments[worker_key]

    @classmethod
    def rate_limit_api_key(cls, api_key: str, rpm_per_key: int = 20):
        """对指定的API key进行速率限制"""
        if not api_key:
            return

        with cls._rate_limiter_lock:
            if api_key not in cls._rate_limiters:
                cls._rate_limiters[api_key] = {
                    'last_request_time': 0,
                    'rpm_per_key': rpm_per_key
                }

            rate_limiter = cls._rate_limiters[api_key]
            now = time.time()
            last_request_time = rate_limiter['last_request_time']
            rpm_per_key = rate_limiter['rpm_per_key']

            # 计算距离上次请求的时间间隔
            time_since_last = now - last_request_time

            # 计算最小请求间隔（秒）
            min_interval = 60.0 / rpm_per_key

            if time_since_last < min_interval:
                # 需要等待的时间
                wait_time = min_interval - time_since_last
                logger.debug(f"⏱️ API key ...{api_key[-6:]} 速率限制，等待 {wait_time:.2f} 秒")

                # 更新最后请求时间
                rate_limiter['last_request_time'] = now + wait_time

                # 等待
                time.sleep(wait_time)
            else:
                # 可以直接执行，更新最后请求时间
                rate_limiter['last_request_time'] = now


# ============================================================================
# Friday Client
# ============================================================================

class FridayClient:
    """OpenAI-compatible LLM client with optional API-key rotation."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        api_url: str = DEFAULT_LLM_BASE_URL,
        api_token: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 65536,
        timeout: int = 300,
        max_retries: int = 5,
        rpm: Optional[int] = None,
        config_path: str = "model_rpm.yaml",
        use_api_key_manager: bool = True,
        realtime_output_dir: Optional[str] = None
    ):
        """
        初始化Friday客户端

        Args:
            model_name: 模型名称
            api_url: API URL
            api_token: API令牌（可选，如果不提供则从配置文件加载）
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 请求超时时间
            max_retries: 最大重试次数
            rpm: 每分钟请求数限制（可选，如果不提供则从配置文件加载）
            config_path: 配置文件路径
            use_api_key_manager: 是否使用API key管理器（并行场景建议开启）
            realtime_output_dir: 实时保存输出的目录（可选）
        """
        self.model_name = model_name
        self.config_path = config_path
        self.use_api_key_manager = use_api_key_manager
        self.realtime_output_dir = realtime_output_dir
        self._request_counter = 0  # 请求计数器，用于生成唯一文件名

        # 加载配置
        self.config_manager = get_config_manager(config_path)

        # 设置API URL
        if not api_url.endswith('/chat/completions'):
            self.api_url = f"{api_url}/chat/completions"
        else:
            self.api_url = api_url

        # 设置API token
        if api_token:
            self.api_token = api_token
            self.api_keys = [api_token]  # 单个token模式
        else:
            # 从配置文件加载API keys
            self.api_keys = self.config_manager.get_api_keys()
            self.api_token = self.api_keys[0] if self.api_keys else ""

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        # 设置RPM限制
        if rpm is not None:
            self.rpm = rpm
        else:
            self.rpm = self.config_manager.get_model_rpm(model_name)

        # RPM限制相关（用于非API key管理器模式）
        self.last_request_time = 0
        self.request_interval = 60.0 / self.rpm if self.rpm else 0
        self._api_idx = 0  # 轮询选择 key 时的指针

        logger.info(f"✅ FridayClient初始化完成: {model_name}")
        logger.info(f"   - API keys数量: {len(self.api_keys)}")
        logger.info(f"   - RPM限制: {self.rpm}")
        logger.info(f"   - 使用API key管理器: {use_api_key_manager}")
        if self.realtime_output_dir:
            logger.info(f"   - 实时输出目录: {self.realtime_output_dir}")

    def _realtime_save(
        self,
        messages: List[Dict[str, Any]],
        response_text: str,
        cost_time: float,
        token_info: Dict[str, int],
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        实时保存LLM输出到本地文件

        Args:
            messages: 请求消息
            response_text: 响应文本
            cost_time: 请求耗时
            token_info: token使用信息
            extra_info: 额外信息（可选）
        """
        if not self.realtime_output_dir:
            return

        try:
            # 确保输出目录存在
            os.makedirs(self.realtime_output_dir, exist_ok=True)

            # 生成唯一文件名
            self._request_counter += 1
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"llm_output_{timestamp}_{self._request_counter:06d}.json"
            file_path = os.path.join(self.realtime_output_dir, filename)

            # 构建保存数据
            save_data = {
                "timestamp": timestamp,
                "model_name": self.model_name,
                "request_id": self._request_counter,
                "messages": self._sanitize_messages_for_save(messages),
                "response": response_text,
                "cost_time_seconds": round(cost_time, 3),
                "token_info": token_info,
            }

            if extra_info:
                save_data["extra_info"] = extra_info

            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 实时保存LLM输出: {file_path}")

        except Exception as e:
            logger.warning(f"实时保存LLM输出失败: {e}")

    def _sanitize_messages_for_save(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清理消息数据用于保存（移除敏感信息和base64图片）
        """
        sanitized = []
        for msg in messages:
            sanitized_msg = {"role": msg.get("role", "")}

            content = msg.get("content", "")
            if isinstance(content, str):
                sanitized_msg["content"] = content
            elif isinstance(content, list):
                # 多模态消息，移除base64图片数据
                sanitized_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            sanitized_content.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:image"):
                                # 标记有图片但不保存base64数据
                                sanitized_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": "[IMAGE_DATA_BASE64]"},
                                    "note": "Base64图片数据已移除"
                                })
                            else:
                                sanitized_content.append(item)
                    else:
                        sanitized_content.append(item)
                sanitized_msg["content"] = sanitized_content

            sanitized.append(sanitized_msg)

        return sanitized

    def single_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Tuple[str, float, Dict[str, int]]:
        """
        发送单次请求（支持文本和图片输入）

        Args:
            messages: 消息列表，格式:
                - 文本: [{"role": "user", "content": "..."}]
                - 多模态: [{"role": "user", "content": [
                    {"type": "text", "text": "..."},
                    {"type": "image_url", "image_url": {"url": "..."}}
                ]}]
            temperature: 温度参数（可选，覆盖默认值）
            max_tokens: 最大token数（可选，覆盖默认值）

        Returns:
            (response_text, cost_time, token_info)
            - response_text: LLM响应文本
            - cost_time: 请求耗时（秒）
            - token_info: token使用信息，包含 prompt_tokens, completion_tokens, total_tokens
        """
        start_time = time.time()

        # 确定使用哪个API key
        if self.use_api_key_manager and len(self.api_keys) > 1:
            # 使用API key管理器分配key（并行场景）
            current_api_key = ApiKeyManager.get_assigned_api_key(self.api_keys)
            ApiKeyManager.rate_limit_api_key(current_api_key, self.rpm)
        else:
            # 使用固定的API key（单线程场景）
            current_api_key = self.api_token
            # RPM限制
            if self.request_interval > 0:
                elapsed = time.time() - self.last_request_time
                if elapsed < self.request_interval:
                    sleep_time = self.request_interval - elapsed
                    time.sleep(sleep_time)

        # 构建请求
        headers = {
            "Authorization": f"Bearer {current_api_key}",
            "Content-Type": "application/json"
        }

        # 构建payload（使用简化格式，避免不支持的参数导致服务器错误）
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": False,
            # "thinking": {"type": "disabled"}  # 关闭 thinking 模式
        }

        # LLM API 使用直连，不走代理（美团API需要国内网络直连）
        # 显式清除环境变量中的代理设置，确保不走代理
        env_proxies = {}
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']
        for var in proxy_vars:
            if var in os.environ:
                env_proxies[var] = os.environ[var]
                del os.environ[var]

        logger.debug(f"已清除代理环境变量: {list(env_proxies.keys())}")

        proxies = {
            "http": None,
            "https": None
        }

        logger.debug(f"将使用直连，不走代理，proxies={proxies}")

        # 重试逻辑：优先在同一轮尝试其他可用 key；若全部限流则等待后继续，直到成功或遇到非限流错误累计超限
        last_error = None
        non_rate_limit_failures = 0
        while True:
            tried_keys = set()
            rate_limited_all = False
            while True:
                current_api_key = self._get_next_api_key(tried_keys)
                if not current_api_key:
                    rate_limited_all = True
                    break
                tried_keys.add(current_api_key)

                # 针对当前 key 做节流
                if self.use_api_key_manager and len(self.api_keys) > 1:
                    ApiKeyManager.rate_limit_api_key(current_api_key, self.rpm)
                else:
                    if self.request_interval > 0:
                        elapsed = time.time() - self.last_request_time
                        if elapsed < self.request_interval:
                            time.sleep(self.request_interval - elapsed)

                headers = {
                    "Authorization": f"Bearer {current_api_key}",
                    "Content-Type": "application/json"
                }

                try:
                    # 禁用SSL验证以解决SSL错误（仅用于内网环境）
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout,
                        verify=False,  # 禁用SSL验证
                        proxies=proxies
                    )

                    # 仅在非2xx且命中限流关键词时认定为限流；200 响应不视为限流
                    if response.status_code == 429 or (response.status_code >= 400 and self._is_rate_limit_error(response.text)):
                        masked_key = f"...{current_api_key[-6:]}" if current_api_key else "***"
                        logger.warning(
                            f"⏱️ API限流 (状态码: {response.status_code}, key: {masked_key})，尝试切换其他可用key"
                        )
                        continue  # 换下一个 key

                    response.raise_for_status()

                    result = response.json()

                    response_text = ""
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if choice.get("message") and "content" in choice["message"]:
                            response_text = choice["message"]["content"]

                    token_info = {
                        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": result.get("usage", {}).get("total_tokens", 0)
                    }

                    cost_time = time.time() - start_time
                    self.last_request_time = time.time()

                    # 实时保存输出
                    self._realtime_save(
                        messages=messages,
                        response_text=response_text,
                        cost_time=cost_time,
                        token_info=token_info
                    )

                    # 恢复被清除的代理环境变量
                    for var, value in env_proxies.items():
                        os.environ[var] = value

                    return response_text, cost_time, token_info

                except requests.exceptions.RequestException as e:
                    # 恢复被清除的代理环境变量
                    for var, value in env_proxies.items():
                        os.environ[var] = value
                    last_error = e
                    error_detail = str(e)
                    if hasattr(e, 'response') and e.response is not None:
                        try:
                            error_json = e.response.json()
                            error_detail = f"{str(e)} | 响应内容: {json.dumps(error_json, ensure_ascii=False)}"
                        except:
                            error_detail = f"{str(e)} | 响应文本: {e.response.text[:200]}"

                    # 检测连接错误和超时错误，视为临时性网络问题进行重试
                    is_network_error = isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError))
                    if is_network_error:
                        logger.warning(f"请求失败-网络错误: {error_detail}")
                        non_rate_limit_failures += 1
                        if non_rate_limit_failures >= self.max_retries:
                            logger.error(f"所有重试都失败: {last_error}")
                            raise Exception(f"LLM请求失败: {last_error}")
                        # 指数退避重试
                        wait_time = min(2 ** non_rate_limit_failures, 32)
                        logger.info(f"网络错误，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue

                    if self._is_rate_limit_error(error_detail):
                        logger.warning(
                            f"请求失败-限流 (key ...{current_api_key[-6:] if current_api_key else '***'})，尝试其他key"
                        )
                        continue  # 换下一个 key

                    # 检测 5xx 服务器错误，视为临时性错误进行重试
                    is_server_error = False
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        if 500 <= status_code < 600:
                            is_server_error = True
                            logger.warning(
                                f"请求失败-服务器错误 ({status_code})，进入重试等待..."
                            )

                    # 非限流错误计数
                    non_rate_limit_failures += 1
                    logger.warning(
                        f"请求失败 (非限流，第{non_rate_limit_failures}/{self.max_retries}次): {error_detail}"
                    )
                    if non_rate_limit_failures >= self.max_retries:
                        logger.error(f"所有重试都失败: {last_error}")
                        raise Exception(f"LLM请求失败: {last_error}")

                    # 5xx 错误使用指数退避重试
                    if is_server_error:
                        # 指数退避: 2, 4, 8, 16, 32 秒
                        wait_time = min(2 ** non_rate_limit_failures, 32)
                        logger.info(f"5xx服务器错误，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue

                    break  # 跳出内层，进入退避

            # 如果本轮所有 key 都限流，等待后继续（不限轮次，直到成功或非限流错误超限）
            # 增加退避时间：指数退避从2^4增加到2^6，并增加基础等待时间
            base_wait = 4.0  # 基础等待时间4秒
            exponential_wait = (2 ** min(non_rate_limit_failures, 6))  # 最大指数从4增加到6（64秒）
            sleep_time = base_wait + exponential_wait + random.uniform(0, 2)
            logger.info(
                f"所有可用key均限流，等待 {sleep_time:.2f} 秒后继续尝试"
            )
            time.sleep(sleep_time)

    def _get_next_api_key(self, tried_keys: set) -> Optional[str]:
        """轮询获取一个未尝试过的API key；若无则返回None。"""
        if not self.api_keys:
            return self.api_token
        for _ in range(len(self.api_keys)):
            key = self.api_keys[self._api_idx % len(self.api_keys)]
            self._api_idx += 1
            if key not in tried_keys:
                return key
        return None

    def _is_rate_limit_error(self, message: str) -> bool:
        """
        检测是否为限流相关错误
        """
        if not message:
            return False
        lowered = message.lower()
        return any(
            keyword in lowered
            for keyword in [
                "429",
                "rate limit",
                "too many requests",
                "达到使用量上限",
                "每分钟请求次数超过限制",
                "请求次数超过限制"
            ]
        )

    def batch_request(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, int]]]:
        """
        批量请求

        Args:
            messages_list: 多个消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            List of (response_text, cost_time, token_info)
        """
        results = []
        for messages in messages_list:
            result = self.single_request(messages, temperature, max_tokens)
            results.append(result)
        return results


def get_friday_client(
    model_name: str = DEFAULT_LLM_MODEL,
    rpm: Optional[int] = None,
    api_url: str = DEFAULT_LLM_BASE_URL,
    api_token: Optional[str] = None,
    config_path: str = "model_rpm.yaml",
    use_api_key_manager: bool = True,
    realtime_output_dir: Optional[str] = None
) -> FridayClient:
    """
    获取Friday客户端实例

    Args:
        model_name: 模型名称
        rpm: 每分钟请求数限制（可选，从配置文件读取）
        api_url: API URL
        api_token: API令牌（可选，从配置文件读取）
        config_path: 配置文件路径
        use_api_key_manager: 是否使用API key管理器（并行场景建议开启）
        realtime_output_dir: 实时保存输出的目录（可选）

    Returns:
        FridayClient实例
    """
    return FridayClient(
        model_name=model_name,
        api_url=api_url,
        api_token=api_token,
        rpm=rpm,
        config_path=config_path,
        use_api_key_manager=use_api_key_manager,
        realtime_output_dir=realtime_output_dir
    )


def get_multiple_friday_clients(
    model_name: str = DEFAULT_LLM_MODEL,
    num_clients: int = 3,
    rpm_per_client: Optional[int] = None,
    api_url: str = DEFAULT_LLM_BASE_URL,
    config_path: str = "model_rpm.yaml",
    realtime_output_dir: Optional[str] = None
) -> List[FridayClient]:
    """
    获取多个Friday客户端实例（用于并行请求）
    每个客户端会自动从API key池中分配不同的key

    Args:
        model_name: 模型名称
        num_clients: 客户端数量
        rpm_per_client: 每个客户端的RPM限制（可选，从配置文件读取）
        api_url: API URL
        config_path: 配置文件路径
        realtime_output_dir: 实时保存输出的目录（可选）

    Returns:
        FridayClient实例列表
    """
    clients = []
    for i in range(num_clients):
        client = FridayClient(
            model_name=model_name,
            api_url=api_url,
            api_token=None,  # 从配置文件读取
            rpm=rpm_per_client,
            config_path=config_path,
            use_api_key_manager=True,  # 启用API key管理器
            realtime_output_dir=realtime_output_dir
        )
        clients.append(client)
    return clients


# ============================================================================
# 向后兼容层 - LLMClient (异步接口包装)
# ============================================================================

class LLMClient:
    """
    LLMClient - 向后兼容的异步接口包装器
    包装FridayClient，提供异步接口以保持与旧代码的兼容性
    """

    def __init__(
        self,
        config=None,
        friday_client: Optional[FridayClient] = None,
        realtime_output_dir: Optional[str] = None
    ):
        """
        初始化LLMClient

        Args:
            config: 旧的LLMConfig对象（可选）
            friday_client: FridayClient实例（可选）
            realtime_output_dir: 实时保存输出的目录（可选）
        """
        if realtime_output_dir and not friday_client:
            # 如果提供了realtime_output_dir但没有friday_client，创建一个
            self.client = get_friday_client(realtime_output_dir=realtime_output_dir)
        elif friday_client:
            self.client = friday_client
        elif config:
            # 从旧的config创建FridayClient
            self.client = FridayClient(
                model_name=getattr(config, 'model', DEFAULT_LLM_MODEL),
                api_url=getattr(config, 'base_url', DEFAULT_LLM_BASE_URL),
                api_token=getattr(config, 'api_key', None),
                temperature=getattr(config, 'temperature', 0.7),
                max_tokens=getattr(config, 'max_tokens', 65536),
                use_api_key_manager=True
            )
        else:
            # 使用默认配置
            self.client = get_friday_client()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        异步生成文本响应（包装同步的FridayClient）

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（暂不支持）

        Returns:
            生成的文本
        """
        import asyncio

        # 构建消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 在executor中运行同步方法
        loop = asyncio.get_event_loop()
        response, _, _ = await loop.run_in_executor(
            None,
            lambda: self.client.single_request(messages, temperature, max_tokens)
        )

        return response

    async def generate_with_images(
        self,
        prompt: str,
        image_urls: List[str],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        多模态生成（图片+文本）
        使用OpenAI格式的多模态消息，支持视觉模型

        Args:
            prompt: 用户提示
            image_urls: 图片URL列表
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            生成的文本
        """
        import asyncio

        # 构建多模态消息内容
        content = [{"type": "text", "text": prompt}]
        for image_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        # 构建消息列表
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        # 在executor中运行同步方法
        loop = asyncio.get_event_loop()
        response, _, _ = await loop.run_in_executor(
            None,
            lambda: self.client.single_request(messages, temperature, max_tokens)
        )

        return response

    async def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_concurrent: int = 5
    ) -> List[str]:
        """
        批量生成响应

        Args:
            prompts: 提示列表
            system_prompt: 共享的系统提示
            max_concurrent: 最大并发数（暂时忽略，使用顺序处理）

        Returns:
            响应列表
        """
        results = []
        for prompt in prompts:
            try:
                result = await self.generate(prompt, system_prompt)
                results.append(result)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        return results


__all__ = [
    "FridayClient",
    "get_friday_client",
    "get_multiple_friday_clients",
    "ConfigManager",
    "ApiKeyManager",
    "get_config_manager",
    "LLMClient"  # 向后兼容
]


# 测试代码
if __name__ == "__main__":
    # 测试FridayClient
    client = get_friday_client(DEFAULT_LLM_MODEL)

    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "1+1等于几？"}
    ]

    try:
        response, cost_time, token_info = client.single_request(messages)
        print(f"响应: {response}")
        print(f"耗时: {cost_time:.2f}秒")
        print(f"Token使用: {token_info}")
    except Exception as e:
        print(f"测试失败: {e}")
