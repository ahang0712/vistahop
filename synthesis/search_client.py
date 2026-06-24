"""
搜索客户端：提供多种真实数据来源的接入
支持: Google Search API, Bing Search API, Serper API, SerpAPI, 本地数据集
"""
import asyncio
import aiohttp
import json
import os
import hashlib
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import re
import os
import sys
from datetime import datetime

# 处理导入路径，支持模块和直接运行
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_grandparent_dir = os.path.dirname(os.path.dirname(_current_dir))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _grandparent_dir not in sys.path:
    sys.path.insert(0, _grandparent_dir)

SearchConfig = None
try:
    from .config import SearchConfig
except ImportError:
    try:
        from config import SearchConfig
    except ImportError:
        pass

# ============================================================
# 缓存配置
# ============================================================
# 缓存目录：项目根目录下的 cache 文件夹
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
WIKIDATA_CACHE_DIR = os.path.join(CACHE_DIR, "wikidata")
WIKIPEDIA_CACHE_DIR = os.path.join(CACHE_DIR, "wikipedia")

# 缓存永不过期
CACHE_EXPIRY = float('inf')


def _ensure_cache_dir():
    """确保缓存目录存在"""
    os.makedirs(WIKIDATA_CACHE_DIR, exist_ok=True)
    os.makedirs(WIKIPEDIA_CACHE_DIR, exist_ok=True)


def _get_cache_path(cache_dir: str, key: str, ext: str = ".json") -> str:
    """生成缓存文件路径"""
    # 使用 MD5 作为文件名（避免特殊字符问题）
    key_hash = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{key_hash}{ext}")


def _load_cached_data(cache_path: str) -> Optional[Dict]:
    """加载缓存数据"""
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # 检查是否过期
        cached_time = cache_data.get('_cached_at', 0)
        if time.time() - cached_time > CACHE_EXPIRY:
            return None

        return cache_data
    except Exception:
        return None


def _save_cache_data(cache_path: str, data: Dict, extra_fields: Dict = None):
    """保存缓存数据"""
    try:
        _ensure_cache_dir()
        cache_data = {
            '_cached_at': time.time(),
            'data': data
        }
        if extra_fields:
            cache_data.update(extra_fields)

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[缓存警告] 无法保存缓存: {e}")


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    images: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.images = self.images or []
        self.metadata = self.metadata or {}


@dataclass
class ImageSearchResult:
    """图片搜索结果"""
    url: str
    thumbnail_url: str
    title: str
    source_url: str
    width: int = 0
    height: int = 0


class BaseSearchClient(ABC):
    """搜索客户端基类"""

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """执行文本搜索"""
        pass

    @abstractmethod
    async def search_images(self, query: str, num_results: int = 5) -> List[ImageSearchResult]:
        """执行图片搜索"""
        pass


class SerperSearchClient(BaseSearchClient):
    """
    Serper API 搜索客户端
    官网: https://serper.dev/
    免费额度: 2500次/月
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://google.serper.dev"

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Google搜索"""
        url = f"{self.base_url}/search"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": num_results,
            "gl": "cn",  # 地区
            "hl": "zh-cn"  # 语言
        }

        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("organic", []):
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            metadata={"position": item.get("position")}
                        ))

                    return results
                else:
                    print(f"Serper搜索失败: {response.status}")
                    return []

    async def search_images(self, query: str, num_results: int = 5) -> List[ImageSearchResult]:
        """Google图片搜索"""
        url = f"{self.base_url}/images"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": num_results
        }

        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("images", []):
                        results.append(ImageSearchResult(
                            url=item.get("imageUrl", ""),
                            thumbnail_url=item.get("thumbnailUrl", ""),
                            title=item.get("title", ""),
                            source_url=item.get("link", ""),
                            width=item.get("imageWidth", 0),
                            height=item.get("imageHeight", 0)
                        ))

                    return results
                else:
                    return []


class BingSearchClient(BaseSearchClient):
    """
    Bing Search API 搜索客户端
    官网: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
    免费额度: 1000次/月
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.search_url = "https://api.bing.microsoft.com/v7.0/search"
        self.image_url = "https://api.bing.microsoft.com/v7.0/images/search"

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Bing网页搜索"""
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": num_results,
            "mkt": "zh-CN"
        }

        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.search_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("webPages", {}).get("value", []):
                        results.append(SearchResult(
                            title=item.get("name", ""),
                            url=item.get("url", ""),
                            snippet=item.get("snippet", "")
                        ))

                    return results
                else:
                    print(f"Bing搜索失败: {response.status}")
                    return []

    async def search_images(self, query: str, num_results: int = 5) -> List[ImageSearchResult]:
        """Bing图片搜索"""
        headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        params = {
            "q": query,
            "count": num_results
        }

        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.image_url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("value", []):
                        results.append(ImageSearchResult(
                            url=item.get("contentUrl", ""),
                            thumbnail_url=item.get("thumbnailUrl", ""),
                            title=item.get("name", ""),
                            source_url=item.get("hostPageUrl", ""),
                            width=item.get("width", 0),
                            height=item.get("height", 0)
                        ))

                    return results
                else:
                    return []


class SerpAPIClient(BaseSearchClient):
    """
    SerpAPI 搜索客户端
    官网: https://serpapi.com/
    免费额度: 100次/月
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Google搜索（通过SerpAPI）"""
        params = {
            "engine": "google",
            "q": query,
            "num": num_results,
            "api_key": self.api_key,
            "hl": "zh-cn",
            "gl": "cn"
        }

        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("organic_results", []):
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", "")
                        ))

                    return results
                else:
                    return []

    async def search_images(self, query: str, num_results: int = 5) -> List[ImageSearchResult]:
        """Google图片搜索（通过SerpAPI）"""
        params = {
            "engine": "google_images",
            "q": query,
            "num": num_results,
            "api_key": self.api_key
        }

        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("images_results", []):
                        results.append(ImageSearchResult(
                            url=item.get("original", ""),
                            thumbnail_url=item.get("thumbnail", ""),
                            title=item.get("title", ""),
                            source_url=item.get("link", ""),
                            width=item.get("original_width", 0),
                            height=item.get("original_height", 0)
                        ))

                    return results
                else:
                    return []


class WikipediaClient(BaseSearchClient):
    """
    Wikipedia API 搜索客户端
    免费无限制使用
    适合获取高质量知识内容

    支持多个镜像（中国友好）
    """

    # 镜像配置
    MIRRORS = {
        "zh": [
            "https://zh.wikipedia.org/w/api.php",  # 中文维基
            "https://wikimedia.org/api/rest_v1",   # Wikimedia CDN
        ],
        "en": [
            "https://en.wikipedia.org/w/api.php",  # 英文维基
            "https://wikimedia.org/api/rest_v1",   # Wikimedia CDN
        ]
    }

    def __init__(self, language: str = "zh", proxy: str = None):
        self.language = language
        self.base_url = f"https://{language}.wikipedia.org/w/api.php"
        self.fallback_url = "https://wikimedia.org/api/rest_v1"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # 如果没有传入proxy，尝试从环境变量获取
        self.proxy = proxy or os.environ.get("https_proxy") or os.environ.get("http_proxy")

    def _get_proxy(self) -> Optional[str]:
        """获取代理配置"""
        if self.proxy:
            return self.proxy
        # 尝试从环境变量获取
        return os.environ.get("https_proxy") or os.environ.get("http_proxy")

    @staticmethod
    def _pick_language_value(values: Dict[str, Dict[str, str]], preferred_languages: List[str]) -> str:
        for lang in preferred_languages:
            value = values.get(lang, {}).get("value", "")
            if value:
                return value
        for value in values.values():
            if isinstance(value, dict) and value.get("value"):
                return value.get("value", "")
        return ""

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """搜索Wikipedia文章"""
        # 生成缓存key
        cache_key = f"search_{query}_{num_results}_{self.language}"
        cache_path = _get_cache_path(WIKIPEDIA_CACHE_DIR, cache_key)

        # 先检查缓存
        cached = _load_cached_data(cache_path)
        if cached:
            print(f"   [缓存命中] Wikipedia搜索: {query}")
            return cached.get('data', [])

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
            "utf8": 1
        }

        proxy = self._get_proxy()
        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.base_url, params=params, headers=self.headers, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get("query", {}).get("search", []):
                        # 清理HTML标签
                        snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))

                        results.append(SearchResult(
                            title=item.get("title", ""),
                            url=f"https://{self.language}.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}",
                            snippet=snippet,
                            metadata={"pageid": item.get("pageid")}
                        ))

                    # 保存到缓存
                    _save_cache_data(cache_path, results)
                    return results
                else:
                    return []

    async def get_article_content(self, title: str) -> Optional[str]:
        """获取Wikipedia文章完整内容"""
        # 生成缓存key
        safe_title = title.replace(' ', '_').replace('/', '_')
        cache_key = f"content_{safe_title}_{self.language}"
        cache_path = _get_cache_path(WIKIPEDIA_CACHE_DIR, cache_key)

        # 先检查缓存
        cached = _load_cached_data(cache_path)
        if cached:
            print(f"   [缓存命中] Wikipedia内容: {title}")
            return cached.get('data', {}).get('extract', '')

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": 1,
            "format": "json"
        }

        proxy = self._get_proxy()
        timeout = aiohttp.ClientTimeout(total=60)  # 设置更长的超时时间
        async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
            async with session.get(self.base_url, params=params, headers=self.headers, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})

                    for page_id, page in pages.items():
                        if page_id != "-1":
                            extract = page.get("extract", "")
                            # 保存到缓存
                            _save_cache_data(cache_path, {'extract': extract, 'title': title})
                            return extract

                return None

    async def get_page_url(self, title: str) -> Optional[str]:
        """获取Wikipedia页面URL"""
        # 生成缓存key
        safe_title = title.replace(' ', '_').replace('/', '_')
        cache_key = f"url_{safe_title}_{self.language}"
        cache_path = _get_cache_path(WIKIPEDIA_CACHE_DIR, cache_key)

        # 先检查缓存
        cached = _load_cached_data(cache_path)
        if cached:
            print(f"   [缓存命中] Wikipedia URL: {title}")
            return cached.get('data', {}).get('url', '')

        params = {
            "action": "query",
            "titles": title,
            "prop": "info",
            "inprop": "url",
            "format": "json"
        }

        proxy = self._get_proxy()
        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.base_url, params=params, headers=self.headers, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})

                    for page_id, page in pages.items():
                        if page_id != "-1":
                            url = page.get("fullurl")
                            # 保存到缓存
                            _save_cache_data(cache_path, {'url': url, 'title': title})
                            return url

                return None

    async def get_page_context(
        self,
        title: str,
        max_links: int = 50,
        max_categories: int = 20
    ) -> Dict[str, Any]:
        """获取Wikipedia页面正文、链接、分类和URL，用于实体扩展证据。"""
        safe_title = title.replace(' ', '_').replace('/', '_')
        cache_key = f"context_{safe_title}_{self.language}_{max_links}_{max_categories}"
        cache_path = _get_cache_path(WIKIPEDIA_CACHE_DIR, cache_key)

        cached = _load_cached_data(cache_path)
        if cached:
            print(f"   [缓存命中] Wikipedia上下文: {title}")
            return cached.get('data', {})

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|links|categories|info",
            "explaintext": 1,
            "pllimit": max_links,
            "cllimit": max_categories,
            "inprop": "url",
            "redirects": 1,
            "format": "json"
        }

        proxy = self._get_proxy()
        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.base_url, params=params, headers=self.headers, proxy=proxy) as response:
                if response.status != 200:
                    return {}

                data = await response.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    if page_id == "-1":
                        continue

                    links = [
                        link.get("title", "")
                        for link in page.get("links", [])
                        if link.get("ns") == 0 and link.get("title")
                    ]
                    categories = [
                        cat.get("title", "").replace("Category:", "")
                        for cat in page.get("categories", [])
                        if cat.get("title")
                    ]
                    result = {
                        "title": page.get("title", title),
                        "url": page.get("fullurl", f"https://{self.language}.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                        "extract": page.get("extract", ""),
                        "links": links[:max_links],
                        "categories": categories[:max_categories],
                        "language": self.language
                    }
                    _save_cache_data(cache_path, result)
                    return result

        return {}

    async def search_images(self, query: str, num_results: int = 5) -> List[ImageSearchResult]:
        """获取Wikipedia文章中的图片"""
        # 生成缓存key
        safe_query = query.replace(' ', '_').replace('/', '_')
        cache_key = f"images_{safe_query}_{num_results}_{self.language}"
        cache_path = _get_cache_path(WIKIPEDIA_CACHE_DIR, cache_key)

        # 先检查缓存
        cached = _load_cached_data(cache_path)
        if cached:
            print(f"   [缓存命中] Wikipedia图片: {query}")
            return cached.get('data', [])

        params = {
            "action": "query",
            "titles": query,
            "prop": "images",
            "imlimit": num_results,
            "format": "json"
        }

        proxy = self._get_proxy()
        async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(self.base_url, params=params, headers=self.headers, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get("query", {}).get("pages", {})
                    results = []

                    for page in pages.values():
                        for img in page.get("images", []):
                            img_title = img.get("title", "")
                            if img_title and not any(x in img_title.lower() for x in ['icon', 'logo', 'symbol']):
                                results.append(ImageSearchResult(
                                    url=f"https://{self.language}.wikipedia.org/wiki/{img_title.replace(' ', '_')}",
                                    thumbnail_url="",
                                    title=img_title,
                                    source_url=f"https://{self.language}.wikipedia.org/wiki/{query.replace(' ', '_')}"
                                ))

                    # 保存到缓存
                    _save_cache_data(cache_path, results)
                    return results[:num_results]
                else:
                    return []


class LocalDatasetClient(BaseSearchClient):
    """
    本地数据集客户端
    从本地JSON/JSONL文件加载数据
    适合使用已有的QA数据集作为来源
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data: List[Dict] = []
        self._load_data()

    def _load_data(self):
        """加载本地数据"""
        if not os.path.exists(self.data_path):
            print(f"警告: 数据文件不存在: {self.data_path}")
            return

        if self.data_path.endswith('.jsonl'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

        print(f"已加载 {len(self.data)} 条本地数据")

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """在本地数据中搜索"""
        results = []
        query_lower = query.lower()

        for item in self.data:
            # 简单的关键词匹配
            text = json.dumps(item, ensure_ascii=False).lower()
            if query_lower in text:
                results.append(SearchResult(
                    title=item.get("title", item.get("question", ""))[:100],
                    url=f"local://{hash(json.dumps(item))}",
                    snippet=item.get("content", item.get("answer", ""))[:500],
                    metadata=item
                ))

                if len(results) >= num_results:
                    break

        return results

    async def search_images(self, query: str, num_results: int = 5) -> List[ImageSearchResult]:
        """本地数据集通常不包含图片搜索"""
        return []

    def get_random_samples(self, count: int = 10) -> List[Dict]:
        """获取随机样本（用于数据增强）"""
        import random
        return random.sample(self.data, min(count, len(self.data)))


class WebPageFetcher:
    """
    网页内容抓取器
    用于获取网页的完整内容
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    async def fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """抓取网页内容"""
        try:
            async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._parse_html(html, url)
                    else:
                        return None
        except Exception as e:
            print(f"抓取网页失败: {url}, 错误: {e}")
            return None

    def _parse_html(self, html: str, url: str) -> Dict[str, Any]:
        """解析HTML内容"""
        soup = BeautifulSoup(html, 'lxml')

        # 移除脚本和样式
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # 提取标题
        title = soup.title.string if soup.title else ""

        # 提取正文
        # 优先查找文章主体
        article = soup.find('article') or soup.find('main') or soup.find('body')
        text = article.get_text(separator='\n', strip=True) if article else ""

        # 提取图片
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src and not src.startswith('data:'):
                # 处理相对路径
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    src = f"{parsed.scheme}://{parsed.netloc}{src}"
                images.append({
                    "url": src,
                    "alt": img.get('alt', ''),
                    "title": img.get('title', '')
                })

        return {
            "url": url,
            "title": title,
            "content": text,  # 返回完整内容，不截断
            "images": images[:20],  # 限制图片数量
            "word_count": len(text.split())
        }


class UnifiedSearchClient:
    """
    统一搜索客户端
    整合多个搜索源，提供统一的接口
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self.clients: Dict[str, BaseSearchClient] = {}
        self.webpage_fetcher = WebPageFetcher(config.timeout)

        self._init_clients()

    def _init_clients(self):
        """初始化搜索客户端"""
        api_key = self.config.search_api_key
        engine = self.config.search_engine.lower()
        proxy = getattr(self.config, 'proxy', None) or os.environ.get("https_proxy") or os.environ.get("http_proxy")

        if api_key:
            if engine == "serper":
                self.clients["serper"] = SerperSearchClient(api_key)
            elif engine == "bing":
                self.clients["bing"] = BingSearchClient(api_key)
            elif engine == "serpapi":
                self.clients["serpapi"] = SerpAPIClient(api_key)

        # Wikipedia 始终可用（免费），传入代理配置
        self.clients["wikipedia"] = WikipediaClient("zh", proxy=proxy)
        self.clients["wikipedia_en"] = WikipediaClient("en", proxy=proxy)

    async def close(self):
        """关闭所有客户端的会话"""
        for client in self.clients.values():
            if hasattr(client, 'close'):
                await client.close()

    def add_local_dataset(self, name: str, path: str):
        """添加本地数据集"""
        self.clients[name] = LocalDatasetClient(path)

    async def search(
        self,
        query: str,
        num_results: int = 10,
        sources: List[str] = None
    ) -> List[SearchResult]:
        """
        执行搜索

        Args:
            query: 搜索查询
            num_results: 结果数量
            sources: 指定搜索源，默认使用所有可用源
        """
        sources = sources or list(self.clients.keys())
        all_results = []

        for source in sources:
            if source in self.clients:
                try:
                    results = await self.clients[source].search(query, num_results)
                    for r in results:
                        r.metadata["source"] = source
                    all_results.extend(results)
                except Exception as e:
                    print(f"搜索源 {source} 失败: {e}")

        return all_results[:num_results * 2]  # 返回双倍结果供选择

    async def search_images(
        self,
        query: str,
        num_results: int = 5
    ) -> List[ImageSearchResult]:
        """执行图片搜索"""
        for client in self.clients.values():
            try:
                results = await client.search_images(query, num_results)
                if results:
                    return results
            except:
                continue
        return []

    async def fetch_webpage(self, url: str) -> Optional[Dict[str, Any]]:
        """抓取网页内容"""
        return await self.webpage_fetcher.fetch(url)

    async def deep_search(
        self,
        query: str,
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        深度搜索：搜索 + 抓取网页内容

        Args:
            query: 搜索查询
            depth: 抓取的结果数量
        """
        # 先搜索
        search_results = await self.search(query, depth * 2)

        # 抓取前N个网页
        detailed_results = []
        for result in search_results[:depth]:
            webpage = await self.fetch_webpage(result.url)
            if webpage:
                detailed_results.append({
                    "search_result": result,
                    "webpage_content": webpage
                })

        return detailed_results


@dataclass
class WikidataEntity:
    """Wikidata实体"""
    entity_id: str  # Q-ID
    label: str
    description: str
    claims: Dict[str, List[Dict]]  # 属性
    neighbors: List[Dict[str, str]]  # 相关实体: {entity_id, label, relation_type}
    labels: Dict[str, str] = field(default_factory=dict)
    descriptions: Dict[str, str] = field(default_factory=dict)
    sitelinks: Dict[str, str] = field(default_factory=dict)


class WikidataClient:
    """
    Wikidata API 客户端

    用于从Wikidata获取实体的属性和邻居关系

    支持多个镜像地址（中国友好型）
    """
    # Wikidata API 镜像（中国友好）
    WIKIMEDIA_API = "https://wikimedia.org/api/rest_v1/mix/search/entity"
    WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

    # 按优先级排序的可用URL
    AVAILABLE_URLS = [
        ("https://www.wikidata.org/w/api.php", "wikidata"),
        ("https://wikimedia.org/api/rest_v1", "wikimedia_cdn"),
    ]

    BASE_URL = "https://www.wikidata.org/w/api.php"

    # 重要的关系类型（P-ID映射）
    # 按类别组织：
    # - 分类关系：instance_of, subclass_of, part_of
    # - 人物关系：spouse, child, parent, employer, educated_at
    # - 组织关系：member_of, subsidiary, owner_of, headquarters
    # - 地理位置：location, located_in_country, country_of_origin
    # - 工作/创作：occupation, field_of_work, notable_work
    # - 时间相关：inception, dissolved, start_time, end_time
    # - 作品/产品：manufacturer, developer, producer
    # - 体育/娱乐：member_of_sports_team, position, genre
    # - 其他：award_received, language, image

    RELATION_PROPERTIES = {
        # === 分类关系 ===
        "instance_of": "P31",           # 是...的实例
        "subclass_of": "P279",          # 是...的子类
        "part_of": "P361",              # 是...的一部分

        # === 人物关系 ===
        "spouse": "P26",                # 配偶
        "child": "P40",                 # 子女
        "parent": "P25",                # 父母
        "employer": "P108",             # 雇主
        "educated_at": "P69",          # 毕业院校
        "doctoral_student": "P185",    # 博士生
        "doctoral_advisor": "P184",    # 博导

        # === 组织关系 ===
        "member_of": "P463",            # 成员
        "subsidiary": "P355",           # 子公司
        "owner_of": "P1830",            # 拥有者
        "headquarters": "P159",         # 总部
        "founded_by": "P112",          # 创始人
        "industry": "P452",             # 行业

        # === 地理位置 ===
        "location": "P276",             # 位于
        "located_in_country": "P17",   # 所在国家
        "country_of_origin": "P495",   # 原产国
        "continent": "P30",             # 洲
        "capital": "P36",              # 首都

        # === 工作/职业 ===
        "occupation": "P106",          # 职业
        "field_of_work": "P101",       # 工作领域
        "position_held": "P39",        # 担任职位

        # === 时间相关 ===
        "inception": "P571",           # 创建时间
        "dissolved": "P576",           # 解散时间
        "start_time": "P580",          # 开始时间
        "end_time": "P582",             # 结束时间
        "publication_date": "P577",    # 发布/出版日期

        # === 作品/创作 ===
        "notable_work": "P800",        # 著名作品
        "author": "P50",               # 作者
        "creator": "P170",             # 创造者
        "manufacturer": "P176",       # 制造商
        "developer": "P178",          # 开发者
        "producer": "P162",           # 制片人
        "record_label": "P264",       # 唱片公司
        "genre": "P136",               # 类型/流派

        # === 体育/娱乐 ===
        "member_of_sports_team": "P54", # 运动队成员
        "position_played": "P413",     # 球队位置/演奏乐器

        # === 成就/荣誉 ===
        "award_received": "P166",      # 获得奖项
        "nominated_for": "P1411",     # 提名

        # === 其他 ===
        "language_of_work": "P407",    # 工作语言
        "original_language": "P364",    # 原始语言
        "country_for_sport": "P1532",   # 体育代表国家
        "use": "P366",                   # 用途
        "made_from_material": "P186",  # 制造材料
        "religion": "P140",             # 宗教
        "native_language": "P103",      # 母语
        "ethnic_group": "P172",         # 民族
        "citizenship": "P27",           # 国籍
        "population": "P1082",         # 人口
        "area": "P2046",                # 面积
        "elevation": "P2044",          # 海拔
        "ticker_symbol": "P249",       # 股票代码
        "iata_code": "P238",            # IATA代码
        "icao_code": "P239",           # ICAO代码
        "callsign": "P2850",           # 呼号
        "named_after": "P138",         # 以...命名
        "described_by_source": "P1343",# 来源描述
        "image": "P18",                # 图片
        "logo_image": "P154",          # Logo
        "locator_map_image": "P242",   # 定位地图
        "coat_of_arms_image": "P94",   # 纹章
        "sex_or_gender": "P21",        # 性别
        "official_website": "P856",    # 官方网站
    }

    def __init__(self, proxy: str = None):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.session = None
        # 如果没有传入proxy，尝试从环境变量获取
        self.proxy = proxy or os.environ.get("https_proxy") or os.environ.get("http_proxy")

    async def close(self):
        """关闭Session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    def _get_proxy(self) -> Optional[str]:
        """获取代理配置"""
        if self.proxy:
            return self.proxy
        # 尝试从环境变量获取
        return os.environ.get("https_proxy") or os.environ.get("http_proxy")

    @staticmethod
    def _pick_language_value(values: Dict[str, Dict[str, str]], preferred_languages: List[str]) -> str:
        for lang in preferred_languages:
            value = values.get(lang, {}).get("value", "")
            if value:
                return value
        for value in values.values():
            if isinstance(value, dict) and value.get("value"):
                return value.get("value", "")
        return ""

    async def _get_session(self):
        """获取异步Session（带代理配置）"""
        if self.session is None or self.session.closed:
            proxy = self._get_proxy()
            timeout = aiohttp.ClientTimeout(total=60)
            if proxy:
                # 使用代理
                self.session = aiohttp.ClientSession(
                    headers=self.headers,
                    timeout=timeout,
                    trust_env=True  # 允许从环境变量读取代理配置
                )
            else:
                self.session = aiohttp.ClientSession(
                    headers=self.headers,
                    timeout=timeout,
                    trust_env=True
                )
        return self.session

    async def search_entity(self, query: str, limit: int = 5, language: str = "en") -> List[Dict]:
        """
        搜索Wikidata实体

        Args:
            query: 搜索关键词
            limit: 返回数量
            language: 搜索语言

        Returns:
            实体列表
        """
        # 生成缓存key
        cache_key = f"search_{query}_{limit}_{language}"
        cache_path = _get_cache_path(WIKIDATA_CACHE_DIR, cache_key)

        # 先检查缓存
        cached = _load_cached_data(cache_path)
        if cached:
            print(f"   [缓存命中] Wikidata搜索: {query}")
            return cached.get('data', [])

        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": language,
            "format": "json",
            "limit": limit
        }

        proxy = self._get_proxy()
        session = await self._get_session()
        async with session.get(self.BASE_URL, params=params, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get("search", [])
                    for item in results:
                        item["search_language"] = language
                    # 保存到缓存
                    _save_cache_data(cache_path, results)
                    return results
                return []

    async def search_entity_multilingual(
        self,
        query: str,
        limit: int = 5,
        languages: List[str] = None
    ) -> List[Dict]:
        """按语言顺序搜索Wikidata实体，并按QID去重合并结果。"""
        languages = languages or ["en", "zh"]
        merged = []
        seen = set()
        for lang in languages:
            try:
                results = await self.search_entity(query, limit=limit, language=lang)
            except Exception as e:
                print(f"   ⚠️ Wikidata搜索失败: query='{query}', language='{lang}', error={e}")
                results = []
            for item in results:
                entity_id = item.get("id")
                if not entity_id or entity_id in seen:
                    continue
                seen.add(entity_id)
                merged.append(item)
        return merged[:limit * len(languages)]

    async def get_entity(self, entity_id: str) -> Optional[WikidataEntity]:
        """
        获取Wikidata实体详情

        Args:
            entity_id: 实体ID (如 Q123)

        Returns:
            WikidataEntity对象
        """
        # 生成缓存key
        cache_key = f"entity_v2_{entity_id}"
        cache_path = _get_cache_path(WIKIDATA_CACHE_DIR, cache_key)

        # 先检查缓存
        cached = _load_cached_data(cache_path)
        if cached:
            print(f"   [缓存命中] Wikidata实体: {entity_id}")
            cached_data = cached.get('data', {})
            return WikidataEntity(
                entity_id=cached_data.get('entity_id', entity_id),
                label=cached_data.get('label', ''),
                description=cached_data.get('description', ''),
                claims=cached_data.get('claims', {}),
                neighbors=[],
                labels=cached_data.get('labels', {}),
                descriptions=cached_data.get('descriptions', {}),
                sitelinks=cached_data.get('sitelinks', {})
            )

        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "props": "labels|descriptions|claims|sitelinks",
            "format": "json",
            "languages": "en|zh",
            "sitefilter": "enwiki|zhwiki"
        }

        proxy = self._get_proxy()
        session = await self._get_session()
        async with session.get(self.BASE_URL, params=params, proxy=proxy) as response:
                if response.status == 200:
                    data = await response.json()
                    entities = data.get("entities", {})
                    if entity_id in entities:
                        entity_data = entities[entity_id]
                        labels = {
                            lang: value.get("value", "")
                            for lang, value in entity_data.get("labels", {}).items()
                            if value.get("value")
                        }
                        descriptions = {
                            lang: value.get("value", "")
                            for lang, value in entity_data.get("descriptions", {}).items()
                            if value.get("value")
                        }
                        sitelinks = {
                            site: value.get("title", "")
                            for site, value in entity_data.get("sitelinks", {}).items()
                            if value.get("title")
                        }
                        entity = WikidataEntity(
                            entity_id=entity_id,
                            label=self._pick_language_value(entity_data.get("labels", {}), ["en", "zh"]),
                            description=self._pick_language_value(entity_data.get("descriptions", {}), ["en", "zh"]),
                            claims=entity_data.get("claims", {}),
                            neighbors=[],
                            labels=labels,
                            descriptions=descriptions,
                            sitelinks=sitelinks
                        )
                        # 保存到缓存
                        _save_cache_data(cache_path, {
                            'entity_id': entity.entity_id,
                            'label': entity.label,
                            'description': entity.description,
                            'claims': entity.claims,
                            'labels': entity.labels,
                            'descriptions': entity.descriptions,
                            'sitelinks': entity.sitelinks
                        })
                        return entity
                return None

    async def get_neighbors(
        self,
        entity_id: str,
        relation_types: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        获取实体的邻居实体

        Args:
            entity_id: 实体ID
            relation_types: 要获取的关系类型列表（如["instance_of", "part_of", "occupations"]）

        Returns:
            邻居实体列表 [{entity_id, label, relation_type, property_id, description, frequency}]
        """
        relation_types = relation_types or list(self.RELATION_PROPERTIES.keys())
        neighbors = []

        # 获取实体详情
        entity = await self.get_entity(entity_id)
        if not entity:
            return []

        # 第一遍：收集所有邻居关系
        temp_neighbors = []
        for rel_type in relation_types:
            prop_id = self.RELATION_PROPERTIES.get(rel_type)
            if not prop_id:
                continue

            claims = entity.claims.get(prop_id, [])
            for claim in claims:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})

                if datavalue.get("type") == "wikibase-entityid":
                    target_id = datavalue.get("value", {}).get("id")
                    if target_id:
                        # 获取邻居标签
                        neighbor_entity = await self.get_entity(target_id)
                        if neighbor_entity:
                            temp_neighbors.append({
                                "entity_id": target_id,
                                "label": neighbor_entity.label,
                                "relation_type": rel_type,
                                "property_id": prop_id,
                                "description": neighbor_entity.description
                            })

        # 第二遍：统计每个实体出现的频次
        entity_counts: Dict[str, int] = {}
        for n in temp_neighbors:
            entity_id_key = n["entity_id"]
            entity_counts[entity_id_key] = entity_counts.get(entity_id_key, 0) + 1

        # 第三遍：合并结果，添加频次
        seen = set()
        for n in temp_neighbors:
            entity_id_key = n["entity_id"]
            if entity_id_key not in seen:
                seen.add(entity_id_key)
                n["frequency"] = entity_counts[entity_id_key]  # 添加频次
                neighbors.append(n)

        return neighbors

    async def get_all_neighbors(self, entity_id: str) -> List[Dict[str, str]]:
        """
        获取实体的所有邻居

        Args:
            entity_id: 实体ID

        Returns:
            所有邻居实体
        """
        all_relation_types = list(self.RELATION_PROPERTIES.keys())
        return await self.get_neighbors(entity_id, all_relation_types)

    async def get_linked_entities(self, entity_id: str) -> List[str]:
        """
        获取与当前实体有链接的所有实体ID

        Args:
            entity_id: 实体ID

        Returns:
            链接实体ID列表
        """
        entity = await self.get_entity(entity_id)
        if not entity:
            return []

        linked_ids = set()
        for prop_id, claims in entity.claims.items():
            for claim in claims:
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})

                if datavalue.get("type") == "wikibase-entityid":
                    target_id = datavalue.get("value", {}).get("id")
                    if target_id:
                        linked_ids.add(target_id)

        return list(linked_ids)
