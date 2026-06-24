"""
VistaHop管道配置模块

VistaHop: A Bottom-up Method for Generating Complex Multi-hop Reasoning Questions

配置类:
- SynthesisConfig: VistaHop管道主配置
- SearchConfig: 搜索配置
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict
import os


@dataclass
class SearchConfig:
    """
    搜索配置

    支持的搜索引擎:
    - serper: Serper.dev API (推荐，2500次/月免费)
    - bing: Bing Search API (1000次/月免费)
    - serpapi: SerpAPI (100次/月免费)
    - wikipedia: Wikipedia API (完全免费，无限制)
    """
    # 搜索API配置
    search_api_key: str = ""
    search_engine: str = "serper"  # serper, bing, serpapi, wikipedia
    max_results_per_query: int = 10
    timeout: int = 30

    # 本地数据集配置（可选）
    local_dataset_path: str = ""  # 本地QA数据集路径 (JSON/JSONL)

    # 网页抓取配置
    enable_webpage_fetch: bool = True  # 是否抓取网页全文
    max_content_length: int = 5000  # 抓取内容最大长度

    # 代理配置
    proxy: str = ""  # 搜索服务使用的代理地址，如 "http://127.0.0.1:7890"


@dataclass
class SynthesisConfig:
    """VistaHop管道配置"""

    # LLM配置
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4.1"))

    # 搜索配置
    search_engine: str = "wikipedia"
    search_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", "") or os.getenv("SERPAPI_KEY", ""))
    proxy: str = ""  # 搜索服务使用的代理地址，如 "http://127.0.0.1:7890"

    def __post_init__(self):
        # 如果没有设置代理，尝试从环境变量获取
        if not self.proxy:
            self.proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy") or ""

    # API镜像配置（中国友好）
    use_api_mirror: bool = True  # 是否使用API镜像
    wikipedia_mirror: str = "wikimedia"  # wikimedia, zh, en
    wikidata_mirror: str = "wikimedia"  # wikimedia, official

    # 证据链配置
    max_chain_depth: int = 2  # 暂时遍历2层
    min_diversity_score: float = 0.5
    nli_confidence_threshold: float = 0.9

    # 问题生成配置
    questions_per_chain: int = 3
    chains_per_entity: int = 1

    # 图片处理配置
    image_url: str = ""
    image_model: str = ""
    first_entity_only: bool = False
    num_entities_for_chains: int = 0  # 限制用于证据链构建的实体数量（0表示不限制）

    # 输出配置
    output_dir: str = "./outputs/synthesis"
    save_intermediate: bool = True

    # VQA生成配置
    generate_vqa: bool = False  # 是否生成VQA
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_KEY", ""))
    vqa_images_per_entity: int = 2  # 每个实体搜索的图片数量
    enable_question_simplify: bool = False  # 是否启用问题简化/重写

    # Stage 6 多链融合配置
    enable_stage6_fusion: bool = False  # 是否启用Stage 6多链融合
    fusion_num_chains: int = 3  # 融合链数量
    fusion_rule: str = "llm_decided"  # llm_decided, add, subtract, multiply, divide, max, min, avg, conditional
    enable_llm_fusion: bool = True  # 是否让LLM决定融合规则（支持非数值结果）

    # 管道控制
    max_stage: int = 6  # 最大运行阶段 (1-6)

    # 条件消融分析配置
    enable_ablation_analysis: bool = True  # 是否启用条件消融分析
    ablation_max_conditions: int = 5  # 最多分析的条件数量
    ablation_batch_size: int = 4  # 消融分析批处理大小
