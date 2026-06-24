"""
第1阶段：数据源与适应 (Data Source and Adaptation)

从图片提取实体、实体过滤和清洗
"""

import json
import sys
import os
from typing import List, Dict, Any, Optional

# 处理导入路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from ._bbox_extractor import extract_bounding_boxes
except ImportError:
    from _bbox_extractor import extract_bounding_boxes


class Stage1Extractor:
    """
    第1阶段：数据源与适应处理器

    负责从图片提取实体及其属性
    """

    def __init__(self, llm_client, config):
        """
        初始化

        Args:
            llm_client: LLM客户端
            config: VistaHop配置
        """
        self.llm_client = llm_client
        self.config = config

    async def extract_entities(
        self,
        entities: Optional[List[str]] = None,
        image_url: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        提取实体

        Args:
            entities: 提供的实体列表
            image_url: 图片URL

        Returns:
            提取的实体列表（带属性）
        """
        result = []
        if entities:
            # 使用提供的实体列表
            result = self._process_provided_entities(entities)
        elif image_url:
            # 从图片提取实体
            result = await self._extract_from_image(image_url)
        else:
            # 使用默认实体
            result = self._get_default_entities()

        # 清洗所有实体的名称
        for entity in result:
            if "name" in entity and entity["name"]:
                entity["cleaned_name"] = self._clean_entity_name(entity["name"])

        return result

    async def _extract_from_image(self, image_url: str) -> List[Dict[str, Any]]:
        """
        从图片提取实体

        Args:
            image_url: 图片URL

        Returns:
            提取的实体列表
        """
        print(f"   🔍 Extracting entities from image...")
        print(f"   📷 Image URL: {image_url[:60]}...")

        try:
            # 使用 bbox extractor 提取实体
            entities = await extract_bounding_boxes(
                llm_client=self.llm_client,
                image_url=image_url,
                output_dir=self.config.output_dir
            )

            if not entities:
                print(f"   ⚠️ No entities extracted")
                return []

            return entities

        except Exception as e:
            print(f"   ❌ Failed to extract entities: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _process_provided_entities(self, entities: List[str]) -> List[Dict[str, Any]]:
        """
        处理提供的实体列表

        Args:
            entities: 实体名称列表

        Returns:
            包含属性的实体列表
        """
        result = []
        for entity in entities:
            cleaned_name = self._clean_entity_name(entity)
            result.append({
                "name": entity,
                "cleaned_name": cleaned_name,
                "source": "provided"
            })
        return result

    def _clean_entity_name(self, name: str) -> str:
        """
        清洗实体名称，删除无关词汇

        Args:
            name: 原始实体名称

        Returns:
            清洗后的实体名称
        """
        import re
        # 删除常见的无关词汇
        patterns_to_remove = [
            r'\bLogo\b',
            r'\blogo\b',
            r'\bIcon\b',
            r'\bicon\b',
            r'\bSymbol\b',
            r'\bsymbol\b',
            r'\bFlag of\b',
            r'\bflag of\b',
            r'\bImage\b',
            r'\bimage\b',
            r'\bPhoto\b',
            r'\bphoto\b',
            r'\bPicture\b',
            r'\bpicture\b',
            r'\bGraphic\b',
            r'\bgraphic\b',
            r'\bSign\b',
            r'\bsign\b',
            r'\bMark\b',
            r'\bmark\b',
            r'\bEmblem\b',
            r'\bemblem\b',
        ]

        cleaned = name
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned)

        # 清理多余空格和连字符
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip(' -_,;:')

        # 如果清洗后为空，返回原始名称
        return cleaned if cleaned else name

    def _get_default_entities(self) -> List[Dict[str, Any]]:
        """
        获取默认实体列表（用于演示）

        Returns:
            默认实体列表
        """
        print(f"   ⚠️ No entities or image provided, using default entities")
        return [
            {"name": "Einstein", "cleaned_name": "Einstein", "source": "default"},
            {"name": "Curie", "cleaned_name": "Curie", "source": "default"}
        ]


# ============================================================================
# 独立运行入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    import asyncio
    import os
    import json
    import sys

    async def main():
        parser = argparse.ArgumentParser(description="Stage1: 实体提取")
        parser.add_argument("--entities", type=str, nargs="+", help="实体列表")
        parser.add_argument("--image-url", type=str, help="图片URL")
        parser.add_argument("--output", type=str, default="./stage1_output.json", help="输出文件")
        parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY", ""), help="API Key")
        parser.add_argument("--base-url", type=str, default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), help="API Base URL")
        parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="模型名称")
        args = parser.parse_args()

        # 处理导入路径
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        if _current_dir not in sys.path:
            sys.path.insert(0, _current_dir)

        # 初始化配置和客户端
        try:
            from .config import SynthesisConfig
            from llm_client import get_friday_client
        except ImportError:
            from .config import SynthesisConfig
            from .llm_client import get_friday_client

        config = SynthesisConfig(
            llm_api_key=args.api_key,
            llm_base_url=args.base_url,
            llm_model=args.model,
            image_url=args.image_url or ""
        )
        llm_client = get_friday_client(config)

        # 运行 Stage1
        extractor = Stage1Extractor(llm_client, config)
        entities = await extractor.extract_entities(
            entities=args.entities,
            image_url=args.image_url
        )

        print(f"\n✅ Stage1 完成，提取到 {len(entities)} 个实体")

        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {args.output}")

    asyncio.run(main())
