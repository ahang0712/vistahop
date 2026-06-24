"""
第2阶段：节点信息构建 (Node Information Construction)

为每个实体构建知识图谱节点，收集节点描述和属性
"""

import json
import re
import sys
import os
from typing import List, Any, Dict

# 处理导入路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from .search_client import WikipediaClient
except ImportError:
    from search_client import WikipediaClient


class Stage2NodeBuilder:
    """
    第2阶段：节点信息构建器

    为实体构建详细的知识图谱节点信息
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

    async def build_node_information(
        self,
        entities: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        构建节点信息

        Args:
            entities: 实体列表

        Returns:
            节点信息列表
        """
        node_info = []

        # 如果只处理第一个实体
        entities_to_process = [entities[0]] if self.config.first_entity_only else entities

        total = len(entities_to_process)

        for idx, entity in enumerate(entities_to_process, 1):
            # 支持两种输入类型：字符串 或 字典
            if isinstance(entity, dict):
                entity_name = entity.get("cleaned_name") or entity.get("name", str(entity))
            else:
                entity_name = str(entity)

            # 打印进度条
            progress = idx / total
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"   {bar} {idx}/{total} ({progress*100:.1f}%) 正在处理: {entity_name[:30]}", end='\r')

            try:
                # 获取Wikipedia信息
                wiki = WikipediaClient("en")
                content = await wiki.get_article_content(entity_name)

                # Build node information
                info = {
                    "entity": entity_name,
                    "description": "",
                    "entity_type": "",
                    "properties": {},
                    "related_entities": []
                }

                if content:
                    # Use LLM to extract node details
                    info = await self._extract_node_details(entity_name, content, info)

                node_info.append(info)

            except Exception as e:
                # 支持两种输入类型
                entity_name = entity.get("cleaned_name") or entity.get("name", str(entity)) if isinstance(entity, dict) else str(entity)
                print(f"   ⚠️ 构建节点 '{entity_name}' 信息失败: {e}")
                node_info.append({
                    "entity": entity_name,
                    "error": str(e)
                })

        # 清除进度条并打印完成信息
        print(" " * 80, end='\r')
        print(f"   ✅ 节点信息构建完成 ({total}/{total})")

        return node_info

    async def _extract_node_details(
        self,
        entity_name: str,
        content: str,
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用LLM提取节点详情

        Args:
            entity_name: 实体名称
            content: Wikipedia内容
            info: 现有节点信息

        Returns:
            更新后的节点信息
        """
        prompt = f"""Extract key information about "{entity_name}" from the following text.

Text content:
{content[:1500]}

Please extract:
1. One-sentence description of what the entity is
2. Main type of entity (person/place/event/object/concept)
3. Key attributes of the entity (achievements, characteristics, etc.)
4. Most related entities to this entity

**Return JSON**:
{{
    "description": "One-sentence description",
    "entity_type": "person/place/event/object/concept",
    "properties": {{
        "key1": "value1",
        "key2": "value2"
    }},
    "related_entities": ["entity1", "entity2", "entity3"]
}}

Return JSON only, no other content."""

        response = await self.llm_client.generate(prompt)

        clean_response = response.strip()
        if clean_response.startswith("```"):
            clean_response = re.sub(r'^```\w*\n?', '', clean_response)
            clean_response = re.sub(r'\n?```$', '', clean_response)

        try:
            data = json.loads(clean_response)
            info["description"] = data.get("description", "")
            info["entity_type"] = data.get("entity_type", "")
            info["properties"] = data.get("properties", {})
            info["related_entities"] = data.get("related_entities", [])
        except json.JSONDecodeError as e:
            print(f"   ⚠️ 解析节点详情失败: {e}")

        return info


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
        parser = argparse.ArgumentParser(description="Stage2: 节点信息构建")
        parser.add_argument("--input", type=str, required=True, help="Stage1输出的JSON文件")
        parser.add_argument("--output", type=str, default="./stage2_output.json", help="输出文件")
        parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY", ""), help="API Key")
        parser.add_argument("--base-url", type=str, default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), help="API Base URL")
        parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="模型名称")
        args = parser.parse_args()

        # 加载输入
        with open(args.input, 'r', encoding='utf-8') as f:
            entities = json.load(f)

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
            llm_model=args.model
        )
        llm_client = get_friday_client(config)

        # 运行 Stage2
        builder = Stage2NodeBuilder(llm_client, config)
        node_info = await builder.build_node_information(entities)

        print(f"\n✅ Stage2 完成，构建了 {len(node_info)} 个节点信息")

        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(node_info, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {args.output}")

    asyncio.run(main())
