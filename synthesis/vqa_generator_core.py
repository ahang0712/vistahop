#!/usr/bin/env python3
"""
VistaHop VQA Generator
将VistaHop生成的问题转换为视觉问答(VQA)格式

功能：
1. 从原图中提取可可视化的实体
2. 使用SerpAPI搜索相关图片
3. 将文本问题转换为VQA格式（添加视觉引用）
4. 生成多模态问答数据

使用方法：
  from synthesis.vqa_generator_core import VistaHopVQAGenerator, VistaHopVQAConfig

  config = VistaHopVQAConfig(
      image_url="https://...",
      serpapi_key="your_api_key",
      images_per_entity=2
  )
  generator = VistaHopVQAGenerator(config, llm_client)
  vqa_result = await generator.generate_vqa_from_questions(synthesis_questions, original_image_url)
"""

import os
import json
import asyncio
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote_plus

# 处理导入路径，支持模块和直接运行
try:
    from .llm_client import LLMClient
    from .search_client import SerpAPIClient, ImageSearchResult
except ImportError:
    try:
        from llm_client import LLMClient
        from search_client import SerpAPIClient, ImageSearchResult
    except ImportError:
        # 如果导入失败，使用 None 作为占位符
        LLMClient = None
        SerpAPIClient = None
        ImageSearchResult = None


# ============= 配置类 =============

@dataclass
class VistaHopVQAConfig:
    """VistaHop VQA Generation Configuration"""
    # 图片相关
    image_url: str = ""
    image_model: str = ""

    # SerpAPI配置
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_KEY", ""))

    # 图片搜索配置
    images_per_entity: int = 2
    min_image_width: int = 400
    min_image_height: int = 400

    # 问题转换配置 - 视觉引用模板（固定使用 shown in the image）
    visual_reference_templates: List[str] = field(default_factory=lambda: [
        "the {entity} shown in the image",
    ])

    # 输出配置
    output_dir: str = "./output/vqa"
    save_images: bool = False

    # 实体过滤配置
    filter_temporal_entities: bool = True
    filter_abstract_entities: bool = True
    min_entity_length: int = 2
    max_entity_length: int = 50

    # 问题简化配置
    enable_question_simplify: bool = True  # 是否启用问题简化/重写

    # Stage1 实体数据（包含视觉描述）
    # 格式: [{"name": "实体名", "description": "视觉描述"}, ...]
    stage1_entities: List[Dict[str, str]] = field(default_factory=list)


# ============= 数据类 =============

@dataclass
class VisualizableEntity:
    """Visualizable Entity extracted from image"""
    name: str
    entity_type: str  # PERSON, PLACE, OBJECT, ORGANIZATION, EVENT
    description: str
    confidence: float
    image_url: str = ""
    image_results: List[ImageSearchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "confidence": self.confidence,
            "image_url": self.image_url,
            "image_results": [r.__dict__ for r in self.image_results]
        }


@dataclass
class VQAItem:
    """VQA Question-Answer Item"""
    id: str
    question: str                      # VQA问题（含视觉引用，简化后）
    answer: str
    original_question: str              # 原始文本问题
    original_answer: str
    target_entity: str                 # 被替换的实体
    images: List[Dict[str, Any]]       # 关联图片列表
    visual_reference: str               # 视觉引用短语
    entity_type: str
    difficulty: str                    # easy/medium/hard
    domain: str
    reasoning_path: List[str]          # 推理路径
    image_url: str = ""                # 原图URL (base64 or http)
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_vqa_question: str = ""       # 原始VQA问题（简化前，仅完成视觉引用替换）
    # 验证相关字段
    leakage_detected: bool = False      # 是否检测到信息泄露
    leakage_reason: str = ""           # 泄露原因
    solver_answer: str = ""            # 做题Agent的答案（仅用文本）
    solver_confidence: float = 0.0     # 做题Agent的置信度
    local_image_path: str = ""         # 本地图片路径（用于保存到文件）

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "original_question": self.original_question,
            "original_answer": self.original_answer,
            "target_entity": self.target_entity,
            "images": self.images,
            "visual_reference": self.visual_reference,
            "entity_type": self.entity_type,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "reasoning_path": self.reasoning_path,
            "metadata": self.metadata,
            "original_vqa_question": self.original_vqa_question,
            "leakage_detected": self.leakage_detected,
            "leakage_reason": self.leakage_reason,
            "solver_answer": self.solver_answer,
            "solver_confidence": self.solver_confidence,
            "local_image_path": self.local_image_path
        }
        # 如果有本地路径，使用本地路径；否则使用原值（http URL 或空）
        if self.local_image_path:
            result["image_url"] = self.local_image_path
        elif self.image_url and not self.image_url.startswith("data:"):
            result["image_url"] = self.image_url
        return result


@dataclass
class VQAGenerationResult:
    """VQA Generation Result"""
    total_input: int = 0
    total_vqa: int = 0
    vqa_items: List[VQAItem] = field(default_factory=list)
    entities_found: List[VisualizableEntity] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now().isoformat(),
            "total_input": self.total_input,
            "total_vqa": self.total_vqa,
            "vqa_items": [item.to_dict() for item in self.vqa_items],
            "entities_found": [e.to_dict() for e in self.entities_found],
            "errors": self.errors
        }


# ============= 主类 =============

class VistaHopVQAGenerator:
    """
    VistaHop VQA生成器

    将VistaHop生成的文本QA转换为视觉问答格式：
    1. 从原图中提取可视化实体
    2. 使用SerpAPI搜索相关图片
    3. 将文本问题转换为VQA格式（添加视觉引用）
    """

    def __init__(self, config: VistaHopVQAConfig, llm_client: Optional[LLMClient] = None):
        """
        初始化VQA生成器

        Args:
            config: VQA配置
            llm_client: LLM客户端（用于实体提取）
        """
        self.config = config
        self.llm_client = llm_client

        # 初始化SerpAPI客户端
        self.serpapi_client = None
        if config.serpapi_key:
            self.serpapi_client = SerpAPIClient(config.serpapi_key)

        # 实体类型关键词映射
        self.entity_type_keywords = {
            "PERSON": ["person", "people", "human", " figure", "celebrity", "actor", "scientist"],
            "PLACE": ["place", "location", "city", "country", "building", "landmark", "tower", "museum"],
            "OBJECT": ["object", "item", "thing", "product", "device", "car", "phone"],
            "ORGANIZATION": ["organization", "company", "institution", "agency", "university", "museum"],
            "EVENT": ["event", "happening", "ceremony", "war", "festival", "conference"],
        }

        # 领域关键词
        self.domain_keywords = {
            "science": ["theory", "scientific", "physics", "chemistry", "biology", "einstein", "newton"],
            "history": ["war", "century", "dynasty", "historical", "ancient", "world war"],
            "geography": ["mountain", "river", "ocean", "continent", "island", "country", "city"],
            "arts": ["painting", "sculpture", "music", "film", "literature", "art", "museum"],
            "business": ["company", "stock", "market", "economic", "corporation", "business"],
        }

    async def generate_vqa_from_questions(
        self,
        synthesis_questions: List[Dict[str, Any]],
        original_image_url: str = ""
    ) -> VQAGenerationResult:
        """
        从VistaHop问题生成VQA

        流程（单次LLM调用完成）：
        1. 从问题中提取根节点实体（仅处理根节点）
        2. 将问题转换为VQA格式（只替换根节点实体）
        3. 使用原始输入图片（不搜索网络图片）

        Args:
            synthesis_questions: VistaHop生成的问题列表
            original_image_url: 原始输入图片URL

        Returns:
            VQAGenerationResult: VQA生成结果
        """
        print("\n" + "=" * 60)
        print("VQA Generation: Converting Text QA to VQA (Root Entity Only)")
        print("=" * 60)

        result = VQAGenerationResult()
        result.total_input = len(synthesis_questions)

        # 保存原始本地路径（用于结果文件）
        local_image_path = original_image_url if not original_image_url.startswith("data:") and not original_image_url.startswith("http") else ""

        # 如果是本地路径，转换为 base64 用于 LLM 调用
        image_for_llm = original_image_url
        if original_image_url and not original_image_url.startswith("data:") and not original_image_url.startswith("http"):
            print(f"   [INFO] Converting local path to base64 for LLM: {original_image_url}")
            image_for_llm = self._convert_local_path_to_base64(original_image_url)
            if not image_for_llm:
                print(f"   [WARNING] Failed to convert local path to base64, using local path as-is")
                image_for_llm = original_image_url

        # Step 1: 提取根节点实体并转换为VQA（单次LLM调用）
        print("\nStep 1: Extract root entities and transform questions to VQA...")
        entities, transformed_questions = await self._extract_root_entities_from_questions(synthesis_questions, image_for_llm)
        result.entities_found = list(entities.values())
        print(f"   Extracted {len(entities)} root entities, transformed {len(transformed_questions)} questions")

        if not entities:
            result.errors.append("No root entities extracted from questions")
            return result

        # Step 2: 构建VQA结果（使用原始输入图片，不搜索网络图片）
        print("\nStep 2: Build VQA results with original image...")
        print(f"   [DEBUG] entities count: {len(entities)}")
        for ename, einfo in entities.items():
            print(f"   [DEBUG] entity: {ename!r} -> {einfo}")
        print(f"   [DEBUG] transformed_questions count: {len(transformed_questions)}")
        for i, tq in enumerate(transformed_questions):
            print(f"   [DEBUG] transformed_questions[{i}]: target_entity={tq.get('target_entity')!r}, "
                  f"vqa_question={tq.get('vqa_question', '')[:80]!r}")

        # Step 3: 简化问题（减少冗余信息）
        print("\nStep 3: Simplify questions...")

        # 创建原始问题到索引的映射
        original_map = {q.get('question', ''): q for q in synthesis_questions}

        # 批量简化问题（如果启用）
        if self.config.enable_question_simplify:
            simplified_questions = await self._simplify_questions_batch(
                [tq.get("vqa_question", "") or tq.get("original_question", "") for tq in transformed_questions]
            )
        else:
            print("   [INFO] Question simplification disabled, using original questions")
            simplified_questions = []

        # 准备用于验证的问题列表
        validation_questions = []
        for idx, tq in enumerate(transformed_questions):
            target_entity = tq.get("target_entity", "")
            entity_info = entities.get(target_entity)
            if entity_info:
                # 应用简化后处理
                vqa_question = tq.get("vqa_question", "") or tq.get("original_question", "")
                if idx < len(simplified_questions) and simplified_questions[idx]:
                    simplified = simplified_questions[idx].strip()
                    if simplified and len(simplified) > 10:
                        vqa_question = simplified

                validation_questions.append({
                    "index": idx,
                    "question": vqa_question,
                    "answer": tq.get("original_answer", ""),
                    "target_entity": target_entity,
                    "original_question": tq.get("original_question", "")
                })

        # Step 4: 验证问题是否存在信息泄露（已移除，在此阶段只负责VQA生成）

        for idx, tq in enumerate(transformed_questions):
            try:
                target_entity = tq.get("target_entity", "")
                entity_info = entities.get(target_entity)

                if entity_info:
                    # 使用LLM已经转换好的问题（这是原始VQA问题，简化之前）
                    vqa_question_original = tq.get("vqa_question", "")
                    if not vqa_question_original:
                        vqa_question_original = tq.get("original_question", "")

                    # 保存原始VQA问题（简化之前）
                    vqa_question = vqa_question_original

                    # 应用简化后处理
                    if idx < len(simplified_questions) and simplified_questions[idx]:
                        simplified = simplified_questions[idx].strip()
                        if simplified and len(simplified) > 10:  # 确保简化后的内容有效
                            vqa_question = simplified

                    # 生成唯一ID
                    qa_id = f"vqa_{hash(tq.get('original_question', '') + target_entity) % 10000000:07d}"

                    # 使用原始输入图片（只有一张）
                    images = []
                    if original_image_url:
                        images.append({
                            "url": local_image_path if local_image_path else original_image_url,
                            "title": "Original input image",
                            "thumbnail": local_image_path if local_image_path else original_image_url,
                            "width": 0,
                            "height": 0,
                            "source": ""
                        })

                    # 确定难度和领域
                    original_question = tq.get("original_question", "")
                    original_q = original_map.get(original_question, {})
                    difficulty = self._estimate_difficulty(original_q)
                    domain = self._determine_domain(original_question)

                    result.vqa_items.append(VQAItem(
                        id=qa_id,
                        question=vqa_question,
                        answer=tq.get("original_answer", ""),
                        original_question=tq.get("original_question", ""),
                        original_answer=tq.get("original_answer", ""),
                        target_entity=target_entity,
                        images=images,
                        visual_reference=tq.get("visual_reference", ""),
                        entity_type=entity_info.entity_type,
                        difficulty=difficulty,
                        domain=domain,
                        reasoning_path=original_q.get("reasoning_path", []),
                        image_url=local_image_path if local_image_path else original_image_url,
                        original_vqa_question=vqa_question_original,  # 保存原始VQA问题（简化前）
                        local_image_path=local_image_path
                    ))

                    # 每生成一个VQA就保存一次（增量保存）
                    await self._save_incremental(result, self.config.output_dir)

            except Exception as e:
                result.errors.append(f"VQA building failed: {e}")

        result.total_vqa = len(result.vqa_items)
        print(f"   Generated {result.total_vqa} VQA questions")

        # 统计各实体的VQA数量
        if result.vqa_items:
            entity_counts = {}
            for item in result.vqa_items:
                entity_counts[item.target_entity] = entity_counts.get(item.target_entity, 0) + 1
            print(f"\nVQA count by root entity:")
            for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"   - {entity}: {count}")

        return result

    async def _extract_visualizable_entities(self, image_url: str) -> List[VisualizableEntity]:
        """从图片中提取可视化实体"""
        if not image_url:
            return []

        if not self.llm_client:
            print("   WARNING: No LLM client configured")
            return []

        try:
            prompt = f"""Analyze this image and extract ALL entities that can be visually represented.

Focus on specific, named entities such as:
- People: specific individuals with names (Einstein, Steve Jobs, etc.)
- Places: landmarks, buildings, cities, countries
- Objects: products, devices, items with specific names
- Organizations: companies, institutions, museums with specific names
- Events: conferences, ceremonies, historical events

For each entity provide:
1. Entity name (in English, as specific as possible)
2. Entity type: PERSON, PLACE, OBJECT, ORGANIZATION, or EVENT
3. Brief description of what you see

Example response format:
[
    {{
        "name": "Eiffel Tower",
        "type": "PLACE",
        "description": "A famous iron lattice tower in Paris, France"
    }},
    {{
        "name": "Apple Store",
        "type": "ORGANIZATION",
        "description": "A retail store of Apple Inc. with glass facade"
    }}
]

Only extract entities that would make good anchors for questions like:
- "What is the {{entity}} in the image?"
- "Who is the {{entity}} in the image?"

Return JSON only, no other content."""

            response = await self.llm_client.generate_with_images(
                prompt=prompt,
                image_urls=[image_url],
                temperature=0.0,
                max_tokens=8192
            )

            entities = self._parse_entities_response(response)
            print(f"   Extracted {len(entities)} entities from image")
            return entities

        except Exception as e:
            print(f"   ERROR: Entity extraction failed: {e}")
            return []

    async def _extract_entities_from_questions(
        self,
        questions: List[Dict[str, Any]]
    ) -> Tuple[List[VisualizableEntity], List[Dict[str, Any]]]:
        """
        从问题中提取所有实体并转换为VQA格式（单次LLM调用）

        同时完成：
        1. 提取可可视化的实体
        2. 将问题转换为VQA格式（假设实体在图片中）

        Args:
            questions: VistaHop生成的问题列表

        Returns:
            Tuple[List[VisualizableEntity], List[Dict]]: 实体列表和转换后的问题
        """
        if not questions:
            return [], []

        if not self.llm_client:
            print("   WARNING: No LLM client configured")
            return [], []

        # 收集所有问题（包含答案 + 根节点视觉描述）
        questions_data = []
        for i, q in enumerate(questions):  # 处理所有问题，无限制
            q_text = q.get('question', '')
            q_answer = q.get('answer', '')

            # 从证据链中提取根节点的视觉描述（如果存在）
            root_visual_description = ""
            root_entity_name = ""
            try:
                evidence_chain = q.get("evidence_chain") or {}
                nodes = evidence_chain.get("nodes") or []
                if nodes:
                    root_node = nodes[0] or {}
                    root_visual_description = root_node.get("description", "") or ""
                    root_entity_name = root_node.get("entity", "") or ""
            except Exception:
                # 提取失败时忽略，不影响主流程
                root_visual_description = ""
                root_entity_name = ""

            questions_data.append({
                "index": i + 1,
                "question": q_text,
                "answer": q_answer,
                "root_entity": root_entity_name,
                "root_visual_description": root_visual_description
            })

        questions_json = json.dumps(questions_data, ensure_ascii=False, indent=2)

        try:
            prompt = f"""You are an expert at converting text QA to Visual QA (VQA).

Your task is to:
1. Transform questions to VQA format by adding visual references.
2. Each question item include a "root_visual_description" describing the ROOT node (root entity) of the reasoning chain as it visually appears in the original image. Whenever this field is non-empty, you MUST:
   - Naturally integrate this root visual description into the VQA question text so that the question explicitly reflects the visual characteristics of the root entity;
   - ALSO OBFUSCATE the root entity in this description: do NOT directly mention its concrete name, but paraphrase it into a vague, indirect visual description (e.g., "a tall iron tower by a river" instead of "the Eiffel Tower").

TRANSFORMATION RULES:
- Replace entity with a natural visual reference based on its entity type
- Recommended replacements:
  - PERSON: "this person", "the individual", "the man/woman", "the figure"
  - PLACE: "this location", "this place", "this spot", "the area"
  - OBJECT: "this object", "this item", "the thing", "the structure"
  - ORGANIZATION: "this organization/company", "the group"
  - EVENT: "this event", "the occasion"
- Naturally integrate root_visual_description into the question (if provided) as a clarifying phrase
- Examples:
- "What is Einstein known for?" → "What is the person shown in the image known for?"
- "Where was Eiffel Tower built?" → "Where was the landmark shown in the image located?"
- "Who discovered radium?" → "Who is the scientist shown in the image?"

OUTPUT FORMAT (JSON):
{{
    "entities": [
        {{
            "name": "entity name",
            "type": "PERSON|PLACE|OBJECT|ORGANIZATION|EVENT",
            "description": "why this can be shown in images"
        }}
    ],
    "transformed_questions": [
        {{
            "original_index": 1,
            "original_question": "...",
            "original_answer": "...",
            "target_entity": "entity name",
            "vqa_question": "natural transformed question with visual context"
        }}
    ]
}}

Input:
{questions_json}

Return JSON only, no other content."""

            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=8192
            )

            # 解析响应
            entities, transformed = self._parse_extraction_response(response)
            print(f"   Extracted {len(entities)} entities, transformed {len(transformed)} questions")
            return entities, transformed

        except Exception as e:
            print(f"   ERROR: Entity extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    async def _extract_root_entities_from_questions(
        self,
        questions: List[Dict[str, Any]],
        image_url: str = ""
    ) -> Tuple[List[VisualizableEntity], List[Dict[str, Any]]]:
        """
        从问题中提取根节点实体并转换为VQA格式（只处理根节点）

        关键区别于 _extract_entities_from_questions：
        1. 只提取根节点实体（seed entity）
        2. 不搜索网络图片（使用原始输入图片）
        3. 每个问题只生成一个VQA条目

        Args:
            questions: VistaHop生成的问题列表
            image_url: 原始输入图片URL（裁剪后的图片）

        Returns:
            Tuple[Dict[str, VisualizableEntity], List[Dict]]: 根节点实体字典和转换后的问题
        """
        if not questions:
            return {}, []

        if not self.llm_client:
            print("   WARNING: No LLM client configured")
            return {}, []

        # 构建 Stage1 实体名称到视觉描述的映射
        stage1_description_map = {}
        if self.config.stage1_entities:
            for entity in self.config.stage1_entities:
                name = entity.get("name", "").strip()
                desc = entity.get("description", "").strip()
                if name and desc:
                    stage1_description_map[name] = desc
            print(f"   [DEBUG] Loaded {len(stage1_description_map)} stage1 entities with visual descriptions")

        # 收集所有问题（包含答案 + 根节点信息）
        questions_data = []
        for i, q in enumerate(questions):
            q_text = q.get('question', '')
            q_answer = q.get('answer', '')

            # 从证据链中提取根节点信息
            root_entity_name = ""
            root_entity_type = "ENTITY"
            try:
                evidence_chain = q.get("evidence_chain") or {}
                nodes = evidence_chain.get("nodes") or []
                if nodes and len(nodes) > 0:
                    root_node = nodes[0] or {}
                    root_entity_name = root_node.get("entity", "") or ""
                    # 尝试从 entity_type 获取类型
                    root_entity_type = root_node.get("entity_type", "ENTITY")
                    # 如果是 seed 类型，尝试根据实体名称判断类型
                    if root_entity_type == "seed":
                        root_entity_type = self._infer_entity_type(root_entity_name)
            except Exception:
                root_entity_name = ""
                root_entity_type = "ENTITY"

            if not root_entity_name:
                print(f"   [DEBUG] q[{i}] SKIP (no root_entity): question={q_text[:60]!r}")
                continue

            # 查找 Stage1 中的视觉描述
            root_visual_description = stage1_description_map.get(root_entity_name, "")

            print(f"   [DEBUG] q[{i}] OK: root_entity={root_entity_name!r} ({root_entity_type}), "
                  f"visual_description={root_visual_description[:40]!r}..., "
                  f"question={q_text[:40]!r}...")

            questions_data.append({
                "index": i + 1,
                "question": q_text,
                "answer": q_answer,
                "root_entity": root_entity_name,
                "root_entity_type": root_entity_type,
                "root_visual_description": root_visual_description
            })

        if not questions_data:
            print("   WARNING: No root entities found in questions")
            return {}, []

        questions_json = json.dumps(questions_data, ensure_ascii=False, indent=2)

        try:
            prompt = f"""You are an expert at converting text QA to Visual QA (VQA).

Your task is to:
1. Extract ONLY the ROOT ENTITY (seed entity) from each question
2. Transform questions to VQA format by replacing the root entity with a visual reference
3. Use the root_entity_type to determine the appropriate visual reference
4. IMPORTANT: Use the "root_visual_description" field to add visual context to the question

EXTRACTION RULES:
- ONLY extract the root entity (the seed entity from the evidence chain)
- The root entity is the starting point of the reasoning chain
- Do NOT extract intermediate or leaf entities

## CRITICAL: USING ROOT_VISUAL_DESCRIPTION
Each question item includes a "root_visual_description" field. This describes how the root entity visually appears in the original image (e.g., "Brand name printed on an outdoor air conditioning unit on the side of the building").

When root_visual_description is provided, you MUST:
- Integrate it naturally into the VQA question to provide visual context
- OBFUSCATE the description: paraphrase it to be vague/indirect WITHOUT revealing the exact entity name
- Example: "Brand name printed on an outdoor air conditioning unit" → "a brand name visible on equipment outside a building"

TRANSFORMATION RULES:
- Replace ONLY the root entity with a natural visual reference based on its entity type
- Use natural visual references that fit the context
- Recommended replacements:
  - PERSON: "this person", "the individual", "the man/woman"
  - PLACE: "this location", "this place", "this spot", "the area"
  - OBJECT: "this object", "this item", "the thing", "the structure"
  - ORGANIZATION: "this organization/company", "the group"
  - EVENT: "this event", "the occasion"
- Integrate root_visual_description (if provided) into the question as a clarifying phrase

Examples:
- Question: "Identify the titan of global logistics whose central operations reside within the 'Gateway to the World.' ..."
  Root Entity: "Hapag-Lloyd" (ORGANIZATION)
  → VQA: "Identify this company whose central operations reside within the 'Gateway to the World.' ..."

- Question: "What is the company shown in the image?"
  Root Entity: "Mitsubishi Electric"
  Root Visual Description: "Brand name printed on an outdoor air conditioning unit on the side of the building to the right."
  → VQA: "What is the brand name visible on the outdoor equipment shown in the image?"

- Question: "A maritime leviathan, distinguished by its sunset-and-sea-toned livery, maintains its seat in a northern hub of the Hanseatic league. ..."
  Root Entity: "Hamburg" (PLACE)
  → VQA: "A maritime leviathan, distinguished by its sunset-and-sea-toned livery, maintains its seat in this city in northern Germany. ..."

OUTPUT FORMAT (JSON):
{{
    "root_entities": [
        {{
            "name": "entity name",
            "type": "PERSON|PLACE|OBJECT|ORGANIZATION|EVENT|ENTITY",
            "description": "description of the entity"
        }}
    ],
    "transformed_questions": [
        {{
            "original_index": 1,
            "original_question": "...",
            "original_answer": "...",
            "target_entity": "root entity name",
            "vqa_question": "transformed question with visual reference",
            "visual_reference": "the [entity type] shown in the image"
        }}
    ]
}}

IMPORTANT: You MUST use "shown in the image" in the visual_reference. Do NOT use other phrases like "depicted in the image", "visible in the image", etc.

## RULE — MULTIPLE INSTANCES OF THE SAME ENTITY TYPE IN THE IMAGE
When the image contains more than one instance of the same entity type (e.g., multiple cars, multiple people, multiple objects):
- Do NOT use "the [entity] shown in the image" (which implies a single referent).
- Use instead "one of the [entities] in t The placeholder “[entity]” should be replaced with the actual entity type, such as car, person, building, animal, etc.
- In short: if there is not exactly one A in the image, the wording must NOT be "the A shown in the image"; use "a certain A in the image" or "one of the A's in the image".

Input:
{questions_json}

Return JSON only, no other content. For example:
- If root entity is "Apple Watch", use "this watch shown in the image"
- If root entity is "John Doe", use "this person shown in the image"
- If root entity is "Paris", use "this city shown in the image"
- If there are multiple cars in the image, use "one of the cars shown in the image" or "a certain car in the image", NOT "the car shown in the image"
"""

            # 根据是否有图片选择调用方式
            if image_url:
                print(f"   [DEBUG] Using image for VQA conversion: {image_url[:80]}...")
                response = await self.llm_client.generate_with_images(
                    prompt=prompt,
                    image_urls=[image_url],
                    temperature=0.0,
                    max_tokens=32768
                )
            else:
                print(f"   [DEBUG] No image provided, using text-only mode")
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=32768
                )

            # 解析响应
            entities, transformed = self._parse_root_extraction_response(response)
            print(f"   Extracted {len(entities)} root entities, transformed {len(transformed)} questions")
            return entities, transformed

        except Exception as e:
            print(f"   ERROR: Root entity extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return {}, []

    async def _simplify_questions_batch(self, questions: List[str]) -> List[str]:
        """
        批量简化问题，减少冗余信息，使问题更简洁

        Args:
            questions: 问题列表

        Returns:
            简化后的问题列表
        """
        if not questions:
            return []

        if not self.llm_client:
            return questions

        # 构建批量处理的 prompt
        questions_data = []
        for i, q in enumerate(questions):
            if q:  # 只处理非空问题
                questions_data.append({
                    "index": i + 1,
                    "question": q
                })

        if not questions_data:
            return questions

        questions_json = json.dumps(questions_data, ensure_ascii=False, indent=2)

        try:
            prompt = f"""{questions_json}

You are an expert at rewriting VQA questions so that they can ONLY be answered by looking at the image — not by reading the question text alone.

## RULE 1 — STRIP ALL DESCRIPTIONS OF THE VISUAL ENTITY
The question may contain long descriptive phrases that effectively name or identify the visual entity through text. These are harmful and must be completely removed.
- Any textual description that reveals or hints at what the visual entity is must be DELETED.
- Replace the entire descriptive phrase with a natural visual reference like "the object shown in the image", "this item shown in the image", or "it" (e.g., "the maker of this object", "the company behind it").
- **EXCEPTION: NEVER delete "shown in the image" phrase.** The phrase "shown in the image" (e.g., "the landmark shown in the image", "this person shown in the image") MUST be kept exactly as-is.

## RULE 2 — USE PLAIN LANGUAGE
Replace verbose/pompous vocabulary with simple equivalents:
- "sovereign jurisdiction / sovereign state / sovereign territory" → "country"
- "fiscal barometer / financial benchmark / arithmetic mean" → "stock index"
- "non-credit-intermediary enterprises / non-monetary-sector titans" → "non-financial companies"
- "headquartered / administrative home / legal domicile" → "based in"
- "architect / pioneer of" → "maker of" / "company behind"
- "quintessential gauge / definitive indicator" → "main indicator"
- Any other roundabout expression → simplest direct equivalent

## RULE 3 — REMOVE UNNECESSARY MODIFIERS
Delete decorative adjectives/adverbs that do NOT affect the answer:
- "renowned / prominent / leading / prestigious / iconic / legendary" → DELETE
- "major / significant / substantial" → DELETE if not crucial for the answer
- "modern / contemporary / traditional / classic" → DELETE if not crucial
- Keep only modifiers that are essential for reasoning (e.g., "largest", "oldest", "first")

## RULE 4 — FUZZY NUMBERS AND DATES
When exact numbers/dates are NOT essential for the answer, DELETE them directly or generalize them:

- DELETE completely if not crucial: "30", "500", "1990", "mid-2010s", "century-old", "5 years", "10%"
- Generalize if deletion loses context: "thirty companies" → "a stock index of companies" / "some companies"
- Keep ONLY if crucial for the answer (e.g., "top 3", "first", "largest", "most")

## RULE 5 — MULTIPLE INSTANCES OF THE SAME ENTITY TYPE IN THE IMAGE
When the image contains more than one instance of the same entity type (e.g., multiple cars, multiple people, multiple objects):
- Do NOT use "the [entity] shown in the image" (which implies a single referent).
- Use instead "one of the [entities] in the image" or "a certain [entity] in the image" (e.g., "one of the cars shown in the image", "a certain person in the image").
- In short: if there is not exactly one A in the image, the wording must NOT be "the A shown in the image"; use "a certain A in the image" or "one of the A's in the image".

## RULE 6 — PRESERVE THE REASONING CHAIN
Keep all facts that are part of the logical path to the answer:
- Named entities NOT describing the visual entity (e.g., index names, company names, place names)
- Quantitative facts ONLY if crucial for reasoning: "thirty companies", "five hundred companies", "price-weighted"
- Unique relationships: "subsidiary of", "built by", "world's largest"

**CRITICAL: "shown in the image" MUST be preserved exactly as-is** (e.g., "the watch shown in the image", "this car shown in the image"). Do NOT modify, shorten, or remove this phrase.

## OUTPUT FORMAT (JSON):
{{
    "simplified_questions": [
        {{
            "original_index": 1,
            "original_question": "...",
            "simplified_question": "rewritten question"
        }}
    ]
}}

Return JSON only, no other content."""

            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=8192
            )

            # 解析响应
            simplified = self._parse_simplified_response(response)
            print(f"   Simplified {len(simplified)} questions")
            return simplified

        except Exception as e:
            print(f"   WARNING: Question simplification failed: {e}")
            return questions

    def _parse_simplified_response(self, response: str) -> List[str]:
        """解析简化后的问题响应"""
        import re

        # 初始化结果列表
        result = []

        # 清理响应
        clean_response = response.strip()
        if clean_response.startswith("```"):
            match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
            if match:
                clean_response = match.group(1).strip()

        try:
            data = json.loads(clean_response)

            # 构建索引到简化问题的映射
            simplified_map = {}
            for item in data.get("simplified_questions", []):
                idx = item.get("original_index", 0)
                simplified = item.get("simplified_question", "").strip()
                if simplified:
                    simplified_map[idx] = simplified

            # 按原始顺序返回
            for i in range(1, len(simplified_map) + 1):
                if i in simplified_map:
                    result.append(simplified_map[i])
                else:
                    result.append("")  # 保持索引对齐

        except json.JSONDecodeError as e:
            print(f"   WARNING: Cannot parse simplified response as JSON: {e}")
        except Exception as e:
            print(f"   WARNING: Error parsing simplified response: {e}")

        return result

    async def _validate_questions_for_leakage(
        self,
        questions: List[Dict[str, Any]],
        max_retries: int = 1
    ) -> Dict[int, Dict[str, Any]]:
        """
        验证问题是否存在信息泄露

        使用多Agent系统：
        1. Solver Agent: 只看文本，不看图片，尝试用自己的知识推理出答案
        2. Judge Agent: 比较 Solver 的答案和真实答案，评估题目是否泄露了信息

        Args:
            questions: 问题列表
            max_retries: 最大重试次数（当检测到泄露时）

        Returns:
            Dict[int, Dict]: 验证结果，key为问题索引
        """
        if not questions:
            return {}

        if not self.llm_client:
            print("   WARNING: No LLM client configured, skipping leakage validation")
            return {}

        results = {}

        # 批量处理（每批5个问题，避免token超限）
        batch_size = 5
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch = questions[batch_start:batch_end]

            print(f"   Validating batch {batch_start//batch_size + 1}/{(len(questions)-1)//batch_size + 1} ({batch_start}-{batch_end})...")

            # 构建验证prompt
            questions_data = []
            for q in batch:
                questions_data.append({
                    "index": q.get("index", 0),
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "target_entity": q.get("target_entity", "")
                })

            questions_json = json.dumps(questions_data, ensure_ascii=False, indent=2)

            try:
                prompt = f"""You are an expert at evaluating VQA questions for "information leakage".

## YOUR TASK
You have two roles:
1. SOLVER: Read the question text ONLY (no image), try to answer using your internal knowledge
2. JUDGE: Determine if the question leaks information that makes the image unnecessary

## DEFINITION OF LEAKAGE
A question "leaks information" if a person (or AI model) can answer it correctly WITHOUT looking at the image, simply by reading the question text.
- The question should require looking at the image to answer correctly
- If the text alone reveals the answer or makes it trivial to guess, it's LEAKAGE

## EVALUATION CRITERIA
For each question, determine:
1. SOLVER_ANSWER: What would you answer based ONLY on the text? (If you don't know, say "I don't know")
2. SOLVER_CONFIDENCE: 0.0-1.0 (how confident is the solver that their answer is correct based on text alone)
3. LEAKAGE_DETECTED: true/false
4. LEAKAGE_REASON: Why is this leakage? (e.g., "The description 'Steve Jobs founded' directly reveals the answer")

## KEY PRINCIPLE
The question should describe the visual entity vaguely (e.g., "this object shown in the image"), but should NOT contain clues that directly identify the entity.
- BAD: "What is the company shown in the image that was founded by Steve Jobs and is headquartered in Cupertino?"
  -> LEAKAGE: "founded by Steve Jobs" + "headquartered in Cupertino" uniquely identifies Apple without seeing image
- GOOD: "What is this company shown in the image?"
  -> NO LEAKAGE: The text alone doesn't reveal what the company is

## EXAMPLES:
Question: "What is this person shown in the image?"
- Solver Answer: "I don't know" (no info given)
- Confidence: 0.0
- Leakage: false

Question: "Who is the scientist shown in the image who discovered radium?"
- Solver Answer: "Marie Curie"
- Confidence: 0.95
- Leakage: true
- Reason: "discovered radium" uniquely identifies Marie Curie without needing the image

Question: "What is this landmark shown in the image?"
- Solver Answer: "I don't know"
- Confidence: 0.0
- Leakage: false

Question: "Where was the Eiffel Tower built?"
- Solver Answer: "Paris, France"
- Confidence: 0.99
- Leakage: true
- Reason: "Eiffel Tower" is mentioned by name, making the image unnecessary

Input:
{questions_json}

OUTPUT FORMAT (JSON):
{{
    "validations": [
        {{
            "original_index": 1,
            "solver_answer": "...",
            "solver_confidence": 0.0-1.0,
            "leakage_detected": true/false,
            "leakage_reason": "..."
        }}
    ]
}}

Return JSON only, no other content."""

                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=8192
                )

                # 解析响应
                batch_results = self._parse_validation_response(response)
                results.update(batch_results)
                print(f"   Validated {len(batch_results)} questions in this batch")

            except Exception as e:
                print(f"   WARNING: Validation failed for batch {batch_start}-{batch_end}: {e}")

        # 统计泄露情况
        leakage_count = sum(1 for v in results.values() if v.get("leakage_detected", False))
        print(f"   Leakage summary: {leakage_count}/{len(results)} questions detected with leakage")

        return results

    def _parse_validation_response(self, response: str) -> Dict[int, Dict[str, Any]]:
        """解析验证响应"""
        import re

        results = {}

        # 清理响应
        clean_response = response.strip()
        if clean_response.startswith("```"):
            match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
            if match:
                clean_response = match.group(1).strip()

        try:
            data = json.loads(clean_response)

            for item in data.get("validations", []):
                idx = item.get("original_index", 0)
                results[idx] = {
                    "solver_answer": item.get("solver_answer", ""),
                    "solver_confidence": float(item.get("solver_confidence", 0.0)),
                    "leakage_detected": item.get("leakage_detected", False),
                    "leakage_reason": item.get("leakage_reason", "")
                }

        except json.JSONDecodeError as e:
            print(f"   WARNING: Cannot parse validation response as JSON: {e}")
        except Exception as e:
            print(f"   WARNING: Error parsing validation response: {e}")

        return results

    def _infer_entity_type(self, entity_name: str) -> str:
        """根据实体名称推断类型"""
        entity_lower = entity_name.lower()

        # 人物关键词
        person_keywords = ["person", "people", "human", " figure", "celebrity", "actor", "scientist", "president", "king", "queen", "emperor"]
        if any(kw in entity_lower for kw in person_keywords):
            return "PERSON"

        # 地点关键词
        place_keywords = ["city", "country", "place", "location", "building", "tower", "museum", "park", "river", "mountain", "lake", "island", "continent"]
        if any(kw in entity_lower for kw in place_keywords):
            return "PLACE"

        # 组织关键词
        org_keywords = ["company", "corporation", "organization", "institution", "agency", "university", "museum", "association"]
        if any(kw in entity_lower for kw in org_keywords):
            return "ORGANIZATION"

        # 物体关键词
        object_keywords = ["object", "item", "thing", "product", "device", "car", "phone", "machine"]
        if any(kw in entity_lower for kw in object_keywords):
            return "OBJECT"

        # 事件关键词
        event_keywords = ["event", "war", "conference", "festival", "ceremony", "olympics"]
        if any(kw in entity_lower for kw in event_keywords):
            return "EVENT"

        # 默认返回未知类型
        return "ENTITY"

    def _parse_root_extraction_response(
        self,
        response: str
    ) -> Tuple[Dict[str, VisualizableEntity], List[Dict[str, Any]]]:
        """解析根节点提取和转换的响应"""
        import re
        import json

        entities = {}
        transformed = []

        # 清理响应 - 尝试多种方式提取JSON
        clean_response = response.strip()

        # 方式1: 尝试提取代码块中的JSON
        json_match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
        if json_match:
            clean_response = json_match.group(1).strip()
        else:
            # 方式2: 尝试找到JSON对象的开始和结束
            # 查找第一个 { 到最后一个 }
            start = clean_response.find('{')
            end = clean_response.rfind('}')
            if start != -1 and end != -1 and end > start:
                clean_response = clean_response[start:end+1]

        # 尝试解析JSON
        data = None
        parse_error = None
        try:
            data = json.loads(clean_response)
        except json.JSONDecodeError as e:
            parse_error = str(e)
            # 方式3: 尝试修复不完整的JSON
            try:
                # 使用更宽松的正则来提取 JSON 对象
                # 先尝试提取整个根对象
                root_match = re.search(r'\{[\s\S]*"root_entities"\s*:\s*\[[\s\S]*\][\s\S]*"transformed_questions"\s*:\s*\[[\s\S]*\][\s\S]*\}', clean_response)
                if root_match:
                    try:
                        data = json.loads(root_match.group(0))
                    except:
                        pass

                # 如果还是失败，尝试更简单的方式：提取每个问题的字段
                if not data:
                    transformed = []
                    # 查找所有 "original_index": 数字 的位置，然后尝试向前向后查找完整对象
                    for match in re.finditer(r'"original_index"\s*:\s*(\d+)', clean_response):
                        idx = int(match.group(1))
                        start = match.start()
                        # 向前找到 {
                        obj_start = clean_response.rfind('{', 0, start)
                        # 向后找到配对的 }
                        # 简单方法：查找下一个 "original_index" 或 "}"
                        end = clean_response.find('}', match.end())
                        if obj_start != -1 and end != -1:
                            obj_str = clean_response[obj_start:end+1]
                            try:
                                obj = json.loads(obj_str)
                                transformed.append(obj)
                            except:
                                # 尝试修复常见的JSON问题
                                # 修复单引号
                                obj_str_fixed = obj_str.replace("'", '"')
                                try:
                                    obj = json.loads(obj_str_fixed)
                                    transformed.append(obj)
                                except:
                                    pass

                    if transformed:
                        data = {"root_entities": [], "transformed_questions": transformed}
            except Exception as e2:
                parse_error = f"{parse_error}; Fix attempt failed: {str(e2)}"

        if data is None:
            print(f"   WARNING: Cannot parse response as JSON: {parse_error}")
            print(f"   Raw response: {clean_response[:1000] if clean_response else 'Empty response'}")
            return entities, transformed

        # 解析根节点实体
        for item in data.get("root_entities", []):
            name = item.get("name", "").strip()
            if name and self._is_valid_entity(name):
                entity_type = item.get("type", "ENTITY").upper()
                entities[name] = VisualizableEntity(
                    name=name,
                    entity_type=entity_type,
                    description=item.get("description", ""),
                    confidence=0.9  # 根节点置信度较高
                )

        # 解析转换后的问题
        for item in data.get("transformed_questions", []):
            target = item.get("target_entity", "").strip()
            original_q = item.get("original_question", "")
            vqa_q = item.get("vqa_question", "").strip()

            # Fallback: 如果 target_entity 为空但有 root_entity，使用 root_entity
            if not target and item.get("root_entity"):
                target = item.get("root_entity", "").strip()

            # 如果 vqa_question 为空，使用 original_question（降级处理）
            if not vqa_q and original_q:
                vqa_q = original_q

            transformed.append({
                "original_index": item.get("original_index", 0),
                "original_question": original_q,
                "original_answer": item.get("original_answer", ""),
                "target_entity": target,
                "vqa_question": vqa_q,
                "visual_reference": item.get("visual_reference", "")
            })

        return entities, transformed

    def _parse_extraction_response(
        self,
        response: str
    ) -> Tuple[List[VisualizableEntity], List[Dict[str, Any]]]:
        """解析提取和转换的响应"""
        import re
        import json

        entities = []
        transformed = []

        # 清理响应
        clean_response = response.strip()
        if clean_response.startswith("```"):
            match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
            if match:
                clean_response = match.group(1).strip()

        try:
            data = json.loads(clean_response)

            # 解析实体
            for item in data.get("entities", []):
                name = item.get("name", "").strip()
                if name and self._is_valid_entity(name):
                    entities.append(VisualizableEntity(
                        name=name,
                        entity_type=item.get("type", "OBJECT").upper(),
                        description=item.get("description", ""),
                        confidence=item.get("confidence", 0.8)
                    ))

            # 解析转换后的问题
            for item in data.get("transformed_questions", []):
                transformed.append({
                    "original_index": item.get("original_index", 0),
                    "original_question": item.get("original_question", ""),
                    "original_answer": item.get("original_answer", ""),
                    "target_entity": item.get("target_entity", ""),
                    "vqa_question": item.get("vqa_question", ""),
                    "visual_reference": item.get("visual_reference", "")
                })

        except json.JSONDecodeError as e:
            print(f"   WARNING: Cannot parse response as JSON: {e}")
        except Exception as e:
            print(f"   WARNING: Error parsing response: {e}")

        return entities, transformed

    def _is_valid_entity(self, name: str) -> bool:
        """验证实体名称是否有效"""
        # 检查长度
        if len(name) < self.config.min_entity_length:
            return False
        if len(name) > self.config.max_entity_length:
            return False

        # 过滤时间实体
        if self.config.filter_temporal_entities:
            if re.match(r'^\d{4}$', name):
                return False

        # 过滤抽象概念
        if self.config.filter_abstract_entities:
            abstract_words = ["theory", "concept", "idea", "philosophy", "method", "process"]
            if any(word.lower() in name.lower() for word in abstract_words):
                if not re.search(r"[A-Z][a-z]+['\"]?\s+\w+", name):
                    return False

        # 必须是英文
        if not re.search(r'[a-zA-Z]', name):
            return False

        return True

    async def _search_entity_images(self, entities: List[VisualizableEntity]):
        """为每个实体搜索图片"""
        if not self.serpapi_client:
            print("   WARNING: No SerpAPI client configured, skipping image search")
            return

        for entity in entities:
            try:
                # 构建搜索查询
                search_query = self._build_image_query(entity)

                # 执行图片搜索
                results = await self.serpapi_client.search_images(
                    query=search_query,
                    num_results=self.config.images_per_entity * 2
                )

                # 过滤图片
                valid_images = self._filter_images(results)
                entity.image_results = valid_images[:self.config.images_per_entity]

                if entity.image_results:
                    entity.image_url = entity.image_results[0].url

                print(f"   {entity.name}: {len(entity.image_results)} images found")

                # 避免请求过快
                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"   WARNING: Image search failed for {entity.name}: {e}")

    def _build_image_query(self, entity: VisualizableEntity) -> str:
        """构建图片搜索查询 - 直接使用实体名称"""
        return entity.name

    def _filter_images(self, images: List[ImageSearchResult]) -> List[ImageSearchResult]:
        """过滤低质量图片"""
        valid = []

        for img in images:
            # 检查尺寸
            if img.width < self.config.min_image_width:
                continue
            if img.height < self.config.min_image_height:
                continue

            # 过滤缩略图
            if "thumb" in img.url.lower() or "preview" in img.url.lower():
                continue

            valid.append(img)

        # 按尺寸排序
        valid.sort(key=lambda x: x.width * x.height, reverse=True)

        return valid

    def _find_replacable_entity(
        self,
        question: str,
        entities: List[VisualizableEntity]
    ) -> Tuple[Optional[str], Optional[VisualizableEntity]]:
        """在问题中找到可替换的实体"""
        question_lower = question.lower()

        for entity in entities:
            entity_name_lower = entity.name.lower()

            # 完全匹配
            if entity_name_lower in question_lower:
                return entity.name, entity

            # 关键词匹配
            keywords = entity.name.split()
            if len(keywords) >= 2:
                matches = sum(1 for kw in keywords if kw.lower() in question_lower)
                if matches >= len(keywords) / 2:
                    return entity.name, entity

        return None, None

    def _convert_local_path_to_base64(self, local_path: str) -> Optional[str]:
        """
        将本地图片路径转换为 base64 编码

        Args:
            local_path: 本地图片路径

        Returns:
            base64 编码的 data URI，失败返回 None
        """
        import base64
        import os

        if not local_path or not os.path.exists(local_path):
            return None

        try:
            ext = os.path.splitext(local_path)[1].lower()
            mime_type = "image/jpeg"
            if ext in [".png"]:
                mime_type = "image/png"
            elif ext in [".gif"]:
                mime_type = "image/gif"
            elif ext in [".webp"]:
                mime_type = "image/webp"

            with open(local_path, 'rb') as f:
                base64_data = base64.b64encode(f.read()).decode('utf-8')

            return f"data:{mime_type};base64,{base64_data}"
        except Exception as e:
            print(f"   [ERROR] Failed to convert {local_path} to base64: {e}")
            return None

    async def _transform_to_vqa(
        self,
        question: Dict[str, Any],
        target_entity: str,
        entity_info: VisualizableEntity,
        original_image_url: str
    ) -> Optional[VQAItem]:
        """将文本问题转换为VQA格式"""
        original_question = question.get("question", "")
        original_answer = question.get("answer", "")

        # 选择视觉引用模板
        template = self.config.visual_reference_templates[
            hash(target_entity) % len(self.config.visual_reference_templates)
        ]
        visual_reference = template.format(entity=target_entity)

        # 替换问题中的实体为视觉引用
        vqa_question = self._replace_entity_with_reference(
            original_question, target_entity, visual_reference
        )

        if not vqa_question:
            return None

        # 准备图片列表
        images = []
        for img_result in entity_info.image_results:
            images.append({
                "url": img_result.url,
                "title": img_result.title,
                "thumbnail": img_result.thumbnail_url,
                "width": img_result.width,
                "height": img_result.height,
                "source": img_result.source_url
            })

        # 生成唯一ID
        qa_id = f"vqa_{hash(f'{original_question}_{target_entity}') % 10000000:07d}"

        # 确定难度和领域
        difficulty = self._estimate_difficulty(question)
        domain = self._determine_domain(original_question)

        return VQAItem(
            id=qa_id,
            question=vqa_question,
            answer=original_answer,
            original_question=original_question,
            original_answer=original_answer,
            target_entity=target_entity,
            images=images,
            visual_reference=visual_reference,
            entity_type=entity_info.entity_type,
            difficulty=difficulty,
            domain=domain,
            reasoning_path=question.get("reasoning_path", []),
            image_url=original_image_url
        )

    def _replace_entity_with_reference(
        self,
        question: str,
        entity: str,
        reference: str
    ) -> str:
        """替换问题中的实体为视觉引用"""
        # 精确替换
        if entity in question:
            return question.replace(entity, reference)

        # 不区分大小写替换
        pattern = re.compile(re.escape(entity), re.IGNORECASE)
        if pattern.search(question):
            return pattern.sub(reference, question)

        # 部分匹配
        words = entity.split()
        for word in words:
            if len(word) > 3 and word.lower() in question.lower():
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                if pattern.search(question):
                    return pattern.sub(reference, question)

        # 前置添加
        if not entity.lower() in question.lower():
            if question.lower().startswith(("where", "what is", "who is", "which")):
                return f"{reference}, {question[0].lower() + question[1:]}"

        return None

    def _estimate_difficulty(self, question: Dict[str, Any]) -> str:
        """估计问题难度"""
        difficulty = question.get("difficulty", "")
        if difficulty in ["easy", "medium", "hard"]:
            return difficulty

        chain_depth = len(question.get("reasoning_path", []))
        if chain_depth <= 2:
            return "easy"
        elif chain_depth <= 4:
            return "medium"
        else:
            return "hard"

    def _determine_domain(self, question: str) -> str:
        """确定问题领域"""
        question_lower = question.lower()

        for domain, keywords in self.domain_keywords.items():
            if any(kw in question_lower for kw in keywords):
                return domain

        return "general"

    async def save_results(
        self,
        result: VQAGenerationResult,
        output_dir: Optional[str] = None
    ):
        """保存VQA生成结果"""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存完整结果
        full_path = os.path.join(output_dir, f"vqa_full_{timestamp}.json")
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"   Saved full results: {full_path}")

        # 保存训练格式
        training_path = os.path.join(output_dir, f"vqa_training_{timestamp}.jsonl")
        with open(training_path, 'w', encoding='utf-8') as f:
            for item in result.vqa_items:
                # 使用本地路径
                image_path = item.local_image_path if item.local_image_path else (item.images[0]["url"] if item.images else "")
                training_sample = {
                    "id": item.id,
                    "question": item.question,  # 简化后的问题
                    "answer": item.answer,
                    "image": image_path,
                    "images": [item.local_image_path if item.local_image_path else img["url"] for img in item.images],
                    "entity_type": item.entity_type,
                    "difficulty": item.difficulty,
                    "domain": item.domain,
                    "original_question": item.original_question,
                    "visual_reference": item.visual_reference,
                    "target_entity": item.target_entity,
                    "original_vqa_question": item.original_vqa_question  # 原始VQA问题（简化前）
                }
                f.write(json.dumps(training_sample, ensure_ascii=False) + "\n")
        print(f"   Saved training format: {training_path}")

    async def _save_incremental(
        self,
        result: VQAGenerationResult,
        output_dir: Optional[str] = None
    ):
        """增量保存VQA结果（每生成一个就保存）"""
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 保存增量结果
        incremental_path = os.path.join(output_dir, "vqa_incremental.json")
        with open(incremental_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


# ============= 便捷函数 =============

async def generate_vqa_from_synthesis_output(
    synthesis_output_dir: str,
    image_url: str,
    serpapi_key: str = "",
    llm_client: Optional[LLMClient] = None,
    output_dir: Optional[str] = None
) -> VQAGenerationResult:
    """
    从VistaHop输出目录生成VQA的便捷函数

    Args:
        synthesis_output_dir: VistaHop输出目录
        image_url: 原图URL
        serpapi_key: SerpAPI密钥
        llm_client: LLM客户端
        output_dir: 输出目录

    Returns:
        VQAGenerationResult: VQA生成结果
    """
    # 加载VistaHop问题
    questions_file = os.path.join(synthesis_output_dir, "questions_raw.json")
    if not os.path.exists(questions_file):
        questions_file = os.path.join(synthesis_output_dir, "synthesis_final_results.json")

    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Cannot find questions file: {synthesis_output_dir}")

    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get("questions", data)

    # 创建配置
    config = VistaHopVQAConfig(
        image_url=image_url,
        serpapi_key=serpapi_key,
        output_dir=output_dir or synthesis_output_dir.replace("_first", "_vqa").replace("_test", "_vqa")
    )

    # 创建生成器并运行
    generator = VistaHopVQAGenerator(config, llm_client)
    result = await generator.generate_vqa_from_questions(questions, image_url)

    # 保存结果
    await generator.save_results(result)

    return result


# ============= 主入口 =============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate VQA from VistaHop output")
    parser.add_argument("--synthesis-dir", type=str, required=True, help="VistaHop output directory")
    parser.add_argument("--image-url", type=str, required=True, help="Original image URL")
    parser.add_argument("--serpapi-key", type=str, default="", help="SerpAPI key")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")

    args = parser.parse_args()

    result = asyncio.run(generate_vqa_from_synthesis_output(
        synthesis_output_dir=args.synthesis_dir,
        image_url=args.image_url,
        serpapi_key=args.serpapi_key,
        output_dir=args.output_dir
    ))

    print(f"\nVQA Generation Complete!")
    print(f"   Input questions: {result.total_input}")
    print(f"   Generated VQA: {result.total_vqa}")
