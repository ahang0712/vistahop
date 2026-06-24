"""
第3阶段：证据链构建 (Evidence Chain Construction)

构建推理证据链、NLI关系验证、多样性评估

此模块整合了 EvidenceChainBuilder 的功能
"""

import asyncio
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# 处理导入路径
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_grandparent_dir = os.path.dirname(os.path.dirname(_current_dir))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _grandparent_dir not in sys.path:
    sys.path.insert(0, _grandparent_dir)

# 尝试导入 llm_client
LLMClient = None
try:
    from .llm_client import LLMClient
except ImportError:
    try:
        from llm_client import LLMClient
    except ImportError:
        pass

# 尝试导入 search_client
UnifiedSearchClient = None
WikipediaClient = None
WikidataClient = None
try:
    from .search_client import UnifiedSearchClient, WikipediaClient, WikidataClient
except ImportError:
    try:
        from search_client import UnifiedSearchClient, WikipediaClient, WikidataClient
    except ImportError:
        pass


# ============================================================================
# 数据类定义
# ============================================================================

class RelationType(Enum):
    """预定义的12种逻辑关系类型（互斥分类）"""
    # 结构关系
    PART_WHOLE = "part_whole"           # 部分-整体
    MEMBER_COLLECTION = "member_collection"  # 成员-集合
    # 依赖关系
    CAUSAL = "causal"                   # 因果
    CONDITIONAL = "conditional"         # 条件
    PURPOSE = "purpose"                 # 目的
    # 时空关系
    TEMPORAL = "temporal"               # 时间
    SPATIAL = "spatial"                 # 空间
    # 比较关系
    COMPARATIVE = "comparative"         # 比较
    SIMILAR = "similar"                 # 相似
    ANTITHETIC = "antithetic"           # 对立
    # 关联关系
    ASSOCIATIVE = "associative"         # 一般关联
    ATTRIBUTIVE = "attributive"         # 属性
    # 无关系
    NONE = "none"


class EntityType(Enum):
    """实体类型枚举（NER分类 + 扩展）"""
    # 基础类型
    PERSON = "PER"           # 人物
    ORGANIZATION = "ORG"     # 组织
    LOCATION = "LOC"         # 地点
    PRODUCT = "PROD"         # 产品
    EVENT = "EVENT"          # 事件
    WORK_OF_ART = "WOA"      # 艺术作品

    # 生物
    ANIMAL = "ANIMAL"        # 动物
    PLANT = "PLANT"          # 植物
    FOOD = "FOOD"            # 食物
    DISH = "DISH"            # 菜品

    # 物品
    VEHICLE = "VEHICLE"      # 交通工具
    MATERIAL = "MATERIAL"    # 材料
    BUILDING = "BUILDING"    # 建筑物
    BRAND = "BRAND"          # 品牌

    # 抽象概念
    ABSTRACT = "ABSTRACT"    # 抽象概念
    TECHNOLOGY = "TECH"      # 科技
    MEDICAL = "MEDICAL"      # 医学术语
    NATURAL_PHENOMENON = "NAT"  # 自然现象

    # 其他
    OTHER = "OTHER"          # 其他


@dataclass
class NLIRelation:
    """NLI关系验证结果"""
    source_entity: str
    target_entity: str
    relation_type: str
    evidence: str
    confidence: float
    is_valid: bool = False
    wikidata_frequency: int = 0  # Wikidata出现频率（越低越冷门）
    llm_familiarity_score: float = 0.5  # LLM评估的熟悉程度 (0-1, 越低越陌生)
    wikidata_relation_type: str = ""
    wikidata_property_id: str = ""
    evidence_sources: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        # 必须同时满足：1) 置信度 >= 0.9, 2) 关系类型不是 "none" 或空
        self.is_valid = self.confidence >= 0.9 and self.relation_type not in ("", "none")


@dataclass
class EntityClassification:
    """实体分类结果"""
    entity: str
    entity_type: EntityType
    confidence: float
    reasoning: str


@dataclass
class EvidenceNode:
    """证据链节点"""
    entity: str
    entity_type: str
    description: str
    depth: int
    source_url: str = ""
    related_entities: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    image_attribute: str = ""
    image_attribute_value: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity": self.entity,
            "entity_type": self.entity_type,
            "description": self.description,
            "depth": self.depth,
            "source_url": self.source_url,
            "related_entities": self.related_entities,
            "properties": self.properties,
            "image_attribute": self.image_attribute,
            "image_attribute_value": self.image_attribute_value
        }


@dataclass
class EvidenceEdge:
    """证据链边"""
    source: str
    target: str
    relation_type: str
    evidence: str
    confidence: float
    is_valid: bool
    wikidata_relation_type: str = ""
    wikidata_property_id: str = ""
    wikidata_frequency: int = 0
    evidence_sources: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "wikidata_relation_type": self.wikidata_relation_type,
            "wikidata_property_id": self.wikidata_property_id,
            "wikidata_frequency": self.wikidata_frequency,
            "evidence_sources": self.evidence_sources
        }


@dataclass
class EvidenceChain:
    """证据链"""
    seed_entity: str
    nodes: List[EvidenceNode]
    edges: List[EvidenceEdge]
    chain_depth: int
    diversity_score: float
    uniqueness_score: float
    image_attribute: str = ""
    image_attribute_value: str = ""
    answer_entity: str = ""
    answer_condition: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed_entity": self.seed_entity,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "chain_depth": self.chain_depth,
            "diversity_score": self.diversity_score,
            "uniqueness_score": self.uniqueness_score,
            "image_attribute": self.image_attribute,
            "image_attribute_value": self.image_attribute_value,
            "answer_entity": self.answer_entity,
            "answer_condition": self.answer_condition
        }

    @property
    def entities(self) -> List[str]:
        return [node.entity for node in self.nodes]


# ============================================================================
# 核心类
# ============================================================================

class EntityClassifier:
    """基于LLM的实体分类器"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def classify_entity(self, entity: str) -> EntityClassification:
        prompt = f"""Please strictly determine whether the following entity is a specific/concrete entity or an abstract/generic concept.

Entity name: {entity}

**Strict Classification Rules**:
- SPECIFIC CONCRETE: Must be a specific, named entity that is uniquely identifiable.
  - Specific persons: "Elon Musk", "Taylor Swift" (NOT "musician", "billionaire")
  - Specific organizations: "Apple Inc.", "Harvard University" (NOT "company", "university")
  - Specific locations: "Eiffel Tower", "Paris" (NOT "city", "building")
  - Specific products: "iPhone 15", "Coca-Cola" (NOT "phone", "soda")
  - Specific events: "2024 Olympics", "World War II" (NOT "war", "sports event")
  - Specific artworks: "Mona Lisa", "Harry Potter" (NOT "book", "painting")
  - Specific animals/plants/vehicles

- ABSTRACT/GENERIC: Non-specific, generic, or conceptual terms.
  - Generic categories: "technology", "music", "food" (NOT specific brand/product)
  - Emotions: "happiness", "love"
  - Concepts: "justice", "freedom", "democracy"
  - Theories: "relativity", "evolution"
  - Generic roles: "teacher", "doctor" (NOT specific person)

**Classification options**:
- PER (PERSON): Specific person name (e.g., "Elon Musk")
- ORG (ORGANIZATION): Specific company, institution (e.g., "Apple Inc.")
- LOC (LOCATION): Specific location (e.g., "Eiffel Tower")
- PROD (PRODUCT): Specific product/brand (e.g., "iPhone 15")
- EVENT (EVENT): Specific event (e.g., "2024 Olympics")
- WOA (WORK_OF_ART): Specific artwork (e.g., "Mona Lisa")
- ANIMAL: Specific animal (e.g., "Golden Retriever")
- PLANT: Specific plant (e.g., "Rose")
- VEHICLE: Specific vehicle (e.g., "Boeing 747")
- FOOD: Food item (e.g., "Sushi", "Pizza")
- DISH: Specific dish name (e.g., "Kung Pao Chicken")
- MATERIAL: Material (e.g., "Steel", "Cotton")
- BUILDING: Building (e.g., "Empire State Building")
- BRAND: Brand name (e.g., "Nike")
- TECH: Technology (e.g., "AI", "Blockchain")
- MEDICAL: Medical term (e.g., "Diabetes")
- NAT: Natural phenomenon (e.g., "Earthquake", "Rainbow")
- OTHER: Other specific/concrete entities
- ABSTRACT: Abstract concept or generic term

**Output format (JSON)**:
{{"entity_type": "ORG", "confidence": 0.95, "reasoning": "reason"}}

Return JSON only, no other content."""

        try:
            response = await self.llm_client.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)

            type_map = {
                "PER": EntityType.PERSON, "ORG": EntityType.ORGANIZATION,
                "LOC": EntityType.LOCATION, "PROD": EntityType.PRODUCT,
                "EVENT": EntityType.EVENT, "WOA": EntityType.WORK_OF_ART,
                "ANIMAL": EntityType.ANIMAL, "PLANT": EntityType.PLANT,
                "VEHICLE": EntityType.VEHICLE, "OTHER": EntityType.OTHER,
                "ABSTRACT": EntityType.ABSTRACT,
                "FOOD": EntityType.FOOD, "DISH": EntityType.DISH,
                "MATERIAL": EntityType.MATERIAL, "BUILDING": EntityType.BUILDING,
                "BRAND": EntityType.BRAND, "TECH": EntityType.TECHNOLOGY,
                "MEDICAL": EntityType.MEDICAL, "NAT": EntityType.NATURAL_PHENOMENON
            }

            return EntityClassification(
                entity=entity,
                entity_type=type_map.get(data.get("entity_type", "OTHER"), EntityType.OTHER),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "")
            )
        except Exception as e:
            return EntityClassification(entity=entity, entity_type=EntityType.OTHER, confidence=0.5, reasoning=f"Error: {e}")

    async def classify_entities_batch(self, entities: List[str]) -> Dict[str, EntityClassification]:
        if not entities:
            return {}

        entities_list = "\n".join([f"{i+1}. {e}" for i, e in enumerate(entities)])

        prompt = f"""Please strictly classify the following {len(entities)} entities as SPECIFIC/CONCRETE or ABSTRACT/GENERIC.

**Entities to classify**:
{entities_list}

**Strict Classification Rules**:
- SPECIFIC CONCRETE: Must be a specific, named entity that is uniquely identifiable.
  - Specific persons: "Elon Musk" (NOT "musician", "billionaire")
  - Specific organizations: "Apple Inc." (NOT "company", "university")
  - Specific locations: "Eiffel Tower" (NOT "city", "building")
  - Specific products: "iPhone 15" (NOT "phone", "soda")
  - Specific events: "2024 Olympics" (NOT "war", "sports event")
  - Specific artworks: "Mona Lisa" (NOT "book", "painting")

- ABSTRACT/GENERIC: Non-specific, generic, or conceptual terms.
  - Generic categories: "technology", "music", "food"
  - Emotions: "happiness", "love"
  - Concepts: "justice", "freedom", "democracy"
  - Generic roles: "teacher", "doctor"

**Classification options**:
- PER (PERSON): Specific person name
- ORG (ORGANIZATION): Specific company/institution
- LOC (LOCATION): Specific location
- PROD (PRODUCT): Specific product/brand
- EVENT (EVENT): Specific event
- WOA (WORK_OF_ART): Specific artwork
- ANIMAL: Specific animal
- PLANT: Specific plant
- VEHICLE: Specific vehicle
- FOOD: Food item (e.g., "Sushi", "Pizza")
- DISH: Specific dish name (e.g., "Kung Pao Chicken")
- MATERIAL: Material (e.g., "Steel", "Cotton")
- BUILDING: Building (e.g., "Empire State Building")
- BRAND: Brand name (e.g., "Nike")
- TECH: Technology (e.g., "AI", "Blockchain")
- MEDICAL: Medical term (e.g., "Diabetes")
- NAT: Natural phenomenon (e.g., "Earthquake", "Rainbow")
- OTHER: Other specific/concrete entities
- ABSTRACT: Abstract concept or generic term

**Output format (JSON array)**:
[
    {{"entity": "Entity1", "entity_type": "ORG", "confidence": 0.95, "reasoning": "reason"}}
]

Return JSON array only, no other content."""

        try:
            response = await self.llm_client.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)
            if not isinstance(data, list):
                return {}

            type_map = {
                "PER": EntityType.PERSON, "ORG": EntityType.ORGANIZATION,
                "LOC": EntityType.LOCATION, "PROD": EntityType.PRODUCT,
                "EVENT": EntityType.EVENT, "WOA": EntityType.WORK_OF_ART,
                "ANIMAL": EntityType.ANIMAL, "PLANT": EntityType.PLANT,
                "VEHICLE": EntityType.VEHICLE, "OTHER": EntityType.OTHER,
                "ABSTRACT": EntityType.ABSTRACT,
                "FOOD": EntityType.FOOD, "DISH": EntityType.DISH,
                "MATERIAL": EntityType.MATERIAL, "BUILDING": EntityType.BUILDING,
                "BRAND": EntityType.BRAND, "TECH": EntityType.TECHNOLOGY,
                "MEDICAL": EntityType.MEDICAL, "NAT": EntityType.NATURAL_PHENOMENON
            }

            classifications = {}
            for item in data:
                entity = item.get("entity", "")
                if not entity:
                    continue
                classifications[entity] = EntityClassification(
                    entity=entity,
                    entity_type=type_map.get(item.get("entity_type", "OTHER"), EntityType.OTHER),
                    confidence=float(item.get("confidence", 0.5)),
                    reasoning=item.get("reasoning", "")
                )

            for entity in entities:
                if entity not in classifications:
                    classifications[entity] = EntityClassification(entity=entity, entity_type=EntityType.OTHER, confidence=0.5, reasoning="Default")

            return classifications
        except Exception as e:
            return {e: await self.classify_entity(e) for e in entities}


class NLIValidator:
    """NLI关系验证器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.relation_types = [rt.value for rt in RelationType]

    async def validate_relation(self, source_entity: str, target_entity: str, relation_type: str, context: str = "") -> NLIRelation:
        prompt = f"""Determine if there is a "{relation_type}" relationship between:

Source: {source_entity}
Target: {target_entity}

Valid types (mutually exclusive):
- part_whole, member_collection, causal, conditional, purpose
- temporal, spatial, comparative, similar, antithetic
- associative, attributive, none

Return JSON:
{{"has_relation": true/false, "relation_type": "type", "confidence": 0.0-1.0, "evidence": "reason"}}

Return JSON only."""

        try:
            response = await self.llm.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)
            return NLIRelation(
                source_entity=source_entity,
                target_entity=target_entity,
                relation_type=data.get("relation_type", ""),
                evidence=data.get("evidence", ""),
                confidence=float(data.get("confidence", 0.0))
            )
        except Exception as e:
            return NLIRelation(source_entity=source_entity, target_entity=target_entity, relation_type="", evidence="", confidence=0.0)

    async def validate_relations_batch(self, seed_entity: str, neighbors: List[Dict[str, Any]], seed_description: str = "") -> List[NLIRelation]:
        if not neighbors:
            return []

        if seed_description:
            seed_info = f"**Seed Entity**: {seed_entity}\n   Description: {seed_description}\n\n"
        else:
            seed_info = ""

        pairs_list = []
        for i, n in enumerate(neighbors, 1):
            entity_name = n.get('entity', '')
            entity_desc = n.get('description', '')
            evidence = n.get("evidence", "")
            evidence_sources = n.get("evidence_sources", []) or []
            source_lines = []
            for src in evidence_sources[:4]:
                source_name = src.get("source", "")
                detail = src.get("detail", "") or src.get("property_id", "") or src.get("url", "")
                if source_name or detail:
                    source_lines.append(f"{source_name}: {detail}".strip(": "))
            pair_text = f"**Pair {i}**: {seed_entity} -> {entity_name} ({n.get('type', '')})"
            if entity_desc:
                pair_text += f"\n   Description: {entity_desc}"
            if evidence:
                pair_text += f"\n   Evidence: {evidence[:700]}"
            if source_lines:
                pair_text += f"\n   Sources: {'; '.join(source_lines)}"
            pairs_list.append(pair_text)

        prompt = f"""Determine NLI relationships for:
{seed_info}{chr(10).join(pairs_list)}

Types (choose ONE, mutually exclusive):
- part_whole: A is part of B or contains B
- member_collection: A belongs to group B
- causal: A causes B
- conditional: A leads to B (if A then B)
- purpose: A is for achieving B
- temporal: A happens before/after/simultaneously with B
- spatial: A is located near/inside/far from B
- comparative: A is bigger/smaller/better/worse than B
- similar: A is similar to B
- antithetic: A is opposite to B
- associative: A is generally related to B (default)
- attributive: A has property B
- none: no relation

Use the provided Wikidata/Wikipedia evidence when available. If the evidence does not support a meaningful relation, choose "none" even if the names seem loosely related.

**Output format (JSON array)**:
[
    {{"index": 1, "relation_type": "part_whole", "confidence": 0.9, "evidence": "reason"}}
]

Return JSON array only."""

        try:
            response = await self.llm.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)
            if not isinstance(data, list):
                return []

            neighbor_list = neighbors
            relations = []

            for item in data:
                idx = item.get("index", 0)
                if idx < 1 or idx > len(neighbor_list):
                    continue

                neighbor = neighbor_list[idx - 1]
                neighbor_entity = neighbor.get("entity", "")
                llm_evidence = item.get("evidence", "")
                external_evidence = neighbor.get("evidence", "")
                if external_evidence and external_evidence not in llm_evidence:
                    evidence = f"{llm_evidence} | Evidence: {external_evidence[:700]}" if llm_evidence else external_evidence[:700]
                else:
                    evidence = llm_evidence
                relations.append(NLIRelation(
                    source_entity=seed_entity,
                    target_entity=neighbor_entity,
                    relation_type=item.get("relation_type", "none") or "",
                    evidence=evidence,
                    confidence=float(item.get("confidence", 0.0)),
                    wikidata_frequency=neighbor.get("wikidata_frequency", 0),
                    wikidata_relation_type=neighbor.get("type", ""),
                    wikidata_property_id=neighbor.get("property_id", ""),
                    evidence_sources=neighbor.get("evidence_sources", []) or []
                ))

            return relations
        except Exception as e:
            return []


class DiversityEvaluator:
    """多样性评估器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def calculate_semantic_diversity(self, entities: List[str], relations: List[str]) -> float:
        if len(entities) <= 1:
            return 1.0
        entity_diversity = min(1.0, len(entities) / 10)
        if len(relations) > 0:
            unique_relations = len(set(relations))
            relation_diversity = min(1.0, unique_relations / len(relations))
        else:
            relation_diversity = 0.0
        return 0.6 * entity_diversity + 0.4 * relation_diversity

    def calculate_expansion_priority(self, neighbor_info: Dict[str, Any], existing_relations: List[str], chain_entities: List[str]) -> float:
        relation_type = neighbor_info.get('relation_type', 'unknown')
        confidence = neighbor_info.get('confidence', 0.5)
        entity = neighbor_info.get('entity', '')

        novelty = 1.0 - (existing_relations.count(relation_type) / max(1, len(existing_relations)))
        entity_novelty = 0.1 if entity in chain_entities else 1.0

        return 0.4 * novelty + 0.3 * 1.0 + 0.2 * entity_novelty + 0.1 * confidence


class EvidenceChainBuilder:
    """证据链构建器 - 实现VistaHop论文中的证据链构建流程"""

    def __init__(self, llm_client: LLMClient, search_client: UnifiedSearchClient = None, max_depth: int = 5, min_diversity: float = 0.5, confidence_threshold: float = 0.9, proxy: str = None, llm_model: str = "gemini-3-flash-preview"):
        self.llm = llm_client
        self.llm_model = llm_model
        self.search_client = search_client or UnifiedSearchClient()

        client_proxy = proxy
        if not client_proxy and hasattr(self.search_client, 'config'):
            client_proxy = getattr(self.search_client.config, 'proxy', None)

        self.wikipedia = WikipediaClient("en", proxy=client_proxy)
        self.wikipedia_zh = WikipediaClient("zh", proxy=client_proxy)
        self.wikidata = WikidataClient(proxy=client_proxy)

        self.max_depth = max_depth
        self.min_diversity = min_diversity
        self.confidence_threshold = confidence_threshold

        self.nli_validator = NLIValidator(llm_client)
        self.diversity_evaluator = DiversityEvaluator(llm_client)
        self.entity_classifier = EntityClassifier(llm_client)
        # 全局已选实体缓存，避免同一图片处理过程中重复选择相同的实体
        self._chosen_entities: Set[str] = set()

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        return re.sub(r"\s+", " ", (name or "").replace("_", " ").strip().lower())

    @staticmethod
    def _claim_entity_ids(claims: Dict[str, List[Dict]], prop_id: str) -> Set[str]:
        ids = set()
        for claim in claims.get(prop_id, []) or []:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            if datavalue.get("type") == "wikibase-entityid":
                value = datavalue.get("value", {})
                entity_id = value.get("id")
                if entity_id:
                    ids.add(entity_id)
        return ids

    async def _resolve_wikidata_entity(self, entity_name: str, languages: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """多语言搜索 + label/description/sitelink/instance_of 消歧，避免直接取第一个QID。"""
        languages = languages or ["en", "zh"]
        try:
            if hasattr(self.wikidata, "search_entity_multilingual"):
                search_results = await self.wikidata.search_entity_multilingual(entity_name, limit=5, languages=languages)
            else:
                search_results = await self.wikidata.search_entity(entity_name, limit=5)
        except Exception:
            search_results = []

        if not search_results:
            return None

        query_norm = self._normalize_entity_name(entity_name)
        scored = []
        for rank, result in enumerate(search_results):
            entity_id = result.get("id")
            if not entity_id:
                continue
            detail = await self.wikidata.get_entity(entity_id)
            if not detail:
                continue

            labels = set(detail.labels.values()) if getattr(detail, "labels", None) else set()
            if result.get("label"):
                labels.add(result.get("label", ""))
            label_norms = {self._normalize_entity_name(label) for label in labels if label}
            description = detail.description or result.get("description", "")
            description_lower = description.lower()
            sitelinks = getattr(detail, "sitelinks", {}) or {}
            instance_ids = self._claim_entity_ids(detail.claims, "P31")

            score = max(0, 5 - rank)
            if query_norm in label_norms:
                score += 10
            elif any(query_norm and (query_norm in label or label in query_norm) for label in label_norms):
                score += 4
            if description:
                score += 1
            if sitelinks.get("enwiki"):
                score += 3
            if sitelinks.get("zhwiki"):
                score += 2
            if "Q4167410" in instance_ids or "disambiguation" in description_lower:
                score -= 20
            if "wikimedia category" in description_lower or "category page" in description_lower:
                score -= 8

            scored.append({
                "id": entity_id,
                "label": detail.label or result.get("label", entity_name),
                "description": description,
                "claims": detail.claims,
                "labels": getattr(detail, "labels", {}) or {},
                "sitelinks": sitelinks,
                "score": score,
                "search_result": result
            })

        if not scored:
            return None

        scored.sort(key=lambda item: item["score"], reverse=True)
        best = scored[0]
        if best["score"] < 0 and len(scored) > 1:
            return scored[1]
        return best

    async def _get_wikipedia_context(self, entity_label: str, sitelinks: Dict[str, str] = None) -> Dict[str, Any]:
        sitelinks = sitelinks or {}
        attempts = [
            ("en", sitelinks.get("enwiki") or entity_label),
            ("zh", sitelinks.get("zhwiki") or entity_label)
        ]
        for language, title in attempts:
            if not title:
                continue
            client = self.wikipedia if language == "en" else self.wikipedia_zh
            try:
                context = await client.get_page_context(title, max_links=50, max_categories=20)
            except Exception:
                context = {}
            if context and (context.get("extract") or context.get("links") or context.get("categories")):
                return context
        return {}

    @staticmethod
    def _find_sentence_with_mention(text: str, mention: str, max_len: int = 320) -> str:
        if not text or not mention:
            return ""
        mention_norm = mention.lower()
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if mention_norm in sentence.lower():
                return sentence[:max_len]
        return ""

    def _build_wikidata_evidence(self, source_entity: str, neighbor: Dict[str, Any], source_context: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
        target = neighbor.get("label", "")
        relation_type = neighbor.get("relation_type", "related")
        property_id = neighbor.get("property_id", "")
        sentence = self._find_sentence_with_mention(source_context.get("extract", ""), target)
        evidence_parts = [f"Wikidata {property_id} ({relation_type}) links {source_entity} to {target}."]
        sources = [{
            "source": "wikidata",
            "property_id": property_id,
            "detail": relation_type,
            "target_entity_id": neighbor.get("entity_id", "")
        }]
        if sentence:
            evidence_parts.append(f"Wikipedia sentence: {sentence}")
            sources.append({
                "source": "wikipedia",
                "url": source_context.get("url", ""),
                "detail": sentence
            })
        return " ".join(evidence_parts), sources

    @staticmethod
    def _is_noisy_wikipedia_link(title: str) -> bool:
        title = (title or "").strip()
        if not title:
            return True
        title_lower = title.lower()
        noisy_prefixes = (
            "list of ",
            "lists of ",
            "index of ",
            "outline of ",
            "timeline of ",
            "glossary of ",
        )
        noisy_contains = (
            "disambiguation",
            "bibliography",
            "discography",
            "filmography",
            "template:",
            "category:",
            "portal:",
            "draft:",
            "help:",
            "wikipedia:",
        )
        if title_lower.startswith(noisy_prefixes):
            return True
        if any(token in title_lower for token in noisy_contains):
            return True
        if re.fullmatch(r"\d{3,4}", title) or re.fullmatch(r"\d{1,2}(st|nd|rd|th)? century", title_lower):
            return True
        if re.fullmatch(r"\d{4} in .+", title_lower) or re.fullmatch(r".+ in \d{4}", title_lower):
            return True
        return False

    @staticmethod
    def _is_noisy_wikipedia_category(category: str) -> bool:
        category = (category or "").strip()
        if not category:
            return True
        category_lower = category.lower()
        noisy_prefixes = (
            "articles ",
            "all articles",
            "all pages",
            "pages ",
            "wikipedia ",
            "cs1 ",
            "webarchive ",
            "use ",
            "commons category",
            "short description",
            "coordinates ",
            "hidden categories",
        )
        noisy_contains = (
            "articles with",
            "articles needing",
            "articles lacking",
            "articles containing",
            "all stub articles",
            "stub articles",
            "maintenance",
            "cleanup",
            "unreferenced",
            "wikidata",
            "infobox",
            "disambiguation",
            "redirects",
            "births",
            "deaths",
        )
        if category_lower.startswith(noisy_prefixes):
            return True
        if any(token in category_lower for token in noisy_contains):
            return True
        if re.fullmatch(r"\d{3,4}s? .*", category_lower):
            return True
        return False

    def _build_wikipedia_candidates(self, source_context: Dict[str, Any], current_entity: str) -> List[Dict[str, Any]]:
        if not source_context:
            return []
        candidates = []
        current_norm = self._normalize_entity_name(current_entity)
        for title in source_context.get("links", [])[:30]:
            if not title or self._normalize_entity_name(title) == current_norm:
                continue
            if self._is_noisy_wikipedia_link(title):
                continue
            evidence = f"Wikipedia page for {current_entity} links to {title}."
            sentence = self._find_sentence_with_mention(source_context.get("extract", ""), title)
            if sentence:
                evidence += f" Sentence: {sentence}"
            candidates.append({
                "entity": title,
                "type": "wikipedia_link",
                "property_id": "",
                "wikidata_frequency": 1,
                "description": f"Linked from the Wikipedia page for {current_entity}",
                "relation": f"Wikipedia link from {current_entity}",
                "entity_id": "",
                "frequency": "medium",
                "occurrence_count": 0,
                "evidence": evidence,
                "source_url": source_context.get("url", ""),
                "evidence_sources": [{
                    "source": "wikipedia",
                    "url": source_context.get("url", ""),
                    "detail": "page link"
                }]
            })

        for category in source_context.get("categories", [])[:12]:
            category_name = category.replace("_", " ").strip()
            if not category_name or self._normalize_entity_name(category_name) == current_norm:
                continue
            if self._is_noisy_wikipedia_category(category_name):
                continue
            candidates.append({
                "entity": category_name,
                "type": "wikipedia_category",
                "property_id": "",
                "wikidata_frequency": 1,
                "description": f"Category on the Wikipedia page for {current_entity}",
                "relation": f"Wikipedia category of {current_entity}",
                "entity_id": "",
                "frequency": "medium",
                "occurrence_count": 0,
                "evidence": f"Wikipedia page for {current_entity} belongs to category {category_name}.",
                "source_url": source_context.get("url", ""),
                "evidence_sources": [{
                    "source": "wikipedia",
                    "url": source_context.get("url", ""),
                    "detail": "page category"
                }]
            })
        return candidates

    def _merge_neighbor_candidates(self, neighbors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for neighbor in neighbors:
            name = (neighbor.get("entity") or "").strip()
            if not name:
                continue
            key = self._normalize_entity_name(name)
            if key not in merged:
                merged[key] = neighbor
                continue

            existing = merged[key]
            existing_sources = existing.get("evidence_sources", []) or []
            existing_sources.extend(neighbor.get("evidence_sources", []) or [])
            existing["evidence_sources"] = existing_sources[:8]
            if neighbor.get("property_id") and not existing.get("property_id"):
                existing["property_id"] = neighbor.get("property_id", "")
                existing["type"] = neighbor.get("type", existing.get("type", ""))
            if neighbor.get("entity_id") and not existing.get("entity_id"):
                existing["entity_id"] = neighbor.get("entity_id", "")
            if neighbor.get("description") and not existing.get("description"):
                existing["description"] = neighbor.get("description", "")
            if neighbor.get("evidence"):
                evidence = existing.get("evidence", "")
                if neighbor["evidence"] not in evidence:
                    existing["evidence"] = f"{evidence} {neighbor['evidence']}".strip()[:1000]
        return list(merged.values())

    async def close(self):
        """释放内部资源（避免 aiohttp ClientSession 泄漏）"""
        try:
            if hasattr(self, "wikidata") and self.wikidata:
                await self.wikidata.close()
        except Exception:
            # 关闭失败不应影响主流程退出
            pass

    async def build_chain(self, seed_entity: str, num_chains: int = 3, image_attribute: str = "", image_attribute_value: str = "", stage2_nodes: List[Dict] = None) -> List[EvidenceChain]:
        print(f"\n🔗 开始为实体 '{seed_entity}' 构建证据链...")

        existing_entities = {seed_entity}

        print(f"   📚 Step 1: 收集邻居实体...")

        # 严格筛选：只收集非抽象实体
        neighbors = await self._collect_neighbors(seed_entity, existing_entities, filter_level=0)

        if not neighbors:
            print(f"   ⚠️ 未找到邻居实体")
            return []

        print(f"   ✅ 找到 {len(neighbors)} 个邻居实体")

        # 基于 Stage2 properties 用 LLM 预筛选“更有线索/更可推理”的下一跳
        seed_props = self._extract_properties_from_stage2(stage2_nodes, seed_entity)
        # 从 stage2_nodes 提取种子实体的 description
        seed_description = self._extract_description_from_stage2(stage2_nodes, seed_entity)
        if seed_props:
            print(f"   🧠 基于种子实体 properties 进行邻居预筛选...")
            original_count = len(neighbors)
            neighbors = await self._select_neighbors_by_properties(
                current_entity=seed_entity,
                neighbors=neighbors,
                current_properties=seed_props,
                top_k=10
            )
            if not neighbors:
                print(f"   ⚠️ properties 预筛选后没有可用邻居")
                return []
            print(f"   ✅ properties 预筛选后保留 {len(neighbors)}/{original_count} 个候选邻居")

        print(f"   🔍 Step 2: NLI关系验证...")
        validated_edges = await self._validate_relations(seed_entity, neighbors, seed_description=seed_description)

        valid_edges = [e for e in validated_edges if e.is_valid]
        print(f"   ✅ {len(valid_edges)}/{len(validated_edges)} 条边通过NLI验证")

        if not valid_edges:
            return []

        print(f"   🌳 Step 3: 深度优先扩展...")
        chains = await self._depth_first_expansion(
            seed_entity,
            valid_edges,
            num_chains,
            existing_entities,
            image_attribute,
            image_attribute_value,
            max_depth=self.max_depth,
            stage2_nodes=stage2_nodes
        )

        print(f"   ✅ 生成了 {len(chains)} 条证据链")

        for chain in chains:
            chain.image_attribute = image_attribute
            chain.image_attribute_value = image_attribute_value or seed_entity

        return chains

    async def _collect_neighbors(self, entity: str, existing_entities: Optional[Set[str]] = None, filter_level: int = 0) -> List[Dict[str, Any]]:
        existing_entities = existing_entities or set()

        try:
            print(f"      🔍 从Wikidata多语言搜索并消歧实体 '{entity}'...")
            resolved_entity = await self._resolve_wikidata_entity(entity)

            if not resolved_entity:
                return []

            entity_id = resolved_entity.get("id")
            entity_label = resolved_entity.get("label", entity)
            sitelinks = resolved_entity.get("sitelinks", {}) or {}

            print(f"      ✅ 消歧选择Wikidata实体: {entity_label} ({entity_id}, score={resolved_entity.get('score', 0):.1f})")

            source_context = await self._get_wikipedia_context(entity_label, sitelinks)

            print(f"      📚 获取Wikidata邻居实体...")
            wikidata_neighbors = await self.wikidata.get_all_neighbors(entity_id)

            print(f"      ✅ 找到 {len(wikidata_neighbors)} 个Wikidata邻居")

            all_neighbors = []
            for n in wikidata_neighbors:
                neighbor_label = n.get("label", "")
                if not neighbor_label:
                    continue

                relation_type = n.get("relation_type", "related")
                property_id = n.get("property_id", "P0")
                wikidata_freq = n.get("frequency", 1)
                evidence, evidence_sources = self._build_wikidata_evidence(entity_label, n, source_context)

                all_neighbors.append({
                    "entity": neighbor_label,
                    "type": relation_type,
                    "property_id": property_id,
                    "wikidata_frequency": wikidata_freq,
                    "description": n.get("description", ""),
                    "relation": f"{relation_type} of {entity}",
                    "entity_id": n.get("entity_id", ""),
                    "frequency": "high",
                    "occurrence_count": 0,
                    "evidence": evidence,
                    "source_url": source_context.get("url", ""),
                    "evidence_sources": evidence_sources
                })

            wikipedia_neighbors = self._build_wikipedia_candidates(source_context, entity_label)
            if wikipedia_neighbors:
                print(f"      ✅ 从Wikipedia links/categories补充 {len(wikipedia_neighbors)} 个候选")
                all_neighbors.extend(wikipedia_neighbors)

            unique_neighbors = self._merge_neighbor_candidates(all_neighbors)
            existing_norms = {self._normalize_entity_name(item) for item in existing_entities}
            filtered_neighbors = [
                n for n in unique_neighbors
                if n.get("entity") and self._normalize_entity_name(n.get("entity")) not in existing_norms
            ]

            if not filtered_neighbors:
                return []

            print(f"      🏷️ 使用NER分类器过滤...")
            neighbor_entities = [n.get("entity", "") for n in filtered_neighbors]
            classifications = await self.entity_classifier.classify_entities_batch(neighbor_entities)

            # 严格筛选：排除抽象实体，保留其他所有实体类型（不限于视觉实体）
            valid_entities = set()
            abstract_entities = set()
            for entity, classification in classifications.items():
                if classification.entity_type == EntityType.ABSTRACT:
                    abstract_entities.add(entity)
                else:
                    # 保留所有非抽象实体（PERSON, LOCATION, ORGANIZATION, PRODUCT 等）
                    valid_entities.add(entity)

            # 记录被排除的实体（抽象实体）- 需要在过滤前记录原始编号
            original_neighbors = filtered_neighbors.copy()

            # 重新过滤：只保留非抽象实体
            filtered_neighbors = [n for n in filtered_neighbors if n.get("entity") not in abstract_entities]

            if not filtered_neighbors:
                return []

            # 打印调试信息：找出被排除的实体在原始列表中的编号
            excluded_entities_info = []
            for i, n in enumerate(original_neighbors):
                entity = n.get("entity", "")
                if entity in abstract_entities:
                    excluded_entities_info.append(f"#{i+1}: {entity}")

            # 打印调试信息
            kept_entities_info = []
            for i, n in enumerate(original_neighbors):
                entity = n.get("entity", "")
                if entity in valid_entities:
                    kept_entities_info.append(f"#{i+1}: {entity}")

            if excluded_entities_info:
                print(f"      ✅ 严格筛选后保留 {len(filtered_neighbors)} 个有效实体 (排除{len(abstract_entities)}个抽象实体: {', '.join(excluded_entities_info[:5])}{'...' if len(excluded_entities_info) > 5 else ''})")
                print(f"      📋 保留的实体: {', '.join(kept_entities_info[:10])}{'...' if len(kept_entities_info) > 10 else ''}")
            else:
                print(f"      ✅ 严格筛选后保留 {len(filtered_neighbors)} 个有效实体")
                print(f"      📋 保留的实体: {', '.join(kept_entities_info[:10])}{'...' if len(kept_entities_info) > 10 else ''}")

            return filtered_neighbors[:20]

        except Exception as e:
            print(f"   ⚠️ 获取邻居实体失败: {e}")
            return []

    def _extract_properties_from_stage2(self, stage2_nodes: Optional[List[Dict[str, Any]]], entity: str) -> Dict[str, Any]:
        if not stage2_nodes:
            return {}
        for node in stage2_nodes:
            if node.get("entity") == entity:
                props = node.get("properties", {}) or {}
                # 允许 properties 为非空 dict
                return props if isinstance(props, dict) else {}
        return {}

    def _extract_description_from_stage2(self, stage2_nodes: Optional[List[Dict[str, Any]]], entity: str) -> str:
        """从 stage2_nodes 的 entities_found 中提取实体的 description"""
        if not stage2_nodes:
            return ""
        for node in stage2_nodes:
            if node.get("entity") == entity:
                entities_found = node.get("entities_found", [])
                for e in entities_found:
                    if e.get("name") == entity:
                        return e.get("description", "") or ""
        return ""

    async def _fetch_entity_properties_from_wikidata(self, entity_name: str) -> Dict[str, Any]:
        """
        从 Wikidata 获取实体的属性信息

        Args:
            entity_name: 实体名称

        Returns:
            属性字典
        """
        try:
            resolved_entity = await self._resolve_wikidata_entity(entity_name)
            if not resolved_entity:
                return {}

            entity_id = resolved_entity.get("id")
            if not entity_id:
                return {}

            # 获取实体详情
            entity = await self.wikidata.get_entity(entity_id)
            if not entity:
                return {}

            # 收集所有需要解析的实体 ID
            entity_ids_to_resolve = set()

            # 提取重要属性
            properties = {}

            # 添加标签和描述
            if entity.label:
                properties["label"] = entity.label
            if entity.description:
                properties["description"] = entity.description
            if getattr(entity, "sitelinks", None):
                properties["wikipedia_sitelinks"] = entity.sitelinks

            # 从 claims 中提取重要属性
            if entity.claims:
                # 提取 P31 (instance of) - 类型
                if "P31" in entity.claims:
                    instances = []
                    for claim in entity.claims["P31"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            value = datavalue.get("value", {})
                            instances.append(value.get("id", ""))
                            entity_ids_to_resolve.add(value.get("id", ""))
                    if instances:
                        properties["instance_of"] = instances[:3]  # 最多保留3个

                # 提取 P569 (date of birth) / P570 (date of death)
                if "P569" in entity.claims:
                    dates = []
                    for claim in entity.claims["P569"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "time":
                            dates.append(datavalue.get("value", ""))
                    if dates:
                        properties["date_of_birth"] = dates[0]

                if "P570" in entity.claims:
                    dates = []
                    for claim in entity.claims["P570"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "time":
                            dates.append(datavalue.get("value", ""))
                    if dates:
                        properties["date_of_death"] = dates[0]

                # 提取 P19 (place of birth)
                if "P19" in entity.claims:
                    places = []
                    for claim in entity.claims["P19"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            value = datavalue.get("value", {})
                            places.append(value.get("id", ""))
                            entity_ids_to_resolve.add(value.get("id", ""))
                    if places:
                        properties["place_of_birth"] = places[0]

                # 提取 P27 (country of citizenship)
                if "P27" in entity.claims:
                    countries = []
                    for claim in entity.claims["P27"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            value = datavalue.get("value", {})
                            countries.append(value.get("id", ""))
                            entity_ids_to_resolve.add(value.get("id", ""))
                    if countries:
                        properties["country_of_citizenship"] = countries[0]

                # 提取 P276 (location)
                if "P276" in entity.claims:
                    locations = []
                    for claim in entity.claims["P276"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            value = datavalue.get("value", {})
                            locations.append(value.get("id", ""))
                            entity_ids_to_resolve.add(value.get("id", ""))
                    if locations:
                        properties["location"] = locations[0]

                # 提取 P281 (country)
                if "P281" in entity.claims:
                    countries = []
                    for claim in entity.claims["P281"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "string":
                            countries.append(datavalue.get("value", ""))
                    if countries:
                        properties["postal_code"] = countries[0]

                # 提取 P37 (official language)
                if "P37" in entity.claims:
                    languages = []
                    for claim in entity.claims["P37"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            value = datavalue.get("value", {})
                            languages.append(value.get("id", ""))
                            entity_ids_to_resolve.add(value.get("id", ""))
                    if languages:
                        properties["official_language"] = languages[0]

                # 提取 P1705 (spoken text language)
                if "P1705" in entity.claims:
                    languages = []
                    for claim in entity.claims["P1705"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            value = datavalue.get("value", {})
                            languages.append(value.get("id", ""))
                            entity_ids_to_resolve.add(value.get("id", ""))
                    if languages:
                        properties["language_of_work"] = languages[0]

                # 提取 P571 (inception)
                if "P571" in entity.claims:
                    dates = []
                    for claim in entity.claims["P571"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "time":
                            dates.append(datavalue.get("value", ""))
                    if dates:
                        properties["inception"] = dates[0]

                # 提取 P159 (headquarters)
                if "P159" in entity.claims:
                    headquarters = []
                    for claim in entity.claims["P159"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "wikibase-entityid":
                            value = datavalue.get("value", {})
                            headquarters.append(value.get("id", ""))
                            entity_ids_to_resolve.add(value.get("id", ""))
                    if headquarters:
                        properties["headquarters"] = headquarters[0]

                # 提取 P1128 (employees)
                if "P1128" in entity.claims:
                    counts = []
                    for claim in entity.claims["P1128"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "quantity":
                            counts.append(datavalue.get("value", ""))
                    if counts:
                        properties["employees"] = counts[0]

                # 提取 P2132 (revenue)
                if "P2132" in entity.claims:
                    revenues = []
                    for claim in entity.claims["P2132"]:
                        mainsnak = claim.get("mainsnak", {})
                        datavalue = mainsnak.get("datavalue", {})
                        if datavalue.get("type") == "quantity":
                            revenues.append(str(datavalue.get("value", "")))
                    if revenues:
                        properties["revenue"] = revenues[0]

            # 批量解析实体 ID 为标签
            if entity_ids_to_resolve:
                id_to_label = await self._resolve_entity_ids_to_labels(list(entity_ids_to_resolve))

                # 将 ID 替换为标签
                if "instance_of" in properties and isinstance(properties["instance_of"], list):
                    properties["instance_of"] = [id_to_label.get(id, id) for id in properties["instance_of"]]

                if "place_of_birth" in properties:
                    properties["place_of_birth"] = id_to_label.get(properties["place_of_birth"], properties["place_of_birth"])

                if "country_of_citizenship" in properties:
                    properties["country_of_citizenship"] = id_to_label.get(properties["country_of_citizenship"], properties["country_of_citizenship"])

                if "location" in properties:
                    properties["location"] = id_to_label.get(properties["location"], properties["location"])

                if "official_language" in properties:
                    properties["official_language"] = id_to_label.get(properties["official_language"], properties["official_language"])

                if "language_of_work" in properties:
                    properties["language_of_work"] = id_to_label.get(properties["language_of_work"], properties["language_of_work"])

                if "headquarters" in properties:
                    properties["headquarters"] = id_to_label.get(properties["headquarters"], properties["headquarters"])

            # 只有当有实际属性时才返回
            if len(properties) > 1:  # 除了 label 至少还有一个
                return properties
            return {}

        except Exception as e:
            print(f"      ⚠️ 从 Wikidata 获取属性失败: {e}")
            return {}

    async def _resolve_entity_ids_to_labels(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        批量将 Wikidata 实体 ID 解析为标签

        Args:
            entity_ids: 实体 ID 列表

        Returns:
            ID 到标签的映射
        """
        if not entity_ids:
            return {}

        id_to_label = {}

        # 由于 Wikidata API 一次只能获取有限数量的实体，我们分批处理
        batch_size = 50
        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i:i + batch_size]

            try:
                params = {
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels",
                    "format": "json",
                    "languages": "en"
                }

                proxy = getattr(self, '_proxy', None)
                if not proxy and hasattr(self, 'wikidata'):
                    proxy = getattr(self.wikidata, '_proxy', None)

                session = await self.wikidata._get_session()
                async with session.get(self.wikidata.BASE_URL, params=params, proxy=proxy) as response:
                    if response.status == 200:
                        data = await response.json()
                        entities = data.get("entities", {})
                        for entity_id, entity_data in entities.items():
                            label = entity_data.get("labels", {}).get("en", {}).get("value", "")
                            if label:
                                id_to_label[entity_id] = label
            except Exception as e:
                print(f"      ⚠️ 批量解析实体标签失败: {e}")

        return id_to_label

    async def _select_neighbors_by_properties(
        self,
        current_entity: str,
        neighbors: List[Dict[str, Any]],
        current_properties: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        用 LLM 基于当前实体 properties 在候选邻居中做预筛选。
        目标：更贴合 properties 线索，同时偏向“鲜为人知”(wikidata_frequency 低)。
        """
        if not neighbors:
            return []
        if not current_properties:
            return neighbors[:top_k]

        # 只把 properties 的前若干条摘要给模型，避免 token 爆炸
        prop_items = list(current_properties.items())
        prop_items = prop_items[:12]
        props_text = "\n".join([f"- {k}: {str(v)[:160]}" for k, v in prop_items])

        # 构造候选邻居列表（限制数量避免超长）
        candidates = neighbors[:30]
        cand_lines = []
        for i, n in enumerate(candidates, 1):
            label = n.get("entity", "")
            desc = (n.get("description", "") or "")[:120]
            evidence = (n.get("evidence", "") or "")[:160]
            freq = n.get("wikidata_frequency", 0)
            rel = n.get("type", "")
            cand_lines.append(f"{i}. {label} | rel={rel} | freq={freq} | {desc} | evidence={evidence}")

        prompt = f"""You are selecting the NEXT-hop entities for a reasoning evidence chain.

Current entity: {current_entity}

Current entity properties (hints):
{props_text}

Candidate neighbors (choose up to {min(top_k, 10)}):
{chr(10).join(cand_lines)}

Selection rules:
- Prefer candidates that are strongly suggested by the properties/hints.
- Prefer more obscure/less-known entities when ties (lower freq is more obscure).
- Avoid generic/abstract concepts.
- Choose a diverse set (do not pick near-duplicates).

Return JSON only:
{{"indices": [1,2,3]}}
"""
        try:
            resp = await self.llm.generate(prompt)
            clean = resp.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```\w*\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean)
            data = json.loads(clean)
            indices = data.get("indices", [])
            if not isinstance(indices, list):
                return neighbors[:top_k]
            picked = []
            seen = set()
            for idx in indices:
                if not isinstance(idx, int):
                    continue
                if idx < 1 or idx > len(candidates):
                    continue
                n = candidates[idx - 1]
                name = (n.get("entity") or "").strip()
                if not name:
                    continue
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                picked.append(n)
                if len(picked) >= top_k:
                    break
            return picked or neighbors[:top_k]
        except Exception:
            return neighbors[:top_k]

    async def _validate_relations(self, seed_entity: str, neighbors: List[Dict[str, Any]], seed_description: str = "") -> List[NLIRelation]:
        if not neighbors:
            return []

        print(f"         🔍 开始批量NLI验证: {len(neighbors)} 个实体对")

        relations = await self.nli_validator.validate_relations_batch(
            seed_entity=seed_entity, neighbors=neighbors, seed_description=seed_description
        )

        return relations

    async def _assess_familiarity_with_llm(self, edges: List[NLIRelation]) -> List[NLIRelation]:
        """
        用 LLM 评估实体的熟悉程度（鲜为人知的优先）。
        返回更新后的 edges 列表，包含 llm_familiarity_score。
        """
        if not edges:
            return []

        # 构造 prompt
        entities = [e.target_entity for e in edges]

        prompt = f"""You are a knowledge graph expert. Assess how familiar/common each entity is to the general public.

Entities to assess:
{chr(10).join([f"{i+1}. {e}" for i, e in enumerate(entities)])}

For each entity, rate its familiarity on a scale of 0 to 1:
- 0.0 = extremely obscure/uncommon (e.g., a small village, niche technical term)
- 0.5 = moderately known (e.g., a well-known company, city, or historical event)
- 1.0 = extremely common/famous (e.g., a world-famous person, major country, common concept)

IMPORTANT SELECTION GUIDELINES:
- We will REJECT entities with score >= 0.4 (too familiar), so please score conservatively
- PREFER entities with score < 0.4: specific companies, lesser-known cities, technical terms, niche products, specific events
- AVOID scoring >= 0.4 for: major countries (USA, China, France, etc.), continents (Asia, Europe), world-famous landmarks (Eiffel Tower, Great Wall), globally famous people (Einstein), common concepts
- ALWAYS assign score 1.0 to encyclopedia/reference-type entities: Wikipedia, Britannica, encyclopedias, dictionaries, atlases, thesauruses, general reference works, "list of X" type entities, and any entity whose primary purpose is to CONTAIN information rather than to be a specific information target (e.g., "Encyclopedia", "Wikipedia", "Oxford English Dictionary", "World Atlas")
- When scoring, give lower scores to entities that are more specific and harder for average people to recognize

Return JSON only:
{{"familiarity_scores": [{{"entity": "entity_name", "score": 0.5}}, ...]}}

Notes:
- Focus on how familiar the entity is to an average educated person
- Consider the entity's context (e.g., "Apple" the company is common)
- Give lower scores to entities that are more specific rather than generic
- Target score < 0.4 for the entities to pass our filter
- If an entity is a reference/knowledge repository (encyclopedia, dictionary, atlas, etc.), ALWAYS score it 1.0 regardless of how well-known it is
"""

        try:
            resp = await self.llm.generate(prompt)
            clean = resp.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```\w*\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean)

            data = json.loads(clean)
            scores_map = {}
            for item in data.get("familiarity_scores", []):
                entity = item.get("entity", "")
                score = item.get("score", 0.5)
                if entity:
                    scores_map[entity] = score

            # 打印调试信息：每个实体的评分
            print(f"         📊 LLM 熟悉度评分结果:")
            for entity, score in scores_map.items():
                print(f"            - {entity}: {score:.2f}")

            # 更新 edges
            for edge in edges:
                if edge.target_entity in scores_map:
                    edge.llm_familiarity_score = scores_map[edge.target_entity]
                else:
                    edge.llm_familiarity_score = 0.5  # 默认值

            return edges

        except Exception as e:
            print(f"         ⚠️ LLM 熟悉度评估失败: {e}")
            # 返回默认分数
            for edge in edges:
                edge.llm_familiarity_score = 0.5
            return edges

    async def _depth_first_expansion(self, seed_entity: str, valid_edges: List[NLIRelation], num_chains: int, existing_entities: Optional[Set[str]] = None, image_attribute: str = "", image_attribute_value: str = "", max_depth: int = 5, stage2_nodes: List[Dict] = None) -> List[EvidenceChain]:
        existing_entities = existing_entities or {seed_entity}

        # 构建实体->properties的映射
        entity_properties_map = {}
        if stage2_nodes:
            for node in stage2_nodes:
                entity = node.get("entity", "")
                props = node.get("properties", {})
                if entity and props:
                    entity_properties_map[entity] = props

        # 使用递归实现深度优先扩展（从多个第一层邻居并行出发）
        all_chains = []

        seed_node = EvidenceNode(
            entity=seed_entity,
            entity_type="seed",
            description="Seed entity",
            depth=0,
            source_url="",
            image_attribute=image_attribute,
            image_attribute_value=image_attribute_value or seed_entity,
            properties=entity_properties_map.get(seed_entity, {})
        )

        # 检查种子实体的properties是否为空
        if seed_entity not in entity_properties_map or not entity_properties_map[seed_entity]:
            print(f"      ⚠️ 种子实体 '{seed_entity}' 没有properties信息，尝试从 Wikidata 获取...")
            # 从 Wikidata 获取属性作为 fallback
            try:
                wiki_props = await self._fetch_entity_properties_from_wikidata(seed_entity)
                if wiki_props:
                    seed_node.properties = wiki_props
                    entity_properties_map[seed_entity] = wiki_props
                    print(f"      ✅ 种子实体从 Wikidata 获取到 properties")
                else:
                    print(f"      ⚠️ 种子实体无法获取有效 properties，跳过")
                    return []
            except Exception as e:
                print(f"      ⚠️ 种子实体获取 properties 失败: {e}")
                return []

        # 按鲜为人知程度排序（LLM 评估的熟悉度越低越鲜为人知），同时参考置信度
        # 优先选择 LLM 认为越陌生的实体
        sorted_edges = sorted(valid_edges, key=lambda e: (
            e.llm_familiarity_score if hasattr(e, 'llm_familiarity_score') else 0.5,
            -e.confidence
        ))

        print(f"      📊 有效边数量: {len(sorted_edges)}，需要链数: {num_chains}，最大深度: {max_depth}")

        # 递归扩展函数
        async def expand_recursive(
            current_entity: str,
            current_nodes: List[EvidenceNode],
            current_edges: List[EvidenceEdge],
            current_depth: int,
            visited: Set[str]
        ):
            nonlocal all_chains

            # 达到目标深度，生成链
            if current_depth >= max_depth:
                if len(current_nodes) > 1:  # 至少要有一跳
                    chain = EvidenceChain(
                        seed_entity=seed_entity,
                        nodes=current_nodes[:],
                        edges=current_edges[:],
                        chain_depth=current_depth,
                        diversity_score=0.0,
                        uniqueness_score=0.0,
                        image_attribute=image_attribute,
                        image_attribute_value=image_attribute_value or seed_entity
                    )
                    chain.diversity_score = self.diversity_evaluator.calculate_semantic_diversity(
                        chain.entities, [e.relation_type for e in chain.edges]
                    )
                    all_chains.append(chain)
                return

            # 如果已经收集了足够的链，提前终止
            if len(all_chains) >= num_chains * 3:  # 多收集一些，后面筛选
                return

            # 收集当前节点的邻居
            try:
                neighbors = await self._collect_neighbors(current_entity, visited)
            except Exception as e:
                print(f"      ⚠️ 收集邻居失败: {e}")
                return

            if not neighbors:
                print(f"      ⚠️ 节点 '{current_entity}' 没有邻居，尝试其他分支...")
                return

            # properties 驱动的邻居预筛选：用当前节点的 properties 让 LLM 先挑更可能的下一跳
            current_props = self._extract_properties_from_stage2(stage2_nodes, current_entity)
            if current_props:
                neighbors = await self._select_neighbors_by_properties(
                    current_entity=current_entity,
                    neighbors=neighbors,
                    current_properties=current_props,
                    top_k=10
                )
                if not neighbors:
                    print(f"      ⚠️ 节点 '{current_entity}' properties 预筛选后无邻居，放弃此分支")
                    return

            # 检查当前节点的 properties 是否为空
            current_props = entity_properties_map.get(current_entity, {})
            if not current_props:
                print(f"      ⚠️ 节点 '{current_entity}' 没有properties信息，尝试从 Wikidata 补充...")
                try:
                    # 尝试从 Wikidata 获取属性
                    wiki_props = await self._fetch_entity_properties_from_wikidata(current_entity)
                    if wiki_props:
                        current_props = wiki_props
                        entity_properties_map[current_entity] = wiki_props
                        print(f"      ✅ 节点 '{current_entity}' 从 Wikidata 补充到 properties")
                    else:
                        print(f"      ⚠️ 节点 '{current_entity}' 无法从 Wikidata 获取 properties，放弃此分支")
                        return
                except Exception as e:
                    print(f"      ⚠️ 节点 '{current_entity}' 获取 properties 失败: {e}，放弃此分支")
                    return

            # NLI验证
            validated = await self._validate_relations(current_entity, neighbors)
            valid_edges = [e for e in validated if e.is_valid]

            if not valid_edges:
                return

            # 用 LLM 评估实体的熟悉程度（鲜为人知的优先）
            print(f"         🧠 用 LLM 评估 {len(valid_edges)} 个实体的熟悉程度...")
            valid_edges = await self._assess_familiarity_with_llm(valid_edges)

            # 过滤掉 LLM 熟悉度评分 >= 0.4 的实体（太熟悉的不要）
            original_count = len(valid_edges)
            valid_edges = [e for e in valid_edges if e.llm_familiarity_score < 0.4]
            filtered_count = original_count - len(valid_edges)
            if filtered_count > 0:
                print(f"         🔽 过滤掉 {filtered_count} 个过于熟悉的实体 (llm_familiarity_score >= 0.4)")

            # 记录被过滤的实体作为备选（用于回溯）
            filtered_edges = [e for e in validated if e.is_valid and e.llm_familiarity_score >= 0.4]
            if filtered_edges:
                print(f"         📋 保留 {len(filtered_edges)} 个备选实体用于回溯")

            if not valid_edges:
                # 尝试使用被过滤的备选实体
                if filtered_edges:
                    print(f"         ⚠️ 所有邻居都被过滤，尝试使用备选实体...")
                    valid_edges = filtered_edges
                else:
                    return

            # 按鲜为人知程度排序（LLM 评估的熟悉度越低越鲜为人知），同时参考置信度
            # 优先选择 LLM 认为越陌生的实体
            valid_edges = sorted(valid_edges, key=lambda e: (
                e.llm_familiarity_score if hasattr(e, 'llm_familiarity_score') else 0.5,
                -e.confidence
            ))
            # 每次扩展的分支数至少要满足生成链的数量需求
            branches = max(num_chains, min(3, len(valid_edges)))

            # 过滤掉已经被缓存选过的实体
            available_edges = [e for e in valid_edges if e.target_entity not in self._chosen_entities]
            if not available_edges:
                # 所有候选实体都被选过，返回上一层（回溯）
                print(f"         ⬆️ 所有候选实体都已在缓存中，放弃此分支，回溯...")
                return

            for edge in available_edges[:branches]:
                # 创建新节点和边
                new_node = EvidenceNode(
                    entity=edge.target_entity,
                    entity_type=edge.relation_type,
                    description=edge.evidence,
                    depth=current_depth + 1,
                    source_url="",
                    image_attribute=image_attribute,
                    image_attribute_value=image_attribute_value or seed_entity,
                    properties=entity_properties_map.get(edge.target_entity, {})
                )

                # 检查新节点的 properties 是否为空，如果为空则尝试补充
                if not new_node.properties:
                    print(f"      ⚠️ 节点 '{new_node.entity}' 没有 properties 信息，尝试从 Wikidata 补充...")
                    try:
                        # 尝试从 Wikidata 获取属性
                        wiki_props = await self._fetch_entity_properties_from_wikidata(new_node.entity)
                        if wiki_props:
                            new_node.properties = wiki_props
                            entity_properties_map[new_node.entity] = wiki_props
                            print(f"      ✅ 节点 '{new_node.entity}' 从 Wikidata 补充到 properties")
                        else:
                            print(f"      ⚠️ 节点 '{new_node.entity}' 无法从 Wikidata 获取 properties，跳过此分支")
                            continue
                    except Exception as e:
                        print(f"      ⚠️ 节点 '{new_node.entity}' 获取 properties 失败: {e}，跳过此分支")
                        continue

                new_edge = EvidenceEdge(
                    source=current_entity,
                    target=edge.target_entity,
                    relation_type=edge.relation_type,
                    evidence=edge.evidence,
                    confidence=edge.confidence,
                    is_valid=True,
                    wikidata_relation_type=getattr(edge, 'wikidata_relation_type', ''),
                    wikidata_property_id=getattr(edge, 'wikidata_property_id', ''),
                    wikidata_frequency=getattr(edge, 'wikidata_frequency', 0),
                    evidence_sources=getattr(edge, 'evidence_sources', [])
                )

                # 将该实体加入全局缓存（只有通过所有检查的实体才加入）
                self._chosen_entities.add(edge.target_entity)

                # 递归扩展
                new_visited = visited | {edge.target_entity}
                await expand_recursive(
                    edge.target_entity,
                    current_nodes + [new_node],
                    current_edges + [new_edge],
                    current_depth + 1,
                    new_visited
                )

        # 从每个第一层邻居开始扩展（跳过已在缓存中的实体）
        # 先从已排序的列表中过滤掉缓存中的实体
        first_layer_candidates = [e for e in sorted_edges if e.target_entity not in self._chosen_entities]
        if not first_layer_candidates:
            print(f"      ⚠️ 所有第一层候选实体都已在缓存中，无法扩展，回溯...")
            return []
        for idx, edge in enumerate(first_layer_candidates[:min(5, len(first_layer_candidates))], 1):
            if len(all_chains) >= num_chains:
                break

            print(f"      🔗 [{idx}/{min(5, len(first_layer_candidates))}] 从 '{edge.target_entity}' 开始扩展 (置信度: {edge.confidence:.2f}, 熟悉度: {edge.llm_familiarity_score:.2f})")

            target_node = EvidenceNode(
                entity=edge.target_entity,
                entity_type=edge.relation_type,
                description=edge.evidence,
                depth=1,
                source_url="",
                image_attribute=image_attribute,
                image_attribute_value=image_attribute_value or seed_entity,
                properties=entity_properties_map.get(edge.target_entity, {})
            )

            # 检查第一层节点的 properties 是否为空，如果为空则尝试补充
            if not target_node.properties:
                print(f"      ⚠️ 第一层节点 '{target_node.entity}' 没有 properties 信息，尝试从 Wikidata 补充...")
                try:
                    # 尝试从 Wikidata 获取属性
                    wiki_props = await self._fetch_entity_properties_from_wikidata(target_node.entity)
                    if wiki_props:
                        target_node.properties = wiki_props
                        entity_properties_map[target_node.entity] = wiki_props
                        print(f"      ✅ 第一层节点 '{target_node.entity}' 从 Wikidata 补充到 properties")
                    else:
                        print(f"      ⚠️ 第一层节点 '{target_node.entity}' 无法从 Wikidata 获取 properties，跳过此分支")
                        continue
                except Exception as e:
                    print(f"      ⚠️ 第一层节点 '{target_node.entity}' 获取 properties 失败: {e}，跳过此分支")
                    continue

            evidence_edge = EvidenceEdge(
                source=seed_entity,
                target=edge.target_entity,
                relation_type=edge.relation_type,
                evidence=edge.evidence,
                confidence=edge.confidence,
                is_valid=True,
                wikidata_relation_type=getattr(edge, 'wikidata_relation_type', ''),
                wikidata_property_id=getattr(edge, 'wikidata_property_id', ''),
                wikidata_frequency=getattr(edge, 'wikidata_frequency', 0),
                evidence_sources=getattr(edge, 'evidence_sources', [])
            )

            # 将该实体加入全局缓存
            self._chosen_entities.add(edge.target_entity)

            # 开始递归扩展
            await expand_recursive(
                edge.target_entity,
                [seed_node, target_node],
                [evidence_edge],
                1,
                existing_entities | {edge.target_entity}
            )

        # 按多样性评分排序，选取top N
        all_chains.sort(key=lambda c: c.diversity_score, reverse=True)
        final_chains = all_chains[:num_chains]

        # 确保链深度正确
        for chain in final_chains:
            chain.chain_depth = len(chain.nodes) - 1

        # Step 4: 使用 LLM 优化链（合并同义节点并重新延伸）
        if final_chains:
            print(f"   🧠 Step 4: LLM 优化证据链（合并同义节点）...")
            final_chains = await self._optimize_chains_with_llm(
                final_chains,
                seed_entity,
                image_attribute,
                image_attribute_value,
                stage2_nodes
            )
            print(f"   ✅ LLM 优化完成，最终生成 {len(final_chains)} 条证据链")

        return final_chains

    async def _optimize_chains_with_llm(
        self,
        chains: List[EvidenceChain],
        seed_entity: str,
        image_attribute: str = "",
        image_attribute_value: str = "",
        stage2_nodes: List[Dict] = None
    ) -> List[EvidenceChain]:
        """
        使用 LLM 优化证据链：
        1. 检测链中是否有语义重复的节点（同义词/相似实体）
        2. 如果有，合并这些节点
        3. 合并后重新延伸链到规定的跳数
        """
        entity_properties_map = {}
        if stage2_nodes:
            for node in stage2_nodes:
                entity = node.get("entity", "")
                props = node.get("properties", {})
                if entity and props:
                    entity_properties_map[entity] = props

        optimized_chains = []

        for idx, chain in enumerate(chains):
            # 提取链中所有实体
            entities_in_chain = [node.entity for node in chain.nodes]

            # 构建英文 LLM prompt
            prompt = f"""You are a knowledge graph reasoning expert. Analyze if there are semantically duplicate nodes in the following evidence chain (synonyms, similar entities, parent-child relationships, etc.).

Seed entity: {seed_entity}
Current evidence chain nodes: {entities_in_chain}

Determine if there are duplicate nodes in the chain. If yes, provide the node pairs that need to be merged and the suggested merged entity name.

Output format (JSON):
{{
    "has_duplicates": true/false,
    "duplicate_groups": [
        {{"nodes": ["Node A", "Node B"], "merged_name": "Merged Name"}}
    ]
}}

Notes:
1. If no duplicates, return "has_duplicates": false
2. The merged entity should be a more general or specific concept
3. Only consider duplicate relationships between directly connected nodes"""

            try:
                # 使用 llm.generate 方法
                response = await self.llm.generate(
                    prompt=prompt,
                    temperature=0.0
                )
                result_text = response

                # 解析 JSON 结果
                import json
                import re
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = {"has_duplicates": False}

                if result.get("has_duplicates", False):
                    duplicate_groups = result.get("duplicate_groups", [])
                    print(f"      🔄 Chain {idx+1}: Found {len(duplicate_groups)} duplicate groups, merging and re-extending...")

                    # 合并重复节点
                    merged_chain = await self._merge_duplicate_nodes(
                        chain,
                        duplicate_groups,
                        entity_properties_map,
                        seed_entity,
                        image_attribute,
                        image_attribute_value
                    )

                    # 如果合并后链深度不足，重新延伸
                    if merged_chain and len(merged_chain.nodes) - 1 < self.max_depth:
                        print(f"      🔄 Chain {idx+1}: Re-extending chain to target depth {self.max_depth}...")
                        extended_chain = await self._re_extend_chain(
                            merged_chain,
                            entity_properties_map,
                            seed_entity,
                            image_attribute,
                            image_attribute_value
                        )
                        if extended_chain:
                            optimized_chains.append(extended_chain)
                        else:
                            optimized_chains.append(merged_chain)
                    elif merged_chain:
                        optimized_chains.append(merged_chain)
                    else:
                        optimized_chains.append(chain)
                else:
                    optimized_chains.append(chain)

            except Exception as e:
                print(f"      ⚠️ Chain {idx+1}: LLM optimization failed: {e}")
                optimized_chains.append(chain)

        return optimized_chains

    async def _merge_duplicate_nodes(
        self,
        chain: EvidenceChain,
        duplicate_groups: List[Dict],
        entity_properties_map: Dict,
        seed_entity: str,
        image_attribute: str,
        image_attribute_value: str
    ) -> EvidenceChain:
        """
        合并证据链中的重复节点，合并后对新建的边进行NLI验证
        """
        # 创建节点和边的副本
        nodes = chain.nodes[:]
        edges = chain.edges[:]

        for group in duplicate_groups:
            nodes_to_merge = group.get("nodes", [])
            merged_name = group.get("merged_name", "")

            if not nodes_to_merge or not merged_name:
                continue

            # 找到这些节点在链中的位置
            indices_to_remove = []
            for node_name in nodes_to_merge:
                for i, node in enumerate(nodes):
                    if node.entity == node_name:
                        indices_to_remove.append(i)
                        break

            if not indices_to_remove:
                continue

            # 按索引降序排序，避免删除时索引变化
            indices_to_remove.sort(reverse=True)

            # 创建合并后的新节点
            merged_node = EvidenceNode(
                entity=merged_name,
                entity_type=nodes[indices_to_remove[0]].entity_type if indices_to_remove else "merged",
                description=f"Merged from: {', '.join(nodes_to_merge)}",
                depth=nodes[indices_to_remove[0]].depth if indices_to_remove else 0,
                source_url="",
                image_attribute=image_attribute,
                image_attribute_value=image_attribute_value or seed_entity,
                properties=entity_properties_map.get(merged_name, {})
            )

            # 移除被合并的节点
            for idx in indices_to_remove:
                if idx < len(nodes):
                    nodes.pop(idx)

            # 在第一个被移除的位置插入合并后的节点
            if indices_to_remove:
                insert_idx = min(indices_to_remove)
                nodes.insert(insert_idx, merged_node)

        # 重建边，并对每条新边进行NLI验证
        new_edges = []
        for i in range(len(nodes) - 1):
            source_entity = nodes[i].entity
            target_entity = nodes[i + 1].entity

            # NLI验证：检查source -> target的语义关系是否成立
            validated = await self._validate_relations(source_entity, [
                {"entity": target_entity, "type": "unknown", "relation_type": "unknown", "description": ""}
            ])

            if validated and validated[0].is_valid:
                nli_rel = validated[0]
                new_edge = EvidenceEdge(
                    source=source_entity,
                    target=target_entity,
                    relation_type=nli_rel.relation_type,
                    evidence=nli_rel.evidence,
                    confidence=nli_rel.confidence,
                    is_valid=nli_rel.is_valid,
                    wikidata_relation_type=getattr(nli_rel, 'wikidata_relation_type', ''),
                    wikidata_property_id=getattr(nli_rel, 'wikidata_property_id', ''),
                    wikidata_frequency=nli_rel.wikidata_frequency,
                    evidence_sources=getattr(nli_rel, 'evidence_sources', [])
                )
            else:
                # NLI验证失败：降级为低置信度边，但仍保留（后续可过滤）
                new_edge = EvidenceEdge(
                    source=source_entity,
                    target=target_entity,
                    relation_type="unknown",
                    evidence="NLI validation failed after node merge",
                    confidence=0.0,
                    is_valid=False,
                    wikidata_frequency=0
                )
            new_edges.append(new_edge)

        # 返回新的链
        return EvidenceChain(
            seed_entity=seed_entity,
            nodes=nodes,
            edges=new_edges,
            chain_depth=len(nodes) - 1,
            diversity_score=chain.diversity_score,
            uniqueness_score=chain.uniqueness_score,
            image_attribute=image_attribute,
            image_attribute_value=image_attribute_value or seed_entity
        )

    async def _re_extend_chain(
        self,
        chain: EvidenceChain,
        entity_properties_map: Dict,
        seed_entity: str,
        image_attribute: str,
        image_attribute_value: str
    ) -> EvidenceChain:
        """
        重新延伸证据链到规定的跳数
        """
        current_entity = chain.nodes[-1].entity if chain.nodes else seed_entity
        visited = {node.entity for node in chain.nodes}
        current_nodes = chain.nodes[:]
        current_edges = chain.edges[:]

        while len(current_nodes) - 1 < self.max_depth:
            # 收集当前最后节点的邻居
            neighbors = await self._collect_neighbors(current_entity, visited, filter_level=0)

            if not neighbors:
                break

            # 预筛选
            current_props = entity_properties_map.get(current_entity, {})
            if current_props:
                neighbors = await self._select_neighbors_by_properties(
                    current_entity=current_entity,
                    neighbors=neighbors,
                    current_properties=current_props,
                    top_k=10
                )

            if not neighbors:
                break

            # NLI 验证
            validated = await self._validate_relations(current_entity, neighbors)
            valid_edges = [e for e in validated if e.is_valid]

            if not valid_edges:
                break

            # 按鲜为人知程度排序
            valid_edges = sorted(valid_edges, key=lambda e: (
                e.wikidata_frequency if hasattr(e, 'wikidata_frequency') and e.wikidata_frequency else 999,
                -e.confidence
            ))

            # 选择最佳下一跳
            best_edge = valid_edges[0]

            # 创建新节点
            new_node = EvidenceNode(
                entity=best_edge.target_entity,
                entity_type=best_edge.relation_type,
                description=best_edge.evidence,
                depth=len(current_nodes),
                source_url="",
                image_attribute=image_attribute,
                image_attribute_value=image_attribute_value or seed_entity,
                properties=entity_properties_map.get(best_edge.target_entity, {})
            )

            # 创建新边
            new_edge = EvidenceEdge(
                source=current_entity,
                target=best_edge.target_entity,
                relation_type=best_edge.relation_type,
                evidence=best_edge.evidence,
                confidence=best_edge.confidence,
                is_valid=True,
                wikidata_relation_type=getattr(best_edge, 'wikidata_relation_type', ''),
                wikidata_property_id=getattr(best_edge, 'wikidata_property_id', ''),
                wikidata_frequency=getattr(best_edge, 'wikidata_frequency', 0),
                evidence_sources=getattr(best_edge, 'evidence_sources', [])
            )

            current_nodes.append(new_node)
            current_edges.append(new_edge)
            current_entity = best_edge.target_entity
            visited.add(current_entity)

        return EvidenceChain(
            seed_entity=seed_entity,
            nodes=current_nodes,
            edges=current_edges,
            chain_depth=len(current_nodes) - 1,
            diversity_score=self.diversity_evaluator.calculate_semantic_diversity(
                [n.entity for n in current_nodes], [e.relation_type for e in current_edges]
            ),
            uniqueness_score=0.0,
            image_attribute=image_attribute,
            image_attribute_value=image_attribute_value or seed_entity
        )


# ============================================================================
# Stage3 处理器
# ============================================================================

class Stage3EvidenceChainBuilder:
    """
    第3阶段：证据链构建器

    负责构建多跳推理证据链
    """

    def __init__(self, llm_client, search_client, config):
        # 优先从 config 获取代理，如果没有则从环境变量获取
        proxy = getattr(config, 'proxy', None)
        if not proxy:
            proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")

        self._builder = EvidenceChainBuilder(
            llm_client=llm_client,
            search_client=search_client,
            max_depth=config.max_chain_depth,
            min_diversity=getattr(config, 'min_diversity_score', 0.5),
            confidence_threshold=getattr(config, 'nli_confidence_threshold', 0.9),
            proxy=proxy,
            llm_model=getattr(config, 'llm_model', 'gemini-3-flash-preview')
        )
        self.config = config

    async def build_chain(self, entity_name: str, num_chains: int = 3, image_attribute: str = "", image_attribute_value: str = "", stage2_nodes: List[Dict] = None) -> List[EvidenceChain]:
        return await self._builder.build_chain(
            seed_entity=entity_name,
            num_chains=num_chains,
            image_attribute=image_attribute,
            image_attribute_value=image_attribute_value,
            stage2_nodes=stage2_nodes
        )

    async def close(self):
        """关闭 Stage3 内部可能持有的会话"""
        if hasattr(self, "_builder") and self._builder and hasattr(self._builder, "close"):
            await self._builder.close()


# ============================================================================
# 独立运行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    async def main():
        parser = argparse.ArgumentParser(description="Stage3: 证据链构建")
        parser.add_argument("--input", type=str, required=True, help="Stage2输出的JSON文件")
        parser.add_argument("--output", type=str, default="./stage3_output.json", help="输出文件")
        parser.add_argument("--chains", type=int, default=1, help="每个实体生成的链数量")
        parser.add_argument("--max-depth", type=int, default=5, help="最大深度")
        parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY", ""), help="API Key")
        parser.add_argument("--base-url", type=str, default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), help="API Base URL")
        parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="模型名称")
        parser.add_argument("--proxy", type=str, default="http://127.0.0.1:7890", help="代理地址")
        args = parser.parse_args()

        with open(args.input, 'r', encoding='utf-8') as f:
            node_info = json.load(f)

        try:
            from .config import SynthesisConfig
            from llm_client import get_friday_client
            from search_client import UnifiedSearchClient
        except ImportError:
            from .config import SynthesisConfig
            from .llm_client import get_friday_client
            from .search_client import UnifiedSearchClient

        config = SynthesisConfig(
            llm_api_key=args.api_key,
            llm_base_url=args.base_url,
            llm_model=args.model,
            max_chain_depth=args.max_depth,
            proxy=args.proxy
        )
        llm_client = get_friday_client(config)
        search_client = UnifiedSearchClient(proxy=args.proxy)

        builder = Stage3EvidenceChainBuilder(llm_client, search_client, config)

        all_chains = []
        for node in node_info:
            entity_name = node.get("entity")
            if not entity_name:
                continue
            print(f"🔗 为实体 {entity_name} 构建证据链...")
            chains = await builder.build_chain(entity_name=entity_name, num_chains=args.chains)
            all_chains.extend(chains)

        print(f"\n✅ Stage3 完成，构建了 {len(all_chains)} 条证据链")

        result = []
        for chain in all_chains:
            result.append(chain.to_dict())

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {args.output}")

    asyncio.run(main())
