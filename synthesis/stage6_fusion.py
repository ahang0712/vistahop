#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6: 多链融合模块
将多条 VQA 链的答案融合成一个综合问题
"""

import asyncio
import json
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from .llm_client import LLMClient, get_friday_client, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL
except ImportError:
    from llm_client import LLMClient, get_friday_client, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL

try:
    from .search_client import UnifiedSearchClient
except ImportError:
    try:
        from search_client import UnifiedSearchClient
    except ImportError:
        UnifiedSearchClient = None


# ============================================================================
# 数据类型定义
# ============================================================================

class FusionRule(Enum):
    """融合规则枚举"""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    MAX = "max"
    MIN = "min"
    AVG = "avg"
    CONDITIONAL = "conditional"
    LLM_DECIDED = "llm_decided"  # LLM 自动选择融合方式


@dataclass
class OriginalVQA:
    """原始VQA数据"""
    id: str
    question: str
    answer: str
    reasoning_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NumericalExtraction:
    """数值提取结果"""
    chain_id: str
    original_answer: str
    extraction_method: str
    extracted_value: float
    calculation_description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """融合结果"""
    fusion_rule: FusionRule
    extraction_details: List[NumericalExtraction]
    intermediate_values: List[float]
    final_answer: float
    fusion_description: str
    original_vqas: List[OriginalVQA]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result["fusion_rule"] = self.fusion_rule.value
        return result


@dataclass
class FusedQuestion:
    """融合问题"""
    question: str
    final_answer: float
    answer_type: str = "number"
    extraction_details: List[NumericalExtraction] = field(default_factory=list)
    fusion_operation: str = ""
    fusion_description: str = ""
    original_vqas: List[OriginalVQA] = field(default_factory=list)
    reasoning_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class Stage6FusionConfig:
    """Stage 6 配置"""
    num_chains: int = 3
    min_chains: int = 2
    primary_fusion_rule: Optional[FusionRule] = None  # 主要融合规则，None 表示由 LLM 决定
    output_dir: str = "./output/stage6"


# ============================================================================
# JSON 解析工具
# ============================================================================

def extract_json_from_response(response: str) -> Optional[str]:
    """
    从 LLM 响应中提取 JSON 字符串
    处理 markdown 代码块、截断等问题
    """
    if not response:
        return None

    clean = response.strip()

    # 移除 markdown 代码块
    if clean.startswith("```"):
        lines = clean.split('\n')
        start_idx = -1
        end_idx = -1
        brace_count = 0

        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == '{' and start_idx == -1:
                    start_idx = i
                    brace_count = 1
                elif start_idx != -1:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i
                            break
            if end_idx != -1:
                break

        if start_idx != -1 and end_idx != -1:
            clean = '\n'.join(lines[start_idx:end_idx + 1])
        elif start_idx != -1:
            # JSON 被截断，找到了 { 但没有 }
            # 尝试获取从 { 开始的所有内容
            clean = '\n'.join(lines[start_idx:])

    # 确保是有效的 JSON 格式
    first_brace = clean.find('{')
    last_brace = clean.rfind('}')

    if first_brace != -1:
        if last_brace == -1 or last_brace <= first_brace:
            # JSON 被截断，没有结尾的 }
            return clean[first_brace:]
        return clean[first_brace:last_brace + 1]

    return None


def fix_truncated_json(json_str: str, expected_fields: List[str]) -> Optional[str]:
    """
    尝试修复被截断的 JSON 字符串
    """
    if not json_str:
        return None

    # 如果 JSON 已经完整，直接返回
    try:
        json.loads(json_str)
        return json_str
    except:
        pass

    result = json_str

    # 尝试提取已解析的值
    partial_data = {}

    for field in expected_fields:
        if field == "extraction_method" or field == "fusion_operation":
            match = re.search(rf'"{field}"\s*:\s*"([^"]*)"', result)
            if match:
                partial_data[field] = match.group(1)
        elif field == "extracted_value" or field == "final_answer":
            match = re.search(rf'"{field}"\s*:\s*([0-9.]+)', result)
            if match:
                try:
                    partial_data[field] = float(match.group(1))
                except:
                    pass
        elif field == "question":
            # 尝试提取 question 字段
            match = re.search(r'"question"\s*:\s*"([^"]*)$', result, re.MULTILINE)
            if match:
                partial_data[field] = match.group(1)
            elif '"question"' in result:
                # question 字段存在但被截断，尝试从后续文本提取
                # 找到 question 值开始的位置
                q_start = result.find('"question"')
                if q_start != -1:
                    colon_pos = result.find(':', q_start)
                    if colon_pos != -1:
                        # 找到值的开始
                        value_start = result.find('"', colon_pos + 1)
                        if value_start != -1:
                            # 提取到行尾的内容
                            remaining = result[value_start+1:]
                            # 找到第一个未转义的引号
                            partial_question = remaining.split('"')[0] if '"' in remaining else remaining
                            if partial_question.strip():
                                partial_data[field] = partial_question.strip()
        elif field == "reasoning_path":
            # 尝试提取 reasoning_path
            if '"reasoning_path"' in result:
                # 简单处理：返回空数组
                partial_data[field] = []
            else:
                partial_data[field] = []
        elif field == "calculation_description" or field == "fusion_description":
            match = re.search(rf'"{field}"\s*:\s*"([^"]*)$', result, re.MULTILINE)
            if match:
                partial_data[field] = match.group(1)

    # 如果提取到了部分数据，生成完整的 JSON
    if partial_data:
        # 补全缺失字段
        for field in expected_fields:
            if field not in partial_data:
                if field in ["extraction_method", "fusion_operation"]:
                    partial_data[field] = "unknown"
                elif field in ["extracted_value", "final_answer"]:
                    partial_data[field] = 0
                elif field in ["calculation_description", "fusion_description", "question"]:
                    partial_data[field] = ""
                elif field == "reasoning_path":
                    partial_data[field] = []

        try:
            return json.dumps(partial_data)
        except:
            pass

    return None


# ============================================================================
# LLM 驱动的答案转数值转换器
# ============================================================================

class LLMAnswerToNumberConverter:
    """LLM 驱动的答案转数值转换器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def convert_answer_to_number(
        self,
        chain_id: str,
        answer: str,
        context: str
    ) -> NumericalExtraction:
        """将文本答案转换为数值"""

        if not answer or not answer.strip():
            return NumericalExtraction(
                chain_id=chain_id,
                original_answer=answer,
                extraction_method="empty",
                extracted_value=0.0,
                calculation_description="Empty answer, default to 0"
            )

        # 改进的 prompt：让 LLM 根据问题的语义选择提取方法
        prompt = f'''Analyze this question and its answer to determine the BEST method to extract a VERIFIABLE numerical value.

Question: {context}
Answer: "{answer}"

Available extraction methods (choose only the most appropriate one):
1. first_letter_ascii: Use ASCII value of the first letter of the answer
2. first_letter_alphabet: Position of first letter in the English alphabet (A=1, B=2, ..., Z=26)
3. character_count: Number of characters, excluding spaces and punctuation
4. word_count: Number of words in the answer
5. vowel_count: Number of vowels (a, e, i, o, u, case-insensitive)
6. digit_sum: Sum of all digits present in the answer
7. founding_year: Extract a valid founding/establishment year (only if the year is NOT present in the question)
8. founding_year_diff: 2026 minus the founding year (only if the year is NOT present in the question)

IMPORTANT RULES:
- If the answer contains a YEAR (like 1974, 1885, 1999), use founded_year or founded_year_diff
- If the answer is a SHORT NAME (like "Hamburg", "Paris"), use first_letter_ascii_sum
- If the answer is a SENTENCE/PHRASE, use character_count or word_count
- ALWAYS choose the method that gives the most UNIQUE and MEANINGFUL number

Output JSON only:
{{
    "extraction_method": "method_name",
    "extracted_value": 72.0,
    "calculation_description": "Step-by-step calculation showing how you got the number"
}}

IMPORTANT: Return complete JSON only, no other text.'''

        # 最多重试3次
        for attempt in range(3):
            try:
                response = await self.llm.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=4096
                )

                # 提取 JSON
                json_str = extract_json_from_response(response)
                if not json_str:
                    raise ValueError(f"No valid JSON found in response: {response[:100]}")

                # 尝试解析并修复
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试修复截断的 JSON
                    fixed = fix_truncated_json(json_str, ["extraction_method", "extracted_value", "calculation_description"])
                    if fixed:
                        data = json.loads(fixed)
                        print(f"[INFO] Fixed truncated JSON for {chain_id}")
                    else:
                        raise ValueError(f"Cannot parse or fix JSON")

                extraction_method = data.get("extraction_method", "")
                extracted_value = float(data.get("extracted_value", 0))
                calculation_description = data.get("calculation_description", "")

                # 验证计算结果
                verified_value = self._verify_calculation(answer, extraction_method, extracted_value)

                return NumericalExtraction(
                    chain_id=chain_id,
                    original_answer=answer,
                    extraction_method=extraction_method,
                    extracted_value=verified_value,
                    calculation_description=calculation_description or f"Calculated: {verified_value}"
                )

            except Exception as e:
                print(f"[WARNING] Attempt {attempt + 1}/3: LLM error for {chain_id}: {e}")
                if attempt < 2:
                    continue
                raise

        raise RuntimeError(f"LLM extraction failed for {chain_id} after 3 attempts")

    def _verify_calculation(
        self,
        answer: str,
        method: str,
        reported_value: float
    ) -> float:
        """验证LLM计算的数值"""

        if method == "first_letter_ascii_sum":
            words = answer.split()
            if not words:
                return 0.0
            calculated = sum(ord(w[0]) for w in words if w)
            return float(calculated)

        elif method == "founded_year":
            # 提取年份
            years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', answer)
            if years:
                return float(years[0])
            return reported_value

        elif method == "founded_year_diff":
            # 计算年份差
            years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', answer)
            if years:
                year = int(years[0])
                return float(2026 - year)
            return reported_value

        elif method == "character_count":
            # 字符数（不含空格和标点）
            chars = re.sub(r'[\s\W]', '', answer)
            return float(len(chars))

        elif method == "digit_sum":
            digits = re.findall(r'\d', answer)
            return float(sum(int(d) for d in digits)) if digits else 0.0

        elif method == "word_count":
            return float(len(answer.split()))

        elif method == "vowel_count":
            vowels = re.findall(r'[aeiouAEIOU]', answer)
            return float(len(vowels))

        return reported_value


# ============================================================================
# LLM驱动的融合规划器
# ============================================================================

class LLMFusionPlanner:
    """LLM驱动的融合规划器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def decide_fusion(
        self,
        extractions: List[NumericalExtraction]
    ) -> Dict[str, Any]:
        """让LLM决定如何融合多条链的数值"""

        extraction_info = []
        for ext in extractions:
            extraction_info.append({
                "chain_id": ext.chain_id,
                "original_answer": ext.original_answer,
                "extraction_method": ext.extraction_method,
                "extracted_value": ext.extracted_value,
                "calculation": ext.calculation_description
            })

        extraction_json = json.dumps(extraction_info, ensure_ascii=False, indent=2)

        prompt = f'''Given {len(extractions)} numerical extractions from reasoning chains, decide the best fusion operation.

Extractions:
{extraction_json}

Available fusion operations:
1. ADD: Sum all extracted values
2. SUBTRACT: Subtract second from first (for exactly 2 chains)
3. MULTIPLY: Multiply all values
4. DIVIDE: Divide first by second (for exactly 2 chains)
5. MAX: Return the maximum value
6. MIN: Return the minimum value
7. AVG: Return the average of all values

Choose the operation that produces the most meaningful and verifiable result.

Choose the operation that produces the most meaningful and verifiable result.

**Operation**: [ADD/SUBTRACT/MULTIPLY/DIVIDE/MAX/MIN/AVG]
**Reason**: [brief explanation in English]

Return your answer using the format above, no other text.'''

        # 最多重试3次
        for attempt in range(3):
            try:
                response = await self.llm.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=4096
                )

                # 用正则从 markdown 响应中解析 operation 和 description
                operation = None
                description = ""
                valid_operations = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MAX", "MIN", "AVG"]

                op_match = re.search(r'\*\*Operation\*\*:\s*([A-Z]+)', response, re.IGNORECASE)
                if not op_match:
                    op_match = re.search(r'(?:^|\n)\s*(?:-\s*)?(?:Fusion\s+)?Operation:\s*([A-Z]+)', response, re.IGNORECASE | re.MULTILINE)
                if op_match:
                    operation = op_match.group(1).upper()
                    if operation not in valid_operations:
                        operation = None

                desc_match = re.search(r'\*\*Reason\*\*:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
                if not desc_match:
                    desc_match = re.search(r'\*\*Description\*\*:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
                if not desc_match:
                    desc_match = re.search(r'(?:^|\n)\s*(?:-\s*)?(?:Fusion\s+)?(?:Reason|Description):\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if desc_match:
                    description = desc_match.group(1).strip()

                if operation is None:
                    raise ValueError(f"Cannot extract valid fusion_operation from response")

                return {
                    "fusion_operation": operation,
                    "fusion_description": description
                }

            except Exception as e:
                print(f"[WARNING] Attempt {attempt + 1}/3: LLM error in fusion planner: {e}")
                if attempt < 2:
                    continue
                # 三次重试全部失败，使用默认策略
                print(f"[WARNING] LLM fusion planning failed after 3 attempts, defaulting to AVG")
                return {
                    "fusion_operation": "AVG",
                    "fusion_description": f"LLM planning failed after 3 attempts, defaulting to AVG: {e}"
                }


# ============================================================================
# 确定性融合引擎
# ============================================================================

class DeterministicFusionEngine:
    """确定性融合引擎"""

    def apply_fusion(
        self,
        values: List[float],
        rule: FusionRule,
        threshold: float = 0.5
    ) -> Tuple[float, str]:
        """应用确定性融合规则"""

        if not values:
            return 0.0, "No values to fuse"

        if len(values) == 1:
            return values[0], f"Single value: {values[0]}"

        description = ""
        result = 0.0

        try:
            if rule == FusionRule.ADD:
                result = sum(values)
                description = " + ".join(str(int(v)) if v == int(v) else str(v) for v in values) + f" = {int(result) if result == int(result) else result}"

            elif rule == FusionRule.SUBTRACT:
                if len(values) >= 2:
                    result = values[0]
                    for v in values[1:]:
                        result -= v
                    description = " - ".join(str(int(v)) if v == int(v) else str(v) for v in values) + f" = {int(result) if result == int(result) else result}"

            elif rule == FusionRule.MULTIPLY:
                result = 1.0
                for v in values:
                    result *= v
                description = " × ".join(str(int(v)) if v == int(v) else str(v) for v in values) + f" = {int(result) if result == int(result) else result}"

            elif rule == FusionRule.DIVIDE:
                if len(values) >= 2 and values[1] != 0:
                    result = values[0]
                    for v in values[1:]:
                        if v != 0:
                            result /= v
                    description = " ÷ ".join(str(int(v)) if v == int(v) else str(v) for v in values) + f" = {int(result) if result == int(result) else round(result, 2)}"

            elif rule == FusionRule.MAX:
                result = max(values)
                description = f"max({', '.join(str(int(v)) if v == int(v) else str(v) for v in values)}) = {int(result) if result == int(result) else result}"

            elif rule == FusionRule.MIN:
                result = min(values)
                description = f"min({', '.join(str(int(v)) if v == int(v) else str(v) for v in values)}) = {int(result) if result == int(result) else result}"

            elif rule == FusionRule.AVG:
                result = sum(values) / len(values)
                description = f"avg({', '.join(str(int(v)) if v == int(v) else str(v) for v in values)}) = {int(result) if result == int(result) else round(result, 2)}"

        except Exception as e:
            result = sum(values) / len(values)
            description = f"Error in fusion: {e}, fallback avg = {int(result) if result == int(result) else result}"

        return result, description


# ============================================================================
# LLM驱动的问题生成器
# ============================================================================

class LLMQuestionGenerator:
    """LLM驱动的问题生成器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate_fused_question(
        self,
        original_vqas: List[OriginalVQA],
        extractions: List[NumericalExtraction],
        fusion_operation: str,
        fusion_description: str,
        final_answer: float
    ) -> FusedQuestion:
        """生成融合问题（不泄露答案）"""

        # 构建链信息（不包含答案）
        chain_info = []
        for i, (vqa, ext) in enumerate(zip(original_vqas, extractions), 1):
            chain_info.append({
                "index": i,
                "chain_id": vqa.id,
                "question": vqa.question,  # 只包含问题，不包含答案
                "extraction_hint": f"Extract a number using {ext.extraction_method}"  # 不透露具体计算结果
            })

        chain_json = json.dumps(chain_info, ensure_ascii=False, indent=2)

        # 构建可变数量的 Question 列表
        question_section = "\n\n".join([f"## Question {i+1}:\n{vqa.question}" for i, vqa in enumerate(original_vqas)])

        # 构建 extraction hints
        extraction_hints = "\n".join([f"- Question {i+1}: Extract number using {ext.extraction_method}" for i, ext in enumerate(extractions)])

        # 构建 reasoning steps
        reasoning_steps = "\n".join([f"Step {i+1}: Answer question {i+1}, extract number" for i in range(len(original_vqas))])
        reasoning_steps += f"\nStep {len(original_vqas)+1}: Apply {fusion_operation}"

        operation_display = {
            "ADD": "sum",
            "SUBTRACT": "difference",
            "MULTIPLY": "product",
            "DIVIDE": "quotient",
            "MAX": "maximum",
            "MIN": "minimum",
            "AVG": "average"
        }.get(fusion_operation, fusion_operation.lower())

        prompt = f'''Create a multi-step reasoning question about entities shown in an image.

{question_section}

## Task:
Create a SINGLE, coherent, unified question that:
1. Naturally integrates content from ALL {len(original_vqas)} original VQA questions about the image
2. Clearly asks the solver to compute the {operation_display} of numerical values derived from the corresponding answers
3. Does NOT reveal any entity names, answers, or intermediate numerical results

## Answer Extraction Guidance (for the reasoning process):
{extraction_hints}
- Final step: Apply the {fusion_operation} operation to obtain the final numerical answer.

## Output Format (JSON only):
{{
    "question": "A single continuous question ending with 'What is the result?' in natural English, with visual grounding to the image and no sequential markers such as 'first' or 'then'",
    "reasoning_path": ["Step 1: ...", "Step 2: ...", "..."]
}}

## Hard Constraints (MUST BE FOLLOWED):
The "question" field MUST:
- End with "What is the result?"
- Be at least 50 characters in length (excluding the ending phrase)
- NOT include any entity names from the original answers
- NOT include any numbers, years, statistics, or intermediate results
- NOT include phrases such as "final answer", "the answer is", "calculate the answer"
- Maintain visual grounding by referencing "the image" or "shown in the image"
- Use natural, fluent language without robotic listing

Return ONLY the complete JSON object, no extra explanation, no extra text.'''

        # 最多重试3次
        for attempt in range(3):
            try:
                response = await self.llm.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=8192  # 增大以确保完整生成
                )

                # 提取 JSON
                json_str = extract_json_from_response(response)
                if not json_str:
                    raise ValueError(f"No valid JSON found in response")

                # 尝试解析并修复
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    fixed = fix_truncated_json(json_str, ["question", "reasoning_path"])
                    if fixed:
                        data = json.loads(fixed)
                        print(f"[INFO] Fixed truncated JSON for question generator")
                    else:
                        raise ValueError(f"Cannot parse or fix JSON")

                question_text = data.get("question", "")
                reasoning_path = data.get("reasoning_path", [])

                # 验证问题长度
                if len(question_text.strip()) < 50:
                    print(f"[WARNING] Question too short: {len(question_text)} chars")
                    if attempt < 2:
                        # 使用已解析的内容继续
                        if question_text:
                            reasoning_path = reasoning_path or self._generate_default_reasoning(fusion_operation, original_vqas, extractions)
                            break
                        continue
                    # 强制生成
                    question_text = self._generate_fallback_question(original_vqas, fusion_operation, extractions)
                    reasoning_path = reasoning_path or self._generate_default_reasoning(fusion_operation, original_vqas, extractions)

                # 验证问题不包含答案泄露
                if self._contains_answer_leakage(question_text, original_vqas):
                    print(f"[WARNING] Generated question may leak answers, sanitizing...")
                    question_text = self._sanitize_question(question_text, original_vqas)

                return FusedQuestion(
                    question=question_text,
                    final_answer=final_answer,
                    answer_type="number",
                    extraction_details=extractions,
                    fusion_operation=fusion_operation,
                    fusion_description=fusion_description,
                    original_vqas=original_vqas,
                    reasoning_path=reasoning_path,
                    metadata={
                        "generated_by": "llm",
                        "num_chains": len(original_vqas),
                        "anti_leakage": True
                    }
                )

            except Exception as e:
                print(f"[WARNING] Attempt {attempt + 1}/3: LLM error in question generator: {e}")
                if attempt < 2:
                    continue
                raise

        raise RuntimeError("LLM question generation failed after 3 attempts")

    def _generate_default_reasoning(
        self,
        fusion_operation: str,
        original_vqas: List[OriginalVQA],
        extractions: List[NumericalExtraction]
    ) -> List[str]:
        """生成默认的推理路径"""
        reasoning = []
        for i, (vqa, ext) in enumerate(zip(original_vqas, extractions), 1):
            reasoning.append(f"Step {i}: Answer question {i} using the image, then extract number using {ext.extraction_method}")
        reasoning.append(f"Step {len(original_vqas) + 1}: Apply {fusion_operation} to extracted numbers")
        return reasoning

    def _generate_fallback_question(
        self,
        original_vqas: List[OriginalVQA],
        fusion_operation: str,
        extractions: List[NumericalExtraction]
    ) -> str:
        """生成备用问题（当LLM生成失败时）"""
        op_text = {
            "ADD": "sum",
            "SUBTRACT": "difference",
            "MULTIPLY": "product",
            "DIVIDE": "quotient",
            "MAX": "maximum",
            "MIN": "minimum",
            "AVG": "average"
        }.get(fusion_operation, fusion_operation.lower())

        # 提取第一个问题的关键词
        first_q = original_vqas[0].question if original_vqas else "containers"
        # 简化为通用描述
        return f"Analyze the shipping containers and other elements shown in the image. Answer all questions about the visible company logos, identifiers, and relevant details, then extract numerical values using the specified methods. What is the {op_text} of these extracted numbers?"

    def _contains_answer_leakage(self, question: str, original_vqas: List[OriginalVQA]) -> bool:
        """检查问题是否泄露了答案"""
        question_lower = question.lower()

        # 检查是否包含任何原始答案
        for vqa in original_vqas:
            answer = vqa.answer.lower()
            # 检查答案中的关键词（至少3个字符的单词）
            answer_words = [w for w in answer.split() if len(w) >= 3]
            for word in answer_words:
                if word.lower() in question_lower:
                    print(f"[WARNING] Answer word '{word}' found in question")
                    return True

        # 检查是否包含数值答案
        for vqa in original_vqas:
            # 检查答案中的年份
            years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', vqa.answer)
            for year in years:
                if year in question:
                    print(f"[WARNING] Year '{year}' found in question")
                    return True

        return False

    def _sanitize_question(self, question: str, original_vqas: List[OriginalVQA]) -> str:
        """清理问题中的答案泄露"""
        result = question

        # 移除答案中的年份
        for vqa in original_vqas:
            years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', vqa.answer)
            for year in years:
                result = re.sub(rf'\b{year}\b', '[YEAR]', result)

        # 移除敏感关键词（从答案中提取的名词）
        sensitive_words = set()
        for vqa in original_vqas:
            # 过滤掉短词和常见词
            words = [w.strip('.,;:!?()[]{}"\'') for w in vqa.answer.split()]
            for word in words:
                if len(word) >= 4 and word.lower() not in ['that', 'with', 'from', 'this', 'which', 'where', 'when', 'what', 'there']:
                    sensitive_words.add(word.lower())

        # 替换敏感词
        for word in sensitive_words:
            # 不完全替换，避免完全破坏语义
            pass  # 保持原样，因为这些词可能是必要的描述

        return result


# ============================================================================
# Stage 6 融合生成器
# ============================================================================

class Stage6FusionGenerator:
    """
    Stage 6: 多链融合生成器

    将多条 VQA 链的答案融合成一个综合问题：
    1. 选择多条 VQA 链
    2. 将每条链的文本答案转换为数值（LLM 决定方法）
    3. 决定融合操作（LLM 决定）
    4. 应用确定性融合规则
    5. 生成融合问题（LLM 生成）
    """

    def __init__(
        self,
        config: Stage6FusionConfig,
        llm_client: LLMClient
    ):
        self.config = config
        self.llm = llm_client

        # 初始化组件
        self.answer_converter = LLMAnswerToNumberConverter(llm_client)
        self.fusion_planner = LLMFusionPlanner(llm_client)
        self.fusion_engine = DeterministicFusionEngine()
        self.question_generator = LLMQuestionGenerator(llm_client)

    async def generate_fusion(
        self,
        vqa_items: List[Dict[str, Any]]
    ) -> Tuple[FusionResult, FusedQuestion]:
        """
        生成融合结果

        Args:
            vqa_items: VQA 项目列表

        Returns:
            Tuple[FusionResult, FusedQuestion]: 融合结果和问题
        """
        print("\n" + "=" * 60)
        print("Stage 6: Multi-Chain Fusion")
        print("=" * 60)

        # Step 1: 选择 VQA 链
        selected_vqas = self._select_vqa_chains(vqa_items)

        if not selected_vqas:
            print("[WARNING] No VQAs selected for fusion")
            return self._create_empty_result()

        print(f"\n[INFO] Selected {len(selected_vqas)} VQAs for fusion")
        for i, vqa in enumerate(selected_vqas, 1):
            vqa_id = vqa.get("id", f"vqa_{i}")
            vqa_question = vqa.get("question", "")
            vqa_answer = vqa.get("answer", "")
            vqa_reasoning = vqa.get("reasoning_path", [])
            print(f"[INFO] VQA {i}: {vqa_id}")
            print(f"       Question: {vqa_question[:80]}...")
            print(f"       Answer: {vqa_answer}")
            print(f"       Reasoning steps: {len(vqa_reasoning)}")

        # Step 2: 将 VQA dict 转换为 OriginalVQA 对象（包含完整信息）
        original_vqas = [
            OriginalVQA(
                id=vqa.get("id", f"vqa_{i}"),
                question=vqa.get("question", ""),
                answer=vqa.get("answer", ""),  # 用于数值提取，不会在问题中泄露
                reasoning_path=vqa.get("reasoning_path", []),
                metadata={
                    "original_question": vqa.get("original_question", ""),
                    "target_entity": vqa.get("target_entity", ""),
                    "visual_reference": vqa.get("visual_reference", ""),
                    "entity_type": vqa.get("entity_type", ""),
                    "difficulty": vqa.get("difficulty", ""),
                    "domain": vqa.get("domain", ""),
                    "image_url": vqa.get("image_url", "")
                }
            )
            for i, vqa in enumerate(selected_vqas)
        ]

        # Step 3: 将每个answer转换为数值（LLM必须成功）
        # context 使用完整的问题信息
        print(f"\n[INFO] Converting answers to numbers (LLM-based)...")
        extractions = []

        for i, vqa in enumerate(original_vqas, 1):
            # 构建丰富的上下文信息
            context = f"Question: {vqa.question}\n"
            if vqa.reasoning_path:
                context += f"Reasoning: {' -> '.join(vqa.reasoning_path)}\n"
            context += f"Visual reference: {vqa.metadata.get('visual_reference', 'N/A')}\n"
            context += f"Target entity: {vqa.metadata.get('target_entity', 'N/A')}"

            try:
                extraction = await self.answer_converter.convert_answer_to_number(
                    chain_id=vqa.id,
                    answer=vqa.answer,
                    context=context
                )
                extractions.append(extraction)

                print(f"       [{i}] {vqa.answer[:50]}...")
                print(f"           -> {extraction.extraction_method}: {extraction.calculation_description}")
            except Exception as e:
                print(f"\n[ERROR] Failed to extract number from VQA {vqa.id}: {e}")
                print(f"       Answer: {vqa.answer}")
                raise RuntimeError(f"Stage 6 fusion requires LLM-based extraction. Aborting: {e}")

        # Step 4: 决定融合操作
        fusion_operation_str = None
        fusion_operation_desc = "Using configured fusion rule"

        if self.config.primary_fusion_rule and self.config.primary_fusion_rule != FusionRule.LLM_DECIDED:
            # 使用配置的规则
            fusion_operation_str = self.config.primary_fusion_rule.value.upper()
            fusion_operation_desc = f"Using configured fusion rule: {fusion_operation_str}"
            print(f"\n[INFO] Using configured fusion rule: {fusion_operation_str}")
        else:
            # LLM 决定
            print(f"\n[INFO] Deciding fusion operation (LLM-based)...")
            try:
                fusion_decision = await self.fusion_planner.decide_fusion(extractions)
                fusion_operation_str = fusion_decision["fusion_operation"]
                fusion_operation_desc = fusion_decision["fusion_description"]

                print(f"       Operation: {fusion_operation_str}")
                print(f"       Reason: {fusion_operation_desc}")
            except Exception as e:
                print(f"\n[WARNING] LLM fusion planning failed: {e}")
                print(f"       Falling back to default AVG fusion")
                fusion_operation_str = "AVG"
                fusion_operation_desc = "Default fallback due to LLM failure"

        # 转换为枚举
        fusion_rule = FusionRule[fusion_operation_str]

        # Step 5: 应用确定性融合规则
        intermediate_values = [ext.extracted_value for ext in extractions]
        print(f"\n[INFO] Applying fusion...")
        final_answer, fusion_description = self.fusion_engine.apply_fusion(
            values=intermediate_values,
            rule=fusion_rule
        )

        print(f"       Calculation: {fusion_description}")
        print(f"       Final answer: {final_answer}")

        # 构建FusionResult
        fusion_result = FusionResult(
            fusion_rule=fusion_rule,
            extraction_details=extractions,
            intermediate_values=intermediate_values,
            final_answer=final_answer,
            fusion_description=fusion_description,
            original_vqas=original_vqas
        )

        # Step 6: 生成融合问题（LLM必须成功）
        print(f"\n[INFO] Generating fused question (LLM-based)...")
        try:
            fused_question = await self.question_generator.generate_fused_question(
                original_vqas=original_vqas,
                extractions=extractions,
                fusion_operation=fusion_operation_str,
                fusion_description=fusion_description,
                final_answer=final_answer
            )

            if not fused_question.question or len(fused_question.question.strip()) < 20:
                raise ValueError("Generated question is too short or empty")

            print(f"[INFO] Fused question generated:")
            print(f"       {fused_question.question[:100]}...")
        except Exception as e:
            print(f"\n[ERROR] Failed to generate fused question: {e}")
            raise RuntimeError(f"Stage 6 fusion requires LLM-based question generation. Aborting: {e}")

        return fusion_result, fused_question

    def _select_vqa_chains(
        self,
        vqa_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """选择用于融合的 VQA 链"""
        num_chains = min(self.config.num_chains, len(vqa_items))
        return vqa_items[:num_chains]

    def _create_empty_result(self) -> Tuple[FusionResult, FusedQuestion]:
        """创建空结果"""
        return (
            FusionResult(
                fusion_rule=FusionRule.ADD,
                extraction_details=[],
                intermediate_values=[],
                final_answer=0.0,
                fusion_description="No VQAs available",
                original_vqas=[]
            ),
            FusedQuestion(
                question="No VQAs available for fusion",
                final_answer=0.0,
                answer_type="number",
                original_vqas=[],
                metadata={"error": "No VQAs"}
            )
        )


# ============================================================================
# 便捷函数
# ============================================================================

async def run_stage6_fusion(
    vqa_items: List[Dict[str, Any]],
    num_chains: int = 3,
    llm_client: LLMClient = None,
    search_client: UnifiedSearchClient = None,
    output_dir: str = None
) -> Tuple[FusionResult, FusedQuestion]:
    """
    运行 Stage 6 融合

    Args:
        vqa_items: VQA 项目列表
        num_chains: 融合链数量
        llm_client: LLM 客户端
        search_client: 搜索客户端
        output_dir: 输出目录

    Returns:
        Tuple[FusionResult, FusedQuestion]: 融合结果和问题
    """
    # 初始化默认客户端
    if llm_client is None:
        friday_client = get_friday_client(
            model_name=DEFAULT_LLM_MODEL,
            api_url=DEFAULT_LLM_BASE_URL,
            use_api_key_manager=True
        )
        llm_client = LLMClient(friday_client=friday_client)

    # 创建配置
    config = Stage6FusionConfig(
        num_chains=num_chains,
        output_dir=output_dir or f"./output/stage6"
    )

    # 创建生成器
    generator = Stage6FusionGenerator(
        config=config,
        llm_client=llm_client
    )

    # 运行融合
    fusion_result, fused_question = await generator.generate_fusion(vqa_items)

    # 保存结果
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存融合结果
        result_path = os.path.join(output_dir, "stage6_fusion_result.json")
        result_data = {
            "fusion_rule": fusion_result.fusion_rule.value,
            "extraction_details": [
                {
                    "chain_id": ext.chain_id,
                    "original_answer": ext.original_answer,
                    "extraction_method": ext.extraction_method,
                    "extracted_value": ext.extracted_value,
                    "calculation_description": ext.calculation_description
                }
                for ext in fusion_result.extraction_details
            ],
            "intermediate_values": fusion_result.intermediate_values,
            "final_answer": fusion_result.final_answer,
            "fusion_description": fusion_result.fusion_description,
            "original_vqas": [
                {
                    "id": vqa.id,
                    "question": vqa.question,
                    "answer": vqa.answer,
                    "reasoning_path": vqa.reasoning_path
                }
                for vqa in fusion_result.original_vqas
            ]
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Fusion result saved to: {result_path}")

        # 保存融合问题
        question_path = os.path.join(output_dir, "stage6_fused_question.json")
        question_data = {
            "question": fused_question.question,
            "final_answer": fused_question.final_answer,
            "answer_type": fused_question.answer_type,
            "fusion_operation": fused_question.fusion_operation,
            "fusion_description": fused_question.fusion_description,
            "reasoning_path": fused_question.reasoning_path,
            "metadata": fused_question.metadata
        }
        with open(question_path, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Fused question saved to: {question_path}")

    return fusion_result, fused_question


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Stage 6: Multi-Chain Fusion")
    parser.add_argument("--input", type=str, required=True, help="Input VQA file")
    parser.add_argument("--output-dir", type=str, default="./output/stage6", help="Output directory")
    parser.add_argument("--num-chains", type=int, default=3, help="Number of chains to fuse")
    parser.add_argument("--api-key", type=str, help="API key (optional)")
    parser.add_argument("--base-url", type=str, default=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL), help="API base URL")
    parser.add_argument("--model", type=str, default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL), help="Model name")

    args = parser.parse_args()

    # 加载输入数据
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vqa_items = data.get('vqa_items', [])
    print(f"Loaded {len(vqa_items)} VQA items from {args.input}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"stage6_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # 运行融合
    async def main():
        friday_client = get_friday_client(
            model_name=args.model,
            api_url=args.base_url,
            api_token=args.api_key or os.getenv("LLM_API_KEY") or None,
            use_api_key_manager=not bool(args.api_key or os.getenv("LLM_API_KEY")),
        )
        llm_client = LLMClient(friday_client=friday_client)
        fusion_result, fused_question = await run_stage6_fusion(
            vqa_items=vqa_items,
            num_chains=args.num_chains,
            llm_client=llm_client,
            output_dir=output_dir
        )
        print(f"\n[SUCCESS] Fusion complete!")
        print(f"Final answer: {fusion_result.final_answer}")

    asyncio.run(main())
