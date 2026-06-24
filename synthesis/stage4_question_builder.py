"""
Stage 4: Question Construction

Question generation, obfuscation, iterative refinement

This module integrates the QuestionBuilder functionality
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# 处理导入路径
import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)  # shujuhecheng/modules
_grandparent_dir = os.path.dirname(_parent_dir)  # shujuhecheng

if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
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

# 尝试导入 stage3_evidence_chain_builder
EvidenceChain = None
EvidenceNode = None
try:
    from .stage3_evidence_chain_builder import EvidenceChain, EvidenceNode
except ImportError:
    try:
        from stage3_evidence_chain_builder import EvidenceChain, EvidenceNode
    except ImportError:
        pass


# ============================================================================
# 数据类定义
# ============================================================================

class QuestionType(Enum):
    """Question types"""
    BRIDGE = "bridge"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass
class SynthesisQuestion:
    """VistaHop generated question"""
    id: str
    question: str
    answer: str
    evidence_chain: Dict[str, Any]
    constraints: List[str]
    reasoning_path: List[str]
    difficulty_score: float
    uniqueness_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "evidence_chain": self.evidence_chain,
            "constraints": self.constraints,
            "reasoning_path": self.reasoning_path,
            "difficulty_score": self.difficulty_score,
            "uniqueness_score": self.uniqueness_score
        }


@dataclass
class Condition:
    """识别的单个条件"""
    index: int
    text: str
    entity_mentioned: Optional[str] = None


@dataclass
class AblationResult:
    """消融实验结果"""
    original_question: str
    original_answer: str
    condition: Condition
    ablated_question: str
    solver_answer: str
    solver_correct: bool
    condition_critical: bool  # True=关键, False=冗余


# ============================================================================
# 核心类
# ============================================================================

class ReverseQuestionGenerator:
    """Reverse question generator (now generates forward questions where answer is the last node)"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def generate(self, chain: EvidenceChain, num_questions: int = 3) -> List[Dict[str, Any]]:
        questions = []

        for i in range(num_questions):
            question = await self._generate_single_question(chain, i)
            if question:
                questions.append(question)

        return questions

    async def _generate_single_question(self, chain: EvidenceChain, question_index: int) -> Optional[Dict[str, Any]]:
        # 获取最后一个节点作为答案
        last_node = chain.nodes[-1] if chain.nodes else None
        if not last_node:
            return None

        answer_entity = last_node.entity

        # 构建正向推理链的描述（从种子实体到目标答案）
        reasoning_chain_desc = ""
        for i, node in enumerate(chain.nodes, 1):
            if i == 1:
                reasoning_chain_desc += f"Level 0: {node.entity} (Start)\n"
            else:
                edge = None
                for e in chain.edges:
                    if e.target == node.entity:
                        edge = e
                        break
                relation = edge.relation_type if edge else "related"
                evidence = edge.evidence if edge and edge.evidence else ""

                # 使用自然语言描述关系，包含证据
                if evidence:
                    reasoning_chain_desc += f"Level {i-1} -> Level {i}: {chain.nodes[i-2].entity} --[{relation}]--> {node.entity}\n   Evidence: {evidence}\n"
                else:
                    reasoning_chain_desc += f"Level {i-1} -> Level {i}: {chain.nodes[i-2].entity} --[{relation}]--> {node.entity}\n"

        # 提取首尾节点的完整信息
        start_node = chain.nodes[0] if chain.nodes else None
        end_node = chain.nodes[-1] if chain.nodes else None

        start_info = ""
        if start_node:
            start_info = f"""Start Node:
- Entity: {start_node.entity}
- Type: {start_node.entity_type}
- Description: {start_node.description}
- Properties: {json.dumps(start_node.properties, ensure_ascii=False) if start_node.properties else "N/A"}
"""

        end_info = ""
        if end_node:
            end_info = f"""End Node:
- Entity: {end_node.entity}
- Type: {end_node.entity_type}
- Description: {end_node.description}
- Properties: {json.dumps(end_node.properties, ensure_ascii=False) if end_node.properties else "N/A"}
"""

        prompt = f"""Based on the following evidence chain, generate a multi-hop reasoning question.

Evidence Chain:
{reasoning_chain_desc}

{start_info}

{end_info}

1. Reverse Entity Order:
- The phrasing of the question should follow the entity order of the reasoning chain in reverse, from the final entity toward the seed entity.

2. Obfuscation & Vagueness:
- Never directly mention the final answer entity or any intermediate entities by name.
- Describe entities indirectly using attributes, categories, or general characteristics.
- Avoid giving away the reasoning path; each step must be subtle.

3. Difficulty & Multi-Step Search:
- The question should require exactly N reasoning steps (where N = number of nodes in the chain).
- Each reasoning step should provide a vague constraint or clue that requires external search or knowledge verification.


4. Clarity of Requirements for Generation:
- Maintain coherence and fluency.
- Keep the entity order reversed but the reasoning chain intact.
- Ensure the question is solvable via logical reasoning over external information, without revealing shortcuts.

Example 1:
Input:
Evidence Chain:
Level 0: Apple Watch (Start)
Level 1 -> Level 2:  Apple Watch --[is_manufactured_by]--> Quanta Computer
Level 2 -> Level 3: Quanta Computer --[is_located_in_country]--> Taiwan
Level 3 -> Level 4: Taiwan --[is_located_in_continent]--> Asia

Start Node:
- Entity: Apple Watch
- Type: seed
- Description: Seed entity

End Node:
- Entity: Asia
- Type: spatial
- Description: Asia is the continent that contains Taiwan.

Output:
A company headquartered on an island in a certain continent with a GDP over 20,000 US dollars, responsible for assembling Apple Watch electronic products, how many aircraft carriers does the most populous country of that continent have?

Example 2:
Evidence Chain:
Level 0: Tesla Model 3 (Start)
Level 1 -> Level 2: Tesla Model 3 --[is_manufactured_by]--> Tesla, Inc.
Level 2 -> Level 3: Tesla, Inc. --[is_headquartered_in_city]--> Palo Alto
Level 3 -> Level 4: Palo Alto --[is_located_in_state]--> California
Level 4 -> Level 5: California --[is_located_in_country]--> United States

Start Node:
- Entity: Tesla Model 3
- Type: seed
- Description: Seed entity

End Node:
- Entity: United States
- Type: spatial
- Description: United States is the country that contains California.

Output:
A company headquartered in a city in a country, responsible for manufacturing Model 3 vehicles, what is the total number of Olympic gold medals won by that country in the most recent Summer Olympics?


IMPORTANT - Anti-Leakage Requirements:
The generated question must be DIFFICULT and require external knowledge search to answer. It should NOT reveal too much information:

1. NEVER directly mention the final answer entity in the question text
2. Make clues VAGUE and INDIRECT - use indirect descriptions rather than specific names
   - BAD: "Which company founded by Elon Musk..."
   - GOOD: "Which aerospace company founded..."
3. Use OBFUSCATED descriptions - describe entities by their attributes rather than names
   - BAD: "SpaceX is headquartered in Hawthorne"
   - GOOD: "A company headquartered in a city..."
4. Avoid giving away the reasoning path - don't make it obvious how to get from A to B
5. The question should be like a puzzle that requires multi-step search and reasoning
6. Use RELATIVE/COMPARATIVE descriptions instead of absolute direct references
   - BAD: "Which company has the most employees...", "The island with the highest population density..."
   - GOOD: "A company that, compared to others in its industry, has a significant workforce..."
   - Use "the largest among...", "the most populous within...", "the highest in..." instead of bare superlatives
   - Use relative positional descriptions: "on one side of...", "in the northern/southern region of..."
   - When describing superlatives, always provide a reference category or comparison group

Question Requirements:
1. The question should require {len(chain.nodes) - 1} steps of reasoning
2. Each reasoning step should provide vague constraint conditions that require search to verify
3. Individual clues should be indirect (but correctly lead to the answer when combined)


Please generate a natural question where the final answer is the last entity in the chain:

```json
{{
    "question": "Generated question",
    "constraints": ["constraint1", "constraint2", ...],
    "reasoning_path": ["reasoning step 1", "reasoning step 2", ...]
}}
```

Return JSON only, no other content."""

        try:
            response = await self.llm.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                import re
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)

            return {
                "question": data.get("question", ""),
                "constraints": data.get("constraints", []),
                "reasoning_path": data.get("reasoning_path", []),
                "chain_depth": chain.chain_depth
            }
        except Exception as e:
            print(f"   ⚠️ Failed to generate question: {e}")
            return None


class QuestionObfuscator:
    """问题混淆器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def obfuscate(self, question: Dict[str, Any]) -> Dict[str, Any]:
        original_question = question.get("question", "")
        constraints = question.get("constraints", [])

        prompt = f"""Please obfuscate the following multi-hop reasoning question.

Original Question: {original_question}

Constraints:
"""
        for i, c in enumerate(constraints, 1):
            prompt += f"{i}. {c}\n"

        prompt += """
Obfuscation Requirements:
1. Replace direct entity names with indirect descriptions.
2. Increase the abstractness while retaining sufficient reasoning clues.

Return JSON:
{
    "obfuscated_question": "Obfuscated question",
    "obfuscated_constraints": ["obfuscated constraint 1", ...]
}

Return JSON only, no other content."""

        try:
            response = await self.llm.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                import re
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)

            return {
                "original_question": original_question,
                "obfuscated_question": data.get("obfuscated_question", ""),
                "obfuscated_constraints": data.get("obfuscated_constraints", []),
                "constraints": constraints
            }
        except Exception as e:
            print(f"   ⚠️ Failed to obfuscate question: {e}")
            return question


class AnswerUniquenessValidator:
    """答案唯一性验证器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def validate(self, question: str, answer: str, evidence_chain: EvidenceChain) -> Tuple[bool, float]:
        all_entities = evidence_chain.entities
        other_entities = [e for e in all_entities if e != answer]

        prompt = f"""Validate whether the answer to the following question is unique.

Question: {question}
Correct Answer: {answer}
Other Possible Answers: {', '.join(other_entities[:5])}

Return JSON:
{{"is_unique": true/false, "uniqueness_score": 0-1}}

Return JSON only."""

        try:
            response = await self.llm.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                import re
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)
            return data.get("is_unique", False), float(data.get("uniqueness_score", 0.5))
        except Exception as e:
            return False, 0.5


class IterativeRefiner:
    """迭代精炼器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.max_iterations = 3

    async def refine(self, question: Dict[str, Any], evidence_chain: EvidenceChain) -> Dict[str, Any]:
        current_question = question.copy()

        for iteration in range(self.max_iterations):
            score = await self._evaluate_quality(current_question)
            current_question["final_score"] = score

            if score >= 0.9:
                break

        return current_question

    async def _evaluate_quality(self, question: Dict[str, Any]) -> float:
        prompt = f"""Evaluate the quality of the following question (0-1):
{question.get('question', '')}

Return JSON:
{{"quality_score": 0-1}}

Return JSON only."""

        try:
            response = await self.llm.generate(prompt)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                import re
                clean_response = re.sub(r'^```\w*\n?', '', clean_response)
                clean_response = re.sub(r'\n?```$', '', clean_response)

            data = json.loads(clean_response)
            return float(data.get("quality_score", 0.5))
        except Exception as e:
            return 0.5


class QuestionBuilder:
    """VistaHop问题构建器"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.reverse_generator = ReverseQuestionGenerator(llm_client)
        self.obfuscator = QuestionObfuscator(llm_client)
        self.uniqueness_validator = AnswerUniquenessValidator(llm_client)
        self.refiner = IterativeRefiner(llm_client)

    async def build_question(self, chain: EvidenceChain, num_questions: int = 3) -> List[SynthesisQuestion]:
        print(f"\n📝 Building questions for evidence chain...")
        print(f"   Seed entity: {chain.seed_entity}")

        # Step 1: Question generation (answer is the last node in the chain)
        raw_questions = await self.reverse_generator.generate(chain, num_questions)

        if not raw_questions:
            return []

        # Step 2-4 已停用（仅注释保留，便于随时恢复）
        # - Step 2: 问题混淆（obfuscation）
        # obfuscated_questions = []
        # for q in raw_questions:
        #     obfuscated = await self.obfuscator.obfuscate(q)
        #     obfuscated_questions.append(obfuscated)
        #
        # - Step 3: 迭代精炼（quality score）
        # refined_questions = []
        # for q in obfuscated_questions:
        #     refined = await self.refiner.refine(q, chain)
        #     refined_questions.append(refined)
        #
        # - Step 4: 答案唯一性验证（uniqueness）
        # validated_questions = []
        # for q in refined_questions:
        #     question_text = q.get('obfuscated_question', q.get('question', ''))
        #
        #     # 获取最后一个节点作为答案
        #     answer = chain.nodes[-1].entity if chain.nodes else chain.seed_entity
        #
        #     is_unique, uniqueness_score = await self.uniqueness_validator.validate(
        #         question_text, answer, chain
        #     )
        #
        #     if is_unique or uniqueness_score > 0.7:
        #         q['uniqueness_score'] = uniqueness_score
        #         validated_questions.append(q)

        # Step 5: 构建最终问题对象（直接使用 raw_questions）
        final_questions = []
        for i, q in enumerate(raw_questions, 1):
            # 使用最后一个节点作为答案
            answer = chain.nodes[-1].entity if chain.nodes else chain.seed_entity

            question_obj = SynthesisQuestion(
                id=f"synthesis_{chain.seed_entity}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}",
                question=q.get('question', ''),
                answer=answer,
                evidence_chain=chain.to_dict(),
                constraints=q.get('constraints', []),
                reasoning_path=q.get('reasoning_path', []),
                difficulty_score=0.5,
                uniqueness_score=0.5
            )
            final_questions.append(question_obj)

        return final_questions


# ============================================================================
# Stage4 Processor
# ============================================================================

class Stage4QuestionBuilder:
    """
    第4阶段：问题构建器

    负责从证据链生成多跳推理问题
    """

    def __init__(self, llm_client, output_dir: str = "./"):
        self._builder = QuestionBuilder(llm_client)
        self.llm_client = llm_client
        self.output_dir = output_dir

    async def build_question(self, chain, num_questions: int = 3) -> List[SynthesisQuestion]:
        return await self._builder.build_question(chain=chain, num_questions=num_questions)

    async def generate_reverse_questions(self, chain, num_questions: int = 3) -> List[Dict[str, Any]]:
        generator = ReverseQuestionGenerator(self.llm_client)
        return await generator.generate(chain, num_questions)

    # ========================================================================
    # 多Agent泄露检测系统
    # ========================================================================

    async def filter_leaked_questions(
        self,
        questions: List[Dict[str, Any]],
        leakage_rewrite_max_rounds: int = 1,
        enable_ablation_analysis: bool = False,
        ablation_max_conditions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        使用多Agent系统处理存在泄露的问题：
        1. Solver Agent: 只看文本，尝试推理答案
        2. Judge Agent: 评估Solver的答案，判断是否泄露
        3. Generator Agent: 根据Judge的反馈重写问题

        循环最多 leakage_rewrite_max_rounds 轮，仍泄露的才过滤

        Args:
            questions: 待验证的问题列表
            leakage_rewrite_max_rounds: 泄露后重写的最大轮数
            enable_ablation_analysis: 是否启用条件消融分析
            ablation_max_conditions: 每个问题最多分析的条件数量

        Returns:
            处理后的问题列表（尽量修复泄露；修复失败的移除）
        """
        if not questions:
            return []

        # 记录每道题的修复轮次
        for q in questions:
            if "leakage_rewrite_rounds" not in q:
                q["leakage_rewrite_rounds"] = 0
            if "judge_feedback" not in q:
                q["judge_feedback"] = ""

        batch_size = 1

        # -----------------------------------------------------------
        # 内部函数：Solver Agent - 只看文本做题
        # -----------------------------------------------------------
        async def solve_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """让Solver Agent只看问题文本，尝试推理答案"""
            batch_json = json.dumps(batch, ensure_ascii=False, indent=2)

            prompt = f"""You are a Solver Agent.

## Task
Given only the question text, try to answer the question.
Focus on whether the question text ALONE gives you enough information to determine the answer.

## Instructions
- Read the question carefully.
- If you can determine the answer from the question text alone, provide your answer.

## Input (JSON)
{batch_json}

## Output format (JSON only)
{{
  "solves": [
    {{
      "original_index": 1,
      "solver_answer": "...",
      "solver_confidence": 0.0-1.0,
      "used_text_clues": "What specific text clues did you use to answer?"
    }}
  ]
}}

Return JSON only. Do not output any other text."""

            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=8192  # 增大 max_tokens
                )

                import re
                clean_response = response.strip()

                if not clean_response:
                    print(f"   ⚠️ Solver returned empty response")
                    return []

                if clean_response.startswith("```"):
                    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
                    if match:
                        clean_response = match.group(1).strip()

                # 尝试解析 JSON，如果失败则尝试修复
                try:
                    data = json.loads(clean_response)
                except json.JSONDecodeError as e:
                    # 尝试修复：提取每个 solve 对象
                    print(f"   ⚠️ Solver JSON parse failed: {e}")
                    print(f"   Raw response: {repr(clean_response[:500])}")
                    solves = []
                    try:
                        # 尝试匹配 solves 数组
                        array_match = re.search(r'"solves"\s*:\s*\[([\s\S]*)\]', clean_response)
                        if array_match:
                            array_content = array_match.group(1)
                            # 提取每个对象 (支持嵌套)
                            for obj_match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\})?[^{}]*\}', array_content):
                                try:
                                    obj_str = obj_match.group(0)
                                    obj = json.loads(obj_str)
                                    solves.append(obj)
                                except:
                                    pass
                    except Exception as e2:
                        print(f"   ⚠️ 正则修复也失败: {e2}")

                    if solves:
                        print(f"   ✅ 正则修复成功，提取了 {len(solves)} 个 solves")
                        return solves
                    return []

                return data.get("solves", [])
            except Exception:
                return []

        # -----------------------------------------------------------
        # 辅助函数：解析 Judge Markdown 格式
        # -----------------------------------------------------------
        def _parse_judge_markdown(markdown_text: str) -> List[Dict[str, Any]]:
            """解析 Judge 返回的 Markdown 格式评价结果"""
            judgments = []

            # 匹配 ### original_index=X, solve_result=Y 格式
            section_pattern = re.compile(
                r'###\s*original_index\s*=\s*(\d+),\s*solve_result\s*=\s*(correct|incorrect)',
                re.IGNORECASE
            )

            # 匹配字段: leakage_detected, leakage_type, model_issue, leakage_reason, judge_feedback
            field_pattern = re.compile(
                r'-\s*(leakage_detected|leakage_type|model_issue|leakage_reason|judge_feedback)\s*:\s*(.+)',
                re.IGNORECASE
            )

            current_judgment = {}
            current_section_idx = -1

            for line in markdown_text.split('\n'):
                section_match = section_pattern.match(line.strip())
                if section_match:
                    # 保存前一个 judgment
                    if current_judgment and current_section_idx >= 0:
                        judgments.append(current_judgment)

                    # 开始新的 judgment
                    original_idx = int(section_match.group(1))
                    solve_result = section_match.group(2)
                    current_section_idx = original_idx  # 用于判断是否保存

                    current_judgment = {
                        "original_index": original_idx,
                        "solve_result": solve_result,
                        "leakage_detected": False,
                        "leakage_type": "none",
                        "model_issue": False,
                        "leakage_reason": "",
                        "judge_feedback": ""
                    }
                    continue

                # 匹配字段
                field_match = field_pattern.match(line.strip())
                if field_match and current_judgment:
                    field_name = field_match.group(1).strip().lower()
                    field_value = field_match.group(2).strip()

                    # 清理值
                    if field_name == "leakage_detected":
                        current_judgment[field_name] = field_value.lower() == "true"
                    elif field_name == "model_issue":
                        current_judgment[field_name] = field_value.lower() == "true"
                    elif field_name == "leakage_type":
                        current_judgment[field_name] = field_value
                    else:
                        current_judgment[field_name] = field_value

            # 保存最后一个 judgment
            if current_judgment and current_section_idx >= 0:
                judgments.append(current_judgment)

            if judgments:
                print(f"   ✅ Markdown解析成功，提取了 {len(judgments)} 个 judgments")

            return judgments

        # -----------------------------------------------------------
        # 内部函数：Judge Agent - 评估Solver答案，判断是否泄露
        # -----------------------------------------------------------
        async def judge_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """让Judge Agent评估是否泄露，区分模型能力问题和题目问题"""
            batch_json = json.dumps(batch, ensure_ascii=False, indent=2)

            prompt = f"""You are a Judge Agent.

## Task
Evaluate EACH question in the input list. You MUST output a judgment for EVERY question in the input.
Do NOT skip any question.
IMPORTANT: There are EXACTLY {len(batch)} questions in the input.

## Input
The input is a JSON list of questions, where each question contains:
- index: The question number (use this as original_index in output)
- question: The question text
- answer: The expected answer
- solver_answer: The answer given by the Solver
- solver_confidence: The confidence score from Solver (0.0-1.0)
- used_text_clues: What clues the Solver used

{batch_json}

## Evaluation Steps (for EACH question)

### Step 1: Check if Solver Answer is CORRECT
- Compare solver_answer with answer
- If they match (or solver_answer contains the key part of answer) -> correct
- If they don't match OR solver said "cannot determine" -> incorrect

### Step 2: If CORRECT - Check for Information Leakage
- Did the Solver use UNNECESSARY/EXTRA information from the question to get the answer?
- If Solver needed ALL key details -> NO leakage (leakage_detected: false)
- If Solver used only partial/clue-like information -> LEAKAGE (leakage_detected: true)

### Step 3: If INCORRECT - Determine Root Cause
- MODEL_ISSUE: The question is valid but the model failed to reason correctly
- QUESTION_ISSUE: The question itself has problems (information redundancy, etc)

## Output Requirements
- You MUST output judgments for EACH question in the input
- Do NOT skip any question.
- Use the Markdown format below

## Output Format (Markdown)

```
## Judgments

### original_index=0, solve_result=correct/incorrect
- leakage_detected: true/false
- leakage_type: none/redundant_info
- model_issue: true/false
- leakage_reason: ...
- judge_feedback: ...

### original_index=1, solve_result=correct/incorrect
- leakage_detected: true/false
- leakage_type: none/redundant_info
- model_issue: true/false
- leakage_reason: ...
- judge_feedback: ...

...
```

IMPORTANT: Use the original_index from the input (e.g., 0, 1, 2...) as the section header, NOT sequential numbers (1, 2, 3...).

Return the judgments in Markdown format as shown above."""

            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=8192  # 增大 max_tokens
                )

                import re
                clean_response = response.strip()

                # DEBUG: 打印原始响应
                print(f"   [DEBUG Judge] Judge raw response (first 1000 chars): {repr(clean_response[:1000])}")

                if not clean_response:
                    print(f"   ⚠️ Judge返回空响应")
                    return []

                # 改进：更健壮地提取 JSON 块
                if clean_response.startswith("```"):
                    # 尝试匹配 ```json ... ``` 格式
                    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
                    if match:
                        clean_response = match.group(1).strip()
                    else:
                        # 如果没有结束标记，尝试提取 ```json 之后的所有内容
                        match = re.search(r'```(?:\w+)?\s*([\s\S]*)', clean_response)
                        if match:
                            clean_response = match.group(1).strip()

                # 清理响应，直接解析 Markdown 格式
                # 移除可能的代码块标记
                if clean_response.startswith("```"):
                    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
                    if match:
                        clean_response = match.group(1).strip()

                # 直接解析 Markdown 格式
                judgments = _parse_judge_markdown(clean_response)

                return judgments
            except Exception as e:
                print(f"   ⚠️ Judge批次失败: {e}")
                return []

        # -----------------------------------------------------------
        # 内部函数：Generator Agent - 根据Judge反馈重写问题
        # -----------------------------------------------------------
        async def generator_rewrite(leaked_validations: List[Dict[str, Any]]) -> None:
            """根据Judge的feedback重写泄露的问题"""
            rewrite_items: List[Dict[str, Any]] = []
            for v in leaked_validations:
                # 兼容两种字段名："index" 或 "original_index"
                idx = v.get("original_index") if v.get("original_index") is not None else v.get("index")
                if idx is None or idx < 0 or idx >= len(questions):
                    print(f"   ⚠️ 跳过无效索引: {idx}")
                    continue
                q = questions[idx]
                rewrite_items.append({
                    "original_index": idx,
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "target_entity": q.get("target_entity", ""),
                    "evidence_chain": q.get("evidence_chain", {}),
                    "judge_feedback": v.get("judge_feedback", ""),
                    "leakage_reason": v.get("leakage_reason", "")
                })
                print(f"   [Generator] 准备重写问题 {idx}: {q.get('question', '')[:50]}...")

            if not rewrite_items:
                return

            total_batches = (len(rewrite_items) - 1) // batch_size + 1
            for batch_start in range(0, len(rewrite_items), batch_size):
                batch_end = min(batch_start + batch_size, len(rewrite_items))
                batch = rewrite_items[batch_start:batch_end]
                print(f"   [Generator] Rewrite batch {batch_start//batch_size + 1}/{total_batches}...")
                rewritten = await _generator_rewrite_batch(batch)

                for item in rewritten:
                    idx = item.get("original_index")
                    new_q = item.get("rewritten_question")
                    if idx is None or new_q is None:
                        print(f"   ⚠️ Generator返回结果缺少必要字段: idx={idx}, rewritten_question={new_q}")
                        continue
                    if 0 <= idx < len(questions) and isinstance(new_q, str) and new_q.strip():
                        # 保存重写后的问题到临时字段，供后续验证使用
                        questions[idx]["rewritten_question"] = new_q.strip()
                        questions[idx]["rewritten_answer"] = questions[idx].get("answer", "")
                        questions[idx]["leakage_rewrite_rounds"] = int(questions[idx].get("leakage_rewrite_rounds", 0)) + 1
                        print(f"   [Generator] 问题 {idx} 已重写: {new_q[:60]}...")

        # -----------------------------------------------------------
        # 独立的Generator重写方法
        # -----------------------------------------------------------
        async def _generator_rewrite_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            batch_json = json.dumps(batch, ensure_ascii=False, indent=2)

            prompt = f"""You are a Generator Agent.

## Goal
Rewrite each QA question based on the Judge's feedback to eliminate information leakage while maintaining the multi-hop reasoning structure.
The question must still be answerable from the image, but the answer should NOT be inferable from the question text alone.

## Evidence Chain Based Question Generation Rules
When rewriting questions, you MUST follow these rules from the evidence chain based QA generation:

### 1. Reverse Entity Order
- The phrasing of the question should follow the entity order of the reasoning chain in reverse, from the final entity toward the seed entity.

### 2. Obfuscation & Vagueness
- NEVER directly mention the final answer entity or any intermediate entities by name.
- Describe entities indirectly using attributes, categories, or general characteristics.
- Avoid giving away the reasoning path; each step must be subtle.

### 3. Difficulty & Multi-Step Search
- The question should require exactly N reasoning steps (where N = number of nodes in the chain - 1).
- Each reasoning step should provide a vague constraint or clue that requires external search or verification.

### 4. Anti-Leakage Requirements
- NEVER directly mention the final answer entity in the question text
- Make clues VAGUE and INDIRECT - use indirect descriptions rather than specific names
- Use OBFUSCATED descriptions - describe entities by their attributes rather than names
- Avoid giving away the reasoning path - don't make it obvious how to get from A to B
- The question should be like a puzzle that requires multi-step search and reasoning

## Input Format (JSON)
Each item contains:
- question: Current QA question
- answer: Expected answer
- evidence_chain: The reasoning chain (nodes with entity, description, relationship)
- judge_feedback: Feedback from Judge on why the question leaks

{batch_json}

## Instructions
- Use the "judge_feedback" field to understand WHY the question leaks and HOW to fix it.
- Use the "evidence_chain" to understand the reasoning path: the chain starts from the visual entity (root node) and follows relationships to reach the answer.
- When rewriting:
  - The visual entity in the root node of the evidence chain should be replaced with a visual reference (e.g., "the object shown in the image", "it").
  - Do NOT reveal the identity of the visual entity through text descriptions.
  - Keep the reasoning chain (intermediate nodes and relationships) that leads to the answer.
  - Use vague/indirect descriptions instead of specific names for entities.
- Keep the expected answer type the same.
- Keep the language consistent with the original English.

## Output Format (JSON only)
{{
  "rewrites": [
    {{
      "original_index": 1,
      "rewritten_question": "..."
    }}
  ]
}}

Return JSON only. Do not output any other text."""

            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.01,
                    max_tokens=8192  # 增大 max_tokens
                )

                import re
                clean_response = response.strip()
                if not clean_response:
                    print(f"   ⚠️ Generator返回空响应")
                    return []

                if clean_response.startswith("```"):
                    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
                    if match:
                        clean_response = match.group(1).strip()

                # 尝试解析 JSON，如果失败则尝试修复
                try:
                    data = json.loads(clean_response)
                except json.JSONDecodeError as e:
                    # 尝试修复：提取每个 rewrite 对象
                    print(f"   ⚠️ Generator JSON解析失败: {e}")
                    print(f"   Raw response: {repr(clean_response[:500])}")
                    rewrites = []
                    try:
                        # 尝试匹配 rewrites 数组
                        array_match = re.search(r'"rewrites"\s*:\s*\[([\s\S]*)\]', clean_response)
                        if array_match:
                            array_content = array_match.group(1)
                            # 提取每个对象 (支持嵌套)
                            for obj_match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\})?[^{}]*\}', array_content):
                                try:
                                    obj_str = obj_match.group(0)
                                    obj = json.loads(obj_str)
                                    rewrites.append(obj)
                                except:
                                    pass
                    except Exception as e2:
                        print(f"   ⚠️ 正则修复也失败: {e2}")

                    if rewrites:
                        print(f"   ✅ 正则修复成功，提取了 {len(rewrites)} 个 rewrites")
                        return rewrites
                    return []

                rewrites = data.get("rewrites", [])
                if isinstance(rewrites, list):
                    return rewrites
                return []
            except Exception as e:
                print(f"   ⚠️ Generator批次失败: {e}")
                return []

        # -----------------------------------------------------------
        # 主循环：Solver -> Judge -> (Generator -> Solver -> Judge)
        # -----------------------------------------------------------

        # 准备验证数据
        validation_items: List[Dict[str, Any]] = []
        for idx, q in enumerate(questions):
            validation_items.append({
                "index": idx,
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "target_entity": q.get("target_entity", "")
            })

        # Step 1: 对所有题目跑 Solver -> Judge
        async def run_solver_judge() -> List[Dict[str, Any]]:
            # DEBUG: 打印验证项数量
            print(f"   [DEBUG] Running Solver+Judge on {len(validation_items)} items")
            if validation_items:
                print(f"   [DEBUG] First validation item: {json.dumps(validation_items[0], ensure_ascii=False)[:1000]}")

            # 分批调用 Solver
            solver_results: List[Dict[str, Any]] = []
            total_batches = (len(validation_items) - 1) // batch_size + 1

            # DEBUG: 打印批次信息
            print(f"   [DEBUG] Total batches: {total_batches}, batch_size: {batch_size}")

            for batch_start in range(0, len(validation_items), batch_size):
                batch_end = min(batch_start + batch_size, len(validation_items))
                batch = validation_items[batch_start:batch_end]
                print(f"   [Solver] 做题批次 {batch_start//batch_size + 1}/{total_batches}...")
                solver_results.extend(await solve_batch(batch))

            # DEBUG: 打印 solver 结果
            print(f"   [DEBUG] Solver returned {len(solver_results)} results")
            if solver_results:
                print(f"   [DEBUG] First solver result: {json.dumps(solver_results[0], ensure_ascii=False)[:500]}")

            # 把 Solver 结果合并进 validation_items
            solver_map = {r.get("original_index"): r for r in solver_results if "original_index" in r}
            for item in validation_items:
                idx = item.get("index")
                if idx in solver_map:
                    item["solver_answer"] = solver_map[idx].get("solver_answer", "")
                    item["solver_confidence"] = solver_map[idx].get("solver_confidence", 0.0)
                    item["used_text_clues"] = solver_map[idx].get("used_text_clues", "")

            # 分批调用 Judge
            judge_results: List[Dict[str, Any]] = []
            for batch_start in range(0, len(validation_items), batch_size):
                batch_end = min(batch_start + batch_size, len(validation_items))
                batch = validation_items[batch_start:batch_end]

                # DEBUG: 打印 batch 内容
                print(f"   [DEBUG] Judge batch size: {len(batch)}")
                if batch:
                    print(f"   [DEBUG] First item in judge batch: {json.dumps(batch[0], ensure_ascii=False)[:500]}")

                print(f"   [Judge] 评价批次 {batch_start//batch_size + 1}/{total_batches}...")
                result = await judge_batch(batch)
                print(f"   [DEBUG] Judge batch returned {len(result)} results")
                judge_results.extend(result)

            return judge_results

        # 只运行一次 Solver + Judge，不循环
        # 如果判断为泄露，则先尝试重写，重写后再次验证
        print(f"\n   === Leakage Detection & Rewriting ===")

        # 1. Solver + Judge
        judgments = await run_solver_judge()

        # DEBUG: 打印 judgments 的内容
        print(f"   [DEBUG] Judge returned {len(judgments)} judgments")
        if judgments:
            print(f"   [DEBUG] First judgment: {json.dumps(judgments[0], ensure_ascii=False)[:500]}")

        # 2. Classify issues
        model_issue_indices = set()
        question_issue_indices = set()
        redundant_info_indices = set()

        for j in judgments:
            idx = j.get("original_index")
            if not isinstance(idx, int):
                continue

            solve_result = j.get("solve_result", "incorrect")  # correct / incorrect
            leakage_detected = j.get("leakage_detected", False)
            model_issue = j.get("model_issue", False)

            # DEBUG: 打印每个 judgment 的内容
            print(f"   [DEBUG] Judgment {idx}: solve_result={solve_result}, leakage_detected={leakage_detected}, model_issue={model_issue}")

            # Write Judge feedback back to questions
            if 0 <= idx < len(questions):
                questions[idx]["judge_feedback"] = j.get("judge_feedback", "")
                questions[idx]["leakage_reason"] = j.get("leakage_reason", "")
                questions[idx]["solve_result"] = solve_result
                questions[idx]["model_issue"] = model_issue

            # Classify
            if solve_result == "correct":
                if leakage_detected:
                    redundant_info_indices.add(idx)  # correct but redundant info
            else:  # solve_result == "incorrect"
                if model_issue:
                    model_issue_indices.add(idx)  # model can't solve, don't rewrite
                else:
                    question_issue_indices.add(idx)  # question issue, need rewrite

        # 统计
        print(f"   📊 Statistics (Round 1):")
        print(f"      - Correct but redundant info leakage: {len(redundant_info_indices)}")
        print(f"      - Incorrect and question issue: {len(question_issue_indices)}")
        print(f"      - Incorrect but model issue: {len(model_issue_indices)}")

        # 3. 多轮重写（最多 2 轮）
        max_rewrite_rounds = 2
        rewrite_indices = redundant_info_indices | question_issue_indices
        need_filter_indices = set()

        for round_num in range(1, max_rewrite_rounds + 1):
            if not rewrite_indices:
                print(f"\n   ✅ No questions need rewriting in round {round_num}")
                break

            print(f"\n   === Rewriting Round {round_num}/{max_rewrite_rounds}: {len(rewrite_indices)} questions ===")

            # 准备需要重写的问题
            rewrite_items = []
            for idx in rewrite_indices:
                q = questions[idx]
                rewrite_items.append({
                    "index": idx,
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "judge_feedback": q.get("judge_feedback", ""),
                    "leakage_reason": q.get("leakage_reason", "")
                })

            # 调用重写函数
            await generator_rewrite(rewrite_items)

            # 4. 重新运行 Solver + Judge 验证重写后的问题
            print(f"\n   === Re-verify rewritten questions ===")

            # 收集重写后的验证项（检查 rewritten_question 字段是否有值）
            rewritten_validation_items = []
            for idx in rewrite_indices:
                q = questions[idx]
                # 检查是否有重写后的问题
                rewritten_q = q.get("rewritten_question", "")
                if rewritten_q and isinstance(rewritten_q, str) and rewritten_q.strip():
                    rewritten_validation_items.append({
                        "index": idx,
                        "question": rewritten_q,
                        "answer": q.get("rewritten_answer", q.get("answer", ""))
                    })
                    print(f"   [DEBUG] 已收集重写问题 {idx}: {rewritten_q[:50]}...")
                else:
                    print(f"   ⚠️ 问题 {idx} 没有重写后的问题")

            if not rewritten_validation_items:
                print(f"   ⚠️ No rewritten questions to verify")
                need_filter_indices |= rewrite_indices
                break

            # 对重写后的问题需要先运行 Solver，然后再运行 Judge
            print(f"   [Solver] 重新验证重写后的问题...")
            solver_results_recheck: List[Dict[str, Any]] = []
            total_solver_batches = (len(rewritten_validation_items) - 1) // batch_size + 1
            for batch_start in range(0, len(rewritten_validation_items), batch_size):
                batch_end = min(batch_start + batch_size, len(rewritten_validation_items))
                batch = rewritten_validation_items[batch_start:batch_end]
                print(f"   [Solver Re-verify] Batch {batch_start//batch_size + 1}/{total_solver_batches}...")
                results = await solve_batch(batch)
                solver_results_recheck.extend(results)

            # 把 Solver 结果合并进 rewritten_validation_items
            solver_map_recheck = {r.get("original_index"): r for r in solver_results_recheck if "original_index" in r}
            for item in rewritten_validation_items:
                idx = item.get("index")
                if idx in solver_map_recheck:
                    item["solver_answer"] = solver_map_recheck[idx].get("solver_answer", "")
                    item["solver_confidence"] = solver_map_recheck[idx].get("solver_confidence", 0.0)
                    item["used_text_clues"] = solver_map_recheck[idx].get("used_text_clues", "")

            # 然后运行 Judge
            rewritten_judgments = []
            total_judge_batches = (len(rewritten_validation_items) - 1) // batch_size + 1
            for batch_start in range(0, len(rewritten_validation_items), batch_size):
                batch_end = min(batch_start + batch_size, len(rewritten_validation_items))
                batch = rewritten_validation_items[batch_start:batch_end]
                print(f"   [Judge Re-verify] Batch {batch_start//batch_size + 1}/{total_judge_batches}...")
                result = await judge_batch(batch)
                rewritten_judgments.extend(result)

            # 更新重写后问题的状态，确定下一轮需要重写的问题
            still_need_rewrite_indices = set()
            for j in rewritten_judgments:
                idx = j.get("original_index")
                if not isinstance(idx, int):
                    continue

                solve_result = j.get("solve_result", "incorrect")
                leakage_detected = j.get("leakage_detected", False)
                model_issue = j.get("model_issue", False)

                # 如果重写后仍然有问题，需要继续重写（如果是最后一轮则过滤）
                if solve_result == "correct" and leakage_detected:
                    if round_num < max_rewrite_rounds:
                        still_need_rewrite_indices.add(idx)
                    else:
                        need_filter_indices.add(idx)
                elif solve_result == "incorrect" and not model_issue:
                    if round_num < max_rewrite_rounds:
                        still_need_rewrite_indices.add(idx)
                    else:
                        need_filter_indices.add(idx)
                else:
                    # 重写成功，更新问题为重写后的版本
                    if 0 <= idx < len(questions):
                        if "rewritten_question" in questions[idx]:
                            questions[idx]["question"] = questions[idx].pop("rewritten_question")
                            questions[idx]["answer"] = questions[idx].pop("rewritten_answer")
                        questions[idx]["judge_feedback"] = j.get("judge_feedback", "")
                        questions[idx]["solve_result"] = solve_result
                        questions[idx]["leakage_rewrite_rounds"] = round_num

                # 更新问题的状态
                if 0 <= idx < len(questions):
                    questions[idx]["judge_feedback"] = j.get("judge_feedback", "")
                    questions[idx]["solve_result"] = solve_result

            print(f"   📊 Round {round_num}: {len(still_need_rewrite_indices)} questions still need rewrite")

            # 下一轮重写的索引
            rewrite_indices = still_need_rewrite_indices

        # 如果还有剩余的 rewrite_indices（最后一轮之后还有问题的）
        if rewrite_indices:
            need_filter_indices |= rewrite_indices

        # 过滤问题
        filtered_questions = []
        for idx, q in enumerate(questions):
            if idx in need_filter_indices:
                print(f"   ⚠️ 过滤问题 {idx}: {q.get('question', '')[:50]}...")
                continue
            filtered_questions.append(q)

        print(f"   ✅ 过滤后剩余 {len(filtered_questions)}/{len(questions)} 个问题")

        # ------------------------------------------------------------------
        # 条件消融分析 (Condition Ablation Analysis)
        # ------------------------------------------------------------------
        if enable_ablation_analysis and filtered_questions:
            print(f"\n🔬 Step 5b: 执行条件消融分析...")
            try:
                analysis_results = await self.analyze_conditions(
                    filtered_questions,
                    max_conditions=ablation_max_conditions
                )
                # 保存分析结果
                self._save_ablation_results(analysis_results)
                print(f"   消融分析完成: 生成了 {len(analysis_results)} 份分析报告")
            except Exception as e:
                print(f"   ⚠️ 条件消融分析失败: {e}")

        return filtered_questions

    # ============================================================================
    # 条件消融分析 (Condition Ablation Analysis)
    # ============================================================================

    async def analyze_conditions(
        self,
        questions: List[Dict[str, Any]],
        max_conditions: int = 5,
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        分析每个问题的条件并执行消融实验

        Args:
            questions: 问题列表
            max_conditions: 每个问题最多分析的条件数量
            batch_size: LLM批处理大小

        Returns:
            每个问题的消融分析结果列表
        """
        print(f"\n🔬 条件消融分析: 分析 {len(questions)} 个问题")

        # ------------------------------------------------------------------
        # Step 1: 条件识别 - 用模型提取问题中的条件
        # ------------------------------------------------------------------
        async def extract_conditions_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """批量提取问题中的条件"""
            batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
            prompt = f"""You are a question analysis assistant. Please analyze the following questions and identify all constraint conditions in each question.

## Task
- Identify each independent constraint in the question that can be removed
- Conditions typically describe entity attributes, locations, relationships, etc.
- Output in JSON format with all identified conditions

## Input Format
{batch_json}

## Output Format (JSON only)
{{
  "extractions": [
    {{
      "original_index": 0,
      "conditions": [
        {{
          "index": 0,
          "text": "完整描述的条件文本",
          "entity_mentioned": "条件中提到的实体名（如果没有则填null）"
        }},
        ...
      ]
    }}
  ]
}}

Return JSON only. Do not output any other text."""

            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=8192
                )

                import re
                clean_response = response.strip()
                if not clean_response:
                    return [{"original_index": q.get("index", i), "conditions": []} for i, q in enumerate(batch)]

                if clean_response.startswith("```"):
                    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
                    if match:
                        clean_response = match.group(1).strip()

                try:
                    data = json.loads(clean_response)
                except json.JSONDecodeError:
                    array_match = re.search(r'"extractions"\s*:\s*\[([\s\S]*)\]', clean_response)
                    if array_match:
                        return [{"original_index": q.get("index", i), "conditions": []} for i, q in enumerate(batch)]
                    return [{"original_index": q.get("index", i), "conditions": []} for i, q in enumerate(batch)]

                return data.get("extractions", [])
            except Exception as e:
                print(f"   ⚠️ 条件提取批次失败: {e}")
                return [{"original_index": q.get("index", i), "conditions": []} for i, q in enumerate(batch)]

        # ------------------------------------------------------------------
        # Step 2: 条件移除 - 移除指定条件生成新问题
        # ------------------------------------------------------------------
        async def remove_condition_batch(
            items: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """批量移除条件生成新问题"""
            batch_json = json.dumps(items, ensure_ascii=False, indent=2)
            prompt = f"""You are a question rewriting assistant. Please remove the specified condition from each question to generate a new question.

## Task
- Remove the single specified condition from the question
- Keep the question grammatically correct and fluent
- Do not change other conditions
- You may adjust sentence structure to keep the question meaningful

## Input Format
{batch_json}

## Output Format (JSON only)
{{
  "rewrites": [
    {{
      "original_index": 0,
      "condition_index": 0,
      "ablated_question": "改写后的新问题"
    }}
  ]
}}

Return JSON only. Do not output any other text."""

            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.01,
                    max_tokens=4096
                )

                import re
                clean_response = response.strip()
                if not clean_response:
                    return [{"original_index": it.get("original_index", 0), "condition_index": it.get("condition_index", 0), "ablated_question": it.get("original_question", "")} for it in items]

                if clean_response.startswith("```"):
                    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
                    if match:
                        clean_response = match.group(1).strip()

                try:
                    data = json.loads(clean_response)
                except json.JSONDecodeError:
                    return [{"original_index": it.get("original_index", 0), "condition_index": it.get("condition_index", 0), "ablated_question": it.get("original_question", "")} for it in items]

                return data.get("rewrites", [])
            except Exception as e:
                print(f"   ⚠️ 条件移除批次失败: {e}")
                return [{"original_index": it.get("original_index", 0), "condition_index": it.get("condition_index", 0), "ablated_question": it.get("original_question", "")} for it in items]

        # ------------------------------------------------------------------
        # Step 3: Solver验证消融后的问题
        # ------------------------------------------------------------------
        async def solve_ablated_batch(
            items: List[Dict[str, Any]]
        ) -> List[Dict[str, Any]]:
            """批量验证消融后的问题"""
            batch_json = json.dumps(items, ensure_ascii=False, indent=2)
            prompt = f"""You are a Solver Agent.

## Task
Answer the question based on the provided context. If you cannot answer, provide your best guess.

## Input Format
{batch_json}

## Output Format (JSON only)
{{
  "answers": [
    {{
      "original_index": 0,
      "condition_index": 0,
      "solver_answer": "your answer",
      "solver_confidence": 0.8
    }}
  ]
}}

Return JSON only. Do not output any other text."""

            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    temperature=0.0,
                    max_tokens=4096
                )

                import re
                clean_response = response.strip()
                if not clean_response:
                    return [{"original_index": it.get("original_index", 0), "condition_index": it.get("condition_index", 0), "solver_answer": "", "solver_confidence": 0.0} for it in items]

                if clean_response.startswith("```"):
                    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', clean_response)
                    if match:
                        clean_response = match.group(1).strip()

                try:
                    data = json.loads(clean_response)
                except json.JSONDecodeError:
                    return [{"original_index": it.get("original_index", 0), "condition_index": it.get("condition_index", 0), "solver_answer": "", "solver_confidence": 0.0} for it in items]

                return data.get("answers", [])
            except Exception as e:
                print(f"   ⚠️ 消融问题Solver批次失败: {e}")
                return [{"original_index": it.get("original_index", 0), "condition_index": it.get("condition_index", 0), "solver_answer": "", "solver_confidence": 0.0} for it in items]

        # ------------------------------------------------------------------
        # 主流程
        # ------------------------------------------------------------------
        analysis_results: List[Dict[str, Any]] = []

        # Step 1: 提取所有问题的条件
        print(f"   Step 1: 提取问题条件...")
        extraction_map: Dict[int, List[Dict]] = {}

        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch = [{"index": i, "question": q.get("question", ""), "answer": q.get("answer", "")}
                     for i, q in enumerate(questions[batch_start:batch_end])]

            print(f"   [Condition Extraction] Batch {batch_start//batch_size + 1}...")
            extractions = await extract_conditions_batch(batch)

            for ext in extractions:
                idx = ext.get("original_index", 0)
                conditions = ext.get("conditions", [])[:max_conditions]
                extraction_map[batch_start + idx] = conditions

        # 初始化分析结果
        for i, q in enumerate(questions):
            analysis_results.append({
                "question_id": q.get("id", f"q_{i}"),
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "num_conditions": len(extraction_map.get(i, [])),
                "conditions": [],
                "ablation_results": [],
                "summary": {}
            })

        # Step 2 & 3: 对每个条件的每个问题执行消融
        print(f"   Step 2-3: 执行消融实验...")
        ablate_items: List[Dict] = []

        for q_idx, q in enumerate(questions):
            conditions = extraction_map.get(q_idx, [])
            for cond in conditions:
                ablate_items.append({
                    "original_index": q_idx,
                    "condition_index": cond.get("index", 0),
                    "original_question": q.get("question", ""),
                    "original_answer": q.get("answer", ""),
                    "condition_text": cond.get("text", ""),
                    "condition_entity": cond.get("entity_mentioned")
                })

        # 分批处理消融
        for batch_start in range(0, len(ablate_items), batch_size):
            batch_end = min(batch_start + batch_size, len(ablate_items))
            batch = ablate_items[batch_start:batch_end]

            print(f"   [Ablation] Batch {batch_start//batch_size + 1}, processing {len(batch)} ablations...")

            # 2a: 移除条件生成新问题
            rewrites = await remove_condition_batch(batch)
            rewrite_map = {(r.get("original_index", 0), r.get("condition_index", 0)): r.get("ablated_question", "")
                          for r in rewrites}

            # 2b: Solver验证新问题
            solve_items = []
            for item in batch:
                key = (item.get("original_index", 0), item.get("condition_index", 0))
                ablated_q = rewrite_map.get(key, item.get("original_question", ""))
                solve_items.append({
                    "original_index": item.get("original_index", 0),
                    "condition_index": item.get("condition_index", 0),
                    "question": ablated_q,
                    "expected_answer": item.get("original_answer", "")
                })

            answers = await solve_ablated_batch(solve_items)
            answer_map = {(a.get("original_index", 0), a.get("condition_index", 0)): a
                         for a in answers}

            # 合并结果
            for item in batch:
                q_idx = item.get("original_index", 0)
                c_idx = item.get("condition_index", 0)
                key = (q_idx, c_idx)

                solver_result = answer_map.get(key, {})
                solver_answer = solver_result.get("solver_answer", "")

                # 判断正确性：简单字符串匹配
                expected = item.get("original_answer", "").lower().strip()
                actual = solver_answer.lower().strip()
                solver_correct = expected in actual or actual in expected or expected == actual

                if q_idx < len(analysis_results):
                    analysis_results[q_idx]["conditions"].append({
                        "index": c_idx,
                        "text": item.get("condition_text", ""),
                        "entity_mentioned": item.get("condition_entity")
                    })
                    analysis_results[q_idx]["ablation_results"].append({
                        "condition_index": c_idx,
                        "condition_text": item.get("condition_text", ""),
                        "ablated_question": rewrite_map.get(key, item.get("original_question", "")),
                        "solver_answer": solver_answer,
                        "solver_correct": solver_correct,
                        "condition_critical": not solver_correct
                    })

        # Step 4: 生成汇总报告
        print(f"   Step 4: 生成汇总报告...")
        for result in analysis_results:
            ablation = result.get("ablation_results", [])
            critical = [r for r in ablation if r.get("condition_critical", False)]
            redundant = [r for r in ablation if not r.get("condition_critical", True)]

            result["summary"] = {
                "critical_conditions": [r.get("condition_index") for r in critical],
                "redundant_conditions": [r.get("condition_index") for r in redundant],
                "redundancy_ratio": len(redundant) / len(ablation) if ablation else 0.0,
                "total_conditions": len(ablation)
            }

        return analysis_results

    def _save_ablation_results(
        self,
        results: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """保存消融分析结果"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"ablation_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "ablation_analysis.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"   💾 消融分析结果已保存至: {output_path}")
        return output_path


# ============================================================================
# 独立运行入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    async def main():
        parser = argparse.ArgumentParser(description="Stage4: 问题构建")
        parser.add_argument("--input", type=str, required=True, help="Stage3输出的JSON文件")
        parser.add_argument("--output", type=str, default="./stage4_output.json", help="输出文件")
        parser.add_argument("--questions-per-chain", type=int, default=1, help="每条链生成的问题数")
        parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY", ""), help="API Key")
        parser.add_argument("--base-url", type=str, default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), help="API Base URL")
        parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="模型名称")
        args = parser.parse_args()

        with open(args.input, 'r', encoding='utf-8') as f:
            chains_data = json.load(f)

        # 修复导入路径
        # (sys 和 os 已在文件顶部导入)
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _parent_dir = os.path.dirname(_current_dir)  # shujuhecheng/modules
        _grandparent_dir = os.path.dirname(_parent_dir)  # shujuhecheng

        # 添加路径
        paths_to_add = [
            _current_dir,
            _parent_dir,
            _grandparent_dir,
        ]
        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)

        # 导入配置和客户端
        try:
            from .config import SynthesisConfig
            from .llm_client import LLMClient, get_friday_client
        except ImportError:
            from config import SynthesisConfig
            from llm_client import LLMClient, get_friday_client

        config = SynthesisConfig(
            llm_api_key=args.api_key,
            llm_base_url=args.base_url,
            llm_model=args.model
        )
        friday_client = get_friday_client(
            model_name=config.llm_model,
            api_url=config.llm_base_url,
            api_token=config.llm_api_key or None,
            use_api_key_manager=not bool(config.llm_api_key),
        )
        llm_client = LLMClient(friday_client=friday_client)

        builder = Stage4QuestionBuilder(llm_client)

        all_questions = []
        for chain_data in chains_data:
            nodes = chain_data.get("nodes", [])
            if not nodes:
                continue

            from stage3_evidence_chain_builder import EvidenceChain, EvidenceNode
            chain = EvidenceChain(
                seed_entity=nodes[0].get("entity", "") if nodes else "",
                nodes=[],
                edges=[],
                chain_depth=len(nodes),
                diversity_score=0.0,
                uniqueness_score=0.0
            )
            for node_data in nodes:
                node = EvidenceNode(
                    entity=node_data.get("entity", ""),
                    entity_type=node_data.get("entity_type", ""),
                    description=node_data.get("description", ""),
                    depth=node_data.get("depth", 0)
                )
                chain.nodes.append(node)

            print(f"📝 为链构建问题...")
            questions = await builder.build_question(chain=chain, num_questions=args.questions_per_chain)
            all_questions.extend(questions)

        print(f"\n✅ Stage4 完成，生成了 {len(all_questions)} 个问题")

        result = []
        for q in all_questions:
            result.append(q.to_dict())

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {args.output}")

    asyncio.run(main())
