"""
第5阶段：VQA生成 (VQA Generation)

生成视觉问答数据
包含多Agent泄露检测系统（Generator / Solver / Judge）
"""

import os
import json
import sys
import asyncio
from typing import List, Dict, Any, Optional

# 处理导入路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from .vqa_generator_core import VistaHopVQAGenerator, VistaHopVQAConfig, VQAItem
    from .stage4_question_builder import Stage4QuestionBuilder
except ImportError:
    from synthesis.vqa_generator_core import VistaHopVQAGenerator, VistaHopVQAConfig, VQAItem
    from synthesis.stage4_question_builder import Stage4QuestionBuilder


class Stage5VQAGenerator:
    """
    第5阶段：VQA生成器

    负责生成视觉问答数据
    包含多Agent泄露检测系统：
    - Generator (出题): 负责生成/重写问题
    - Solver (做题): 只看文本，尝试推理答案
    - Judge (评价): 评估Solver的答案，判断是否泄露
    """

    def __init__(
        self,
        config,
        llm_client,
        enable_leakage_check: bool = True,
        leakage_rewrite_max_rounds: int = 1,
    ):
        """
        初始化

        Args:
            config: VistaHop配置
            llm_client: LLM客户端
            enable_leakage_check: 是否启用泄露检测
            leakage_rewrite_max_rounds: 泄露后重写的最大轮数（0=不重写直接过滤）
        """
        self.config = config
        self.llm_client = llm_client
        self.enable_leakage_check = enable_leakage_check
        self.leakage_rewrite_max_rounds = max(0, int(leakage_rewrite_max_rounds))

        # 初始化 Stage4QuestionBuilder（用于泄露检测）
        self.stage4_builder = Stage4QuestionBuilder(llm_client)

    async def generate_vqa(
        self,
        questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成VQA

        Args:
            questions: VistaHop生成的问题列表

        Returns:
            VQA生成结果
        """
        if not self.config.generate_vqa:
            return {"total_vqa": 0, "vqa_items": [], "entities_found": []}

        generator = self._init_vqa_generator()
        if not generator:
            return {"total_vqa": 0, "vqa_items": [], "entities_found": []}

        # Step 5: 多Agent泄露检测（在VQA生成之前进行）
        # Generator -> Solver -> Judge 循环
        if self.enable_leakage_check:
            print("\n🔍 Step 5: 执行多Agent泄露检测（Generator->Solver->Judge循环）...")
            questions = await self.stage4_builder.filter_leaked_questions(
                questions,
                leakage_rewrite_max_rounds=self.leakage_rewrite_max_rounds,
                enable_ablation_analysis=getattr(self.config, 'enable_ablation_analysis', False),
                ablation_max_conditions=getattr(self.config, 'ablation_max_conditions', 5)
            )
            print(f"   泄露检测完成: 剩余 {len(questions)} 个问题待处理")
        else:
            print("\n⏭️ Step 5: 跳过泄露检测")

        # Step 6: 生成VQA
        result = await generator.generate_vqa_from_questions(
            synthesis_questions=questions,
            original_image_url=self.config.image_url
        )

        # 保存VQA结果
        await generator.save_results(result)

        return {
            "total_vqa": result.total_vqa,
            "vqa_items": [item.to_dict() for item in result.vqa_items],
            "entities_found": [e.to_dict() for e in result.entities_found],
            "errors": result.errors
        }

    # ========================================================================
    # 旧版兼容方法（VQA生成后检测用，可保留）
    # ========================================================================

    async def _validate_leakage(self, result) -> Any:
        """验证问题是否存在信息泄露（在VQA生成之后，用于分析）"""
        # 兼容旧代码，暂时保留
        if not getattr(result, "vqa_items", None):
            return result
        # 这里可以复用上面的逻辑，但为了不影响主流程，暂时不做处理
        return result

    async def _validate_batch(self, batch: List[Dict]) -> List[Dict]:
        """兼容旧接口，内部调用Judge"""
        # 这个方法旧版还在用，但我们已经重构到多Agent系统里了
        # 这里为了兼容，返回空让它不干扰主流程
        return []

    def _init_vqa_generator(self) -> Optional[VistaHopVQAGenerator]:
        """初始化VQA生成器"""
        if not self.config.generate_vqa:
            return None

        if not self.config.image_url:
            print("   ⚠️ 未配置图片URL，跳过VQA生成")
            return None

        # 加载 Stage1 实体数据（包含视觉描述）
        stage1_entities = []
        stage1_path = os.path.join(self.config.output_dir, "stage1_entities.json")
        if os.path.exists(stage1_path):
            try:
                with open(stage1_path, 'r', encoding='utf-8') as f:
                    stage1_data = json.load(f)
                # 提取 name 和 description 字段
                for entity in stage1_data:
                    name = entity.get("name", "").strip()
                    description = entity.get("description", "").strip()
                    if name and description:
                        stage1_entities.append({
                            "name": name,
                            "description": description
                        })
                print(f"   [INFO] Loaded {len(stage1_entities)} stage1 entities with visual descriptions")
            except Exception as e:
                print(f"   [WARNING] Failed to load stage1_entities.json: {e}")
        else:
            print(f"   [DEBUG] stage1_entities.json not found at {stage1_path}")

        vqa_config = VistaHopVQAConfig(
            image_url=self.config.image_url,
            serpapi_key=self.config.serpapi_key,
            images_per_entity=self.config.vqa_images_per_entity,
            output_dir=self.config.output_dir.replace("_first", "_vqa").replace("_test", "_vqa"),
            stage1_entities=stage1_entities,
            enable_question_simplify=getattr(self.config, 'enable_question_simplify', True)
        )

        return VistaHopVQAGenerator(vqa_config, self.llm_client)

    async def save_vqa_results(
        self,
        vqa_result: Dict[str, Any]
    ) -> None:
        """保存VQA结果"""
        # 兼容旧代码
        pass
