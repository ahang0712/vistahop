"""
VistaHop管道核心模块

VistaHop: A Bottom-up Method for Generating Complex Multi-hop Reasoning Questions

主管道类:
- SynthesisPipeline: VistaHop主管道类

五个阶段：
1. 第1阶段：数据源与适应 (Data Source and Adaptation)
   - 从图片提取实体
   - 实体过滤和清洗

2. 第2阶段：节点信息构建 (Node Information Construction)
   - 为每个实体构建知识图谱节点
   - 收集节点描述和属性

3. 第3阶段：证据链构建 (Evidence Chain Construction)
   - 构建推理证据链
   - NLI关系验证
   - 多样性评估

4. 第4阶段：问题构建 (Final Question Construction)
   - 反向问题生成
   - 问题混淆
   - 迭代精炼

5. 第5阶段：VQA生成 (VQA Generation)
   - 生成视觉问答数据
"""

import asyncio
import os
from typing import List, Dict, Any, Optional

# 尝试相对导入，如果失败则使用绝对导入
# 注意：确保优先从本地 synthesis 包导入，避免与父目录的 config.py 冲突
try:
    from .config import SynthesisConfig, SearchConfig
except ImportError:
    try:
        from config import SynthesisConfig, SearchConfig
    except ImportError:
        SynthesisConfig = None
        SearchConfig = None

try:
    from .result import SynthesisResult
except ImportError:
    try:
        from .synthesis_result import SynthesisResult
    except ImportError:
        SynthesisResult = None

try:
    from .llm_client import LLMClient, get_friday_client
except ImportError:
    try:
        from llm_client import LLMClient, get_friday_client
    except ImportError:
        LLMClient = None
        get_friday_client = None

try:
    from .search_client import UnifiedSearchClient
except ImportError:
    try:
        from search_client import UnifiedSearchClient
    except ImportError:
        UnifiedSearchClient = None

# 导入拆分的阶段模块
try:
    from .stage1_extractor import Stage1Extractor
except ImportError:
    try:
        from stage1_extractor import Stage1Extractor
    except ImportError:
        Stage1Extractor = None

try:
    from .stage2_node_builder import Stage2NodeBuilder
except ImportError:
    try:
        from stage2_node_builder import Stage2NodeBuilder
    except ImportError:
        Stage2NodeBuilder = None

try:
    from .stage3_evidence_chain_builder import Stage3EvidenceChainBuilder
except ImportError:
    try:
        from stage3_evidence_chain_builder import Stage3EvidenceChainBuilder
    except ImportError:
        Stage3EvidenceChainBuilder = None

try:
    from .stage4_question_builder import Stage4QuestionBuilder
except ImportError:
    try:
        from stage4_question_builder import Stage4QuestionBuilder
    except ImportError:
        Stage4QuestionBuilder = None

try:
    from .result_saver import ResultSaver
except ImportError:
    try:
        from result_saver import ResultSaver
    except ImportError:
        ResultSaver = None

try:
    from .stage5_vqa_generator import Stage5VQAGenerator
except ImportError:
    try:
        from stage5_vqa_generator import Stage5VQAGenerator
    except ImportError:
        Stage5VQAGenerator = None

try:
    from .stage6_fusion import Stage6FusionGenerator, run_stage6_fusion, FusionRule
except ImportError:
    try:
        from stage6_fusion import Stage6FusionGenerator, run_stage6_fusion, FusionRule
    except ImportError:
        Stage6FusionGenerator = None
        run_stage6_fusion = None
        FusionRule = None


class SynthesisPipeline:
    """
    VistaHop主管道

    整合五个阶段实现端到端的多跳推理问题生成

    Attributes:
        config: VistaHop配置
        llm_client: LLM客户端
        search_client: 搜索客户端
        statistics: 统计信息
    """

    def __init__(self, config: SynthesisConfig):
        """
        初始化VistaHop管道

        Args:
            config: VistaHop配置对象
        """
        self.config = config

        # Initialize LLM client
        self.llm_client = self._init_llm_client()

        # Initialize search client
        search_config = SearchConfig(
            search_api_key=self.config.search_api_key,
            search_engine=self.config.search_engine,
            proxy=self.config.proxy  # 传递代理配置
        )
        self.search_client = UnifiedSearchClient(search_config)

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

        # Statistics (must be initialized before _init_components)
        self.statistics: Dict[str, Any] = {}

        # Initialize stage components
        self._init_components()

    def _init_llm_client(self) -> LLMClient:
        """Initialize LLM client"""
        friday_client = get_friday_client(
            model_name=self.config.llm_model,
            api_url=self.config.llm_base_url,
            api_token=self.config.llm_api_key if self.config.llm_api_key else None,
            use_api_key_manager=not bool(self.config.llm_api_key)
        )
        return LLMClient(friday_client=friday_client)

    def _init_components(self):
        """Initialize stage components"""
        # Stage 1: Entity extractor
        self.stage1_extractor = Stage1Extractor(self.llm_client, self.config)

        # Stage 2: Node builder
        self.stage2_node_builder = Stage2NodeBuilder(self.llm_client, self.config)

        # Stage 3: Evidence chain builder
        self.stage3_evidence_chain_builder = Stage3EvidenceChainBuilder(
            self.llm_client,
            self.search_client,
            self.config
        )

        # Stage 4: Question builder
        self.stage4_question_builder = Stage4QuestionBuilder(self.llm_client)

        # Stage 5: VQA generator
        self.vqa_generator = Stage5VQAGenerator(self.config, self.llm_client, enable_leakage_check=True)

        # Stage 6: Multi-Chain Fusion
        if Stage6FusionGenerator:
            self.stage6_fusion = None  # Lazy initialization
        else:
            self.stage6_fusion = None

        # Result saver
        self.result_saver = ResultSaver(self.config, self.statistics)

    async def run(
        self,
        entities: Optional[List[str]] = None,
        image_url: Optional[str] = None
    ) -> SynthesisResult:
        """
        Run VistaHop pipeline

        Args:
            entities: Entity list (extract from image if None)
            image_url: Image URL (for entity extraction)

        Returns:
            SynthesisResult: Processing result
        """
        result = SynthesisResult()
        try:
            print("=" * 70)
            print("🚀 VistaHop: Multi-hop Reasoning Question Generation Pipeline")
            print("=" * 70)

            # =====================================================
            # Stage 1: Data Source and Adaptation
            # =====================================================
            print("\n📋 Stage 1: Data Source and Adaptation")
            print("-" * 50)

            # 使用 Stage1Extractor 处理实体提取
            entities_result = await self.stage1_extractor.extract_entities(
                entities=entities,
                image_url=image_url
            )

            if entities_result:
                result.entities = entities_result
                self.statistics["total_entities"] = len(result.entities)
                print(f"   📊 Total entities: {len(result.entities)}")
                # 保存 Stage1 中间结果
                if self.config.save_intermediate:
                    self.result_saver.save_intermediate("stage1_entities.json", result.entities)
            else:
                result.errors.append("No entities available")
                return result

            # 如果只运行阶段1，则在此停止
            if self.config.max_stage == 1:
                print("\n✅ Stage 1 completed (max_stage=1)")
                self.result_saver.save_final_results(result)
                self.result_saver.print_summary(result)
                return result

            # =====================================================
            # Determine entities to process (moved before Stage 2)
            # Priority: first_entity_only (1) > num_entities_for_chains (N)
            # =====================================================
            if self.config.first_entity_only:
                entities_to_process = [result.entities[0]]
            elif self.config.num_entities_for_chains > 0:
                entities_to_process = result.entities[:self.config.num_entities_for_chains]
            else:
                entities_to_process = result.entities

            total_entities = len(entities_to_process)
            print(f"[INFO] Processing {total_entities} entities for pipeline")

            # =====================================================
            # Stage 2: Node Information Construction
            # =====================================================
            print("\n📚 Stage 2: Node Information Construction")
            print("-" * 50)

            node_info = await self.stage2_node_builder.build_node_information(entities_to_process)
            result.statistics["node_info"] = node_info

            print(f"   ✅ Node information construction completed")
            print(f"   📊 Built detailed information for {len(node_info)} nodes")
            # 保存 Stage2 中间结果
            if self.config.save_intermediate:
                self.result_saver.save_intermediate("stage2_nodes.json", node_info)

            # 如果只运行阶段2，则在此停止
            if self.config.max_stage == 2:
                print("\n✅ Stage 2 completed (max_stage=2)")
                self.result_saver.save_final_results(result)
                self.result_saver.print_summary(result)
                return result

            # =====================================================
            # Stage 3: Evidence Chain Construction
            # =====================================================
            print("\n🔗 Stage 3: Evidence Chain Construction")
            print("-" * 50)

            all_chains = []

            print(f"[INFO] Processing {total_entities} entities for evidence chains")

            for idx, entity in enumerate(entities_to_process, 1):
                # Print progress bar
                progress = idx / total_entities
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)

                # 支持两种输入类型：字符串 或 字典
                if isinstance(entity, dict):
                    entity_name = entity.get("cleaned_name") or entity.get("name", str(entity))
                    # 提取属性信息
                    image_attribute = entity.get("image_attribute", "")  # 如 "集装箱上出现最多的公司"
                    image_attribute_value = entity.get("image_attribute_value", "")  # 如 "Hapag-Lloyd"
                else:
                    entity_name = str(entity)
                    image_attribute = ""
                    image_attribute_value = ""

                print(f"\n   {bar} {idx}/{total_entities} ({progress*100:.1f}%) Processing: {entity_name[:30]}")

                chains = await self.stage3_evidence_chain_builder.build_chain(
                    entity_name,
                    num_chains=self.config.chains_per_entity,
                    image_attribute=image_attribute,
                    image_attribute_value=image_attribute_value or entity_name,
                    stage2_nodes=node_info
                )
                all_chains.extend(chains)

            result.evidence_chains = [chain.to_dict() for chain in all_chains]
            self.statistics["total_chains"] = len(all_chains)

            print(f"\n   📊 证据链总数: {len(all_chains)}")

            # 保存中间结果
            if self.config.save_intermediate:
                self.result_saver.save_intermediate("stage3_evidence_chains.json", result.evidence_chains)

            # 如果只运行阶段3，则在此停止
            if self.config.max_stage == 3:
                print("\n✅ Stage 3 completed (max_stage=3)")
                self.result_saver.save_final_results(result)
                self.result_saver.print_summary(result)
                return result

            # =====================================================
            # Stage 4: Question Construction
            # =====================================================
            print("\n❓ 第4阶段：问题构建")
            print("-" * 50)

            all_questions = []
            total_chains = len(all_chains)

            for idx, chain in enumerate(all_chains, 1):
                # 打印进度条
                progress = idx / total_chains
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '░' * (bar_length - filled_length)
                print(f"   {bar} {idx}/{total_chains} ({progress*100:.1f}%) 正在生成问题...", end='\r')

                questions = await self.stage4_question_builder.build_question(
                    chain,
                    num_questions=self.config.questions_per_chain
                )
                all_questions.extend(questions)

            # 清除进度条
            print(" " * 80, end='\r')

            result.questions = [q.to_dict() for q in all_questions]
            self.statistics["total_questions"] = len(all_questions)

            print(f"\n   📊 生成问题总数: {len(all_questions)}")

            # 保存中间结果
            if self.config.save_intermediate:
                self.result_saver.save_intermediate("stage4_questions.json", result.questions)

            # 如果只运行阶段4，则在此停止
            if self.config.max_stage == 4:
                print("\n✅ Stage 4 completed (max_stage=4)")
                self.result_saver.save_final_results(result)
                self.result_saver.print_summary(result)
                return result

            # =====================================================
            # Stage 5: VQA Generation (Optional)
            # =====================================================
            if self.config.generate_vqa:
                print("\n🎨 第5阶段：VQA生成")
                print("-" * 50)

                vqa_result = await self.vqa_generator.generate_vqa(result.questions)
                self.statistics["vqa_generation"] = vqa_result

                # 保存VQA结果
                await self.vqa_generator.save_vqa_results(vqa_result)

                # 将VQA结果保存到result中（用于Stage 6）
                result.vqa_items = vqa_result.get("vqa_items", [])

            # 如果只运行阶段5，则在此停止
            if self.config.max_stage == 5:
                print("\n✅ Stage 5 completed (max_stage=5)")
                self.result_saver.save_final_results(result)
                self.result_saver.print_summary(result)
                return result

            # =====================================================
            # Stage 6: Multi-Chain Fusion (Optional)
            # =====================================================
            if self.config.enable_stage6_fusion and Stage6FusionGenerator:
                print("\n🔢 第6阶段：多链融合")
                print("-" * 50)

                # 检查是否有足够的 VQA items 进行融合
                if not result.vqa_items or len(result.vqa_items) < 2:
                    print("⚠️ 跳过 Stage 6：需要至少 2 个 VQA items 进行融合")
                else:
                    # 导入配置
                    from .stage6_fusion import Stage6FusionConfig
                    try:
                        from .stage6_fusion import FusionRule as FRule
                    except ImportError:
                        from stage6_fusion import FusionRule as FRule

                    # 创建Stage 6配置
                    fusion_rule = FRule.LLM_DECIDED
                    if self.config.fusion_rule.lower() == "llm_decided":
                        fusion_rule = FRule.LLM_DECIDED
                        print("[INFO] LLM will decide the fusion operation")
                    elif self.config.fusion_rule.lower() == "add":
                        fusion_rule = FRule.ADD
                    elif self.config.fusion_rule.lower() == "subtract":
                        fusion_rule = FRule.SUBTRACT
                    elif self.config.fusion_rule.lower() == "multiply":
                        fusion_rule = FRule.MULTIPLY
                    elif self.config.fusion_rule.lower() == "divide":
                        fusion_rule = FRule.DIVIDE
                    elif self.config.fusion_rule.lower() == "max":
                        fusion_rule = FRule.MAX
                    elif self.config.fusion_rule.lower() == "min":
                        fusion_rule = FRule.MIN
                    elif self.config.fusion_rule.lower() == "avg":
                        fusion_rule = FRule.AVG
                    elif self.config.fusion_rule.lower() == "conditional":
                        fusion_rule = FRule.CONDITIONAL
                    else:
                        fusion_rule = FRule.LLM_DECIDED

                    stage6_config = Stage6FusionConfig(
                        num_chains=self.config.fusion_num_chains,
                        primary_fusion_rule=fusion_rule
                    )

                    # 初始化Stage 6融合生成器
                    self.stage6_fusion = Stage6FusionGenerator(
                        config=stage6_config,
                        llm_client=self.llm_client
                    )

                    # 执行融合
                    fusion_result, fused_question = await self.stage6_fusion.generate_fusion(
                        vqa_items=result.vqa_items
                    )

                    # 保存融合结果
                    result.numerical_answer = fusion_result.final_answer
                    result.fusion_chains = [
                        {
                            "id": v.id,
                            "question": v.question,
                            "answer": v.answer,
                            "reasoning_path": v.reasoning_path,
                            "metadata": v.metadata
                        }
                        for v in fusion_result.original_vqas
                    ]
                    result.fusion_result = fusion_result.to_dict()

                    # 保存融合问题
                    if fused_question:
                        result.fused_question = fused_question.to_dict()

                    # 保存融合结果到文件
                    self.result_saver.save_stage6_results(fusion_result, fused_question)

                    self.statistics["stage6_fusion"] = fusion_result.to_dict()

                    print(f"\n   📊 多链融合完成:")
                    print(f"      - 融合规则: {fusion_result.fusion_rule.value}")
                    print(f"      - 中间值: {fusion_result.intermediate_values}")
                    print(f"      - 最终答案: {fusion_result.final_answer}")
                    if fused_question:
                        print(f"\n   📝 融合问题:")
                        print(f"      {fused_question.question[:100]}...")

            # =====================================================
            # Final: Save Results
            # =====================================================
            # 保存最终结果
            self.result_saver.save_final_results(result)

            # 打印摘要
            self.result_saver.print_summary(result)

            return result
        finally:
            # 清理资源：确保任何提前 return / 异常都能关闭 aiohttp session
            try:
                if hasattr(self, "stage3_evidence_chain_builder") and hasattr(self.stage3_evidence_chain_builder, "close"):
                    await self.stage3_evidence_chain_builder.close()
            except Exception:
                pass
            try:
                if hasattr(self, "search_client") and self.search_client:
                    await self.search_client.close()
            except Exception:
                pass
