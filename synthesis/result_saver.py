"""
结果保存器 (Result Saver)

负责保存中间结果、最终结果和导出训练格式
"""

import os
import json
from datetime import datetime
from typing import Any, Dict

from .result import SynthesisResult


class ResultSaver:
    """
    结果保存器

    负责所有结果保存相关操作
    """

    def __init__(self, config, statistics: Dict[str, Any]):
        """
        初始化

        Args:
            config: VistaHop配置
            statistics: 统计信息字典
        """
        self.config = config
        self.statistics = statistics

    def save_intermediate(self, filename: str, data: Any):
        """
        保存中间结果

        Args:
            filename: 文件名
            data: 要保存的数据
        """
        filepath = os.path.join(self.config.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"   💾 已保存中间结果: {filepath}")

    def save_final_results(self, result: SynthesisResult):
        """
        保存最终结果

        Args:
            result: SynthesisResult结果对象
        """
        # 保存完整结果
        filepath = os.path.join(self.config.output_dir, "synthesis_final_results.json")

        statistics = {**self.statistics, **result.statistics}
        final_data = {
            "generated_at": datetime.now().isoformat(),
            "statistics": statistics,
            "entities": result.entities,
            "evidence_chains_count": len(result.evidence_chains),
            "evidence_chains": result.evidence_chains,
            "questions_count": len(result.questions),
            "questions": result.questions,
            "vqa_items_count": len(result.vqa_items),
            "vqa_items": result.vqa_items,
            "numerical_answer": result.numerical_answer,
            "fusion_chains": result.fusion_chains,
            "fusion_result": result.fusion_result,
            "fused_question": result.fused_question,
            "errors": result.errors
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 最终结果已保存: {filepath}")

        # 导出为训练格式 (JSONL)
        self.export_training_format(result)

    def export_training_format(self, result: SynthesisResult):
        """
        导出为训练格式

        Args:
            result: SynthesisResult结果对象
        """
        filepath = os.path.join(self.config.output_dir, "training_data.jsonl")

        with open(filepath, 'w', encoding='utf-8') as f:
            for q in result.questions:
                training_sample = {
                    "id": q.get("id", ""),
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "constraints": q.get("constraints", []),
                    "reasoning_path": q.get("reasoning_path", []),
                    "evidence_chain": q.get("evidence_chain", {}),
                    "difficulty_score": q.get("difficulty_score", 0.0),
                    "uniqueness_score": q.get("uniqueness_score", 0.0)
                }
                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')

        print(f"💾 训练数据已导出: {filepath}")

    def print_summary(self, result: SynthesisResult):
        """
        打印结果摘要

        Args:
            result: SynthesisResult结果对象
        """
        print("\n" + "=" * 70)
        print("📊 VistaHop管道执行完成 - 结果摘要")
        print("=" * 70)

        print(f"\n📌 实体统计:")
        print(f"   - 总实体数: {len(result.entities)}")

        print(f"\n🔗 证据链统计:")
        print(f"   - 总链数: {len(result.evidence_chains)}")

        print(f"\n❓ 问题统计:")
        print(f"   - 总问题数: {len(result.questions)}")

        # 计算平均分数
        if result.questions:
            avg_difficulty = sum(
                q.get("difficulty_score", 0) for q in result.questions
            ) / len(result.questions)
            avg_uniqueness = sum(
                q.get("uniqueness_score", 0) for q in result.questions
            ) / len(result.questions)

            print(f"\n📈 质量指标:")
            print(f"   - 平均难度分数: {avg_difficulty:.3f}")
            print(f"   - 平均唯一性分数: {avg_uniqueness:.3f}")

        print(f"\n💾 输出目录: {self.config.output_dir}")

        if result.errors:
            print(f"\n⚠️ 错误信息:")
            for error in result.errors[:5]:
                print(f"   - {error}")

    def save_stage6_results(self, fusion_result, fused_question):
        """
        保存 Stage 6 融合结果

        Args:
            fusion_result: 融合结果对象 (FusionResult)
            fused_question: 融合问题对象 (FusedQuestion)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存融合结果详情
        fusion_data = fusion_result.to_dict()
        fusion_path = os.path.join(self.config.output_dir, f"stage6_fusion_result_{timestamp}.json")
        with open(fusion_path, 'w', encoding='utf-8') as f:
            json.dump(fusion_data, f, ensure_ascii=False, indent=2)
        print(f"   💾 已保存融合结果: {fusion_path}")

        # 保存融合问题
        if fused_question:
            question_data = fused_question.to_dict()
            question_path = os.path.join(self.config.output_dir, f"stage6_fused_question_{timestamp}.json")
            with open(question_path, 'w', encoding='utf-8') as f:
                json.dump(question_data, f, ensure_ascii=False, indent=2)
            print(f"   💾 已保存融合问题: {question_path}")
