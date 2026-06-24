"""VistaHop data synthesis pipeline."""

from .config import SynthesisConfig, SearchConfig
from .pipeline import SynthesisPipeline
from .result import SynthesisResult
from .stage1_extractor import Stage1Extractor
from .stage2_node_builder import Stage2NodeBuilder
from .stage3_evidence_chain_builder import Stage3EvidenceChainBuilder
from .stage4_question_builder import Stage4QuestionBuilder
from .stage5_vqa_generator import Stage5VQAGenerator

try:
    from .stage6_fusion import (
        FusedQuestion,
        FusionResult,
        FusionRule,
        Stage6FusionConfig,
        Stage6FusionGenerator,
        run_stage6_fusion,
    )
except ImportError:
    FusedQuestion = None
    FusionResult = None
    FusionRule = None
    Stage6FusionConfig = None
    Stage6FusionGenerator = None
    run_stage6_fusion = None

__all__ = [
    "SynthesisConfig",
    "SearchConfig",
    "SynthesisPipeline",
    "SynthesisResult",
    "Stage1Extractor",
    "Stage2NodeBuilder",
    "Stage3EvidenceChainBuilder",
    "Stage4QuestionBuilder",
    "Stage5VQAGenerator",
    "Stage6FusionConfig",
    "Stage6FusionGenerator",
    "FusionRule",
    "FusionResult",
    "FusedQuestion",
    "run_stage6_fusion",
]
