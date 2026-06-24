"""
VistaHop管道结果模块

VistaHop: A Bottom-up Method for Generating Complex Multi-hop Reasoning Questions

结果类:
- SynthesisResult: VistaHop管道处理结果
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SynthesisResult:
    """VistaHop处理结果"""
    entities: List[str] = field(default_factory=list)
    evidence_chains: List[Dict[str, Any]] = field(default_factory=list)
    questions: List[Dict[str, Any]] = field(default_factory=list)
    vqa_items: List[Dict[str, Any]] = field(default_factory=list)  # Stage 5 VQA结果
    numerical_answer: Optional[float] = field(default=None)  # Stage 6 融合最终答案
    fusion_chains: List[Dict[str, Any]] = field(default_factory=list)  # Stage 6 融合链
    fusion_result: Optional[Dict[str, Any]] = field(default=None)  # Stage 6 完整融合结果
    fused_question: Optional[Dict[str, Any]] = field(default=None)  # Stage 6 融合问题
    statistics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
