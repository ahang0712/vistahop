#!/usr/bin/env python3
"""Command line entry point for the VistaHop synthesis pipeline."""

import argparse
import asyncio
import os
from datetime import datetime

from .config import SynthesisConfig
from .pipeline import SynthesisPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VistaHop data synthesis pipeline (stage1-stage6)")

    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""), help="LLM API key")
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"), help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4.1"), help="LLM model name")

    parser.add_argument("--entities", nargs="+", default=[], help="Seed entities. If omitted, --image-url is used for stage1 extraction.")
    parser.add_argument("--image-url", default="", help="Image URL or local image path for entity extraction and VQA grounding")
    parser.add_argument("--first-entity-only", action="store_true", help="Only build chains for the first extracted entity")
    parser.add_argument("--num-entities-for-chains", type=int, default=0, help="Limit how many entities are used for chain construction")

    parser.add_argument("--chains-per-entity", type=int, default=1)
    parser.add_argument("--questions-per-chain", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-stage", type=int, default=6, choices=[1, 2, 3, 4, 5, 6])

    parser.add_argument("--output-dir", default="./outputs/synthesis", help="Output directory prefix")
    parser.add_argument("--no-intermediate", action="store_true", help="Do not save stage intermediate files")

    parser.add_argument("--search-engine", default=os.getenv("VISTAHOP_SEARCH_ENGINE", "wikipedia"), choices=["wikipedia", "serper", "bing", "serpapi"])
    parser.add_argument("--search-api-key", default=os.getenv("SERPER_API_KEY", "") or os.getenv("SERPAPI_KEY", ""))
    parser.add_argument("--proxy", default=os.getenv("HTTPS_PROXY", "") or os.getenv("HTTP_PROXY", ""))
    parser.add_argument("--wikipedia-mirror", default="wikimedia", choices=["wikimedia", "zh", "en"])
    parser.add_argument("--wikidata-mirror", default="wikimedia", choices=["wikimedia", "official"])

    parser.add_argument("--generate-vqa", action="store_true", help="Enable stage5 VQA generation")
    parser.add_argument("--serpapi-key", default=os.getenv("SERPAPI_KEY", ""), help="SerpAPI key for image search")
    parser.add_argument("--vqa-images", type=int, default=2)
    parser.add_argument("--no-simplify", action="store_true", help="Disable VQA question simplification")

    parser.add_argument("--enable-stage6-fusion", action="store_true", help="Enable stage6 multi-chain fusion")
    parser.add_argument("--fusion-num-chains", type=int, default=3)
    parser.add_argument(
        "--fusion-rule",
        default="llm_decided",
        choices=["llm_decided", "add", "subtract", "multiply", "divide", "max", "min", "avg", "conditional"],
    )
    parser.add_argument("--no-llm-fusion", action="store_true", help="Use rule-based fusion instead of LLM-decided fusion")
    return parser


async def main() -> None:
    args = build_parser().parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.rstrip("/")
    if not output_dir.endswith(timestamp):
        output_dir = f"{output_dir}_{timestamp}"

    config = SynthesisConfig(
        llm_api_key=args.api_key,
        llm_base_url=args.base_url,
        llm_model=args.model,
        search_engine=args.search_engine,
        search_api_key=args.search_api_key,
        proxy=args.proxy,
        wikipedia_mirror=args.wikipedia_mirror,
        wikidata_mirror=args.wikidata_mirror,
        image_url=args.image_url,
        chains_per_entity=args.chains_per_entity,
        questions_per_chain=args.questions_per_chain,
        max_chain_depth=args.max_depth,
        output_dir=output_dir,
        save_intermediate=not args.no_intermediate,
        first_entity_only=args.first_entity_only,
        num_entities_for_chains=0 if args.first_entity_only else args.num_entities_for_chains,
        generate_vqa=args.generate_vqa,
        serpapi_key=args.serpapi_key,
        vqa_images_per_entity=args.vqa_images,
        enable_question_simplify=not args.no_simplify,
        max_stage=args.max_stage,
        enable_stage6_fusion=args.enable_stage6_fusion,
        fusion_num_chains=args.fusion_num_chains,
        fusion_rule=args.fusion_rule,
        enable_llm_fusion=not args.no_llm_fusion,
    )

    pipeline = SynthesisPipeline(config)
    result = await pipeline.run(
        entities=args.entities or None,
        image_url=args.image_url or None,
    )

    print("\nVistaHop synthesis complete")
    print(f"Output directory: {output_dir}")
    print(f"Questions: {len(result.questions)}")
    print(f"VQA items: {len(result.vqa_items)}")
    if result.fused_question:
        print("Stage6 fused question generated")


if __name__ == "__main__":
    asyncio.run(main())
