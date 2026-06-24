#!/usr/bin/env python3
"""
_bbox_extractor - VistaHop本地实体提取模块

从图片中提取实体（BBX Extractor的本地版本）
"""

import re
import os
import asyncio
from typing import List, Dict, Any, Optional

try:
    from .llm_client import LLMClient
except ImportError:
    # 如果在独立模式下运行，提供一个 mock
    LLMClient = None


async def extract_bounding_boxes(
    llm_client,
    image_url: str,
    output_dir: str = None,
    model_name: str = None
) -> List[Dict]:
    """
    统一的实体提取接口

    从图片中提取实体名称列表。

    Args:
        llm_client: LLM客户端（支持视觉模型）
        image_url: 图片URL或本地文件路径
        output_dir: 输出目录
        model_name: 模型名称（可选）

    Returns:
        List[Dict]: 实体列表，每个包含 name 等信息
    """
    return await extract_bounding_boxes_no_resize(llm_client, image_url, output_dir)


async def extract_bounding_boxes_no_resize(
    llm_client,
    image_url: str,
    output_dir: str = None
) -> List[Dict]:
    """
    从图片中提取实体（支持本地图片文件）

    默认仅抽取实体名称。

    Args:
        llm_client: LLM客户端（支持视觉模型）
        image_url: 图片URL或本地文件路径
        output_dir: 输出目录

    Returns:
        List[Dict]: 实体列表
    """
    try:
        print(f"   🔍 提取实体（使用原始图片）...", flush=True)

        # 判断是本地文件还是网络URL
        is_local_file = False
        file_path = None

        if image_url.startswith('file://'):
            file_path = image_url[7:]
            is_local_file = True
        elif os.path.isabs(image_url) and os.path.exists(image_url):
            file_path = image_url
            is_local_file = True

        print(f"   图片{'路径' if is_local_file else 'URL'}: {image_url[:80]}...", flush=True)

        # 获取图片尺寸
        try:
            import requests
            from PIL import Image
            from io import BytesIO
            import base64

            print(f"   📥 {'读取本地文件' if is_local_file else '下载图片'}...", flush=True)

            if is_local_file:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"本地文件不存在: {file_path}")
                original_img = Image.open(file_path)
                print(f"   📁 本地文件: {file_path}", flush=True)
            else:
                img_response = requests.get(image_url, timeout=10, stream=True)
                img_response.raise_for_status()
                original_img = Image.open(BytesIO(img_response.content))

            original_img_width, original_img_height = original_img.size
            print(f"   📏 原始图片尺寸: {original_img_width}x{original_img_height}", flush=True)

            # 不裁剪，直接使用原始图片
            # 如果图片太大，进行等比例缩放
            max_dimension = 4096
            processed_img = original_img

            if original_img_width > max_dimension or original_img_height > max_dimension:
                # 等比例缩放
                ratio = min(max_dimension / original_img_width, max_dimension / original_img_height)
                new_width = int(original_img_width * ratio)
                new_height = int(original_img_height * ratio)
                processed_img = original_img.resize((new_width, new_height), Image.LANCZOS)
                processed_width, processed_height = new_width, new_height
                print(f"   📐 等比例缩放: {new_width}x{new_height}", flush=True)
            else:
                processed_width, processed_height = original_img_width, original_img_height

            # 强制转换到RGB
            processed_img = processed_img.convert('RGB') if processed_img.mode != 'RGB' else processed_img

            processed_width, processed_height = processed_img.size
            print(f"   ✅ 处理后尺寸: {processed_width}x{processed_height}", flush=True)

            # 转换为base64
            buffered = BytesIO()
            processed_img.save(buffered, format="JPEG", quality=95)
            processed_img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            processed_image_url = f"data:image/jpeg;base64,{processed_img_base64}"

            print(f"   ✅ 图片已转换为base64格式", flush=True)

        except Exception as e:
            print(f"   ⚠️ 无法读取图片: {e}，使用原始URL", flush=True)
            processed_image_url = image_url
            original_img_width, original_img_height = 6000, 3000
            processed_width, processed_height = 1000, 1000

        # 构建prompt
        size_info = f"This image is {processed_width}×{processed_height} pixels."
        coord_range = f"Coordinates should be in pixel values for this {processed_width}×{processed_height} image."

        # 第一次推理：提取实体名称列表
        entity_list_prompt = f"""Analyze this image and identify distinctive, specific NAMED ENTITIES that can be VISUALIZED in images.

⚠️ IMPORTANT: {size_info}

CRITERIA:
- Must be a SPECIFIC named entity (person's name, brand name, product name, landmark name, road/street name, etc.)
- Must be something you could take a photo of
- Must be visually distinctive in this image

EXAMPLES OF GOOD ENTITIES:
- Albert Einstein
- Toyota
- Eiffel Tower
- iPhone
- Statue of Liberty
- Hollywood Boulevard
- Fifth Avenue
- Route 66

EXAMPLES OF BAD ENTITIES (AVOID):
- Person (use specific name if known)
- Car (use brand/model if visible)
- Building (use specific name if known)
- Tree
- Generic object

OUTPUT FORMAT:
Return a list of specific named entities, one per line.
Example:
Albert Einstein
Toyota Camry
Eiffel Tower
Coca Cola
Port of Callao

List as many distinct named entities as possible. Prioritize unique, recognizable entities."""

        print(f"   🔍 [Pass 1] 提取实体名称列表...", flush=True)
        entity_list_response = await llm_client.generate_with_images(
            prompt=entity_list_prompt,
            image_urls=[processed_image_url],
            temperature=0.1,
            max_tokens=4096
        )

        # 解析实体名称列表
        entity_names = []
        try:
            cleaned_list = entity_list_response.strip()
            cleaned_list = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_list, flags=re.MULTILINE)
            cleaned_list = re.sub(r'\n?```\s*$', '', cleaned_list, flags=re.MULTILINE)
            cleaned_list = cleaned_list.strip()

            for line in cleaned_list.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # 清理前缀
                name = re.sub(r'^\d+[\.\)]\s*', '', line)
                name = re.sub(r'^[-•]\s*', '', name)
                name = name.strip()

                if len(name) >= 2 and name.isprintable():
                    entity_names.append(name)

            print(f"   ✅ [Pass 1] 找到 {len(entity_names)} 个实体名称", flush=True)

        except Exception as e:
            print(f"   ⚠️ [Pass 1] 解析实体名称失败: {e}", flush=True)
            return []

        if not entity_names:
            print(f"   ⚠️ 未找到任何实体", flush=True)
            return []

        # 第二次推理：为每个实体生成视觉描述
        # 为了节省token，我们将所有实体名称发送给LLM，让它一次性生成所有描述
        entities_str = "\n".join([f"- {name}" for name in entity_names])

        description_prompt = f"""Analyze this image and provide a brief visual description for each entity listed below.
The description should explain how each specific named entity appears in THIS specific image.

⚠️ IMPORTANT: {size_info}

ENTITY LIST:
{entities_str}

OUTPUT FORMAT:
For each entity, output on a separate line with the description:
Entity Name | Brief visual description of how this entity appears in the image

Example:
Eiffel Tower | Large iron lattice tower in the center background
Toyota Camry | White sedan with "CAMRY" badge on the rear
Coca Cola | Red and white logo on a vending machine

Be specific to THIS image. If an entity is not clearly visible, state "Not clearly visible"."""

        print(f"   🔍 [Pass 2] 生成实体视觉描述...", flush=True)
        grounding_response = await llm_client.generate_with_images(
            prompt=description_prompt,
            image_urls=[processed_image_url],
            temperature=0.1,
            max_tokens=8192
        )
        print(f"   ✅ [Pass 2] 描述生成完成", flush=True)

        # 解析第二次推理的响应 (Entity Name | Description 格式)
        boxes = []
        try:
            cleaned_response = grounding_response.strip()
            cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.MULTILINE)
            cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response, flags=re.MULTILINE)
            cleaned_response = cleaned_response.strip()

            lines = cleaned_response.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 尝试分割实体名称和描述
                if '|' in line:
                    parts = line.split('|', 1)
                    entity_name = parts[0].strip()
                    entity_desc = parts[1].strip() if len(parts) > 1 else ""
                else:
                    # 如果没有 |，则尝试清理旧格式
                    entity_name = line
                    entity_desc = ""

                # 清理实体名称
                entity_name = re.sub(r'^\d+[\.\)]\s*', '', entity_name)
                entity_name = re.sub(r'^[-•]\s*', '', entity_name)
                entity_name = entity_name.strip()
                entity_name = re.sub(r'\s*\[[^\]]+\]\s*', '', entity_name)
                if ':' in entity_name:
                    entity_name = entity_name.split(':')[0].strip()

                # 过滤掉不在原始列表中的实体（可选）
                # 如果需要严格匹配，可以取消下面的注释
                # if entity_name not in entity_names:
                #     continue

                if len(entity_name) >= 2 and entity_name.isprintable():
                    box_info = {
                        "name": entity_name,
                        "description": entity_desc,
                        "processed_size": {"width": processed_width, "height": processed_height},
                        "original_size": {"width": original_img_width, "height": original_img_height}
                    }
                    boxes.append(box_info)
                    if entity_desc:
                        print(f"   ✅ 解析到实体: {entity_name} | {entity_desc[:50]}...", flush=True)
                    else:
                        print(f"   ✅ 解析到实体: {entity_name}", flush=True)

            if boxes:
                print(f"   ✅ 成功解析 {len(boxes)} 个实体", flush=True)

        except Exception as e:
            print(f"   ⚠️ 解析失败: {e}", flush=True)
            boxes = []

        if not boxes:
            print(f"   ⚠️ 未找到实体", flush=True)
            return []

        print(f"   ✅ 找到 {len(boxes)} 个实体", flush=True)

        # 验证和过滤实体
        valid_boxes = []
        seen_names = set()
        for box in boxes:
            name = box.get("name", "").strip()

            if not name or len(name) < 2:
                continue

            if name.lower() in seen_names:
                continue

            seen_names.add(name.lower())
            valid_boxes.append(box)

        return valid_boxes

    except Exception as e:
        print(f"   ❌ 提取实体失败: {e}")
        import traceback
        traceback.print_exc()
        return []
