import os
import json
import asyncio
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 导入两个核心生成器和它们的数据模型
from backend.OutlineGenerate import (
    stream_outline_with_llm, 
    GenerateParams as OutlineGenerateParams,
    OutlineChapterList
)
from backend.ContentGenerate import (
    stream_content_with_llm,
    ContentGenerateParams,
    OutlineData as ContentOutlineData,
    Chapter as ContentChapter
)

# --- 1. 加载环境变量 ---
load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- 2. FastAPI 应用设置 ---
app = FastAPI(title="Emergency Backend", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. 核心业务逻辑 ---
async def stream_generator(params: OutlineGenerateParams):
    """
    重构后的流式生成器，分两个阶段：
    1. 流式生成大纲。
    2. 收集完整大纲后，流式生成详细内容。
    """
    full_outline_obj = None
    
    try:
        # --- 阶段 1: 生成大纲 ---
        logging.info("阶段 1: 开始生成大纲...")
        yield f"event: status_update\ndata: {json.dumps({'message': '正在连接大模型并撰写大纲...'}, ensure_ascii=False)}\n\n"
        print(f"event: status_update\ndata: {json.dumps({'message': '正在连接大模型并撰写大纲...'}, ensure_ascii=False)}\n\n")

        final_outline_chunk = None
        async for chunk in stream_outline_with_llm(params):
            if chunk:
                final_outline_chunk = chunk
                # 实时将部分大纲数据块发送给前端
                yield f"event: outline_chunk\ndata: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
                print(f"{json.dumps(chunk.model_dump(), ensure_ascii=False)}")
        
        if final_outline_chunk:
            full_outline_obj = final_outline_chunk
            logging.info("大纲生成成功并解析完毕。")
            yield f"event: outline_finished\ndata: {json.dumps(full_outline_obj.model_dump(), ensure_ascii=False)}\n\n"
            print(f"event: outline_finished\ndata: {json.dumps(full_outline_obj.model_dump(), ensure_ascii=False)}\n\n")
        else:
            raise Exception("大纲生成未能返回任何数据。")

        # --- 阶段 2: 生成内容 ---
        logging.info("阶段 2: 开始生成详细内容...")
        yield f"event: status_update\ndata: {json.dumps({'message': '大纲生成完毕，正在生成详细内容...'}, ensure_ascii=False)}\n\n"
        print(f"event: status_update\ndata: {json.dumps({'message': '大纲生成完毕，正在生成详细内容...'}, ensure_ascii=False)}\n\n")
        
        content_params = ContentGenerateParams(
            scene=params.scene,
            outline=ContentOutlineData(
                outline=[ContentChapter(**chapter.dict()) for chapter in full_outline_obj.outline]
            )
        )

        full_content = ""
        async for content_chunk in stream_content_with_llm(content_params):
            event = content_chunk.get("event")
            data = content_chunk.get("data")
            if event and data is not None:
                json_data = json.dumps(data, ensure_ascii=False)
                yield f"event: {event}\ndata: {json_data}\n\n"
                print(f"{json_data}")
                
                if event == 'content_chunk':
                    full_content += data
                elif event in ['chapter_start', 'section_start']:
                    full_content += data.get('title_md', '')

        # --- 结束 ---
        logging.info("全部内容生成完毕。")
        await asyncio.sleep(0.5)
        final_data = {
            "message": "预案已生成！",
            "full_content": full_content,
            "full_outline": full_outline_obj.dict()
        }
        yield f"event: full_content_finished\ndata: {json.dumps(final_data, ensure_ascii=False)}\n\n"

    except Exception as e:
        logging.error(f"流式生成过程中发生错误: {e}", exc_info=True)
        error_message = f"生成失败: {str(e)}"
        yield f"event: error\ndata: {json.dumps({'error': error_message}, ensure_ascii=False)}\n\n"

# --- 4. 定义 API 接口 ---
@app.post("/api/generate")
async def generate_streaming_endpoint(api_params: OutlineGenerateParams = Body(...)):
    """
    处理流式生成请求的端点。
    """
    return StreamingResponse(
        stream_generator(api_params),
        media_type="text/event-stream"
    )
