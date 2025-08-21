import os
import json
import asyncio
import logging
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.pydantic_v1 import BaseModel, Field

# 导入 LangChain 的核心函数
from backend.OutlineGenerate import stream_outline_with_llm

# --- 1. 加载环境变量 ---
load_dotenv()

# --- 2. 定义数据模型 ---
class GenerateParams(BaseModel):
    """传递给 LangChain 链的参数模型 (Pydantic V1)"""
    scene: str = Field(..., description="突发事件场景描述")

class Chapter(BaseModel):
  """定义单个章节，包含主标题和二级标题列表。"""
  title: str = Field(description="主章节的标题")
  sections: List[str] = Field(description="该章节下的二级标题列表")

class OutlineChapterList(BaseModel):
  """用于接收 LLM 生成的、包含多个章节对象的列表。"""
  outline: List[Chapter] = Field(
    description="一个由章节对象组成的数组，每个对象都包含主标题和二级标题列表。"
  )


# --- 3. 初始化 FastAPI 应用 ---
app = FastAPI(title="Emergency Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. 定义核心业务逻辑 ---
async def stream_generator(params: GenerateParams):
    """
    流式生成器，调用 LangChain 并将结果实时推送给前端。
    """
    try:
        # 发送初始状态
        yield f"event: status_update\ndata: {json.dumps({'message': '正在连接大模型并撰写大纲...'}, ensure_ascii=False)}\n\n"

        # 调用核心逻辑流式生成大纲
        async for chunk in stream_outline_with_llm(params):
            if chunk:
                yield f"event: outline_chunk\ndata: {json.dumps(chunk.dict(), ensure_ascii=False)}\n\n"
        
        # 发送结束信号
        await asyncio.sleep(0.5)
        yield f"event: end\ndata: {json.dumps({'message': '大纲生成完毕'}, ensure_ascii=False)}\n\n"

    except Exception as e:
        logging.error(f"流式生成过程中发生错误: {e}", exc_info=True)
        error_message = f"生成失败: {str(e)}"
        yield f"event: error\ndata: {json.dumps({'error': error_message}, ensure_ascii=False)}\n\n"

# --- 5. 定义 API 接口 ---
@app.post("/api/generate")
async def generate_streaming_endpoint(api_params: GenerateParams = Body(...)):
    """
    处理流式生成请求的端点。
    """
    # **核心修复：将 FastAPI 的 Pydantic V2 模型转换为 LangChain 的 V1 模型**
    langchain_params = GenerateParams(**api_params.dict())

    return StreamingResponse(
        stream_generator(langchain_params),
        media_type="text/event-stream"
    )

# 用于非流式接口的占位符
@app.post("/api/outline", response_model=OutlineChapterList)
async def generate_outline_endpoint(api_params: GenerateParams):
    raise NotImplementedError("非流式接口待实现")
