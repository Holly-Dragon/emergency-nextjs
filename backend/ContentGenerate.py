import os
from typing import List, Dict, Any, AsyncGenerator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import json

# --- 1. 加载环境变量 ---
load_dotenv()

# --- 2. 定义数据模型 ---
# 注意：这些模型应与 OutlineGenerate.py 中的定义保持一致，以便模块间交互。
class Chapter(BaseModel):
  """定义单个章节，包含主标题和二级标题列表。"""
  title: str = Field(description="主章节的标题")
  sections: List[str] = Field(description="该章节下的二级标题列表")

class OutlineData(BaseModel):
  """包含多个章节对象的列表。"""
  outline: List[Chapter] = Field(
    description="一个由章节对象组成的数组，每个对象都包含主标题和二级标题列表。"
  )

class ContentGenerateParams(BaseModel):
    """传递给内容生成函数的参数模型"""
    scene: str = Field(..., description="突发事件场景描述")
    outline: OutlineData = Field(..., description="应急预案的整体大纲")
    # references: str = Field("", description="相关的参考资料") # 可选的参考资料

# --- 3. 初始化 LangChain 组件 ---
# 系统提示词
SYSTEM_PROMPT = """# 角色
你是应急预案编撰专家。具备深厚的应急预案知识和突发事件处理经验，能够根据突发事件场景编撰实践性强、有追责效力的应急预案文本。

## 技能：根据主题扩写段落
- 你负责预案的扩写工作，即根据预案大纲的某一部分，扩写预案，在大纲小标题的基础上补全对应的内容段落。
- 用户会提供突发事件的场景描述、预案的整体大纲、你正在写的部分、你已经写好的上一个段落，供你理解和学习。
- 扩写前应仔细分析突发事件场景和整体大纲，确保预案编撰的内容真实可信，具备切实可行性。
- 扩写时保证预案语言严谨官方，保持内容连贯、衔接自然。
- 所输出的段落内容必须结构合理，条理清晰。

## 限制
- 使用 Markdown 格式回复。
- 生成的段落需严格符合用户给定的突发事件、大纲要求。 不需要输出与预案内容无关的描述。
- 内容的长度应与大纲层级相适应，通常为300-500字左右，但可以根据突发事件的复杂程度和重要性进行适当调整。
"""

# 用户提示词模板
USER_PROMPT_TEMPLATE = """请根据 {subtitle} 这个主题生成一段预案文本。

以下信息供你参考：
预案处理的突发事件：{scene}
预案的整体大纲：{outline_str}
你正在写的部分： {subtitle}
你已经写好的上一个段落：{last_paragraph}
"""
# 相关参考资料：{references} # 如有需要可添加

llm = ChatOpenAI(
    model="ep-20250821165017-bzjpq",
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_PROMPT_TEMPLATE)
])

output_parser = StrOutputParser()

# --- 4. 构建并暴露 LangChain 链 ---
content_generation_chain = prompt | llm | output_parser

# --- 5. 核心业务逻辑 ---
async def stream_content_with_llm(params: ContentGenerateParams) -> AsyncGenerator[Dict[str, Any], None]:
    """
    异步流式生成完整的应急预案内容。
    遍历大纲，逐个生成章节和小节内容，并以结构化的字典形式流式返回。
    """
    last_paragraph = "无"
    outline_str = json.dumps([chapter.dict() for chapter in params.outline.outline], ensure_ascii=False, indent=2)

    for chapter in params.outline.outline:
        # 流式输出主章节标题
        chapter_title_md = f"## {chapter.title}\n\n"
        yield {"event": "chapter_start", "data": {"title": chapter.title, "title_md": chapter_title_md}}
        last_paragraph = chapter_title_md

        for section in chapter.sections:
            # 流式输出子章节标题
            section_title_md = f"### {section}\n\n"
            yield {"event": "section_start", "data": {"title": section, "title_md": section_title_md}}
            
            input_data = {
                "scene": params.scene,
                "outline_str": outline_str,
                "subtitle": f"{chapter.title} - {section}",
                "last_paragraph": last_paragraph,
                # "references": params.references
            }

            # 流式调用LLM生成段落内容
            current_paragraph = ""
            async for chunk in content_generation_chain.astream(input_data):
                yield {"event": "content_chunk", "data": chunk}
                current_paragraph += chunk
            
            # 添加换行符以分隔段落
            yield {"event": "content_chunk", "data": "\n\n"}
            last_paragraph = section_title_md + current_paragraph
            # 发送小节结束信号
            yield {"event": "section_end", "data": {"title": section}}

async def generate_content_with_llm(params: ContentGenerateParams) -> str:
    """
    异步一次性生成完整的应急预案内容。
    遍历大纲，逐个生成章节和小节内容，最后返回完整的字符串。
    """
    full_content = ""
    last_paragraph = "无"
    outline_str = json.dumps([chapter.dict() for chapter in params.outline.outline], ensure_ascii=False, indent=2)

    for chapter in params.outline.outline:
        chapter_title_md = f"## {chapter.title}\n\n"
        full_content += chapter_title_md
        last_paragraph = chapter_title_md

        for section in chapter.sections:
            section_title_md = f"### {section}\n\n"
            # full_content += section_title_md
            
            input_data = {
                "scene": params.scene,
                "outline_str": outline_str,
                "subtitle": f"{chapter.title} - {section}",
                "last_paragraph": last_paragraph,
                # "references": params.references
            }

            # 一次性调用LLM生成段落内容
            paragraph_content = await content_generation_chain.ainvoke(input_data)
            
            full_content += paragraph_content
            full_content += "\n\n"
            
            last_paragraph = section_title_md + paragraph_content
            
    return full_content


# --- 6. 用于独立测试的入口 ---
if __name__ == "__main__":
    import asyncio

    async def main():
        """本地异步测试函数"""
        print("--- Running Content Generation Test ---")
        
        # 假设这是从 OutlineGenerate.py 获得的大纲
        test_outline_data = {
            "outline": [
                {"title": "1 总则", "sections": ["1.1 编制目的", "1.2 编制依据"]},
                {"title": "2 组织机构和职责", "sections": ["2.1 领导机构", "2.2 办事机构"]}
            ]
        }

        test_params = ContentGenerateParams(
            scene="广州市发生里氏 4.5 级地震，震源深度 10 公里，部分老旧房屋出现裂痕，有人员被困报告。",
            outline=OutlineData.parse_obj(test_outline_data)
        )
        
        try:
            print(f"--- Generating content for scene: {test_params.scene} ---\n")
            full_content = await generate_content_with_llm(test_params)
            # full_content = ""
            # async for chunk in stream_content_with_llm(test_params):
            #     print(chunk, end="", flush=True)
            #     full_content += chunk
            
            print("\n\n--- ✅ Generation successful! ---")
            # print("\n--- Full Content ---")
            print(full_content)

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

    # 运行异步测试
    asyncio.run(main())