import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# 在所有其他代码之前加载环境变量
# 这确保了无论此文件是作为脚本直接运行还是作为模块导入，
# OPENAI_API_KEY 等配置都能被正确加载。
load_dotenv()



from .app import OutlineChapterList, Chapter, GenerateParams


# --- 2. 初始化 LangChain 组件 ---
llm = ChatOpenAI(
    model="ep-20250821165017-bzjpq",
    temperature=0.5,
)
output_parser = PydanticOutputParser(pydantic_object=OutlineChapterList)

# --- 3. 创建提示词模板 ---
prompt_template = PromptTemplate(
    template="""# 角色
你是应急预案编撰专家，擅长根据各种突发事件场景，生成逻辑清晰、结构合理的应急预案大纲。

## 技能：根据突发事件场景生成大纲
- 根据用户提供的突发事件场景和要求，生成一份详细的大纲。
- 大纲应包含主章节标题和子章节标题，体现预案逻辑结构。
- 所输出的大纲内容必须逻辑清晰、层次分明，符合应急预案编撰指南。

## 限制
- 大纲中的标题应简洁清晰，避免过于冗长和复杂的表述。
- 输出格式必须为 JSON，其值为一个对象数组。每个对象代表一个主章节，包含 `title` (主章节标题) 和 `sections` (二级标题字符串数组)。
- 子章节大纲深度不超过3层。
- 严格遵循下面的 JSON 格式输出。

## 示例
用户输入：无锡市惠山古镇景区发生踩踏事件，造成2人死亡
你的输出：
```json
{{
  "outline": [
    {{
      "title": "1 总则",
      "sections": ["1.1 编制目的", "1.2 编制依据", "1.3 工作原则", "1.4 适用范围", "1.5 旅游突发事件分类"]
    }},
    {{
      "title": "2 组织机构和职责",
      "sections": ["2.1 领导机构", "2.2 办事机构", "2.3 现场处置机构", "2.4 专家组"]
    }},
    {{
      "title": "3 预警",
      "sections": ["3.1 预警级别", "3.2 预警发布", "3.3 预警级别调整"]
    }},
    {{
      "title": "4 应急处置",
      "sections": ["4.1 信息报告与共享", "4.2 先期处置", "4.3 分级响应", "4.4 指挥与协调", "4.5 信息发布", "4.6 应急结束"]
    }},
    {{
      "title": "5 后期处置",
      "sections": ["5.1 善后工作", "5.2 保险理赔", "5.3 调查与评估"]
    }},
    {{
      "title": "6 应急保障",
      "sections": ["6.1 通信保障", "6.2 应急队伍保障", "6.3 紧急安置场所保障", "6.4 交通运输保障", "6.5 医疗卫生保障", "6.6 治安保障", "6.7 经费保障"]
    }},
    {{
      "title": "7 监督管理",
      "sections": ["7.1 宣传教育培训", "7.2 演练", "7.3 奖惩", "7.4 预案管理"]
    }}
  ]
}}
```

# 用户请求
请根据突发事件场景描述，撰写针对地级市层面的应急预案大纲。场景描述为：{scene}
""",
    input_variables=["scene"],
)

# --- 4. 构建并暴露 LangChain 链 ---
outline_generation_chain = prompt_template | llm | output_parser

async def stream_outline_with_llm(params: GenerateParams):
    """
    使用 LangChain 和大模型异步流式生成预案大纲。
    """
    input_data = {"scene": params.scene}
    # 使用 astream 方法进行流式调用
    async for chunk in outline_generation_chain.astream(input_data):
        yield chunk

async def generate_outline_with_llm(params: GenerateParams) -> OutlineChapterList:
    """
    使用 LangChain 和大模型异步生成预案大纲。
    """
    input_data = {"scene": params.scene}
    # result_chapter_list 是一个 Pydantic v1 版本的模型实例
    result_chapter_list: OutlineChapterList = await outline_generation_chain.ainvoke(input_data)
    
    # --- 版本兼容性修复 ---
    # 原始实现（在 Pydantic v1/v2 混用时可能导致 TypeError）:
    # outline_items = [
    #     OutlineItem(title=chapter.title, sections=chapter.sections)
    #     for chapter in result_chapter_list.outline
    # ]

    # 新实现：通过字典进行安全转换
    # 1. 先将 LangChain 输出的 Pydantic v1 模型转换为 Python 字典
    v1_model_dict = result_chapter_list.dict()
    
    # 2. 从字典创建 FastAPI 需要的 Pydantic v2 模型，确保兼容性
    outline_items = [
        Chapter(title=chapter['title'], sections=chapter['sections'])
        for chapter in v1_model_dict.get('outline', [])
    ]

    return OutlineChapterList(outline=outline_items)

# # --- 5. 用于独立测试的入口 ---
# if __name__ == "__main__":
#     import asyncio

#     async def main():
#         """本地异步测试函数"""
#         print("--- Running Outline Generation Test ---")
#         # 使用在文件顶部定义的导入
#         test_params = GenerateParams(
#             scene="广州市发生里氏 4.5 级地震，震源深度 10 公里，部分老旧房屋出现裂痕，有人员被困报告。"
#         )
        
#         try:
#             result = await generate_outline_with_llm(test_params)
#             print("✅ Generation successful!")
#             # 格式化输出以便阅读
#             for i, item in enumerate(result.outline):
#                 print(f"\n--- Chapter {i+1}: {item.title} ---")
#                 for section in item.sections:
#                     print(f"  - {section}")
#         except Exception as e:
#             print(f"❌ An error occurred: {e}")

#     # 运行异步测试
#     asyncio.run(main())
