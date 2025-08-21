import os
import json
import re
import jieba
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain.schema import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.output_parsers import BaseOutputParser

from rouge_chinese import Rouge
from tqdm import tqdm
import pandas as pd
# from langchain.chat_models import ChatOllama
# from langchain.retrievers import BM25Retriever


# 配置本地知识库路径
KB_DIR = "/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/plansDatabase0331"
# 配置文本切割数据库存储路径
CONTENT_DB_DIR = "/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/contentsave"
# 配置向量数据库存储路径
VECTOR_DB_DIR = "/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/embsave"
# 定义场景类型
SCENE_TYPES = ["convid", "fire", "food_security", "travel_accident", "typhoon", "earthquake", "traffic_accident", "pollution"]
# 定义场景匹配数据库
SCENE_MATCH_DB = {
    "convid": "plansDatabase4",
    "fire": "plansDatabase3",
    "food_security": "plansDatabase8",
    "travel_accident": "plansDatabase6",
    "typhoon": "plansDatabase1",
    "earthquake": "plansDatabase5",
    "traffic_accident": "plansDatabase2",
    "pollution": "plansDatabase7",
    "Unknown": ''
}


# 初始化 Ollama 嵌入和模型
embeddings = OllamaEmbeddings(model="bge-m3")
# 用于生成应急预案的大模型
llm = OllamaLLM(model="deepseek-r1", temperature=0.1, base_url="http://localhost:6006") # llama3.1 qwen2.5 gemma2 deepseek-r1 glm4
# llm = OllamaLLM(model="llama3.1", temperature=0.1, base_url="http://localhost:11434")
# 用于查询改写的新本地部署大模型，这里假设模型名为 new_model
query_rewrite_llm = OllamaLLM(model="glm4", base_url="http://localhost:6006") # 后续可以更换为更好的LLM
# query_rewrite_llm = OllamaLLM(model="glm4", base_url="http://localhost:11434")

def remove_think_tag(text):
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text

def determine_scene(scene_description):
    """
    根据场景描述判断场景类型
    :param scene_description: 场景描述
    :return: 场景类型
    """
    # SCENE_TYPES = ["convid", "fire", "food_security", "travel_accident", "typhoon", "earthquake", "traffic_accident", "pollution"]

    prompt = PromptTemplate(
        input_variables=["description", "scene_types"],
    #     template="根据以下场景描述，判断场景类型。场景描述为：{description}。注意直接输出场景类型，无需输出分析内容。场景类型只能从{scene_types}中选择。"
        template="根据以下场景描述，判断场景类型。场景描述为：{description}。注意直接输出主要场景类型，例如踩踏事件属于travel_acctdent，因暴雨导致地铁停运属于traffic_accident。无需输出分析内容。场景类型只能从{scene_types}中选择。"
    )

    # output_parser = StrOutputParser()
    # chain = LLMChain(llm=llm, prompt=prompt)
    # result = chain.run({"description": scene_description, "scene_types": SCENE_TYPES})


    chain = prompt | llm
    result = chain.invoke({"description": scene_description, "scene_types": SCENE_TYPES})
    result = remove_think_tag(result)
    for scene in SCENE_TYPES:
        if scene in result:
            return scene
    return 'Unknown'

def load_and_split_documents(scene):
    """
    加载并分割指定场景的文档
    :param scene: 场景名称
    :return: 分割后的文档列表
    """
    
    read_content_dir = os.path.join(CONTENT_DB_DIR, f'{scene}_content')
    load_content_dir = os.path.join(KB_DIR, f'{SCENE_MATCH_DB[scene]}')

    if not os.path.exists(os.path.join(read_content_dir)):

        loader = DirectoryLoader(load_content_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, # chunk_size：块的最大大小，其中大小由 length_function 确定。
            chunk_overlap=200, # 块之间的目标重叠。重叠块有助于减轻上下文在块之间划分时信息丢失。
            length_function=len, # 确定块大小的函数
            is_separator_regex=False, # 分隔符列表（默认为 ["\n\n", "\n", " ", ""]）是否应解释为正则表达式。
        )
        docs = text_splitter.split_documents(documents)
        # 元数据修改
        for doc in docs:
            doc.metadata['scene'] = scene
        
        # 切割后数据存储
        os.makedirs(read_content_dir, exist_ok=True)
        for i, doc in enumerate(docs):
            file_path = os.path.join(read_content_dir, f"doc_{i}.json")
            data = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
    else:
        docs = read_saved_docs(read_content_dir)

    return docs

def read_saved_docs(read_content_dir):
    """
    读取之前保存的文档
    :param scene: 场景名称
    :return: 文档列表
    """
    # save_dir = os.path.join("split_docs", read_content_dir)
    docs = []
    for file_name in os.listdir(read_content_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(read_content_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                doc = Document(page_content=data["page_content"], metadata=data["metadata"])
                docs.append(doc)
    return docs

def get_vectorstore(scene):
    """
    获取指定场景的向量数据库
    :param scene: 场景名称
    :return: 向量数据库对象
    """
    vector_db_path = os.path.join(VECTOR_DB_DIR, f"{scene}_index")
    if not os.path.exists(vector_db_path):
        docs = load_and_split_documents(scene)
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectorstore.save_local(os.path.join(VECTOR_DB_DIR, f"{scene}_index"))
        print(f"{scene}_index保存向量成功!")
    else:
        vectorstore = FAISS.load_local(os.path.join(VECTOR_DB_DIR, f"{scene}_index"), embeddings, allow_dangerous_deserialization=True)
        print(f"{scene}_index加载向量成功!")
    return vectorstore


# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """
    Output parser for a list of lines.
    """

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines)) # Remove empty lines

def query_multi_rewrite(query):
    """
    对查询进行多改写
    :param query: 原始查询
    :return: 改写后的查询列表
    """
    # prompt = PromptTemplate(
    #     input_variables=["query"],
    #     template="请对以下查询进行多种方式改写：{query}"
    # )
    # chain = LLMChain(llm=query_rewrite_llm, prompt=prompt)
    # result = chain.run(query)

    output_parser = LineListOutputParser()

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        你是一个人工智能语言助手。你的任务是对用户给出的问题进行转述，以便从向量数据库中检索相关文档。通过对用户问题生成多个不同角度的表述，克服基于距离的相似度搜索的局限性。请保留用户输入关键词，并将原问题和转述问题的额外三个问题用换行符分隔后直接给出，不要包含`原始问题`和`转述问题`字样。
        原始问题：{question}""",
    )
    chain = query_prompt | query_rewrite_llm | output_parser
    results = chain.invoke({"question": query})
    return results


def get_retrievers(scene):
    """
    获取指定场景的检索器
    :param scene: 场景名称
    :return: 集成检索器对象
    """

    docs = load_and_split_documents(scene)
    vectorstore = get_vectorstore(scene)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 1
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
    # reranker = CohereRerank()
    # ensemble_retriever = reranker.augment_retriever(ensemble_retriever)
    return ensemble_retriever


def get_emergency_knowledge(scene_description, query):
    """
    生成应急预案文本
    :param scene_description: 场景描述
    :param query: 用户查询
    :return: 生成的应急预案文本
    """
    # 意图识别
    scene = determine_scene(scene_description)
    print(f"场景类型为：{scene}")
    # 场景对应向量库匹配
    retriever = get_retrievers(scene)
    # query多条件改写
    rewritten_queries = query_multi_rewrite(query)
    all_results = []
    for q in rewritten_queries:
        results = retriever.invoke(q)
        all_results.extend(results)
    unique_results = list({doc.page_content: doc for doc in all_results}.values())
    
    prompt = PromptTemplate(
        input_variables=["documents", "query"],
        template="根据以下文档内容，回答问题：{query}\n{documents}"
    )
    # chain = LLMChain(llm=llm, prompt=prompt)
    chain = prompt | llm

    doc_content = "\n".join([doc.page_content for doc in unique_results])
    result = chain.invoke({"documents": doc_content, "query": query})
    return remove_think_tag(result), scene

def generate_plan_RAG(knowledge, scene_description, task):

    prompt = PromptTemplate(
        input_variables=["knowledge", "scene_description", "task"],
        template="你是应急预案编撰专家，请结合突发事件情景和参考文档，{task}，注意只需撰写给定部分，不要包括多余内容。\n情景为：{scene_description}\n参考文档为{knowledge}"
    )
    chain = prompt | llm
    result = chain.invoke({"knowledge": knowledge, "scene_description": scene_description, "task": task})
    return remove_think_tag(result)

def generate_plan_direct(scene_description, task):
    prompt = PromptTemplate(
        input_variables=["scene_description", "task"],
        template="你是应急预案编撰专家，请结合突发事件情景，{task}，注意只需撰写给定部分，不要包括多余内容。\n情景为：{scene_description}"
    )
    chain = prompt | llm
    result = chain.invoke({"scene_description": scene_description, "task": task})
    return remove_think_tag(result)

# 方便直接print和写出
def process_plans(text):
    text = text.replace('#', '')
    text = text.replace('*', '')
    text = text.replace('-', '')
    text = text.replace(' ', '').replace('.', '')
    text = text.replace('\n\n', '\n')
    text = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]', '', text)
    return text

# 服务于文本评估
def pro_content(text):
    text = text.replace('#', '')
    text = text.replace('*', '')
    text = text.replace('-', '')
    text = text.replace(' ', '').replace('.', '')
    text = text.replace('\n\n', '\n')
    text = re.sub(r'（\d+）', '', text)
    text = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩]', '', text)
    text = re.sub(r'[一二三四五六七八九十]', '', text)
    text = re.sub(r'\d+', '', text)
    words = jieba.cut(text)
    stopwords = ['#', '*', '\t', '-', '、', '(', ')', '\n']
    flitered = [word for word in words if word not in stopwords]
    text = ' '.join(flitered)
    return text

def compare_plans(plan_direct, plan_rag, plan_reference):
    rouge = Rouge()
    scores_rag = rouge.get_scores(plan_rag, plan_reference)
    scores_direct = rouge.get_scores(plan_direct, plan_reference)

    return scores_rag[0], scores_direct[0]
    
def launch_experiment(epoch, search_key, query, task):
    temp_result = {}
    scene_description = search_dict[search_key]['scene_description']
    reference_path = '/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/reference'
    task_path = task.replace("撰写", '')
    with open(os.path.join(reference_path, f'{task_path}', f"{search_key}.txt")) as f:
        plan_reference = f.read()
        plan_reference = pro_content(plan_reference)

    for e in tqdm(range(epoch)):
        plan_direct = generate_plan_direct(scene_description, task)
        plan_direct = process_plans(plan_direct)


        knowledge, scene = get_emergency_knowledge(scene_description, query)
        plan_rag = generate_plan_RAG(knowledge, scene_description, task)
        plan_rag = process_plans(plan_rag)

        scores_rag, scores_direct = compare_plans(pro_content(plan_direct), pro_content(plan_rag), plan_reference)
        temp_result[e] = {
            "scene": scene,
            "scores_rag": scores_rag,
            "scores_direct": scores_direct,
            "plan_direct": plan_direct,
            "plan_rag": plan_rag
        }
    return temp_result

def get_reference(task, event):
    reference_path = '/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/reference'
    task_path = task.replace("撰写", '')
    with open(os.path.join(reference_path, f'{task_path}', f"{event}.txt")) as f:
        plan_reference = f.read()
        plan_reference = pro_content(plan_reference)
    return plan_reference

def combine_actions(model, text):
    if model == 'deepseekr1':
        model = 'deepseek-r1'
    llm = OllamaLLM(model=model, temperature=0.1, base_url="http://localhost:6006") # llama3.1 qwen2.5 gemma2 deepseek-r1 glm4
    # llm = OllamaLLM(model=model, temperature=0.1, base_url="http://localhost:11434")

    prompt = PromptTemplate(
        input_variables=["text"],
        template="你是应急预案编撰专家，请完全根据参考文本，重新撰写应急响应预案，要求包括级别与启动条件、行动和措施、信息报告和发布以及调整与终止四部分，保证内容完整性和上下文逻辑清晰、语义连贯、避免中英混杂。参考文本为：{text}"
    )
    chain = prompt | llm
    result = chain.invoke({"text": text})
    if model == 'deepseek-r1':
        return process_plans(remove_think_tag(result))
    return process_plans(result)

def launch_sub_experiment():
    # 将应急响应的分任务组合，生成完整的应急响应文本
    rewrite_path = '/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/results/events_all_combined.xlsx'
    df = pd.read_excel(rewrite_path)

    for idx, grouped in df.groupby(['模型', '场景', '任务']):
        model, event, task = idx
        reference_plan = get_reference(task, event)

        for idx, row in grouped.iterrows():
            rag_plan_oraginal, direct_plan = row['rag_plan'], row['direct_plan']
            rag_plan_rewrite = combine_actions(model, rag_plan_oraginal)

            rouge = Rouge()
            scores_rag_oraginal = rouge.get_scores(rag_plan_oraginal, reference_plan)
            scores_rag_rewrite = rouge.get_scores(rag_plan_rewrite, reference_plan)
            scores_direct = rouge.get_scores(direct_plan, reference_plan)

def generate_app(scene_description):
    pass
    

    

if __name__ == "__main__":

    query_list = ['应急响应的内容是？', '应急保障的内容是？']
    task_list = ['撰写应急响应的级别与启动条件', '撰写应急响应的行动和措施', '撰写应急响应的信息报告和发布', '撰写应急响应的调整与终止', '撰写应急保障']
    scene_description_list = [
        '江苏省某养殖场附近村落出现不明疫情，发病症状与新冠相似，引发了当地村民的恐慌。',
        '江苏省某市居民区因电动车飞线充电引发火灾，现已造成46人死亡。',
        '江苏省某食品加工厂在生产过程中违规添加非食用物质，现已造成一人中毒死亡。',
        '江苏省某景区发生踩踏事件，人流量严重拥堵，预计疏散时间大于24小时。',
        '一场强大的台风正向江苏省逼近，气象台已发布台风橙色预警。',
        '江苏省某区域发生5.8级地震，时间为上午8:00，震中地区为一所实验小学。',
        '江苏省南京市地铁2号线因暴雨无法通车，造成交通瘫痪，预计经济损失2亿元。',
        '江苏省一沿湖化工企业发生了化学品泄漏事故，据企业上报，由于夏季高温催化，泄漏物已扩散至湖面。'
    ]

    search_dict = {
        "疫情": {"scene_description": "江苏省某养殖场附近村落出现不明疫情，发病症状与新冠相似，引发了当地村民的恐慌。"},
        "火灾": {"scene_description": "江苏省某市居民区因电动车飞线充电引发火灾，现已造成46人死亡。"},
        "食品安全": {"scene_description": "江苏省某食品加工厂在生产过程中违规添加非食用物质，现已造成一人中毒死亡。"},
        "文化旅游": {"scene_description": "江苏省某景区发生踩踏事件，人流量严重拥堵，预计疏散时间大于24小时。"},
        "台风": {"scene_description": "一场强大的台风正向江苏省逼近，气象台已发布台风橙色预警。"},
        "地震": {"scene_description": "江苏省某区域发生5.8级地震，时间为上午8:00，震中地区为一所实验小学。",},
        "交通事故": {"scene_description": "江苏省南京市地铁2号线因暴雨无法通车，造成交通瘫痪，预计经济损失2亿元。"},
        "环境污染": {"scene_description": "江苏省一沿湖化工企业发生了化学品泄漏事故，据企业上报，由于夏季高温催化，泄漏物已扩散至湖面。"}
    }

    result_dict = {
        "疫情": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
        "火灾": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
        "食品安全": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
        "文化旅游": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
        "台风": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
        "地震": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
        "交通事故": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
        "环境污染": {"query": {"应急响应的内容是？": {"task": {"撰写应急响应的级别与启动条件": {}, "撰写应急响应的行动和措施": {}, "撰写应急响应的信息报告和发布": {}, "撰写应急响应的调整与终止": {}}}, "应急保障的内容是？": {"task": {"撰写应急保障": {"result": {}}}}}},
    }




    # # ---------------------生成实验----------------------
    # scene_description = "江苏省某区域发生5.8级地震，时间为上午8:00，震中地区为一所实验小学。"
    # query = "应急响应的内容是？" # 应急保障的内容是？
    # task = "撰写应急响应的级别与启动条件" # 
    # search_key = "交通事故"  # 文化旅游 交通事故 
    # scene_description = search_dict[search_key]['scene_description']

    # reference_path = '/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/reference'
    # for query in query_list:
    #     if query == "应急响应的内容是？":
    #         for task in task_list:
    #             result_temp = launch_experiment(10, search_key, query, task)
    #             result_dict[search_key]["query"][query]["task"][task] = result_temp
    #     elif query == "应急保障的内容是？":
    #         task = "撰写应急保障"
    #         result_temp = launch_experiment(10, search_key, query, task)
    #         result_dict[search_key]["query"][query]["task"][task] = result_temp
    
    # generate_path = "/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/generated"
    # with open(os.path.join(generate_path, f"{search_key}.json"), 'w', encoding='utf-8') as f:
    #     json.dump(result_dict, f, ensure_ascii=False, indent=4)



    # # ---------------------直接生成----------------------
    # reference_path = '/Users/wangqian/codes/PythonCodes/硕士项目/AI开发/data/reference'
    # with open(os.path.join(reference_path, '应急响应的级别与启动条件', "疫情.txt")) as f:
    #     plan_reference = f.read()
    #     plan_reference = pro_content(plan_reference)

   
    # plan_direct = generate_plan_direct(scene_description, task)
    # print("直接生成的应急预案：")
    # print(process_plans(plan_direct))


    # knowledge, scene = get_emergency_knowledge(scene_description, query)
    # plan_rag = generate_plan_RAG(knowledge, scene_description, task)
    # print("RAG生成的应急预案：")
    # print(process_plans(plan_rag))


    # plan_direct = pro_content(plan_direct)
    # plan_rag = pro_content(plan_rag)
    # scores_rag, scores_direct = compare_plans(plan_direct, plan_rag, plan_reference)
    # print(f"RAG生成的应急预案分数：")
    # print(f"召回率：\n ROUGE-1: {scores_rag['rouge-1']['r']} ROUGE-2: {scores_rag['rouge-2']['r']} ROUGE-L: {scores_rag['rouge-l']['r']}")
    # print(f"精确度：\n ROUGE-1: {scores_rag['rouge-1']['p']} ROUGE-2: {scores_rag['rouge-2']['p']} ROUGE-L: {scores_rag['rouge-l']['p']}")
    # print(f"直接生成的应急预案分数：")
    # print(f"召回率：\n ROUGE-1: {scores_direct['rouge-1']['r']} ROUGE-2: {scores_direct['rouge-2']['r']} ROUGE-L: {scores_direct['rouge-l']['r']}")
    # print(f"精确度：\n ROUGE-1: {scores_direct['rouge-1']['p']} ROUGE-2: {scores_direct['rouge-2']['p']} ROUGE-L: {scores_direct['rouge-l']['p']}")
    
    # 重组与否的得分