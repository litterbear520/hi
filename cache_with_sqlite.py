#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能对话系统 - 集成向量数据库缓存和在线搜索功能

特点：
1. 使用SQLite存储问答对和向量嵌入
2. 使用FAISS进行向量相似度搜索
3. 集成Tavily在线搜索工具
4. 基于LangGraph实现工作流图
5. 实现思考-观察-行动模式的搜索代理
"""

# 标准库导入
import os
import re
import uuid
import sqlite3
import time
import json
from typing import List, Dict, Any, Optional, Tuple

# 第三方库导入
import numpy as np
import dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults
from langgraph.graph import START, MessagesState, StateGraph

#------------------------------------------------------------------------------
# 配置部分
#------------------------------------------------------------------------------

# 加载环境变量
dotenv.load_dotenv()

# 系统配置
DATA_DIR = "data"  # 数据存储目录
DB_PATH = os.path.join(DATA_DIR, "qa_cache.db")  # SQLite数据库路径
SIMILARITY_THRESHOLD = 0.7  # 缓存命中阈值（余弦相似度）
SHORT_TEXT_LENGTH = 5  # 短文本定义（字符数）
SHORT_TEXT_THRESHOLD = 0.1  # 短文本相似度阈值

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)

# 搜索代理的提示词模板
SEARCH_AGENT_PROMPT = """你是一个智能助手，配备了搜索工具来获取最新信息。请按照以下步骤回答问题：

1. 思考：分析用户的问题，判断是否需要搜索外部信息。考虑以下因素：
   - 问题是否涉及实时性数据（如新闻、当前事件、实时行情等）
   - 问题是否涉及专业知识或需要最新的信息支持
   - 问题是否超出了你的知识范围

2. 决策：如果需要搜索，请明确指出你将搜索什么关键词。如果不需要搜索，请直接回答。

3. 搜索：如果需要搜索，你将使用搜索工具获取信息。

4. 分析：分析搜索结果，提取相关信息。

5. 回答：基于搜索结果和你的知识，给出全面、准确的回答。在回答中引用搜索到的信息源。

注意：如果用户的问题是简单的问候、问好或者是不需要外部信息的问题，请直接回答，不要使用搜索工具。
"""

#------------------------------------------------------------------------------
# 模型和工具初始化
#------------------------------------------------------------------------------

# 初始化大模型
model = ChatTongyi(
    model="qwen3-235b-a22b",
)

# 初始化嵌入模型
embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), 
    model="text-embedding-v2"  # 使用阿里云千问的文本嵌入模型
)

# 初始化搜索工具
search_tool = TavilySearchResults(
    max_results=3,
)

#------------------------------------------------------------------------------
# 数据库和向量存储
#------------------------------------------------------------------------------

def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度"""
    # 确保向量是numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 计算余弦相似度: cos(θ) = (A·B)/(|A|·|B|)
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    # 避免除以零
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)


def init_sqlite_db():
    """初始化SQLite数据库结构"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建问答表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS qa_pairs (
        id TEXT PRIMARY KEY,
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 创建向量表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS vector_embeddings (
        id TEXT PRIMARY KEY,
        question_id TEXT NOT NULL,
        vector BLOB NOT NULL,
        FOREIGN KEY (question_id) REFERENCES qa_pairs(id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("SQLite数据库初始化完成")


def init_faiss_index():
    """初始化FAISS向量索引
    
    创建一个初始文档，因为FAISS不能使用空列表初始化
    """
    initial_texts = ["初始化文档"]
    vector_db = FAISS.from_texts(
        texts=initial_texts, 
        embedding=embeddings,
        metadatas=[{"is_init": True, "answer": ""}]
    )
    print("创建了新的FAISS索引（使用初始化文档）")
    return vector_db


def load_qa_from_sqlite_to_faiss():
    """从SQLite数据库加载问答对到FAISS向量索引"""
    global vector_db
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 获取所有问答对
    cursor.execute("SELECT id, question, answer FROM qa_pairs")
    qa_pairs = cursor.fetchall()
    
    if not qa_pairs:
        print("SQLite数据库中没有问答对")
        conn.close()
        return
    
    # 重新创建FAISS索引（包含初始化文档）
    vector_db = init_faiss_index()
    
    # 添加所有问答对到FAISS
    texts = []
    metadatas = []
    ids = []
    
    for qa_id, question, answer in qa_pairs:
        texts.append(question)
        metadatas.append({"answer": answer, "id": qa_id})
        ids.append(qa_id)
    
    # 将所有问答对添加到FAISS
    if texts:
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"从SQLite加载了 {len(texts)} 个问答对到FAISS")
    
    conn.close()


# 初始化数据库和向量索引
init_sqlite_db()
vector_db = init_faiss_index()
load_qa_from_sqlite_to_faiss()

#------------------------------------------------------------------------------
# 缓存检查与存储
#------------------------------------------------------------------------------

def check_cache(query: str) -> Tuple[bool, Optional[str]]:
    """检查查询是否在缓存中
    
    特殊处理：
    1. 对于短文本（小于等于5个字符），先进行精确匹配
    2. 对于较短文本（小于等于10个字符），使用更低的相似度阈值
    
    Args:
        query: 要检查的查询文本
        
    Returns:
        (is_hit, answer): 是否命中缓存及缓存的回答
    """
    # 如果向量数据库为空，直接返回未命中
    if len(vector_db.index_to_docstore_id) <= 1:  # 只有初始化文档
        return False, None
    
    print(f"检查缓存: {query}")
    
    # 对轻量请求进行精确匹配
    if len(query) <= SHORT_TEXT_LENGTH:
        # 对于非常短的文本，先尝试精确匹配
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, answer FROM qa_pairs WHERE question = ?", (query,))
        exact_match = cursor.fetchone()
        conn.close()
        
        if exact_match:
            print(f"找到精确匹配: {query}")
            return True, exact_match[1]
    
    # 获取查询的向量嵌入
    query_embedding = embeddings.embed_query(query)
    
    # 根据文本长度决定相似度阈值
    current_threshold = SHORT_TEXT_THRESHOLD if len(query) <= 10 else SIMILARITY_THRESHOLD
    
    # 搜索相似问题 - 使用相似度搜索
    results = vector_db.similarity_search_with_score(query, k=3)  # 获取前3个结果
    
    if not results:
        print("没有找到相似问题")
        return False, None
    
    # 获取最相似的文档
    doc, score = results[0]
    doc_id = doc.metadata.get("id")
    
    # 计算相似度
    similarity = 0.0
    
    # 尝试使用精确的余弦相似度
    if doc_id:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT vector FROM vector_embeddings WHERE question_id = ?", (doc_id,))
            vector_result = cursor.fetchone()
            
            if vector_result:
                # 从二进制转换回向量
                doc_embedding_str = vector_result[0].decode()
                doc_embedding = eval(doc_embedding_str)
                
                # 使用余弦相似度
                similarity = cosine_similarity(query_embedding, doc_embedding)
                print(f"最相似问题: {doc.page_content}")
                print(f"余弦相似度: {similarity:.4f} (阈值: {current_threshold})")
            else:
                # 如果没有在SQLite中找到向量，使用FAISS的分数
                similarity = 1.0 - score
                print(f"最相似问题: {doc.page_content}")
                print(f"FAISS相似度: {similarity:.4f} (阈值: {current_threshold})")
        except Exception as e:
            print(f"计算余弦相似度时出错: {e}")
            similarity = 1.0 - score
        finally:
            conn.close()
    else:
        # 如果没有ID，使用FAISS的分数
        similarity = 1.0 - score
        print(f"最相似问题: {doc.page_content}")
        print(f"FAISS相似度: {similarity:.4f} (阈值: {current_threshold})")
    
    # 跳过初始化文档
    if doc.metadata.get("is_init", False) or doc.page_content == "初始化文档":
        print("跳过初始化文档")
        return False, None
    
    # 检查是否超过相似度阈值
    if similarity >= current_threshold:
        answer = doc.metadata.get("answer", "")
        print(f"缓存命中! 相似度: {similarity:.4f} >= 阈值: {current_threshold}")
        return True, answer
    
    print(f"缓存未命中. 相似度: {similarity:.4f} < 阈值: {current_threshold}")
    return False, None

def store_qa_pair(question: str, answer: str) -> str:
    """将问答对存储到SQLite和向量数据库
    
    Args:
        question: 要存储的问题
        answer: 要存储的回答
        
    Returns:
        新创建的问答对ID
    """
    # 检查是否已存在相同问题
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM qa_pairs WHERE question = ?", (question,))
    existing = cursor.fetchone()
    
    if existing:
        doc_id = existing[0]
        # 更新现有问答对
        cursor.execute("UPDATE qa_pairs SET answer = ? WHERE id = ?", (answer, doc_id))
        conn.commit()
        conn.close()
        print(f"更新已存在的问答对 - ID: {doc_id}")
        return doc_id
    
    # 创建新的问答对
    doc_id = str(uuid.uuid4())
    
    # 获取问题的向量嵌入
    question_embedding = embeddings.embed_query(question)
    
    try:
        # 存储问答对
        cursor.execute(
            "INSERT INTO qa_pairs (id, question, answer) VALUES (?, ?, ?)",
            (doc_id, question, answer)
        )
        
        # 将向量转换为二进制格式存储
        vector_binary = sqlite3.Binary(str(question_embedding).encode())
        
        # 存储向量嵌入
        cursor.execute(
            "INSERT INTO vector_embeddings (id, question_id, vector) VALUES (?, ?, ?)",
            (str(uuid.uuid4()), doc_id, vector_binary)
        )
        
        conn.commit()
        print(f"已存储问答对到SQLite - 问题: {question[:30]}{'...' if len(question) > 30 else ''}")
    except Exception as e:
        conn.rollback()
        print(f"存储到SQLite失败: {e}")
        return None
    finally:
        conn.close()
    
    # 存储到FAISS向量数据库
    try:
        vector_db.add_texts(
            texts=[question],
            metadatas=[{"answer": answer, "id": doc_id}],
            ids=[doc_id]
        )
        print(f"已存储问答对到FAISS")
    except Exception as e:
        print(f"存储到FAISS失败: {e}")
    
    return doc_id

#------------------------------------------------------------------------------
# 搜索代理与核心工作流
#------------------------------------------------------------------------------

def should_search(thinking_content: str) -> bool:
    """根据模型的思考过程判断是否需要搜索
    
    简化版本：默认返回true，让系统总是进行搜索
    
    Args:
        thinking_content: 模型的思考过程内容（已不使用）
    
    Returns:
        是否需要搜索
    """
    # 始终返回true，表示需要搜索
    return True


def perform_web_search(query: str) -> str:
    """执行网络搜索
    
    Args:
        query: 搜索查询
        
    Returns:
        格式化的搜索结果
    """
    try:
        # 调用Tavily搜索工具
        search_results = search_tool.invoke({"query": query})
        
        # 格式化搜索结果
        formatted_results = "搜索结果:\n"
        for i, result in enumerate(search_results):
            formatted_results += f"{i+1}. {result['title']}\n   链接: {result['url']}\n   摘要: {result['content'][:200]}...\n\n"
            
        return formatted_results
    except Exception as e:
        return f"搜索时出错: {str(e)}"


def search_agent(state: MessagesState) -> Dict[str, List]:
    """搜索代理函数 - 实现思考-观察-行动模式
    
    工作流程:
    1. 先检查缓存，如果命中则直接返回缓存的回答
    2. 如果缓存未命中，让模型思考是否需要搜索
    3. 如果需要搜索，执行搜索并基于结果生成回答
    4. 如果不需要搜索，直接调用模型生成回答
    5. 将生成的问答对存入缓存
    """
    # 获取最后一条人类消息
    last_message = state["messages"][-1]
    
    if not isinstance(last_message, HumanMessage):
        # 如果不是人类消息，直接返回当前消息
        return {"messages": state["messages"]}
        
    query = last_message.content
    print(f"\n处理查询: {query}")
    
    # 1. 首先检查缓存
    is_hit, cached_answer = check_cache(query)
    if is_hit and cached_answer:
        print("缓存命中! 使用缓存的回答")
        return {"messages": state["messages"] + [AIMessage(content=cached_answer)]}
    
    # 2. 缓存未命中，让模型思考是否需要搜索
    thinking_prompt = f"{SEARCH_AGENT_PROMPT}\n\n用户问题: {query}\n\n请先思考这个问题是否需要搜索外部信息，然后决定是否进行搜索。如果需要搜索，请指出要搜索的关键词。"
    thinking_response = model.invoke([HumanMessage(content=thinking_prompt)])
    thinking_content = thinking_response.content if isinstance(thinking_response, AIMessage) else thinking_response[-1].content
    
    # 3. 分析模型的思考过程，决定是否搜索
    need_search = should_search(thinking_content)
    
    final_response = None
    
    if need_search:
        # 4a. 执行搜索 - 直接使用原始查询
        print(f"搜索代理决定搜索: {query}")
        search_results = perform_web_search(query)
        
        # 生成最终回答
        final_prompt = f"{SEARCH_AGENT_PROMPT}\n\n用户问题: {query}\n\n{search_results}\n\n请基于以上搜索结果和你的知识，给出全面、准确的回答。"
        final_response = model.invoke([HumanMessage(content=final_prompt)])
    else:
        # 4b. 直接调用模型
        print("搜索代理决定不需要搜索")
        final_response = model.invoke(state["messages"])
    
    # 5. 处理响应并存储到缓存
    if isinstance(final_response, AIMessage):
        final_content = final_response.content
        store_qa_pair(query, final_content)
        return {"messages": state["messages"] + [final_response]}
    elif isinstance(final_response, list) and final_response and isinstance(final_response[-1], AIMessage):
        final_content = final_response[-1].content
        store_qa_pair(query, final_content)
        return {"messages": state["messages"] + final_response}
    else:
        # 处理其他情况
        if hasattr(final_response, 'content'):
            store_qa_pair(query, final_response.content)
        return {"messages": state["messages"] + [final_response]}

# 删除原来的call_model函数，因为现在我们使用搜索代理来处理所有查询

# 删除原来的should_search函数，因为现在我们使用搜索代理来判断是否需要搜索

#------------------------------------------------------------------------------
# 工作流图定义
#------------------------------------------------------------------------------

def build_workflow():
    """创建工作流图"""
    # 创建工作流图
    workflow = StateGraph(state_schema=MessagesState)
    
    # 添加搜索代理节点
    workflow.add_node("agent", search_agent)
    
    # 定义起始边
    workflow.add_edge(START, "agent")
    
    # 编译工作流图
    return workflow.compile()

# 创建应用
app = build_workflow()

#------------------------------------------------------------------------------
# 交互式聊天功能
#------------------------------------------------------------------------------

def interactive_chat():
    """交互式聊天功能
    
    允许用户与智能助手进行对话，支持缓存和搜索功能
    """
    print("欢迎使用智能助手（集成缓存和搜索功能）")
    print("输入'退出'或'exit'或'quit'结束对话")
    print("==========================================================")
    
    # 初始化消息历史
    messages = []
    
    while True:
        # 获取用户输入
        user_input = input("\n用户: ")
        
        # 检查是否退出
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("谢谢使用，再见！")
            break
        
        # 添加用户消息
        messages.append(HumanMessage(content=user_input))
        
        try:
            # 调用应用
            result = app.invoke({"messages": messages})
            
            # 更新消息历史
            if "messages" in result:
                messages = result["messages"]
                
                # 打印助手回复
                if messages and isinstance(messages[-1], AIMessage):
                    print(f"\n助手: {messages[-1].content}")
        except Exception as e:
            print(f"\n处理请求时出错: {str(e)}")
            # 如果发生错误，删除最后一条消息，以防重复处理
            if messages:
                messages.pop()

#------------------------------------------------------------------------------
# 主程序入口
#------------------------------------------------------------------------------

def main():
    """主程序入口"""
    print("\n智能对话系统 - 启动中...")
    print("* 已加载向量数据库缓存")
    print("* 已集成Tavily搜索工具")
    print("* 已创建思考-观察-行动搜索代理")
    print("\n系统已准备就绪\n")
    
    # 运行交互式聊天
    interactive_chat()

if __name__ == "__main__":
    main()
