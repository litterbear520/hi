#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
智能对话系统 - 集成向量数据库缓存、知识库和在线搜索功能

特点：
1. 使用SQLite存储问答对和向量嵌入
2. 使用FAISS进行向量相似度搜索
3. 集成Tavily在线搜索工具
4. 基于LangGraph实现工作流图
5. 实现思考-观察-行动模式的搜索代理
6. 支持RAG检索增强生成
7. 支持上传多种格式文件作为知识库
8. 使用FastAPI构建前后端分离应用
"""

# 标准库导入
import os
import re
import uuid
import sqlite3
import time
import json
import tempfile
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, AsyncGenerator
import threading
import base64
from io import BytesIO
from PIL import Image

# 导入对话管理工具
from conversation_utils import (
    create_conversation,
    get_conversation,
    get_all_conversations,
    update_conversation_title,
    update_conversation_time,
    delete_conversation,
    get_conversation_messages,
    add_message_to_conversation,
    generate_cute_name
)

# 第三方库导入
import numpy as np
import dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# LangChain 导入
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.utils import AddableDict
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import (
    TextLoader, 
    CSVLoader, 
    PyPDFLoader, 
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import START, MessagesState, StateGraph
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 导入图像理解工具
from vision_tools import VisionAnalysisTool
from thinking_tools import get_thinking_response, stream_thinking_response, thinking_tool

#------------------------------------------------------------------------------
# 配置部分
#------------------------------------------------------------------------------

# 加载环境变量
dotenv.load_dotenv()

# 系统配置
DATA_DIR = "data"  # 数据存储目录
DB_PATH = os.path.join(DATA_DIR, "qa_cache.db")  # SQLite数据库路径
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")  # 知识库目录
SIMILARITY_THRESHOLD = 0.7  # 缓存命中阈值（余弦相似度），恢复使用余弦相似度
SHORT_TEXT_LENGTH = 5  # 短文本定义（字符数）
SHORT_TEXT_THRESHOLD = 0.6  # 短文本相似度阈值（余弦相似度）

# 记忆管理配置
MAX_TOKEN_LIMIT = 2000  # 记忆的最大token限制
MAX_TURNS_FOR_FULL_MEMORY = 10  # 使用完整记忆的最大对话轮数

# 全局记忆存储 - 为每个对话维护独立的记忆
conversation_memories = {}

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# 搜索代理的提示词模板
SEARCH_AGENT_PROMPT = """你是一个支持联网搜索的智能助手，配备了搜索工具来获取最新信息。用户已经启用了联网搜索功能，这意味着你可以在需要时主动使用搜索工具。

如果提供了搜索结果，请直接基于这些结果回答用户的问题。如果还没有搜索结果，请判断是否需要搜索：

以下问题需要搜索：
1. 与最新新闻、时事、实时信息相关的问题
2. 关于当前事件、资讯、行情的问题
3. 关于特定产品、服务、公司的最新信息
4. 涉及价格、评价、对比的问题
5. 当前热点话题和趋势

不需要搜索的问题：
1. 基础定义、概念解释
2. 历史事件、基础知识
3. 个人意见、建议或意向
4. 简单的问候、打招呼或聊天

如果你判断需要搜索，请立即告知用户你需要搜索信息。在使用搜索结果时，请合理引用信息源并加以个人解读分析。

如果你判断不需要搜索，请直接基于你的知识回答问题。
"""

# RAG提示词模板
RAG_PROMPT = """你是一个专业的助手，擅长根据提供的参考文档回答问题。

请根据以下参考文档回答用户的问题。如果参考文档中没有足够的信息来回答问题，请明确说明并尽力基于你自己的知识提供一个有帮助的回答。

参考文档:
{context}

用户问题: {question}

请提供一个详细、准确且有帮助的回答。如果引用了参考文档中的信息，请确保准确表述，不要添加不存在的内容。
"""

#------------------------------------------------------------------------------
# 模型和工具初始化
#------------------------------------------------------------------------------

# 模型配置字典
AVAILABLE_MODELS = {
    'qwen3-235b-a22b': {
        'name': 'Qwen3-235B',
        'description': '最新的Qwen3模型，支持深度思考',
        'supports_thinking': True,
        'supports_vision': False
    },
    'qwen-plus': {
        'name': 'Qwen-Plus',
        'description': '平衡性能和效率的模型',
        'supports_thinking': False,
        'supports_vision': False
    },
    'qwen-plus-latest': {
        'name': 'Qwen-Plus-Latest',
        'description': '最新版本，支持深度思考开关',
        'supports_thinking': True,
        'supports_vision': False
    },
    'qwen-max': {
        'name': 'Qwen-Max',
        'description': '最强大的通用模型',
        'supports_thinking': False,
        'supports_vision': False
    },
    'qwen-vl-max': {
        'name': 'Qwen-VL-Max',
        'description': '多模态视觉理解模型',
        'supports_thinking': False,
        'supports_vision': True
    },
    'deepseek-r1': {
        'name': 'DeepSeek-R1',
        'description': '深度思考模型，自动开启思考过程',
        'supports_thinking': True,
        'supports_vision': False,
        'force_thinking': True
    }
}

# 初始化模型字典
models = {}

def get_model(model_id: str, enable_thinking: bool = False):
    """获取指定的模型实例"""
    try:
        # 根据模型ID选择不同的模型
        if model_id == "qwen3-235b-a22b":
            return ChatTongyi(
                model="qwen3-235b-a22b",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                temperature=0.7,
                max_tokens=8192,
                streaming=True
            )
        elif model_id == "qwen-plus-latest":
            return ChatTongyi(
                model="qwen-plus-latest",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                temperature=0.7,
                max_tokens=8192,
                streaming=True
            )
        elif model_id == "deepseek-r1":
            # deepseek-r1使用标准模型，思考功能由thinking_tool处理
            return ChatTongyi(
                model="deepseek-r1",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                temperature=0.7,
                max_tokens=8192,
                streaming=True
            )
        elif model_id == "qwen-plus":
            return ChatTongyi(
                model="qwen-plus",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                temperature=0.7,
                max_tokens=8192,
                streaming=True
            )
        elif model_id == "qwen-vl-max":
            # 视觉理解模型
            return ChatTongyi(
                model="qwen-vl-max",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                temperature=0.7,
                max_tokens=8192,
                streaming=True
            )
        else:  # 默认使用 qwen-max
            return ChatTongyi(
                model="qwen-max",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                temperature=0.7,
                max_tokens=8192,
                streaming=True
            )
    except Exception as e:
        print(f"创建模型时出错: {str(e)}")
        # 如果出错，返回默认模型
        return ChatTongyi(
            model="qwen-max",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.7,
            max_tokens=8192,
            streaming=True
        )

# 初始化大模型
model = get_model("qwen-max", True)

# 初始化嵌入模型
embeddings = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"), 
    model="text-embedding-v2"  # 使用阿里云千问的文本嵌入模型
)

# 初始化搜索工具
search_tool = TavilySearchResults(
    max_results=3,
)

# 初始化图像理解工具
vision_tool = VisionAnalysisTool()

#------------------------------------------------------------------------------
# 记忆管理功能
#------------------------------------------------------------------------------

def get_conversation_memory(conversation_id: str) -> ConversationSummaryBufferMemory:
    """获取或创建对话记忆"""
    if conversation_id not in conversation_memories:
        # 创建新的记忆实例
        conversation_memories[conversation_id] = ConversationSummaryBufferMemory(
            llm=model,
            max_token_limit=MAX_TOKEN_LIMIT,
            return_messages=True
        )
    return conversation_memories[conversation_id]

def load_conversation_history(conversation_id: str):
    """从数据库加载对话历史到记忆中"""
    if not conversation_id:
        return
    
    memory = get_conversation_memory(conversation_id)
    
    # 从数据库获取对话消息
    try:
        messages = get_conversation_messages(conversation_id)
        
        # 将历史消息添加到记忆中
        for msg in messages:
            memory.save_context(
                {"input": msg["user_message"]},
                {"output": msg["ai_message"]}
            )
        
        print(f"✅ 已加载对话历史到记忆: {len(messages)} 条消息")
        
    except Exception as e:
        print(f"❌ 加载对话历史失败: {str(e)}")

def get_conversation_context(conversation_id: str, current_message: str) -> List:
    """获取对话上下文，基于数据库聊天记录构建上下文"""
    if not conversation_id:
        return [HumanMessage(content=current_message)]
    
    try:
        # 直接从数据库获取对话历史
        messages = get_conversation_messages(conversation_id)
        
        # 构建上下文消息列表
        context_messages = []
        
        # 如果对话轮数较少，使用完整历史
        if len(messages) <= MAX_TURNS_FOR_FULL_MEMORY:
            print(f"🧠 使用完整对话历史: {len(messages)} 条消息")
            
            # 将历史消息转换为LangChain消息格式
            for msg in messages:
                context_messages.append(HumanMessage(content=msg["user_message"]))
                context_messages.append(AIMessage(content=msg["ai_message"]))
        
        else:
            # 对话较长，使用摘要 + 最近消息
            print(f"🧠 对话较长({len(messages)}条)，使用摘要模式")
            
            # 获取最近的几轮对话
            recent_messages = messages[-6:]  # 最近3轮对话
            
            # 生成早期对话的摘要
            early_messages = messages[:-6]
            if early_messages:
                summary_text = f"早期对话摘要(共{len(early_messages)}轮):\n"
                for i, msg in enumerate(early_messages[:3]):  # 取前3轮作为摘要示例
                    summary_text += f"Q: {msg['user_message'][:50]}...\n"
                    summary_text += f"A: {msg['ai_message'][:50]}...\n"
                if len(early_messages) > 3:
                    summary_text += f"...等共{len(early_messages)}轮对话"
                
                # 添加摘要消息
                context_messages.append(HumanMessage(content=summary_text))
            
            # 添加最近的对话
            for msg in recent_messages:
                context_messages.append(HumanMessage(content=msg["user_message"]))
                context_messages.append(AIMessage(content=msg["ai_message"]))
        
        # 添加当前消息
        context_messages.append(HumanMessage(content=current_message))
        
        print(f"✅ 构建上下文完成: 总共{len(context_messages)}条消息")
        return context_messages
        
    except Exception as e:
        print(f"❌ 获取对话上下文失败: {str(e)}")
        # 如果出错，返回当前消息
        return [HumanMessage(content=current_message)]

def save_to_memory(conversation_id: str, user_message: str, ai_message: str):
    """保存新的对话到数据库（替换内存记忆）"""
    if not conversation_id or not user_message or not ai_message:
        return
    
    try:
        # 直接保存到数据库，不使用内存记忆
        add_message_to_conversation(conversation_id, user_message, ai_message)
        print(f"✅ 已保存到对话数据库: {conversation_id} - {user_message[:30]}...")
    except Exception as e:
        print(f"❌ 保存对话到数据库失败: {str(e)}")

def get_conversation_summary(conversation_id: str) -> str:
    """获取对话摘要（用于长对话）"""
    if not conversation_id:
        return ""
    
    try:
        messages = get_conversation_messages(conversation_id)
        if not messages:
            return ""
        
        # 构建摘要文本
        summary = f"对话摘要(共{len(messages)}轮):\n"
        
        # 取前几轮和最近几轮对话
        preview_count = min(2, len(messages))
        for i in range(preview_count):
            msg = messages[i]
            summary += f"Q{i+1}: {msg['user_message'][:40]}...\n"
            summary += f"A{i+1}: {msg['ai_message'][:40]}...\n"
        
        if len(messages) > preview_count:
            summary += f"... 中间省略{len(messages) - preview_count}轮对话 ...\n"
        
        return summary
        
    except Exception as e:
        print(f"❌ 生成对话摘要失败: {str(e)}")
        return ""

def clear_conversation_memory(conversation_id: str):
    """清除指定对话的记忆（已不再需要，保留接口兼容性）"""
    print(f"✅ 对话记忆管理已简化，使用数据库存储: {conversation_id}")

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


def check_and_upgrade_db():
    """检查并升级数据库结构"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 检查qa_pairs表是否有conversation_id列
    cursor.execute("PRAGMA table_info(qa_pairs)")
    columns = cursor.fetchall()
    has_conversation_id = False
    
    for column in columns:
        if column[1] == 'conversation_id':
            has_conversation_id = True
            break
    
    # 如果没有conversation_id列，添加它
    if not has_conversation_id:
        print("正在升级数据库: 添加conversation_id列到qa_pairs表")
        try:
            cursor.execute("ALTER TABLE qa_pairs ADD COLUMN conversation_id TEXT")
            conn.commit()
            print("数据库升级成功")
        except Exception as e:
            print(f"数据库升级失败: {str(e)}")
    
    conn.close()

def init_sqlite_db():
    """初始化SQLite数据库结构"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # 创建对话表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
        ''')
        
        # 创建问答表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            conversation_id TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
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
        
        # 创建知识库文件表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_files (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed BOOLEAN DEFAULT 0
        )
        ''')
        
        # 在同一个连接中检查并升级数据库
        # 检查qa_pairs表是否有conversation_id列
        cursor.execute("PRAGMA table_info(qa_pairs)")
        columns = cursor.fetchall()
        has_conversation_id = False
        
        for column in columns:
            if column[1] == 'conversation_id':
                has_conversation_id = True
                break
        
        # 如果没有conversation_id列，添加它
        if not has_conversation_id:
            print("正在升级数据库: 添加conversation_id列到qa_pairs表")
            cursor.execute("ALTER TABLE qa_pairs ADD COLUMN conversation_id TEXT")
            conn.commit()
            print("数据库升级成功")
        
        conn.commit()
        print("初始化SQLite数据库完成")
    
    except Exception as e:
        print(f"初始化数据库错误: {str(e)}")
    finally:
        conn.close()


def init_faiss_index():
    """初始化FAISS向量索引
    
    创建一个初始文档，因为FAISS不能使用空列表初始化
    """
    initial_texts = ["初始化文档"]
    vector_db = FAISS.from_texts(
        texts=initial_texts, 
        embedding=embeddings,
        metadatas=[{"is_init": True, "answer": "", "source": "init"}]
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
        metadatas.append({"answer": answer, "id": qa_id, "source": "cache"})
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
# 知识库处理函数
#------------------------------------------------------------------------------

def get_document_loader(file_path, file_type):
    """根据文件类型获取适当的文档加载器"""
    loaders = {
        "txt": lambda: TextLoader(file_path),
        "csv": lambda: CSVLoader(file_path),
        "pdf": lambda: PyPDFLoader(file_path),
        "md": lambda: UnstructuredMarkdownLoader(file_path),
        "xls": lambda: UnstructuredExcelLoader(file_path),
        "xlsx": lambda: UnstructuredExcelLoader(file_path),
        "doc": lambda: UnstructuredWordDocumentLoader(file_path),
        "docx": lambda: UnstructuredWordDocumentLoader(file_path),
        "json": lambda: JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
    }
    
    loader_func = loaders.get(file_type.lower())
    if not loader_func:
        raise ValueError(f"不支持的文件类型: {file_type}")
    
    return loader_func()


def process_knowledge_file(file_id, file_path, file_type):
    """处理知识库文件并添加到向量数据库"""
    global vector_db
    
    try:
        # 首先检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 {file_path}")
            return False
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print(f"错误: 文件为空 {file_path}")
            return False
        
        print(f"开始处理文件: {file_path} (类型: {file_type}, 大小: {file_size} 字节)")
        
        try:
            # 尝试加载文档
            loader = get_document_loader(file_path, file_type)
            documents = loader.load()
            print(f"成功加载文档: {len(documents)} 个文档")
        except Exception as loader_error:
            print(f"Error loading {file_path}: {str(loader_error)}")
            # 对于文本文件，尝试使用基本的文件读取
            if file_type.lower() == 'txt':
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    from langchain_core.documents import Document
                    documents = [Document(page_content=content, metadata={'source': file_path})]
                    print(f"使用备用方法成功读取文本文件: {len(content)} 字符")
                except Exception as basic_error:
                    print(f"基本读取也失败: {str(basic_error)}")
                    return False
            else:
                return False
        
        # 文本分割器 - 优化中文文档分割
        # 定义中文友好的分隔符
        chinese_separators = [
            "\n\n",  # 段落分隔
            "\n",    # 行分隔
            "。",    # 中文句号
            "！",    # 中文感叹号
            "？",    # 中文问号
            "；",    # 中文分号
            "，",    # 中文逗号
            "、",    # 中文顿号
            " ",     # 空格
            "",      # 字符级别分割
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # 适当减小chunk大小，更适合中文
            chunk_overlap=100,  # 减小重叠，避免过多重复
            length_function=len,
            separators=chinese_separators,  # 使用中文友好的分隔符
            keep_separator=True,  # 保留分隔符，保持语义完整性
        )
        
        # 分割文档
        try:
            docs = text_splitter.split_documents(documents)
            print(f"成功分割文档: {len(docs)} 个片段")
        except Exception as split_error:
            print(f"分割文档时出错: {str(split_error)}")
            return False
        
        # 提取文本和元数据
        texts = [doc.page_content for doc in docs]
        metadatas = []
        
        for doc in docs:
            metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            metadata["source"] = f"knowledge_{file_id}"
            metadata["file_path"] = file_path
            metadata["file_type"] = file_type
            metadatas.append(metadata)
        
        # 添加到向量数据库
        try:
            vector_db.add_texts(texts=texts, metadatas=metadatas)
            print(f"成功添加到向量数据库: {len(texts)} 个文档片段")
        except Exception as db_error:
            print(f"添加到向量数据库时出错: {str(db_error)}")
            return False
        
        # 更新数据库状态
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE knowledge_files SET processed = 1 WHERE id = ?", (file_id,))
        conn.commit()
        conn.close()
        
        print(f"成功处理文件 {file_path}, 添加了 {len(texts)} 个文档片段")
        return True
    
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"处理文件时出错: {str(e)}\n{trace}")
        return False


def add_file_to_knowledge_base(file_path, original_filename, file_type):
    """添加文件到知识库"""
    file_id = str(uuid.uuid4())
    
    # 保存文件信息到数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO knowledge_files (id, filename, file_path, file_type) VALUES (?, ?, ?, ?)",
        (file_id, original_filename, file_path, file_type)
    )
    conn.commit()
    conn.close()
    
    # 异步处理文件
    success = process_knowledge_file(file_id, file_path, file_type)
    
    return {
        "file_id": file_id,
        "filename": original_filename,
        "success": success
    }


def get_knowledge_base_files():
    """获取知识库中的所有文件"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, file_type, upload_time, processed FROM knowledge_files")
    files = [
        {
            "id": row[0],
            "filename": row[1],
            "file_type": row[2],
            "upload_time": row[3],
            "processed": bool(row[4])
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return files


def query_knowledge_base(query, top_k=3):
    """查询知识库，返回最相关的文档"""
    if not vector_db:
        return []
    
    # 进行向量检索（不使用过滤器）
    docs_with_scores = vector_db.similarity_search_with_score(
        query=query,
        k=top_k * 3  # 获取更多结果，然后手动过滤
    )
    
    results = []
    for doc, score in docs_with_scores:
        # 排除初始化文档和缓存文档
        if doc.metadata.get("is_init", False) or doc.metadata.get("source", "") == "cache":
            continue
        
        # 只保留top_k个结果
        if len(results) >= top_k:
            break
            
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score)
        })
    
    return results


#------------------------------------------------------------------------------
# 图片处理函数
#------------------------------------------------------------------------------

def process_images(images_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """处理图片数据，确保URL格式正确"""
    processed_images = []
    
    for image_data in images_data:
        try:
            # 确保图片数据包含必要字段
            if not isinstance(image_data, dict):
                continue
                
            # 获取图片URL
            image_url = image_data.get('url', '')
            if not image_url:
                # 如果没有URL，尝试从data字段构建
                data = image_data.get('data', '')
                if data and data.startswith('data:image'):
                    image_url = data
                else:
                    continue
            
            # 确保URL格式正确
            if not image_url.startswith(('http://', 'https://', 'data:image')):
                # 如果是base64数据但没有前缀，添加前缀
                if ',' in image_url:
                    image_url = f"data:image/jpeg;base64,{image_url.split(',')[-1]}"
                else:
                    image_url = f"data:image/jpeg;base64,{image_url}"
            
            processed_image = {
                'url': image_url,
                'name': image_data.get('name', 'image'),
                'size': image_data.get('size', 0)
            }
            
            processed_images.append(processed_image)
            
        except Exception as e:
            print(f"处理图片数据时出错: {str(e)}")
            continue
    
    return processed_images

def create_vision_message(text: str, images: List[Dict[str, Any]]) -> HumanMessage:
    """创建包含图片的消息"""
    content = []
    
    # 添加文本内容
    if text:
        content.append({
            "type": "text",
            "text": text
        })
    
    # 添加图片内容 - 修复格式以符合千问VL模型要求
    for img in images:
        content.append({
            "type": "image",
            "image": f"data:{img['type']};base64,{img['data']}"
        })
    
    return HumanMessage(content=content)


#------------------------------------------------------------------------------
# 缓存检查与存储
#------------------------------------------------------------------------------

def check_cache(query: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """检查查询是否在缓存中
    
    特殊处理：
    1. 对于短文本（小于等于5个字符），先进行精确匹配
    2. 对于较短文本（小于等于10个字符），使用更低的相似度阈值
    
    Args:
        query: 要检查的查询文本
        
    Returns:
        (is_hit, answer, id): 是否命中缓存、缓存的回答及其ID
    """
    if not query or not query.strip():
        return False, None, None
    
    query = query.strip()
    
    # 对于短文本，先尝试精确匹配
    if len(query) <= SHORT_TEXT_LENGTH:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, answer FROM qa_pairs WHERE question = ?", (query,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return True, result[1], result[0]
    
    # 没有精确匹配，执行向量搜索
    try:
        # 首先获取查询的向量嵌入
        query_embedding = embeddings.embed_query(query)
        
        # 执行向量搜索（获取更多结果进行手动相似度计算）
        docs_with_scores = vector_db.similarity_search_with_score(
            query=query,
            k=10  # 获取更多结果，然后手动计算余弦相似度
        )
        
        # 如果没有结果，返回未命中
        if not docs_with_scores:
            print("向量搜索无结果")
            return False, None, None
        
        # 手动过滤出缓存的问题并计算余弦相似度
        cache_results = []
        for doc, faiss_score in docs_with_scores:
            if doc.metadata.get("source") == "cache":
                # 获取文档的向量嵌入并计算余弦相似度
                doc_embedding = embeddings.embed_query(doc.page_content)
                cosine_sim = cosine_similarity(query_embedding, doc_embedding)
                cache_results.append((doc, cosine_sim))
        
        if not cache_results:
            print("没有找到缓存的问题")
            return False, None, None
        
        # 按余弦相似度排序，获取最相似的结果
        cache_results.sort(key=lambda x: x[1], reverse=True)
        doc, similarity = cache_results[0]
        
        # 对于短文本，使用更低的阈值
        threshold = SHORT_TEXT_THRESHOLD if len(query) <= 10 else SIMILARITY_THRESHOLD
        
        print(f"缓存查询: {query[:30]}..., 最佳匹配余弦相似度: {similarity:.4f}, 阈值: {threshold}")
        
        # 余弦相似度越大越相似
        if similarity >= threshold:
            print(f"缓存命中: {doc.page_content[:30]}...")
            return True, doc.metadata.get("answer"), doc.metadata.get("id")
        else:
            print(f"相似度不足，未命中缓存")
        
    except Exception as e:
        print(f"向量搜索错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return False, None, None


def store_qa_pair(question: str, answer: str) -> str:
    """将问答对存储到SQLite和向量数据库
    
    Args:
        question: 要存储的问题
        answer: 要存储的回答
        
    Returns:
        新创建的问答对ID
    """
    if not question or not answer:
        return ""
    
    question = question.strip()
    answer = answer.strip()
    
    # 生成唯一ID
    qa_id = str(uuid.uuid4())
    
    try:
        # 存储到SQLite
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 插入问答对
        cursor.execute(
            "INSERT INTO qa_pairs (id, question, answer) VALUES (?, ?, ?)",
            (qa_id, question, answer)
        )
        
        conn.commit()
        conn.close()
        
        # 存储到向量数据库
        vector_db.add_texts(
            texts=[question],
            metadatas=[{"answer": answer, "id": qa_id, "source": "cache"}],
            ids=[qa_id]
        )
        
        print(f"已缓存问答对: {question[:30]}... -> {answer[:30]}...")
        return qa_id
        
    except Exception as e:
        print(f"存储问答对时出错: {str(e)}")
        return ""


#------------------------------------------------------------------------------
# 搜索代理与RAG
#------------------------------------------------------------------------------

def format_knowledge_results(results):
    """格式化知识库搜索结果为可读格式"""
    if not results:
        return "没有找到相关知识库内容。"
    
    formatted = "找到的相关知识：\n\n"
    for i, item in enumerate(results, 1):
        source = item['metadata'].get('source', '未知来源')
        if source.startswith('knowledge_'):
            source = item['metadata'].get('file_path', '未知文件')
            if isinstance(source, str) and os.path.exists(source):
                source = os.path.basename(source)
        
        formatted += f"[{i}] 来源: {source}\n"
        formatted += f"内容: {item['content'][:200]}{'...' if len(item['content']) > 200 else ''}\n\n"
    
    return formatted


def perform_web_search(query: str) -> str:
    """执行网络搜索，直接调用搜索API
    
    Args:
        query: 搜索查询
        
    Returns:
        搜索结果文本
    """
    try:
        print(f"🔍 开始搜索: {query}")
        
        # 直接调用搜索API
        search_results = search_tool.invoke({"query": query})
        
        if not search_results:
            return "搜索没有返回结果。"
        
        # 格式化搜索结果
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            title = result.get('title', '无标题')
            content = result.get('content', '')
            url = result.get('url', '')
            
            formatted_result = f"[{i}] {title}\n"
            if content:
                # 限制内容长度
                content = content[:300] + "..." if len(content) > 300 else content
                formatted_result += f"内容: {content}\n"
            if url:
                formatted_result += f"来源: {url}\n"
            
            formatted_results.append(formatted_result)
        
        result_text = "\n".join(formatted_results)
        print(f"✅ 搜索完成，找到 {len(search_results)} 个结果")
        return result_text
        
    except Exception as e:
        error_msg = f"搜索失败: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg


#------------------------------------------------------------------------------
# LangGraph 工作流（简化版）
#------------------------------------------------------------------------------

# 创建应用
app = StateGraph(MessagesState)


#------------------------------------------------------------------------------
# FastAPI应用
#------------------------------------------------------------------------------

# API模型
class ChatRequest(BaseModel):
    message: str
    web_search: bool = False
    deep_thinking: bool = False
    model: str = "qwen-max"
    conversation_id: Optional[str] = None
    images: Optional[List[Dict[str, Any]]] = None


class ConversationRequest(BaseModel):
    title: Optional[str] = None


class ConversationUpdateRequest(BaseModel):
    title: str


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    success: bool
    message: str = ""


# 创建FastAPI应用
fastapi_app = FastAPI(title="智能对话系统API")

# 配置CORS中间件
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加静态文件服务（用于前端静态资源）
from pathlib import Path
templates_dir = Path("templates")
if templates_dir.exists():
    try:
        fastapi_app.mount("/static", StaticFiles(directory="templates"), name="static")
        print("✅ 静态文件服务已配置")
    except Exception as e:
        print(f"⚠️ 静态文件服务配置失败: {e}")

# 全局停止标志管理
stop_flags = {}
stop_flags_lock = threading.Lock()

def set_stop_flag(websocket_id: str, value: bool):
    """设置停止标志"""
    with stop_flags_lock:
        stop_flags[websocket_id] = value

def get_stop_flag(websocket_id: str) -> bool:
    """获取停止标志"""
    with stop_flags_lock:
        return stop_flags.get(websocket_id, False)

def clear_stop_flag(websocket_id: str):
    """清除停止标志"""
    with stop_flags_lock:
        stop_flags.pop(websocket_id, None)

# 存储WebSocket连接
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.websocket_ids = {}  # 存储websocket到ID的映射

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        websocket_id = str(uuid.uuid4())
        self.active_connections.append(websocket)
        self.websocket_ids[websocket] = websocket_id
        set_stop_flag(websocket_id, False)
        return websocket_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        websocket_id = self.websocket_ids.pop(websocket, None)
        if websocket_id:
            clear_stop_flag(websocket_id)

    def get_websocket_id(self, websocket: WebSocket) -> str:
        return self.websocket_ids.get(websocket, "")

    async def send_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"发送消息失败: {str(e)}")
            self.disconnect(websocket)


manager = ConnectionManager()


@fastapi_app.get("/")
async def read_root():
    """返回前端页面"""
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    # 检查前端文件是否存在
    index_file = Path("templates") / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    else:
        return {
            "message": "智能对话系统API",
            "error": "前端文件 templates/index.html 不存在",
            "api_docs": "/docs",
            "suggestion": "请确保 templates/index.html 文件存在"
        }


@fastapi_app.post("/chat")
async def chat(request: ChatRequest):
    """处理聊天请求并返回回答"""
    if not request.message.strip():
        return JSONResponse(content={"错误": "消息不能为空"}, status_code=400)
    
    # 如果有对话 ID，验证它的存在性
    conversation_id = request.conversation_id
    if conversation_id:
        conversation = get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(content={"错误": "对话不存在"}, status_code=404)
    else:
        # 如果没有指定对话，创建一个新对话
        new_conversation = create_conversation()
        conversation_id = new_conversation["id"]
    
    # 构建输入状态
    state = {
        "messages": [HumanMessage(content=request.message)],
        "web_search_requested": request.web_search
    }
    
    # 调用工作流应用
    result = app.invoke(state)
    
    # 提取回答
    ai_message = result["messages"][-1]
    
    # 将问答对添加到对话中
    add_message_to_conversation(conversation_id, request.message, ai_message.content)
    
    return {
        "response": ai_message.content,
        "conversation_id": conversation_id
    }


async def generate_chat_response(message: str, web_search: bool, deep_thinking: bool = False, model_id: str = "qwen-max", images: Optional[List[Dict[str, Any]]] = None, conversation_id: Optional[str] = None, websocket_id: str = None) -> AsyncGenerator[str, None]:
    """生成聊天回答的异步生成器"""
    full_response = ""
    thinking_process = ""
    
    # 处理图片数据
    has_images = images and len(images) > 0
    processed_images = []
    if has_images:
        processed_images = process_images(images)
        print(f"处理了 {len(processed_images)} 张图片")
    
    try:
        # 1. 首先检查缓存（如果没有图片和网络搜索）
        if not has_images and not web_search:
            is_hit, cached_answer, cache_id = check_cache(message)
            if is_hit:
                print(f"✅ 缓存命中，返回缓存结果: {cache_id}")
                yield cached_answer
                full_response = cached_answer
                return
        
        # 2. 加载对话历史到记忆中（如果有对话ID）
        if conversation_id:
            load_conversation_history(conversation_id)
        
        # 3. 获取对话上下文（包含历史记忆）
        context_messages = get_conversation_context(conversation_id, message)
        
        # 4. 如果开启了联网搜索，直接调用搜索API
        search_results = ""
        if web_search:
            search_results = perform_web_search(message)
            # 将搜索结果添加到当前消息中
            search_context = f"\n\n联网搜索结果：\n{search_results}\n\n请基于以上搜索结果回答用户的问题。"
            message_with_search = message + search_context
            # 更新上下文消息的最后一条
            context_messages[-1] = HumanMessage(content=message_with_search)
        else:
            message_with_search = message
        
        # 5. 查询知识库
        knowledge_results = query_knowledge_base(message_with_search)
        if knowledge_results:
            knowledge_context = f"\n\n知识库内容：\n{format_knowledge_results(knowledge_results)}\n\n"
            message_with_search += knowledge_context
            # 更新上下文消息的最后一条
            context_messages[-1] = HumanMessage(content=message_with_search)
        
        # 6. 处理图片理解（保持原有逻辑）
        if has_images and not web_search:
            print("🖼️ 使用视觉模型处理图片")
            
            try:
                # 创建视觉模型
                vision_model = get_model('qwen-vl-max', False)
                
                # 构建包含图片的消息 - 使用正确的OpenAI格式
                message_content = []
                
                # 添加文本问题
                if message.strip():
                    message_content.append({
                        "type": "text", 
                        "text": message
                    })
                else:
                    message_content.append({
                        "type": "text", 
                        "text": "请分析这张图片"
                    })
                
                # 添加图片，使用正确的格式
                for img in processed_images:
                    image_url = img.get('url', '')
                    if image_url:
                        # 确保使用正确的image_url格式
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        })
                
                # 创建消息
                vision_message = HumanMessage(content=message_content)
                
                # 流式生成回答
                async for chunk in vision_model.astream([vision_message]):
                    if websocket_id and get_stop_flag(websocket_id):
                        yield "\n\n[对话已停止]"
                        return
                    
                    chunk_content = chunk.content
                    full_response += chunk_content
                    yield chunk_content
                
            except Exception as e:
                error_msg = f"图片理解失败: {str(e)}"
                print(f"详细错误信息: {error_msg}")
                
                # 如果是格式错误，尝试使用您示例中的格式
                try:
                    print("尝试使用备用格式处理图片")
                    # 使用您的image_test.py中的格式
                    from image_test import QwenVLChatModel
                    
                    # 创建QwenVL模型实例
                    qwen_vl = QwenVLChatModel(streaming=True)
                    
                    # 提取图片URL
                    image_urls = []
                    for img in processed_images:
                        image_urls.append(img.get('url', ''))
                    
                    # 创建包含图像的消息（使用您的示例格式）
                    vision_message = HumanMessage(
                        content=message if message.strip() else "请分析这张图片",
                        additional_kwargs={"images": image_urls}
                    )
                    
                    # 生成响应
                    result = qwen_vl._generate([vision_message])
                    full_response = result.generations[0].message.content
                    yield full_response
                    
                except Exception as backup_error:
                    error_msg = f"图片理解完全失败: 主要错误: {str(e)}, 备用方法错误: {str(backup_error)}"
                    print(f"备用方法也失败: {str(backup_error)}")
                    yield error_msg
                    return
        
        else:
            # 7. 使用普通模型处理文本（使用上下文记忆）
            current_model = get_model(model_id, deep_thinking)
            
            # 对于支持思考的模型，使用思考工具
            if deep_thinking and model_id in ['qwen3-235b-a22b', 'qwen-plus-latest', 'deepseek-r1']:
                thinking_started = False
                content_started = False
                
                for chunk_data in thinking_tool.think_and_respond(
                    question=message_with_search,
                    model=model_id,
                    enable_thinking=True
                ):
                    if websocket_id and get_stop_flag(websocket_id):
                        yield "\n\n[对话已停止]"
                        return
                    
                    # 处理思考内容 - 立即流式输出
                    if chunk_data["type"] == "thinking" and chunk_data["thinking"]:
                        # 如果是第一次输出思考内容，添加标题
                        if not thinking_started:
                            yield "**🤔 思考过程：**\n"
                            thinking_started = True
                        
                        # 立即流式输出思考内容，不累积
                        yield chunk_data["thinking"]
                    
                    # 处理回答内容
                    if chunk_data["type"] == "content" and chunk_data["content"]:
                        # 如果是第一次输出回答内容，添加分隔和标题
                        if not content_started:
                            if thinking_started:
                                yield "\n\n**💡 回答：**\n"
                            content_started = True
                        
                        full_response += chunk_data["content"]
                        yield chunk_data["content"]
                    
                    if chunk_data["type"] == "error":
                        yield chunk_data["content"]
                        return
            else:
                # 普通流式响应（使用上下文记忆）
                async for chunk in current_model.astream(context_messages):
                    if websocket_id and get_stop_flag(websocket_id):
                        yield "\n\n[对话已停止]"
                        return
                    
                    chunk_content = chunk.content
                    full_response += chunk_content
                    yield chunk_content
        
        # 8. 保存对话到记忆中
        if conversation_id and message and full_response:
            save_to_memory(conversation_id, message, full_response)
    
    except Exception as e:
        error_msg = f"生成回答时出错: {str(e)}"
        print(error_msg)
        yield error_msg
        return
    
    # 注意：不要在这里使用finally块，因为这是一个异步生成器
    # QA保存逻辑将在WebSocket处理函数中处理


@fastapi_app.post("/chat/stream")
async def stream_chat(request: Request):
    """流式处理聊天请求并返回回答"""
    data = await request.json()
    message = data.get("message", "")
    web_search = data.get("web_search", False)
    conversation_id = data.get("conversation_id", None)
    
    if not message.strip():
        return JSONResponse(content={"error": "消息不能为空"}, status_code=400)
    
    # 验证对话存在性或创建新对话
    if conversation_id:
        conversation = get_conversation(conversation_id)
        if not conversation:
            return JSONResponse(content={"error": "对话不存在"}, status_code=404)
    else:
        # 如果没有指定对话，创建一个新对话
        new_conversation = create_conversation()
        conversation_id = new_conversation["id"]
        
    # 开始流式生成回答
    return StreamingResponse(
        generate_chat_response(message, web_search, conversation_id=conversation_id),
        media_type="text/plain"
    )


@fastapi_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    websocket_id = await manager.connect(websocket)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            
            # 检查是否是停止请求
            if parsed_data.get("action") == "stop":
                # 设置停止标志
                set_stop_flag(websocket_id, True)
                # 发送停止确认
                await manager.send_message(json.dumps({"stopped": True}), websocket)
                continue
            
            message = parsed_data.get("message", "")
            web_search = parsed_data.get("web_search", False)
            deep_thinking = parsed_data.get("deep_thinking", False)
            model_id = parsed_data.get("model", "qwen-max")
            images = parsed_data.get("images", None)
            conversation_id = parsed_data.get("conversation_id")
            
            # 确保message是字符串类型
            if isinstance(message, list):
                # 如果message是列表，提取文本内容
                text_parts = []
                for item in message:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                message = " ".join(text_parts).strip()
            elif not isinstance(message, str):
                message = str(message)
            
            # 如果message为空但有图片，设置默认消息
            if not message.strip() and images:
                message = "请分析这张图片"
            
            if not message.strip() and not images:
                await manager.send_message(json.dumps({"error": "消息不能为空"}), websocket)
                continue
            
            # 重置停止标志
            set_stop_flag(websocket_id, False)
            
            # 如果没有对话 ID，创建一个新对话
            if not conversation_id:
                # 创建新对话
                new_conversation = create_conversation()
                conversation_id = new_conversation["id"]
                # 将新对话的信息发送给客户端
                await manager.send_message(json.dumps({"new_conversation": new_conversation}), websocket)
            
            # 如果启用了网络搜索，立即发送正在搜索的通知
            if web_search:
                await manager.send_message(json.dumps({"chunk": "正在进行网络搜索...请稍候\n\n"}), websocket)
            
            # 流式生成回答（搜索将在generate_chat_response中处理）
            full_response = ""
            try:
                async for chunk in generate_chat_response(message, web_search, deep_thinking, model_id, images, conversation_id, websocket_id):
                    # 检查连接是否还活跃
                    if websocket not in manager.active_connections:
                        break
                    await manager.send_message(json.dumps({"chunk": chunk}), websocket)
                    full_response += chunk
                
                # 发送完成信号（只有在没有停止的情况下）
                if not get_stop_flag(websocket_id):
                    # 保存QA对话到缓存和对话记录
                    try:
                        # 保存到缓存（如果没有图片）
                        if not images and full_response and message:
                            store_qa_pair(message, full_response)
                            print(f"✅ 已保存QA对话到缓存: {message[:30]}...")
                        
                        # 保存到对话记录
                        if conversation_id and message and full_response:
                            add_message_to_conversation(conversation_id, message, full_response)
                            print(f"✅ 已保存QA对话到对话记录: 对话ID {conversation_id}")
                            
                    except Exception as save_error:
                        print(f"❌ 保存QA对话时出错: {str(save_error)}")
                    
                    await manager.send_message(json.dumps({
                        "complete": True, 
                        "full_response": full_response,
                        "conversation_id": conversation_id
                    }), websocket)
                else:
                    # 如果是停止的，发送停止完成信号
                    await manager.send_message(json.dumps({
                        "stopped_complete": True,
                        "partial_response": full_response,
                        "conversation_id": conversation_id
                    }), websocket)
            except Exception as e:
                # 处理生成过程中的错误
                error_msg = f"生成回答时出错: {str(e)}"
                await manager.send_message(json.dumps({"error": error_msg}), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket处理出错: {str(e)}")
        try:
            await manager.send_message(json.dumps({"error": f"连接出错: {str(e)}"}), websocket)
        except:
            pass
        manager.disconnect(websocket)


@fastapi_app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件到知识库"""
    # 检查文件扩展名
    original_filename = file.filename
    file_extension = original_filename.split(".")[-1].lower()
    
    supported_extensions = ["pdf", "csv", "txt", "md", "xls", "xlsx", "docx", "doc", "json"]
    
    if file_extension not in supported_extensions:
        return FileUploadResponse(
            file_id="",
            filename=original_filename,
            success=False,
            message=f"不支持的文件类型。支持的格式：{', '.join(supported_extensions)}"
        )
    
    # 创建唯一文件名
    unique_filename = f"{str(uuid.uuid4())}.{file_extension}"
    file_path = os.path.join(KNOWLEDGE_DIR, unique_filename)
    
    # 保存文件
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # 处理文件并添加到知识库
    result = add_file_to_knowledge_base(file_path, original_filename, file_extension)
    
    return FileUploadResponse(
        file_id=result["file_id"],
        filename=original_filename,
        success=result["success"],
        message="文件上传成功并已添加到知识库" if result["success"] else "文件上传成功但处理失败"
    )


@fastapi_app.get("/knowledge/files")
async def list_knowledge_files():
    """获取知识库中的所有文件"""
    return get_knowledge_base_files()


@fastapi_app.delete("/knowledge/files/{file_id}")
async def delete_knowledge_file(file_id: str):
    """从知识库中删除文件"""
    try:
        # 首先获取文件信息
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM knowledge_files WHERE id = ?", (file_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return {"success": False, "message": "文件不存在"}
        
        file_path = result[0]
        
        # 从数据库中删除文件记录
        cursor.execute("DELETE FROM knowledge_files WHERE id = ?", (file_id,))
        conn.commit()
        
        # 尝试删除物理文件
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已删除物理文件: {file_path}")
        except Exception as file_error:
            print(f"删除物理文件时出错: {str(file_error)}")
        
        # 重建向量数据库 - 删除文件对应的文档
        global vector_db
        try:
            # 获取当前所有文档
            docs = vector_db.similarity_search_with_score("", k=1000)
            
            # 创建新的FAISS索引
            vector_db = init_faiss_index()
            
            # 重新添加非删除文件的文档
            for doc, _ in docs:
                doc_source = doc.metadata.get("source", "")
                if not doc_source.startswith(f"knowledge_{file_id}"):
                    vector_db.add_texts([doc.page_content], [doc.metadata])
            
            print(f"已从向量数据库中删除文件 {file_id} 对应的文档")
        except Exception as vector_error:
            print(f"更新向量数据库时出错: {str(vector_error)}")
        
        conn.close()
        
        return {"success": True, "message": "已成功删除文件"}
    except Exception as e:
        print(f"删除知识库文件时出错: {str(e)}")
        return {"success": False, "message": f"删除文件时出错: {str(e)}"}


@fastapi_app.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """删除指定的对话"""
    try:
        print(f"尝试删除对话: {conversation_id}")
        # 验证对话存在性
        conversation = get_conversation(conversation_id)
        if not conversation:
            return {"success": False, "message": "对话不存在"}
        
        # 执行删除
        success = delete_conversation(conversation_id)
        if success:
            return {"success": True, "message": "对话删除成功"}
        else:
            return {"success": False, "message": "删除对话时出错"}
    except Exception as e:
        print(f"删除对话时出错: {str(e)}")
        return {"success": False, "message": f"删除对话时出错: {str(e)}"}


@fastapi_app.put("/conversations/{conversation_id}")
async def rename_conversation_endpoint(conversation_id: str, request: Request):
    """重命名对话"""
    try:
        # 获取请求数据
        data = await request.json()
        new_title = data.get("title")
        
        if not new_title or not new_title.strip():
            return {"success": False, "message": "新标题不能为空"}
        
        # 验证对话存在性
        conversation = get_conversation(conversation_id)
        if not conversation:
            return {"success": False, "message": "对话不存在"}
        
        # 执行重命名
        success = update_conversation_title(conversation_id, new_title)
        if success:
            return {"success": True, "message": "对话重命名成功", "title": new_title}
        else:
            return {"success": False, "message": "重命名对话时出错"}
    except Exception as e:
        print(f"重命名对话时出错: {str(e)}")
        return {"success": False, "message": f"重命名对话时出错: {str(e)}"}


@fastapi_app.get("/supported-formats")
async def get_supported_formats():
    """获取支持的文件格式"""
    return {
        "formats": [
            {"extension": "pdf", "description": "PDF文档"},
            {"extension": "csv", "description": "CSV表格文件"},
            {"extension": "txt", "description": "文本文件"},
            {"extension": "md", "description": "Markdown文档"},
            {"extension": "xls", "description": "Excel工作簿 (旧版)"},
            {"extension": "xlsx", "description": "Excel工作簿"},
            {"extension": "docx", "description": "Word文档"},
            {"extension": "doc", "description": "Word文档 (旧版)"},
            {"extension": "json", "description": "JSON数据文件"}
        ]
    }


# 对话管理API
@fastapi_app.post("/conversations")
async def create_new_conversation(request: ConversationRequest = None):
    """创建新对话"""
    title = None
    if request and request.title:
        title = request.title
    
    conversation = create_conversation(title)
    return conversation


@fastapi_app.get("/conversations")
async def list_conversations():
    """获取所有对话列表"""
    return get_all_conversations()


@fastapi_app.get("/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    """获取单个对话详情"""
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    return conversation


@fastapi_app.put("/conversations/{conversation_id}")
async def update_conversation(conversation_id: str, request: ConversationUpdateRequest):
    """更新对话信息"""
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    success = update_conversation_title(conversation_id, request.title)
    if not success:
        raise HTTPException(status_code=500, detail="更新对话失败")
    
    return {"id": conversation_id, "title": request.title, "updated": True}


@fastapi_app.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """删除对话"""
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    success = delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=500, detail="删除对话失败")
    
    return {"id": conversation_id, "deleted": True}


@fastapi_app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages_endpoint(conversation_id: str):
    """获取对话中的所有消息"""
    conversation = get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    messages = get_conversation_messages(conversation_id)
    return messages


@fastapi_app.get("/random-name")
async def get_random_name():
    """获取随机可爱的名字"""
    try:
        return {"name": generate_cute_name()}
    except Exception as e:
        print(f"生成随机名字时出错: {str(e)}")
        return {"name": f"对话 {datetime.now().strftime('%m-%d %H:%M')}"}


@fastapi_app.get("/models")
async def get_available_models():
    """获取可用的模型列表"""
    return {
        "models": [
            {
                "id": "qwen-max",
                "name": "Qwen-Max",
                "description": "最强大的通用模型",
                "supports_vision": False,
                "supports_thinking": False
            },
            {
                "id": "qwen-plus", 
                "name": "Qwen-Plus",
                "description": "平衡性能和效率的模型",
                "supports_vision": False,
                "supports_thinking": False
            },
            {
                "id": "qwen-plus-latest",
                "name": "Qwen-Plus-Latest",
                "description": "最新版本，支持思考和非思考切换",
                "supports_vision": False,
                "supports_thinking": True
            },
            {
                "id": "qwen3-235b-a22b",
                "name": "Qwen3-235B",
                "description": "最新的Qwen3模型，支持深度思考",
                "supports_vision": False,
                "supports_thinking": True
            },
            {
                "id": "qwen-vl-max",
                "name": "Qwen-VL-Max",
                "description": "视觉理解模型，支持图片分析",
                "supports_vision": True,
                "supports_thinking": False
            },
            {
                "id": "deepseek-r1",
                "name": "DeepSeek-R1",
                "description": "深度思考模型，自动开启思考过程",
                "supports_vision": False,
                "supports_thinking": True,
                "force_thinking": True
            }
        ]
    }


@fastapi_app.post("/screenshot")
async def trigger_screenshot():
    """触发截图功能"""
    try:
        # 这里可以集成截图工具，比如使用pyautogui
        # 由于安全考虑，这里返回一个提示信息
        return {
            "success": True,
            "message": "请使用系统截图工具（如Windows的Win+Shift+S）进行截图，然后粘贴到聊天框中"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"截图功能出错: {str(e)}"
        }


#------------------------------------------------------------------------------
# 交互式聊天（命令行）
#------------------------------------------------------------------------------

def interactive_chat():
    """交互式聊天功能
    
    允许用户与智能助手进行对话，支持缓存和搜索功能
    """
    print("\n欢迎使用智能对话系统!")
    print("输入'exit'或'quit'退出对话\n")
    
    history = {"messages": []}
    
    while True:
        # 获取用户输入
        user_input = input("\n你: ")
        
        # 检查是否退出
        if user_input.lower() in ["exit", "quit"]:
            print("\n谢谢使用，再见!")
            break
        
        # 检查是否启用搜索
        web_search = False
        if user_input.startswith("/search "):
            user_input = user_input[8:].strip()  # 移除指令前缀
            web_search = True
            print("(已启用联网搜索)")
        
        # 更新状态
        state = {
            "messages": history["messages"] + [HumanMessage(content=user_input)],
            "web_search_requested": web_search
        }
        
        # 调用应用
        result = app.invoke(state)
        
        # 提取回答并打印
        ai_message = result["messages"][-1]
        print(f"\n助手: {ai_message.content}")
        
        # 更新历史
        history["messages"] = result["messages"]


#------------------------------------------------------------------------------
# 主程序入口
#------------------------------------------------------------------------------

def main():
    """启动应用的主入口点"""
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    print("智能对话系统正在启动...")
    main()
