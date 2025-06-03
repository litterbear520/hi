#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对话管理工具模块 - 提供对话创建、查询、删除等功能
支持随机生成可爱的对话名称
"""

import os
import uuid
import sqlite3
import random
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# 数据库路径
DATA_DIR = "data"  # 数据存储目录
DB_PATH = os.path.join(DATA_DIR, "qa_cache.db")  # SQLite数据库路径

# 随机名称生成用的词汇表
ADJECTIVES = [
    "可爱的", "聪明的", "调皮的", "机智的", "快乐的", "萌萌的", "温柔的", "淘气的", 
    "活泼的", "灵动的", "善良的", "勇敢的", "软萌的", "甜美的", "呆萌的", "慵懒的",
    "雪白的", "蓬松的", "敏捷的", "忠诚的", "憨厚的", "机灵的", "顽皮的", "迷你的"
]

ANIMALS = [
    "猫咪", "小狗", "兔子", "小熊", "松鼠", "小鹿", "浣熊", "小狐狸", 
    "熊猫", "小象", "小猴", "考拉", "海豚", "小鸟", "小羊", "小鹤",
    "企鹅", "小猫头鹰", "小鸭子", "小仓鼠", "小刺猬", "小狮子", "小老虎", "小斑马"
]

def generate_cute_name() -> str:
    """生成一个可爱的名字，由形容词和动物名称组成"""
    adjective = random.choice(ADJECTIVES)
    animal = random.choice(ANIMALS)
    return f"{adjective}{animal}"


def create_conversation(title: Optional[str] = None) -> Dict[str, Any]:
    """创建一个新的对话
    
    Args:
        title: 对话标题，如果为None，则生成一个随机可爱的名称
        
    Returns:
        新创建的对话信息
    """
    conversation_id = str(uuid.uuid4())
    
    if title is None:
        title = generate_cute_name()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, is_active) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, title, current_time, current_time, 1)
        )
        
        conn.commit()
        conn.close()
        
        return {
            "id": conversation_id,
            "title": title,
            "created_at": current_time,
            "updated_at": current_time,
            "is_active": True
        }
    except Exception as e:
        print(f"创建对话时出错: {str(e)}")
        return {}


def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """获取对话信息
    
    Args:
        conversation_id: 对话ID
        
    Returns:
        对话信息
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, title, created_at, updated_at, is_active FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "id": result[0],
                "title": result[1],
                "created_at": result[2],
                "updated_at": result[3],
                "is_active": bool(result[4])
            }
        
        return {}
    except Exception as e:
        print(f"获取对话信息时出错: {str(e)}")
        return {}


def get_all_conversations() -> List[Dict[str, Any]]:
    """获取所有对话
    
    Returns:
        对话列表，按更新时间降序排序
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, title, created_at, updated_at, is_active FROM conversations WHERE is_active = 1 ORDER BY updated_at DESC"
        )
        
        results = cursor.fetchall()
        conn.close()
        
        conversations = []
        for result in results:
            conversations.append({
                "id": result[0],
                "title": result[1],
                "created_at": result[2],
                "updated_at": result[3],
                "is_active": bool(result[4])
            })
        
        return conversations
    except Exception as e:
        print(f"获取所有对话时出错: {str(e)}")
        return []


def update_conversation_title(conversation_id: str, title: str) -> bool:
    """更新对话标题
    
    Args:
        conversation_id: 对话ID
        title: 新标题
        
    Returns:
        是否更新成功
    """
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, current_time, conversation_id)
        )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"更新对话标题时出错: {str(e)}")
        return False


def update_conversation_time(conversation_id: str) -> bool:
    """更新对话的最后修改时间
    
    Args:
        conversation_id: 对话ID
        
    Returns:
        是否更新成功
    """
    max_retries = 3
    retry_delay = 0.5  # 毫秒
    
    for attempt in range(max_retries):
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 设置超时时间为5秒
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (current_time, conversation_id)
            )
            
            conn.commit()
            conn.close()
            
            return True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                # 数据库锁定，等待一段时间后重试
                print(f"数据库锁定，第{attempt+1}次重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数递增等待时间
            else:
                print(f"更新对话时间时出错: {str(e)}")
                return False
        except Exception as e:
            print(f"更新对话时间时出错: {str(e)}")
            return False
    
    return False


def delete_conversation(conversation_id: str) -> bool:
    """删除对话（软删除）
    
    Args:
        conversation_id: 对话ID
        
    Returns:
        是否删除成功
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 软删除，将is_active设为0
        cursor.execute(
            "UPDATE conversations SET is_active = 0 WHERE id = ?",
            (conversation_id,)
        )
        
        conn.commit()
        conn.close()
        
        return True
    except Exception as e:
        print(f"删除对话时出错: {str(e)}")
        return False


def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """获取对话中的所有消息
    
    Args:
        conversation_id: 对话ID
        
    Returns:
        消息列表，按创建时间升序排序
    """
    max_retries = 3
    retry_delay = 0.5  # 毫秒
    
    for attempt in range(max_retries):
        try:
            # 设置超时时间为5秒
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, question, answer, created_at FROM qa_pairs WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,)
            )
            
            results = cursor.fetchall()
            conn.close()
            
            messages = []
            for result in results:
                messages.append({
                    "id": result[0],
                    "user_message": result[1],
                    "ai_message": result[2],
                    "created_at": result[3]
                })
            
            return messages
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                # 数据库锁定，等待一段时间后重试
                print(f"数据库锁定，获取消息第{attempt+1}次重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数递增等待时间
            else:
                print(f"获取对话消息时出错: {str(e)}")
                return []
        except Exception as e:
            print(f"获取对话消息时出错: {str(e)}")
            # 关闭连接，确保不会泄漏
            try:
                if conn:
                    conn.close()
            except:
                pass
            return []
    
    return []


def add_message_to_conversation(conversation_id: str, question: str, answer: str) -> str:
    """添加消息到对话
    
    Args:
        conversation_id: 对话ID
        question: 用户问题
        answer: AI回答
        
    Returns:
        新创建的消息ID
    """
    max_retries = 3
    retry_delay = 0.5  # 毫秒
    message_id = str(uuid.uuid4())
    
    for attempt in range(max_retries):
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 设置超时时间为5秒
            conn = sqlite3.connect(DB_PATH, timeout=5.0)
            cursor = conn.cursor()
            
            # 使用事务管理
            conn.execute("BEGIN IMMEDIATE")
            
            cursor.execute(
                "INSERT INTO qa_pairs (id, question, answer, created_at, conversation_id) VALUES (?, ?, ?, ?, ?)",
                (message_id, question, answer, current_time, conversation_id)
            )
            
            conn.commit()
            conn.close()
            
            # 由于数据库锁定问题，我们在插入成功后才更新对话时间
            update_conversation_time(conversation_id)
            
            return message_id
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                # 数据库锁定，等待一段时间后重试
                print(f"数据库锁定，消息添加第{attempt+1}次重试...")
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数递增等待时间
            else:
                print(f"添加消息到对话时出错: {str(e)}")
                return ""
        except Exception as e:
            print(f"添加消息到对话时出错: {str(e)}")
            # 关闭连接，确保不会泄漏
            try:
                if conn:
                    conn.close()
            except:
                pass
            return ""
    
    return ""


if __name__ == "__main__":
    # 测试代码
    print(generate_cute_name())
    print(generate_cute_name())
    print(generate_cute_name())
