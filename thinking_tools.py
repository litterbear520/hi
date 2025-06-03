#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度思考工具 - 基于LangChain 0.3和通义千问的思考模式
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
import dotenv
from openai import OpenAI
import json

# 加载环境变量
dotenv.load_dotenv()

class ThinkingTool:
    """基于原生OpenAI API的深度思考工具"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        初始化思考工具
        
        Args:
            api_key: API密钥，默认从环境变量获取
            base_url: API基础URL，默认使用通义千问兼容模式
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
    def think_and_respond(
        self, 
        question: str, 
        system_prompt: str = "You are a helpful assistant.",
        model: str = "qwen-plus-latest",
        enable_thinking: bool = True,
        stream: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        使用深度思考模式回答问题
        
        Args:
            question: 用户问题
            system_prompt: 系统提示词
            model: 使用的模型
            enable_thinking: 是否启用思考模式
            stream: 是否使用流式输出
            
        Yields:
            包含思考内容和回答内容的字典
        """
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                stream=stream,
                extra_body={"enable_thinking": enable_thinking}
            )
            
            for chunk in completion:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    result = {
                        "thinking": None,
                        "content": None,
                        "type": None
                    }
                    
                    # 检查思考内容
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        result["thinking"] = delta.reasoning_content
                        result["type"] = "thinking"
                        yield result
                    
                    # 检查回答内容
                    if hasattr(delta, 'content') and delta.content:
                        result["content"] = delta.content
                        result["type"] = "content"
                        yield result
                        
        except Exception as e:
            yield {
                "thinking": None,
                "content": f"思考过程中出现错误: {str(e)}",
                "type": "error"
            }
    
    def simple_think(
        self, 
        question: str, 
        system_prompt: str = "You are a helpful assistant.",
        model: str = "qwen-plus-latest",
        enable_thinking: bool = True
    ) -> Dict[str, str]:
        """
        简单的思考模式，返回完整的思考内容和回答
        
        Args:
            question: 用户问题
            system_prompt: 系统提示词
            model: 使用的模型
            enable_thinking: 是否启用思考模式
            
        Returns:
            包含thinking_content和response_content的字典
        """
        thinking_content = ""
        response_content = ""
        
        try:
            for chunk in self.think_and_respond(
                question=question,
                system_prompt=system_prompt,
                model=model,
                enable_thinking=enable_thinking,
                stream=True
            ):
                if chunk["type"] == "thinking" and chunk["thinking"]:
                    thinking_content += chunk["thinking"]
                elif chunk["type"] == "content" and chunk["content"]:
                    response_content += chunk["content"]
                elif chunk["type"] == "error":
                    return {
                        "thinking_content": "",
                        "response_content": chunk["content"],
                        "error": True
                    }
            
            return {
                "thinking_content": thinking_content,
                "response_content": response_content,
                "error": False
            }
            
        except Exception as e:
            return {
                "thinking_content": "",
                "response_content": f"思考过程中出现错误: {str(e)}",
                "error": True
            }

# 创建全局实例
thinking_tool = ThinkingTool()

def get_thinking_response(question: str, enable_thinking: bool = True) -> Dict[str, str]:
    """
    获取带思考过程的回答（兼容接口）
    
    Args:
        question: 用户问题
        enable_thinking: 是否启用思考模式
        
    Returns:
        包含思考内容和回答内容的字典
    """
    return thinking_tool.simple_think(
        question=question,
        enable_thinking=enable_thinking
    )

def stream_thinking_response(question: str, enable_thinking: bool = True):
    """
    流式获取带思考过程的回答（兼容接口）
    
    Args:
        question: 用户问题
        enable_thinking: 是否启用思考模式
        
    Yields:
        思考和回答的流式内容
    """
    return thinking_tool.think_and_respond(
        question=question,
        enable_thinking=enable_thinking
    )

# 测试函数
def test_thinking_tool():
    """测试思考工具"""
    print("="*60)
    print("测试深度思考工具")
    print("="*60)
    
    question = "简单解释一下什么是人工智能？"
    
    # 测试开启思考模式
    print("开启思考模式:")
    result = get_thinking_response(question, enable_thinking=True)
    print(f"思考内容长度: {len(result['thinking_content'])}")
    print(f"回答内容长度: {len(result['response_content'])}")
    print(f"思考内容: {result['thinking_content'][:100]}...")
    print(f"回答内容: {result['response_content'][:100]}...")
    
    print("\n" + "="*60)
    
    # 测试关闭思考模式
    print("关闭思考模式:")
    result_off = get_thinking_response(question, enable_thinking=False)
    print(f"思考内容长度: {len(result_off['thinking_content'])}")
    print(f"回答内容长度: {len(result_off['response_content'])}")
    print(f"回答内容: {result_off['response_content'][:100]}...")
    
    print(f"\n思考功能是否有效: {'是' if len(result['thinking_content']) > 0 else '否'}")

if __name__ == "__main__":
    test_thinking_tool() 