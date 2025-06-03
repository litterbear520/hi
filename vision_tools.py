#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片理解工具 - 基于LangChain和阿里云通义千问视觉模型
"""

import os
import base64
from typing import List, Dict, Any, Optional, Type
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from io import BytesIO
from PIL import Image
import dotenv

# 加载环境变量
dotenv.load_dotenv()

class VisionAnalysisInput(BaseModel):
    """图片分析工具的输入模型"""
    image_urls: List[str] = Field(description="图片URL列表")
    question: str = Field(description="关于图片的问题", default="请描述这张图片的内容")

class VisionAnalysisTool(BaseTool):
    """图片理解分析工具"""
    
    name: str = "vision_analysis"
    description: str = "分析图片内容的工具，可以识别图片中的对象、场景、文字等信息。当用户上传图片或提供图片URL时，自动调用此工具进行图片理解。"
    args_schema: Type[BaseModel] = VisionAnalysisInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化视觉模型（不作为Pydantic字段）
        object.__setattr__(self, '_vision_model', ChatTongyi(
            model="qwen-vl-max",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            temperature=0.7,
            max_tokens=8192,
            streaming=False
        ))
    
    @property
    def vision_model(self):
        """获取视觉模型"""
        return getattr(self, '_vision_model', None)
    
    def _run(self, image_urls: List[str], question: str = "请描述这张图片的内容") -> str:
        """运行图片分析"""
        try:
            if not image_urls:
                return "没有提供图片URL"
            
            # 创建包含图片的消息内容
            message_content = []
            
            # 添加文本问题
            message_content.append({
                "type": "text", 
                "text": question
            })
            
            # 添加图片 - 修复格式以符合千问VL模型要求
            for image_url in image_urls:
                if image_url.startswith('data:image'):
                    # 如果是base64格式的data URL，直接使用
                    message_content.append({
                        "type": "image",
                        "image": image_url
                    })
                else:
                    # 如果是普通URL，使用image_url格式
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
            
            # 创建消息
            message = HumanMessage(content=message_content)
            
            # 调用模型
            response = self.vision_model.invoke([message])
            
            return response.content
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"图片分析详细错误: {error_detail}")
            return f"图片分析失败: {str(e)}"
    
    async def _arun(self, image_urls: List[str], question: str = "请描述这张图片的内容") -> str:
        """异步运行图片分析"""
        return self._run(image_urls, question)

def process_image_data(image_data: str, image_name: str = "image") -> str:
    """
    处理base64图片数据，上传到临时存储并返回URL
    这里简化处理，实际项目中可能需要上传到云存储
    """
    try:
        # 解码base64图片
        image_bytes = base64.b64decode(image_data)
        
        # 这里应该上传到云存储，为了演示直接返回data URL
        return f"data:image/jpeg;base64,{image_data}"
        
    except Exception as e:
        print(f"处理图片数据时出错: {str(e)}")
        return None

def create_vision_agent():
    """创建具有图片理解能力的智能体"""
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
    
    # 创建工具列表
    tools = [VisionAnalysisTool()]
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手，具有图片理解能力。当用户提供图片时，你需要：
1. 使用vision_analysis工具来分析图片内容
2. 根据分析结果回答用户的问题
3. 如果用户没有明确问题，请详细描述图片内容

请注意：
- 优先使用工具来分析图片
- 基于分析结果给出准确的回答
- 如果图片分析失败，请告知用户"""),
        ("user", "{input}"),
        ("assistant", "{agent_scratchpad}")
    ])
    
    # 创建模型
    model = ChatTongyi(
        model="qwen-max",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.7,
        max_tokens=8192,
        streaming=True
    )
    
    # 创建agent
    agent = create_tool_calling_agent(model, tools, prompt)
    
    # 创建executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def analyze_images_with_agent(image_urls: List[str], question: str = "请分析这些图片") -> str:
    """使用智能体分析图片"""
    try:
        # 创建智能体
        agent = create_vision_agent()
        
        # 构建输入
        input_text = f"{question}\n图片URL: {', '.join(image_urls)}"
        
        # 执行分析
        result = agent.invoke({"input": input_text})
        
        return result.get("output", "分析失败")
        
    except Exception as e:
        print(f"智能体分析图片时出错: {str(e)}")
        return f"分析失败: {str(e)}"

if __name__ == "__main__":
    # 测试代码
    test_image_url = "https://img.shetu66.com/2023/07/27/1690436791750269.png"
    
    # 测试工具
    tool = VisionAnalysisTool()
    result = tool._run([test_image_url], "这是什么？")
    print("工具测试结果:", result)
    
    # 测试智能体
    agent_result = analyze_images_with_agent([test_image_url], "请详细描述这张图片")
    print("智能体测试结果:", agent_result) 