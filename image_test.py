import os
from typing import List, Dict, Any, Optional
import dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI

dotenv.load_dotenv()


class QwenVLChatModel(BaseChatModel):
    """通义千问VL模型的LangChain封装"""
    
    model_name: str = "qwen-vl-max"
    api_key: Optional[str] = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    streaming: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 使用object.__setattr__来避免Pydantic的字段验证
        object.__setattr__(self, 'client', OpenAI(
            api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.base_url,
        ))
    
    @property
    def _llm_type(self) -> str:
        return "qwen-vl"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成聊天响应"""
        # 转换LangChain消息格式为OpenAI格式
        openai_messages = self._convert_messages(messages)
        
        try:
            if self.streaming:
                return self._generate_streaming(openai_messages, run_manager, **kwargs)
            else:
                return self._generate_non_streaming(openai_messages, **kwargs)
        except Exception as e:
            raise ValueError(f"生成响应时出错: {str(e)}")
    
    def _generate_streaming(
        self, 
        messages: List[Dict], 
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> ChatResult:
        """流式生成响应"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            **kwargs
        )
        
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                if run_manager:
                    run_manager.on_llm_new_token(content)
                else:
                    # 如果没有回调管理器，直接打印
                    print(content, end="")
        
        if not run_manager:
            print()  # 添加换行符
            
        message = AIMessage(content=full_response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _generate_non_streaming(self, messages: List[Dict], **kwargs) -> ChatResult:
        """非流式生成响应"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            **kwargs
        )
        
        content = completion.choices[0].message.content
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        """将LangChain消息格式转换为OpenAI格式"""
        openai_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                # 检查消息内容是否包含图像
                if hasattr(message, 'additional_kwargs') and 'images' in message.additional_kwargs:
                    # 处理包含图像的消息
                    content = []
                    if message.content:
                        content.append({"type": "text", "text": message.content})
                    
                    for image_url in message.additional_kwargs['images']:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        })
                    
                    openai_messages.append({
                        "role": "user",
                        "content": content
                    })
                else:
                    # 纯文本消息
                    openai_messages.append({
                        "role": "user",
                        "content": message.content
                    })
            elif isinstance(message, AIMessage):
                openai_messages.append({
                    "role": "assistant",
                    "content": message.content
                })
        
        return openai_messages


def main():
    """主函数 - 使用LangChain框架实现原有功能"""
    # 创建模型实例，启用流式输出
    llm = QwenVLChatModel(streaming=True)
    
    # 创建包含图像的消息
    message = HumanMessage(
        content="这是什么",
        additional_kwargs={
            "images": ["https://img.shetu66.com/2023/07/27/1690436791750269.png"]
        }
    )
    
    # 生成响应（流式输出）
    result = llm._generate([message])
    
    return result.generations[0].message.content


def demo_different_questions():
    """演示不同问题的图像分析"""
    llm = QwenVLChatModel(streaming=False)  # 非流式，便于展示结果
    
    image_url = "https://img.shetu66.com/2023/07/27/1690436791750269.png"
    
    questions = [
        "这是什么？",
        "请详细描述这张图片",
        "图片中有什么颜色？",
        "这张图片的主要内容是什么？"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n=== 问题 {i}: {question} ===")
        message = HumanMessage(
            content=question,
            additional_kwargs={"images": [image_url]}
        )
        
        result = llm._generate([message])
        print(f"回答: {result.generations[0].message.content}")


if __name__ == "__main__":
    print("=== 使用LangChain框架的图像识别 ===")
    print("流式输出结果:")
    main()
    
    # 演示不同问题
    demo_different_questions()