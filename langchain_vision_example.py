"""
LangChain图像识别使用示例
展示如何使用封装好的QwenVL模型进行图像分析
"""

from image_test import QwenVLChatModel
from langchain_core.messages import HumanMessage


def simple_image_analysis():
    """简单的图像分析示例"""
    # 创建模型实例
    llm = QwenVLChatModel(streaming=True)
    
    # 创建包含图像的消息
    message = HumanMessage(
        content="这是什么",
        additional_kwargs={
            "images": ["https://img.shetu66.com/2023/07/27/1690436791750269.png"]
        }
    )
    
    # 生成响应
    print("分析结果：")
    result = llm._generate([message])
    return result.generations[0].message.content


def batch_image_analysis():
    """批量图像分析示例"""
    llm = QwenVLChatModel(streaming=False)
    
    # 多个图像URL
    image_urls = [
        "https://img.shetu66.com/2023/07/27/1690436791750269.png",
        # 可以添加更多图像URL
    ]
    
    # 创建包含多个图像的消息
    message = HumanMessage(
        content="请分别描述这些图片的内容",
        additional_kwargs={"images": image_urls}
    )
    
    result = llm._generate([message])
    print("批量分析结果：")
    print(result.generations[0].message.content)


def interactive_image_chat():
    """交互式图像对话示例"""
    llm = QwenVLChatModel(streaming=False)
    image_url = "https://img.shetu66.com/2023/07/27/1690436791750269.png"
    
    questions = [
        "这张图片中有什么？",
        "图片的颜色主要是什么？",
        "这个物体可能的用途是什么？",
        "图片的背景是什么样的？"
    ]
    
    print("=== 交互式图像对话 ===")
    for i, question in enumerate(questions, 1):
        print(f"\n问题 {i}: {question}")
        
        message = HumanMessage(
            content=question,
            additional_kwargs={"images": [image_url]}
        )
        
        result = llm._generate([message])
        print(f"回答: {result.generations[0].message.content}")


if __name__ == "__main__":
    print("=== LangChain图像识别示例 ===\n")
    
    # 1. 简单图像分析
    print("1. 简单图像分析（流式输出）:")
    simple_image_analysis()
    
    print("\n" + "="*50 + "\n")
    
    # 2. 批量图像分析
    print("2. 批量图像分析:")
    batch_image_analysis()
    
    print("\n" + "="*50 + "\n")
    
    # 3. 交互式图像对话
    interactive_image_chat() 