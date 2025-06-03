import os
from openai import OpenAI


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 测试问题
question = "简单解释一下什么是人工智能？"

print("="*60)
print("测试1: 开启思考模式 (enable_thinking=True)")
print("="*60)

# 开启思考模式
completion_with_thinking = client.chat.completions.create(
    model="qwen-plus-latest",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ],
    stream=True,
    extra_body={"enable_thinking": True}
)

thinking_content = ""
response_content = ""

for chunk in completion_with_thinking:
    if hasattr(chunk, 'choices') and chunk.choices:
        delta = chunk.choices[0].delta
        
        # 检查思考内容 (reasoning_content)
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            thinking_content += delta.reasoning_content
            print(f"[思考]: {delta.reasoning_content}", end="")
        
        # 检查回答内容
        if hasattr(delta, 'content') and delta.content:
            response_content += delta.content
            print(delta.content, end="")

print(f"\n\n开启思考模式总结:")
print(f"- 思考内容长度: {len(thinking_content)}")
print(f"- 回答内容长度: {len(response_content)}")

print("\n" + "="*60)
print("测试2: 关闭思考模式 (enable_thinking=False)")
print("="*60)

# 关闭思考模式
completion_without_thinking = client.chat.completions.create(
    model="qwen-plus-latest",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ],
    stream=True,
    extra_body={"enable_thinking": False}
)

thinking_content_off = ""
response_content_off = ""

for chunk in completion_without_thinking:
    if hasattr(chunk, 'choices') and chunk.choices:
        delta = chunk.choices[0].delta
        
        # 检查思考内容
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            thinking_content_off += delta.reasoning_content
            print(f"[思考]: {delta.reasoning_content}", end="")
        
        # 检查回答内容
        if hasattr(delta, 'content') and delta.content:
            response_content_off += delta.content
            print(delta.content, end="")

print(f"\n\n关闭思考模式总结:")
print(f"- 思考内容长度: {len(thinking_content_off)}")
print(f"- 回答内容长度: {len(response_content_off)}")

print("\n" + "="*60)
print("对比结果:")
print("="*60)
print(f"开启思考 - 思考长度: {len(thinking_content)}, 回答长度: {len(response_content)}")
print(f"关闭思考 - 思考长度: {len(thinking_content_off)}, 回答长度: {len(response_content_off)}")
print(f"思考功能是否有效: {'是' if len(thinking_content) > len(thinking_content_off) else '否'}")
