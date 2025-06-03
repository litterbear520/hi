#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试修复功能的脚本
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_cache_with_sqlite import check_cache, store_qa_pair

def test_cache_functionality():
    """测试缓存功能"""
    print("🧪 测试缓存功能...")
    
    # 测试存储QA对话
    question = "什么是人工智能？"
    answer = "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
    
    print(f"📝 存储问答对: {question}")
    qa_id = store_qa_pair(question, answer)
    
    if qa_id:
        print(f"✅ 成功存储，ID: {qa_id}")
    else:
        print("❌ 存储失败")
        return False
    
    # 测试缓存查询
    print(f"🔍 查询缓存: {question}")
    is_hit, cached_answer, cached_id = check_cache(question)
    
    if is_hit:
        print(f"✅ 缓存命中! ID: {cached_id}")
        print(f"📖 缓存回答: {cached_answer[:50]}...")
        return True
    else:
        print("❌ 缓存未命中")
        return False

def test_similar_query():
    """测试相似查询的缓存命中"""
    print("\n🧪 测试相似查询缓存...")
    
    similar_question = "AI是什么？"
    print(f"🔍 查询相似问题: {similar_question}")
    is_hit, cached_answer, cached_id = check_cache(similar_question)
    
    if is_hit:
        print(f"✅ 相似查询缓存命中! ID: {cached_id}")
        print(f"📖 缓存回答: {cached_answer[:50]}...")
        return True
    else:
        print("❌ 相似查询缓存未命中")
        return False

if __name__ == "__main__":
    print("🚀 开始测试修复功能...\n")
    
    # 测试缓存功能
    cache_test = test_cache_functionality()
    similar_test = test_similar_query()
    
    print(f"\n📊 测试结果:")
    print(f"缓存存储和查询: {'✅ 通过' if cache_test else '❌ 失败'}")
    print(f"相似查询缓存: {'✅ 通过' if similar_test else '❌ 失败'}")
    
    if cache_test and similar_test:
        print("\n🎉 所有测试通过！缓存功能正常工作。")
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试。") 