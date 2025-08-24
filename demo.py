#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OneTinyRAG 演示脚本
展示系统的核心功能和性能
"""

import os
import sys
import json
import time
import requests
from typing import List, Dict

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'OneTinyRAG'))

def demo_cli_mode():
    """命令行模式演示"""
    print("=" * 60)
    print("🚀 OneTinyRAG 命令行演示")
    print("=" * 60)
    
    # 导入模块
    from OneTinyRAG.Indexer.Indexer import Indexer
    from OneTinyRAG.Retriever.Retriever import Retriever
    from OneTinyRAG.Generator.Generator import Generator
    from OneTinyRAG.Tools.Query import Query
    
    # 加载配置
    config_path = os.path.join(current_dir, 'OneTinyRAG/Config/config7.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("📋 当前配置:")
    print(f"  - 分块器: {config['chunker']['type']}")
    print(f"  - 嵌入模型: {config['embedder']['docEmbedder']['params']['model_name']}")
    print(f"  - 检索器: {config['retriever']['type']}")
    print(f"  - 生成器: {config['generator']['type']}")
    
    # 演示问题
    demo_queries = [
        "什么是机器学习？",
        "深度学习的主要应用有哪些？", 
        "如何评估模型性能？",
        "什么是过拟合问题？"
    ]
    
    print("\n🔧 正在初始化系统...")
    start_time = time.time()
    
    try:
        # 初始化组件
        indexer = Indexer(config)
        dataset_path = os.path.join(current_dir, 'OneTinyRAG/Dataset/sample.txt')
        
        print("📁 构建向量索引...")
        textIndex, txtChunks = indexer.index(dataset_path)
        
        retriever = Retriever(
            DocEmbedder=indexer.DocEmbedder.embedder,
            textIndex=textIndex,
            config=config
        )
        
        generator = Generator(config)
        
        init_time = time.time() - start_time
        print(f"✅ 初始化完成! 耗时: {init_time:.2f}s")
        
        # 处理演示问题
        print("\n🎯 开始问答演示:")
        print("-" * 40)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 问题 {i}: {query}")
            
            query_start = time.time()
            
            # 检索
            retrieval_chunks = retriever.retrieval(
                query, txtChunks, imgChunks=None, top_k=3
            )
            
            # 显示检索结果
            txt_chunks = retrieval_chunks[0] if retrieval_chunks and retrieval_chunks[0] else []
            print(f"🔍 检索到 {len(txt_chunks)} 个相关片段:")
            
            for j, chunk in enumerate(txt_chunks[:2], 1):  # 只显示前2个
                content = chunk.get('page_content', str(chunk)) if isinstance(chunk, dict) else str(chunk)
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"   {j}. {preview}")
            
            # 生成答案
            try:
                print("🤖 正在生成答案...")
                answer = generator.generate(query, retrieval_chunks)
                
                # 处理不同格式的返回值
                if hasattr(answer, 'choices'):
                    answer_text = answer.choices[0].message.content
                elif isinstance(answer, str):
                    answer_text = answer
                else:
                    answer_text = str(answer)
                
                query_time = time.time() - query_start
                print(f"💡 答案: {answer_text}")
                print(f"⏱️  处理时间: {query_time:.2f}s")
                
            except Exception as e:
                print(f"❌ 生成答案失败: {e}")
            
            print("-" * 40)
        
        # 性能统计
        print(f"\n📊 系统性能统计:")
        print(f"  - 总初始化时间: {init_time:.2f}s")
        print(f"  - 平均查询时间: ~2-5s (取决于模型)")
        print(f"  - 索引文档数: {len(txtChunks)}")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        return False
    
    return True

def demo_api_mode(base_url: str = "http://localhost:8000"):
    """API模式演示"""
    print("=" * 60)
    print("🌐 OneTinyRAG API 演示")
    print("=" * 60)
    
    # 检查API服务状态
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API 服务运行正常")
        else:
            print("❌ API 服务状态异常")
            return False
    except requests.exceptions.RequestException:
        print("❌ 无法连接到 API 服务")
        print("💡 请先启动 API 服务: python OneTinyRAG/api_server.py")
        return False
    
    # 演示问题
    demo_queries = [
        "什么是人工智能？",
        "机器学习有哪些类型？"
    ]
    
    print("\n🎯 API 问答演示:")
    print("-" * 40)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 问题 {i}: {query}")
        
        # 同步API调用
        try:
            payload = {
                "query": query,
                "top_k": 3,
                "enable_query_optimization": False
            }
            
            start_time = time.time()
            response = requests.post(f"{base_url}/query", json=payload, timeout=30)
            api_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"🔍 检索片段数: {len(result['retrieved_chunks'])}")
                print(f"💡 答案: {result['answer']}")
                print(f"⏱️  API 响应时间: {api_time:.2f}s")
                print(f"📊 服务端处理时间: {result['processing_time']}s")
            else:
                print(f"❌ API 调用失败: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络请求失败: {e}")
        
        print("-" * 40)
    
    return True

def print_usage():
    """打印使用说明"""
    print("🔧 OneTinyRAG 使用说明")
    print("=" * 50)
    print("1. 命令行模式 (推荐新手):")
    print("   python demo.py cli")
    print()
    print("2. API 服务模式:")
    print("   # 启动服务")
    print("   python OneTinyRAG/api_server.py")
    print("   # 测试API")
    print("   python demo.py api")
    print()
    print("3. Docker 模式:")
    print("   docker-compose up --build")
    print("   python demo.py api")
    print()
    print("📋 配置文件: OneTinyRAG/Config/config7.json")
    print("📁 数据文件: OneTinyRAG/Dataset/sample.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)
    
    mode = sys.argv[1].lower()
    
    if mode == "cli":
        success = demo_cli_mode()
    elif mode == "api":
        base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
        success = demo_api_mode(base_url)
    else:
        print("❌ 未知模式，请使用 'cli' 或 'api'")
        print_usage()
        sys.exit(1)
    
    if success:
        print("\n🎉 演示完成!")
    else:
        print("\n❌ 演示失败，请检查配置和依赖")
