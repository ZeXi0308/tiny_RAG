#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
混合检索效果对比测试
比较 Dense-only vs BM25+Dense 的检索效果
"""

import os
import sys
import json
import time
from typing import List, Dict

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'OneTinyRAG'))

from OneTinyRAG.Indexer.Indexer import Indexer
from OneTinyRAG.Retriever.Retriever import Retriever

def test_retrieval_comparison():
    """对比测试：Dense vs Hybrid检索"""
    print("🚀 混合检索效果对比测试")
    print("=" * 60)
    
    # 测试查询
    test_queries = [
        "西红柿炒蛋怎么做？",
        "机器学习的基本原理",
        "Python编程入门",
        "深度学习和神经网络"
    ]
    
    # 加载配置
    config_path_original = os.path.join(current_dir, 'OneTinyRAG/Config/config7.json')
    config_path_hybrid = os.path.join(current_dir, 'OneTinyRAG/Config/config_hybrid.json')
    
    with open(config_path_original, 'r', encoding='utf-8') as f:
        config_original = json.load(f)
    
    with open(config_path_hybrid, 'r', encoding='utf-8') as f:
        config_hybrid = json.load(f)
    
    print("📋 配置加载完成")
    print(f"  - 原始配置: {config_original['retriever']['type']}")
    print(f"  - 混合配置: {config_hybrid['retriever']['type']}")
    
    # 构建索引（两种配置共享）
    print("\n🔧 构建向量索引...")
    indexer = Indexer(config_original)
    dataset_path = os.path.join(current_dir, 'OneTinyRAG/Dataset/sample.txt')
    textIndex, txtChunks = indexer.index(dataset_path)
    print(f"✅ 索引构建完成，共 {len(txtChunks)} 个文档片段")
    
    # 初始化两种检索器
    print("\n🎯 初始化检索器...")
    
    # Dense-only 检索器
    retriever_dense = Retriever(
        DocEmbedder=indexer.DocEmbedder.embedder,
        textIndex=textIndex,
        config=config_original
    )
    
    # Hybrid 检索器
    retriever_hybrid = Retriever(
        DocEmbedder=indexer.DocEmbedder.embedder,
        textIndex=textIndex,
        config=config_hybrid
    )
    
    print("✅ 检索器初始化完成")
    
    # 对比测试
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 测试 {i}: {query}")
        print("-" * 40)
        
        # Dense检索
        start_time = time.time()
        dense_results = retriever_dense.retrieval(query, txtChunks, imgChunks=None, top_k=3)
        dense_time = time.time() - start_time
        
        # Hybrid检索
        start_time = time.time()
        hybrid_results = retriever_hybrid.retrieval(query, txtChunks, imgChunks=None, top_k=3)
        hybrid_time = time.time() - start_time
        
        # 显示结果对比
        print(f"🔍 Dense-only 检索 (耗时: {dense_time:.3f}s):")
        dense_chunks = dense_results[0] if dense_results and dense_results[0] else []
        for j, chunk in enumerate(dense_chunks[:3], 1):
            content = chunk.get('page_content', str(chunk)) if isinstance(chunk, dict) else str(chunk)
            print(f"  [{j}] {content[:80]}...")
        
        print(f"\n🔄 Hybrid 检索 (耗时: {hybrid_time:.3f}s):")
        hybrid_chunks = hybrid_results[0] if hybrid_results and hybrid_results[0] else []
        for j, chunk in enumerate(hybrid_chunks[:3], 1):
            content = chunk.get('page_content', str(chunk)) if isinstance(chunk, dict) else str(chunk)
            scores = chunk.get('scores', {}) if isinstance(chunk, dict) else {}
            
            score_info = ""
            if scores:
                score_info = f" [Hybrid: {scores.get('hybrid', 0):.3f}, BM25: {scores.get('bm25', 0):.3f}, Dense: {scores.get('dense', 0):.3f}]"
            
            print(f"  [{j}] {content[:60]}...{score_info}")
        
        # 记录结果
        results.append({
            'query': query,
            'dense_time': dense_time,
            'hybrid_time': hybrid_time,
            'dense_results': len(dense_chunks),
            'hybrid_results': len(hybrid_chunks)
        })
    
    # 总结
    print("\n📊 测试总结")
    print("=" * 60)
    
    avg_dense_time = sum(r['dense_time'] for r in results) / len(results)
    avg_hybrid_time = sum(r['hybrid_time'] for r in results) / len(results)
    
    print(f"平均检索时间:")
    print(f"  - Dense-only: {avg_dense_time:.3f}s")
    print(f"  - Hybrid:     {avg_hybrid_time:.3f}s")
    print(f"  - 时间开销: {(avg_hybrid_time/avg_dense_time-1)*100:+.1f}%")
    
    print(f"\n功能对比:")
    print(f"  ✅ Dense-only: 纯语义匹配")
    print(f"  ✅ Hybrid:     语义 + 关键词匹配，带详细评分")
    
    print(f"\n🎯 结论:")
    print(f"  - 混合检索提供了更丰富的匹配策略")
    print(f"  - BM25权重可调，适应不同查询类型")
    print(f"  - 检索结果包含详细评分，便于调试和分析")
    
    return results

if __name__ == "__main__":
    try:
        results = test_retrieval_comparison()
        print(f"\n🎉 测试完成！共执行 {len(results)} 个查询的对比测试")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

