#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
    

"""

import os
import sys
import json
from Indexer.Indexer import Indexer
from Retriever.Retriever import Retriever
from Generator.Generator import Generator
from Tools.Query import Query
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)



os.environ["TOKENIZERS_PARALLELISM"] = "false"
with open(os.path.join(current_dir, 'Config/config7.json'), 'r') as js:
    config = json.load(js)
query = "西红柿炒蛋怎么做的？"
# 启用查询优化：将用户问题结构化为任务图（结果会保存为 query_before/after.json）
try:
    user_query, task_dict, final_dict = Query(user_query=query, config=config)
except Exception as e:
    print("查询阶段跳过或失败:", e)
    user_query, task_dict, final_dict = query, {}, {}
indexer = Indexer(config)
# 直接指定示例数据文件，确保可以被处理
DATASET_PATH = os.path.join(current_dir, "Dataset/sample.txt")
textIndex, txtChunks = indexer.index(DATASET_PATH)
# 按关键字参数传入，避免位置参数错位
retriever = Retriever(DocEmbedder=indexer.DocEmbedder.embedder, textIndex=textIndex, config=config)
retrievalChunks = retriever.retrieval(query, txtChunks, imgChunks=None, top_k=3)
print("Top-K 检索片段:")
for i, ck in enumerate((retrievalChunks[0] or [])):
    if isinstance(ck, dict):
        print(f"[{i+1}] {ck.get('page_content','')[:200]}")
    else:
        print(f"[{i+1}] {str(ck)[:200]}")
# 生成阶段：若未配置本地模型或API，可能失败，捕获后提示
try:
    generator = Generator(config)
    result = generator.generate(query, retrievalChunks)
    print("生成结果:", result)
except Exception as e:
    print("生成阶段跳过或失败:", e)
