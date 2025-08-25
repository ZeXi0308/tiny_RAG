#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI服务端 - OneTinyRAG 问答系统
提供RESTful API接口，支持同步和流式问答
"""

import os
import sys
import json
import time
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Indexer.Indexer import Indexer
from Retriever.Retriever import Retriever
from Generator.Generator import Generator
from Tools.Query import Query

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量存储模型组件
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 启动时初始化模型"""
    logger.info("正在初始化 RAG 系统...")
    
    try:
        # 加载配置
        config_path = os.path.join(current_dir, 'Config/config7.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 初始化索引器
        indexer = Indexer(config)
        dataset_path = os.path.join(current_dir, "Dataset/sample.txt")
        
        logger.info("构建向量索引...")
        textIndex, txtChunks = indexer.index(dataset_path)
        
        # 初始化检索器
        retriever = Retriever(
            DocEmbedder=indexer.DocEmbedder.embedder, 
            textIndex=textIndex, 
            config=config
        )
        
        # 初始化生成器
        generator = Generator(config)
        
        # 存储到全局状态
        app_state.update({
            'config': config,
            'indexer': indexer,
            'retriever': retriever,
            'generator': generator,
            'txtChunks': txtChunks,
            'textIndex': textIndex
        })
        
        logger.info("RAG 系统初始化完成!")
        yield
        
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise
    
    # 清理资源
    logger.info("正在清理资源...")

# 创建 FastAPI 应用
app = FastAPI(
    title="OneTinyRAG API",
    description="智能文档问答系统 - 基于检索增强生成(RAG)",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求/响应模型
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1, max_length=500)
    top_k: Optional[int] = Field(3, description="检索数量", ge=1, le=10)
    enable_query_optimization: Optional[bool] = Field(False, description="是否启用查询优化")

class RetrievalResult(BaseModel):
    content: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    success: bool
    query: str
    answer: str
    retrieved_chunks: List[RetrievalResult]
    processing_time: float
    timestamp: str

class StreamResponse(BaseModel):
    type: str  # "chunk" | "metadata" | "error" | "done"
    content: str
    metadata: Optional[Dict[str, Any]] = None

# API 路由
@app.get("/", summary="健康检查")
async def root():
    """API 健康检查"""
    return {
        "message": "OneTinyRAG API 运行正常",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health", summary="系统状态检查")
async def health_check():
    """检查系统组件状态"""
    try:
        is_ready = all(key in app_state for key in ['retriever', 'generator'])
        return {
            "status": "healthy" if is_ready else "initializing",
            "components": {
                "indexer": "indexer" in app_state,
                "retriever": "retriever" in app_state,
                "generator": "generator" in app_state
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="系统状态检查失败")

@app.post("/query", response_model=QueryResponse, summary="同步问答")
async def query_sync(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    同步问答接口
    
    - **query**: 用户问题
    - **top_k**: 检索文档数量 (1-10)
    - **enable_query_optimization**: 是否启用查询优化
    """
    start_time = time.time()
    
    try:
        logger.info(f"收到查询: {request.query}")
        
        # 检查系统状态
        if not all(key in app_state for key in ['retriever', 'generator']):
            raise HTTPException(status_code=503, detail="系统尚未初始化完成，请稍后重试")
        
        # 查询优化（可选）
        user_query = request.query
        if request.enable_query_optimization:
            try:
                user_query, task_dict, final_dict = Query(
                    user_query=request.query, 
                    config=app_state['config']
                )
                logger.info(f"查询优化完成: {user_query}")
            except Exception as e:
                logger.warning(f"查询优化失败，使用原始查询: {e}")
                user_query = request.query
        
        # 检索相关文档
        retrieval_chunks = app_state['retriever'].retrieval(
            user_query, 
            app_state['txtChunks'], 
            imgChunks=None, 
            top_k=request.top_k
        )
        
        # 构建检索结果
        retrieved_results = []
        txt_chunks = retrieval_chunks[0] if retrieval_chunks and retrieval_chunks[0] else []
        
        for i, chunk in enumerate(txt_chunks):
            if isinstance(chunk, dict):
                content = chunk.get('page_content', str(chunk))
                metadata = chunk.get('metadata', {})
            else:
                content = str(chunk)
                metadata = {}
            
            retrieved_results.append(RetrievalResult(
                content=content[:200] + "..." if len(content) > 200 else content,
                metadata=metadata
            ))
        
        # 生成答案
        try:
            answer = app_state['generator'].generate(user_query, retrieval_chunks)
            # 处理生成器返回的不同格式
            if hasattr(answer, 'choices'):  # OpenAI 格式
                answer = answer.choices[0].message.content
            elif not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            answer = f"抱歉，生成答案时出现错误: {str(e)}"
        
        processing_time = time.time() - start_time
        
        # 记录日志
        background_tasks.add_task(
            log_query_result, 
            request.query, 
            answer, 
            processing_time, 
            len(retrieved_results)
        )
        
        return QueryResponse(
            success=True,
            query=request.query,
            answer=answer,
            retrieved_chunks=retrieved_results,
            processing_time=round(processing_time, 3),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")

@app.post("/query/stream", summary="流式问答")
async def query_stream(request: QueryRequest):
    """
    流式问答接口 - 实时返回生成过程
    
    返回 Server-Sent Events (SSE) 格式的流式数据
    """
    
    async def generate_stream():
        start_time = time.time()
        
        try:
            # 发送开始信号
            yield f"data: {json.dumps({'type': 'start', 'content': '开始处理查询...'})}\n\n"
            
            # 检索阶段
            yield f"data: {json.dumps({'type': 'retrieval', 'content': '正在检索相关文档...'})}\n\n"
            
            user_query = request.query
            if request.enable_query_optimization:
                try:
                    user_query, _, _ = Query(user_query=request.query, config=app_state['config'])
                    yield f"data: {json.dumps({'type': 'optimization', 'content': f'查询优化: {user_query}'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'warning', 'content': f'查询优化失败: {str(e)}'})}\n\n"
            
            retrieval_chunks = app_state['retriever'].retrieval(
                user_query, 
                app_state['txtChunks'], 
                imgChunks=None, 
                top_k=request.top_k
            )
            
            # 发送检索结果
            txt_chunks = retrieval_chunks[0] if retrieval_chunks and retrieval_chunks[0] else []
            yield f"data: {json.dumps({'type': 'retrieval_done', 'content': f'检索到 {len(txt_chunks)} 个相关文档'})}\n\n"
            
            # 生成阶段
            yield f"data: {json.dumps({'type': 'generation', 'content': '正在生成答案...'})}\n\n"
            
            try:
                answer = app_state['generator'].generate(user_query, retrieval_chunks)
                if hasattr(answer, 'choices'):
                    answer = answer.choices[0].message.content
                elif not isinstance(answer, str):
                    answer = str(answer)
                
                # 模拟流式输出（可根据实际生成器调整）
                words = answer.split()
                for i, word in enumerate(words):
                    yield f"data: {json.dumps({'type': 'chunk', 'content': word + ' '})}\n\n"
                    if i % 5 == 0:  # 每5个词休息一下
                        time.sleep(0.1)
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': f'生成失败: {str(e)}'})}\n\n"
                return
            
            # 发送完成信号
            processing_time = time.time() - start_time
            yield f"data: {json.dumps({'type': 'done', 'content': '处理完成', 'metadata': {'processing_time': round(processing_time, 3)}})}\n\n"
            
        except Exception as e:
            logger.error(f"流式处理失败: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'处理失败: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/config", summary="获取系统配置")
async def get_config():
    """获取当前系统配置信息"""
    if 'config' not in app_state:
        raise HTTPException(status_code=503, detail="系统尚未初始化")
    
    config = app_state['config'].copy()
    # 移除敏感信息
    return {
        "chunker": config.get("chunker", {}),
        "embedder": config.get("embedder", {}),
        "retriever": config.get("retriever", {}),
        "generator": {
            "type": config.get("generator", {}).get("type"),
            # 不返回API密钥等敏感信息
        }
    }

# 后台任务
async def log_query_result(query: str, answer: str, processing_time: float, chunk_count: int):
    """记录查询结果到日志"""
    logger.info(f"查询完成 - 问题: {query[:50]}... | 处理时间: {processing_time:.3f}s | 检索片段数: {chunk_count}")

# 启动服务
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
