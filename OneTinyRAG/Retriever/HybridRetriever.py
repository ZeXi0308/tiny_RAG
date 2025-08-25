#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Retriever: BM25 + Dense Vector Retrieval
Combines lexical search (BM25) with semantic search (dense vectors) for improved recall
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
import jieba
import re
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    chunk_id: int
    bm25_score: float
    dense_score: float
    hybrid_score: float
    metadata: Optional[Dict[str, Any]] = None

class HybridRetriever:
    """混合检索器：BM25 + Dense Vector"""
    
    def __init__(self, 
                 dense_embedder=None, 
                 dense_index=None,
                 bm25_weight: float = 0.6,
                 dense_weight: float = 0.4,
                 language: str = "chinese",
                 fusion_method: str = "weighted_sum",
                 normalization_method: str = "min_max"):
        """
        初始化混合检索器
        
        Args:
            dense_embedder: 句向量模型
            dense_index: FAISS向量索引
            bm25_weight: BM25权重
            dense_weight: 密集向量权重
            language: 语言设置，影响分词策略
            fusion_method: 融合方法 ("weighted_sum", "harmonic_mean", "geometric_mean", "max", "rrf")
            normalization_method: 归一化方法 ("min_max", "z_score", "rank")
        """
        self.dense_embedder = dense_embedder
        self.dense_index = dense_index
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.language = language
        self.fusion_method = fusion_method
        self.normalization_method = normalization_method
        
        # BM25索引将在构建时初始化
        self.bm25_index = None
        self.corpus_texts = []
        self.tokenized_corpus = []
        
    def _tokenize_text(self, text: str) -> List[str]:
        """文本分词"""
        if self.language == "chinese":
            # 中文分词
            tokens = list(jieba.cut(text))
            # 过滤停用词和标点
            tokens = [token.strip() for token in tokens 
                     if token.strip() and not re.match(r'^[^\w]+$', token)]
        else:
            # 英文分词
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        return tokens
    
    def build_bm25_index(self, chunks: List) -> None:
        """构建BM25索引"""
        print(f"🔧 构建 BM25 索引，共 {len(chunks)} 个文档片段...")
        
        # 提取文本内容
        self.corpus_texts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get('page_content', str(chunk))
            elif hasattr(chunk, 'page_content'):
                text = chunk.page_content
            else:
                text = str(chunk)
            self.corpus_texts.append(text)
        
        # 分词
        self.tokenized_corpus = [self._tokenize_text(text) for text in self.corpus_texts]
        
        # 构建BM25索引
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        print(f"✅ BM25 索引构建完成")
    
    def _bm25_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """BM25检索"""
        if self.bm25_index is None:
            raise ValueError("BM25索引未构建，请先调用 build_bm25_index()")
        
        # 查询分词
        query_tokens = self._tokenize_text(query)
        
        # BM25评分
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # 获取top_k结果
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        results = [(idx, bm25_scores[idx]) for idx in top_indices]
        
        return results
    
    def _dense_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """密集向量检索"""
        if self.dense_embedder is None or self.dense_index is None:
            return []
        
        # 查询向量化
        query_embedding = self.dense_embedder.encode(query, normalize_embeddings=True)
        query_embedding = np.array([query_embedding])
        
        # FAISS检索
        distances, indices = self.dense_index.search(query_embedding, top_k)
        
        # 转换为余弦相似度（FAISS IndexFlatIP 返回内积）
        results = [(indices[0][i], distances[0][i]) for i in range(len(indices[0]))]
        
        return results
    
    def _normalize_scores(self, scores: List[float], method: str = "min_max") -> List[float]:
        """
        分数归一化到[0,1]
        
        Args:
            scores: 原始分数列表
            method: 归一化方法 ("min_max", "z_score", "rank")
        """
        if not scores:
            return scores
        
        scores_array = np.array(scores)
        
        if method == "min_max":
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score == min_score:
                return [1.0] * len(scores)
            return ((scores_array - min_score) / (max_score - min_score)).tolist()
            
        elif method == "z_score":
            # Z-score标准化，然后sigmoid映射到[0,1]
            mean_score = scores_array.mean()
            std_score = scores_array.std()
            if std_score == 0:
                return [0.5] * len(scores)
            z_scores = (scores_array - mean_score) / std_score
            return (1 / (1 + np.exp(-z_scores))).tolist()
            
        elif method == "rank":
            # 基于排名的归一化
            ranks = np.argsort(np.argsort(scores_array))
            return (ranks / (len(scores) - 1)).tolist()
            
        else:
            return self._normalize_scores(scores, "min_max")
    
    def _compute_hybrid_score(self, bm25_score: float, dense_score: float, 
                             fusion_method: str = "weighted_sum") -> float:
        """
        计算混合评分
        
        Args:
            bm25_score: BM25分数 [0,1]
            dense_score: Dense分数 [0,1]  
            fusion_method: 融合方法
                - "weighted_sum": 加权求和
                - "harmonic_mean": 调和平均
                - "geometric_mean": 几何平均
                - "max": 取最大值
                - "rrf": Reciprocal Rank Fusion
        """
        if fusion_method == "weighted_sum":
            return self.bm25_weight * bm25_score + self.dense_weight * dense_score
            
        elif fusion_method == "harmonic_mean":
            # 调和平均：2ab/(a+b)
            if bm25_score + dense_score == 0:
                return 0.0
            return 2 * bm25_score * dense_score / (bm25_score + dense_score)
            
        elif fusion_method == "geometric_mean":
            # 几何平均：sqrt(ab)
            return np.sqrt(bm25_score * dense_score)
            
        elif fusion_method == "max":
            # 取最大值
            return max(bm25_score, dense_score)
            
        elif fusion_method == "rrf":
            # Reciprocal Rank Fusion (简化版)
            # RRF = 1/(k + rank), 这里用分数近似排名
            k = 60  # RRF常数
            bm25_rrf = 1 / (k + (1 - bm25_score) * 100)
            dense_rrf = 1 / (k + (1 - dense_score) * 100)
            return bm25_rrf + dense_rrf
            
        else:
            return self.bm25_weight * bm25_score + self.dense_weight * dense_score
    
    def hybrid_search(self, 
                     query: str, 
                     chunks: List,
                     top_k: int = 3,
                     retrieval_top_k: int = 50) -> List[RetrievalResult]:
        """
        混合检索主函数
        
        Args:
            query: 查询文本
            chunks: 文档片段列表
            top_k: 最终返回的结果数量
            retrieval_top_k: 每个检索器的召回数量
            
        Returns:
            排序后的检索结果列表
        """
        print(f"🔍 混合检索: '{query[:30]}...'")
        
        # 确保BM25索引已构建
        if self.bm25_index is None:
            self.build_bm25_index(chunks)
        
        # BM25检索
        bm25_results = self._bm25_search(query, retrieval_top_k)
        bm25_scores_dict = {idx: score for idx, score in bm25_results}
        
        # Dense检索
        dense_results = self._dense_search(query, retrieval_top_k)
        dense_scores_dict = {idx: score for idx, score in dense_results}
        
        # 合并候选集
        all_candidates = set(bm25_scores_dict.keys()) | set(dense_scores_dict.keys())
        
        # 分数归一化
        bm25_scores = [bm25_scores_dict.get(idx, 0.0) for idx in all_candidates]
        dense_scores = [dense_scores_dict.get(idx, 0.0) for idx in all_candidates]
        
        normalized_bm25 = self._normalize_scores(bm25_scores)
        normalized_dense = self._normalize_scores(dense_scores)
        
        # 混合评分（支持多种融合策略）
        hybrid_results = []
        for i, candidate_idx in enumerate(all_candidates):
            if candidate_idx >= len(chunks):
                continue
                
            bm25_score = normalized_bm25[i]
            dense_score = normalized_dense[i]
            
            # 融合策略选择
            hybrid_score = self._compute_hybrid_score(
                bm25_score, dense_score, 
                fusion_method=getattr(self, 'fusion_method', 'weighted_sum')
            )
            
            # 获取原始内容
            chunk = chunks[candidate_idx]
            if isinstance(chunk, dict):
                content = chunk.get('page_content', str(chunk))
                metadata = chunk.get('metadata', {})
            elif hasattr(chunk, 'page_content'):
                content = chunk.page_content
                metadata = getattr(chunk, 'metadata', {})
            else:
                content = str(chunk)
                metadata = {}
            
            result = RetrievalResult(
                content=content,
                chunk_id=candidate_idx,
                bm25_score=bm25_score,
                dense_score=dense_score,
                hybrid_score=hybrid_score,
                metadata=metadata
            )
            hybrid_results.append(result)
        
        # 按混合分数排序
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # 返回top_k结果
        final_results = hybrid_results[:top_k]
        
        print(f"📊 检索完成: BM25({len(bm25_results)}) + Dense({len(dense_results)}) → Top-{len(final_results)}")
        
        # 打印调试信息
        for i, result in enumerate(final_results):
            print(f"  [{i+1}] Score: {result.hybrid_score:.3f} "
                  f"(BM25: {result.bm25_score:.3f}, Dense: {result.dense_score:.3f}) "
                  f"| {result.content[:60]}...")
        
        return final_results

class HybridRetrievalAdapter:
    """适配器：让HybridRetriever兼容原有的Retriever接口"""
    
    def __init__(self, embedder, index, config: dict = None):
        """
        初始化适配器
        
        Args:
            embedder: 句向量模型
            index: FAISS索引
            config: 配置字典（可选，兼容旧接口）
        """
        # 兼容旧接口：如果第三个参数不是config而是其他类型，则设为默认配置
        if config is None or not isinstance(config, dict):
            config = {}
        
        retriever_config = config.get("retriever", {})
        hybrid_config = retriever_config.get("hybrid", {})
        
        self.hybrid_retriever = HybridRetriever(
            dense_embedder=embedder,
            dense_index=index,
            bm25_weight=hybrid_config.get("bm25_weight", 0.6),
            dense_weight=hybrid_config.get("dense_weight", 0.4),
            language=hybrid_config.get("language", "chinese")
        )
        
    def retrieval_txt(self, query: str, chunks: List, top_k: int = 3) -> List:
        """
        文本检索方法（兼容 CosinRetriever 接口）
        
        Returns:
            检索结果列表
        """
        # 执行混合检索
        hybrid_results = self.hybrid_retriever.hybrid_search(
            query=query,
            chunks=chunks,
            top_k=top_k
        )
        
        # 转换为原有格式
        text_results = []
        for result in hybrid_results:
            chunk_dict = {
                'page_content': result.content,
                'metadata': result.metadata or {},
                'chunk_id': result.chunk_id,
                'scores': {
                    'bm25': result.bm25_score,
                    'dense': result.dense_score,
                    'hybrid': result.hybrid_score
                }
            }
            text_results.append(chunk_dict)
        
        return text_results
    
    def retrieval_img(self, query: str, chunks: List, top_k: int = 3) -> List:
        """
        图像检索方法（兼容接口，暂不实现）
        """
        return []
    
    def retrieval(self, query: str, chunks: List, imgChunks=None, top_k: int = 3) -> List:
        """
        兼容原有接口的检索方法
        
        Returns:
            [text_results, img_results] 格式，保持向后兼容
        """
        text_results = self.retrieval_txt(query, chunks, top_k)
        return [text_results, imgChunks]  # 保持原有的 [text, img] 格式
