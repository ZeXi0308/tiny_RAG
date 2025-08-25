#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-12 23:41:04
    

"""

from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

class DeepseekAPIGenerator:
    def __init__(self):
        super().__init__()
    def generate(self, query, retrievalChunks: List[str]) -> str:
        client = OpenAI(api_key="sk-xxx", base_url="https://api.deepseek.com")
        llm_model = "deepseek-reasoner"
        context = ""
        for i, chunk in enumerate(retrievalChunks):
            context += f"reference infomation {i+1}: \n{chunk}\n\n"

        prompt = f"根据参考文档回答问题：{query}\n\n{context}"
        messages=[{"role": "system", "content": ""},
                {"role": "user", "content": f"{prompt}"},]

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                stream=False
            )
            return response
        except Exception as e:
            raise ValueError(f"DeepseekAPIGenerator_retrieval error: {e}")


class DeepseekOllamaGenerator:
    def __init__(self):
        super().__init__()
    def generate(self, query, retrievalChunks: List[str]) -> str:
        llm = OllamaLLM(model="deepseek-r1:7b")
        context = ""
        for i, chunk in enumerate(retrievalChunks):
            context += f"reference infomation {i+1}: \n{chunk}\n\n"

        templatel_prompt = "根据参考文档回答问题{query}\n\n{context}"
        # 创建 RAG Prompt 模板（LCEL）
        QA_PROMPT = PromptTemplate(input_variables=["query", "context"], template=templatel_prompt)
        chain = QA_PROMPT | llm

        try:
            response = chain.invoke({"context": context, "query": query})
            return response
        except Exception as e:
            raise ValueError(f"DeepseekAPIGenerator_retrieval error: {e}")


