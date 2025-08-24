# OneTinyRAG

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)
![FAISS](https://img.shields.io/badge/FAISS-1.7+-orange.svg)

**轻量级、可插拔的检索增强生成（RAG）框架**

支持 BM25+Dense 混合检索 | FAISS 向量索引 | 多格式文档处理 | FastAPI 服务化

</div>

## ✨ 特性

- 🔍 **混合检索**: BM25 关键词检索 + Dense 语义检索，支持多种融合策略
- 📚 **多格式支持**: PDF、TXT、JSON 文档解析与处理
- 🧩 **可插拔架构**: 配置驱动的组件化设计，易于扩展
- ⚡ **高性能**: FAISS 向量索引，支持批量处理与并发
- 🌐 **服务化**: FastAPI RESTful API，支持流式输出
- 🤖 **多模型**: 支持 DeepSeek API 和本地 Ollama 部署
- 🔧 **智能分块**: 递归、Token、语义、元数据多种分块策略
- 📊 **任务编排**: 基于依赖图的智能工作流调度

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 8GB+ RAM (推荐)
- GPU (可选，用于本地模型推理)

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/OneTinyRAG.git
cd OneTinyRAG

# 安装依赖
pip install -r requirements.txt

# 或使用一键启动脚本
chmod +x start.sh
./start.sh
```

### 基础使用

#### 1. 命令行模式

```bash
# 使用默认配置运行
python OneTinyRAG/app.py

# 使用混合检索配置
python OneTinyRAG/app.py --config OneTinyRAG/Config/config_hybrid.json
```

#### 2. API 服务模式

```bash
# 启动 FastAPI 服务
python api_server.py

# 测试 API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是检索增强生成？", "config_name": "config_hybrid"}'
```

#### 3. Python 集成

```python
from OneTinyRAG.Indexer.Indexer import Indexer
from OneTinyRAG.Retriever.Retriever import Retriever
from OneTinyRAG.Generator.Generator import Generator

# 初始化组件
config = {...}  # 加载配置
indexer = Indexer(config)
retriever = Retriever(config)
generator = Generator(config)

# 构建索引
embedder, index, documents = indexer.index("path/to/documents")

# 检索相关文档
results = retriever.retrieval(query="用户问题", embedder=embedder, index=index)

# 生成回答
answer = generator.generate(query="用户问题", retrieved_docs=results)
```

## 📁 项目结构

```
OneTinyRAG/
├── Indexer/           # 文档处理与索引构建
│   ├── DataProcessor.py    # PDF/TXT/JSON 解析器
│   ├── Chunker.py          # 多种分块策略
│   ├── Embedder.py         # 向量化与 FAISS 索引
│   └── Indexer.py          # 索引构建门面类
├── Retriever/         # 检索模块
│   ├── Retrieval.py        # 基础检索器
│   ├── HybridRetriever.py  # BM25+Dense 混合检索
│   └── Retriever.py        # 检索器统一接口
├── Generator/         # 生成模块
│   ├── Generator.py        # 生成器接口
│   └── Generate.py         # DeepSeek/Ollama 实现
├── Tools/             # 查询优化与工作流
│   ├── Query.py            # 查询结构化
│   ├── Workflow.py         # 任务图调度
│   └── Utils.py            # 工具函数
├── Config/            # 配置文件
├── Dataset/           # 示例数据
└── app.py            # 主程序入口
```

## ⚙️ 配置说明

### 基础配置 (config7.json)

```json
{
  "chunker": {
    "type": "MetaDataChunker",
    "params": {
      "chunk_size": 128,
      "language": "chinese"
    }
  },
  "embedder": {
    "type": "MetaDataEmbedder",
    "params": {
      "model_name": "BAAI/bge-small-zh-v1.5"
    }
  },
  "retriever": {
    "type": "CosinRetriever",
    "params": {
      "top_k": 5
    }
  },
  "generator": {
    "type": "DeepSeekGenerator",
    "params": {
      "api_key": "your_api_key"
    }
  }
}
```

### 混合检索配置 (config_hybrid.json)

```json
{
  "retriever": {
    "type": "HybridRetriever",
    "params": {
      "top_k": 10,
      "bm25_weight": 0.3,
      "dense_weight": 0.7,
      "language": "chinese",
      "fusion_method": "weighted_sum",
      "normalization_method": "min_max"
    }
  }
}
```

## 🔧 核心功能

### 1. 混合检索策略

- **BM25 检索**: 基于词频的关键词匹配
- **Dense 检索**: 基于语义向量的相似度检索
- **融合方法**: 加权求和、调和平均、几何平均、RRF 等
- **归一化**: Min-Max、Z-score、Rank 标准化

### 2. 智能分块策略

- **RecursiveChunker**: 递归字符分割
- **TokenChunker**: 基于 Token 数量分割
- **SemanticChunker**: 基于 spaCy/NLTK 的语义分割
- **MetaDataChunker**: 保留元数据的滑窗分块

### 3. 多格式文档处理

- **PDF**: 提取文本内容和元数据
- **TXT**: 智能文本清洗和格式化
- **JSON**: 学术论文格式特化处理

### 4. 任务编排系统

- **依赖分析**: 自动构建任务依赖图
- **并发执行**: 基于 asyncio 的高效调度
- **容错处理**: 优雅降级和错误恢复

## 📊 API 文档

### 核心端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/query` | POST | 标准查询接口 |
| `/stream` | POST | 流式输出接口 |
| `/health` | GET | 健康检查 |
| `/configs` | GET | 获取可用配置 |

### 查询示例

```bash
# 标准查询
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RAG 系统如何工作？",
    "config_name": "config_hybrid",
    "top_k": 5
  }'

# 流式查询
curl -X POST "http://localhost:8000/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "解释检索增强生成",
    "config_name": "config7"
  }'
```

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t onetinyrag:latest .

# 运行容器
docker run -p 8000:8000 -v $(pwd)/OneTinyRAG/Dataset:/app/OneTinyRAG/Dataset onetinyrag:latest

# 使用 Docker Compose
docker-compose up -d
```

## 🧪 测试与评估

```bash
# 运行混合检索测试
python test_hybrid_retrieval.py

# 运行完整演示
python demo.py
```

## 🛠️ 开发指南

### 添加新的检索器

1. 在 `Retriever/` 目录下创建新的检索器类
2. 继承基础接口并实现 `retrieval_txt` 方法
3. 在 `Mappers/Mappers.py` 中注册新检索器
4. 更新配置文件支持新参数

### 添加新的分块策略

1. 在 `Indexer/Chunker.py` 中实现新的分块类
2. 继承 `BaseChunker` 并实现 `chunk_text` 方法
3. 在映射器中注册新分块器

## 📈 性能优化

- **批量处理**: 支持文档批量嵌入
- **索引持久化**: FAISS 索引保存与加载
- **内存优化**: 大文档流式处理
- **并发控制**: 可配置的线程池大小

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📝 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

**⭐ 如果这个项目对你有帮助，请给个 Star！⭐**

</div>
