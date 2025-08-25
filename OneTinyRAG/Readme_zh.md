```markdown
# OneTinyRAG - 智能文档助手


实现核心的检索增强生成(RAG)能力，现开放社区驱动开发！

## 开发计划
- [2025.4.23] 🔥 多模态处理
- [2025.4.23] 🔥 元数据分块
- [2025.4.10] 🔥 多格式文件自动处理 ✅
- [2025.4.10] 🔥 Ollama + Deepseek 本地部署 ✅
- [2025.4.10] 🔥 语义分块扩展(Spacy & NLTK) ✅

## 快速开始
### 安装依赖
```bash
conda activate DeepseekRag
pip install -r requirements.txt
```

### 运行项目
```bash
bash app.sh
```

## 项目结构

```
/OneTinyRAG/
├── app.sh                 # 主启动脚本
├── app.py                 # 主程序入口
├── Readme.md              # 英文文档
├── Readme_zh.md           # 中文文档
├── Indexer/               # 索引模块
│   ├── Indexer.py         # 索引管理核心
│   ├── Embedder.py        # 嵌入模型抽象层
│   ├── DataProcessor.py   # 文档处理器
│   └── Chunker.py         # 文本分块策略
├── Generator/             # 生成模块
│   └── Generator.py       
├── Retriever/             # 检索模块
│   └── Retriever.py
└── config/                # 配置目录
│   └── config.json
└── Tutorial/              # 教程文档
  ├── Ollama_zh.md         
  └── Ollama.md
```

## 配置说明
通过`config/config1.json`配置：
• 嵌入模型参数
• 分块大小/重叠量
• 相似度阈值
• 大语言模型API端点

## 扩展指南
1. 在`Indexer.py`中注册组件
2. 在`DataProcessor.py`中添加文档处理器
3. 在`Chunker.py`中实现新的文本分割器
4. 在`Embedder.py`中扩展嵌入模型
5. 在`Retriever.py`中注册组件  
6. 在`Retriever.py`中开发新的检索器
7. 在`Generator.py`中注册组件
8. 在`Generator.py`中实现新的生成器

## 📊 性能基准测试
