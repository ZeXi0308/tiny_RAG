# OneTinyRAG Docker 镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app/OneTinyRAG
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY OneTinyRAG/ /app/OneTinyRAG/
COPY requirements.txt /app/

# 创建缓存目录
RUN mkdir -p /app/.cache/huggingface

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 预下载模型（可选，减少首次启动时间）
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')" || true

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "/app/OneTinyRAG/api_server.py"]
