# TravelAssistant 云端镜像：FastAPI 后端 + Celery worker + Gradio 前端共用此镜像
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8

# 云端 embedding 走通义 API（EMBEDDING_PROVIDER=qwen），无需 torch/sentence-transformers，镜像轻量
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 复制代码（.dockerignore 已排除 .env / .git / 本地数据）
COPY . .

EXPOSE 8001 7860

# 默认启动 FastAPI；worker 用 docker-compose 覆盖 command
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
