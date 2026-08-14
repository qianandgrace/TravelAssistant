import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env（无论从哪个目录启动都定位到项目根）
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Config:
    """统一的配置类，集中管理所有常量"""
    # 日志持久化存储
    LOG_FILE = "logfile/app.log"
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    MAX_BYTES = 5*1024*1024,
    BACKUP_COUNT = 3

    # PostgreSQL数据库配置参数
    DB_URI = os.getenv("DB_URI", "postgresql://gq210:123456@localhost:5433/postgres?sslmode=disable")
    MIN_SIZE = 5
    MAX_SIZE = 10

    # Redis数据库配置参数
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    SESSION_TIMEOUT = 300
    TTL = 3600
    CELERY_BROKER_URL = "redis://localhost:6379/0"
    TASK_TTL = 3600

    # openai:调用gpt模型,qwen:调用阿里通义千问大模型,deepseek:调用deepseek大模型
    # 默认 qwen；可用环境变量 LLM_TYPE 覆盖
    LLM_TYPE = os.getenv("LLM_TYPE", "qwen")

    # LLM 回退链：某一家无 key / 调用失败时依次切换下一家
    LLM_FALLBACK_CHAIN = ["qwen", "deepseek", "openai"]


    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8001



config = Config()