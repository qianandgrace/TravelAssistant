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
    MAX_BYTES = 5 * 1024 * 1024
    BACKUP_COUNT = 3
    # 日志级别（DEBUG/INFO/WARNING/ERROR），默认 DEBUG 便于观察各流程节点
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

    # PostgreSQL数据库配置参数
    DB_URI = os.getenv("DB_URI", "postgresql://gq210:123456@localhost:5433/postgres?sslmode=disable")
    MIN_SIZE = 5
    MAX_SIZE = 10

    # Redis数据库配置参数（docker-compose 实际映射为宿主机 6379）
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    # 会话/任务在 Redis 中的过期时间（秒），可用 /session/ttl 接口动态调整
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "300"))
    TTL = int(os.getenv("SESSION_TTL", "3600"))
    TASK_TTL = int(os.getenv("TASK_TTL", "3600"))

    # Celery：broker 与 result backend 都走 Redis，可用环境变量覆盖
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

    # openai:调用gpt模型,qwen:调用阿里通义千问大模型,deepseek:调用deepseek大模型
    # 默认 qwen；可用环境变量 LLM_TYPE 覆盖
    LLM_TYPE = os.getenv("LLM_TYPE", "qwen")

    # LLM 回退链：某一家无 key / 调用失败时依次切换下一家
    LLM_FALLBACK_CHAIN = ["qwen", "deepseek", "openai"]

    # LLM 单次调用超时（秒）。规划等大输出场景 qwen-max 生成 3~4k token 需要几十秒，
    # 30s 的默认超时会导致 planner 必然超时（ReadTimeout）+ 90s 空等后才回退到下一家。
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))


    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8001



config = Config()