"""长短期记忆：Postgres + pgvector 实现（LangGraph 官方语义检索方案）。

- 短期记忆（thread 级）：AsyncPostgresSaver 作为 graph 的 checkpointer。
- 长期记忆：AsyncPostgresStore 开启语义索引，两个命名空间：
    ("memory", "episodic")  用户偏好 + 每次行程经验
    ("memory", "semantic")  旅游知识（LLM 提炼）
  embedding 复用 utils.llm.get_embedding_model()（bge-base-zh-v1.5，768 维）。
"""
import datetime
import logging
from uuid import uuid4

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from utils.config import config
from utils.llm import get_embedding_model

logger = logging.getLogger(__name__)

# 语义索引配置：对 value["content"] 字段做 embedding。
# 惰性构建（首次进入 __aenter__ 时才加载 bge 模型），避免 FastAPI server 一 import 就加载 embedding。
_STORE_INDEX = None


def _get_store_index() -> dict:
    global _STORE_INDEX
    if _STORE_INDEX is None:
        _STORE_INDEX = {
            "dims": 768,
            "embed": get_embedding_model(),
            "fields": ["content"],
        }
    return _STORE_INDEX


# 命名空间基础（传入 user_id 时按用户隔离）
_BASE_EPISODIC_NS = ("memory", "episodic")
_BASE_SEMANTIC_NS = ("memory", "semantic")


class MemoryManager:
    """封装长短期记忆的异步上下文管理器。

    用法：
        async with MemoryManager() as memory:
            graph = build_workflow(map_tools, memory=memory)
            ...
    传 user_id 时长期记忆按用户隔离（命名空间加一层 user_id），多用户互不可见；
    不传则保持全局共享（CLI 传参模式）。
    """

    def __init__(self, conn_string: str | None = None, user_id: str | None = None):
        self.conn_string = conn_string or config.DB_URI
        self.user_id = user_id
        self._store = None
        self._store_cm = None
        self._saver = None
        self._saver_cm = None

    # ---- 命名空间（按用户隔离）----
    def _episodic_ns(self):
        return ("memory", self.user_id, "episodic") if self.user_id else _BASE_EPISODIC_NS

    def _semantic_ns(self):
        return ("memory", self.user_id, "semantic") if self.user_id else _BASE_SEMANTIC_NS

    def _notes_ns(self):
        return ("memory", self.user_id, "notes") if self.user_id else ("memory", "notes")

    def _retrieve_prefix(self):
        return ("memory", self.user_id) if self.user_id else ("memory",)

    # ---- 生命周期 ----
    async def __aenter__(self) -> "MemoryManager":
        self._store_cm = AsyncPostgresStore.from_conn_string(
            self.conn_string, index=_get_store_index()
        )
        self._store = await self._store_cm.__aenter__()
        self._saver_cm = AsyncPostgresSaver.from_conn_string(self.conn_string)
        self._saver = await self._saver_cm.__aenter__()
        await self._ensure_pgvector()
        await self._store.setup()
        await self._saver.setup()
        logger.info("记忆服务已连接并初始化（postgres@%s）", self.conn_string.split("@")[-1])
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        for cm in (self._saver_cm, self._store_cm):
            if cm is not None:
                try:
                    await cm.__aexit__(None, None, None)
                except Exception as e:  # noqa: BLE001
                    logger.warning("关闭记忆连接失败：%s", e)

    @property
    def checkpointer(self):
        """供 graph.compile(checkpointer=...) 使用的短期记忆 checkpointer。"""
        return self._saver

    async def _ensure_pgvector(self) -> None:
        """确保 pgvector 扩展存在（store 的向量列依赖它）。"""
        with psycopg.connect(self.conn_string) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ---- 长期记忆：写入 ----
    async def save_episodic(
        self,
        destination: str,
        days: int,
        preference: str,
        itinerary: str,
        weather: str,
    ) -> None:
        """保存一次「经验」：偏好 + 本次行程，作为 episodic 记忆。"""
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefs = preference.strip() or "（未提供明确偏好）"
        content = (
            f"用户对{destination}规划{days}天行程。"
            f"用户偏好：{prefs}。"
            f"天气概况：{weather[:100]}。"
            f"行程摘要：{itinerary[:500]}"
        )
        key = f"{destination}_{now}_{uuid4().hex[:6]}"
        await self._store.aput(
            self._episodic_ns(),
            key,
            {
                "content": content,
                "destination": destination,
                "days": days,
                "preference": preference,
                "itinerary": itinerary,
                "weather": weather,
            },
        )
        logger.info("已保存 episodic 记忆：%s", key)

    async def save_semantic(self, destination: str, knowledge_items: list[str]) -> None:
        """把 LLM 提炼的通用旅游知识逐条保存为 semantic 记忆。"""
        for text in knowledge_items:
            text = text.strip()
            if not text:
                continue
            key = f"{destination}_{uuid4().hex[:6]}"
            await self._store.aput(
                self._semantic_ns(),
                key,
                {
                    "content": f"关于{destination}的旅游知识：{text}",
                    "destination": destination,
                    "knowledge": text,
                },
            )
        if knowledge_items:
            logger.info("已保存 %d 条 semantic 旅游知识", len(knowledge_items))

    # ---- 长期记忆：检索 ----
    async def retrieve(self, query: str, limit: int = 6) -> str:
        """语义检索该用户命名空间下的 episodic/semantic/notes，返回可直接注入 prompt 的文本。"""
        items = await self._store.asearch(
            self._retrieve_prefix(), query=query, limit=limit
        )
        if not items:
            return ""
        labels = {"episodic": "偏好/经验", "semantic": "旅游知识", "notes": "用户笔记"}
        lines = []
        for it in items:
            last = it.namespace[-1] if it.namespace else ""
            label = labels.get(last, "记忆")
            lines.append(f"- [{label}] {it.value.get('content', '')}")
        return "\n".join(lines)

    # ---- 长期记忆：用户自定义笔记 ----
    async def write_user_memory(self, user_id: str, text: str) -> None:
        """写一条用户自定义的长期记忆（如偏好设置），存入该用户命名空间。"""
        text = text.strip()
        if not text:
            return
        key = f"note_{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        await self._store.aput(
            self._notes_ns(),
            key,
            {"content": text, "user_id": user_id},
        )
        logger.info("已保存用户 %s 的长期记忆：%s", user_id, key)
