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

# 语义索引配置：对 value["content"] 字段做 embedding
_STORE_INDEX = {
    "dims": 768,
    "embed": get_embedding_model(),
    "fields": ["content"],
}

EPISODIC_NS = ("memory", "episodic")
SEMANTIC_NS = ("memory", "semantic")


class MemoryManager:
    """封装长短期记忆的异步上下文管理器。

    用法：
        async with MemoryManager() as memory:
            graph = build_workflow(map_tools, memory=memory)
            ...
    """

    def __init__(self, conn_string: str | None = None):
        self.conn_string = conn_string or config.DB_URI
        self._store = None
        self._store_cm = None
        self._saver = None
        self._saver_cm = None

    # ---- 生命周期 ----
    async def __aenter__(self) -> "MemoryManager":
        self._store_cm = AsyncPostgresStore.from_conn_string(
            self.conn_string, index=_STORE_INDEX
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
            EPISODIC_NS,
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
                SEMANTIC_NS,
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
        """语义检索 episodic + semantic，返回可直接注入 prompt 的文本。"""
        items = await self._store.asearch(
            ("memory",), query=query, limit=limit
        )
        if not items:
            return ""
        lines = []
        for it in items:
            ns = "偏好/经验" if it.namespace == EPISODIC_NS else "旅游知识"
            lines.append(f"- [{ns}] {it.value.get('content', '')}")
        return "\n".join(lines)
