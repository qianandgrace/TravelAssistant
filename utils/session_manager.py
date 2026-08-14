"""Redis 会话管理器：存储用户/会话/任务状态，支持动态 TTL 与故障恢复。

键设计：
  sess:users                        -> SET(user_id)
  sess:user:{uid}:sessions          -> ZSET(session_id -> last_updated)，按最近排序
  sess:user:{uid}:{sid}:tasks       -> SET(task_id)
  sess:task:{uid}:{sid}:{tid}       -> HASH{status, last_query, last_response(JSON), last_updated}
  sess:user:{uid}:{sid}:ttl         -> 会话级 TTL 覆盖（秒，可选）

会话状态枚举：idle / pending / running / interrupted / completed / error / not_found。
会话状态存 Redis -> 客户端断线重连、服务端/worker 重启后都可查询与恢复（图状态在 Postgres checkpointer）。
"""
import json
import logging
import time
from typing import Any, Optional

import redis.asyncio as aioredis

from utils.config import Config

logger = logging.getLogger(__name__)

STATUS_IDLE = "idle"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_INTERRUPTED = "interrupted"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"
STATUS_NOT_FOUND = "not_found"

_USERS_KEY = "sess:users"


def _user_sessions_key(user_id: str) -> str:
    return f"sess:user:{user_id}:sessions"


def _session_tasks_key(user_id: str, session_id: str) -> str:
    return f"sess:user:{user_id}:{session_id}:tasks"


def _task_key(user_id: str, session_id: str, task_id: str) -> str:
    return f"sess:task:{user_id}:{session_id}:{task_id}"


def _session_ttl_key(user_id: str, session_id: str) -> str:
    return f"sess:user:{user_id}:{session_id}:ttl"


class RedisSessionManager:
    """基于 redis.asyncio 的会话管理器。"""

    def __init__(self, url: Optional[str] = None):
        self._redis = aioredis.from_url(url or Config.REDIS_URL, decode_responses=True)

    async def close(self) -> None:
        await self._redis.aclose()

    @staticmethod
    def _now() -> float:
        return time.time()

    # ---- 基础存在性 ----
    async def user_id_exists(self, user_id: str) -> bool:
        return bool(await self._redis.sismember(_USERS_KEY, user_id))

    async def session_id_exists(self, user_id: str, session_id: str) -> bool:
        return await self._redis.zscore(_user_sessions_key(user_id), session_id) is not None

    async def session_task_id_exists(self, user_id: str, session_id: str, task_id: str) -> bool:
        return bool(await self._redis.sismember(_session_tasks_key(user_id, session_id), task_id))

    # ---- TTL ----
    async def _resolve_ttl(self, user_id: str, session_id: str, ttl: Optional[int]) -> int:
        """TTL 优先级：显式参数 > 会话级覆盖 > 全局默认。"""
        if ttl:
            return ttl
        override = await self._redis.get(_session_ttl_key(user_id, session_id))
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        return int(Config.TTL)

    async def set_session_ttl(self, user_id: str, session_id: Optional[str], ttl: int) -> int:
        """动态调整会话过期时间；session_id 为空则作用于该用户全部会话。

        Returns: 受影响的任务数。
        """
        affected = 0
        session_ids = [session_id] if session_id else await self.get_all_session_ids(user_id)
        for sid in session_ids:
            await self._redis.set(_session_ttl_key(user_id, sid), ttl)
            task_ids = await self._redis.smembers(_session_tasks_key(user_id, sid))
            for tid in task_ids:
                await self._redis.expire(_task_key(user_id, sid, tid), ttl)
                affected += 1
        logger.info("已调整 %s 的会话过期时间为 %ds，涉及 %d 个任务", user_id, ttl, affected)
        return affected

    # ---- 会话/任务写入 ----
    async def create_session(
        self,
        user_id: str,
        session_id: str,
        task_id: str,
        status: str = STATUS_IDLE,
        last_query: Optional[str] = None,
        last_response: Optional[dict] = None,
        last_updated: Optional[float] = None,
        ttl: Optional[int] = None,
    ) -> None:
        await self._redis.sadd(_USERS_KEY, user_id)
        await self._redis.zadd(_user_sessions_key(user_id), {session_id: last_updated or self._now()})
        await self._redis.sadd(_session_tasks_key(user_id, session_id), task_id)
        await self._write_task(user_id, session_id, task_id, status, last_query,
                               last_response, last_updated, ttl)

    async def update_session(
        self,
        user_id: str,
        session_id: str,
        task_id: str,
        status: Optional[str] = None,
        last_query: Optional[str] = None,
        last_response: Optional[dict] = None,
        last_updated: Optional[float] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """更新任务状态（未提供的字段保持不变），并刷新会话活跃时间与 TTL。"""
        await self._redis.zadd(_user_sessions_key(user_id), {session_id: last_updated or self._now()})
        await self._write_task(user_id, session_id, task_id, status, last_query,
                               last_response, last_updated, ttl)

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        user_id: str,
        session_id: str,
        last_response: Optional[dict] = None,
        last_query: Optional[str] = None,
    ) -> None:
        await self.update_session(user_id, session_id, task_id, status=status,
                                  last_query=last_query, last_response=last_response)

    async def _write_task(
        self,
        user_id: str,
        session_id: str,
        task_id: str,
        status: Optional[str],
        last_query: Optional[str],
        last_response: Optional[dict],
        last_updated: Optional[float],
        ttl: Optional[int],
    ) -> None:
        key = _task_key(user_id, session_id, task_id)
        mapping: dict[str, Any] = {"last_updated": str(last_updated or self._now())}
        if status is not None:
            mapping["status"] = status
        if last_query is not None:
            mapping["last_query"] = last_query
        if last_response is not None:
            mapping["last_response"] = json.dumps(last_response, ensure_ascii=False, default=str)
        if mapping:
            await self._redis.hset(key, mapping=mapping)
        ttl_sec = await self._resolve_ttl(user_id, session_id, ttl)
        if ttl_sec:
            await self._redis.expire(key, ttl_sec)

    # ---- 查询 ----
    async def get_session_by_task(self, user_id: str, session_id: str, task_id: str) -> dict:
        data = await self._redis.hgetall(_task_key(user_id, session_id, task_id))
        result: dict = {}
        for k, v in data.items():
            if k == "last_response":
                try:
                    result[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    result[k] = None
            elif k == "last_updated":
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = None
            else:
                result[k] = v
        return result

    async def get_all_session_ids(self, user_id: str) -> list[str]:
        """该用户的全部会话，按最近活跃排序（新的在前）。"""
        members = await self._redis.zrevrange(_user_sessions_key(user_id), 0, -1)
        return list(members)

    async def get_user_active_session_id(self, user_id: str) -> str:
        """最近一次活跃的会话 ID。"""
        members = await self._redis.zrevrange(_user_sessions_key(user_id), 0, 0)
        return members[0] if members else ""

    async def get_task_status(self, user_id: str, session_id: str) -> list[dict]:
        """指定会话下所有任务的 id + 状态。"""
        task_ids = await self._redis.smembers(_session_tasks_key(user_id, session_id))
        result = []
        for tid in task_ids:
            data = await self.get_session_by_task(user_id, session_id, tid)
            result.append({"task_id": tid, "status": data.get("status", "not_found")})
        result.sort(key=lambda x: x["task_id"])
        return result

    async def get_session_count(self) -> int:
        user_ids = await self._redis.smembers(_USERS_KEY)
        total = 0
        for uid in user_ids:
            total += await self._redis.zcard(_user_sessions_key(uid))
        return total

    async def get_all_users_session_ids(self) -> dict[str, list[str]]:
        user_ids = await self._redis.smembers(_USERS_KEY)
        result = {}
        for uid in user_ids:
            result[uid] = await self.get_all_session_ids(uid)
        return result

    # ---- 删除 ----
    async def delete_task(self, user_id: str, session_id: str, task_id: str) -> None:
        await self._redis.delete(_task_key(user_id, session_id, task_id))
        await self._redis.srem(_session_tasks_key(user_id, session_id), task_id)

    async def delete_session(self, user_id: str, session_id: str, task_id: Optional[str] = None) -> None:
        """删除指定任务，或（task_id=None）删除整个会话（含其全部任务）。"""
        if task_id:
            await self.delete_task(user_id, session_id, task_id)
            return
        task_ids = await self._redis.smembers(_session_tasks_key(user_id, session_id))
        for tid in task_ids:
            await self._redis.delete(_task_key(user_id, session_id, tid))
        await self._redis.delete(_session_tasks_key(user_id, session_id))
        await self._redis.zrem(_user_sessions_key(user_id), session_id)
        await self._redis.delete(_session_ttl_key(user_id, session_id))


_session_manager: Optional[RedisSessionManager] = None


def get_session_manager() -> RedisSessionManager:
    """返回全局单例（FastAPI lifespan 与 Celery worker 共用）。"""
    global _session_manager
    if _session_manager is None:
        _session_manager = RedisSessionManager()
    return _session_manager
