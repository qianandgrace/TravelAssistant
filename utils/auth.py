"""用户注册/登录：Postgres users 表 + pbkdf2 密码哈希。

- user_id 用 UUID（内部标识，避免用户名撞长期记忆命名空间组分），username 仅用于展示与登录。
- 密码哈希格式：`pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>`，用 stdlib，无需额外依赖。
- 连接池为模块级单例，server 启动时 open_user_store()，关闭时 close_user_store()。
"""
import hashlib
import hmac
import logging
import secrets
import uuid
from dataclasses import dataclass
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from utils.config import config

logger = logging.getLogger(__name__)

_HASH_ALGO = "pbkdf2_sha256"
_PBKDF2_ITERATIONS = 100_000


@dataclass
class User:
    """已认证用户。id 为内部 UUID（user_id）。"""
    id: str
    username: str


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return f"{_HASH_ALGO}${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, expected = stored.split("$")
    except (ValueError, AttributeError):
        return False
    if algo != _HASH_ALGO:
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
    )
    return hmac.compare_digest(digest.hex(), expected)


# ---------- 连接池（模块级单例） ----------
_pool: Optional[AsyncConnectionPool] = None


def _get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            config.DB_URI,
            min_size=1,
            max_size=5,
            open=False,
            kwargs={"row_factory": dict_row},
        )
    return _pool


async def init_user_store() -> None:
    """建 users 表并打开连接池（服务启动时调用）。"""
    pool = _get_pool()
    if pool.closed:
        await pool.open()
    async with pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
    logger.info("用户表 users 就绪")


async def close_user_store() -> None:
    global _pool
    if _pool is not None and not _pool.closed:
        await _pool.close()
    _pool = None


# ---------- 业务操作 ----------
async def create_user(username: str, password: str) -> tuple[bool, str, Optional[User]]:
    """注册。返回 (是否成功, 消息, User|None)。重名 / 并发重复插入由唯一约束兜底。"""
    username = (username or "").strip()
    pool = _get_pool()
    uid = str(uuid.uuid4())
    phash = _hash_password(password)
    async with pool.connection() as conn:
        try:
            await conn.execute(
                "INSERT INTO users (id, username, password_hash) VALUES (%s, %s, %s)",
                (uid, username, phash),
            )
        except Exception as e:  # noqa: BLE001 - 唯一约束冲突等
            logger.warning("注册用户 %s 失败：%s", username, e)
            return False, "用户名已存在", None
    logger.info("注册成功：%s (user_id=%s)", username, uid)
    return True, "注册成功", User(id=uid, username=username)


async def verify_user(username: str, password: str) -> Optional[User]:
    """登录校验。成功返回 User，否则 None。"""
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = %s",
            ((username or "").strip(),),
        )
        row = await cur.fetchone()
    if row and _verify_password(password, row["password_hash"]):
        return User(id=str(row["id"]), username=row["username"])
    return None
