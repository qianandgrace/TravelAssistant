"""Pydantic 请求/响应模型，供 FastAPI 后端（server.py）使用。"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    """注册新用户。"""
    username: str = Field(min_length=2, max_length=32, pattern=r"^[\w-]+$")
    password: str = Field(min_length=6, max_length=64)


class LoginRequest(BaseModel):
    """用户名 + 密码登录。"""
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=64)


class AgentRequest(BaseModel):
    """提交一次异步智能体运行。"""
    user_id: str
    session_id: Optional[str] = None   # 不传则新建
    task_id: Optional[str] = None      # 不传则新建
    query: str                         # 自然语言输入，如「9月2号到9月5号去天津，轻松点」
    system_message: Optional[str] = None
    ttl: Optional[int] = None          # 可选：本次会话的 Redis 过期时间（秒）


class ResumeRequest(BaseModel):
    """恢复被中断的智能体运行（HITL）。command 直接作为 Command(resume=...) 的值。"""
    user_id: str
    session_id: str
    task_id: str
    command: dict[str, Any] = Field(default_factory=dict)  # 如 {"action": "accept"}


class LongMemRequest(BaseModel):
    user_id: str
    memory_info: str


class TtlRequest(BaseModel):
    """动态调整会话过期时间。"""
    user_id: str
    session_id: Optional[str] = None   # 不传则作用于该用户全部会话
    ttl: int = Field(ge=60, le=86400 * 30)


# ---- 响应模型 ----
class SessionInfoResponse(BaseModel):
    session_ids: list[str] = Field(default_factory=list)


class ActiveSessionInfoResponse(BaseModel):
    active_session_id: str = ""


class TaskInfoResponse(BaseModel):
    task_ids: list[dict] = Field(default_factory=list)  # [{"task_id":..., "status":...}]


class SessionStatusResponse(BaseModel):
    user_id: str
    session_id: str
    task_id: str
    status: str = "not_found"
    last_query: Optional[str] = None
    last_updated: Optional[float] = None
    last_response: Optional[dict] = None
    message: Optional[str] = None


class SystemInfoResponse(BaseModel):
    sessions_count: int = 0
    active_users: dict[str, list[str]] = Field(default_factory=dict)
