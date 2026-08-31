# TravelAssistant 上云部署指南

> 目标：**只在云端不推理，本地保留 torch**。云端 embedding 走通义 API（`EMBEDDING_PROVIDER=qwen`），镜像不带 torch；本地仍是 bge 本地模型。本指南基于阿里云轻量服务器 `47.93.40.210` + Docker CE。

## 1. 上云步骤（相对本地的差异）

### 本地运行方式（对照基准）

- Postgres(pgvector) + Redis 用根目录 `docker-compose.yml`，映射到 `localhost:5433` / `6379`
- Python 用 conda 环境 `travel_assistant`
- embedding 用 bge 本地模型（torch），模型在 `C:\Users\qian gao\models\BAAI\bge-base-zh-v1___5`

### 上云五步

1. **准备云 .env**：`cp .env.example .env` 填入真实 key（AMAP / QWEN / DEEPSEEK / LAOZHANG / TAVILY）
   - `DB_URI` / `REDIS_URL` 指向 docker 网络服务名 `postgres:5432` / `redis:6379`（不是 localhost）
   - `EMBEDDING_PROVIDER=qwen`
2. **传代码上服务器**：Xshell SFTP 或 `scp` 把项目传上去，或 `git clone` 后只填 .env
3. **启动编排**：服务器已装 Docker CE，项目根目录下 `docker compose -f docker-compose.prod.yml up -d --build`（首次 build 要几分钟）
4. **安全组放行端口**：阿里云控制台安全组放行 **8001 / 7860**（22 已放行，用于 SSH）
5. **访问**：`http://47.93.40.210:8001`（API）/ `http://47.93.40.210:7860`（Gradio UI）

### 核心差异表

| 项 | 本地 | 云端 |
|---|---|---|
| embedding | bge 本地模型（**torch**） | 通义 API `text-embedding-v4`，768 维，**不推理** |
| 运行载体 | conda 直跑 | Docker（`python:3.12-slim`，不带 torch） |
| 依赖清单 | `requirements.txt` | `requirements-docker.txt`（去 pygraphviz/grandalf/sentence-transformers） |
| DB/Redis 地址 | `localhost:5433` / `6379` | 服务名 `postgres:5432` / `redis:6379` |
| 对外端口 | 5433 / 6379 | 只 8001 / 7860，pg/redis 不映射公网 |
| 前端 host | 默认 `127.0.0.1` | `GRADIO_HOST=0.0.0.0` |
| Celery worker | `--pool=solo`（Windows 需要） | `celery -A utils.tasks worker` |

### 部署文件说明（项目根）

| 文件 | 作用 |
|---|---|
| `Dockerfile` | `python:3.12-slim`，**不带 torch**，装 `requirements-docker.txt` |
| `requirements-docker.txt` | requirements.txt 去掉 pygraphviz/grandalf/sentence-transformers，保留 langchain-community / langchain-huggingface（llm.py 顶部 import 需要） |
| `docker-compose.prod.yml` | postgres(pgvector) / redis / web:8001 / worker / webui:7860 五服务；pg、redis 不映射公网端口 |
| `.env.example` | 云 .env 模板，DB_URI/REDIS_URL 用 docker 网络服务名 |
| `.dockerignore` | 排除 .env/.git/pgdata/logfile/graph/test 等 |

## 2. 启动 / 关闭

### 启动（服务器上，进入部署目录）

```bash
docker compose -f docker-compose.prod.yml up -d --build   # 启动，首次 build 较久
docker compose -f docker-compose.prod.yml ps              # 查看 5 个服务状态
docker compose -f docker-compose.prod.yml logs -f web     # 实时看 web 日志
```

### 关闭（按递进程度）

```bash
docker compose -f docker-compose.prod.yml stop    # 停止容器，数据卷保留，可随时 up 恢复
docker compose -f docker-compose.prod.yml down    # 删容器，仍保留 pgdata 数据卷
docker compose -f docker-compose.prod.yml down -v # 连数据卷一起删 → 数据库清空！慎用
docker compose -f docker-compose.prod.yml restart web   # 只重启单个服务
```

日常"关掉再开"用 `down`（容器删了、数据还在），重新访问就再 `up -d` 一次。

## 3. 公网 IP vs 私网 IP

- **47.93.40.210（公网）**：阿里云 NAT 映射、公网可路由，是服务器对外的大门。从外网（Xshell）登录走 `公网 IP + 22 端口`；对外服务的 8001/7860、安全组放行规则都作用在它上面。
- **172.21.31.91（私网）**：只在阿里云 VPC 内网可见，公网不可达。用途：同一账号下其他云资源内网互访，或阿里云控制台内部管理。`172.21.x.x` 是阿里云 VPC 默认私网段（概念上类似家用路由的 `192.168.x.x`）。
- **关键区别**：公网侧只能用公网 IP，私网 IP 外部连不上（除非在同一 VPC 内有机器）。Xshell「公网 IP + 22 密码」能登录说明安全组 22 已放行，后续只需再放行 8001/7860。
