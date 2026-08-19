# TravelAssistant · 旅行行程规划助手

基于 **LangGraph** 的对话式旅行行程规划器：自然语言输入（如「9月2号到9月5号去东京，轻松点」）→ 意图识别 → 实体抽取 → 攻略研究 → 逐日行程生成 → **人工确认（HITL）** → 长期记忆沉淀。前端为 Gradio 对话界面，后端为 FastAPI + Celery 异步任务，Redis 存会话状态，Postgres 存长短期记忆。

## 架构

```
Gradio(webapp.py) ──HTTP──▶ FastAPI(server.py) ──▶ Redis(会话状态 + broker/result)
                                      │                    ▲
                                      │ delay()            │ worker 写回状态
                                      ▼                    │
                                Celery worker(utils/tasks.py) ──▶ LangGraph(utils/workflow.py)
                                                                      │
                                                                Postgres(checkpointer 短期 / store 长期)
```

| 组件 | 文件 | 职责 |
|---|---|---|
| 前端 | `webapp.py` | Gradio 对话界面、历史会话、HITL 裁决按钮、导出 |
| 后端 | `server.py` | FastAPI 路由、会话管理、导出端点、静态图代理（预留） |
| 异步任务 | `utils/tasks.py` | Celery worker：意图分诊、实体抽取、跑 LangGraph、状态回写 |
| 工作流 | `utils/workflow.py` | LangGraph 状态机（节点 + 两条 HITL 中断） |
| 会话 | `utils/session_manager.py` | Redis 会话/任务状态（TTL 动态调整） |
| 记忆 | `utils/memory.py` | Postgres store（长期）+ checkpointer（短期） |
| 渲染 | `utils/render_html.py` | 产品视图 HTML（时间线/地图/攻略下拉）+ 导出 |

## 环境与依赖

```bash
# 推荐使用 conda 环境
conda create -n travel_assistant python=3.11
conda activate travel_assistant
pip install -r requirements.txt
```

依赖服务：**Redis**（宿主机 `6379`）、**Postgres + pgvector**（`5433`，docker-compose 已就绪）。

## 配置（.env）

所有密钥只放项目根目录的 `.env`，**绝不硬编码、绝不提交**（`.env` 已在 `.gitignore`）：

```
AMAP_MAPS_API_KEY=          # 高德 MCP key（地理编码/POI/路线/天气/图片）
QWEN_API_KEY=               # LLM 主模型（回退链：qwen → deepseek → openai）
DEEPSEEK_API_KEY=
LAOZHANG_API_KEY=
TAVILY_API_KEY=             # 攻略搜索；缺失则研究优雅降级为空来源
LOG_LEVEL=DEBUG             # 日志级别，默认 DEBUG（开发期观察各流程节点）
DB_URI=postgresql://gq210:123456@localhost:5433/postgres?sslmode=disable
```

## 启动

```bash
# 1) Celery worker（Windows 必须 --pool=solo；--loglevel=DEBUG 才能看到节点级 DEBUG 日志）
python -m celery -A utils.tasks.celery_app worker --loglevel=DEBUG --pool=solo

# 2) 后端
python server.py                 # FastAPI @ :8001

# 3) 前端
python webapp.py                 # Gradio @ :7860
```

日志会同时输出到控制台和 `logfile/app.log`。每个流程节点都有 `DEBUG` 日志：
`[geocode]` → `[get_weather]` → `[search_pois]` → `[retrieve_memory]` → `[do_research]` → `[plan_itinerary]` → `[review_itinerary]` → `[extract_memory]` → `[save_memory]`；任务层有 `[intent]` / `[entity]` / `[modify]` / `[graph]`。

## 意图识别与路由

入口 `utils/tasks.py::_run_invoke` 先做**意图分诊**（`classify_intent`），再决定是否进图——这是刻意的：拒绝/问答/推荐等轻量意图不必为一句回复付全套图构建成本：

| 意图 | 路由 | 处理 |
|---|---|---|
| 非旅游（写程序/修 bug…） | `refuse` | 直接礼貌拒绝，不进图 |
| 旅游问答（签证/景点…） | `reply` | LLM 直接回答，不进图 |
| 目的地推荐（不知道去哪） | `recommend` | LLM 生成目的地卡片（含真实 POI 图），不进图 |
| 规划 / 修改已有行程 | `graph` | 抽取实体 → 复用上下文（修改时）→ 进 LangGraph |

修改行程（如「第三天太累删两个景点」）：用户通常不重复目的地，因此抽取结果里没有明确新目的地时，从同一会话上一轮 checkpointer **复用 destination/days/preference**，原始改动指令经 `state.query` 原样交给 planner。

## 确定行程（HITL）逻辑

规划类请求进图后的完整流程，两处中断由用户在 Gradio 上裁决：

```
geocode ─┬─ search_pois ── retrieve_memory ── summarize ── do_research ── plan_itinerary
         └─ get_weather  （并行）
              → enrich_routes（坐标/路线） → enrich_images（真实图片）
              → ──★ 中断1：review_itinerary 接受 / 编辑 / 拒绝 ──
                     │ 接受  ─→ extract_memory → ★ 中断2：confirm_memory 保存 / 跳过 → 完成
                     │ 拒绝  ─→（带反馈意见）回 plan_itinerary 重新规划 → 再回中断1
                     └ 编辑  ─→ 以编辑文本作为行程 → 视为已接受 → 进中断2
```

- **中断1「确定行程」**：默认接受；「拒绝」会携带修改意见重规划；「编辑」直接采用编辑后的文本。
- **中断2「保存记忆」**：保存则写 episodic（本次经验）+ semantic（通用旅游知识）到 Postgres `store` 表；跳过则仅完成本次行程。保存失败（best-effort）**不影响**本次行程。
- 观察日志即可跟随每一步：`[review_itinerary] 用户选择「接受」…`、`[save_memory] 已保存到 Postgres store 表…`。

## 数据库：长短期记忆分别在哪个表

| 表 | 内容 | 查看方式 |
|---|---|---|
| `store` | **长期记忆**（episodic/semantic/notes） | `SELECT prefix, key, value FROM store;` |
| `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` | **短期记忆**：每线程（session）的图状态与消息 | 由 LangGraph checkpointer 管理 |

长期记忆命名空间（按用户隔离）：`memory.<user_id>.episodic`、`memory.<user_id>.semantic`、`memory.<user_id>.notes`；内容在 `value`（jsonb）里，`content` 字段为可注入 prompt 的文本。CLI（`main.py`，不传 user_id）用全局命名空间 `memory.*`。

> 排查提示：选了「保存」却在库里找不到？请查 `store` 表而不是 `checkpoints` —— 长期记忆只进 `store`。

## 攻略来源下拉

研究阶段（`utils/research.py`）用 Tavily 搜索真实攻略/游记来源（**绝不编造**；key 缺失时 sources 为空）。产品视图把来源渲染成可展开下拉（`<details>`），逐条给出**可点击链接**（`target=_blank`，标题与 URL 均转义防注入），Web 端与导出的 HTML 共用同一渲染函数。

## 地图

服务端把每日有坐标的 POI 画成内联 SVG（等距圆柱投影 + cos(lat) 修正，GCJ-02 坐标，按天配色折线 + 全局编号）。SVG 支持**滚轮缩放、拖拽平移、＋/−/复位按钮**（纯前端，无外部库）。注意：SVG 只画行程 POI 点与路线，**不含真实底图**，放大缩小看到的是点/线的空间关系。

## 图片

景点/美食卡片图片来自高德 POI 真实照片；无图时用占位图 + `loading=lazy` 懒加载，不阻塞布局。所有真实图片来自 POI/search API，**不由 LLM 生成 URL**。

## HTML 导出

Web 端「导出 HTML」按钮 → 后端 `/agent/export` → 生成独立 HTML（产品视图 + 地图 + 攻略下拉 + 打印样式）。**Key 安全**：导出文件只含业务字段，服务端 API Key（高德/Tavily/各家 LLM）的取值绝不会出现在导出 HTML 中（有哨兵值单测保证）。文件名自动净化非法字符，空目的地回退 `行程.html`。

## 错误处理约定

非关键节点**优雅降级、绝不崩溃**：天气失败→空、POI 失败→空列表、研究失败→空结构、记忆检索失败→空、对话摘要失败→本轮不压缩、记忆保存失败→不影响行程、意图识别失败→默认规划。核心数据（目的地 geocode 失败、行程 JSON 两次生成失败）会明确报错而非静默。

## 测试

环境无 pytest，测试为普通 Python 脚本（`if __name__ == "__main__"`），需服务运行时用 HTTP e2e：

```bash
# 离线（无需服务）
python test_intent_router.py        # 意图识别 11/11
python test_phase9_units.py         # 修改行程合并 + 节点降级
python test_phase7_frontend.py      # 前端渲染单测（e2e 部分需服务）
python test_phase8_export.py        # 导出单测 + Key 安全（e2e 部分需服务）

# 需 worker + server 运行（先按「启动」起服务）
python test_phase9_acceptance.py    # 验收 A-D：完整东京规划/国庆推荐/Java拒绝/修改行程
python test_phase1_e2e.py ...       # 各阶段 e2e（test_phase1_e2e.py ~ test_phase8_export.py）
```

Windows 下控制台若显示中文乱码，用 `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python xxx.py` 运行。

## 命令行 CLI

`python main.py 杭州 2 --preference "轻松好吃"`（传参式，含同样的 HITL 确认；不建用户隔离的长期记忆命名空间，兼容老用法）。
