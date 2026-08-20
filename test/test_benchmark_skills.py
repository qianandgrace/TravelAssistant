"""Skills 渐进式披露对比基准：workflow / react × 有/无 skills（4 方案）。

用法：
    PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python test/test_benchmark_skills.py

环境变量：
    SMOKE=N          只跑前 N 问（SMOKE=0 只构建 harness、不发真实 LLM 调用）。
                     默认全量 10 问。
    BENCHMARK_MODEL  对比模型，默认 deepseek（如 BENCHMARK_MODEL=qwen）。

与 test/test_benchmark_workflow_vs_react.py 完全对齐的计量方法：
同一 10 问、同一契约打分（check_contract）、同一 token 累计（_Usage + monkey-patch
acall_with_fallback 强制单模型）。唯一新增变量是 skills 渐进式披露：

    workflow        : build_workflow(map_tools, memory=None)
    workflow_skills : build_workflow(map_tools, memory=None, skills=load_all_skills())
    react           : build_react_agent(map_tools, with_skills=False)
    react_skills    : build_react_agent(map_tools, with_skills=True)

输出：benchmark_out/skills/result_skills.json + benchmark_out/skills/q{id}_{system}.txt
（system ∈ workflow / workflow_skills / react / react_skills，标签本身已区分是否带 skills）
"""
import asyncio
import json
import logging
import os
import sys
import time

from langchain_core.messages import AIMessage, HumanMessage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)  # 使 import test_benchmark_workflow_vs_react 可用

import utils.llm  # noqa: E402
import test_benchmark_workflow_vs_react as base  # noqa: E402  (复用 QUESTIONS / check_contract)
from utils.entity_extractor import extract_travel_entities  # noqa: E402
from utils.intent_router import NON_TRAVEL_REPLY, classify_intent  # noqa: E402
from utils.llm import get_single_llm  # noqa: E402
from utils.prompts import TRAVEL_QA_PROMPT  # noqa: E402
from utils.react_agent import build_react_agent  # noqa: E402
from utils.recommendation import recommend_destinations  # noqa: E402
from utils.skills import load_all_skills  # noqa: E402
from utils.tools import get_map_tools  # noqa: E402
from utils.workflow import build_workflow  # noqa: E402

MODEL = os.getenv("BENCHMARK_MODEL", "deepseek")
SMOKE = int(os.getenv("SMOKE", "0")) if os.getenv("SMOKE") else 10

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

OUT_DIR = os.path.join("benchmark_out", "skills")  # 独立子目录，避免与 baseline 输出混淆
QUESTIONS = base.QUESTIONS[:SMOKE] if SMOKE else []

# ---------------------------------------------------------------- token 累计（同 baseline）
class _Usage:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add(self, msg):
        um = getattr(msg, "usage_metadata", None) or {}
        self.input_tokens += um.get("input_tokens") or 0
        self.output_tokens += um.get("output_tokens") or 0
        self.calls += 1


_USAGE = _Usage()
_ORIG_ACALL = utils.llm.acall_with_fallback


async def _patched_acall(messages, llm_chain=None):
    resp = await _ORIG_ACALL(messages, llm_chain=[MODEL])
    _USAGE.add(resp)
    return resp


def _install_patch():
    utils.llm.acall_with_fallback = _patched_acall
    for mod in (utils.entity_extractor, utils.intent_router,
                utils.recommendation, utils.research, utils.workflow):
        mod.acall_with_fallback = _patched_acall


# ---------------------------------------------------------------- workflow 路径（两个变体共用）
async def _travel_reply(query: str) -> str:
    prompt = TRAVEL_QA_PROMPT.format(query=query)
    res = await utils.llm.acall_with_fallback([HumanMessage(content=prompt)])
    return str(res.content).strip()


async def run_workflow(query: str, ctx) -> dict:
    """ctx = map_tools；复刻 baseline 的 run_workflow（无 skills）。"""
    map_tools = ctx
    graph = _workflow_graphs[0]  # plain
    return await _run_workflow_graph(query, map_tools, graph)


async def run_workflow_skills(query: str, ctx) -> dict:
    map_tools = ctx
    graph = _workflow_graphs[1]  # skills
    return await _run_workflow_graph(query, map_tools, graph)


_workflow_graphs: list = []  # 进程级缓存：主流程 build 一次，两个 runner 复用


async def _run_workflow_graph(query: str, map_tools, graph) -> dict:
    _USAGE.reset()
    t0 = time.perf_counter()
    intent_info = await classify_intent(query)
    intent = intent_info.get("intent", "TRAVEL_PLANNING")
    route = intent_info.get("route", "graph")
    answer = ""
    if route == "refuse":
        answer = NON_TRAVEL_REPLY
    elif route == "recommend":
        rec = await recommend_destinations(query)
        cards = (rec or {}).get("destinations") or []
        answer = "推荐目的地：\n" + "\n".join(
            f"- {c.get('destination', '')}（{c.get('country', '')}）：{c.get('reason', '')}"
            for c in cards
        )
    elif route == "reply":
        answer = await _travel_reply(query)
    else:
        parsed = await extract_travel_entities(query)
        ginput = {
            "destination": parsed.get("destination", ""),
            "days": parsed.get("days", 3),
            "preference": parsed.get("preference", ""),
            "start_date": parsed.get("start_date", ""),
            "query": query,
        }
        state = await graph.ainvoke(ginput)
        answer = str(state.get("itinerary") or state.get("summary") or "")
    elapsed = time.perf_counter() - t0
    return {
        "answer": answer,
        "time": elapsed,
        "input_tokens": _USAGE.input_tokens,
        "output_tokens": _USAGE.output_tokens,
        "calls": _USAGE.calls,
        "intent": intent,
        "route": route,
    }


# ---------------------------------------------------------------- ReAct 路径
async def run_react(query: str, agent) -> dict:
    t0 = time.perf_counter()
    state = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 30},
    )
    elapsed = time.perf_counter() - t0
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    answer = last.content if isinstance(last, AIMessage) else str(last or "")
    if isinstance(answer, list):
        answer = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in answer)
    tin = tout = 0
    for m in msgs:
        um = getattr(m, "usage_metadata", None) or {}
        tin += um.get("input_tokens") or 0
        tout += um.get("output_tokens") or 0
    return {
        "answer": answer,
        "time": elapsed,
        "input_tokens": tin,
        "output_tokens": tout,
        "calls": len([m for m in msgs if isinstance(m, AIMessage)]),
        "intent": "react",
        "route": "react_skills" if "skills" in getattr(agent, "name", "") else "react",
    }


# ---------------------------------------------------------------- 主流程
async def main() -> int:
    _install_patch()
    os.makedirs(OUT_DIR, exist_ok=True)
    map_tools = await get_map_tools()

    # 构建 4 个方案（一次构建，全量复用；SMOKE=0 也构建以验证 harness）
    skills = load_all_skills()
    _workflow_graphs.append(build_workflow(map_tools, memory=None))
    _workflow_graphs.append(build_workflow(map_tools, memory=None, skills=skills))
    react_plain = build_react_agent(map_tools=map_tools, model=get_single_llm(MODEL), with_skills=False)
    react_skill = build_react_agent(map_tools=map_tools, model=get_single_llm(MODEL), with_skills=True)

    if not QUESTIONS:
        print(f"[SMOKE=0] 4 方案 harness 构建成功（workflow / workflow_skills / react / react_skills），未发起真实调用。")
        return 0

    print(f"对比模型：{MODEL}  共 {len(QUESTIONS)} 问")
    systems = (
        ("workflow", run_workflow, map_tools),
        ("workflow_skills", run_workflow_skills, map_tools),
        ("react", run_react, react_plain),
        ("react_skills", run_react, react_skill),
    )

    results = []
    for q in QUESTIONS:
        print(f"\n=== Q{q['id']} {q['query']} ===")
        for system, runner, ctx in systems:
            try:
                r = await runner(q["query"], ctx)
            except Exception as e:  # noqa: BLE001 - 单题失败不中断整个基准
                logging.getLogger("benchmark").exception("[%s] Q%d 失败", system, q["id"])
                r = {"answer": f"[ERROR] {type(e).__name__}: {e}", "time": 0,
                     "input_tokens": 0, "output_tokens": 0, "calls": 0,
                     "intent": "error", "route": str(e)}
            ok, why = base.check_contract(q, r["answer"])
            r.update({"qid": q["id"], "system": system, "query": q["query"],
                      "contract_ok": ok, "contract_why": why})
            results.append(r)
            print(f"  [{system}] {r['time']:6.1f}s  in={r['input_tokens']:>6} "
                  f"out={r['output_tokens']:>5} 契约={'通过' if ok else '未过'}  {why}")
            with open(os.path.join(OUT_DIR, f"q{q['id']}_{system}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(r["answer"])

    _print_summary(results)
    with open(os.path.join(OUT_DIR, "result_skills.json"), "w", encoding="utf-8") as f:
        json.dump({"questions": QUESTIONS, "model": MODEL, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已写入 {OUT_DIR}/result_skills.json 与逐题输出文件。")
    return 0


def _agg(results: list, system: str) -> dict:
    rows = [r for r in results if r["system"] == system]
    return {
        "time": sum(r["time"] for r in rows),
        "input": sum(r["input_tokens"] for r in rows),
        "output": sum(r["output_tokens"] for r in rows),
        "pass": sum(1 for r in rows if r["contract_ok"]),
        "n": len(rows),
    }


def _print_summary(results: list) -> None:
    names = ("workflow", "workflow_skills", "react", "react_skills")
    data = {n: _agg(results, n) for n in names}
    print("\n" + "=" * 90)
    print(f"{'指标':<14}{'Workflow':>14}{'WF+Skills':>16}{'ReAct':>14}{'ReAct+Skills':>16}")
    print("-" * 90)
    print(f"{'总耗时(s)':<14}"
          f"{data['workflow']['time']:>14,.1f}{data['workflow_skills']['time']:>16,.1f}"
          f"{data['react']['time']:>14,.1f}{data['react_skills']['time']:>16,.1f}")
    print(f"{'总 input token':<14}"
          f"{data['workflow']['input']:>14,}{data['workflow_skills']['input']:>16,}"
          f"{data['react']['input']:>14,}{data['react_skills']['input']:>16,}")
    print(f"{'总 output token':<14}"
          f"{data['workflow']['output']:>14,}{data['workflow_skills']['output']:>16,}"
          f"{data['react']['output']:>14,}{data['react_skills']['output']:>16,}")
    print(f"{'契约通过率':<14}"
          f"{data['workflow']['pass']}/{data['workflow']['n']:>8}"
          f"{data['workflow_skills']['pass']}/{data['workflow_skills']['n']:>10}"
          f"{data['react']['pass']}/{data['react']['n']:>8}"
          f"{data['react_skills']['pass']}/{data['react_skills']['n']:>10}")
    print("=" * 90)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main()))
