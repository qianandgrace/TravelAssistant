"""Workflow vs ReAct 架构对比基准（10 问 × 2 系统：token / 耗时 / 准确度契约）。

用法：
    PYTHONPATH=. PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python test/test_benchmark_workflow_vs_react.py

两个执行路径都**进程内**跑（跳过 HTTP / celery / redis 开销，只测「推理」本身）：

- workflow : utils/tasks.py::_run_invoke 的推理部分（意图分诊 -> 路由 -> 实体抽取
             -> build_workflow(map_tools, memory=None) 固定流水线）。
- react    : utils/react_agent.py::build_react_agent（langchain.agents.create_agent
             构建的 ReAct 循环，LLM 自主决定调用工具）。

模型：默认两端都用 deepseek（同模型公平对比），可用环境变量 BENCHMARK_MODEL
覆盖（如 BENCHMARK_MODEL=qwen）。为了让对比只看架构差异，workflow 路径会强制
acall_with_fallback 的 llm_chain=[model]（禁用回退链），ReAct 用 get_single_llm(model)。

token 测量：
- workflow：monkey-patch utils.llm.acall_with_fallback（并重绑到各模块已
  `from ... import` 的引用），累加每次返回 AIMessage.usage_metadata。
- react   ：遍历 agent.ainvoke 返回的全部 messages 累加 usage_metadata。
  注：ReAct 每轮模型调用会把此前全部上下文重发，因此 input_tokens 累加的是
  「真实付费量」（每轮都按完整上下文计费）。

输出：benchmark_out/<qid>_<system>.txt（原始回答，供人工复核）
     + benchmark_out/result.json（全部结构化数据）+ 控制台对比表。
"""
import asyncio
import json
import logging
import os
import re
import sys
import time

from langchain_core.messages import AIMessage, HumanMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import utils.llm  # noqa: E402
from utils.entity_extractor import extract_travel_entities  # noqa: E402
from utils.intent_router import NON_TRAVEL_REPLY, classify_intent  # noqa: E402
from utils.llm import get_single_llm  # noqa: E402
from utils.prompts import TRAVEL_QA_PROMPT  # noqa: E402
from utils.react_agent import build_react_agent  # noqa: E402
from utils.recommendation import recommend_destinations  # noqa: E402
from utils.tools import get_map_tools  # noqa: E402
from utils.workflow import build_workflow  # noqa: E402

# 对比模型：默认 deepseek，可用环境变量覆盖（如 BENCHMARK_MODEL=qwen）
MODEL = os.getenv("BENCHMARK_MODEL", "deepseek")

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")

OUT_DIR = "benchmark_out"

# ---------------------------------------------------------------- 10 个问题
QUESTIONS = [
    {"id": 1, "query": "9月2号到9月6号去东京，轻松点", "intent": "plan", "days": 5, "dest": "东京"},
    {"id": 2, "query": "帮我规划一下杭州4天行程", "intent": "plan", "days": 4, "dest": "杭州"},
    {"id": 3, "query": "去杭州，约5天", "intent": "plan", "days": 5, "dest": "杭州"},
    {"id": 4, "query": "8月20号到8月23号去天津玩，希望行程有一天能在海边玩耍", "intent": "plan", "days": 4, "dest": "天津"},
    {"id": 5, "query": "国庆5天不知道去哪", "intent": "recommend"},
    {"id": 6, "query": "有什么适合亲子游的地方推荐一下", "intent": "recommend"},
    {"id": 7, "query": "东京有哪些值得去的地方？", "intent": "qa", "keywords": ["东京", "浅草", "银座", "富士"]},
    {"id": 8, "query": "去日本需要签证吗", "intent": "qa", "keywords": ["签证", "免签", "护照"]},
    {"id": 9, "query": "写一个Java程序", "intent": "nontravel"},
    {"id": 10, "query": "帮我修复一下这个bug", "intent": "nontravel"},
]

# ---------------------------------------------------------------- token 累计
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
    # 强制单模型，让两端做同模型公平对比（禁用 workflow 的回退链）
    resp = await _ORIG_ACALL(messages, llm_chain=[MODEL])
    _USAGE.add(resp)
    return resp


def _install_patch():
    """给 utils.llm.acall_with_fallback 打补丁，并重绑各模块已 import 的引用。"""
    utils.llm.acall_with_fallback = _patched_acall
    for mod in (
        utils.entity_extractor,
        utils.intent_router,
        utils.recommendation,
        utils.research,
        utils.workflow,
    ):
        mod.acall_with_fallback = _patched_acall


# ---------------------------------------------------------------- 契约打分
_CHINESE_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_DAY_RE = re.compile(r"第\s*([0-9]+|[一二两三四五六七八九十]+)\s*天")


def _day_numbers(text: str) -> set:
    """数出文本里出现的「第 N 天」天数集合（覆盖中文数字与阿拉伯数字）。"""
    nums = set()
    for m in _DAY_RE.finditer(text):
        g = m.group(1)
        if g.isdigit():
            nums.add(int(g))
        else:
            nums.add(_CHINESE_NUM.get(g, -1))
    return nums


def check_contract(q: dict, answer: str) -> tuple:
    """规则化契约。返回 (通过与否, 说明)。"""
    a = answer or ""
    intent = q["intent"]
    if intent == "plan":
        days = _day_numbers(a)
        ok_days = q["days"] in days
        ok_dest = q["dest"] in a
        return ok_days and ok_dest, (
            f"天数{q['days']}={'✓' if ok_days else '✗'}(含{len(days)}天) "
            f"目的地「{q['dest']}」={'✓' if ok_dest else '✗'}"
        )
    if intent == "recommend":
        ok = len(a.strip()) > 30 and any(w in a for w in ("推荐", "建议", "适合", "目的地", "打卡"))
        return ok, ("给出推荐" if ok else "未给出明确目的地推荐")
    if intent == "qa":
        hit = [k for k in q["keywords"] if k in a]
        return bool(hit), (f"命中关键词 {hit}" if hit else f"未命中 {q['keywords']}")
    if intent == "nontravel":
        refused = any(w in a for w in ("抱歉", "只处理", "无法", "不能", "不负责", "不是", "旅行"))
        no_itin = not _DAY_RE.search(a)
        ok = refused and no_itin
        return ok, f"拒绝={refused} 无行程={no_itin}"
    return False, "未知 intent"


# ---------------------------------------------------------------- workflow 路径
async def _travel_reply(query: str) -> str:
    """TRAVEL_QA / DESTINATION_RECOMMENDATION 直接回答（对齐 utils/tasks.py::_travel_reply）。"""
    prompt = TRAVEL_QA_PROMPT.format(query=query)
    res = await utils.llm.acall_with_fallback([HumanMessage(content=prompt)])
    return str(res.content).strip()


async def run_workflow(query: str, map_tools) -> dict:
    """复刻 _run_invoke 的推理部分（无 HTTP/celery/redis），返回结构化结果。"""
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
        graph = build_workflow(map_tools, memory=None)
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
    """create_agent 构建的 ReAct agent：单轮 ainvoke，从全部 messages 累计 token。"""
    t0 = time.perf_counter()
    state = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 30},
    )
    elapsed = time.perf_counter() - t0
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    answer = last.content if isinstance(last, AIMessage) else str(last or "")
    if isinstance(answer, list):  # 部分模型 content 是文本块列表
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
        "route": "react",
    }


# ---------------------------------------------------------------- 主流程
async def main() -> int:
    _install_patch()
    os.makedirs(OUT_DIR, exist_ok=True)
    map_tools = await get_map_tools()
    print(f"对比模型：{MODEL}")
    react_agent = build_react_agent(map_tools=map_tools, model=get_single_llm(MODEL))

    results = []
    for q in QUESTIONS:
        print(f"\n=== Q{q['id']} {q['query']} ===")
        for system, runner, ctx in (
            ("workflow", run_workflow, map_tools),
            ("react", run_react, react_agent),
        ):
            try:
                r = await runner(q["query"], ctx)
            except Exception as e:  # noqa: BLE001 - 单题失败不中断整个基准
                logging.getLogger("benchmark").exception("[%s] Q%d 失败", system, q["id"])
                r = {"answer": f"[ERROR] {type(e).__name__}: {e}", "time": 0,
                     "input_tokens": 0, "output_tokens": 0, "calls": 0,
                     "intent": "error", "route": str(e)}
            ok, why = check_contract(q, r["answer"])
            r.update({"qid": q["id"], "system": system, "query": q["query"],
                      "contract_ok": ok, "contract_why": why})
            results.append(r)
            print(f"  [{system}] {r['time']:6.1f}s  in={r['input_tokens']:>6} "
                  f"out={r['output_tokens']:>5} 契约={'通过' if ok else '未过'}  {why}")
            with open(os.path.join(OUT_DIR, f"q{q['id']}_{system}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(r["answer"])

    _print_summary(results)
    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"questions": QUESTIONS, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n详细结果已写入 {OUT_DIR}/result.json 与逐题输出文件。")
    return 0


def _print_summary(results: list) -> None:
    def agg(system):
        rows = [r for r in results if r["system"] == system]
        return {
            "time": sum(r["time"] for r in rows),
            "input": sum(r["input_tokens"] for r in rows),
            "output": sum(r["output_tokens"] for r in rows),
            "pass": sum(1 for r in rows if r["contract_ok"]),
            "n": len(rows),
        }

    wf, rc = agg("workflow"), agg("react")
    print("\n" + "=" * 72)
    print(f"{'指标':<16}{'Workflow':>18}{'ReAct':>18}{'差异':>18}")
    print("-" * 72)
    print(f"{'总耗时(s)':<16}{wf['time']:>18,.1f}{rc['time']:>18,.1f}"
          f"{rc['time'] - wf['time']:>+18,.1f}")
    print(f"{'总 input token':<16}{wf['input']:>18,}{rc['input']:>18,}"
          f"{rc['input'] - wf['input']:>+18,}")
    print(f"{'总 output token':<16}{wf['output']:>18,}{rc['output']:>18,}"
          f"{rc['output'] - wf['output']:>+18,}")
    wt, rt = wf["input"] + wf["output"], rc["input"] + rc["output"]
    pct = f"{(rt / wt - 1) * 100:+.0f}%" if wt else "n/a"
    print(f"{'总 token':<16}{wt:>18,}{rt:>18,}{pct:>18}")
    print(f"{'契约通过率':<16}{wf['pass']}/{wf['n']:<4}   {rc['pass']}/{rc['n']}")
    print("=" * 72)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main()))
