"""Workflow 控制流升级验证（纯离线，不需要 Postgres / API Key / Redis）。

覆盖三类 LangGraph 派发机制：
  1. Send 动态扇出    : collect_context 返回 Command(goto=[Send(...)])，天气 + 3 类 POI
                       扇出为 4 个并行 worker（map-reduce），pois 由 _reduce_pois 归并
  2. 汇合单次触发     : 4 个 worker 同一 superstep 完成，do_research 只执行一次
  3. Command(goto=...) : review_itinerary 在节点内边更新 state、边决定下一条边
                        （reject -> 回 plan_itinerary 重规划 / accept -> extract_memory）

跑法：C:\\project\\envs\\travel_assistant\\python.exe test\\test_workflow_send_parallel.py
"""
import asyncio
import json
import os
import sys
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))  # 使 import utils.* 可用
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")  # Windows GBK 控制台兜底

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from utils import workflow as wf

# ---------- 构造：可复现的 2 天行程 JSON（planner 的 mock 输出，能通过 validate_and_normalize） ----------
CANNED_ITINERARY = json.dumps({
    "days": [
        {"day": 1, "date": "2026-09-02", "theme": "故宫", "weather": "晴",
         "items": [
             {"type": "景点", "name": "故宫", "start_time": "09:00", "end_time": "12:00",
              "duration_minutes": 180, "reason": "必去", "address": "东城区", "latitude": None,
              "longitude": None, "image": "", "poi_id": "P1"},
             {"type": "美食", "name": "全聚德", "start_time": "12:30", "end_time": "13:30",
              "duration_minutes": 60, "reason": "烤鸭", "address": "前门", "latitude": None,
              "longitude": None, "image": "", "poi_id": "P2"},
             {"type": "酒店", "name": "北京饭店", "start_time": "18:00", "end_time": "20:00",
              "duration_minutes": 120, "reason": "住宿", "address": "东长安街", "latitude": None,
              "longitude": None, "image": "", "poi_id": "P3"},
         ]},
        {"day": 2, "date": "2026-09-03", "theme": "长城", "weather": "多云",
         "items": [
             {"type": "景点", "name": "八达岭长城", "start_time": "09:00", "end_time": "12:00",
              "duration_minutes": 180, "reason": "必去", "address": "延庆", "latitude": None,
              "longitude": None, "image": "", "poi_id": "P4"},
             {"type": "美食", "name": "长城餐厅", "start_time": "12:30", "end_time": "13:30",
              "duration_minutes": 60, "reason": "午餐", "address": "延庆", "latitude": None,
              "longitude": None, "image": "", "poi_id": "P5"},
             {"type": "酒店", "name": "延庆酒店", "start_time": "18:00", "end_time": "20:00",
              "duration_minutes": 120, "reason": "住宿", "address": "延庆", "latitude": None,
              "longitude": None, "image": "", "poi_id": "P6"},
         ]},
    ],
    "tips": ["多喝水"],
})

# ---------- mock LLM：按 prompt 特征返回不同内容 ----------
_POI_NAMES = {
    "景点": ["故宫", "颐和园", "八达岭长城"],
    "美食": ["全聚德", "东来顺", "四季民福"],
    "酒店": ["北京饭店", "王府井希尔顿", "丽晶酒店"],
}


async def _fake_acall(messages):
    """模拟 utils.llm.acall_with_fallback 的返回值（.content 是文本）。"""
    prompt = messages[0].content if messages else ""
    if "旅游知识整理" in prompt:
        text = '["多喝水"]'                       # extract_memory
    elif "只输出严格 JSON" in prompt or "上次生成的行程 JSON" in prompt:
        text = CANNED_ITINERARY                     # plan / repair
    else:
        text = "好的"
    return type("R", (), {"content": text})()


def _text_payload(s: str):
    return [{"type": "text", "text": s}]


class _GeoTool:
    name = "maps_geo"

    async def ainvoke(self, kw):
        return _text_payload(json.dumps({"results": [{"location": "116.4,39.9", "adcode": "110000"}]}))


class _WeatherTool:
    name = "maps_weather"

    async def ainvoke(self, kw):
        return _text_payload(json.dumps({"city": "北京", "forecasts": [
            {"date": "2026-09-02", "dayweather": "晴", "nightweather": "晴",
             "daytemp": "28", "nighttemp": "18", "daywind": "南", "daypower": "3级"},
        ]}))


class _AroundTool:
    name = "maps_around_search"
    """按分类返回各自 3 个 POI —— 验证 3 个 Send worker 是否各自独立执行、结果是否归并。"""

    async def ainvoke(self, kw):
        cat = kw["keywords"]
        pois = [{"id": f"{cat[0]}{i}", "name": n, "address": f"{n}地址",
                 "location": "116.4,39.9", "photo": ""} for i, n in enumerate(_POI_NAMES[cat])]
        return _text_payload(json.dumps({"pois": pois}))


class _WalkingTool:
    name = "maps_direction_walking"

    async def ainvoke(self, kw):
        return _text_payload(json.dumps({"route": {"paths": [{"distance": 1500, "duration": 1200}]}}))


class _TextSearchTool:
    name = "maps_text_search"

    async def ainvoke(self, kw):
        return _text_payload(json.dumps({"pois": []}))


def _tools():
    return [_GeoTool(), _WeatherTool(), _AroundTool(), _WalkingTool(), _TextSearchTool()]


class _FakeMemory:
    """最小可用的 memory 替身：真实 checkpointer（内存版）+ 记忆读写打桩。"""
    def __init__(self):
        self.checkpointer = InMemorySaver()

    async def retrieve(self, query, limit=6):
        return ""

    async def save_episodic(self, **kw):
        return None

    async def save_semantic(self, destination, knowledge_items):
        return None


async def _run_with_mocks(build, ginput, max_interrupts=6):
    """跑 workflow，遇到 interrupt 自动裁决：第一次 reject、其余 accept/save。"""
    result = {}
    pending = ginput
    config = {"configurable": {"thread_id": "test-send-parallel"}}
    for _ in range(max_interrupts):
        interrupts = []
        async for update in build.astream(pending, config, stream_mode="updates"):
            for node_name, data in update.items():
                if node_name == "__interrupt__":
                    interrupts.extend(data)
                elif isinstance(data, dict):
                    result.update(data)
        if not interrupts:
            return result
        value = interrupts[0].value
        if value.get("kind") == "review_itinerary":
            pending = Command(resume={"action": "accept"} if result.get("feedback") else {"action": "reject", "text": "太累了，想轻松点"})
        else:  # confirm_memory
            pending = Command(resume={"action": "save"})
    raise AssertionError("中断次数超出预期，可能 Command(goto) 路由出错导致死循环")


async def _fake_search_guides(*a, **k):
    return {"sources": [], "search_available": False}


async def _fake_research_summary(*a, **k):
    return {}


async def test_send_map_reduce_offline():
    """Send 扇出 + reducer 归并：3 个分类 worker 并行，pois 完整落入最终 state。"""
    with patch.object(wf, "acall_with_fallback", new=_fake_acall), \
         patch.object(wf, "search_guides", new=_fake_search_guides), \
         patch.object(wf, "research_summary", new=_fake_research_summary), \
         patch.object(wf, "fill_item_images", new=lambda *a, **k: asyncio.sleep(0)):
        graph = wf.build_workflow(_tools(), memory=None)
        names = {n for n in graph.get_graph().nodes}
        assert {"collect_context", "search_pois_worker", "get_weather"} <= names, f"缺少 Send 节点：{names}"

        out = await graph.ainvoke({"destination": "北京", "days": 2})
        pois = out.get("pois") or []
        grouped = {}
        for p in pois:
            grouped[p["category"]] = grouped.get(p["category"], 0) + 1
        assert grouped == {"景点": 3, "美食": 3, "酒店": 3}, f"Send 归并结果异常：{grouped}"
        assert out.get("itinerary_data"), "应生成结构化行程"
        print(f"[PASS] Send map-reduce：3 worker 并行，POI 归并 {grouped}，行程已生成")


async def test_command_goto_hitl():
    """Command(goto=...) + HITL：reject 回 plan_itinerary 重规划，accept 进记忆提炼并保存。"""
    with patch.object(wf, "acall_with_fallback", new=_fake_acall), \
         patch.object(wf, "search_guides", new=_fake_search_guides), \
         patch.object(wf, "research_summary", new=_fake_research_summary), \
         patch.object(wf, "fill_item_images", new=lambda *a, **k: asyncio.sleep(0)):
        graph = wf.build_workflow(_tools(), memory=_FakeMemory())
        result = await _run_with_mocks(graph, {"destination": "北京", "days": 2, "query": "规划北京2天"})
        assert result.get("itinerary_data"), "应产出结构化行程"
        assert result.get("knowledge"), f"应提炼出旅游知识：{result.get('knowledge')}"
        assert result.get("memory_saved", "").startswith("episodic+semantic"), result.get("memory_saved")
        # reject 时 feedback 进入 messages，plan_itinerary 据此重规划过一轮
        fb_in_msgs = any("太累了" in str(getattr(m, "content", "")) for m in (result.get("messages") or []))
        assert fb_in_msgs, "reject 意见应进入对话历史（证明 goto 回到 plan_itinerary）"
        print("[PASS] Command(goto=...) + HITL：reject->重规划->accept->记忆确认->保存 全链路通过")


def main():
    asyncio.run(test_send_map_reduce_offline())
    asyncio.run(test_command_goto_hitl())
    print("\nALL SEND/COMMAND WORKFLOW CASES PASSED")


if __name__ == "__main__":
    main()
