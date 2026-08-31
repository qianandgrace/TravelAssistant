"""Phase 9 单元：修改行程上下文合并 + 错误处理硬化（非关键节点降级）。

  1. _merge_modification_context：未抽到新目的地复用上一轮 / 抽到则保留 / 无上一轮按新规划
  2. get_weather 服务失败 -> 降级 {"weather": ""}（不抛异常）
  3. search_pois 服务失败 -> 降级 {"pois": []}（不抛异常）
  4. do_research 摘要 LLM 失败 -> 降级 {"research": {}}（不抛异常）
"""
import asyncio


def unit_merge():
    from utils.tasks import _merge_modification_context

    q = "第三天太累删两个景点"
    prior = {"destination": "东京", "days": 5, "preference": "轻松", "start_date": "2026-09-02"}

    # 1) 未抽到新目的地（destination 兜底成 query）+ 有上一轮 -> 复用上一轮
    parsed = {"destination": q, "days": 3, "preference": "", "start_date": "", "end_date": ""}
    out = _merge_modification_context(dict(parsed), dict(prior), q)
    assert out["destination"] == "东京", "应复用上一轮目的地"
    assert out["days"] == 5, "应复用上一轮天数"
    assert out["preference"] == "轻松" and out["start_date"] == "2026-09-02"
    print("[PASS] 单元：修改行程未指明目的地 -> 复用上一轮上下文")

    # 2) 抽到明确新目的地 -> 保留抽取结果
    parsed2 = {"destination": "青岛", "days": 3, "preference": "", "start_date": "", "end_date": ""}
    out2 = _merge_modification_context(dict(parsed2), dict(prior), "把行程改成去青岛3天")
    assert out2["destination"] == "青岛" and out2["days"] == 3, "应保留明确新目的地"
    print("[PASS] 单元：修改行程明确换目的地 -> 保留抽取结果")

    # 3) 无上一轮上下文 -> 保留抽取结果（按新规划处理，不崩）
    out3 = _merge_modification_context(dict(parsed), {}, q)
    assert out3["destination"] == q
    print("[PASS] 单元：修改行程无上一轮 -> 按新规划处理")


def _tool(name, fail=False):
    class _T:
        def __init__(self):
            self.name = name
            self.fail = fail

        async def ainvoke(self, *args, **kw):
            if self.fail:
                raise RuntimeError(f"{self.name} 服务不可用")
            return [{"type": "text", "text": "{}"}]

    return _T()


def _nodes():
    from utils.workflow import create_nodes
    return create_nodes([
        _tool("maps_geo"),
        _tool("maps_weather", fail=True),
        _tool("maps_around_search", fail=True),
        _tool("maps_direction_walking"),
        _tool("maps_text_search"),
    ])


async def _weather_poi_degrade():
    (geocode, get_weather, collect_context, plan_itinerary,
     do_research, enrich_routes, enrich_images, search_pois_worker) = _nodes()
    w = await get_weather({"city": "东京"})
    assert w == {"weather": ""}, f"天气失败应降级为空，实际 {w}"
    cmd = await collect_context({"location": "139.7,35.7", "destination": "东京"})
    sends = list(getattr(cmd, "goto", []) or [])
    assert len(sends) == 4, "collect_context 应派发 4 个 Send 子任务（3 类 POI + 天气）"
    p = await search_pois_worker({"category": "景点", "location": "139.7,35.7"})
    assert p == {"pois": []}, f"POI 失败应降级为空，实际 {p}"
    print("[PASS] 单元：天气/POI 服务失败 -> 降级（不抛异常，不阻断规划）")


async def _research_degrade():
    from unittest.mock import patch
    from utils import workflow
    _, _, _, plan, do_research, *_ = _nodes()
    with patch.object(workflow, "research_summary", side_effect=RuntimeError("研究服务挂了")):
        r = await do_research({"destination": "东京", "preference": "", "days": 3})
    assert r == {"research": {}}, f"研究失败应降级为空结构，实际 {r}"
    print("[PASS] 单元：研究摘要 LLM 失败 -> 降级为空结构")


def main():
    unit_merge()
    asyncio.run(_weather_poi_degrade())
    asyncio.run(_research_degrade())
    print("\nALL PHASE9 UNIT CASES PASSED")


if __name__ == "__main__":
    main()
