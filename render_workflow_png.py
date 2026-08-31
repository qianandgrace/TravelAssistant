"""把升级后的 workflow（memory 分支）渲染成 PNG：静态边实线、Send/Command 动态边红色虚线。

用法：C:\\project\\envs\\travel_assistant\\python.exe render_workflow_png.py
产出：graph/workflow_upgraded.png（pygraphviz + graphviz，需要系统装 graphviz）
"""
import os

import pygraphviz as pgv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph", "workflow_upgraded.png")


def main() -> None:
    a = pgv.AGraph(directed=True)
    a.graph_attr.update(rankdir="TB", ranksep="0.55", nodesep="0.45", bgcolor="white")
    a.node_attr.update(fontname="Microsoft YaHei", fontsize=11, shape="box",
                       style="rounded,filled", color="#4a6fa5")
    a.edge_attr.update(fontname="Microsoft YaHei", fontsize=9)

    def add(n, fill="#eef4fb", shape="box"):
        a.add_node(n, fillcolor=fill, shape=shape)

    def static(s, t):
        a.add_edge(s, t, style="solid", color="#333333")

    def dynamic(s, t, label):
        # constraint=false：动态边不参与层级计算，避免虚线把布局拉乱
        a.add_edge(s, t, style="dashed", color="#c0392b", label=label, constraint="false")

    # 入口/出口
    add("START", fill="#d5e8d4", shape="ellipse")
    add("END", fill="#d5e8d4", shape="ellipse")
    # 地图/上下文采集
    add("geocode", fill="#eef4fb")
    add("collect_context", fill="#fff2cc")   # Send 扇出
    add("search_pois_worker", fill="#eef4fb")
    add("get_weather", fill="#eef4fb")
    add("do_research", fill="#eef4fb")
    # 记忆
    add("retrieve_memory", fill="#f3e8fd")
    add("summarize_conversation", fill="#f3e8fd")
    add("extract_memory", fill="#f3e8fd")
    add("save_memory", fill="#f3e8fd")
    # 生成/渲染
    add("plan_itinerary", fill="#eef4fb")
    add("enrich_routes", fill="#eef4fb")
    add("enrich_images", fill="#eef4fb")
    # HITL
    add("review_itinerary", fill="#fdeaea")

    # 静态边（实线）
    static("START", "geocode")
    static("geocode", "collect_context")
    static("search_pois_worker", "retrieve_memory")
    static("get_weather", "retrieve_memory")
    static("retrieve_memory", "summarize_conversation")
    static("summarize_conversation", "do_research")
    static("do_research", "plan_itinerary")
    static("plan_itinerary", "enrich_routes")
    static("enrich_routes", "enrich_images")
    static("enrich_images", "review_itinerary")
    static("extract_memory", "save_memory")
    static("save_memory", "END")

    # 动态边（红色虚线）
    dynamic("collect_context", "search_pois_worker", "Send ×3")
    dynamic("collect_context", "get_weather", "Send")
    dynamic("review_itinerary", "plan_itinerary", "reject · Command(goto)")
    dynamic("review_itinerary", "extract_memory", "accept · Command(goto)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    a.layout(prog="dot")
    a.draw(OUT, format="png")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
