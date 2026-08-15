"""Phase 1 验证：意图识别分类器。

用法：
    "C:/project/envs/travel_assistant/python.exe" test_intent_router.py

用例覆盖用户验收要求的关键场景：
  - 「东京有哪些值得去的地方？」必须识别为 TRAVEL（TRAVEL_QA），不能因无"旅游"字样判成 NON_TRAVEL
  - 「国庆5天不知道去哪」 -> DESTINATION_RECOMMENDATION
  - 「写一个Java程序」  -> NON_TRAVEL（礼貌拒绝）
  - 「第三天太累了删掉两个景点」 -> ITINERARY_MODIFICATION
  - 常规规划输入        -> TRAVEL_PLANNING
"""
import asyncio
import sys

from utils.intent_router import classify_intent

CASES = [
    ("9月2号到9月5号去天津，轻松点", "TRAVEL_PLANNING"),
    ("国庆5天不知道去哪", "DESTINATION_RECOMMENDATION"),
    ("有什么适合亲子游的地方推荐一下", "DESTINATION_RECOMMENDATION"),
    ("东京有哪些值得去的地方？", "TRAVEL_QA"),
    ("去日本需要签证吗", "TRAVEL_QA"),
    ("写一个Java程序", "NON_TRAVEL"),
    ("帮我修复一下这个bug", "NON_TRAVEL"),
    ("第三天太累了，删掉两个景点", "ITINERARY_MODIFICATION"),
    ("把第二天的午餐换成火锅", "ITINERARY_MODIFICATION"),
    ("帮我规划一下杭州4天行程", "TRAVEL_PLANNING"),
    ("去杭州，约5天", "TRAVEL_PLANNING"),  # 「选择这个目的地」按钮产生的查询
]


async def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    passed = 0
    for query, expected in CASES:
        info = await classify_intent(query)
        ok = info["intent"] == expected
        passed += ok
        mark = "OK " if ok else "XX "
        print(f"{mark}[期望 {expected:<28}] 实际 {info['intent']:<28} | {info['reason']} | query={query}")
    print(f"\n{passed}/{len(CASES)} 通过")
    if passed != len(CASES):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
