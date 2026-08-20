# LangGraph Workflow 版行程规划节点使用的 prompt：
# 把高德采集到的 POI + 天气 + 天数合成逐日行程
ITINERARY_PLANNER_PROMPT = """你是一名专业的旅游行程规划师。请根据下面提供的信息，为游客规划 {days} 天的逐日行程。

【目的地】{destination}
【用户偏好】{preference}
【天气预报】
{weather}
【攻略研究摘要】（来自攻略/游记与旅游知识的研究结论，供参考，无则忽略）
{research}
【可用的景点 / 美食 / 酒店 POI 数据】
{pois_text}
【相关记忆】（历史经验与旅游知识，供参考，无则忽略）
{memories}
【历史对话】（此前同次对话的摘要与近期往来，供延续上下文，无则忽略）
{history}
【修改意见】（若用户拒绝了上一版行程并给出意见，请据此调整，无则忽略）
{feedback}
【技能规范】（命中意图时的领域指令，供参考，无则忽略）
{skill}

要求：
1. 按天组织行程，每天给出上午 / 下午 / 晚上的具体安排，路线尽量顺路、不绕路。
2. 合理搭配景点、餐厅和住宿：景点挑选值得去的，餐厅给出本地特色，酒店每天推荐 2~3 家。
3. 结合天气预报给出穿衣 / 出行 / 行程调整建议（如下雨时优先安排室内场馆）。
4. 若提供了用户偏好，请优先满足偏好（如喜欢历史就多安排历史景点）；若相关记忆中有该目的地的历史经验或知识，尽量复用并延续。
5. 酒店与餐厅若 POI 数据中缺少评分/价格，可用常识补充，但须标注"待确认"。
6. 输出使用 Markdown：每天一个二级标题（如 ## 第 1 天），内容简洁、可直接执行，不要输出无关解释。
7. 若有【修改意见】，请先解决其中的问题（如调整景点、时间、强度等）再输出新行程。"""

# 结构化行程生成 prompt（输出严格 JSON，后端校验，坐标/路线/图片由地图服务计算，LLM 不编造）
ITINERARY_PLANNER_JSON_PROMPT = """你是专业旅游行程规划师。请根据下面信息为 {destination} 规划 {days} 天的逐日行程，只输出严格 JSON，不要输出任何 JSON 以外的文字。

【目的地】{destination}
【出行日期】{dates}
【用户偏好】{preference}
【天气预报】
{weather}
【攻略研究摘要】
{research}
【可用的景点/美食/酒店 POI 数据】（含编号，供引用 poi_id）
{pois_text}
【相关记忆】（历史经验与旅游知识，供参考，无则忽略）
{memories}
【历史对话】（此前的摘要与近期往来，供延续上下文，无则忽略）
{history}
【修改意见】（若用户拒绝上一版行程并给出意见，请先解决其中的问题再输出，无则忽略）
{feedback}
【技能规范】（命中意图时的领域指令，供参考，无则忽略）
{skill}

输出 JSON 结构（务必严格遵守）：
{{
  "days": [
    {{
      "day": 1,
      "date": "YYYY-MM-DD",
      "theme": "当日主题（如 西湖文化游）",
      "weather": "当日天气摘要（如 晴 22~28°C）",
      "items": [
        {{
          "type": "景点|美食|酒店|交通|自由活动|其他",
          "name": "地点/事项名称",
          "start_time": "HH:MM",
          "end_time": "HH:MM",
          "duration_minutes": 90,
          "reason": "为什么安排这里，1 句",
          "address": "地址",
          "latitude": null,
          "longitude": null,
          "image": "",
          "poi_id": "对应 POI 数据的编号则填，否则填 null"
        }}
      ],
      "route": {{"points": [], "distance_km": 0, "estimated_minutes": 0}}
    }}
  ],
  "tips": ["行程提示1", "行程提示2"]
}}

要求：
1. 逐日安排，每天 3~8 个 items，按时间顺序（上午/下午/晚上），路线尽量顺路。
2. 合理搭配景点/美食/酒店；结合天气（雨天优先室内）与用户偏好；【修改意见】非空时优先解决。
3. latitude/longitude 一律 null、image 一律空字符串、route 用空数组和 0——坐标/路线/图片由系统后续用地图服务计算，你不得编造坐标。
4. 每个 item 尽量填写 POI 数据中的编号到 poi_id。
5. date 按出行日期逐日递增。
6. 只输出 JSON，不要任何解释。"""

# 行程 JSON 非法时，带原文与错误让 LLM 修复（最多一次）
ITINERARY_REPAIR_PROMPT = """上次生成的行程 JSON 不合法，请修正后重新只输出合法 JSON。

【期望天数】{days}
【上次输出】
{invalid}

【错误】
{error}

要求：
- 输出合法 JSON：{{"days": [{{day, date, theme, weather, items: [{{type,name,start_time,end_time,duration_minutes,reason,address,latitude:null,longitude:null,image:"",poi_id}}], route: {{points:[],distance_km:0,estimated_minutes:0}}}}], "tips": []}}
- days 数组长度必须等于 {days}
- 每天 items 至少 3 个、按时间顺序、每个 item 必须有非空 name
- 只输出 JSON，不要解释。"""

# save_memory 节点使用的 prompt：把行程提炼成可复用的通用旅游知识（semantic 记忆）
KNOWLEDGE_EXTRACTOR_PROMPT = """你是旅游知识整理助手。请根据下面的行程，提炼出关于「{destination}」这个目的地的通用旅游知识，供以后规划同地或类似目的地时复用。

通用旅游知识指：季节/天气特点与应对、交通出行提示、美食特色、住宿建议、值得去的景点类型、避坑经验等，与具体某天的排期无关。

【行程】
{itinerary}

要求：
1. 只输出一条条独立的、可复用的知识点，每条一句话。
2. 输出格式为 JSON 字符串数组，例如 ["知识1", "知识2"]。
3. 如果没有值得提炼的知识，输出 []。
4. 不要输出数组以外的任何解释文字。"""

# summarize_conversation 节点使用的 prompt：把被裁掉的旧对话合并进累计摘要（短期记忆扩容）
HISTORY_SUMMARIZER_PROMPT = """你是对话摘要助手。请把下面的「更早的对话历史」合并进「已有摘要」，生成一段精炼、连贯的中文累计摘要，供后续规划同一行程时延续上下文。

请用第三人称、事实化表述，保留关键信息：目的地、天数、用户偏好、已确认的行程要点、用户曾给出的修改意见与最终结果。

【已有摘要】
{existing_summary}

【更早的对话历史】
{history}

要求：只输出合并后的完整摘要（若已有摘要为空则只总结对话历史），不要输出任何解释、标题或标记。"""

# 对话式输入 -> 意图识别（Intent Router）
# 注意：不要凭是否包含"旅游"字样判断——如「东京有哪些值得去的地方？」是 TRAVEL_QA
INTENT_ROUTER_PROMPT = """你是旅游助手的意图识别器。请判断用户这句输入属于哪一类意图，只输出 JSON。

【用户输入】
{query}

【意图类别】
- TRAVEL_PLANNING: 用户给出了目的地（或可推断的目的地）、日期/天数等信息，需要规划具体行程。如「9月2号到9月5号去天津，轻松点」「帮我把杭州4天的行程排一下」
- DESTINATION_RECOMMENDATION: 用户没有目的地、不知道去哪，需要推荐目的地。如「国庆5天不知道去哪」「有什么适合亲子游的地方推荐一下」
- TRAVEL_QA: 旅游相关的问题，只需回答、不需要规划行程。如「东京有哪些值得去的地方」「去日本需要签证吗」「上海有什么好吃的」
- ITINERARY_MODIFICATION: 用户想修改已生成的行程（删减/增补/调整某天的安排）。如「第三天太累了，删掉两个景点」「把第二天的午餐换成火锅」
- NON_TRAVEL: 与旅游完全无关的请求。如编程问题、无关事务、与旅游无关的闲聊提问

【要求】
1. 不要仅凭是否包含「旅游/旅行/出行」字样来判断：例如「东京有哪些值得去的地方？」是 TRAVEL_QA 而不是 NON_TRAVEL。
2. 寒暄（你好、谢谢）归为 TRAVEL_QA。
3. 只输出 JSON：{{"intent": "TRAVEL_PLANNING", "reason": "一句话理由"}}，不要输出任何其他文字。"""

# 旅游问答（TRAVEL_QA 分支用）
TRAVEL_QA_PROMPT = """你是资深旅游顾问。请简洁地回应用户关于旅游的问题或推荐请求。

【问题】
{query}

要求：
1. 结合你的旅游知识给出具体、可执行的建议；若是求推荐目的地，给出 3~5 个目的地及一句话理由。
2. 不要编造实时数据（价格、营业时间、门票等），不确定的注明「以官方/最新信息为准」。
3. 回答精炼，200 字以内，分点或简短段落。"""

# 目的地推荐（DESTINATION_RECOMMENDATION 分支）：输出结构化 JSON 卡片
DESTINATION_RECOMMENDATION_PROMPT = """你是资深旅游规划师。用户还没有确定目的地，请根据出行背景推荐 3~6 个目的地。

【出行背景】
{query}

【今日日期】
{today}

只输出 JSON，格式：
{{"destinations": [
  {{
    "destination": "城市或地区中文名",
    "country": "国家/地区",
    "reason": "推荐理由，2~3 句，结合出行背景说清为什么适合",
    "best_for": "适合人群/场景，如 亲子/情侣/文化游/美食游",
    "recommended_days": 5,
    "estimated_budget_level": "经济 或 中等 或 奢华",
    "image": "",
    "highlights": ["亮点1", "亮点2", "亮点3"]
  }}
]}}

要求：
1. 推荐 3~6 个目的地，差异要明显（城市/自然/人文/海边等不同类型）。
2. destination 用中文常用名；recommended_days 为 1~30 的整数。
3. image 一律填空字符串（图片由系统从地图服务获取，不要编造图片 URL）。
4. 只输出 JSON，不要任何解释文字。"""

# 攻略/游记研究摘要（Travel Research Pipeline）：紧凑结构，非整篇文章
TRAVEL_RESEARCH_SUMMARY_PROMPT = """你是旅游研究分析师。请为规划 {destination} 的 {days} 天行程，整理一份紧凑的研究摘要。

【目的地】{destination}
【用户偏好】{preference}
【攻略来源】
{sources_note}

【今日日期】
{today}

输出 JSON，字段：
- area_clusters: ["景点集中的区域（如：西湖景区、老城区）"]
- common_routes: ["常见走法（如：D1 市中心 -> D2 西湖）"]
- popular_combinations: ["热门搭配（如：灵隐寺 + 西湖）"]
- transportation_tips: ["交通提示（如：地铁/渡轮/班车）"]
- avoid: ["避坑提醒（如：XX 节假日人多、XX 门票需预约）"]
- practical_tips: ["实用建议（门票/预约/季节/穿着/行程强度）"]

要求：
1. 每条一句话，精炼；每类 3~8 条。
2. 不要输出 sources 字段（来源由系统单独管理）。
3. 只输出 JSON，不要任何解释文字。"""

# 推荐 JSON 非法时，带原文让 LLM 修复（最多一次）
DESTINATION_REPAIR_PROMPT = """上次生成的目的地推荐 JSON 不合法，请修正后重新只输出合法 JSON。

【出行背景】
{query}

【今日日期】
{today}

【上次输出】
{invalid}

【错误原因】
{error}

要求：
- 必须输出 JSON 对象 {{"destinations": [3~6 个目的地卡片]}}
- 每个卡片字段：destination, country, reason, best_for, recommended_days(整数), estimated_budget_level, image(空字符串), highlights(字符串数组)
- 只输出 JSON，不要解释文字。"""

# 对话式输入 -> 关键实体抽取（目的地 / 起止日期 / 偏好）
ENTITY_EXTRACTOR_PROMPT = """你是旅游行程信息抽取助手。请从用户的一句话里抽取出旅游规划需要的关键信息。

【用户输入】
{query}

【今天日期】{today}

要求：输出一个 JSON 对象，字段：
- destination: 目的地（城市或地区名，去掉"去/到/在"等语气词）；若无法确定则为空字符串
- start_date: 出发日期，格式 YYYY-MM-DD；用户说"X月X号/日"时结合"今天"推断年份，若该日期今年已过则推为明年；无法确定则空字符串
- end_date: 返程日期，格式 YYYY-MM-DD（含当天）；只说了天数时按 start_date + 天数推算；无法确定则空字符串
- preference: 出行偏好（如 轻松/历史/美食/亲子/爬山 等，可多个用逗号分隔）；无则空字符串

只输出 JSON，不要任何解释文字。例如：{{"destination": "天津", "start_date": "2026-09-02", "end_date": "2026-09-05", "preference": "轻松"}}"""