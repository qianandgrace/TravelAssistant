"""前端产品视图渲染（Phase 7）：行程时间线卡片 + SVG 路线地图 + 目的地推荐卡片。

纯函数：输入 itinerary_data / 推荐卡片，输出自包含 HTML（内嵌 CSS），
无外部 JS / 图片依赖（地图为内联 SVG；图片用 <img loading="lazy"> 懒加载）。
webapp.py（Gradio 前端）与 Phase 8（HTML 导出）共用本模块。

坐标说明：高德返回 GCJ-02 坐标。SVG 地图用等距柱状投影 + cos(lat) 校正，
保证城市尺度下形状大致正确；单点/无坐标时优雅降级。
"""
import html
import math
import re

# item 类型 -> 标签配色（文字色, 背景色）
TYPE_COLORS = {
    "景点": ("#0ea5e9", "#e0f2fe"),
    "美食": ("#d97706", "#fef3c7"),
    "酒店": ("#7c3aed", "#ede9fe"),
    "交通": ("#059669", "#d1fae5"),
    "自由活动": ("#64748b", "#f1f5f9"),
    "其他": ("#94a3b8", "#f1f5f9"),
}
TYPE_DEFAULT = ("#64748b", "#f1f5f9")

# 每日路线配色（地图折线 / 图例 / 卡片头部色条）
DAY_COLORS = ["#4f7cff", "#2e9e6b", "#e07b39", "#9b5de5", "#f15b6c", "#00b8d9"]

_CSS = """
.tp-root{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;color:#1f2937;line-height:1.5}
.tp-header{border:1px solid #e5e7eb;border-left:4px solid #4f7cff;border-radius:10px;padding:12px 16px;background:#fff;margin-bottom:14px}
.tp-header .tp-h1{font-size:18px;font-weight:700;margin:0 0 4px}
.tp-header .tp-meta{font-size:13px;color:#6b7280;display:flex;flex-wrap:wrap;gap:4px 14px}
.tp-layout{display:grid;grid-template-columns:55% 45%;gap:16px;align-items:start}
.tp-col-map{position:sticky;top:8px}
@media(max-width:900px){.tp-layout{grid-template-columns:1fr}.tp-col-map{position:static}}
.tp-day{border:1px solid #e5e7eb;border-radius:12px;margin-bottom:14px;overflow:hidden;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.tp-day-top{display:flex;align-items:center;flex-wrap:wrap;gap:8px 12px;padding:10px 14px;background:#f8fafc;border-bottom:1px solid #eef2f7}
.tp-day-badge{background:#4f7cff;color:#fff;border-radius:6px;padding:2px 8px;font-weight:700;font-size:13px}
.tp-day-date{font-size:12px;color:#6b7280}
.tp-day-theme{font-weight:600;font-size:14px}
.tp-day-weather{font-size:12px;color:#b45309;background:#fef9c3;padding:1px 8px;border-radius:99px}
.tp-day-route{font-size:12px;color:#059669;background:#ecfdf5;padding:1px 8px;border-radius:99px}
.tp-items{list-style:none;margin:0;padding:4px 0}
.tp-item{display:flex;gap:12px;padding:10px 14px;border-top:1px solid #f3f4f6}
.tp-item:first-child{border-top:none}
.tp-item-num{flex:0 0 22px;height:22px;border-radius:50%;background:#eef2f7;color:#475569;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;margin-top:2px}
.tp-item-img{flex:0 0 92px;width:92px;height:66px;object-fit:cover;border-radius:8px;background:#f1f5f9}
.tp-item-ph{flex:0 0 92px;width:92px;height:66px;border-radius:8px;background:#f1f5f9;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:11px}
.tp-item-body{flex:1;min-width:0}
.tp-item-title{display:flex;align-items:center;flex-wrap:wrap;gap:6px}
.tp-item-time{font-size:12px;color:#6b7280;font-family:ui-monospace,Consolas,monospace}
.tp-item-name{font-weight:600;font-size:14px}
.tp-item-type{font-size:11px;padding:0 7px;border-radius:99px}
.tp-item-reason{font-size:12.5px;color:#4b5563;margin-top:2px}
.tp-item-addr{font-size:11.5px;color:#9ca3af;margin-top:2px}
.tp-tips{border:1px solid #fde68a;background:#fffbeb;border-radius:10px;padding:10px 14px;margin-top:6px}
.tp-tips b{color:#92400e;font-size:13px}
.tp-tips li{font-size:12.5px;color:#78350f;margin:2px 0}
.tp-src{margin-top:8px}
.tp-src details{border:1px solid #e5e7eb;border-radius:10px;background:#fff;padding:0}
.tp-src summary{cursor:pointer;font-size:12.5px;color:#374151;padding:8px 12px;list-style:none;display:flex;align-items:center;gap:6px}
.tp-src summary::-webkit-details-marker{display:none}
.tp-src summary::before{content:"▸";transition:transform .15s;font-size:11px;color:#64748b}
.tp-src details[open] summary::before{transform:rotate(90deg)}
.tp-src ul{list-style:none;margin:0;padding:4px 12px 10px 26px}
.tp-src li{margin:3px 0}
.tp-src a{font-size:12.5px;color:#0369a1;text-decoration:none;word-break:break-all}
.tp-src a:hover{text-decoration:underline}
.tp-src .tp-src-num{font-size:12.5px;color:#64748b;font-weight:600}
.tp-mapcard{border:1px solid #e5e7eb;border-radius:12px;padding:10px;background:#fff}
.tp-mapcard h4{margin:0 0 6px;font-size:13px;color:#374151}
.tp-mapzoom{position:relative}
.tp-mapzoom-tools{position:absolute;top:8px;right:8px;z-index:5;display:flex;gap:4px}
.tp-mapzoom-tools button{width:26px;height:26px;border:1px solid #cbd5e1;border-radius:6px;background:#fff;color:#334155;font-size:14px;line-height:1;cursor:pointer;box-shadow:0 1px 2px rgba(0,0,0,.06)}
.tp-mapzoom-tools button:hover{background:#f1f5f9}
.tp-mapviewport{overflow:hidden;position:relative;border-radius:10px;cursor:grab;touch-action:none;user-select:none;-webkit-user-select:none}
.tp-mapviewport.grabbing{cursor:grabbing}
.tp-mapviewport svg{display:block;width:100%;height:auto;transform-origin:0 0}
.tp-rec-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.tp-rec-card{border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;background:#fff}
.tp-rec-card img{width:100%;height:130px;object-fit:cover;display:block;background:#f1f5f9}
.tp-rec-body{padding:10px 12px}
.tp-rec-name{font-weight:700;font-size:15px}
.tp-rec-meta{font-size:12px;color:#6b7280;margin:2px 0 4px}
.tp-rec-reason{font-size:12.5px;color:#4b5563}
.tp-rec-tags{font-size:11.5px;color:#0369a1;margin-top:6px}
"""


def _esc(v) -> str:
    return html.escape(str(v or ""), quote=True)


# 地图缩放平移 JS：滚轮缩放（以光标为中心）、拖拽平移、＋/−/复位按钮。
# 作用域限定在 .tp-mapviewport 内；dataset.zoomReady 防止重复绑定。
_MAP_ZOOM_JS = """<script>
(function(){
  var vp=document.querySelector('.tp-mapviewport');
  if(!vp||vp.dataset.zoomReady)return; vp.dataset.zoomReady='1';
  var svg=vp.querySelector('svg'); if(!svg)return;
  var scale=1,tx=0,ty=0,drag=null;
  function apply(){ svg.style.transform='translate('+tx+'px,'+ty+'px) scale('+scale+')'; }
  function clampDrag(){ /* 缩放为1时不允许拖出视野 */ }
  function zoomAt(px,py,f){
    var ns=Math.max(1,Math.min(6,scale*f));
    var k=ns/scale;
    tx=px-(px-tx)*k; ty=py-(py-ty)*k; scale=ns; apply();
  }
  var btns=vp.querySelectorAll('.tp-mapzoom-tools button');
  for(var i=0;i<btns.length;i++){
    btns[i].addEventListener('click',function(e){
      e.stopPropagation();
      var z=this.getAttribute('data-z');
      var w=vp.clientWidth,h=vp.clientHeight;
      if(z==='in')zoomAt(w/2,h/2,1.25);
      else if(z==='out')zoomAt(w/2,h/2,0.8);
      else{scale=1;tx=0;ty=0;apply();}
    });
  }
  vp.addEventListener('wheel',function(e){
    e.preventDefault();
    var r=vp.getBoundingClientRect();
    zoomAt(e.clientX-r.left,e.clientY-r.top, e.deltaY<0?1.15:0.87);
  },{passive:false});
  vp.addEventListener('mousedown',function(e){ if(e.button!==0)return; drag={x:e.clientX-tx,y:e.clientY-ty}; vp.classList.add('grabbing'); });
  window.addEventListener('mousemove',function(e){ if(!drag)return; tx=e.clientX-drag.x; ty=e.clientY-drag.y; apply(); });
  window.addEventListener('mouseup',function(){ drag=null; vp.classList.remove('grabbing'); });
  vp.addEventListener('touchstart',function(e){ var t=e.touches[0]; drag={x:t.clientX-tx,y:t.clientY-ty}; vp.classList.add('grabbing'); },{passive:true});
  vp.addEventListener('touchmove',function(e){ if(!drag||e.touches.length!==1)return; var t=e.touches[0]; tx=t.clientX-drag.x; ty=t.clientY-drag.y; apply(); e.preventDefault(); },{passive:false});
  vp.addEventListener('touchend',function(){ drag=null; vp.classList.remove('grabbing'); });
})();
</script>"""


def _km_width(lng_span: float, lat_mid: float) -> float:
    return abs(lng_span) * 111.0 * math.cos(math.radians(lat_mid))


def render_svg_map(days: list, width: int = 560, height: int = 440) -> str:
    """把每天有坐标的 item 画成 SVG：按天配色折线 + 全局编号 marker + 图例。

    无任何坐标时返回占位文本，保证永不崩。
    """
    pts = []  # (day_index, name, lng, lat)
    for di, day in enumerate(days):
        for it in day.get("items") or []:
            lng, lat = it.get("longitude"), it.get("latitude")
            if lng is None or lat is None:
                continue
            try:
                pts.append((di, it.get("name", ""), float(lng), float(lat)))
            except (TypeError, ValueError):
                continue
    if not pts:
        return (
            "<svg width='%d' height='%d' viewBox='0 0 %d %d' style='width:100%%;height:auto;"
            "border-radius:10px;background:#f8fafc'>"
            "<rect width='100%%' height='100%%' fill='#f8fafc'/>"
            "<text x='50%%' y='50%%' fill='#94a3b8' font-size='13' text-anchor='middle'>"
            "暂无地图数据（坐标缺失）</text></svg>" % (width, height, width, height)
        )

    lngs = [p[2] for p in pts]
    lats = [p[3] for p in pts]
    min_lng, max_lng = min(lngs), max(lngs)
    min_lat, max_lat = min(lats), max(lats)
    lng_span = max_lng - min_lng
    lat_span = max_lat - min_lat
    if lng_span < 1e-6 and lat_span < 1e-6:
        lng_span = lat_span = 0.01
        min_lng -= 0.005
        min_lat -= 0.005
    elif lng_span < 1e-6:
        lng_span = 0.01
        min_lng -= 0.005
    elif lat_span < 1e-6:
        lat_span = 0.01
        min_lat -= 0.005

    w_km = _km_width(lng_span, (max_lat + min_lat) / 2)
    h_km = abs(lat_span) * 111.0
    ratio = (h_km or 1e-9) / (w_km or 1e-9)
    if ratio > 1:
        view_w = max(1, int(height / ratio))
        view_h = height
    else:
        view_w = width
        view_h = max(1, int(width * ratio))
    pad = 0.08
    padx = int(view_w * pad)
    pady = int(view_h * pad)
    inner_w = view_w - 2 * padx
    inner_h = view_h - 2 * pady
    if inner_w < 1:
        inner_w = 1
    if inner_h < 1:
        inner_h = 1

    def X(lng: float) -> float:
        return padx + (lng - min_lng) / lng_span * inner_w

    def Y(lat: float) -> float:
        return pady + (max_lat - lat) / lat_span * inner_h

    parts = [(
        "<svg width='%d' height='%d' viewBox='0 0 %d %d' style='width:100%%;height:auto;"
        "border-radius:10px;background:#f8fafc'>"
    ) % (width, height, view_w, view_h)]

    for di in range(len(days)):
        line_pts = [(X(p[2]), Y(p[3])) for p in pts if p[0] == di]
        if len(line_pts) >= 2:
            path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in line_pts)
            color = DAY_COLORS[di % len(DAY_COLORS)]
            parts.append(
                f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2.2' "
                f"stroke-opacity='0.85' stroke-linecap='round' stroke-linejoin='round'/>"
            )

    for idx, (di, name, lng, lat) in enumerate(pts, 1):
        x, y = X(lng), Y(lat)
        color = DAY_COLORS[di % len(DAY_COLORS)]
        parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='7' fill='{color}' stroke='#fff' stroke-width='1.5'>"
            f"<title>{_esc(name)}</title></circle>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{y + 2.6:.1f}' font-size='9' fill='#fff' text-anchor='middle' "
            f"font-weight='bold'>{idx}</text>"
        )

    ly = view_h - 14
    parts.append(f"<text x='{padx}' y='{ly + 3}' font-size='10' fill='#64748b'>每日路线</text>")
    for di in range(len(days)):
        cx = padx + 52 + di * 46
        color = DAY_COLORS[di % len(DAY_COLORS)]
        parts.append(
            f"<line x1='{cx - 8}' y1='{ly}' x2='{cx + 8}' y2='{ly}' stroke='{color}' stroke-width='2.5'/>"
        )
        parts.append(f"<text x='{cx + 11}' y='{ly + 3}' font-size='9' fill='#475569'>D{di + 1}</text>")
    parts.append("</svg>")
    svg = "".join(parts)
    return (
        "<div class='tp-mapzoom'>"
        "<div class='tp-mapzoom-tools'>"
        "<button type='button' data-z='in' title='放大'>＋</button>"
        "<button type='button' data-z='out' title='缩小'>－</button>"
        "<button type='button' data-z='reset' title='复位'>⟲</button>"
        "</div>"
        f"<div class='tp-mapviewport'>{svg}</div>"
        "</div>"
        + _MAP_ZOOM_JS
    )


def _render_day_card(day: dict, item_start: int) -> str:
    day_no = day.get("day", "")
    date = _esc(day.get("date"))
    theme = _esc(day.get("theme"))
    weather = _esc(day.get("weather"))
    route = day.get("route") or {}
    km = route.get("distance_km") or 0
    minutes = route.get("estimated_minutes") or 0
    di = max(0, (day_no or 1) - 1)
    badge_color = DAY_COLORS[di % len(DAY_COLORS)]

    chips = []
    if weather:
        chips.append(f"<span class='tp-day-weather'>{weather}</span>")
    if km:
        chips.append(f"<span class='tp-day-route'>路线 {km} km · 约 {minutes} 分钟</span>")
    chips_html = "".join(chips)

    items_html = []
    for i, it in enumerate(day.get("items") or []):
        num = item_start + i + 1
        itype = it.get("type") or "其他"
        fg, bg = TYPE_COLORS.get(itype, TYPE_DEFAULT)
        img = str(it.get("image") or "").strip()
        if img:
            media = (
                f"<img class='tp-item-img' src='{_esc(img)}' alt='{_esc(it.get('name'))}' "
                f"loading='lazy' referrerpolicy='no-referrer'>"
            )
        else:
            media = "<div class='tp-item-ph'>暂无图片</div>"
        t = f"{_esc(it.get('start_time'))}~{_esc(it.get('end_time'))}".strip("~")
        time_span = f"<span class='tp-item-time'>{_esc(t)}</span>" if t else ""
        reason = f"<div class='tp-item-reason'>{_esc(it.get('reason'))}</div>" if it.get("reason") else ""
        addr = f"<div class='tp-item-addr'>{_esc(it.get('address'))}</div>" if it.get("address") else ""
        items_html.append(
            f"<li class='tp-item'>"
            f"<span class='tp-item-num'>{num}</span>{media}"
            f"<div class='tp-item-body'>"
            f"<div class='tp-item-title'>{time_span}<span class='tp-item-name'>{_esc(it.get('name'))}</span>"
            f"<span class='tp-item-type' style='color:{fg};background:{bg}'>{_esc(itype)}</span></div>"
            f"{reason}{addr}"
            f"</div></li>"
        )

    return (
        f"<section class='tp-day'>"
        f"<div class='tp-day-top'>"
        f"<span class='tp-day-badge' style='background:{badge_color}'>第 {day_no} 天</span>"
        f"<span class='tp-day-theme'>{theme}</span>"
        f"<span class='tp-day-date'>{date}</span>{chips_html}"
        f"</div>"
        f"<ul class='tp-items'>{''.join(items_html)}</ul>"
        f"</section>"
    )


def render_itinerary_product_html(result: dict) -> str:
    """行程产品视图 HTML。result 含 itinerary_data + destination/days/preference/start_date 等。

    桌面端左侧 55% 时间线、右侧 45% 地图（sticky）；窄屏堆叠。
    """
    data = result.get("itinerary_data") or {}
    days = data.get("days") or []
    destination = result.get("destination") or ""
    preference = result.get("preference") or ""
    day_count = result.get("days") or len(days)
    start_date = result.get("start_date") or ""
    tips = data.get("tips") or []
    sources = data.get("sources") or []

    total_km = sum((d.get("route") or {}).get("distance_km") or 0 for d in days)
    total_min = sum((d.get("route") or {}).get("estimated_minutes") or 0 for d in days)
    last_date = days[-1].get("date") if days else ""

    meta = []
    if start_date:
        date_txt = last_date if last_date and last_date != start_date else start_date
        meta.append(f"出行：{start_date} 起" + (f" · {date_txt} 止" if last_date else ""))
    if day_count:
        meta.append(f"{day_count} 天")
    if preference:
        meta.append(f"偏好：{preference}")
    if total_km:
        meta.append(f"全程步行约 {total_km} km / {total_min} 分钟")
    meta_html = "".join(f"<span>{_esc(m)}</span>" for m in meta)

    header = (
        f"<div class='tp-header'>"
        f"<div class='tp-h1'>{_esc(destination)} · 行程规划</div>"
        f"<div class='tp-meta'>{meta_html}</div>"
        f"</div>"
    )

    # 全局连续编号（地图 marker 与卡片编号一一对应）
    item_start = 0
    day_html = ""
    for day in days:
        day_html += _render_day_card(day, item_start)
        item_start += len(day.get("items") or [])

    tips_html = ""
    if tips:
        tips_html = "<div class='tp-tips'><b>小贴士</b><ul>" + "".join(
            f"<li>{_esc(t)}</li>" for t in tips
        ) + "</ul></div>"
    # 攻略/游记来源：可展开下拉，逐条给出可点击链接（真实来源，绝不编造）
    if sources:
        items_html = "".join(
            f"<li><a href='{_esc(s.get('url', ''))}' target='_blank' rel='noopener'>"
            f"{_esc(s.get('title') or s.get('url') or '（未命名来源）')}</a></li>"
            for s in sources if s.get("url") or s.get("title")
        )
        if items_html:
            src_html = (
                "<div class='tp-src'><details><summary>"
                "<span class='tp-src-num'>参考了 " + str(len(sources)) + " 个攻略/游记来源</span>"
                "<span style='color:#9ca3af;font-size:11px'>（点击展开查看链接）</span>"
                "</summary><ul>" + items_html + "</ul></details></div>"
            )
        else:
            src_html = ""
    else:
        src_html = ""

    timeline = (
        f"<div class='tp-col-main'>{day_html}{tips_html}{src_html}</div>"
    )
    mapcard = (
        f"<div class='tp-mapcard'><h4>行程地图（编号与左侧卡片一致）</h4>{render_svg_map(days)}</div>"
    )
    layout = f"<div class='tp-layout'>{timeline}<div class='tp-col-map'>{mapcard}</div></div>"

    return f"<style>{_CSS}</style><div class='tp-root'>{header}{layout}</div>"


def render_itinerary_export_html(result: dict) -> str:
    """把行程产品视图包装成可独立打开/分享/打印的完整 HTML 文档（Phase 8 导出）。

    Key 安全：本函数只使用 result 里的业务字段（目的地/行程/图片 URL），
    不含任何服务端 API Key；SVG 地图为内联生成，图片来自高德 CDN 公开 URL。
    """
    body = render_itinerary_product_html(result)
    destination = _esc(result.get("destination") or "行程")
    return (
        "<!DOCTYPE html>\n<html lang='zh-CN'>\n<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        f"<title>{destination} · 行程规划</title>\n"
        "<style>@media print{body{background:#fff;padding:0}}"
        ".tp-export-wrap{max-width:1100px;margin:0 auto}"
        ".tp-export-foot{margin:14px auto 0;padding-top:12px;border-top:1px solid #e5e7eb;"
        "color:#94a3b8;font-size:12px;max-width:1100px}</style>\n"
        "</head>\n<body style='background:#f1f5f9;padding:16px'>\n"
        f"<div class='tp-export-wrap'>{body}</div>\n"
        "<div class='tp-export-foot'>由 TravelAssistant 生成 · 地图与行程文字离线可查看 · "
        "图片加载需要网络 · 坐标/图片来自高德开放平台</div>\n"
        "</body>\n</html>\n"
    )


def export_filename(destination: str) -> str:
    """生成安全的导出文件名：去除非法字符、限长；空则回退『行程』。"""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(destination or "").strip())[:40]
    return f"行程_{safe}.html" if safe else "行程.html"


def render_rec_cards_html(cards: list) -> str:
    """目的地推荐卡片 HTML（含真实图片 + 亮点标签 + 懒加载）。"""
    if not cards:
        return "<div class='tp-root'>暂无推荐</div>"
    grid = []
    for c in cards:
        dest = _esc(c.get("destination"))
        country = _esc(c.get("country"))
        days = c.get("recommended_days") or ""
        budget = _esc(c.get("estimated_budget_level") or "")
        reason = _esc(c.get("reason"))
        tags = " · ".join(_esc(t) for t in (c.get("highlights") or [])[:3])
        best = _esc(c.get("best_for") or "")
        meta_parts = [f"约 {days} 天" if days else "", f"预算 {budget}" if budget else ""]
        meta = "　".join(p for p in meta_parts if p)
        img = str(c.get("image") or "").strip()
        img_html = (
            f"<img src='{_esc(img)}' alt='{dest}' loading='lazy' referrerpolicy='no-referrer'>"
            if img else "<div style='height:130px;background:#f1f5f9'></div>"
        )
        tags_parts = [best] if best else []
        if tags:
            tags_parts.append(tags)
        tags_line = "　".join(tags_parts)
        grid.append(
            f"<div class='tp-rec-card'>{img_html}"
            f"<div class='tp-rec-body'>"
            f"<div class='tp-rec-name'>{dest}<span style='color:#9ca3af;font-weight:400'>（{country}）</span></div>"
            f"<div class='tp-rec-meta'>{meta}</div>"
            f"<div class='tp-rec-reason'>{reason}</div>"
            f"<div class='tp-rec-tags'>{tags_line}</div>"
            f"</div></div>"
        )
    return f"<style>{_CSS}</style><div class='tp-root'><div class='tp-rec-grid'>{''.join(grid)}</div></div>"
