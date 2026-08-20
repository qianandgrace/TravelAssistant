"""Skills 渐进式披露：领域指令按需加载。

每个 Skill = skills/<name>/SKILL.md：
- YAML frontmatter 里的 name/description 常驻 prompt（一行，省 token）；
- 正文是命中该意图后的完整指令，按需注入（workflow 传入 build_workflow(skills=...)、
  react 通过 dynamic_prompt middleware 在每轮请求里注入）。

不新增领域知识、不引入网络来源：正文从 utils/prompts.py 对应意图的 prompt 要点提炼，
保证与无 skills 的 baseline 可比。
"""
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


@dataclass(frozen=True)
class Skill:
    """一个意图型 Skill：name/description 常驻，content 按需加载。"""
    name: str
    description: str
    content: str

    def body(self) -> str:
        """拼接成注入 prompt 的完整技能块（带标题，便于模型区分）。"""
        return f"【技能：{self.description}】\n{self.content}"


def _parse_skill_file(path: Path) -> Skill:
    """解析单个 SKILL.md：手工 split --- 取 frontmatter(yaml) + 正文。"""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---")
    if len(parts) < 3:
        raise ValueError(f"{path} 不是合法 SKILL.md（缺 --- 围栏）")
    meta = yaml.safe_load(parts[1]) or {}
    body = "---".join(parts[2:]).strip()
    name = str(meta.get("name") or path.parent.name).strip()
    return Skill(
        name=name,
        description=str(meta.get("description") or "").strip(),
        content=body,
    )


def load_skill(name: str) -> Skill:
    """按 name 读 skills/<name>/SKILL.md。"""
    path = SKILLS_DIR / name / "SKILL.md"
    return _parse_skill_file(path)


def load_all_skills() -> list[Skill]:
    """读 skills/*/SKILL.md 全部技能（目录名与 frontmatter.name 不一致时以 frontmatter 为准）。"""
    skills = []
    if not SKILLS_DIR.exists():
        logger.warning("skills 目录不存在：%s", SKILLS_DIR)
        return skills
    for path in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        try:
            skills.append(_parse_skill_file(path))
        except Exception as e:  # noqa: BLE001 - 单个 skill 解析失败不阻断其余
            logger.warning("解析 skill 失败（跳过）：%s -> %s", path, e)
    return skills


def skill_descriptions(skills: list[Skill] | None = None) -> str:
    """常驻 prompt 片段：一行一个 skill 的 name — description，供匹配触发。"""
    items = skills if skills is not None else load_all_skills()
    if not items:
        return "（无可用技能）"
    lines = [f"- {s.name}：{s.description}" for s in items]
    return "\n".join(lines)


_INTENT_TO_SKILL = {
    "TRAVEL_PLANNING": "itinerary-planning",
    "ITINERARY_MODIFICATION": "itinerary-planning",
    "DESTINATION_RECOMMENDATION": "destination-recommendation",
    "TRAVEL_QA": "travel-qa",
    "NON_TRAVEL": "non-travel",
}


def select_skill_for_query(query: str, skills: list[Skill] | None = None) -> Skill | None:
    """规则型选择器：免费、确定性，命中意图后返回对应的单个 Skill。

    复用 utils.intent_router._fallback_intent 的意图判断（不额外调 LLM，
    token 计量干净）；未命中任何意图（不可能，兜底是 TRAVEL_PLANNING）时返回 None。
    """
    from utils.intent_router import _fallback_intent  # 局部导入避免循环依赖

    items = skills if skills is not None else load_all_skills()
    if not items:
        return None
    intent = _fallback_intent(query or "")
    name = _INTENT_TO_SKILL.get(intent)
    return next((s for s in items if s.name == name), None)
