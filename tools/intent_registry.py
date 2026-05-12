
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class IntentSpec:

    name: str
    description: str
    description_en: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    params_en: Optional[Dict[str, Any]] = None
    when_to_use: List[str] = field(default_factory=list)
    when_to_use_en: Optional[List[str]] = None
    examples: List[str] = field(default_factory=list)
    examples_en: Optional[List[str]] = None
    priority: int = 100

    expose_condition: Optional[Callable] = field(default=None, compare=False, hash=False)


class IntentRegistry:

    _intents: Dict[str, IntentSpec] = {}
    _order: List[str] = []

    @classmethod
    def register(cls, spec: IntentSpec) -> None:
        cls._intents[spec.name] = spec
        if spec.name not in cls._order:
            cls._order.append(spec.name)

    @classmethod
    def get(cls, name: str) -> Optional[IntentSpec]:
        return cls._intents.get(name)

    @classmethod
    def all(cls) -> List[IntentSpec]:
        return sorted(cls._intents.values(), key=lambda s: (s.priority, s.name))

    @classmethod
    def active(cls, ctx=None) -> List[IntentSpec]:
        result = []
        for spec in cls.all():
            if spec.expose_condition is None:
                result.append(spec)
            elif ctx is not None:
                try:
                    if spec.expose_condition(ctx):
                        result.append(spec)
                except Exception:
                    result.append(spec)  # expose on error
            else:
                result.append(spec)
        return result

    @classmethod
    def render_actions_block(cls, language: str = "zh", ctx=None) -> str:
        lang = str(language or "zh").lower()
        use_en = lang.startswith("en")
        lines: List[str] = []
        tools = cls.active(ctx) if ctx is not None else cls.all()
        for idx, spec in enumerate(tools, 1):
            desc = spec.description_en if (use_en and spec.description_en) else spec.description
            params = spec.params_en if (use_en and spec.params_en is not None) else spec.params
            when_to_use = spec.when_to_use_en if (use_en and spec.when_to_use_en is not None) else spec.when_to_use
            examples = spec.examples_en if (use_en and spec.examples_en is not None) else spec.examples

            lines.append(f"{idx}. {spec.name}: {desc}")
            if params:
                lines.append(f"   - params: {params}")
            if when_to_use:
                for w in when_to_use:
                    lines.append(f"   - {w}")
            if examples:
                lines.append("   - Examples:" if use_en else "   - 示例：")
                for ex in examples[:4]:
                    lines.append(f"     {ex}")
            lines.append("")
        return "\n".join(lines).strip()

    @classmethod
    def render_compact_block(cls, language: str = "zh", ctx=None) -> str:
        """Render a compact skill block for SkillDispatcher (single LLM call).

        Each skill gets:
          - 1 line: name + short description (≤50 chars)
          - 1 line: params (key list only, if any)
          - 1 line: most important usage hint

        Target: ~500-800 tokens total for all active skills.
        """
        lang = str(language or "zh").lower()
        use_en = lang.startswith("en")
        lines: List[str] = []
        tools = cls.active(ctx) if ctx is not None else cls.all()
        for idx, spec in enumerate(tools, 1):
            desc = spec.description_en if (use_en and spec.description_en) else spec.description
            params = spec.params_en if (use_en and spec.params_en is not None) else spec.params
            when_to_use = spec.when_to_use_en if (use_en and spec.when_to_use_en is not None) else spec.when_to_use

            # Truncate description to first sentence, max 50 chars
            short_desc = desc.split(". ")[0].split("。")[0].strip()
            if len(short_desc) > 50:
                short_desc = short_desc[:47] + "..."
            lines.append(f"{idx}. {spec.name}: {short_desc}")

            # Params as key list
            if params:
                keys = list(params.keys())
                lines.append(f"   params: {', '.join(keys)}")

            # Max 1 core hint, strip emoji
            for w in (when_to_use or [])[:1]:
                clean = w.replace("🔥 ", "").replace("🔥", "").strip()
                if clean:
                    # Also truncate hint to 80 chars
                    if len(clean) > 80:
                        clean = clean[:77] + "..."
                    lines.append(f"   - {clean}")

        return "\n".join(lines).strip()

    @classmethod
    def render_rules_block(cls, language: str = "zh", ctx=None) -> str:
        lang = str(language or "zh").lower()
        use_en = lang.startswith("en")
        rules: List[str] = []
        tools = cls.active(ctx) if ctx is not None else cls.all()
        for spec in tools:
            when_to_use = spec.when_to_use_en if (use_en and spec.when_to_use_en is not None) else spec.when_to_use
            for w in when_to_use:
                if "🔥" in w or "=> " in w or "→" in w or "MUST" in w:
                    rules.append(w)
        if not rules:
            return ""
        seen = set()
        uniq = []
        for r in rules:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return "\n".join([f"- {r}" for r in uniq]).strip()
