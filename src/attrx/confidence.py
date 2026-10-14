"""Агрегация кандидатов от разных источников в одно значение + калибруемая уверенность."""
from __future__ import annotations

from collections import defaultdict

from .schema import Candidate

SOURCE_WEIGHT = {"human": 1.0, "seller": 1.0, "rules": 1.0, "ml": 0.9, "llm": 0.85}


def aggregate(cands: list[Candidate]) -> tuple[str | float | None, float, list[Candidate], bool]:
    """Возвращает (значение, уверенность, поддерживающие кандидаты, есть_конфликт).

    Согласие независимых источников повышает уверенность (noisy-OR),
    конфликт — снижает на величину поддержки альтернативы.
    """
    if not cands:
        return None, 0.0, [], False
    groups: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        key = str(c.value).lower() if isinstance(c.value, str) else f"{float(c.value):g}"
        groups[key].append(c)

    def support(group: list[Candidate]) -> float:
        best_by_source: dict[str, float] = {}
        for c in group:
            s = min(0.99, c.score * SOURCE_WEIGHT[c.source])
            best_by_source[c.source] = max(best_by_source.get(c.source, 0.0), s)
        miss = 1.0
        for s in best_by_source.values():
            miss *= 1 - s
        return 1 - miss

    ranked = sorted(groups.values(), key=support, reverse=True)
    top = support(ranked[0])
    conflict = len(ranked) > 1
    conf = top - 0.5 * support(ranked[1]) if conflict else top
    return ranked[0][0].value, round(max(0.0, min(1.0, conf)), 3), ranked[0], conflict
