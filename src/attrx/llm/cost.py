"""Стоимость и латентность LLM-слоя: client.stats() → «₽ на 1000 карточек» по тарифам config/llm_prices.yaml.

Два сценария из кейса («LLM на каждую карточку дорого → гибрид»):
- every_card — LLM вызывается на каждой карточке;
- hybrid — LLM только для хвоста: llm_share — доля карточек, где пайплайн реально позвал LLM
  (считается прогоном пайплайна; повторные карточки дополнительно экономит кэш).
Для on-prem платим не за токены, а за час GPU: (амортизация + электричество) × время на карточку.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..config import get_settings


def load_prices(path: Path | str | None = None) -> dict:
    p = Path(path) if path else get_settings().llm_prices_path
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {"providers": {}}


def hourly_cost(h: dict) -> float:
    """₽/час работы GPU-сервера: амортизация капзатрат по рабочим часам + электричество."""
    hours = float(h.get("amortization_years", 3)) * 8760 * float(h.get("utilization", 1.0))
    return float(h.get("capex_rub", 0)) / hours + float(h.get("power_kw", 0)) * float(h.get("electricity_rub_per_kwh", 0))


def cost_per_call(p: dict, prompt_tokens: float, completion_tokens: float, usd_rub: float,
                  seconds_per_call: float | None = None) -> float | None:
    """₽ за один вызов LLM у провайдера p (None — тариф неизвестен)."""
    if "hourly" in p:
        per_hour = hourly_cost(p["hourly"])
        if p.get("throughput_cards_per_hour"):
            return per_hour / float(p["throughput_cards_per_hour"])
        return per_hour * seconds_per_call / 3600 if seconds_per_call else None
    pin, pout = p.get("input_per_1k"), p.get("output_per_1k")
    if pin is None and p.get("input_per_1k_usd") is not None:
        pin, pout = p["input_per_1k_usd"] * usd_rub, p["output_per_1k_usd"] * usd_rub
    if pin is None or pout is None:
        return None
    return (prompt_tokens * pin + completion_tokens * pout) / 1000


def per_1000_cards(stats: dict, n_cards: int | None = None, *, llm_share: float | None = None,
                   prices: dict | None = None, seconds_per_call: float | None = None) -> dict:
    """Стоимость на 1000 карточек по всем провайдерам из прайса для «LLM на каждую карточку» и «гибрида».

    stats — client.stats() за прогон (нужны calls, prompt_tokens, completion_tokens, latency_s.avg);
    n_cards — сколько карточек было в прогоне (тогда доля гибрида = calls / n_cards, если llm_share не задан);
    seconds_per_call — время вызова для on-prem (по умолчанию средняя латентность из stats).
    """
    prices = prices or load_prices()
    calls = max(int(stats.get("calls") or 0), 1)
    pt = (stats.get("prompt_tokens") or 0) / calls
    ct = (stats.get("completion_tokens") or 0) / calls
    sec = seconds_per_call or (stats.get("latency_s") or {}).get("avg")
    share = llm_share
    if share is None and n_cards:
        share = min(1.0, (stats.get("calls") or 0) / n_cards)
    usd_rub = float(prices.get("usd_rub", 90))
    rows = []
    for pid, p in (prices.get("providers") or {}).items():
        c = cost_per_call(p, pt, ct, usd_rub, sec)
        rows.append({
            "provider": pid, "title": p.get("title", pid),
            "rub_per_call": None if c is None else round(c, 5),
            "every_card_rub_per_1000": None if c is None else round(1000 * c, 2),
            "hybrid_rub_per_1000": None if c is None or share is None else round(1000 * c * share, 2),
            "note": p.get("note", ""),
        })
    return {
        "tokens_per_call": {"prompt": round(pt, 1), "completion": round(ct, 1)},
        "seconds_per_call": None if sec is None else round(sec, 3),
        "hours_per_1000_cards_sequential": None if sec is None else round(sec * 1000 / 3600, 2),
        "llm_share_hybrid": share,
        "currency": prices.get("currency", "RUB"),
        "rows": rows,
    }


def cost_table_md(report: dict) -> str:
    """Markdown-таблица из per_1000_cards (для docs/LLM.md и отчёта)."""
    def fmt(v: float | None) -> str:
        return "неизвестно" if v is None else f"{v:,.2f}".replace(",", " ").replace(".", ",")

    head = "| Провайдер | ₽ / 1000 карточек: LLM на каждую | ₽ / 1000 карточек: гибрид |\n|---|---|---|\n"
    return head + "".join(f"| {r['title']} | {fmt(r['every_card_rub_per_1000'])} | "
                          f"{'—' if report['llm_share_hybrid'] is None else fmt(r['hybrid_rub_per_1000'])} |\n"
                          for r in report["rows"])
