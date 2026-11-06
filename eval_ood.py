#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_ood.py — оценка чекпойнта на фиксированном наборе промптов.

Категории:
  CALC  — арифметика: в ответе есть ожидаемое число;
  WEB   — поиск: модель вызвала web_search и в ответе есть факт из tool response;
  GEN   — общий формат: непусто, по-русски, без вырождения, в пределах длины;
  CALIB — калибровка: на болтовню/мета модель НЕ вызывает инструмент.
"""
import argparse
import json
import re
import sys
import importlib.util
from pathlib import Path
from eval_utils import final_answer, contains_number, contains_fact

spec = importlib.util.spec_from_file_location("bn", str(Path(__file__).resolve().parent / "build_neuroset.py"))
bn = importlib.util.module_from_spec(spec)
sys.modules["bn"] = bn
spec.loader.exec_module(bn)
import torch

CALC = [
    ("посчитай: 50 - 18", "32"),
    ("сколько будет 2 * 2", "4"),
    ("Вычисли 123 + 456", "579"),
    ("Сколько будет (12 + 8) * 5?", "100"),
    ("Чему равен sqrt из 144?", "12"),
    ("Посчитай: 1000 / 8", "125"),
    ("Вычисли: 250 - 90 + 40", "200"),
    ("Сколько будет 17 * 13?", "221"),
]
WEB = [
    ("поищи столицу Франции", "Париж"),
    ("поищи высоту Эвереста", "8848"),
    ("какая столица Австралии?", "Канберра"),
    ("кто такой Пушкин?", "Пушкин"),
    ("расскажи про Байкал", "Байкал"),
    ("какая глубина Марианской впадины?", "глубоч"),
    ("столица Японии?", "Токио"),
    ("кто написал Война и мир?", "Толст"),
    ("река Амазонка", "Амазонк"),
    ("кто такой Ломоносов?", "Ломоносов"),
    ("скорость света?", "299"),
    ("что такое кофе?", "кофе"),
]
GEN = [
    "привет, как дела?",
    "Объясни в двух предложениях, что такое дружба.",
    "Придумай название для кофейни.",
    "Дай совет, как начать бегать по утрам.",
    "Что лучше: чай или кофе? Ответь кратко.",
    "Напиши короткое поздравление с днём рождения.",
    "Назови три цвета радуги.",
    "Придумай слоган для книжного магазина.",
]
CALIB = [
    "привет!",
    "как настроение?",
    "спасибо за помощь!",
    "что ты умеешь делать?",
    "ты человек или программа?",
]


def cjk(s):
    return bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", s))


def degenerate(s):
    w = re.findall(r"[\wё]+", s.lower())
    if len(w) < 6:
        return False
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return max(grams.count(g) for g in set(grams)) >= 3


def score(run, device="cuda", max_new=120, tools=True):
    cfg = bn.Config(mode="chat", tools=tools)
    run_dir, payload, run_cfg = bn.load_run(run)
    run_cfg.max_new = max_new
    run_cfg.temperature = 0.0
    run_cfg.tools = tools
    model, tok, pack = bn.load_model_from_run(run_dir, payload, run_cfg, device)

    res = {"calc": [0, 0], "web": [0, 0], "gen": [0, 0], "calib": [0, 0], "lines": []}

    def gen(prompt):
        return bn.generate_text(model, tok, pack, bn.prompt_messages(prompt, run_cfg),
                                run_cfg, device, allow_tools=tools, max_new=max_new,
                                verbose_tools=False)

    def rec(kind, q, a, note, ok):
        res[kind][1] += 1
        res[kind][0] += int(ok)
        res["lines"].append((kind.upper(), q, a, note, ok))

    for q, exp in CALC:
        a = gen(q)
        rec("calc", q, a, f"ожид. {exp}", contains_number(final_answer(a), exp))
    for q, fact in WEB:
        a = gen(q)
        rec("web", q, a, f"факт {fact}", contains_fact(final_answer(a), fact) and "<tool_response>" in a)
    for q in GEN:
        a = gen(q)
        answer = final_answer(a)
        ok = bool(answer) and not cjk(answer) and not degenerate(answer) and 3 <= len(answer) <= 1500
        rec("gen", q, a, "формат", ok)
    for q in CALIB:
        a = gen(q)
        ok = "<tool_call>" not in a and bool(a) and not cjk(a) and len(a) <= 1500
        rec("calib", q, a, "без tool_call", ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    res = score(args.run, device=args.device, max_new=args.max_new, tools=not args.no_tools)
    if args.show:
        for kind, q, a, note, ok in res["lines"]:
            print(f"[{'OK ' if ok else 'BAD'}] {kind} | {q}\n      -> {a[:200].replace(chr(10), ' | ')}")
    print("\n===== ИТОГ =====")
    tot = [0, 0]
    for k in ("calc", "web", "gen", "calib"):
        got, n = res[k]
        print(f"{k:6s}: {got}/{n}")
        tot[0] += got
        tot[1] += n
    print(f"TOTAL: {tot[0]}/{tot[1]} = {100*tot[0]/max(1,tot[1]):.0f}%")
    if args.json_out:
        json.dump(dict(res, schema_version=2, note="Heuristic marker/format checks, not semantic accuracy"),
                  open(args.json_out, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
