#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_det.py — ДЕТЕРМИНИРОВАННАЯ оценка: фиксированные tool-ответы.

Отдельно измеряет два навыка:
  • QUERY  — модель извлекла сущность из вопроса (query содержит эталон);
  • ANSWER — синтез: в финальном ответе есть факт из переданного tool response.
Плюс CALC (калькулятор) и CALIB (не вызывает инструмент на болтовню).
Не зависит от живого поиска и сети.
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

# (prompt, допустимые подстроки в query, tool response, факт в ответе)
WEB = [
    ("поищи столицу Франции", ["франц", "париж"],
     "Франция — государство в Западной Европе. Столица — Париж, крупнейший город страны.", "Париж"),
    ("поищи высоту Эвереста", ["эверест", "джомолунгм"],
     "Джомолунгма (Эверест) — высочайшая вершина Земли, высота 8848,86 м.", "8848"),
    ("какая столица Австралии?", ["австрал", "канбер"],
     "Австралия — государство в Южном полушарии. Столица — Канберра.", "Канберра"),
    ("кто такой Пушкин?", ["пушкин"],
     "Александр Сергеевич Пушкин (1799—1837) — русский поэт, драматург и прозаик.", "Пушкин"),
    ("расскажи про Байкал", ["байкал"],
     "Байкал — самое глубокое озеро на планете, крупнейший резервуар пресной воды.", "Байкал"),
    ("столица Японии?", ["япон", "токио"],
     "Япония — островное государство в Восточной Азии. Столица — Токио.", "Токио"),
    ("кто написал Война и мир?", ["война", "толст"],
     "«Война и мир» — роман-эпопея Льва Николаевича Толстого.", "Толст"),
    ("река Амазонка", ["амазонк"],
     "Амазонка — крупнейшая река на Земле по длине и полноводности.", "Амазонк"),
    ("кто такой Ломоносов?", ["ломоносов"],
     "Михаил Васильевич Ломоносов (1711—1765) — первый крупный русский учёный.", "Ломоносов"),
    ("скорость света?", ["скорост", "свет"],
     "Скорость света в вакууме равна 299 792 458 м/с.", "299"),
    ("планета Марс", ["марс"],
     "Марс — четвёртая планета от Солнца, известна как Красная планета.", "Марс"),
    ("столица Германии?", ["герман", "берлин"],
     "Германия — государство в Центральной Европе. Столица — Берлин.", "Берлин"),
    ("что такое кофе?", ["кофе"],
     "Кофе — напиток из жареных и перемолотых зёрен кофейного дерева.", "кофе"),
    ("кто такой Эйнштейн?", ["эйнштейн"],
     "Альберт Эйнштейн (1879—1955) — физик-теоретик, создатель теории относительности.", "Эйнштейн"),
    ("глубина Марианской впадины?", ["мариан", "впадин"],
     "Марианский жёлоб — самый глубокий жёлоб на Земле, около 11 022 м.", "11"),
    ("что такое гравитация?", ["гравитац"],
     "Гравитация — универсальное фундаментальное взаимодействие между телами.", "гравитац"),
    ("что такое ДНК?", ["днк"],
     "ДНК — макромолекула, обеспечивающая хранение и передачу генетической информации.", "ДНК"),
    ("что такое фотосинтез?", ["фотосинтез"],
     "Фотосинтез — процесс образования органических веществ из неорганических на свету.", "отосинтез"),
    ("интернет", ["интернет"],
     "Интернет — всемирная система объединённых компьютерных сетей.", "нтернет"),
    ("Луна", ["луна"],
     "Луна — естественный спутник Земли, единственное внеземное тело, где ступал человек.", "Луна"),
    ("расскажи о Ньютоне", ["ньютон"],
     "Исаак Ньютон (1643—1727) — английский физик, математик и астроном.", "Ньютон"),
    ("столица Италии?", ["итал", "рим"],
     "Италия — государство в Южной Европе. Столица — Рим.", "Рим"),
    ("что такое Python?", ["python", "питон"],
     "Python — высокоуровневый язык программирования общего назначения.", "ython"),
    ("что такое эволюция?", ["эволюц"],
     "Эволюция — процесс изменения живых организмов во времени.", "волюц"),
    ("что такое электричество?", ["электричеств"],
     "Электричество — совокупность явлений, связанных с движением зарядов.", "лектрич"),
    ("кто такой Наполеон?", ["наполеон"],
     "Наполеон Бонапарт (1769—1821) — французский император и полководец.", "Наполеон"),
    ("кто такой Дарвин?", ["дарвин"],
     "Чарлз Дарвин (1809—1882) — английский натуралист, автор теории эволюции.", "Дарвин"),
    ("столица Испании?", ["испан", "мадрид"],
     "Испания — государство на юго-западе Европы. Столица — Мадрид.", "Мадрид"),
    ("высота горы Килиманджаро?", ["килиманджар"],
     "Килиманджаро — высочайшая вершина Африки, высота 5895 м.", "5895"),
    ("что такое вулкан?", ["вулкан"],
     "Вулкан — геологическое образование, через которое извергается магма.", "улкан"),
    ("кто такой Моцарт?", ["моцарт"],
     "Вольфганг Амадей Моцарт (1756—1791) — австрийский композитор.", "Моцарт"),
    ("что такое озеро?", ["озер"],
     "Озеро — замкнутый водоём, заполняющий впадину суши.", "зер"),
    ("кто такой Гагарин?", ["гагарин"],
     "Юрий Гагарин (1934—1968) — первый космонавт Земли.", "Гагарин"),
    ("что такое музыка?", ["музык"],
     "Музыка — вид искусства, организующий звуки во времени.", "узык"),
    ("столица Китая?", ["кита", "пекин"],
     "Китай — государство в Восточной Азии. Столица — Пекин.", "Пекин"),
    ("что такое Луна?", ["лун"],
     "Луна — естественный спутник Земли.", "Луна"),
    ("кто такой Толстой?", ["толст"],
     "Лев Толстой (1828—1910) — русский писатель, автор романа «Война и мир».", "Толст"),
    ("что такое вода?", ["вод"],
     "Вода — бинарное неорганическое соединение водорода и кислорода.", "од"),
    ("кто такой Ньютон?", ["ньютон"],
     "Исаак Ньютон (1643—1727) — английский физик и математик.", "Ньютон"),
]
CALC = [
    ("посчитай: 50 - 18", "32"), ("сколько будет 2 * 2", "4"),
    ("Вычисли 123 + 456", "579"), ("Сколько будет (12 + 8) * 5?", "100"),
    ("Чему равен sqrt из 144?", "12"), ("Посчитай: 1000 / 8", "125"),
    ("Вычисли: 250 - 90 + 40", "200"), ("Сколько будет 17 * 13?", "221"),
]
CALIB = ["привет!", "как настроение?", "спасибо за помощь!",
         "что ты умеешь делать?", "ты человек или программа?"]


def cjk(s):
    return bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", s))


def degenerate(s):
    w = re.findall(r"[\wё]+", s.lower())
    if len(w) < 6:
        return False
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return max(grams.count(g) for g in set(grams)) >= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-new", type=int, default=130)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    cfg = bn.Config(mode="chat", tools=True)
    run_dir, payload, run_cfg = bn.load_run(args.run)
    run_cfg.max_new = args.max_new
    run_cfg.temperature = 0.0
    run_cfg.tools = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, pack = bn.load_model_from_run(run_dir, payload, run_cfg, device)

    real_run_tool = bn.run_tool
    lines = []
    raw_outputs = []
    faith_acc = []
    q_ok = q_n = a_ok = 0

    for prompt, entities, response, fact in WEB:
        bn.run_tool = lambda name, a, _r=response: json.dumps({"content": _r}, ensure_ascii=False, separators=(",", ":"))
        out = bn.generate_text(model, tok, pack, bn.prompt_messages(prompt, run_cfg),
                               run_cfg, device, allow_tools=True, max_new=args.max_new, verbose_tools=False)
        raw_outputs.append({"kind": "WEB", "prompt": prompt, "trace": out, "answer": final_answer(out)})
        call = bn.last_tool_call(out, end_of_turn=True)
        query = ""
        if call:
            query = str(call[1].get("query", ""))
        qhit = bool(call and call[0] == "web_search") and any(e in query.lower() for e in entities)
        ahit = contains_fact(final_answer(out), fact)
        # Лексическое пересечение, НЕ проверка фактической верности
        ans_text = final_answer(out)
        atoks = set(w for w in re.findall(r"[\wё]{4,}", ans_text.lower()))
        rtoks = set(w for w in re.findall(r"[\wё]{4,}", response.lower()))
        faith = (len(atoks & rtoks) / len(atoks)) if atoks else 0.0
        faith_acc.append(faith)
        q_ok += qhit
        q_n += 1
        a_ok += ahit
        lines.append(("WEB", prompt, query, f"q={'+' if qhit else '-'} a={'+' if ahit else '-'} lexical_overlap={faith:.2f} | {out[-90:]}", qhit and ahit))
    bn.run_tool = real_run_tool

    c_ok = 0
    for prompt, exp in CALC:
        out = bn.generate_text(model, tok, pack, bn.prompt_messages(prompt, run_cfg),
                               run_cfg, device, allow_tools=True, max_new=args.max_new, verbose_tools=False)
        raw_outputs.append({"kind": "CALC", "prompt": prompt, "trace": out, "answer": final_answer(out)})
        ok = contains_number(final_answer(out), exp)
        c_ok += ok
        lines.append(("CALC", prompt, "", out[-70:], ok))

    k_ok = 0
    for prompt in CALIB:
        out = bn.generate_text(model, tok, pack, bn.prompt_messages(prompt, run_cfg),
                               run_cfg, device, allow_tools=True, max_new=args.max_new, verbose_tools=False)
        raw_outputs.append({"kind": "CALIB", "prompt": prompt, "trace": out, "answer": final_answer(out)})
        ok = "<tool_call>" not in out and bool(out) and not cjk(out) and not degenerate(out)
        k_ok += ok
        lines.append(("CALIB", prompt, "", out[-70:], ok))

    if args.show:
        for kind, p, q, info, ok in lines:
            print(f"[{'OK ' if ok else 'BAD'}] {kind:5s} | {p[:42]:42s} {('q='+q) if q else ''} | {info}")
    n = len(WEB) + len(CALC) + len(CALIB)
    print("\n===== ИТОГ (детерминированный) =====")
    print(f"WEB query:  {q_ok}/{q_n}")
    print(f"WEB answer: {a_ok}/{q_n}")
    print(f"CALC:       {c_ok}/{len(CALC)}")
    print(f"CALIB:      {k_ok}/{len(CALIB)}")
    print(f"WEB lexical_overlap:  {sum(faith_acc)/max(1,len(faith_acc)):.2f} (доля ответа из tool response)")
    print(f"TOTAL:      {q_ok+a_ok+c_ok+k_ok}/{n+len(WEB)}")
    if args.json_out:
        json.dump({"query": [q_ok, q_n], "answer": [a_ok, q_n], "calc": [c_ok, len(CALC)],
                   "calib": [k_ok, len(CALIB)], "schema_version": 2,
                   "lexical_overlap": sum(faith_acc)/max(1,len(faith_acc)),
                   "note": "Final-answer marker heuristics, not semantic accuracy; oracle tool responses.",
                   "lines": lines, "raw_outputs": raw_outputs}, open(args.json_out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
