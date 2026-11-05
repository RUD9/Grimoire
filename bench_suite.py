#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_suite.py — единый прогон сравнений Compact Dual Dynamic LLM с
конкурентами того же веса (30-40M).

Метрики:
  • BPB (bits-per-byte) на held-out RU и EN корпусах — не зависит от словаря;
  • instruction-following на одних и тех же русских промптах (автооценка +
    сырые ответы для ручной проверки);
  • скорость генерации (tok/s) в fp16 на GPU.

Результаты сохраняются инкрементально в bench_results.json, чтобы прогон можно
было прервать/продолжить.

Запуск:  python bench_suite.py [ours|hf|instr|speed|all]
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import struct
import sys
import time
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

_orig_load = torch.load
def _load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)
torch.load = _load

ROOT = Path("/media/gamedisk3/Grimoire")
OUT = ROOT / "bench_results.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUB = Path.home() / ".cache" / "huggingface" / "hub"

RU_CORPUS = (ROOT / "bench_corpus.txt").read_text(encoding="utf-8")
EN_CORPUS = (ROOT / "bench_corpus_en.txt").read_text(encoding="utf-8")

# ---------------------------------------------------------------- модели
# lang — язык, на котором модель в лучших условиях (вопросы задаются на нём).
OURS = [
    dict(label="ours_36m (assistant8)", run="runs/assistant8", lang="ru"),
    dict(label="ours_base_36m (night_glm9q)", run="runs/night_glm9q", lang="ru"),
    dict(label="ours_129m (assistant_big6)", run="runs/assistant_big6", lang="ru"),
]
HF = [
    dict(label="Veyra-30M-Instruct", name="veyra-ai/Veyra-30M-Instruct", instruct=True, lang="en"),
    dict(label="Veyra2-Mango-30M-Instruct", name="veyra-ai/Veyra2-Mango-30M-Instruct", instruct=True, lang="en"),
    dict(label="Stentor-30M-Instruct", name="StentorLabs/Stentor-30M-Instruct", instruct=True, lang="en"),
    dict(label="Tensor-2-40m-instruct", name="VGribAI/Tensor-2-40m-instruct", instruct=True, lang="ru"),
    dict(label="TinyStories-Instruct-33M", name="roneneldan/TinyStories-Instruct-33M", instruct=True, lang="en"),
    dict(label="TinyStories-45M", name="razor5050/TinyStories-45M", instruct=False, lang="en"),
    dict(label="ru-babylm-urop-shivan", name="climb-mao/russian-babylm-urop-shivan", instruct=False, lang="ru"),
    dict(label="ru-babylm-ft", name="elliepreed/russian-babylm-ft", instruct=False, lang="ru"),
]

# Одинаковые по смыслу 8 задач на русском и английском: каждая модель получает
# вопросы на СВОЁМ языке (лучшие условия), результаты сравниваются напрямую.
INSTR_PROMPTS_RU = [
    "привет! как дела?",
    "сколько будет 47 * 13?",
    "какая столица Франции?",
    "кто такой Пушкин?",
    "что такое вода?",
    "расскажи про планету Марс",
    "дай совет, как начать бегать по утрам",
    "придумай название для кофейни",
]
INSTR_PROMPTS_EN = [
    "hello! how are you?",
    "what is 47 * 13?",
    "what is the capital of France?",
    "who is Pushkin?",
    "what is water?",
    "tell me about the planet Mars",
    "give advice on how to start running in the morning",
    "come up with a name for a coffee shop",
]
INSTR_PROMPTS = {"ru": INSTR_PROMPTS_RU, "en": INSTR_PROMPTS_EN}

# ожидаемые маркеры для автооценки (lower-case); None = «связный ответ»
INSTR_EXPECT = {
    "ru": {
        "привет! как дела?": None,
        "сколько будет 47 * 13?": ["611"],
        "какая столица Франции?": ["париж"],
        "кто такой Пушкин?": ["пушкин", "поэт", "александр"],
        "что такое вода?": ["вода", "водород", "кислород", "жидкост", "h2o"],
        "расскажи про планету Марс": ["марс", "красн", "планет", "солнц"],
        "дай совет, как начать бегать по утрам": None,
        "придумай название для кофейни": None,
    },
    "en": {
        "hello! how are you?": None,
        "what is 47 * 13?": ["611"],
        "what is the capital of France?": ["paris"],
        "who is Pushkin?": ["pushkin", "poet", "alexander"],
        "what is water?": ["water", "hydrogen", "oxygen", "liquid", "h2o"],
        "tell me about the planet Mars": ["mars", "red", "planet", "sun"],
        "give advice on how to start running in the morning": None,
        "come up with a name for a coffee shop": None,
    },
}


def _slug(name: str) -> str:
    return name.replace("/", "--")


def _hf_dir(name: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(name)


def _ensure_safetensors(name: str) -> str:
    d = _hf_dir(name)
    bins = glob.glob(os.path.join(d, "*.bin"))
    safes = glob.glob(os.path.join(d, "*.safetensors"))
    if bins and not safes:
        from safetensors.torch import save_file
        sd = torch.load(bins[0], map_location="cpu", weights_only=False)
        sd = {k: v.contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
        try:
            save_file(sd, os.path.join(d, "model.safetensors"))
        except Exception:
            sd.pop("lm_head.weight", None)
            sd.pop("model.lm_head.weight", None)
            save_file(sd, os.path.join(d, "model.safetensors"))
    return d


def _load_hf(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = _ensure_safetensors(name)
    try:
        tok = AutoTokenizer.from_pretrained(d)
    except Exception:
        try:
            from transformers import PreTrainedTokenizerFast
            tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(d, "tokenizer.json"))
        except Exception:
            # sentencepiece-токенизаторы (TinyStories-45M и др.)
            from transformers import LlamaTokenizerFast
            tok = LlamaTokenizerFast(vocab_file=os.path.join(d, "sp_model.model"))
    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32).to(DEV).eval()
    return model, tok, d


def st_params(path: str) -> int:
    """Число параметров в safetensors без дублей tied-весов."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    seen = {}
    total = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        p = 1
        for s in v.get("shape", []):
            p *= s
        seen[k] = p
    for k, p in seen.items():
        if k.endswith("lm_head.weight") and any(
                x in seen for x in ("transformer.wte.weight", "model.embed_tokens.weight")):
            continue
        total += p
    return total


# ---------------------------------------------------------------- BPB
def bpb_ours(run: str, corpus: str, max_len: int = 512):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bn", str(ROOT / "build_neuroset.py"))
    bn = importlib.util.module_from_spec(spec)
    sys.modules["bn"] = bn
    spec.loader.exec_module(bn)
    run_dir, payload, cfg = bn.load_run(run)
    model, tok, pack = bn.load_model_from_run(run_dir, payload, cfg, DEV)
    model.eval()
    old_to_new = bn.get_forward_map(pack, len(tok))
    ids = [int(old_to_new[i]) for i in tok.encode(corpus, add_special_tokens=False)]
    nbytes = len(corpus.encode("utf-8"))
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(ids), max_len):
            chunk = ids[s:s + max_len]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk], device=DEV)
            logits = model(x)[0].float()
            logp = F.log_softmax(logits, dim=-1)
            tgt = torch.tensor(chunk[1:], device=DEV)
            total += -logp[:-1].gather(1, tgt[:, None]).sum().item()
            n += len(chunk) - 1
    del model
    torch.cuda.empty_cache()
    return total / math.log(2) / nbytes, n


def bpb_hf(name: str, corpus: str, max_len: int = 512):
    model, tok, _ = _load_hf(name)
    ids = tok.encode(corpus)
    nbytes = len(corpus.encode("utf-8"))
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(ids), max_len):
            chunk = ids[s:s + max_len]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk], device=DEV)
            logits = model(x).logits[0].float()
            logp = F.log_softmax(logits, dim=-1)
            tgt = torch.tensor(chunk[1:], device=DEV)
            total += -logp[:-1].gather(1, tgt[:, None]).sum().item()
            n += len(chunk) - 1
    del model
    torch.cuda.empty_cache()
    return total / math.log(2) / nbytes, n


# ---------------------------------------------------------------- instr
def cjk(s: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", s))


def degenerate(s: str) -> bool:
    w = re.findall(r"[\wё]+", s.lower())
    if len(w) < 6:
        return False
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return max(grams.count(g) for g in set(grams)) >= 3


def score_instr(prompt: str, out: str, lang: str) -> bool:
    if not out or cjk(out) or degenerate(out):
        return False
    exp = INSTR_EXPECT[lang].get(prompt)
    if exp is None:
        return len(out.strip()) >= 4
    low = out.lower()
    return any(e in low for e in exp)


def instr_ours(run: str, lang: str = "ru", max_new: int = 150):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bn", str(ROOT / "build_neuroset.py"))
    bn = importlib.util.module_from_spec(spec)
    sys.modules["bn"] = bn
    spec.loader.exec_module(bn)
    run_dir, payload, cfg = bn.load_run(run)
    cfg.max_new = max_new
    cfg.temperature = 0.0
    cfg.tools = True
    model, tok, pack = bn.load_model_from_run(run_dir, payload, cfg, DEV)
    out = {}
    for p in INSTR_PROMPTS[lang]:
        a = bn.generate_text(model, tok, pack, bn.prompt_messages(p, cfg), cfg, DEV,
                             allow_tools=True, max_new=max_new, verbose_tools=False)
        out[p] = a.strip()
    del model
    torch.cuda.empty_cache()
    return out


def instr_hf(name: str, instruct: bool, lang: str = "en", max_new: int = 100):
    model, tok, _ = _load_hf(name)
    out = {}
    for p in INSTR_PROMPTS[lang]:
        if instruct:
            try:
                ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                              add_generation_prompt=True, return_tensors="pt")
            except Exception:
                ids = tok(f"User: {p}\nAssistant:", return_tensors="pt").input_ids
        else:
            ids = tok(p, return_tensors="pt").input_ids
        ids = ids.to(DEV)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        out[p] = tok.decode(g[0, ids.size(1):], skip_special_tokens=True).strip()
    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- speed
def speed_ours(run: str, max_new: int = 50, warmup: int = 3, runs: int = 10):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bn", str(ROOT / "build_neuroset.py"))
    bn = importlib.util.module_from_spec(spec)
    sys.modules["bn"] = bn
    spec.loader.exec_module(bn)
    run_dir, payload, cfg = bn.load_run(run)
    model, tok, pack = bn.load_model_from_run(run_dir, payload, cfg, DEV)
    cfg.tools = False
    msgs = bn.prompt_messages("привет", cfg)
    old_ids = tok.encode(bn.build_prompt_text(msgs), add_special_tokens=False)
    o2n = bn.get_forward_map(pack, len(tok))
    ids = torch.tensor([[int(o2n[t]) for t in old_ids]], device=DEV)
    anchor = torch.tensor([ids.size(1)], device=DEV)

    def gen():
        logits, cache = model.prefill_cached(ids, anchor)
        nxt = logits.argmax(-1, keepdim=True)
        toks = [int(nxt.item())]
        for _ in range(max_new - 1):
            logits = model.step_cached(nxt, cache)
            nxt = logits.argmax(-1, keepdim=True)
            toks.append(int(nxt.item()))
        return toks

    for _ in range(warmup):
        gen()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        toks = gen()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return runs * len(toks) / dt, len(toks)


def speed_hf(name: str, max_new: int = 50, warmup: int = 3, runs: int = 10):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model, tok, _ = _load_hf(name)
    ids = tok("привет", return_tensors="pt").input_ids.to(DEV)
    if getattr(tok, "chat_template", None):
        try:
            ids = tok.apply_chat_template([{"role": "user", "content": "привет"}],
                                          add_generation_prompt=True, return_tensors="pt").to(DEV)
        except Exception:
            pass
    kw = dict(max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False,
              pad_token_id=tok.eos_token_id)
    with torch.no_grad():
        for _ in range(warmup):
            model.generate(ids, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            g = model.generate(ids, **kw)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n_gen = int(g.shape[1] - ids.shape[1])
    del model
    torch.cuda.empty_cache()
    return runs * n_gen / dt, n_gen


# ---------------------------------------------------------------- main
def load_results():
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8"))
    return {"meta": {"device": str(DEV),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
            "bpb": {}, "instr": {}, "instr_score": {}, "speed": {}}


def save(res):
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    modes = set(sys.argv[1:]) or {"all"}
    if "all" in modes:
        modes = {"ours", "hf", "instr", "speed"}
    res = load_results()
    print(f"Device: {DEV} | {res['meta'].get('gpu')} | modes={sorted(modes)}")

    if modes & {"ours", "hf"}:
        for m in OURS:
            label, run = m["label"], m["run"]
            for lang, corpus in (("RU", RU_CORPUS), ("EN", EN_CORPUS)):
                key = f"{label}|{lang}"
                if key in res["bpb"]:
                    continue
                try:
                    b, n = bpb_ours(run, corpus)
                    res["bpb"][key] = round(b, 3)
                    print(f"[BPB] {label:30s} {lang} = {b:.3f} ({n} tok)", flush=True)
                    save(res)
                except Exception as e:
                    print(f"[BPB] {label} {lang} ERR {e}")
        for m in HF:
            label, name = m["label"], m["name"]
            for lang, corpus in (("RU", RU_CORPUS), ("EN", EN_CORPUS)):
                key = f"{label}|{lang}"
                if key in res["bpb"]:
                    continue
                try:
                    b, n = bpb_hf(name, corpus)
                    res["bpb"][key] = round(b, 3)
                    print(f"[BPB] {label:30s} {lang} = {b:.3f} ({n} tok)", flush=True)
                    save(res)
                except Exception as e:
                    print(f"[BPB] {label} {lang} ERR {str(e)[:90]}")

    if "instr" in modes:
        for m in OURS:
            label = m["label"]
            if label in res["instr"]:
                continue
            try:
                out = instr_ours(m["run"], m["lang"])
                res["instr"][label] = out
                res["instr_score"][label] = [
                    sum(score_instr(p, a, m["lang"]) for p, a in out.items()), len(out)]
                print(f"[INSTR] {label:30s} ({m['lang']}) {res['instr_score'][label]}", flush=True)
                save(res)
            except Exception as e:
                print(f"[INSTR] {label} ERR {e}")
        for m in HF:
            label = m["label"]
            if label in res["instr"]:
                continue
            try:
                out = instr_hf(m["name"], m["instruct"], m["lang"])
                res["instr"][label] = out
                res["instr_score"][label] = [
                    sum(score_instr(p, a, m["lang"]) for p, a in out.items()), len(out)]
                print(f"[INSTR] {label:30s} ({m['lang']}) {res['instr_score'][label]}", flush=True)
                save(res)
            except Exception as e:
                print(f"[INSTR] {label} ERR {str(e)[:90]}")

    if "speed" in modes:
        for m in OURS:
            label = m["label"]
            if label in res["speed"]:
                continue
            try:
                tps, n = speed_ours(m["run"])
                res["speed"][label] = round(tps, 2)
                print(f"[SPEED] {label:30s} {tps:.2f} tok/s", flush=True)
                save(res)
            except Exception as e:
                print(f"[SPEED] {label} ERR {e}")
        for m in HF:
            label = m["label"]
            if label in res["speed"]:
                continue
            try:
                tps, n = speed_hf(m["name"])
                res["speed"][label] = round(tps, 2)
                print(f"[SPEED] {label:30s} {tps:.2f} tok/s", flush=True)
                save(res)
            except Exception as e:
                print(f"[SPEED] {label} ERR {str(e)[:90]}")

    # параметры моделей
    if "params" not in res:
        res["params"] = {}
    for m in OURS:
        p = ROOT / m["run"] / "config.json"
        if p.exists():
            res["params"][m["label"]] = json.loads(p.read_text()).get("params", {}).get("TOTAL")
    for m in HF:
        if res["params"].get(m["label"]):
            continue
        try:
            model, _, _ = _load_hf(m["name"])
            res["params"][m["label"]] = sum(p.numel() for p in model.parameters())
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    save(res)
    print("\n" + json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
