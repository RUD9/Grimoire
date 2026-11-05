#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_neuroset.py — единый самодостаточный скрипт для обучения Compact Dual Dynamic LLM.

Внутри:
  • вся архитектура (Qwen2-слой + рекуррентная генерация весов + byte-morph патчер),
  • compact-vocab со стратегиями сжатия словаря и склейкой весов,
  • подготовка данных из папки, которую вы указываете вручную (--data-dir),
  • обучение по времени (--hours) с чекпойнтами, resume, авто-валидацией и сэмплами,
  • интерактивный чат и бенч с инструментами calculate / web_search.

Ничего не импортирует из вашего проекта — единый файл. Требует torch + transformers
(для Qwen tokenizer / базового слоя), numpy.

Примеры (путь к датасету задаётся вручную всегда):
  python build_neuroset.py --preset qwen36m --data-dir data_final3 --hours 1 --yes
  python build_neuroset.py --preset night-scratch --data-dir data_ready --hours 8 --yes
  python build_neuroset.py --train-file data/train.jsonl --val-file data/val.jsonl --hours 3 --yes
  python build_neuroset.py                          # интерактивный визард (спросит путь)
  python build_neuroset.py --mode chat --run runs/night --prompt "привет"
  python build_neuroset.py --mode bench --run runs/night
  python build_neuroset.py --mode plan --preset night-scratch --data-dir data_ready

Пресеты:
  qwen36m        — 35.8M, инициализация слоем Qwen2.5-0.5B, compact vocab 11264
  night-scratch  — 22.5M, обучение с нуля на ночь (RX 6600 и подобные)
  night-scratch-pro — 35.2M, чуть шире, если хватает VRAM
  smoke          — 2-минутная проверка на CPU
"""

from __future__ import annotations

import argparse
import ast
import bisect
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if not Path("/proc/driver/nvidia").exists():
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")  # RX 6600 / ROCm
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_ckpt
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
        Qwen2RotaryEmbedding,
    )
except Exception as exc:  # pragma: no cover
    print(f"[!] Не удалось импортировать transformers/torch: {exc}")
    print("    Установите зависимости: pip install -r requirements.txt")
    raise

DEFAULT_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
IM_START, IM_END, ENDOFTEXT = 151644, 151645, 151643

# Эти литералы должны декодироваться побайтно точно. Иначе сжатый словарь может
# заменить `arguments` на `arg` и сделать tool call непарсируемым.
PROTOCOL_VOCAB_TEXTS = (
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    '{"name":"calculate","arguments":{"expression":"50 - 18"}}',
    '{"name":"web_search","arguments":{"query":"столица Франции"}}',
    "calculate", "web_search", "arguments", "expression", "query",
    "0123456789", "{}[]\":,.-+*/()",
)


# ======================================================================================
# Конфиг и пресеты
# ======================================================================================

PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen36m": dict(
        init="qwen", base_layer=6, base_layers=1, depth=8,
        vocab_size=11264, vocab_strategy="coverage", coverage=0.999, merge_weights="mean",
        max_seq_len=512, batch_size=4, grad_accum=4, lr=1.5e-5, lr_base=3e-6,
        max_tokens=15_000_000, hours=3.0, eval_seconds=240, sample_seconds=900, save_minutes=15,
    ),
    "night-scratch": dict(
        init="scratch", d_model=512, n_heads=8, d_head=64, kv_heads=2,
        base_layers=2, depth=8, vocab_size=16384, vocab_strategy="coverage", coverage=0.999,
        merge_weights="mean", max_seq_len=512, batch_size=8, grad_accum=4,
        max_tokens=40_000_000, lr=8e-4, lr_base=8e-4, hours=8.0, eval_seconds=300, sample_seconds=900, save_minutes=15,
        grad_checkpoint=True,
    ),
    "night-scratch-pro": dict(
        init="scratch", d_model=640, n_heads=10, d_head=64, kv_heads=2,
        base_layers=2, depth=8, vocab_size=24576, vocab_strategy="coverage", coverage=0.9995,
        merge_weights="mean", max_seq_len=576, batch_size=6, grad_accum=4,
        max_tokens=40_000_000, lr=7e-4, lr_base=7e-4, hours=8.0, eval_seconds=300, sample_seconds=900, save_minutes=15,
        grad_checkpoint=True,
    ),
    "smoke": dict(
        init="scratch", d_model=128, n_heads=2, d_head=64, kv_heads=1,
        base_layers=1, depth=2, vocab_size=512, vocab_strategy="coverage", coverage=0.999,
        merge_weights="mean", max_seq_len=96, batch_size=4, grad_accum=1,
        lr=1e-3, lr_base=1e-3, max_tokens=1_000_000, hours=0.03, eval_seconds=20, sample_seconds=30, save_minutes=1,
        no_samples=False, eval_seconds_min=10,
    ),
}


@dataclass
class Config:
    mode: str = "train"
    preset: str = ""

    # данные
    # Путь к датасету задаётся вручную всегда: --data-dir ПАПКА (train.jsonl/val.jsonl)
    # или --train-file/--val-file. Автосборки и фолбэка на «готовый» датасет нет.
    data_dir: str = ""
    train_file: str = ""
    val_file: str = ""
    # Только для построения нового compact-словаря: позволяет учить короткую
    # persona-фазу, но сохранить покрытие токенов основного корпуса.
    vocab_data_dir: str = ""
    max_samples: int = 0
    max_tokens: int = 30_000_000
    val_samples: int = 1000
    val_batches: int = 40
    allow_tiny_data: bool = False          # сознательно разрешить большой Qwen-ран на <1M токенов
    split_tool_segments: bool = True       # резать tool-трассы на сегменты с независимым якорем
    toolcall_weight: float = 1.0           # вес loss на токенах tool call (опция, 1.0 = выкл)

    # модель
    init: str = "scratch"                     # qwen | scratch
    model_name: str = "auto"                  # HF id или локальная папка
    tokenizer_name: str = ""
    base_layer: int = 6
    base_layers: int = 2
    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64
    kv_heads: int = 2
    intermediate_size: int = 0                # 0 = 4 * d_model
    attention_bias: bool = True
    rope_theta: float = 1e6
    rms_eps: float = 1e-6
    depth: int = 8
    gen_mlp: int = 128
    factory_prefix: int = 0                  # 0 = legacy; >0 = fixed causal conditioning prefix
    max_seq_len: int = 384

    # словарь
    vocab_size: int = 16384
    vocab_strategy: str = "coverage"          # freq | coverage | stratified | full
    coverage: float = 0.999
    merge_weights: str = "mean"               # keep | mean
    vocab_pack: str = ""
    init_ckpt: str = ""                       # стартовые веса (не resume: новый ран)
    init_skip_embeddings: bool = False         # при новом pack не переносить mapping-зависимые тензоры
    init_extend_vocab: bool = False            # новый pack сохранил прежний ID-префикс и добавил строки

    # обучение
    seed: int = 42
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 8e-4
    lr_base: float = 0.0                      # 0 = как lr
    weight_decay: float = 0.01
    warmup: int = 100
    grad_clip: float = 1.0
    min_lr_ratio: float = 0.1
    hours: float = 8.0
    extra_hours: float = 0.0
    max_steps: int = 0
    mask: str = "assistant"                   # assistant | none
    amp: str = "auto"                         # auto | fp16 | bf16 | none
    device: str = "auto"                      # auto | cuda | cpu

    # рантайм
    out: str = ""
    resume: str = ""
    run: str = ""
    # Teacher-generated corpus / bridge / DF-OPD (в соответствующих --mode)
    student_run: str = ""
    teacher: str = "Qwen/Qwen2.5-1.5B-Instruct"
    teacher_data_teacher: str = "Qwen/Qwen2.5-3B-Instruct"
    teacher_data_backend: str = "transformers"  # transformers | llamacpp
    teacher_data_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    teacher_data_parallel: int = 1
    teacher_data_samples: int = 128
    teacher_data_max_new: int = 192
    teacher_data_min_accept: float = 0.80
    # Explicit operator override: write only already-accepted rows even when
    # the aggregate gate is below threshold. Off by default.
    teacher_data_use_accepted: bool = False
    teacher_data_seed: int = 42
    bridge_samples: int = 128
    bridge_steps: int = 300
    bridge_max_new: int = 96
    bridge_batch_size: int = 2
    bridge_lr: float = 7e-6
    bridge_lr_base: float = 1.5e-6
    bridge_save_every: int = 100
    # On-policy DF-OPD: экспериментальный этап только после успешного bridge.
    teacher_device: str = "auto"
    teacher_dtype: str = "fp16"
    df_questions: int = 64
    df_steps: int = 600
    df_max_new: int = 48
    df_rollout_temp: float = 0.75
    df_top_p: float = 0.92
    df_kd_temperature: float = 1.2
    df_hidden_weight: float = 0.01
    df_anchor_dir: str = ""
    df_anchor_weight: float = 0.20
    df_lr: float = 1.5e-6
    df_lr_base: float = 4e-7
    df_save_every: int = 100
    eval_seconds: int = 240
    save_minutes: int = 15
    sample_seconds: int = 600
    log_seconds: int = 10
    no_samples: bool = False
    demo: bool = True
    demo_force: bool = False
    demo_max_new: int = 64
    export_dtype: str = "fp16"
    compile: bool = False
    grad_checkpoint: bool = False
    opt: str = "adamw"                            # adamw | muon (муон = фабрика Muon'ом, остальное AdamW)
    muon_lr: float = 0.02

    # инференс
    tools: bool = True
    max_new: int = 192
    temperature: float = 0.0
    rep_penalty: float = 1.15
    no_repeat_ngram: int = 4                # запрет повторяющихся n-грамм при генерации (0 = выкл)
    prompt: str = ""
    # ВАЖНО: по умолчанию контекст между сообщениями НЕ переносится.
    # Нейросеть не обучена удерживать контекст нескольких сообщений: в SFT каждый
    # ход диалога независим (даже в многоходовых сэмплах ответ не опирается на
    # предыдущие реплики), поэтому каждое сообщение обрабатывается отдельно.
    # Флаг --chat-context включает склейку истории (экспериментально).
    chat_context: bool = False

    # служебное
    yes: bool = False
    dry_run: bool = False

    def arch(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "attention_bias": self.attention_bias,
            "rope_theta": self.rope_theta,
            "rms_eps": self.rms_eps,
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "kv_heads": self.kv_heads,
            "intermediate_size": self.intermediate_size or self.d_model * 4,
            "base_layers": self.base_layers,
            "depth": self.depth,
            "gen_mlp": self.gen_mlp,
            "factory_prefix": self.factory_prefix,
            "max_seq_len": self.max_seq_len,
        }


def apply_preset(cfg: Config, name: str) -> Config:
    if name not in PRESETS:
        return cfg
    d = dict(PRESETS[name])
    d.pop("eval_seconds_min", None)
    for k, v in d.items():
        setattr(cfg, k, v)
    cfg.preset = name
    return cfg


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Обучение Compact Dual Dynamic LLM: визард + флаги",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", default=None, choices=["train", "chat", "bench", "plan", "teacher-probe", "teacher-data", "bridge", "distill", "teacher"], help="режим")
    p.add_argument("--preset", default=None, choices=list(PRESETS.keys()) + ["custom"], help="готовый пресет")
    p.add_argument("--config", default="", help="JSON-конфиг (плоский или сохранённый config.json рана)")

    # данные
    p.add_argument("--data-dir", default=None)
    p.add_argument("--vocab-data-dir", default=None,
                   help="готовый основной train.jsonl только для частот нового compact-словаря")
    p.add_argument("--train-file", default=None)
    p.add_argument("--val-file", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--val-samples", type=int, default=None)
    p.add_argument("--val-batches", type=int, default=None, help="сколько батчей val считать (40 = быстро)")
    p.add_argument("--allow-tiny-data", action="store_true", default=None,
                   help="явно разрешить большой Qwen-ран на <1M train-токенов (только для отладки)")

    # модель
    p.add_argument("--init", default=None, choices=["qwen", "scratch"])
    p.add_argument("--model-name", default=None, help="HF id или локальная папка Qwen")
    p.add_argument("--tokenizer", dest="tokenizer_name", default=None)
    p.add_argument("--base-layer", type=int, default=None)
    p.add_argument("--base-layers", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--d-head", type=int, default=None)
    p.add_argument("--kv-heads", type=int, default=None)
    p.add_argument("--intermediate-size", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--gen-mlp", type=int, default=None)
    p.add_argument("--factory-prefix", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)

    # словарь
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--vocab-strategy", default=None, choices=["freq", "coverage", "stratified", "full"])
    p.add_argument("--coverage", type=float, default=None)
    p.add_argument("--merge-weights", default=None, choices=["keep", "mean"])
    p.add_argument("--vocab-pack", default=None, help="готовый compact_vocab.pt")
    p.add_argument("--init-ckpt", default=None,
                   help="стартовые веса (state_dict) для нового рана, напр. unified_dynamic_persona.pt")
    p.add_argument("--init-skip-embeddings", action="store_true", default=None,
                   help="с --init-ckpt не переносить embed/lm_head/token_bytes; нужно при новом vocab pack")
    p.add_argument("--init-extend-vocab", action="store_true", default=None,
                   help="с --init-ckpt перенести прежний prefix embeddings в расширенный vocab pack")

    # обучение
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-base", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--min-lr-ratio", type=float, default=None)
    p.add_argument("--hours", type=float, default=None, help="бюджет обучения в часах")
    p.add_argument("--minutes", type=float, default=None, help="альтернатива --hours")
    p.add_argument("--extra-hours", type=float, default=None, help="добавить времени при resume")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--mask", default=None, choices=["assistant", "none"])
    p.add_argument("--amp", default=None, choices=["auto", "fp16", "bf16", "none"])
    p.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])

    # рантайм
    p.add_argument("--out", default=None, help="папка рана")
    p.add_argument("--resume", default=None, help="продолжить с чекпойнта или папки рана")
    p.add_argument("--run", default=None, help="папка рана для chat/bench")
    p.add_argument("--student-run", default=None, help="исходный run для --mode distill")
    p.add_argument("--teacher", default=None, help="white-box teacher >=1.5B для --mode bridge/distill")
    p.add_argument("--teacher-data-teacher", default=None,
                   help="teacher для --mode teacher-data; при llamacpp это лишь метка в отчёте")
    p.add_argument("--teacher-data-backend", choices=["transformers", "llamacpp"], default=None,
                   help="transformers fp16 GPU или локальный OpenAI-совместимый llama-server")
    p.add_argument("--teacher-data-url", default=None,
                   help="URL /v1/chat/completions локального llama-server")
    p.add_argument("--teacher-data-parallel", type=int, default=None,
                   help="число параллельных запросов к llama-server (--parallel на server должен быть не меньше)")
    p.add_argument("--teacher-data-samples", type=int, default=None,
                   help="сколько teacher-generated curriculum-пар собрать")
    p.add_argument("--teacher-data-max-new", type=int, default=None,
                   help="максимум новых токенов teacher-ответа")
    p.add_argument("--teacher-data-min-accept", type=float, default=None,
                   help="минимальная доля принятых teacher-ответов; иначе fail closed")
    p.add_argument("--teacher-data-use-accepted", action="store_true", default=None,
                   help="явно сохранить только уже accepted rows при провале aggregate gate; rejected rows никогда не добавляются")
    p.add_argument("--teacher-data-seed", type=int, default=None)
    p.add_argument("--bridge-samples", type=int, default=None,
                   help="сколько teacher-generated user→answer пар собрать для --mode bridge")
    p.add_argument("--bridge-steps", type=int, default=None,
                   help="число CE-шагов student по фиксированным teacher ответам")
    p.add_argument("--bridge-max-new", type=int, default=None,
                   help="максимум токенов teacher-ответа при сборке bridge")
    p.add_argument("--bridge-batch-size", type=int, default=None,
                   help="batch student bridge; 2 безопасен для 8 GB")
    p.add_argument("--bridge-lr", type=float, default=None)
    p.add_argument("--bridge-lr-base", type=float, default=None)
    p.add_argument("--bridge-save-every", type=int, default=None)
    p.add_argument("--teacher-device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--teacher-dtype", choices=["fp16", "bf16", "fp32"], default=None)
    p.add_argument("--df-questions", type=int, default=None)
    p.add_argument("--df-steps", type=int, default=None)
    p.add_argument("--df-max-new", type=int, default=None)
    p.add_argument("--df-rollout-temp", type=float, default=None)
    p.add_argument("--df-top-p", type=float, default=None)
    p.add_argument("--df-kd-temperature", type=float, default=None)
    p.add_argument("--df-hidden-weight", type=float, default=None)
    p.add_argument("--df-anchor-dir", default=None)
    p.add_argument("--df-anchor-weight", type=float, default=None)
    p.add_argument("--df-lr", type=float, default=None)
    p.add_argument("--df-lr-base", type=float, default=None)
    p.add_argument("--df-save-every", type=int, default=None)
    p.add_argument("--eval-seconds", type=int, default=None)
    p.add_argument("--log-seconds", type=int, default=None)
    p.add_argument("--save-minutes", type=int, default=None)
    p.add_argument("--sample-seconds", type=int, default=None)
    p.add_argument("--no-samples", action="store_true", default=None)
    p.add_argument("--no-demo", action="store_true", default=None,
                   help="не генерировать финальное демо (полезно на CPU)")
    p.add_argument("--demo", action="store_true", default=None,
                   help="генерировать демо даже на CPU с большой моделью")
    p.add_argument("--demo-max-new", type=int, default=None)
    p.add_argument("--export-dtype", default=None, choices=["fp16", "fp32"])
    p.add_argument("--compile", action="store_true", default=None)
    p.add_argument("--grad-checkpoint", action="store_true", default=None,
                   help="recompute вместо хранения активаций (меньше VRAM, медленнее)")
    p.add_argument("--opt", default=None, choices=["adamw", "muon"],
                   help="muon: ортогонализованный импульс для матриц фабрики (~-30..50%% шагов до того же loss)")
    p.add_argument("--muon-lr", type=float, default=None, help="lr для Muon-группы (по умолч. 0.02)")

    # инференс
    p.add_argument("--tools", dest="tools", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-new", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--rep-penalty", type=float, default=None)
    p.add_argument("--chat-context", action="store_true", default=None,
                   help="переносить историю сообщений в чате (по умолчанию выкл: "
                        "модель не обучена удерживать контекст нескольких сообщений)")
    p.add_argument("--prompt", default=None)

    p.add_argument("--plan", action="store_true", default=None,
                   help="то же, что --mode plan: замер скорости и прогноз")
    p.add_argument("--toolcall-weight", type=float, default=None)
    p.add_argument("--no-repeat-ngram", type=int, default=None)
    p.add_argument("--yes", "-y", action="store_true", default=None, help="без вопросов")
    p.add_argument("--dry-run", action="store_true", default=None, help="только собрать модель, без обучения")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    if args.config:
        path = Path(args.config)
        data = json.loads(path.read_text(encoding="utf-8"))
        if "config" in data and isinstance(data["config"], dict):
            data = data["config"]
        valid = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in valid:
                setattr(cfg, k, v)
    if args.preset and args.preset != "custom":
        apply_preset(cfg, args.preset)

    # CLI поверх пресета
    skip = {"mode", "preset", "config", "minutes", "yes", "no_samples", "compile",
            "dry_run", "tools", "grad_checkpoint", "no_demo", "demo", "init_skip_embeddings",
            "init_extend_vocab"}
    for f in fields(Config):
        if f.name in skip:
            continue
        if hasattr(args, f.name):
            v = getattr(args, f.name)
            if v is not None:
                setattr(cfg, f.name, v)
    if args.mode:
        cfg.mode = args.mode
    if args.plan:
        cfg.mode = "plan"
    if args.minutes is not None and args.hours is None:
        cfg.hours = args.minutes / 60.0
    if args.yes:
        cfg.yes = True
    if args.no_samples:
        cfg.no_samples = True
    if args.no_demo:
        cfg.demo = False
    if args.demo:
        cfg.demo = True
        cfg.demo_force = True
    if args.compile:
        cfg.compile = True
    if args.grad_checkpoint:
        cfg.grad_checkpoint = True
    if args.init_skip_embeddings:
        cfg.init_skip_embeddings = True
    if args.init_extend_vocab:
        cfg.init_extend_vocab = True
    if args.dry_run:
        cfg.dry_run = True
    if args.tools is not None:
        cfg.tools = args.tools
    return cfg


# ======================================================================================
# Архитектура (1:1 устройство из model_engine.py, с обобщением размеров)
# ======================================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """1:1 из model_engine.py (и работает на torch < 2.4, где нет nn.RMSNorm)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        norm = x * torch.rsqrt(var + self.eps)
        return (self.weight * norm).to(dt)


class SafeByteMorphPatcher(nn.Module):
    """Байтовый «морфологический» патчер: 8 байт токена -> вектор d_model."""

    def __init__(self, d_model: int = 896, byte_dim: int = 32, k: int = 8):
        super().__init__()
        self.byte_embed = nn.Embedding(256, byte_dim)
        self.net = nn.Sequential(
            nn.Linear(k * byte_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        b, s, k = byte_ids.shape
        emb = self.byte_embed(byte_ids.long()).view(b, s, k * self.byte_embed.embedding_dim)
        return self.net(emb)


# SDPA в сгенерированных блоках. ИЗМЕРЕНО на RX 6600 (ROCm, fp16):
# 5.0k tok/s против 5.2k ручного внимания — медленнее, бэкенд mem-eff/math.
# По умолчанию ВЫКЛ. Отложено до другой карты: NEUROSET_SDPA=1
_USE_SDPA = (os.environ.get("NEUROSET_SDPA", "0") == "1"
             and hasattr(F, "scaled_dot_product_attention"))


class CanonicalGeneratedBlock(nn.Module):
    """Один «сгенерированный» слой: веса приходят из фабрики, структура — как у Qwen2."""

    def __init__(self, d_model: int = 896, n_heads: int = 14, d_head: int = 64, gen_mlp: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.gen_mlp = gen_mlp
        self.scale = 1.0 / math.sqrt(d_head)
        self.norm1 = RMSNorm(d_model, eps=1e-6)
        self.norm2 = RMSNorm(d_model, eps=1e-6)

    def forward(self, h, attn_weights, mlp_weights, pos_emb, causal_mask=None):
        b, s, d = h.shape
        h_norm1 = self.norm1(h)
        h_heads = h_norm1.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        w_q, w_k, w_v, w_o = attn_weights
        q = torch.matmul(h_heads, w_q)
        k = torch.matmul(h_heads, w_k)
        v = torch.matmul(h_heads, w_v)

        cos, sin = pos_emb
        q, k = apply_rope(q, k, cos, sin)

        if _USE_SDPA:
            # без материализации scores [b,h,s,s]: ~59МБ трафика на глубину экономим
            ctx = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True, scale=self.scale)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if causal_mask is not None:
                scores = scores + causal_mask
            probs = F.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v)
        attn_heads = torch.matmul(ctx, w_o)
        attn_out = attn_heads.transpose(1, 2).contiguous().view(b, s, d)

        h_mid = h + attn_out
        h_norm2 = self.norm2(h_mid)
        h_mid_heads = h_norm2.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        w_in, w_out = mlp_weights
        gate = F.silu(torch.matmul(h_mid_heads, w_in))
        mlp_heads = torch.matmul(gate, w_out)
        mlp_out = mlp_heads.transpose(1, 2).contiguous().view(b, s, d)
        return attn_out + mlp_out


class AttentivePooler(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, h, mask=None):
        scores = torch.matmul(h, self.query) / math.sqrt(h.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (h * weights).sum(dim=1)


class CanonicalHeadwiseFactory(nn.Module):
    """Генерирует per-head веса attention/MLP по контексту и глубине."""

    def __init__(self, d_model: int = 896, n_heads: int = 14, d_head: int = 64,
                 max_depth: int = 8, gen_mlp: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.gen_mlp = gen_mlp

        self.pooler = AttentivePooler(d_model)
        self.proj_in = nn.Linear(d_model * 2, d_model)
        self.depth_emb = nn.Embedding(max_depth, d_model)
        self.gru = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        dh2 = d_head * d_head
        self.attn_gen = nn.Sequential(
            nn.Linear(d_head, 128),
            nn.GELU(),
            nn.Linear(128, 4 * dh2),
        )
        self.mlp_gen = nn.Sequential(
            nn.Linear(d_head, 128),
            nn.GELU(),
            nn.Linear(128, 2 * d_head * gen_mlp),
        )
        with torch.no_grad():
            # W_o и W_out стартуют с нуля — слой изначально «ничего не добавляет»
            self.attn_gen[-1].weight[3 * dh2:4 * dh2].zero_()
            self.attn_gen[-1].bias[3 * dh2:4 * dh2].zero_()
            self.mlp_gen[-1].weight[d_head * gen_mlp:2 * d_head * gen_mlp].zero_()
            self.mlp_gen[-1].bias[d_head * gen_mlp:2 * d_head * gen_mlp].zero_()

    def forward(self, h, g_prev, depth_idx, mask=None, anchor_lens=None):
        b = h.size(0)
        # ANCHORED-FIX: с якорем ctx/frontier считаются ТОЛЬКО по промпт-части
        # (позиции < anchor_lens): веса слоёв перестают видеть ответ, режим
        # train совпадает с генерацией. None = прежнее поведение (1:1 model_engine.py).
        if anchor_lens is not None:
            pos = torch.arange(h.size(1), device=h.device)
            base_mask = mask if mask is not None else torch.ones(
                (b, h.size(1)), dtype=torch.bool, device=h.device)
            am = (pos[None, :] < anchor_lens.reshape(-1, 1)) & base_mask
            ctx = self.pooler(h.float(), am)
            rows = torch.arange(b, device=h.device)
            frontier = h[rows, (anchor_lens.reshape(-1) - 1).clamp(min=0)].float()
        else:
            ctx = self.pooler(h.float(), mask)
            # 1:1 с model_engine.py: frontier — последняя позиция батча (с паддингом)
            frontier = h[:, -1, :].float()
        fused = self.proj_in(torch.cat([ctx, frontier], dim=-1))
        d_vec = self.depth_emb(torch.full((b,), depth_idx, dtype=torch.long, device=h.device))
        g_next = self.gru(fused + d_vec, g_prev.float())
        gn = self.norm(g_next)
        g_heads = gn.view(b, self.n_heads, self.d_head)

        raw_attn = self.attn_gen(g_heads)
        dh2 = self.d_head * self.d_head
        w_q = raw_attn[:, :, 0:dh2].view(b, self.n_heads, self.d_head, self.d_head)
        w_k = raw_attn[:, :, dh2:2 * dh2].view(b, self.n_heads, self.d_head, self.d_head)
        w_v = raw_attn[:, :, 2 * dh2:3 * dh2].view(b, self.n_heads, self.d_head, self.d_head)
        w_o = raw_attn[:, :, 3 * dh2:4 * dh2].view(b, self.n_heads, self.d_head, self.d_head)

        raw_mlp = self.mlp_gen(g_heads)
        w_in = raw_mlp[:, :, 0:self.d_head * self.gen_mlp].view(
            b, self.n_heads, self.d_head, self.gen_mlp)
        w_out = raw_mlp[:, :, self.d_head * self.gen_mlp:].view(
            b, self.n_heads, self.gen_mlp, self.d_head)
        return g_next, (w_q, w_k, w_v, w_o), (w_in, w_out)


class CompactAsymmetricLLM(nn.Module):
    """Архитектура из model_engine.py: 1..N базовых слоёв Qwen2 + depth генераций."""

    def __init__(
        self,
        base_layers: Sequence[nn.Module],
        rotary_emb: nn.Module,
        norm: nn.Module,
        compact_weights: torch.Tensor,
        token_bytes: torch.Tensor,
        depth: int = 8,
        n_heads: int = 14,
        d_head: int = 64,
        gen_mlp: int = 128,
        pad_id: int = 0,
        grad_checkpoint: bool = False,
        factory_prefix: int = 0,
    ):
        super().__init__()
        assert len(base_layers) >= 1
        assert n_heads * d_head == compact_weights.shape[1], "n_heads*d_head != d_model"

        self.depth = depth
        self.n_heads = n_heads
        self.d_head = d_head
        self.gen_mlp = gen_mlp
        self.pad_id = pad_id
        self.grad_checkpoint = grad_checkpoint
        self.factory_prefix = factory_prefix

        self.single_base = len(base_layers) == 1
        if self.single_base:
            self.base_layer = base_layers[0]
        else:
            self.base_layers = nn.ModuleList(list(base_layers))
        self.rotary_emb = rotary_emb
        self.norm = norm

        v_compact, d_model = compact_weights.shape
        self.V_compact = v_compact
        self.D_model = d_model
        self.embed_tokens = nn.Embedding(v_compact, d_model)
        self.embed_tokens.weight.data.copy_(compact_weights.float())

        self.register_buffer("token_bytes", token_bytes.to(torch.uint8), persistent=True)
        self.morph_patcher = SafeByteMorphPatcher(d_model=d_model)

        self.lm_head = nn.Linear(d_model, v_compact, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # WEIGHT-TIED

        for p in self.norm.parameters():
            p.requires_grad_(False)
        for p in self.embed_tokens.parameters():
            p.requires_grad_(True)
        for p in self.base_layer.parameters() if self.single_base else self.base_layers.parameters():
            p.requires_grad_(True)

        self.gen_layer = CanonicalGeneratedBlock(d_model=d_model, n_heads=n_heads,
                                                 d_head=d_head, gen_mlp=gen_mlp)
        self.factory = CanonicalHeadwiseFactory(d_model=d_model, n_heads=n_heads,
                                                d_head=d_head, max_depth=depth, gen_mlp=gen_mlp)
        self.g_init = nn.Parameter(0.02 * torch.randn(1, d_model))
        self.depth_scale = nn.Parameter(torch.zeros(depth))

    @staticmethod
    def _layer_forward(layer, h, pos_ids, pos_emb, causal_mask):
        # transformers 4.44 не знает position_embeddings — там слой считает RoPE сам
        try:
            res = layer(h, position_ids=pos_ids, position_embeddings=pos_emb,
                        attention_mask=causal_mask)
        except TypeError:
            res = layer(h, position_ids=pos_ids, attention_mask=causal_mask)
        return res[0] if isinstance(res, tuple) else res

    def _base_forward(self, h, pos_ids, pos_emb, causal_mask):
        if self.single_base:
            return self._layer_forward(self.base_layer, h, pos_ids, pos_emb, causal_mask)
        out = h
        for layer in self.base_layers:
            out = self._layer_forward(layer, out, pos_ids, pos_emb, causal_mask)
        return out

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False,
                anchor_lens: Optional[torch.Tensor] = None):
        """Логиты; для distillation-пайплайна опционально возвращает final hidden.

        При обычном train/chat `return_hidden=False`, поэтому устройство и
        поведение исходной модели не меняются.

        ANCHORED-FIX: anchor_lens (опционально, [B]) — длины промпт-части;
        фабрика генерирует веса слоёв только из каузально доступного контекста,
        утечка будущего через пул/фронтир устранена.
        """
        b, s = input_ids.shape
        device = input_ids.device

        safe_ids = input_ids.clamp(0, self.V_compact - 1)
        base_emb = self.embed_tokens(safe_ids)
        b_ids = self.token_bytes[safe_ids]
        morph_emb = self.morph_patcher(b_ids)
        h = (base_emb + 0.1 * morph_emb).float()

        pos_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        try:
            pos_emb = self.rotary_emb(h, pos_ids)
        except TypeError:  # старые transformers
            pos_emb = self.rotary_emb(h, seq_len=s)
        causal_mask = torch.triu(
            torch.full((s, s), -1e4, device=device), diagonal=1
        )[None, None, :, :]

        tok_mask = input_ids != self.pad_id
        g_t = self.g_init.expand(b, -1).contiguous()
        h_prev = h

        # ANCHORED-FIX: при factory_prefix>0 фабрика в train видит только первые
        # prefix позиций — обрезаем якорь тем же префиксом, чтобы train и
        # генерация оставались в одном режиме (тогда досветка не требуется).
        if anchor_lens is not None and self.factory_prefix:
            anchor_lens = anchor_lens.clamp(max=self.factory_prefix)

        for l in range(self.depth):
            prefix = min(self.factory_prefix, s) if self.factory_prefix else s
            g_t, attn_w, mlp_w = self.factory(h[:, :prefix], g_t, l,
                                               mask=tok_mask[:, :prefix],
                                               anchor_lens=anchor_lens)
            if self.grad_checkpoint and self.training:
                base_step = _grad_ckpt(self._base_forward, h, pos_ids, pos_emb, causal_mask,
                                       use_reentrant=False)
                gen_step = _grad_ckpt(self.gen_layer, h, attn_w, mlp_w, pos_emb, causal_mask,
                                      use_reentrant=False)
            else:
                base_step = self._base_forward(h, pos_ids, pos_emb, causal_mask)
                gen_step = self.gen_layer(h, attn_w, mlp_w, pos_emb, causal_mask=causal_mask)
            h_combined = base_step + gen_step
            scale = torch.sigmoid(self.depth_scale[l])
            h = h_prev + scale * (h_combined - h_prev)
            h_prev = h

        h_final = self.norm(h.to(self.norm.weight.dtype))
        logits = self.lm_head(h_final)
        return (logits, h_final) if return_hidden else logits

    # ------------------------------------------------------------------
    # KV-кэш: инкрементальная генерация. В рамках одного сегмента якорь
    # фабрики постоянен -> сгенерированные веса постоянны -> K/V всех слоёв
    # валидны, каждый новый токен стоит O(1) вместо O(s) пересчёта.
    # Эквивалентность проверяется тестом (логиты совпадают с полным форвардом).
    # ------------------------------------------------------------------

    def _gen_attn(self, h, attn_weights, mlp_weights, pos_emb, causal_mask=None,
                  kv_store=None):
        """1:1 с CanonicalGeneratedBlock.forward + опциональный K/V-кэш."""
        b, s, d = h.shape
        gl = self.gen_layer
        h_norm1 = gl.norm1(h)
        h_heads = h_norm1.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        w_q, w_k, w_v, w_o = attn_weights
        q = torch.matmul(h_heads, w_q)
        k = torch.matmul(h_heads, w_k)
        v = torch.matmul(h_heads, w_v)

        cos, sin = pos_emb
        q, k = apply_rope(q, k, cos, sin)

        if kv_store is not None:
            k_prev, v_prev = kv_store
            if k_prev is not None:
                k = torch.cat([k_prev, k], dim=2)
                v = torch.cat([v_prev, v], dim=2)
            kv_store[0], kv_store[1] = k, v
            scores = torch.matmul(q, k.transpose(-2, -1)) * gl.scale
            probs = F.softmax(scores, dim=-1)      # q позади всех ключей — маска не нужна
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * gl.scale
            if causal_mask is not None:
                scores = scores + causal_mask
            probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v)

        attn_heads = torch.matmul(ctx, w_o)
        attn_out = attn_heads.transpose(1, 2).contiguous().view(b, s, d)

        h_mid = h + attn_out
        h_norm2 = gl.norm2(h_mid)
        h_mid_heads = h_norm2.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

        w_in, w_out = mlp_weights
        gate = F.silu(torch.matmul(h_mid_heads, w_in))
        mlp_heads = torch.matmul(gate, w_out)
        mlp_out = mlp_heads.transpose(1, 2).contiguous().view(b, s, d)
        return attn_out + mlp_out

    def _pos_emb(self, h, pos_ids):
        try:
            return self.rotary_emb(h, pos_ids)
        except TypeError:
            return self.rotary_emb(h, seq_len=int(pos_ids.max().item()) + 1)

    @torch.no_grad()
    def prefill_cached(self, input_ids: torch.Tensor, anchor_lens: torch.Tensor):
        """Полный прогон промпта с построением K/V-кэшей и весов фабрики."""
        from transformers.cache_utils import DynamicCache
        b, s = input_ids.shape
        device = input_ids.device
        base_layers = [self.base_layer] if self.single_base else list(self.base_layers)
        n_base = len(base_layers)

        safe_ids = input_ids.clamp(0, self.V_compact - 1)
        base_emb = self.embed_tokens(safe_ids)
        b_ids = self.token_bytes[safe_ids]
        h = (base_emb + 0.1 * self.morph_patcher(b_ids)).float()

        pos_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        pos_emb = self._pos_emb(h, pos_ids)
        causal_mask = torch.triu(
            torch.full((s, s), -1e4, device=device), diagonal=1
        )[None, None, :, :]

        base_caches = [[DynamicCache() for _ in range(n_base)] for _ in range(self.depth)]
        tok_mask = input_ids != self.pad_id
        g_t = self.g_init.expand(b, -1).contiguous()
        h_prev = h
        gen_kvs: List[List[Optional[torch.Tensor]]] = [[None, None] for _ in range(self.depth)]
        weights_per_depth = []

        for l in range(self.depth):
            g_t, attn_w, mlp_w = self.factory(h, g_t, l, mask=tok_mask,
                                              anchor_lens=anchor_lens)
            base_step = h
            for bi, layer in enumerate(base_layers):
                out = layer(base_step, attention_mask=causal_mask, position_ids=pos_ids,
                            past_key_values=base_caches[l][bi], use_cache=True,
                            cache_position=pos_ids, position_embeddings=pos_emb)
                base_step = out[0] if isinstance(out, tuple) else out
            gen_step = self._gen_attn(h, attn_w, mlp_w, pos_emb, causal_mask,
                                      kv_store=gen_kvs[l])
            h_combined = base_step + gen_step
            scale = torch.sigmoid(self.depth_scale[l])
            h = h_prev + scale * (h_combined - h_prev)
            h_prev = h
            weights_per_depth.append((attn_w, mlp_w))

        h_final = self.norm(h.to(self.norm.weight.dtype))
        logits = self.lm_head(h_final)
        cache = {"base": base_caches, "gen_kv": gen_kvs,
                 "weights": weights_per_depth, "len": s}
        return logits[:, -1, :].float(), cache

    @torch.no_grad()
    def step_cached(self, next_ids: torch.Tensor, cache: Dict[str, Any]):
        """Один новый токен по кэшу: базовый слой через HF-cache, gen-слой вручную."""
        b = next_ids.shape[0]
        s_prev = int(cache["len"])
        device = next_ids.device
        base_layers = [self.base_layer] if self.single_base else list(self.base_layers)

        safe_ids = next_ids.clamp(0, self.V_compact - 1)
        h = (self.embed_tokens(safe_ids)
             + 0.1 * self.morph_patcher(self.token_bytes[safe_ids])).float()

        pos = torch.full((b, 1), s_prev, dtype=torch.long, device=device)
        pos_emb = self._pos_emb(h, pos)
        h_prev = h

        for l in range(self.depth):
            attn_w, mlp_w = cache["weights"][l]
            base_step = h
            for bi, layer in enumerate(base_layers):
                out = layer(base_step, attention_mask=None, position_ids=pos,
                            past_key_values=cache["base"][l][bi], use_cache=True,
                            cache_position=pos, position_embeddings=pos_emb)
                base_step = out[0] if isinstance(out, tuple) else out
            gen_step = self._gen_attn(h, attn_w, mlp_w, pos_emb, None,
                                      kv_store=cache["gen_kv"][l])
            h_combined = base_step + gen_step
            scale = torch.sigmoid(self.depth_scale[l])
            h = h_prev + scale * (h_combined - h_prev)
            h_prev = h

        h_final = self.norm(h.to(self.norm.weight.dtype))
        logits = self.lm_head(h_final)
        cache["len"] = s_prev + 1
        return logits[:, -1, :].float()


def human_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}k"
    return str(n)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    groups: Dict[str, int] = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    groups["TOTAL"] = sum(p.numel() for p in model.parameters())
    return groups


# ======================================================================================
# Токенизатор, словарь и compact-веса
# ======================================================================================

def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2/byte-level BPE: byte -> unicode-символ."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def token_metadata(tok) -> Tuple[int, List[str], List[bytes]]:
    """Для каждого id: decoded-строка (для prefix-fallback) и сырые байты (для byte-патчера)."""
    vocab = tok.get_vocab()
    n = max(vocab.values()) + 1
    surfaces = [""] * n
    raw_bytes: List[bytes] = [b""] * n
    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}
    for token, idx in vocab.items():
        raw = bytearray()
        for ch in token:
            b = u2b.get(ch)
            raw.append(b if b is not None else (ord(ch) if ord(ch) < 256 else 63))
        rb = bytes(raw)
        raw_bytes[idx] = rb
        try:
            surfaces[idx] = rb.decode("utf-8")
        except Exception:
            surfaces[idx] = ""  # обрывок UTF-8 — для prefix-матчинга не годится
    return n, surfaces, raw_bytes


def token_bytes_matrix(raw_bytes: Sequence[bytes], kept: Sequence[int]) -> torch.Tensor:
    arr = np.zeros((len(kept), 8), dtype=np.uint8)
    for i, old in enumerate(kept):
        rb = raw_bytes[old][:8]
        arr[i, :len(rb)] = np.frombuffer(rb, dtype=np.uint8)
    return torch.from_numpy(arr)


def get_forward_map(pack: Dict[str, Any], n_old: int = 0) -> np.ndarray:
    """old_id -> new_id как numpy-массив. Поддерживает dict (старый формат) и tensor."""
    x = pack.get("old_to_new")
    if isinstance(x, dict):
        n = max(n_old, (max(x.keys()) + 1) if x else 0)
        arr = np.zeros(n, dtype=np.int64)
        for k, v in x.items():
            k = int(k)
            if 0 <= k < n:
                arr[k] = int(v)
        return arr
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.int64)
    if n_old and len(arr) < n_old:
        arr = np.pad(arr, (0, n_old - len(arr)), constant_values=0)
    return arr


def get_inverse_map(pack: Dict[str, Any], v_size: int = 0) -> np.ndarray:
    """new_id -> old_id как numpy-массив. Поддерживает dict и tensor."""
    x = pack.get("new_to_old")
    if isinstance(x, dict):
        n = max(v_size, (max(x.keys()) + 1) if x else 0)
        arr = np.zeros(n, dtype=np.int64)
        for k, v in x.items():
            k = int(k)
            if 0 <= k < n:
                arr[k] = int(v)
        return arr
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.int64)


def select_vocab(freq: np.ndarray, surfaces: Sequence[str], tok, cfg: Config,
                 log) -> Tuple[List[int], float]:
    n = len(freq)
    pad_old = tok.convert_tokens_to_ids("<|endoftext|>")
    if pad_old is None or pad_old < 0:
        pad_old = 0
    specials = []
    for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i >= 0:
            specials.append(i)

    if cfg.vocab_strategy == "full" or cfg.vocab_size >= n:
        # pad всегда первым: id 0 = <|endoftext|>, даже в режиме без сжатия
        kept = [pad_old] + [i for i in range(n) if i != pad_old]
        total = freq.sum()
        cov = float(freq[kept].sum() / total) if total else 0.0
        return kept, cov

    size = max(64, min(cfg.vocab_size, n))
    kept: List[int] = [pad_old]
    kept_set = {pad_old}
    for s in specials:
        if s not in kept_set:
            kept.append(s)
            kept_set.add(s)

    # Не полагаемся на частоты tool-разметки: в обычном SFT-корпусе эти BPE
    # токены редки, но они обязаны быть в compact vocabulary, иначе fallback
    # меняет `arguments` на `arg` и ломает JSON на инференсе.
    protocol_ids: List[int] = []
    for text in PROTOCOL_VOCAB_TEXTS:
        try:
            protocol_ids.extend(int(i) for i in tok.encode(text, add_special_tokens=False))
        except Exception:
            continue
    for i in protocol_ids:
        if 0 <= i < n and i not in kept_set and len(kept) < size:
            kept.append(i)
            kept_set.add(i)

    order = np.argsort(-freq, kind="stable")
    total = float(freq.sum()) or 1.0

    if cfg.vocab_strategy == "stratified":
        guarantee = set()
        for i, s in enumerate(surfaces):
            if not s or len(s) > 2:
                continue
            ch = s[-1] if s.startswith(" ") else s
            if len(ch) == 1 and (ch.isdigit() or ch.isalpha() or ch in " .,!?;:-—()\"'«»"):
                guarantee.add(i)
        for i in sorted(guarantee, key=lambda x: -freq[x]):
            if len(kept) >= size:
                break
            if i not in kept_set:
                kept.append(i)
                kept_set.add(i)

    if cfg.vocab_strategy == "coverage":
        cum = 0.0
        for i in order:
            if len(kept) >= size:
                break
            if i in kept_set:
                continue
            cum += float(freq[i])
            kept.append(int(i))
            kept_set.add(int(i))
            if cum / total >= cfg.coverage:
                break
        if len(kept) < size:
            for i in order:
                if len(kept) >= size:
                    break
                if i not in kept_set:
                    kept.append(int(i))
                    kept_set.add(int(i))
    else:  # freq / добор после stratified
        for i in order:
            if len(kept) >= size:
                break
            if i in kept_set:
                continue
            kept.append(int(i))
            kept_set.add(int(i))

    kept = kept[:size]
    cov = float(freq[kept].sum() / total)
    log(f"  vocab: strategy={cfg.vocab_strategy}, size={len(kept)}, "
        f"покрытие токенов={cov*100:.2f}%")
    return kept, cov


def build_vocab_pack(
    tok,
    embeddings: Optional[torch.Tensor],
    freq: np.ndarray,
    cfg: Config,
    log=print,
) -> Dict[str, Any]:
    """Создаёт compact_vocab.pt-совместимый пакет."""
    t0 = time.time()
    n_tok, surfaces, raw_bytes = token_metadata(tok)
    d_model = embeddings.shape[1] if embeddings is not None else cfg.d_model

    freq = freq[:n_tok].astype(np.float64)
    kept, coverage = select_vocab(freq, surfaces, tok, cfg, log)
    v = len(kept)

    kept_map = np.zeros(n_tok, dtype=np.int32)
    surface_to_new: Dict[str, int] = {}
    for new_id, old in enumerate(kept):
        kept_map[old] = new_id
        s = surfaces[old]
        if s and old != kept[0]:
            surface_to_new.setdefault(s, new_id)

    # fallback: редкие (выкинутые) токены -> самый длинный префикс из словаря
    if cfg.vocab_strategy != "full":
        for old in range(n_tok):
            if kept_map[old] != 0 or old == kept[0]:
                continue
            s = surfaces[old]
            mapped = 0
            if s:
                for ln in range(min(len(s) - 1, 24), 0, -1):
                    nid = surface_to_new.get(s[:ln])
                    if nid is not None:
                        mapped = nid
                        break
            kept_map[old] = mapped

    old_to_new_t = torch.from_numpy(kept_map.astype(np.int64))
    new_to_old_t = torch.tensor(kept, dtype=torch.long)
    # dict-версии — совместимость с исходным test_unified_agent.py (.get)
    old_to_new = {i: int(v) for i, v in enumerate(kept_map)}
    new_to_old = {i: int(v) for i, v in enumerate(kept)}

    # веса
    if embeddings is None:
        gen = torch.Generator().manual_seed(cfg.seed)
        compact = torch.randn(v, d_model, generator=gen) * 0.02
    else:
        emb = embeddings[:n_tok].float().cpu()
        compact = emb[torch.tensor(kept, dtype=torch.long)].clone()
        if cfg.merge_weights == "mean" and cfg.vocab_strategy != "full":
            # «сворачиваем» выкинутые токены в их префиксные цели с весом частоты
            sums = np.zeros((v, d_model), dtype=np.float32)
            wsum = np.zeros(v, dtype=np.float64)
            freq_np = freq
            valid_old = np.nonzero(freq_np > 0)[0]
            emb_np = emb.numpy()
            for start in range(0, len(valid_old), 4096):
                chunk = valid_old[start:start + 4096]
                targets = kept_map[chunk].astype(np.int64)
                mask = targets > 0  # не трогаем pad-строку
                if not mask.any():
                    continue
                chunk = chunk[mask]
                targets = targets[mask]
                w = freq_np[chunk].astype(np.float32)
                np.add.at(sums, targets, emb_np[chunk] * w[:, None])
                np.add.at(wsum, targets, w)
            for i in range(1, v):
                if wsum[i] > 0:
                    compact[i] = torch.from_numpy(sums[i] / wsum[i])
            log(f"  веса: свернул редкие токены в префиксы (mean, {len(valid_old)} ненулевых)")

    token_bytes = token_bytes_matrix(raw_bytes, kept)

    pack = {
        "compact_weights": compact.float().contiguous(),
        "token_bytes": token_bytes,
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
        "old_to_new_tensor": old_to_new_t,
        "new_to_old_tensor": new_to_old_t,
        "kept_old_ids": torch.tensor(kept, dtype=torch.long),
        "meta": {
            "strategy": cfg.vocab_strategy,
            "vocab_size": v,
            "coverage": coverage,
            "tokenizer_len": n_tok,
            "d_model": d_model,
            "n_heads": cfg.n_heads,
            "d_head": cfg.d_head,
            "merge_weights": cfg.merge_weights,
            "pad_old_id": int(kept[0]),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    log(f"  vocab pack готов за {time.time()-t0:.1f} c: {v} токенов x {d_model}")
    return pack


def has_tokenizer_files(path: Path) -> bool:
    return any((path / f).exists() for f in (
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model"))


def resolve_tokenizer_source(cfg: Config, log=print) -> str:
    if cfg.tokenizer_name:
        return cfg.tokenizer_name
    if cfg.model_name and cfg.model_name != "auto":
        p = Path(cfg.model_name)
        if p.is_dir() and has_tokenizer_files(p):
            return str(p)
        if not p.exists():
            return cfg.model_name
    for pat in ("data_raw/qwen*", "models/qwen*", "qwen*", "../data_raw/qwen*"):
        for p in sorted(Path(".").glob(pat)):
            if p.is_dir() and has_tokenizer_files(p):
                log(f"[i] Локальный токенизатор: {p}")
                return str(p)
    return DEFAULT_QWEN


def resolve_model_source(cfg: Config, log=print) -> str:
    if cfg.model_name and cfg.model_name != "auto":
        return cfg.model_name
    for pat in ("data_raw/qwen*", "models/qwen*", "qwen*", "../data_raw/qwen*"):
        for p in sorted(Path(".").glob(pat)):
            if p.is_dir() and (p / "config.json").exists() and any(
                    (p / f).exists() for f in ("model.safetensors", "pytorch_model.bin")):
                log(f"[i] Локальная модель Qwen: {p}")
                return str(p)
    return DEFAULT_QWEN


def load_tokenizer(cfg: Config, log=print):
    src = resolve_tokenizer_source(cfg, log)
    log(f"[i] Токенизатор: {src}")
    tok = AutoTokenizer.from_pretrained(src, use_fast=True)
    return tok


def _config_rope_theta(config) -> float:
    """rope_theta из конфига Qwen: transformers 5 хранит его в rope_parameters."""
    rp = getattr(config, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta"):
        return float(rp["rope_theta"])
    v = getattr(config, "rope_theta", None)
    if v:
        return float(v)
    return 1e6


def load_embeddings_and_modules(cfg: Config, log=print):
    """Загружает Qwen: матрицу эмбеддингов, базовые слои, rotary и norm."""
    src = resolve_model_source(cfg, log)
    log(f"[i] Загружаю Qwen из {src} (это может занять минуту)...")
    try:
        qwen = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    except TypeError:
        qwen = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
    qwen.eval()
    try:
        qwen.config._attn_implementation = "eager"
    except Exception:
        pass

    n_layers = len(qwen.model.layers)
    if cfg.base_layer < 0 or cfg.base_layer + cfg.base_layers > n_layers:
        raise ValueError(f"base_layer={cfg.base_layer} + base_layers={cfg.base_layers} "
                         f"не влезает в Qwen ({n_layers} слоёв)")

    layers = []
    for i in range(cfg.base_layer, cfg.base_layer + cfg.base_layers):
        layer = copy.deepcopy(qwen.model.layers[i]).float()
        _force_eager(layer)
        layers.append(layer)
    rotary = copy.deepcopy(qwen.model.rotary_emb)
    norm = copy.deepcopy(qwen.model.norm).float()

    emb = qwen.get_input_embeddings().weight.detach().float().cpu()
    cfg.d_model = emb.shape[1]
    cfg.n_heads = qwen.config.num_attention_heads
    cfg.d_head = getattr(qwen.config, "head_dim", None) or (emb.shape[1] // cfg.n_heads)
    cfg.kv_heads = getattr(qwen.config, "num_key_value_heads", cfg.n_heads)
    cfg.intermediate_size = qwen.config.intermediate_size
    cfg.attention_bias = bool(getattr(qwen.config, "attention_bias", True))
    cfg.rope_theta = _config_rope_theta(qwen.config)
    cfg.rms_eps = float(getattr(qwen.config, "rms_norm_eps", None) or 1e-6)
    log(f"[i] Qwen: d_model={cfg.d_model}, heads={cfg.n_heads}, kv_heads={cfg.kv_heads}, "
        f"d_head={cfg.d_head}, layers={n_layers}; беру слои {cfg.base_layer}..{cfg.base_layer+cfg.base_layers-1}")

    del qwen
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return emb, layers, rotary, norm


def _force_eager(module: nn.Module):
    for m in module.modules():
        c = getattr(m, "config", None)
        if c is not None:
            try:
                c._attn_implementation = "eager"
            except Exception:
                pass
    return module


def make_scratch_modules(cfg: Config):
    d = cfg.d_model
    inter = cfg.intermediate_size or d * 4
    qcfg = Qwen2Config(
        vocab_size=max(cfg.vocab_size, 64),
        hidden_size=d,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.kv_heads,
        head_dim=cfg.d_head,
        intermediate_size=inter,
        num_hidden_layers=cfg.base_layers,
        max_position_embeddings=max(512, cfg.max_seq_len + 8),
        rms_norm_eps=cfg.rms_eps,
        rope_theta=cfg.rope_theta,
        attention_bias=cfg.attention_bias,
        hidden_act="silu",
        tie_word_embeddings=True,
    )
    try:
        qcfg._attn_implementation = "eager"
    except Exception:
        pass
    layers = [_force_eager(Qwen2DecoderLayer(qcfg, i)) for i in range(cfg.base_layers)]
    rotary = Qwen2RotaryEmbedding(qcfg)
    norm = Qwen2RMSNorm(d, cfg.rms_eps)
    return layers, rotary, norm


# ======================================================================================
# Данные: рендер ChatML, маски, кэш токенов
# ======================================================================================

_TOOL_RESP_RE = re.compile(r"<tool_response>.*?</tool_response>", re.S)


def render_chat(messages: Sequence[Dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
    """Возвращает текст в формате Qwen ChatML и char-спаны, которые надо учить.

    Ответы инструментов (<tool_response>...</tool_response>) не учим: их на инференсе
    подставляет рантайм, а не модель.
    """
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    pos = 0
    for m in messages:
        role = str(m.get("role", "user"))
        if role not in ("system", "user", "assistant"):
            role = "user"
        content = str(m.get("content", ""))
        content = content.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
        pre = f"<|im_start|>{role}\n"
        parts.append(pre)
        pos += len(pre)
        start = pos
        parts.append(content)
        pos += len(content)
        end = pos
        post = "<|im_end|>\n"
        parts.append(post)
        pos += len(post)
        if role == "assistant":
            if "<tool_response>" in content:
                cursor = 0
                for tm in _TOOL_RESP_RE.finditer(content):
                    a, b = start + cursor, start + tm.start()
                    if b > a:
                        spans.append((a, b))
                    cursor = tm.end()
                a = start + cursor
                if end > a:
                    spans.append((a, end))
            else:
                spans.append((start, end))
    return "".join(parts), spans


_CHATML_RE = re.compile(r"<\|im_start\|>([A-Za-z_]+)\n(.*?)<\|im_end\|>", re.S)


def chatml_to_messages(text: str) -> List[Dict[str, str]]:
    """Разбирает готовый ChatML (старый persona_corpus.jsonl) в messages."""
    out = []
    for role, content in _CHATML_RE.findall(text):
        if role not in ("system", "user", "assistant"):
            role = "assistant" if role in ("gpt", "bot") else "user"
        out.append({"role": role, "content": content})
    return out


def record_to_messages(rec: Dict[str, Any]) -> Tuple[List[Dict[str, str]], bool]:
    """Приводит строку JSONL к messages. Возвращает (messages, is_plain_text)."""
    if isinstance(rec.get("messages"), list):
        return rec["messages"], bool(rec.get("format") == "text")
    for key in ("conversations", "conversation"):
        val = rec.get(key)
        msgs = []
        if isinstance(val, list):
            for m in val:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from") or "user"
                role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(str(role), str(role))
                content = m.get("content") or m.get("value") or ""
                if role in ("system", "user", "assistant") and str(content).strip():
                    msgs.append({"role": role, "content": str(content)})
        if msgs:
            return msgs, False
    if "instruction" in rec and ("output" in rec or "response" in rec):
        q = str(rec.get("instruction") or "")
        if str(rec.get("input") or "").strip():
            q += "\n\n" + str(rec["input"])
        a = str(rec.get("output") or rec.get("response") or "")
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}], False
    if "prompt" in rec and ("response" in rec or "answer" in rec or "completion" in rec):
        return [{"role": "user", "content": str(rec["prompt"])},
                {"role": "assistant", "content": str(rec.get("response") or rec.get("answer") or rec.get("completion"))}], False
    if "question" in rec and "answer" in rec:
        return [{"role": "user", "content": str(rec["question"])},
                {"role": "assistant", "content": str(rec["answer"])}], False
    text = rec.get("text") or rec.get("content") or ""
    if isinstance(text, str) and text.strip():
        parsed = chatml_to_messages(text)
        if parsed:
            return parsed, False
        return [{"role": "assistant", "content": text}], True
    return [], False


def render_record(rec: Dict[str, Any]) -> Tuple[str, List[Tuple[int, int]]]:
    messages, plain = record_to_messages(rec)
    if not messages:
        return "", []
    if plain:
        joined = "".join(str(m.get("content", "")) for m in messages)
        parsed = chatml_to_messages(joined)
        if parsed:
            return render_chat(parsed)
    if plain or (rec.get("format") == "text" and not _CHATML_RE.search("".join(
            str(m.get("content", "")) for m in messages))):
        text = str(messages[0].get("content", "")).strip()
        if not text:
            return "", []
        # сырой текст тоже оборачиваем в ChatML: модель не должна видеть «голый» текст
        pre = "<|im_start|>assistant\n"
        post = "<|im_end|>\n"
        return pre + text + post, [(len(pre), len(pre) + len(text))]
    return render_chat(messages)


def _split_tool_pieces(messages: Sequence[Dict[str, str]]
                       ) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Сегментирует tool-трассу на две обучающие строки.

    render_chat уже НЕ учит <tool_response> и даёт отдельный спан для вызова и
    для синтеза. Проблема была в якоре фабрики: он берётся от первого
    размеченного токена, поэтому синтез учился весами, посчитанными только по
    промпту — модель не «читала» полученный факт.

    A: контекст до конца вызова (якорь = промпт, утечки нет).
    B: полный текст, но маска — всё, кроме спана вызова, поэтому первый
       размеченный токен = начало синтеза, и якорь включает tool response.
    """
    for i, m in enumerate(messages):
        if str(m.get("role")) != "assistant":
            continue
        content = str(m.get("content", ""))
        if "<tool_response>" not in content:
            continue
        tm = _TOOL_RESP_RE.search(content)
        if tm is None:
            continue
        call_part = content[:tm.start()].rstrip()
        if not call_part:
            continue
        msgs_a = list(messages[:i]) + [{"role": "assistant", "content": call_part}]
        text_a, spans_a = render_chat(msgs_a)
        text_b, spans_b = render_chat(messages)
        # <tool_response> есть и в системном промпте, поэтому ищем спан, сразу
        # за которым (через пробелы) начинается ответ инструмента.
        drop = None
        for sp in spans_b:
            if text_b[sp[1]:sp[1] + 40].lstrip().startswith("<tool_response>"):
                drop = sp
        spans_b2 = [sp for sp in spans_b if sp != drop]
        if not spans_a or not spans_b2:
            continue
        return [(text_a, spans_a), (text_b, spans_b2)]
    return []


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
            if limit and len(recs) >= limit:
                break
    return recs


def prepare_datasets(cfg: Config, tok, log=print) -> Dict[str, Any]:
    """Читает jsonl, токенизирует (с кэшем), считает частоты, готовит Dataset'ы."""
    # Путь к датасету обязателен и задаётся вручную: никаких сборщиков и фолбэков.
    if not cfg.data_dir and not cfg.train_file:
        raise ValueError(
            "Не указан путь к датасету. Задайте --data-dir ПАПКА (с train.jsonl/val.jsonl) "
            "или --train-file ФАЙЛ."
        )
    data_dir = Path(cfg.data_dir) if cfg.data_dir else Path(cfg.train_file).parent
    train_path = Path(cfg.train_file) if cfg.train_file else data_dir / "train.jsonl"
    val_path = Path(cfg.val_file) if cfg.val_file else data_dir / "val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Нет файла {train_path}. Укажите существующий путь через --data-dir ПАПКА "
            f"или --train-file ФАЙЛ."
        )
    train_recs = load_jsonl(train_path, cfg.max_samples)
    val_recs = load_jsonl(val_path, cfg.val_samples) if val_path.exists() else []

    if not val_recs:
        n_val = max(1, int(len(train_recs) * 0.02))
        val_recs = train_recs[-n_val:]
        train_recs = train_recs[:-n_val]
        log(f"[i] val.jsonl пуст — отрезал {len(val_recs)} сэмплов от train")
    if len(train_recs) < 8:
        raise ValueError(f"Слишком мало данных: {len(train_recs)} сэмплов")

    tokenizer_fp = f"{getattr(tok, 'name_or_path', '?')}:{len(tok)}"
    cache_dir = data_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(split: str, recs: int) -> str:
        h = hashlib.sha1()
        h.update(tokenizer_fp.encode())
        # версия рендера/маски: при изменении кода кэш должен инвалидироваться
        h.update(b"render-v9-toolcall-weight|")
        h.update(f"{split}|{cfg.max_seq_len}|{cfg.mask}|{recs}|{cfg.max_tokens}".encode())
        for p in (train_path, val_path):
            if p.exists():
                st = p.stat()
                h.update(f"{p}:{st.st_mtime_ns}:{st.st_size}".encode())
        return h.hexdigest()[:16]

    def get_split(name: str, recs: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
        key = cache_key(name, len(recs))
        path = cache_dir / f"{name}_{key}.npz"
        if path.exists():
            log(f"[i] Кэш токенов: {path}")
            data = np.load(path)
            out = {k: data[k] for k in data.files}
            return out
        log(f"[!] Токенизирую {name}: {len(recs)} сэмплов "
            f"(лимит {max_tokens/1e6:.1f}M токенов)...")
        out = encode_records(recs, tok, cfg, max_tokens, log)
        np.savez(path, **out)
        log(f"    сохранено в кэш: {path} ({out['ids'].size:,} токенов)")
        return out

    train_data = get_split("train", train_recs, cfg.max_tokens)
    val_data = get_split("val", val_recs, min(2_000_000, cfg.max_tokens or 2_000_000))

    freq = np.bincount(train_data["ids"].astype(np.int64), minlength=len(tok))
    log(f"[i] Токенов в train: {train_data['ids'].size:,}, уникальных: {int((freq>0).sum()):,}")

    train_ds = PackedDataset(train_data["ids"], train_data["mask"], train_data["offsets"],
                             weight=train_data.get("weight"))
    val_ds = PackedDataset(val_data["ids"], val_data["mask"], val_data["offsets"],
                           weight=val_data.get("weight"))
    return {
        "train": train_ds,
        "val": val_ds,
        "freq": freq,
        "train_tokens": int(train_data["ids"].size),
        "val_tokens": int(val_data["ids"].size),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }


def encode_records(records: Sequence[Dict[str, Any]], tok, cfg: Config,
                   max_tokens: int, log=print) -> Dict[str, np.ndarray]:
    w_toolcall = float(getattr(cfg, "toolcall_weight", 1.0) or 1.0)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    limit = max_tokens if max_tokens and max_tokens > 0 else 10 ** 12
    ids_buf: List[np.ndarray] = []
    mask_buf: List[np.ndarray] = []
    weight_buf: List[np.ndarray] = []
    offsets: List[int] = [0]
    total_tokens = 0
    batch_texts: List[str] = []
    batch_spans: List[List[Tuple[int, int]]] = []
    n_done = 0
    n_trunc = 0
    n_tail = 0

    def make_mask(ids, offs, spans) -> np.ndarray:
        starts = [o[0] for o in offs]
        mask = np.zeros(len(ids), dtype=np.uint8)
        if cfg.mask == "none":
            mask[:] = 1
            return mask
        for s, e in spans:
            i0 = bisect.bisect_left(starts, s)
            i1 = bisect.bisect_left(starts, e)
            i0 = min(max(i0, 0), len(ids))
            i1 = min(max(i1, i0), len(ids))
            if i1 < len(ids) and ids[i1] == im_end:
                i1 += 1
            if i1 > i0:
                mask[i0:i1] = 1
        return mask

    def flush_batch():
        nonlocal total_tokens, n_done, n_trunc, n_tail
        if not batch_texts:
            return
        enc = tok(batch_texts, add_special_tokens=False, return_offsets_mapping=True,
                  truncation=False)
        for full_ids, full_offs, spans, raw_text in zip(enc["input_ids"], enc["offset_mapping"],
                                                         batch_spans, batch_texts):
            ids, offs = full_ids, full_offs
            if len(ids) > cfg.max_seq_len:
                n_trunc += 1
                ids = ids[:cfg.max_seq_len]
                offs = offs[:cfg.max_seq_len]
            if len(ids) < 3:
                continue
            mask = make_mask(ids, offs, spans)
            if int(mask.sum()) < 1 and len(full_ids) > cfg.max_seq_len and cfg.mask != "none":
                # хвост длинного сэмпла: ответ в конце важнее начала промпта
                ids2 = full_ids[-cfg.max_seq_len:]
                offs2 = full_offs[-cfg.max_seq_len:]
                mask2 = make_mask(ids2, offs2, spans)
                if int(mask2.sum()) >= 1:
                    ids, offs, mask = ids2, offs2, mask2
                    n_tail += 1
            if int(mask.sum()) < 1:
                continue
            ids_arr = np.asarray(ids, dtype=np.int32)
            ids_buf.append(ids_arr)
            mask_buf.append(mask)
            # вес 3.0 на токенах внутри <tool_call>...</tool_call> (там формируется запрос)
            w = np.ones(len(ids), dtype=np.float32)
            if w_toolcall > 1.0:
                for tm in re.finditer(r"<tool_call>.*?</tool_call>", raw_text, re.S):
                    for ti, (ca, cb) in enumerate(offs):
                        if ca < tm.end() and cb > tm.start():
                            w[ti] = w_toolcall
            weight_buf.append(w)
            offsets.append(offsets[-1] + len(ids_arr))
            total_tokens += len(ids_arr)
            n_done += 1
        batch_texts.clear()
        batch_spans.clear()

    for i, rec in enumerate(records):
        try:
            messages, _plain = record_to_messages(rec)
            text, spans = render_record(rec)
        except Exception:
            continue
        if not text.strip():
            continue
        pieces = None
        if getattr(cfg, "split_tool_segments", False) and messages:
            pieces = _split_tool_pieces(messages)
        if not pieces:
            pieces = [(text, spans)]
        for piece_text, piece_spans in pieces:
            if not piece_text.strip() or not piece_spans:
                continue
            batch_texts.append(piece_text)
            batch_spans.append(piece_spans)
        if len(batch_texts) >= 64:
            flush_batch()
            if i % 2000 == 0:
                log(f"    {i}/{len(records)} сэмплов, {total_tokens:,} токенов")
            if total_tokens >= limit:
                break
    if total_tokens < limit:
        flush_batch()

    if not ids_buf:
        raise ValueError("После токенизации не осталось ни одного сэмпла (проверьте данные/маску).")
    ids_arr = np.concatenate(ids_buf)
    mask_arr = np.concatenate(mask_buf)
    extra = f", обрезано {n_trunc:,}" if n_trunc else ""
    if n_tail:
        extra += f" (из них {n_tail:,} взято хвостом)"
    log(f"    готово: {n_done:,} сэмплов, {ids_arr.size:,} токенов, "
        f"средняя длина {ids_arr.size/max(1,n_done):.0f}{extra}")
    if n_trunc and n_trunc > 0.1 * max(1, n_trunc + n_done):
        log(f"    [!] {n_trunc/(n_trunc+n_done)*100:.0f}% сэмплов длиннее max_seq_len="
            f"{cfg.max_seq_len}: уменьшите длину сэмплов в датасете или увеличьте seq")
    weight_arr = np.concatenate(weight_buf) if weight_buf else np.ones_like(ids_arr, dtype=np.float32)
    return {
        "ids": ids_arr,
        "mask": mask_arr,
        "weight": weight_arr,
        "offsets": np.asarray(offsets, dtype=np.int64),
    }


class PackedDataset(Dataset):
    def __init__(self, ids: np.ndarray, mask: np.ndarray, offsets: np.ndarray,
                 old_to_new: Optional[np.ndarray] = None, weight: Optional[np.ndarray] = None):
        self.ids = ids
        self.mask = mask
        self.offsets = offsets
        self.weight = weight
        self.map = old_to_new  # None = идентичность (кэш всегда в old id)

    def __len__(self):
        return len(self.offsets) - 1

    def set_map(self, mapping: np.ndarray):
        self.map = mapping

    def __getitem__(self, i: int):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        old = self.ids[a:b].astype(np.int64)
        x = self.map[old].astype(np.int64) if self.map is not None else old
        m = self.mask[a:b]
        y = np.where(m > 0, x, -100).astype(np.int64)
        if self.weight is not None:
            w = self.weight[a:b].astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)
        return torch.from_numpy(x), torch.from_numpy(y)


def collate_fn(batch):
    has_w = len(batch[0]) == 3
    max_len = max(b[0].size(0) for b in batch)
    ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    weights = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, b in enumerate(batch):
        x, y = b[0], b[1]
        ids[i, :x.size(0)] = x
        labels[i, :y.size(0)] = y
        if has_w:
            weights[i, :b[2].size(0)] = b[2]
    return (ids, labels, weights) if has_w else (ids, labels)


# ======================================================================================
# Сборка модели
# ======================================================================================

def build_model_and_pack(cfg: Config, tok, freq: np.ndarray, log=print):
    embeddings = None
    layers = rotary = norm = None

    if cfg.vocab_pack and not Path(cfg.vocab_pack).exists():
        raise FileNotFoundError(f"Нет файла vocab pack: {cfg.vocab_pack}")
    if cfg.vocab_pack and Path(cfg.vocab_pack).exists():
        log(f"[i] Загружаю готовый vocab pack: {cfg.vocab_pack}")
        pack = torch.load(cfg.vocab_pack, map_location="cpu", weights_only=False)
        old_map = get_forward_map(pack, len(tok))
        if len(old_map) < len(tok):
            print(f"[!] vocab_pack покрывает {len(old_map)} из {len(tok)} токенов — "
                  f"недостающие будут маппиться в 0")
        pack["old_to_new"] = old_map  # дальше работаем с массивом
        pack["new_to_old"] = get_inverse_map(pack, int(pack["compact_weights"].shape[0]))
        vmeta = pack.get("meta", {}) or {}
        if vmeta.get("n_heads") and vmeta.get("d_head"):
            cfg.n_heads, cfg.d_head = int(vmeta["n_heads"]), int(vmeta["d_head"])
        d_pack = int(pack["compact_weights"].shape[1])
        if cfg.n_heads * cfg.d_head != d_pack and d_pack % max(1, cfg.d_head) == 0:
            cfg.n_heads = d_pack // cfg.d_head
            print(f"[i] vocab_pack: d_model={d_pack}, ставлю n_heads={cfg.n_heads}, d_head={cfg.d_head}")
    else:
        if cfg.init == "qwen":
            embeddings, layers, rotary, norm = load_embeddings_and_modules(cfg, log)
        pack = build_vocab_pack(tok, embeddings, freq, cfg, log)

    if layers is None and cfg.init == "qwen":
        # pack уже готов, но слои нужно достать отдельно (без эмбеддингов)
        emb_tmp, layers, rotary, norm = load_embeddings_and_modules(cfg, log)
        del emb_tmp

    compact = pack["compact_weights"].float()
    d_prev = cfg.d_model
    cfg.d_model = compact.shape[1]
    cfg.vocab_size = compact.shape[0]
    if layers is None and cfg.init == "qwen" and d_prev and d_prev != cfg.d_model:
        raise ValueError(
            f"vocab_pack обучен на d_model={cfg.d_model}, а Qwen даёт d_model={d_prev}. "
            f"Возьмите pack от этой же инициализации или --init scratch с нужными размерами."
        )
    if layers is None:
        layers, rotary, norm = make_scratch_modules(cfg)
    if cfg.n_heads * cfg.d_head != cfg.d_model:
        raise ValueError(f"n_heads({cfg.n_heads}) * d_head({cfg.d_head}) != d_model({cfg.d_model})")

    model = CompactAsymmetricLLM(
        base_layers=layers,
        rotary_emb=rotary,
        norm=norm,
        compact_weights=compact,
        token_bytes=torch.as_tensor(pack["token_bytes"], dtype=torch.uint8),
        depth=cfg.depth,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        gen_mlp=cfg.gen_mlp,
        pad_id=0,
        grad_checkpoint=cfg.grad_checkpoint,
        factory_prefix=cfg.factory_prefix,
    )
    return model, pack


def build_scaffold(cfg: Config, meta: Dict[str, Any],
                   token_bytes: Optional[torch.Tensor] = None) -> CompactAsymmetricLLM:
    """Собирает модель со случайными весами для загрузки чекпойнта (без Qwen)."""
    d = meta["d_model"]
    qcfg = Qwen2Config(
        vocab_size=meta["vocab_size"],
        hidden_size=d,
        num_attention_heads=meta["n_heads"],
        num_key_value_heads=meta["kv_heads"],
        head_dim=meta["d_head"],
        intermediate_size=meta["intermediate_size"],
        num_hidden_layers=meta["base_layers"],
        max_position_embeddings=max(512, meta["max_seq_len"] + 8),
        rms_norm_eps=meta["rms_eps"],
        rope_theta=meta["rope_theta"],
        attention_bias=meta["attention_bias"],
        hidden_act="silu",
        tie_word_embeddings=True,
    )
    try:
        qcfg._attn_implementation = "eager"
    except Exception:
        pass
    layers = [_force_eager(Qwen2DecoderLayer(qcfg, i)) for i in range(meta["base_layers"])]
    rotary = Qwen2RotaryEmbedding(qcfg)
    norm = Qwen2RMSNorm(d, meta["rms_eps"])
    gen = torch.Generator().manual_seed(1234)
    weights = torch.randn(meta["vocab_size"], d, generator=gen) * 0.02
    if token_bytes is None:
        token_bytes = torch.zeros(meta["vocab_size"], 8, dtype=torch.uint8)
    return CompactAsymmetricLLM(
        base_layers=layers, rotary_emb=rotary, norm=norm,
        compact_weights=weights, token_bytes=token_bytes,
        depth=meta["depth"], n_heads=meta["n_heads"], d_head=meta["d_head"],
        gen_mlp=meta["gen_mlp"], pad_id=0,
        grad_checkpoint=bool(getattr(cfg, "grad_checkpoint", False)),
        factory_prefix=int(meta.get("factory_prefix", 0)),
    )


# ======================================================================================
# Генерация, инструменты, оценка
# ======================================================================================

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Вычисляет арифметическое выражение (+, -, *, /, //, %, **, скобки).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "например: (50 - 18) * 2"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Ищет краткую справку в русской Википедии и возвращает первое предложение.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "поисковый запрос"}},
                "required": ["query"],
            },
        },
    },
]

def _compact_tools(tools: Any) -> Any:
    """1:1 с описанием инструментов в датасете: имя, короткое описание, типы аргументов."""
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            return tools
    if isinstance(tools, dict):
        for key in ("tools", "functions", "available_tools"):
            if isinstance(tools.get(key), list):
                tools = tools[key]
                break
        else:
            tools = [tools]
    if not isinstance(tools, list):
        return tools
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name") or t.get("name")
        if not name:
            continue
        desc = str(fn.get("description") or t.get("description") or "").strip()[:160]
        params = fn.get("parameters") or fn.get("input_schema") or {}
        props = params.get("properties") if isinstance(params, dict) else {}
        simple = {}
        if isinstance(props, dict):
            for k, v in props.items():
                v = v if isinstance(v, dict) else {}
                simple[k] = {"type": v.get("type", "string")}
                if v.get("enum"):
                    simple[k]["enum"] = v["enum"][:8]
        out.append({
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": simple,
                "required": (params.get("required") if isinstance(params, dict) else []) or [],
            },
        })
    return out or tools


def tools_system_text() -> str:
    tools = _compact_tools(TOOLS_SPEC)
    kept = []
    for t in tools:
        cand = kept + [t]
        if len(json.dumps(cand, ensure_ascii=False, separators=(",", ":"))) > 2500:
            break
        kept = cand
    body = json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
    return (
        "Ты — русскоязычный ассистент с инструментами. Пиши ответы по-русски кириллицей, "
        "коротко и по делу; не смешивай китайский, японский, корейский текст.\n"
        "Правила выбора:\n"
        "1) Любой вопрос о фактах, определениях, людях, местах, событиях, датах, числах, "
        "значениях слов и вообще о внешнем мире — это запрос справки. СНАЧАЛА вызови "
        "web_search с КОРОТКИМ запросом (обычно 1–3 слова — сама сущность из вопроса), "
        "и только после <tool_response> дай короткий ответ строго по найденному тексту. "
        "Пример: «что такое венчик?» -> {\"query\":\"венчик\"}; «кто такой Гагарин?» -> "
        "{\"query\":\"Гагарин\"}.\n"
        "2) Вычисления (включая корни и степени) — инструмент calculate с выражением.\n"
        "3) Без инструментов отвечай только на: приветствия и болтовню, мнения, советы, "
        "поздравления, творческие просьбы и вопросы о тебе самом.\n"
        "Не выдумывай факты. Если в <tool_response> написано «не нашёл», так и скажи и "
        "предложи уточнить запрос.\n"
        f"TOOLS: {body}\n"
        "Вызов инструмента — отдельный блок: ровно <tool_call>, затем один полный валидный "
        "JSON-объект, затем </tool_call> или конец сообщения. Не пиши голый JSON без "
        "<tool_call>, не добавляй текст внутрь блока. После <tool_response> дай обычный ответ."
    )


def _safe_arith(expr: str) -> Optional[float]:
    """Безопасный калькулятор: только числа и арифметика."""
    allowed_bin = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_un = (ast.UAdd, ast.USub)
    allowed_calls = {"abs": abs, "round": round, "int": int, "float": float,
                     "min": min, "max": max, "sqrt": math.sqrt, "pow": math.pow}

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_bin):
            return op(node.op, ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_un):
            v = ev(node.operand)
            return +v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in allowed_calls:
            if node.keywords:
                raise ValueError
            return allowed_calls[node.func.id](*[ev(a) for a in node.args])
        raise ValueError

    def op(o, a, b):
        if isinstance(o, ast.Add):
            return a + b
        if isinstance(o, ast.Sub):
            return a - b
        if isinstance(o, ast.Mult):
            return a * b
        if isinstance(o, ast.Div):
            return a / b
        if isinstance(o, ast.FloorDiv):
            return a // b
        if isinstance(o, ast.Mod):
            return a % b
        if isinstance(o, ast.Pow):
            return a ** b
        raise ValueError

    try:
        tree = ast.parse(expr, mode="eval")
        val = ev(tree)
        if isinstance(val, float):
            val = round(val, 10)
            if val.is_integer():
                val = int(val)
        return val
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Локальная база знаний (KB) + BM25: офлайн-first для web_search.
# Маленькой сети поиск нужнее, чем параметрические знания; KB делает поиск
# детерминированным, мгновенным и доступным без интернета.
# ---------------------------------------------------------------------------

_KB_STATE: Dict[str, Any] = {"loaded": False, "docs": [], "index": None}

# Вопросительные/служебные слова не должны мешать сопоставлению заголовка с запросом.
_KB_STOP = {"кто", "такой", "такая", "что", "такое", "где", "когда", "сколько", "как",
            "это", "чем", "про", "для", "или", "об", "о", "найди", "поищи", "дай",
            "расскажи", "информация", "справка", "такой", "какой", "какая", "какие",
            "главные", "факты", "кратко", "нужно", "уточнить", "проверь", "источник",
            "почему", "зачем", "можно", "есть", "был", "была", "были", "этот", "тот"}


def _bm25_tokens(text: str) -> List[str]:
    toks = re.findall(r"[а-яёa-z0-9]+", text.lower())
    out = []
    two_end = ("ов", "ев", "ам", "ям", "ях", "ах", "ей", "ой", "ый", "ий",
               "ая", "ое", "ые", "ие", "ем", "ом", "ую", "юю", "ть", "ся")
    for w in toks:
        if len(w) > 5 and w[-2:] in two_end:
            w = w[:-2]
        elif len(w) > 4 and w[-1] in "аияыиеуюойь":
            w = w[:-1]
        out.append(w)
    return out


def _kb_load():
    if _KB_STATE["loaded"]:
        return
    kb_dir = Path(__file__).resolve().parent / "kb"
    docs: List[Dict[str, Any]] = []
    seen_titles = set()
    for path in (sorted(kb_dir.glob("kb*.jsonl")) if kb_dir.exists() else []):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                t, txt = str(rec.get("title", "")).strip(), str(rec.get("text", "")).strip()
                key = t.lower()
                if t and txt and key not in seen_titles:
                    seen_titles.add(key)
                    docs.append({"title": t, "text": txt})
    index = None
    cache = kb_dir / "index_bm25.pkl"
    try:
        import pickle
        if cache.exists():
            with cache.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("n") == len(docs):
                index = cached["index"]
        if index is None:
            toks = [_bm25_tokens(d["title"]) * 2 + _bm25_tokens(d["text"]) for d in docs]
            from rank_bm25 import BM25Okapi
            index = BM25Okapi(toks, k1=1.5, b=0.75)
            try:
                with cache.open("wb") as f:
                    pickle.dump({"n": len(docs), "index": index}, f)
            except Exception:
                pass
    except Exception:
        # fallback: собственный BM25 по Counter'ам
        df: Counter = Counter()
        tfs = []
        for d in docs:
            tf = Counter(_bm25_tokens(d["title"]) * 2 + _bm25_tokens(d["text"]))
            tfs.append(tf)
            df.update(tf.keys())
        n = max(1, len(docs))
        avgdl = sum(sum(t.values()) for t in tfs) / n
        idf = {term: math.log(1.0 + (n - df_ + 0.5) / (df_ + 0.5)) for term, df_ in df.items()}
        index = ("counter", tfs, idf, avgdl)
    _KB_STATE.update(loaded=True, docs=docs, index=index)


def kb_search(query: str, top_k: int = 1) -> List[Dict[str, Any]]:
    """BM25-поиск по локальной KB. Возвращает [{title, text, score}] или []."""
    _kb_load()
    docs, index = _KB_STATE["docs"], _KB_STATE["index"]
    if not docs:
        return []
    q = _bm25_tokens(query)
    if not q:
        return []
    if index is None:
        return []
    if not isinstance(index, tuple):
        scores = np.asarray(index.get_scores(q), dtype=float)
        order = np.argsort(-scores)[:max(100, top_k)]
        qtok = set(q) - _KB_STOP
        reranked = []
        for i in order:
            i = int(i)
            tkey = " ".join(_bm25_tokens(docs[i]["title"]))
            qkey = " ".join(q)
            ttok = set(_bm25_tokens(docs[i]["title"])) - _KB_STOP
            bonus = 0.0
            if tkey == qkey:
                bonus = 100.0                              # заголовок точно = запрос
            elif ttok and qtok:
                frac = len(ttok & qtok) / len(ttok)
                cover = len(ttok & qtok) / len(qtok)
                if frac >= 1.0:
                    bonus = 15.0
                elif frac >= 0.5:
                    bonus = 5.0
                if cover >= 1.0 and len(ttok) <= len(qtok):
                    bonus += 8.0
                if ttok and ttok <= qtok:
                    bonus += 2.0
            # короче заголовок -> вероятнее искомая сущность
            bonus -= 0.02 * len(ttok)
            reranked.append((scores[i] + bonus, i))
        reranked.sort(reverse=True)
        return [{"title": docs[i]["title"], "text": docs[i]["text"],
                 "score": round(float(sc), 3)} for sc, i in reranked[:top_k]]
    _, tfs, idf, avgdl = index
    k1, b = 1.5, 0.75
    scored = []
    for i, tf in enumerate(tfs):
        dl = sum(tf.values())
        sc = 0.0
        for term in set(q):
            f = tf.get(term, 0)
            if f:
                sc += idf.get(term, 0.0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        if sc > 0:
            scored.append((sc, i))
    scored.sort(reverse=True)
    return [{"title": docs[i]["title"], "text": docs[i]["text"], "score": round(sc, 3)}
            for sc, i in scored[:top_k]]


def _cap_lead(text: str, max_chars: int = 420) -> str:
    """Лид по границам предложений, не длиннее max_chars (легче синтез для малой модели)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = ""
    for p in parts:
        if out and len(out) + len(p) + 1 > max_chars:
            break
        out = (out + " " + p).strip()
    return (out or text.strip())[:max_chars]


def wiki_search(query: str) -> str:
    """Короткая интернет-справка по exact model-selected query.

    Search и extract выполняются с повтором только при transport failure. Повтор
    не меняет query, не извлекает новый query и не подменяет результат. Snippet
    успешного Search API остаётся честным fallback, если extract endpoint слабее.
    """
    q = query.strip().strip("'\"").strip()
    q = re.sub(r"^(найди|поищи|про|о|об)\s+", "", q, flags=re.IGNORECASE).strip()
    if not q:
        return "не нашёл"

    # 1) Локальная KB: мгновенный офлайн-ответ, но ТОЛЬКО при точном совпадении
    # заголовка и запроса. Иначе — живой поиск Википедии: он разрешает редиректы
    # («Эверест» -> «Джомолунгма») и ранжирует по релевантности.
    qtoks_all = _bm25_tokens(q)
    bare = (not any(t in _KB_STOP for t in qtoks_all)) and len(qtoks_all) <= 4
    kb_hits = kb_search(q, top_k=1)
    if kb_hits:
        hit = kb_hits[0]
        qtok = set(qtoks_all) - _KB_STOP
        ttok = set(_bm25_tokens(hit["title"])) - _KB_STOP
        if ttok and (" ".join(sorted(ttok)) == " ".join(sorted(qtok))
                     or (bare and ttok <= qtok and hit["score"] >= 25.0)):
            return _cap_lead(hit["text"])

    headers = {"User-Agent": "compact-dual-dynamic-llm/1.0"}

    def fetch_json(url: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        # Transient Wikipedia/DNS/TLS errors must not turn a valid model query
        # into fabricated evidence. Retrying the exact URL is transport only.
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data if isinstance(data, dict) else None
            except Exception:
                if attempt == 0:
                    time.sleep(0.25)
        return None

    search_url = ("https://ru.wikipedia.org/w/api.php?action=query&list=search&srsearch="
                  + urllib.parse.quote(q) + "&utf8=&format=json&srlimit=1")
    search = fetch_json(search_url)
    results = search.get("query", {}).get("search", []) if search else []
    first_result = results[0] if isinstance(results, list) and results else {}
    title = str(first_result.get("title") or q) if isinstance(first_result, dict) else q
    search_snippet = ""
    if isinstance(first_result, dict):
        search_snippet = re.sub(r"<[^>]+>", "", str(first_result.get("snippet") or ""))
        search_snippet = re.sub(r"\s+", " ", search_snippet).strip()

    if search_snippet:
        extract_url = ("https://ru.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=1"
                       "&explaintext=1&redirects=1&format=json&titles=" + urllib.parse.quote(title))
        extracted = fetch_json(extract_url)
        pages = extracted.get("query", {}).get("pages", {}) if extracted else {}
        if isinstance(pages, dict):
            for pid, page in pages.items():
                if pid == "-1" or not isinstance(page, dict) or not page.get("extract"):
                    continue
                text = str(page["extract"]).strip()
                text = unicodedata.normalize("NFKC", text)
                text = "".join(c for c in text if unicodedata.category(c) != "Mn")
                # Russian Wikipedia leads can quote original Chinese/Japanese/Korean
                # spellings. They are not needed for a Russian tool transcript and
                # must not become accidental student targets.
                text = re.sub(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", text)
                # Drop pronunciation/transliteration parentheses but preserve a parenthesis
                # that contains a measured number, e.g. «(8848,86 м)».
                text = re.sub(
                    r"\(([^)]*)\)",
                    lambda match: match.group(0) if re.search(r"\d", match.group(1)) else "",
                    text,
                )
                text = re.sub(r"\[[^]]*\]", "", text)
                text = re.sub(r"\s+", " ", text).strip().replace('"', "'")
                if text:
                    # A one-sentence lead often omits precisely the requested
                    # date/number. Return a bounded excerpt from the same retrieved
                    # page rather than a fabricated answer or a new query.
                    return _cap_lead(text)
        return "Википедия (поиск): " + search_snippet[:200]

    # A same-query fallback is useful when Wikipedia has no usable result.
    ddg_url = "https://api.duckduckgo.com/?q=" + urllib.parse.quote(q) + "&format=json&no_html=1&skip_disambig=1"
    d3 = fetch_json(ddg_url)
    if d3:
        text = str(d3.get("AbstractText") or "").strip()
        if not text:
            for item in d3.get("RelatedTopics", []) or []:
                if isinstance(item, dict) and item.get("Text"):
                    text = str(item["Text"]).strip()
                    break
        if text:
            text = re.sub(r"\s+", " ", text).strip().replace('"', "'")
            return "DuckDuckGo: " + text[:200]
    if kb_hits and kb_hits[0]["score"] >= 8.0:
        return _cap_lead(kb_hits[0]["text"])
    return "не нашёл"


def run_tool(name: str, args: Dict[str, Any]) -> str:
    """Выполняет только уже валидированный exact-schema вызов модели."""
    try:
        if name == "calculate":
            val = _safe_arith(args["expression"])
            if val is None:
                return json.dumps({"content": "не удалось вычислить"}, ensure_ascii=False)
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return json.dumps({"content": str(val)}, ensure_ascii=False, separators=(",", ":"))
        if name == "web_search":
            return json.dumps({"content": wiki_search(args["query"])}, ensure_ascii=False, separators=(",", ":"))
    except Exception as exc:
        return json.dumps({"content": f"ошибка инструмента: {exc}"}, ensure_ascii=False)
    # last_tool_call не пропускает сюда unknown tool; оставляем защиту на случай
    # прямого программного вызова, а не подменяем инструмент эвристикой.
    return json.dumps({"content": f"инструмент '{name}' недоступен"}, ensure_ascii=False)


def last_tool_call(text: str, end_of_turn: bool = False) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Разбирает только завершённый моделью, полный JSON-вызов.

    Допустимы два явных завершителя, оба сгенерированы самой моделью: текстовый
    ``</tool_call>`` либо нативный конец assistant-хода Qwen (``end_of_turn``).
    Во втором случае рантайм не дописывает тег и не меняет текст. Неполный JSON,
    Python-словарь, неизвестное имя и неверные аргументы не исполняются.
    """
    start = text.rfind("<tool_call>")
    if start < 0:
        return None
    end = text.find("</tool_call>", start)
    if end < 0:
        if not end_of_turn:
            return None
        end = len(text)
    body = text[start + len("<tool_call>"):end].strip()
    try:
        obj = json.loads(body)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    args = obj.get("arguments")
    required = {"calculate": "expression", "web_search": "query"}
    if name not in required or not isinstance(args, dict):
        return None
    value = args.get(required[name])
    if not isinstance(value, str) or not value.strip():
        return None
    return name, args


def build_prompt_text(messages: Sequence[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m["role"] if m["role"] in ("system", "user", "assistant") else "user"
        content = (str(m["content"]).replace("<|im_start|>", "").replace("<|im_end|>", "")
                   .replace("<|endoftext|>", ""))
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


@torch.no_grad()
def generate_text(
    model: nn.Module,
    tok,
    pack: Dict[str, Any],
    messages: Sequence[Dict[str, str]],
    cfg: Config,
    device: torch.device,
    allow_tools: bool = True,
    max_new: Optional[int] = None,
    verbose_tools: bool = True,
) -> str:
    model.eval()
    v_size = int(pack["compact_weights"].shape[0])
    old_to_new = get_forward_map(pack, len(tok))
    new_to_old = get_inverse_map(pack, v_size).tolist()
    eos_old = (tok.convert_tokens_to_ids("<|im_end|>"),
               tok.convert_tokens_to_ids("<|endoftext|>"))
    eos_new = {int(old_to_new[t]) for t in eos_old if isinstance(t, int) and 0 <= t < len(old_to_new)}
    prompt = build_prompt_text(messages)
    old_ids = tok.encode(prompt, add_special_tokens=False)
    cur = torch.tensor([[int(old_to_new[i]) for i in old_ids]], device=device)
    # не выходим за контекст, на котором учили: длинный промпт режем слева
    limit = max_new if max_new is not None else cfg.max_new
    max_prompt = max(64, cfg.max_seq_len - limit)
    if cur.size(1) > max_prompt:
        cur = cur[:, -max_prompt:]

    committed = ""
    seg: List[int] = []
    generated: List[int] = []
    n_tools = 0
    use_cache = (os.environ.get("NEUROSET_NOCACHE", "0") != "1"
                 and hasattr(model, "prefill_cached"))

    # Сегменты: в рамках сегмента якорь фабрики постоянен (как в train после
    # split_tool_segments). После инжекции tool response начинается новый
    # сегмент: полный prefill заново — фабрика читает полученный факт.
    anchor_lens = torch.tensor([cur.size(1)], device=cur.device)
    logits = None
    cache = None

    def _forward_last():
        nonlocal logits, cache
        if use_cache:
            logits, cache = model.prefill_cached(cur, anchor_lens)
        elif cur.is_cuda and getattr(cfg, "amp", "none") in ("auto", "fp16"):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(cur, anchor_lens=anchor_lens)[:, -1, :].float().clone()
        else:
            logits = model(cur, anchor_lens=anchor_lens)[:, -1, :].float().clone()

    def _new_segment():
        nonlocal anchor_lens
        anchor_lens = torch.tensor([cur.size(1)], device=cur.device)
        _forward_last()

    _forward_last()

    for _ in range(limit):
        # JSON в <tool_call> неизбежно повторяет кавычки, скобки и имена ключей.
        # Обычный repetition penalty ломает валидный JSON, поэтому внутри уже
        # начатого вызова он отключён. На обычный текст это не влияет.
        inside_tool_call = False
        if allow_tools and seg:
            partial = tok.decode([new_to_old[t] for t in seg], skip_special_tokens=False)
            inside_tool_call = "<tool_call>" in partial and "</tool_call>" not in partial
        step_logits = logits
        if len(generated) > 3 and not inside_tool_call:
            step_logits = logits.clone()
            for tid in set(generated[-12:]):
                if 0 <= tid < step_logits.size(-1):
                    step_logits[0, tid] /= cfg.rep_penalty
        # запрет повторов n-грамм (декодирование, не влияет на tool call JSON)
        nrn = int(getattr(cfg, "no_repeat_ngram", 0) or 0)
        if nrn >= 2 and not inside_tool_call and len(generated) >= nrn - 1:
            prefix = tuple(generated[-(nrn - 1):])
            banned = set()
            for gi in range(len(generated) - nrn + 1):
                if tuple(generated[gi:gi + nrn - 1]) == prefix:
                    banned.add(generated[gi + nrn - 1])
            if banned:
                step_logits = step_logits.clone()
                for bt in banned:
                    step_logits[0, bt] = -1e9
        if cfg.temperature and cfg.temperature > 0:
            probs = F.softmax(step_logits / cfg.temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = step_logits.argmax(dim=-1, keepdim=True)
        tid = int(nxt.item())
        if tid in eos_new:
            # Qwen завершает assistant-ход специальным IM_END. Это допустимый
            # терминатор только для уже полного JSON; тег и скобки не дописываются.
            seg_text = tok.decode([new_to_old[t] for t in seg], skip_special_tokens=False)
            call = last_tool_call(seg_text, end_of_turn=True) if allow_tools and n_tools < 2 else None
            if call:
                name, args = call
                res = run_tool(name, args)
                if verbose_tools:
                    print(f"   [tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {res}")
                block = f"\n\n<tool_response>\n{res}\n</tool_response>"
                committed = committed + seg_text + block
                inj = tok.encode(block, add_special_tokens=False)
                inj_new = torch.tensor([[int(old_to_new[i]) for i in inj]], device=device)
                cur = torch.cat([cur, inj_new], dim=1)
                seg = []
                n_tools += 1
                _new_segment()   # фабрика читает tool response
                continue
            break
        cur = torch.cat([cur, nxt], dim=1)
        generated.append(tid)
        seg.append(tid)

        seg_text = tok.decode([new_to_old[t] for t in seg], skip_special_tokens=False)
        if allow_tools and n_tools < 2 and "</tool_call>" in seg_text:
            call = last_tool_call(seg_text)
            if call:
                name, args = call
                res = run_tool(name, args)
                if verbose_tools:
                    print(f"   [tool] {name}({json.dumps(args, ensure_ascii=False)}) -> {res}")
                block = f"\n\n<tool_response>\n{res}\n</tool_response>"
                committed = committed + seg_text + block
                inj = tok.encode(block, add_special_tokens=False)
                inj_new = torch.tensor([[int(old_to_new[i]) for i in inj]], device=device)
                cur = torch.cat([cur, inj_new], dim=1)
                seg = []
                n_tools += 1
                _new_segment()   # фабрика читает tool response
                continue

        # следующий токен: дешёвый шаг по кэшу или полный форвард (fallback)
        if use_cache and cache is not None:
            logits = model.step_cached(nxt, cache)
        elif cur.is_cuda and getattr(cfg, "amp", "none") in ("auto", "fp16"):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(cur, anchor_lens=anchor_lens)[:, -1, :].float().clone()
        else:
            logits = model(cur, anchor_lens=anchor_lens)[:, -1, :].float().clone()

    final = committed + tok.decode([new_to_old[t] for t in seg], skip_special_tokens=True)
    final = final.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return final


def prompt_messages(user: str, cfg: Config) -> List[Dict[str, str]]:
    """Единый системный контекст для каждого сообщения.

    Это LLM с постоянно доступными инструментами, а не эвристический чат-бот:
    при включённом --tools модель всегда получает одну и ту же system-инструкцию
    и сама решает, нужен ли вызов инструмента.
    """
    msgs: List[Dict[str, str]] = []
    if cfg.tools:
        msgs.append({"role": "system", "content": tools_system_text()})
    msgs.append({"role": "user", "content": user})
    return msgs


def short_sample(model, tok, pack, cfg, device, prompt: str, max_new: int = 40) -> str:
    try:
        # ANCHORED-FIX (совместно): system-контекст как в chat/bench/demo —
        # без него промпт вне распределения train (там system есть всегда).
        return generate_text(model, tok, pack, prompt_messages(prompt, cfg),
                             cfg, device, allow_tools=False, max_new=max_new, verbose_tools=False)
    except Exception as exc:
        return f"<ошибка генерации: {exc}>"


def pick_device(cfg: Config) -> torch.device:
    if cfg.device == "cpu":
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_settings(cfg: Config, device: torch.device):
    if cfg.amp == "none":
        return None, False
    if cfg.amp == "auto":
        if device.type == "cuda":
            return torch.float16, True
        return None, False
    if cfg.amp == "fp16":
        if device.type != "cuda":
            return None, False
        return torch.float16, True
    if cfg.amp == "bf16":
        return torch.bfloat16, False
    return None, False


class TrainLogger:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.f = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, msg: str = ""):
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("utf-8", "replace").decode("utf-8"), flush=True)
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def fmt_sec(sec: float) -> str:
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _batch_loss(model, x, y, w=None) -> torch.Tensor:
    # ANCHORED-FIX: якорь фабрики = промпт-часть строки (до первого
    # assistant-токена). logits[p] предсказывает x[p+1]; первый assistant-токен
    # стоит в x на p0+1, значит фабрика видит только позиции < p0+1 — утечка
    # ответа через пул/фронтир устранена. При factory_prefix>0 форвард сам
    # обрезает якорь фиксированным префиксом (train-условия сохраняются).
    lab = y[:, 1:] != -100
    ar = torch.arange(lab.size(1), device=y.device)
    first = torch.where(lab, ar, torch.full_like(ar, lab.size(1))).min(dim=1).values
    anchor_lens = (first + 1).clamp(min=1, max=max(1, x.size(1) - 1))
    logits = model(x[:, :-1], anchor_lens=anchor_lens)
    prefix = getattr(model, "factory_prefix", 0)
    if prefix:
        y = y.clone()
        y[:, :prefix + 1] = -100
    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                         y[:, 1:].reshape(-1), ignore_index=-100, reduction="none")
    if w is not None:
        ww = w[:, 1:].reshape(-1).to(ce.dtype)
        return (ce * ww).sum() / ww.sum().clamp(min=1.0)
    return ce.mean()


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: int = 40) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        b_inp, b_tgt = batch[0].to(device), batch[1].to(device)
        b_w = batch[2].to(device) if len(batch) == 3 else None
        with torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else _nullctx():
            loss = _batch_loss(model, b_inp, b_tgt, b_w)
        if torch.isfinite(loss):
            total += float(loss)
            n += 1
        if n >= max_batches:
            break
    model.train()
    return total / max(1, n)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _plain_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for k, v in model.state_dict().items():
        k = k.replace("_orig_mod.", "")
        sd[k] = v
    return sd


def save_model_state(model: nn.Module, path: Path, dtype: torch.dtype):
    sd = {}
    for k, v in _plain_state_dict(model).items():
        if v.is_floating_point():
            v = v.to(dtype)
        sd[k] = v.detach().cpu()
    if "embed_tokens.weight" in sd and "lm_head.weight" in sd:
        sd["lm_head.weight"] = sd["embed_tokens.weight"]  # не дублируем tied-веса на диске
    torch.save(sd, path)


# ---------------------------------------------------------------------------
# Muon: ортогонализованный импульс для 2D-матриц фабрики (Keller Jordan et al.)
# ---------------------------------------------------------------------------

def _ns5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz итерация: приближённая ортогонализация матрицы градиента."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.t() if transposed else X


class HybridOpt(torch.optim.Optimizer):
    """Один оптимизатор с двумя алгоритмами: algo='muon' для скрытых 2D-матриц,
    algo='adam' (AdamW) для эмбеддингов/норм/GRU/якоря. Интерфейс 1:1 с AdamW —
    train-цикл и lr_at() работают без изменений."""

    def __init__(self, groups):
        seen = set()
        for g in groups:
            for p in g["params"]:
                if id(p) in seen:
                    raise ValueError("параметр попал в две группы")
                seen.add(id(p))
        defaults = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                        algo="adam", momentum=0.95, nesterov=True, ns_steps=5, name="?")
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            algo = group["algo"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if wd:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if algo == "muon":
                    if "buf" not in state:
                        state["buf"] = torch.zeros_like(g)
                    buf = state["buf"]
                    buf.mul_(group["momentum"]).add_(g)
                    u = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                    u = _ns5(u.float(), steps=group["ns_steps"]).to(p.dtype)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(u, alpha=-lr * scale)
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(g)
                        state["exp_avg_sq"] = torch.zeros_like(g)
                        state["t"] = 0
                    state["t"] += 1
                    t = state["t"]
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    dh = (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + group["eps"])
                    p.add_(dh, alpha=-lr)
        return loss


def make_optimizer(model: nn.Module, cfg: Config):
    base_params = list(model.base_layer.parameters()) if model.single_base else list(model.base_layers.parameters())
    emb_params = list(model.embed_tokens.parameters()) + list(model.morph_patcher.parameters())
    factory_params = (list(model.factory.parameters()) + list(model.gen_layer.parameters())
                      + [model.g_init, model.depth_scale])
    lr_base = cfg.lr_base if cfg.lr_base > 0 else cfg.lr
    groups = [
        {"params": base_params, "lr": lr_base, "weight_decay": cfg.weight_decay, "name": "base"},
        {"params": emb_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay, "name": "embed"},
        {"params": factory_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay, "name": "factory"},
    ]
    if getattr(cfg, "opt", "adamw") != "muon":
        return torch.optim.AdamW(groups)
    # Muon забирает только 2D-матрицы генераторов фабрики (attn_gen/mlp_gen/proj_in) —
    # это и есть «скрытые» слои, где ортогонализация даёт выигрыш.
    muon_params, factory_rest = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ("factory.attn_gen", "factory.mlp_gen", "factory.proj_in")) \
                and p.ndim >= 2:
            muon_params.append(p)
    muon_ids = {id(p) for p in muon_params}
    factory_rest = [p for p in factory_params if id(p) not in muon_ids]
    groups[2]["params"] = factory_rest
    for g in groups:
        g["algo"] = "adam"
    groups.append({
        "params": muon_params, "lr": getattr(cfg, "muon_lr", 0.02),
        "weight_decay": cfg.weight_decay, "name": "muon", "algo": "muon",
    })
    return HybridOpt(groups)


def lr_at(cfg: Config, base_lr: float, step: int, elapsed: float, total: float) -> float:
    if step < cfg.warmup:
        return base_lr * (step + 1) / max(1, cfg.warmup)
    p = min(1.0, max(0.0, elapsed / max(1.0, total)))
    return base_lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def resolve_ckpt(path: str) -> Tuple[Path, Path]:
    """Возвращает (run_dir, ckpt_path) из папки рана или файла чекпойнта."""
    p = Path(path)
    if p.is_file():
        run_dir = p.parent.parent if p.parent.name == "checkpoints" else p.parent
        return run_dir, p
    for name in ("last.pt", "final.pt", "best.pt", "last_model.pt"):
        ckpt = p / "checkpoints" / name
        if ckpt.exists():
            return p, ckpt
    return p, p / "checkpoints" / "last.pt"


def train(cfg: Config, tok, model, pack, data, run_dir: Path, logger: TrainLogger):
    device = pick_device(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    if device.type == "cpu":
        torch.set_float32_matmul_precision("high")

    logger("=" * 78)
    logger(f"=== ОБУЧЕНИЕ {cfg.preset or 'custom'} на {device} ===")
    logger("=" * 78)
    params = count_parameters(model)
    n_params_total = params.get("TOTAL", 0)
    logger(f"Параметры: {human_num(params['TOTAL'])}  " +
           " | ".join(f"{k}: {human_num(v)}" for k, v in params.items() if k != "TOTAL"))
    logger(f"Данные: train {data.get('n_train', 0):,} сэмплов / "
           f"{data.get('train_tokens', 0):,} токенов; val {data.get('n_val', 0):,}")
    vmeta = pack.get("meta", {}) or {}
    logger(f"Словарь: {cfg.vocab_size} (стратегия {cfg.vocab_strategy}, "
           f"покрытие {vmeta.get('coverage', 0)*100:.2f}%), depth={cfg.depth}, "
           f"seq={cfg.max_seq_len}")

    mapping = get_forward_map(pack, len(tok))
    data["train"].set_map(mapping)
    data["val"].set_map(mapping)

    train_loader = DataLoader(data["train"], batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(data["val"], batch_size=max(1, cfg.batch_size // 2), shuffle=False,
                            drop_last=False, collate_fn=collate_fn,
                            pin_memory=(device.type == "cuda"))
    if len(train_loader) == 0:
        raise ValueError("Датасет меньше одного батча — увеличьте данные или уменьшите batch_size")

    model.to(device)
    opt = make_optimizer(model, cfg)
    amp_dtype, use_scaler = amp_settings(cfg, device)
    scaler = None
    if use_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    metrics_path = run_dir / "metrics.jsonl"
    history: List[Dict[str, Any]] = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                history.append(json.loads(line))
            except Exception:
                pass
    best_val = min((h["val"] for h in history if "val" in h), default=float("inf"))

    start_step = 0
    elapsed_before = 0.0
    if cfg.resume:
        _, ckpt = resolve_ckpt(cfg.resume)
        if not ckpt.exists():
            raise FileNotFoundError(f"Нет чекпойнта {ckpt}")
        logger(f"[resume] загружаю {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if not (isinstance(state, dict) and "model" in state):
            state = {"model": state, "optimizer": None, "scaler": None,
                     "step": 0, "elapsed": 0.0}
        model_sd = {k.replace("_orig_mod.", ""): v for k, v in state["model"].items()}
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing:
            logger(f"[resume] отсутствуют ключи: {len(missing)} (это нормально при смене depth)")
        if state.get("optimizer"):
            try:
                opt.load_state_dict(state["optimizer"])
            except Exception as exc:
                logger(f"[resume] оптимизатор не загружен: {exc}")
        if scaler is not None and state.get("scaler"):
            try:
                scaler.load_state_dict(state["scaler"])
            except Exception:
                pass
        start_step = int(state.get("step", 0))
        elapsed_before = float(state.get("elapsed", 0.0))
        logger(f"[resume] step={start_step}, уже обучено {fmt_sec(elapsed_before)}")

    total_budget = cfg.hours * 3600.0
    if cfg.extra_hours > 0:
        total_budget = elapsed_before + cfg.extra_hours * 3600.0
    remaining = total_budget - elapsed_before
    if remaining <= 5:
        logger(f"[!] Бюджет исчерпан: {fmt_sec(elapsed_before)} из {fmt_sec(total_budget)}. "
               f"Добавьте время: --extra-hours 2")
        return {"best_val": best_val, "last_val": None, "steps": start_step, "elapsed": elapsed_before}

    logger(f"Бюджет: {fmt_sec(total_budget)} | осталось {fmt_sec(remaining)} | "
           f"batch={cfg.batch_size}x{cfg.grad_accum} | seq={cfg.max_seq_len} | amp={amp_dtype}")
    logger("")

    model.train()
    opt.zero_grad(set_to_none=True)
    step = start_step
    micro = 0
    t0 = time.time()
    trained = elapsed_before
    tokens_seen = 0
    loss_acc, loss_n = 0.0, 0
    last_log = last_eval = last_save = last_sample = 0.0
    best_path = run_dir / "checkpoints" / "best.pt"
    last_path = run_dir / "checkpoints" / "last.pt"
    interrupted = False
    last_val = None

    def curr_elapsed() -> float:
        return trained + (time.time() - t0)

    try:
        stop = False
        while not stop:
            for batch in train_loader:
                el = curr_elapsed()
                if el >= total_budget or (cfg.max_steps and step >= cfg.max_steps):
                    stop = True
                    break
                b_inp, b_tgt = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                b_w = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None
                with torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else _nullctx():
                    loss = _batch_loss(model, b_inp, b_tgt, b_w)
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    micro = 0
                    logger("[!] non-finite loss — пропускаю батч")
                    continue
                scaled = loss / cfg.grad_accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                micro += 1
                tokens_seen += b_inp.numel()
                loss_acc += float(loss.detach())
                loss_n += 1

                if micro % cfg.grad_accum == 0:
                    for g in opt.param_groups:
                        base_lr = g.get("lr", cfg.lr)
                        # храним «базовый» lr отдельно, чтобы cosine не портил его
                        if "_base_lr" not in g:
                            g["_base_lr"] = g["lr"]
                        g["lr"] = lr_at(cfg, g["_base_lr"], step, el, total_budget)
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1

                    el = curr_elapsed()
                    if el - last_log >= cfg.log_seconds:
                        avg = loss_acc / max(1, loss_n)
                        tok_s = tokens_seen / max(1e-6, time.time() - t0)
                        eta = max(0.0, total_budget - el)
                        logger(f"[{step:>6} | {fmt_sec(el)}/{fmt_sec(total_budget)}] "
                               f"loss {avg:.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                               f"{tok_s/1000:.1f}k tok/s | ETA {fmt_sec(eta)}")
                        metrics_path.open("a", encoding="utf-8").write(json.dumps({
                            "step": step, "t": round(el, 1), "loss": round(avg, 5),
                            "lr": opt.param_groups[0]["lr"], "tokens": tokens_seen,
                            "tok_s": round(tok_s, 1),
                        }) + "\n")
                        history.append({"step": step, "t": round(el, 1), "loss": avg})
                        loss_acc, loss_n = 0.0, 0
                        last_log = el

                    if cfg.dry_run and (step - start_step) >= 10:
                        session_steps = max(1, step - start_step)
                        session_time = max(1e-6, time.time() - t0)
                        speed = tokens_seen / session_time
                        steps_planned = int(remaining / (session_time / session_steps))
                        logger("")
                        logger(f"[plan] ~{speed/1000:.1f}k токенов/с, ~{speed*3600/1e6:.0f}M токенов/час")
                        logger(f"[plan] за {fmt_sec(remaining)} успеете ~{steps_planned:,} шагов "
                               f"(~{steps_planned*cfg.batch_size*cfg.grad_accum:,} сэмплов)")
                        return {"best_val": best_val, "last_val": None, "steps": step,
                                "elapsed": curr_elapsed()}

                    if cfg.eval_seconds and el - last_eval >= cfg.eval_seconds:
                        last_eval = el
                        val_loss = evaluate(model, val_loader, device, amp_dtype,
                                            max_batches=cfg.val_batches)
                        last_val = val_loss
                        improved = val_loss < best_val - 1e-6
                        if improved:
                            best_val = val_loss
                            save_model_state(model, best_path,
                                             torch.float16 if cfg.export_dtype == "fp16" else torch.float32)
                        logger(f"   [val @ {step}] {val_loss:.4f} | best {best_val:.4f} | "
                               f"ppl {math.exp(min(20, val_loss)):.1f}"
                               + ("  <- best" if improved else ""))
                        metrics_path.open("a", encoding="utf-8").write(
                            json.dumps({"step": step, "t": round(el, 1), "val": round(val_loss, 5)}) + "\n")
                        last_log = el

                        can_sample = not (device.type == "cpu" and n_params_total > 20_000_000)
                        if (not cfg.no_samples) and cfg.sample_seconds and el - last_sample >= cfg.sample_seconds and can_sample:
                            last_sample = el
                            logger("   [samples]")
                            for p in ("привет бро", "посчитай: 50 - 18"):
                                r = short_sample(model, tok, pack, cfg, device, p)
                                logger(f"     usr> {p}\n     agt> {r[:200]}")
                            model.train()

                    if cfg.save_minutes and (el - last_save) >= cfg.save_minutes * 60:
                        last_save = el
                        run_dir.joinpath("checkpoints").mkdir(exist_ok=True)
                        save_model_state(model, run_dir / "checkpoints" / "last_model.pt",
                                         torch.float16 if cfg.export_dtype == "fp16" else torch.float32)
                        torch.save({
                            "model": {k: v.detach().cpu() for k, v in _plain_state_dict(model).items()},
                            "optimizer": opt.state_dict(),
                            "scaler": scaler.state_dict() if scaler is not None else None,
                            "step": step,
                            "elapsed": el,
                            "config": asdict(cfg),
                        }, last_path)
                        logger(f"   [save] чекпойнт на шаге {step} ({fmt_sec(el)})")

        if stop:
            pass
    except KeyboardInterrupt:
        interrupted = True
        logger("\n[!] Ctrl+C — сохраняю и выхожу...")

    el = curr_elapsed()
    final_dir = run_dir / "checkpoints"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_model_state(model, final_dir / "final.pt",
                     torch.float16 if cfg.export_dtype == "fp16" else torch.float32)
    torch.save({
        "model": {k: v.detach().cpu() for k, v in _plain_state_dict(model).items()},
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "elapsed": el,
        "config": asdict(cfg),
    }, last_path)
    # финальная валидация — всегда, чтобы в отчёте было честное число
    last_val = evaluate(model, val_loader, device, amp_dtype, max_batches=cfg.val_batches)
    metrics_path.open("a", encoding="utf-8").write(
        json.dumps({"step": step, "t": round(el, 1), "val": round(last_val, 5)}) + "\n")
    if last_val < best_val - 1e-6:
        best_val = last_val
        save_model_state(model, best_path,
                         torch.float16 if cfg.export_dtype == "fp16" else torch.float32)
    logger("")
    logger(f"[УСПЕХ] обучение завершено: {step:,} шагов, {fmt_sec(el)}, "
           f"val={last_val:.4f}, best={best_val:.4f}")
    logger(f"  final.pt: {final_dir/'final.pt'}")
    logger(f"  best.pt:  {best_path}")
    return {"best_val": best_val, "last_val": last_val, "steps": step, "elapsed": el}


# ======================================================================================
# Отчёты, конфиг рана, загрузка
# ======================================================================================

def save_run_config(run_dir: Path, cfg: Config, model: nn.Module, pack: Dict[str, Any]):
    payload = {
        "config": asdict(cfg),
        "arch": {
            **cfg.arch(),
            "vocab_size": int(pack["compact_weights"].shape[0]),
            "attention_bias": cfg.attention_bias,
            "rope_theta": cfg.rope_theta,
            "rms_eps": cfg.rms_eps,
        },
        "params": count_parameters(model),
        "vocab_meta": pack.get("meta", {}),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                         encoding="utf-8")


def load_run(run_or_ckpt: str) -> Tuple[Path, Dict[str, Any], Config]:
    run_dir, _ = resolve_ckpt(run_or_ckpt)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"В {run_dir} нет config.json — это не папка рана?")
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    valid = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in payload.get("config", {}).items() if k in valid})
    return run_dir, payload, cfg


def load_model_from_run(run_dir: Path, payload: Dict[str, Any], cfg: Config,
                        device: torch.device, log=print):
    pack_path = run_dir / "compact_vocab.pt"
    if not pack_path.exists():
        pack_path = run_dir / "vocab_pack.pt"
    if not pack_path.exists():
        raise FileNotFoundError(f"Нет compact_vocab.pt в {run_dir}")
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    meta = dict(payload["arch"])
    meta["vocab_size"] = int(pack["compact_weights"].shape[0])
    meta["d_model"] = int(pack["compact_weights"].shape[1])
    meta.setdefault("attention_bias", True)
    meta.setdefault("rope_theta", 1e6)
    meta.setdefault("rms_eps", 1e-6)
    meta.setdefault("kv_heads", meta["n_heads"])
    meta.setdefault("intermediate_size", meta["d_model"] * 4)
    meta.setdefault("base_layers", 1)
    meta.setdefault("depth", getattr(cfg, "depth", 8))
    meta.setdefault("gen_mlp", getattr(cfg, "gen_mlp", 128))
    meta.setdefault("max_seq_len", getattr(cfg, "max_seq_len", 512))
    model = build_scaffold(cfg, meta, token_bytes=pack["token_bytes"])

    ckpt = None
    for name in ("final.pt", "best.pt", "last_model.pt", "last.pt"):
        p = run_dir / "checkpoints" / name
        if p.exists():
            ckpt = p
            break
    if ckpt is None:
        raise FileNotFoundError(f"В {run_dir}/checkpoints нет чекпойнта")
    log(f"[i] Чекпойнт: {ckpt}")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        log(f"[!] Не хватило ключей: {len(missing)}")
    model.to(device).eval()

    tok_dir = run_dir / "tokenizer"
    if (tok_dir / "tokenizer_config.json").exists() or (tok_dir / "tokenizer.json").exists():
        tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
        log(f"[i] Токенизатор рана: {tok_dir}")
    else:
        tok = load_tokenizer(cfg, log)
    return model, tok, pack


def write_loss_svg(history: Sequence[Dict[str, Any]], path: Path):
    """Простой self-contained SVG-график без зависимостей."""
    train = [(h["t"], h["loss"]) for h in history if "loss" in h and "t" in h]
    val = [(h["t"], h["val"]) for h in history if "val" in h and "t" in h]
    if not train and not val:
        return
    xs = [p[0] for p in train + val] or [1]
    ys = [p[1] for p in train + val] or [1]
    xmax = max(xs) or 1
    ymin, ymax = min(ys), max(ys)
    if ymax - ymin < 1e-6:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad
    W, H, L, R, T, B = 900, 340, 70, 24, 30, 48

    def xy(t, v):
        x = L + (W - L - R) * (t / xmax)
        y = T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin))
        return x, y

    # сглаживание train
    smooth = []
    win = max(1, len(train) // 80)
    for i in range(len(train)):
        seg = train[max(0, i - win + 1):i + 1]
        smooth.append((train[i][0], sum(v for _, v in seg) / len(seg)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="#0f1117"/>',
        f'<text x="{L}" y="20" fill="#e5e7eb" font-family="monospace" font-size="14">'
        'Compact Dual Dynamic LLM — loss</text>',
    ]
    # сетка
    for i in range(5):
        y = T + (H - T - B) * i / 4
        v = ymax - (ymax - ymin) * i / 4
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#23283a"/>')
        parts.append(f'<text x="{L-8}" y="{y+4:.1f}" fill="#8b93a7" font-family="monospace" '
                     f'font-size="11" text-anchor="end">{v:.2f}</text>')
    for i in range(5):
        x = L + (W - L - R) * i / 4
        t = xmax * i / 4
        parts.append(f'<text x="{x:.1f}" y="{H-B+20}" fill="#8b93a7" font-family="monospace" '
                     f'font-size="11" text-anchor="middle">{fmt_sec(t)}</text>')
    if smooth:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(t, v) for t, v in smooth))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="#60a5fa" stroke-width="1.6"/>')
    for t, v in val:
        x, y = xy(t, v)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#f59e0b"/>')
    parts.append(f'<text x="{W-R}" y="20" fill="#60a5fa" font-family="monospace" font-size="12" '
                 'text-anchor="end">train</text>')
    parts.append(f'<text x="{W-R}" y="38" fill="#f59e0b" font-family="monospace" font-size="12" '
                 'text-anchor="end">val</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_run_report(run_dir: Path, cfg: Config, pack: Dict[str, Any], data: Dict[str, Any],
                     best_val: float, history: Sequence[Dict[str, Any]], demo: str,
                     steps: int, elapsed: float):
    # параметры уже есть в config.json — читаем оттуда, чтобы не собирать модель повторно
    try:
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        params = saved.get("params", {})
    except Exception:
        params = {}

    last_val = next((h["val"] for h in reversed(history) if "val" in h), float("nan"))
    total = params.get("TOTAL", 0)
    ppl = math.exp(last_val) if last_val == last_val and last_val < 20 else float("nan")
    vmeta = pack.get("meta", {}) or {}

    readme = f"""# {run_dir.name}

**Compact Dual Dynamic LLM** — обучен скриптом `build_neuroset.py`.

| | |
|---|---|
| параметров | **{human_num(total)}** ({total:,}) |
| архитектура | d_model={cfg.d_model}, heads={cfg.n_heads}x{cfg.d_head}, base_layers={cfg.base_layers}, depth={cfg.depth} |
| словарь | {cfg.vocab_size} ({cfg.vocab_strategy}, покрытие {vmeta.get('coverage', 0)*100:.2f}%) |
| обучение | {fmt_sec(elapsed)}, {steps:,} шагов |
| val loss | {last_val:.4f} |
| val ppl | {ppl:.2f} |
| best val | {best_val:.4f} |
| данных | {data.get('n_train', 0):,} сэмплов / {data.get('train_tokens', 0):,} токенов |

![loss](loss_curve.svg)

## Демо

```
{demo}
```

## Чат

```bash
python build_neuroset.py --mode chat --run {run_dir}
```

## Бенч

```bash
python build_neuroset.py --mode bench --run {run_dir}
```

## Файлы

- `checkpoints/final.pt` — веса ({cfg.export_dtype})
- `checkpoints/best.pt` — лучшая валидация
- `checkpoints/last.pt` — полное состояние для `--resume`
- `compact_vocab.pt` — compact-словарь (те же ключи, что в вашем проекте)
- `tokenizer/` — токенизатор
- `metrics.jsonl` — кривая обучения
- `demo.txt` — примеры генерации

> Модель крошечная и обучена за ограниченное время — это демонстрация работоспособности
> архитектуры, а не качества уровня Qwen.
"""
    (run_dir / "README.md").write_text(readme, encoding="utf-8")


# ======================================================================================
# Чат и бенч
# ======================================================================================

BENCH_PROMPTS = [
    "привет бро",
    "посчитай: 50 - 18",
    "поищи столицу франции",
    "сколько будет 2 * 2",
    "расскажи коротко о Москве",
]


def run_bench(cfg: Config, tok, pack, model, device, log=print) -> str:
    lines = []
    for q in BENCH_PROMPTS:
        log(f"usr> {q}")
        ans = generate_text(model, tok, pack, prompt_messages(q, cfg), cfg, device,
                            allow_tools=cfg.tools)
        log(f"agt> {ans}\n")
        lines.append(f"usr> {q}\nagt> {ans}\n")
    text = "\n".join(lines)
    return text


def chat_main(cfg: Config):
    device = pick_device(cfg)
    run_dir, payload, run_cfg = load_run(cfg.run or cfg.resume)
    # инференс-настройки берём из CLI/текущего конфига, архитектуру — из рана
    run_cfg.max_new = cfg.max_new
    run_cfg.temperature = cfg.temperature
    run_cfg.rep_penalty = cfg.rep_penalty
    run_cfg.tools = cfg.tools
    run_cfg.chat_context = cfg.chat_context
    model, tok, pack = load_model_from_run(run_dir, payload, run_cfg, device)

    print("=" * 70)
    print(f"=== ЧАТ: {run_dir.name} ===")
    print("=" * 70)
    if cfg.prompt:
        ans = generate_text(model, tok, pack, prompt_messages(cfg.prompt, run_cfg),
                            run_cfg, device, allow_tools=run_cfg.tools)
        print(f"usr> {cfg.prompt}\nagt> {ans}")
        return
    run_bench(run_cfg, tok, pack, model, device)
    print("=" * 70)
    if run_cfg.chat_context:
        print("Интерактивный режим (контекст между сообщениями: ВКЛ).")
    else:
        print("Интерактивный режим (контекст между сообщениями: ВЫКЛ — "
              "модель не обучена его удерживать).")
    print("Пустая строка / exit — выход, /bench — бенч.\n")
    history: List[Dict[str, str]] = []
    while True:
        try:
            usr = input("usr> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not usr:
            break
        if usr.lower() in ("exit", "quit", "выход"):
            break
        if usr == "/bench":
            run_bench(run_cfg, tok, pack, model, device)
            continue
        messages = prompt_messages(usr, run_cfg)
        # По умолчанию история НЕ подклеивается: модель не обучена удерживать
        # контекст нескольких сообщений, лишняя история только портит генерацию.
        if run_cfg.chat_context:
            messages[-1:-1] = history[-6:]
        ans = generate_text(model, tok, pack, messages,
                             run_cfg, device, allow_tools=run_cfg.tools)
        print(f"agt> {ans}\n")
        history.extend(({"role": "user", "content": usr},
                        {"role": "assistant", "content": ans}))


def bench_main(cfg: Config):
    device = pick_device(cfg)
    run_dir, payload, run_cfg = load_run(cfg.run or cfg.resume)
    run_cfg.tools = cfg.tools
    model, tok, pack = load_model_from_run(run_dir, payload, run_cfg, device)
    text = run_bench(run_cfg, tok, pack, model, device)
    (run_dir / "bench.txt").write_text(text, encoding="utf-8")
    print(f"\n[i] Результаты сохранены в {run_dir/'bench.txt'}")


# ======================================================================================
# Визард
# ======================================================================================

BANNER = r"""
   ______                 __     ____        __      ____
  / ____/___  ____  ____  / /_   / __ \__  __/ /_   / __ \__  ______  ___________
 / /   / __ \/ __ \/ __ \/ __/  / / / / / / / __/  / / / / / / / _ \/ ___/ ___/
/ /___/ /_/ / /_/ / /_/ / /_   / /_/ / /_/ / /_   / /_/ / /_/ /  __/ /  (__  )
\____/\____/ .___/\____/\__/   \____/\__,_/\__/   \____/\__,_/\___/_/  /____/
          /_/
      Compact Dual Dynamic LLM — обучение за ночь
"""


def _ask(prompt: str, default: Any, cast=str):
    if not sys.stdin.isatty():
        return default
    suffix = f" [{default}]" if default != "" else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        if not raw:
            return default
        try:
            return cast(raw)
        except Exception:
            print("  ! Не понял, попробуйте ещё раз")


def _ask_yes(prompt: str, default: bool = True) -> bool:
    if not sys.stdin.isatty():
        return default
    try:
        raw = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not raw:
        return default
    return raw in ("y", "yes", "д", "да", "1")


def _choose_arch(cfg: Config):
    while cfg.init not in ("qwen", "scratch"):
        cfg.init = _ask("Инициализация: qwen (готовый слой) или scratch (с нуля)", cfg.init, str)
    if cfg.init == "qwen":
        cfg.model_name = _ask("Qwen: HF id или локальная папка", cfg.model_name, str)
        print("\n  init=qwen: d_model/heads берутся из Qwen автоматически.")
        cfg.depth = _ask("Глубина рекуррентных генераций (depth)", cfg.depth, int)
        cfg.base_layers = _ask("Сколько слоёв Qwen взять (1 = как в оригинале)", cfg.base_layers, int)
        cfg.base_layer = _ask("С какого слоя Qwen начать", cfg.base_layer, int)
    else:
        cfg.d_model = _ask("d_model", cfg.d_model, int)
        cfg.d_head = _ask("d_head", cfg.d_head, int)
        while True:
            cfg.n_heads = _ask("n_heads (d_model = n_heads * d_head)", cfg.n_heads, int)
            if cfg.n_heads * cfg.d_head == cfg.d_model:
                break
            print(f"  ! {cfg.n_heads} * {cfg.d_head} = {cfg.n_heads*cfg.d_head}, "
                  f"а d_model={cfg.d_model}. Попробуйте ещё.")
        cfg.kv_heads = _ask("kv_heads (GQA в базовом слое)", cfg.kv_heads, int)
        cfg.base_layers = _ask("base_layers", cfg.base_layers, int)
        cfg.depth = _ask("depth (сколько раз прогонять фабрику)", cfg.depth, int)
    cfg.vocab_size = _ask("Размер compact-словаря", cfg.vocab_size, int)
    while cfg.vocab_strategy not in ("freq", "coverage", "stratified", "full"):
        cfg.vocab_strategy = _ask("Стратегия словаря (freq/coverage/stratified/full)",
                                  cfg.vocab_strategy, str)
    cfg.max_seq_len = _ask("max_seq_len", cfg.max_seq_len, int)
    cfg.max_tokens = int(_ask("Лимит токенов обучения (млн)", round(cfg.max_tokens / 1e6, 1), float) * 1e6)
    cfg.batch_size = _ask("batch_size", cfg.batch_size, int)
    cfg.grad_accum = _ask("grad_accum", cfg.grad_accum, int)
    cfg.amp = _ask("amp (auto/fp16/bf16/none)", cfg.amp, str)
    cfg.preset = "custom"


def interactive_wizard(cfg: Config) -> Config:
    print(BANNER)
    # Путь к датасету указывается ВРУЧНУЮ всегда. Никакой автосборки (make_dataset)
    # и никакого фолбэка на «готовый» датасет по умолчанию здесь нет.
    while not cfg.data_dir or not (Path(cfg.data_dir) / "train.jsonl").exists():
        if cfg.data_dir:
            print(f"[!] Не нашёл {Path(cfg.data_dir)/'train.jsonl'}. "
                  f"Укажите путь к готовому датасету вручную.")
        if not sys.stdin.isatty():
            raise SystemExit("Укажите путь к датасету: --data-dir ПАПКА (или --train-file ФАЙЛ).")
        cfg.data_dir = str(_ask("Папка с датасетом (train.jsonl/val.jsonl)", "", str)).strip()
        if not cfg.data_dir:
            print("Отменено.")
            raise SystemExit(1)

    if not cfg.preset:
        print("\nВыбери пресет (Enter = 2):")
        print("  1) qwen36m          35.8M, инициализация Qwen2.5-0.5B, ~3 часа")
        print("  2) night-scratch    22.5M с нуля, ночь на RX 6600 (рекомендую)")
        print("  3) night-scratch-pro 35.2M с нуля, если хватает VRAM")
        print("  4) custom           свои размеры")
        print("  5) smoke            2-минутная проверка")
        choice = str(_ask("Номер или имя", "2"))
        table = {"1": "qwen36m", "2": "night-scratch", "3": "night-scratch-pro",
                 "4": "custom", "5": "smoke"}
        name = table.get(choice, choice)
        if name in PRESETS:
            apply_preset(cfg, name)

    print("\n--- основные настройки ---")
    cfg.hours = float(_ask("Сколько часов учить", cfg.hours, float))
    default_out = cfg.out or f"runs/{cfg.preset or cfg.init}_{datetime.now():%Y%m%d_%H%M}"
    cfg.out = str(_ask("Папка рана", default_out))

    if _ask_yes("\nИзменить архитектуру / словарь / батч?", False):
        _choose_arch(cfg)

    print("\n" + "=" * 70)
    print("ИТОГО:")
    print(f"  пресет:      {cfg.preset or 'custom'}")
    print(f"  init:        {cfg.init}" + (" (слой Qwen)" if cfg.init == "qwen" else " (с нуля)"))
    print(f"  архитектура: d_model={cfg.d_model}, heads={cfg.n_heads}x{cfg.d_head}, "
          f"base_layers={cfg.base_layers}, depth={cfg.depth}, seq={cfg.max_seq_len}")
    print(f"  словарь:     {cfg.vocab_size} ({cfg.vocab_strategy})")
    print(f"  батч:        {cfg.batch_size} x grad_accum {cfg.grad_accum}, amp={cfg.amp}")
    print(f"  бюджет:      {cfg.hours:.2f} ч")
    print(f"  данные:      {cfg.data_dir}")
    print(f"  out:         {cfg.out}")
    print("=" * 70)
    if not _ask_yes("Запускаем обучение?", True):
        print("Отменено.")
        raise SystemExit(0)
    return cfg


def validate_config(cfg: Config):
    # Датасет указывается вручную всегда: путь обязателен для обучения/плана.
    if cfg.mode in ("train", "plan") and not cfg.data_dir and not cfg.train_file:
        raise ValueError(
            "Не указан путь к датасету. Задайте --data-dir ПАПКА (с train.jsonl/val.jsonl) "
            "или --train-file ФАЙЛ."
        )
    if cfg.init_skip_embeddings and cfg.init_extend_vocab:
        raise ValueError("Выберите только один режим переноса vocab: --init-skip-embeddings или --init-extend-vocab")
    if cfg.hours <= 0:
        raise ValueError("--hours должен быть > 0")
    if not (1 <= cfg.depth <= 64):
        raise ValueError("depth должен быть в пределах 1..64")
    if cfg.max_seq_len < 32:
        raise ValueError("max_seq_len слишком мал")
    if cfg.base_layers < 1:
        raise ValueError("base_layers должен быть >= 1")
    if cfg.init == "scratch":
        if cfg.n_heads * cfg.d_head != cfg.d_model:
            if cfg.d_model % cfg.d_head == 0:
                new_heads = cfg.d_model // cfg.d_head
                print(f"[!] n_heads={cfg.n_heads} не сходится: ставлю {new_heads}")
                cfg.n_heads = new_heads
            else:
                raise ValueError("n_heads * d_head должно равняться d_model")
        if cfg.n_heads % cfg.kv_heads != 0:
            raise ValueError("kv_heads должен делить n_heads")
    if cfg.vocab_size < 64:
        raise ValueError("vocab_size слишком мал")


# ======================================================================================
# Основной запуск
# ======================================================================================

def apply_resume(cfg: Config, args: argparse.Namespace) -> Config:
    run_dir, _ = resolve_ckpt(cfg.resume)
    payload_path = run_dir / "config.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"В {run_dir} нет config.json — не могу восстановить конфиг")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    valid = {f.name for f in fields(Config)}
    saved = Config(**{k: v for k, v in payload.get("config", {}).items() if k in valid})
    saved.mode = cfg.mode
    saved.resume = str(cfg.resume)
    saved.out = str(run_dir)
    saved.yes = cfg.yes
    saved.prompt = cfg.prompt
    saved.max_new = cfg.max_new
    saved.temperature = cfg.temperature
    saved.rep_penalty = cfg.rep_penalty
    saved.tools = cfg.tools
    if args.hours is not None or args.minutes is not None:
        saved.hours = cfg.hours
    if args.extra_hours is not None:
        saved.extra_hours = cfg.extra_hours
    if args.device is not None:
        saved.device = cfg.device
    if args.amp is not None:
        saved.amp = cfg.amp
    if args.max_steps is not None:
        saved.max_steps = cfg.max_steps
    # путь к датасету можно переопределить при resume (иначе берётся из конфига рана)
    if args.data_dir is not None:
        saved.data_dir = cfg.data_dir
    if args.train_file is not None:
        saved.train_file = cfg.train_file
    if args.val_file is not None:
        saved.val_file = cfg.val_file
    # рантайм-флаги, которые не имеют смысла в сохранённом конфиге
    if getattr(args, "no_demo", None):
        saved.demo = False
    if getattr(args, "demo", None):
        saved.demo = True
        saved.demo_force = True
    if getattr(args, "demo_max_new", None) is not None:
        saved.demo_max_new = args.demo_max_new
    if getattr(args, "no_samples", None):
        saved.no_samples = True
    if getattr(args, "eval_seconds", None) is not None:
        saved.eval_seconds = args.eval_seconds
    if getattr(args, "save_minutes", None) is not None:
        saved.save_minutes = args.save_minutes
    if getattr(args, "log_seconds", None) is not None:
        saved.log_seconds = args.log_seconds
    return saved


def train_main(cfg: Config) -> Dict[str, Any]:
    resume_dir = None
    if cfg.resume:
        resume_dir, _ = resolve_ckpt(cfg.resume)
    if resume_dir is not None:
        run_dir = Path(resume_dir)
    elif cfg.out:
        run_dir = Path(cfg.out)
    else:
        run_dir = Path("runs") / f"{cfg.preset or cfg.init}_{datetime.now():%Y%m%d_%H%M}"
    run_dir.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    if not cfg.resume and (run_dir / "config.json").exists():
        raise SystemExit(f"[!] {run_dir} уже содержит ран. Используйте --resume {run_dir} "
                         f"или другую папку --out")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    logger = TrainLogger(run_dir / "train.log")
    logger(BANNER if not cfg.resume else "")
    logger(f"[i] run dir: {run_dir}")
    try:
        tok_dir = run_dir / "tokenizer"
        if cfg.resume and (tok_dir / "tokenizer_config.json").exists():
            tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
            logger(f"[i] Токенизатор рана: {tok_dir}")
        else:
            tok = load_tokenizer(cfg, logger)
        data = prepare_datasets(cfg, tok, logger)

        # Для короткой persona-фазы можно считать частоты нового compact-словаря
        # по основному корпусу, не подмешивая его в саму оптимизацию.
        vocab_freq = data["freq"]
        if (not cfg.resume and cfg.vocab_data_dir
                and Path(cfg.vocab_data_dir).resolve() != Path(cfg.data_dir).resolve()):
            vocab_cfg = copy.copy(cfg)
            vocab_cfg.data_dir = cfg.vocab_data_dir
            vocab_cfg.train_file = ""
            vocab_cfg.val_file = ""
            vocab_cfg.max_samples = 0
            logger(f"[i] Частоты compact-словаря: {vocab_cfg.data_dir}")
            vocab_data = prepare_datasets(vocab_cfg, tok, logger)
            vocab_freq = vocab_data["freq"]

        # Защита от случайного «ночного» запуска на smoke-фикстурах: Qwen-архитектура
        # успеет прогнать крошечный корпус тысячи раз и просто его запомнит.
        if (not cfg.dry_run and not cfg.resume and cfg.init == "qwen" and cfg.hours >= 1
                and data["train_tokens"] < 1_000_000 and not cfg.allow_tiny_data):
            raise SystemExit(
                "[СТОП] Для Qwen-рана найдено только "
                f"{data['train_tokens']:,} train-токенов. Это похоже на smoke-фикстуры, "
                "а не на реальный корпус; 8-часовое обучение будет повторять их тысячи раз.\n"
                "Укажите настоящие --data-dir/--train-file (минимум ~1M токенов; "
                "рекомендуется 15–40M). Для намеренного короткого отладочного запуска "
                "добавьте --allow-tiny-data."
            )

        if cfg.resume:
            # resume не должен требовать Qwen: архитектура восстанавливается из config.json,
            # веса — из last.pt, словарь — из compact_vocab.pt
            saved_pack = run_dir / "compact_vocab.pt"
            if not saved_pack.exists():
                saved_pack = run_dir / "vocab_pack.pt"
            if not saved_pack.exists():
                raise FileNotFoundError(f"Нет compact_vocab.pt — не могу продолжить ран")
            payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            pack = torch.load(saved_pack, map_location="cpu", weights_only=False)
            pack["old_to_new"] = get_forward_map(pack, len(tok))
            pack["new_to_old"] = get_inverse_map(pack, int(pack["compact_weights"].shape[0]))
            meta = dict(payload["arch"])
            meta["vocab_size"] = int(pack["compact_weights"].shape[0])
            meta["d_model"] = int(pack["compact_weights"].shape[1])
            model = build_scaffold(cfg, meta, token_bytes=pack["token_bytes"])
            cfg.d_model, cfg.vocab_size = meta["d_model"], meta["vocab_size"]
        else:
            model, pack = build_model_and_pack(cfg, tok, vocab_freq, logger)
        if cfg.init_ckpt and not cfg.resume:
            ck_path = Path(cfg.init_ckpt)
            if not ck_path.exists():
                raise FileNotFoundError(f"Нет --init-ckpt: {ck_path}")
            sd = torch.load(ck_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
                sd = sd["model"]
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            own = model.state_dict()
            good, skipped = {}, []
            mapping_keys = ("embed_tokens.", "lm_head.", "token_bytes")
            prefix_embedding = None
            for k, v in sd.items():
                if cfg.init_skip_embeddings and k.startswith(mapping_keys):
                    skipped.append(k + " (новый vocab pack)")
                elif cfg.init_extend_vocab and k.startswith(mapping_keys):
                    # Расширенный pack сохраняет старые строки в точности на
                    # прежних ID, но полная матрица уже имеет больше строк.
                    # После load_state_dict перенесём только этот общий prefix.
                    if (k in ("embed_tokens.weight", "lm_head.weight") and prefix_embedding is None
                            and k in own and v.ndim == 2 and own[k].ndim == 2
                            and v.shape[1] == own[k].shape[1] and 0 < v.shape[0] < own[k].shape[0]):
                        prefix_embedding = v.detach().cpu()
                    skipped.append(k + " (расширенный vocab pack)")
                elif k in own and tuple(v.shape) == tuple(own[k].shape):
                    good[k] = v
                else:
                    skipped.append(k)
            model.load_state_dict(good, strict=False)
            # при увеличении depth переносим выученные depth-эмбеддинги/скейлы
            with torch.no_grad():
                ck_de = sd.get("factory.depth_emb.weight")
                if ck_de is not None and hasattr(model, "factory"):
                    tgt = model.factory.depth_emb.weight
                    if tgt.shape[0] > ck_de.shape[0] and tgt.shape[1] == ck_de.shape[1]:
                        n = min(ck_de.shape[0], tgt.shape[0])
                        tgt[:n].copy_(ck_de[:n].to(tgt.dtype))
                ck_ds = sd.get("depth_scale")
                if ck_ds is not None and model.depth_scale.shape[0] > ck_ds.shape[0]:
                    n = min(ck_ds.shape[0], model.depth_scale.shape[0])
                    model.depth_scale[:n].copy_(ck_ds[:n].to(model.depth_scale.dtype))
            copied = 0
            if prefix_embedding is not None:
                copied = min(prefix_embedding.shape[0], model.embed_tokens.weight.shape[0])
                with torch.no_grad():
                    model.embed_tokens.weight[:copied].copy_(
                        prefix_embedding[:copied].to(model.embed_tokens.weight.dtype))
                # lm_head weight-tied с embed_tokens.weight, отдельная копия не нужна.
            reason = " (включая vocab-зависимые)" if cfg.init_skip_embeddings else ""
            if copied:
                reason += f"; сохранён embedding-prefix {copied} строк"
            logger(f"[i] init-ckpt: загружено {len(good)} тензоров из {ck_path}"
                   + (f", пропущено {len(skipped)}{reason}" if skipped else reason))
        if cfg.compile:
            try:
                model = torch.compile(model)
                logger("[i] torch.compile включён")
            except Exception as exc:
                logger(f"[!] torch.compile не удался: {exc}")

        if not cfg.dry_run:
            if not (run_dir / "compact_vocab.pt").exists() and not (run_dir / "vocab_pack.pt").exists():
                torch.save(pack, run_dir / "compact_vocab.pt")
                logger(f"[i] compact_vocab.pt сохранён ({cfg.vocab_size} x {cfg.d_model})")
            if not cfg.resume:
                try:
                    tok.save_pretrained(run_dir / "tokenizer")
                except Exception as exc:
                    logger(f"[!] Не удалось сохранить токенизатор: {exc}")
                save_run_config(run_dir, cfg, model, pack)
            elif not (run_dir / "config.json").exists():
                save_run_config(run_dir, cfg, model, pack)

        result = train(cfg, tok, model, pack, data, run_dir, logger)
        if cfg.dry_run:
            return result

        # ---- финальное демо ----
        device = pick_device(cfg)
        model.eval()
        demo_lines = []
        n_params = count_parameters(model).get("TOTAL", 0)
        show_demo = cfg.demo
        if device.type == "cpu" and n_params > 20_000_000 and not cfg.demo_force:
            logger("[i] демо пропущено: CPU + большая модель (включить: --demo)")
            show_demo = False
        if show_demo:
            for q in BENCH_PROMPTS:
                logger(f"\n[demo] usr> {q}")
                try:
                    ans = generate_text(model, tok, pack, prompt_messages(q, cfg), cfg, device,
                                        allow_tools=cfg.tools, max_new=cfg.demo_max_new)
                except Exception as exc:
                    ans = f"<ошибка генерации: {exc}>"
                logger(f"[demo] agt> {ans}")
                demo_lines.append(f"usr> {q}\nagt> {ans}")
        demo_text = "\n\n".join(demo_lines) if demo_lines else "(демо отключено)"
        (run_dir / "demo.txt").write_text(demo_text, encoding="utf-8")

        history = []
        mpath = run_dir / "metrics.jsonl"
        if mpath.exists():
            for line in mpath.read_text(encoding="utf-8").splitlines():
                try:
                    history.append(json.loads(line))
                except Exception:
                    pass
        write_loss_svg(history, run_dir / "loss_curve.svg")
        write_run_report(run_dir, cfg, pack, data, result.get("best_val", float("nan")),
                         history, demo_text, result.get("steps", 0), result.get("elapsed", 0.0))
        logger(f"\n[i] Отчёт: {run_dir/'README.md'}")
        logger(f"[i] Чат:  python build_neuroset.py --mode chat --run {run_dir}")
        return result
    finally:
        logger.close()



# ======================================================================================
# DF-OPD: teacher-generated prompts + on-policy token/hidden distillation
# ======================================================================================

class DFHiddenProjector(nn.Module):
    """Временная training-only проекция h_student -> h_teacher."""
    def __init__(self, d_student: int, d_teacher: int):
        super().__init__()
        self.proj = nn.Linear(d_student, d_teacher, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.float())


def _df_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--teacher-device cuda, но CUDA/ROCm недоступна")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _df_dtype(device: torch.device, name: str) -> torch.dtype:
    if device.type == "cpu" or name == "fp32":
        return torch.float32
    return torch.bfloat16 if name == "bf16" else torch.float16


def _df_teacher_input(tok, system: str, user: str) -> torch.Tensor:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        x = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                    return_tensors="pt")
        return x if torch.is_tensor(x) else torch.tensor([x], dtype=torch.long)
    except Exception:
        return torch.tensor([tok.encode(build_prompt_text(messages), add_special_tokens=False)], dtype=torch.long)


def _df_clean_question(line: str) -> str:
    line = line.strip().strip("•-* ")
    line = re.sub(r"^\s*\d+\s*[.)、:-]\s*", "", line)
    return line.strip(" \t\"'")


def _df_questions(teacher, tok, device: torch.device, count: int, seed: int) -> List[str]:
    """DF: вопросы создает teacher без внешнего prompt corpus."""
    system = "Ты создаёшь учебные задания для русскоязычного ассистента. Не отвечай на задания."
    # Broad, compositional curriculum. Each batch is regenerated during DF-OPD;
    # it is not a fixed answer bank. Include dialogue repair and tool decisions.
    topics = ["повседневный разговор и small talk", "объяснение простых понятий",
              "арифметика и бытовые расчёты", "программирование и отладка",
              "наука и причинные объяснения", "история и география",
              "планирование при ограничениях", "русский язык и перефразирование",
              "уточнение неоднозначной просьбы", "диалог: пользователь поправляет ошибку ассистента",
              "когда уместно вызвать калькулятор", "когда нужен поиск, а когда ответить из общих знаний",
              "многоходовый разговор со сменой темы", "признание неопределённости и аккуратный ответ"]
    rng = random.Random(seed)
    out: List[str] = []
    seen = set()
    attempts = 0
    while len(out) < count and attempts < 20:
        attempts += 1
        topic = topics[(attempts - 1) % len(topics)]
        user = ("Придумай ровно 16 разных коротких запросов пользователя на русском языке по теме: "
                f"{topic}. Смешай обычные вопросы, просьбы объяснить, простую арифметику "
                "и информационные запросы. Верни только нумерованный список запросов, "
                "без ответов и комментариев.")
        ids = _df_teacher_input(tok, system, user).to(device)
        with torch.inference_mode():
            generated = teacher.generate(ids, max_new_tokens=900, do_sample=True, temperature=0.85,
                                         top_p=0.95, repetition_penalty=1.05,
                                         pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        text = tok.decode(generated[0, ids.size(1):], skip_special_tokens=True)
        for line in text.splitlines():
            q = _df_clean_question(line)
            key = " ".join(q.lower().split())
            if 6 <= len(q) <= 220 and key not in seen and not q.endswith(":"):
                seen.add(key)
                out.append(q)
                if len(out) >= count:
                    break
        rng.shuffle(topics)
    if len(out) < max(16, count // 2):
        raise RuntimeError(f"Teacher создал лишь {len(out)} пригодных вопросов из {count}")
    return out[:count]


def _df_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    probs = F.softmax(logits.float() / max(temperature, 1e-4), dim=-1)
    vals, idx = torch.sort(probs, descending=True)
    keep = torch.cumsum(vals, dim=-1) <= top_p
    keep[..., 0] = True
    vals = vals * keep
    vals = vals / vals.sum(dim=-1, keepdim=True)
    return int(idx.gather(-1, torch.multinomial(vals, 1)).item())


@torch.no_grad()
def _df_rollout(model, prompt: List[int], max_new: int, temperature: float,
                top_p: float, eos: set[int], device: torch.device) -> List[int]:
    cur = torch.tensor([prompt], dtype=torch.long, device=device)
    output: List[int] = []
    model.eval()
    for _ in range(max_new):
        nxt = _df_sample(model(cur)[:, -1, :], temperature, top_p)
        if nxt in eos and output:
            break
        output.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
    return output


def _df_copy_scaffold(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "config.json", target / "config.json")
    source_pack = source / "compact_vocab.pt"
    if not source_pack.exists():
        source_pack = source / "vocab_pack.pt"
    shutil.copy2(source_pack, target / "compact_vocab.pt")
    if (source / "tokenizer").exists():
        shutil.copytree(source / "tokenizer", target / "tokenizer")
    (target / "checkpoints").mkdir()


def _df_autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else _nullctx()



def teacher_main(cfg: Config) -> None:
    """Санитарная проверка teacher в ТОЧНОМ runtime system context до DF-OPD."""
    if "0.5b" in cfg.teacher.lower() or "0_5b" in cfg.teacher.lower():
        raise SystemExit("[СТОП] Qwen-0.5B нельзя использовать teacher.")
    device = _df_device(cfg.teacher_device)
    tok = AutoTokenizer.from_pretrained(cfg.teacher, use_fast=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher, torch_dtype=_df_dtype(device, cfg.teacher_dtype), device_map=None
    ).to(device).eval()
    prompts = [cfg.prompt] if cfg.prompt else [
        "привет бро", "расскажи коротко о Москве", "посчитай: 50 - 18", "поищи столицу Франции",
    ]
    system = tools_system_text()
    print("=" * 70)
    print(f"=== TEACHER CHECK: {cfg.teacher} ===")
    print("=" * 70)
    for user in prompts:
        ids = _df_teacher_input(tok, system, user).to(device)
        with torch.inference_mode():
            out = teacher.generate(ids, max_new_tokens=min(128, cfg.max_new), do_sample=False,
                                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        # Runtime проверяет протокол с skip_special_tokens=False. Печатаем тот же
        # raw-вариант: иначе специальный </tool_call> может ошибочно выглядеть пропавшим.
        answer = tok.decode(out[0, ids.size(1):], skip_special_tokens=False).strip()
        print(f"usr> {user}\nagt> {answer}\n")




def _teacher_curriculum_tasks(count: int, seed: int) -> List[Dict[str, str]]:
    """Logic-first tasks plus explicit research tasks for model-chosen web calls.

    Это не готовый chat-датасет. Код создаёт только жизненные ситуации и
    ограничения; teacher формирует содержание ответа, решает вызывать ли tool
    и выбирает его query. ``kind=web`` требует валидный web_search call.
    """
    rng = random.Random(seed)
    projects = ["небольшой Python-скрипт", "README проекта", "учебный план", "письмо клиенту",
                "резервное копирование", "список задач", "презентацию", "план релиза"]
    constraints = [
        "есть только один вечер", "важна надёжность", "нельзя усложнять решение",
        "нужен понятный первый шаг", "времени мало, но результат должен быть проверяемым",
        "нужно объяснить решение новичку"]
    bugs = [
        "программа читает файл и получает FileNotFoundError",
        "цикл неожиданно выполняется бесконечно",
        "функция возвращает None вместо ожидаемого значения",
        "после изменения API тесты перестали проходить",
        "в списке появляются дубликаты, хотя их быть не должно",
        "код работает локально, но падает в CI",
    ]
    decisions = [
        ("быстро сделать прототип", "сначала продумать архитектуру"),
        ("хранить настройки в одном файле", "разделить настройки по окружениям"),
        ("сразу оптимизировать код", "сначала измерить узкое место"),
        ("взять готовую библиотеку", "написать небольшую реализацию самому"),
        ("работать над несколькими задачами одновременно", "закончить одну важную задачу"),
    ]
    audiences = ["новичок", "небольшая команда", "один разработчик", "руководитель проекта",
                 "пользователь без технического опыта", "стажёр", "автор open-source проекта"]
    signals = ["срок сорван", "появились противоречивые требования", "тесты нестабильны",
               "пользователи жалуются на сложность", "данных для решения мало", "ошибка воспроизводится редко"]
    web_topics = [
        # столицы и страны
        "столица Франции", "столица Японии", "столица Бразилии", "столица Канады",
        "столица Австралии", "столица Египта", "столица Индии", "столица Аргентины",
        "столица Норвегии", "столица Португалии", "столица Казахстана", "столица Турции",
        # география
        "высота Эвереста", "глубина Марианской впадины", "река Нил", "озеро Байкал",
        "пустыня Сахара", "вулкан Везувий", "Гренландия", "Амазонка", "Каспийское море",
        "водопад Анхель", "пик Победы", "Килиманджаро", "Суэцкий канал", "Панамский канал",
        # история и персоналии
        "кто такой Пушкин", "кто такой Ломоносов", "кто такой Эйнштейн", "кто такой Кюри",
        "кто такой Дарвин", "кто такой Ньютон", "кто такой Тесла", "кто такой Гагарин",
        "Юрий Гагарин первый полёт в космос", "Пётр Первый", "Екатерина Вторая",
        "Александр Македонский", "Наполеон Бонапарт", "Леонардо да Винчи", "Микеланджело",
        "год падения Берлинской стены", "год полёта человека в космос", "дата высадки на Луну",
        "Первая мировая война причины", "Куликово поле битва", "Ледовое побоище",
        # наука и техника
        "что такое чёрная дыра", "скорость света", "таблица Менделеева", "ДНК молекула",
        "теория относительности", "квантовая запутанность", "фотосинтез",
        "как работает интернет", "протокол HTTP", "что такое Git", "язык программирования Python",
        "нейронная сеть", "квантовый компьютер", "Марс планета", "кольца Сатурна",
        "луна спутник Земли", "комета Галлея", "гравитация", "электричество открытие",
        "пенициллин открытие", "вакцина история", "телескоп Хаббл", "миссия Вояджер",
        # культура и литература
        "роман Война и мир автор", "кто написал Преступление и наказание",
        "кто написал Мёртвые души", "Евгений Онегин сюжет", "кто такой Достоевский",
        "кто такой Толстой Лев", "кто такой Чехов", "Шекспир Гамлет", "Мона Лиза картина",
        "Чёрный квадрат Малевича", "Сикстинская капелла", "Эрмитаж музей история",
        "Что где когда передача", "Балет Лебединое озеро", "опера Евгений Онегин",
        # животные и природа
        "кит синий размер", "дельфин афалина", "коала животное", "кенгуру",
        "белый медведь", "колибри птица", "сова полярная", "гепард скорость",
        "слон африканский", "крокодил нильский", "панда большая", "тигр амурский",
        # техника и быт
        "первый автомобиль история", "история телефона изобретение", "самолёт братьев Райт",
        "Титаник крушение", "метро Московское история", "телевидение изобретение",
        "фотография история изобретения", "криптовалюта биткоин", "искусственный интеллект",
        "первый компьютер ЭНИАК", "интернет история создания", "спутник первый запуск",
    ]
    research_angles = [
        "основной факт", "краткая история", "практический смысл", "контекст применения",
        "проверяемая дата", "связь с современной практикой", "ключевая характеристика",
        "причина известности", "одно важное уточнение", "географический или технический контекст",
        "что полезно запомнить", "как не перепутать с похожим понятием",
    ]
    web_forms = [
        # явные исследовательские инструкции (как раньше)
        "Поищи краткую проверяемую справку: {topic}; интересует {angle}. В query используй короткую основную тему, без пояснений. Вызови web_search и не отвечай до tool response.",
        "Найди актуальную краткую информацию: {topic}, особенно {angle}. Сначала и до tool response выведи только валидный web_search; короткий query по основной теме выбери сам.",
        "Проверь по источнику: {topic}; нужно уточнить {angle}. Вызови web_search с коротким ключевым запросом; финальный ответ дай только после tool response.",
        "Мне нужна внешняя справка о теме «{topic}», а именно {angle}. Сначала сделай web_search по основной теме без лишних слов и не пиши текст после вызова.",
        # естественные вопросы без команды «поищи»: учат строить query из вопроса
        "{qform} Интересует {angle}. Нужен свежий источник — вызови web_search с коротким query по теме и не отвечай до tool response.",
        "{qform} Сначала web_search по основной теме, потом короткий ответ по результату.",
        "{qform} Проверь это поиском: один web_search с коротким query, финальный ответ только после tool response.",
        "{qform} Мне нужно уточнение из источника. Сделай web_search с коротким запросом и жди tool response.",
    ]
    # Context varies the final synthesis need without dictating/replacing the
    # model-selected query. It makes the web curriculum scale beyond the 480
    # topic × angle × form combinations used by the small pilot.
    web_contexts = [
        "для краткой заметки", "для учебного конспекта", "для рабочего решения",
        "для ответа новичку", "для проверки факта в документе", "для карточки проекта",
        "для объяснения коллеге", "для небольшой презентации", "для памятки",
        "для планирования следующего шага", "для ответа клиенту", "для сравнения вариантов",
    ]
    greetings = ["привет бро", "добрый день", "ты тут?", "помоги, пожалуйста"]
    # Scales practical reasoning scenarios without inserting a canned answer.
    logic_contexts = [
        "решение должно быть проверяемым", "важно сначала собрать наблюдение",
        "нельзя делать лишних предположений", "нужен короткий следующий шаг",
        "результат должен быть понятен новичку", "ценна воспроизводимость",
        "ошибка может дорого стоить", "нужно зафиксировать основание решения",
    ]

    def normal_form(i: int, j: int) -> str:
        # i выбирает семейство задачи; j независимо перебирает его параметры.
        # Нельзя использовать i и для параметров: после i % 8 он давал почти
        # постоянный subject и ломал requested 60% logic quota.
        kind = i % 8
        if kind == 0:
            return greetings[j % len(greetings)]
        if kind == 1:
            return (f"Нужно подготовить {projects[j % len(projects)]} за {20 + j % 140} минут, но "
                    f"{constraints[(j // len(projects)) % len(constraints)]}. С чего начать и почему?")
        if kind == 2:
            return (f"Как логично разбирать ситуацию: {bugs[j % len(bugs)]}, причём она повторяется "
                    f"в {1 + j % 9} из 10 запусков? Дай порядок проверки.")
        if kind == 3:
            a, b = decisions[j % len(decisions)]
            return (f"Что разумнее: {a} или {b}, если цена ошибки — {1 + j % 5} условных баллов? "
                    "Назови критерии выбора и сделай вывод.")
        if kind == 4:
            return (f"{audiences[j % len(audiences)].capitalize()} сообщает, что "
                    f"{signals[(j // len(audiences)) % len(signals)]} после {2 + j % 48} изменений. "
                    "Какой минимальный следующий шаг поможет проверить гипотезу?")
        if kind == 5:
            return (f"Нужно выбрать между быстрым исправлением и более надёжным решением для "
                    f"{projects[j % len(projects)]}; откат займёт {5 + j % 55} минут. "
                    "Какие два-три критерия помогут принять решение без лишних предположений?")
        if kind == 6:
            return (f"У {audiences[j % len(audiences)]} есть {2 + j % 9} вариантов и мало контекста. "
                    "Как ответить так, чтобы сначала уточнить цель, а потом предложить следующий шаг?")
        return (f"В задаче «{projects[j % len(projects)]}» сбой может быть вызван неверными входными "
                f"данными или ошибкой алгоритма; на проверку есть {1 + j % 20} минут. "
                "Не называй причину без результата проверки. Предложи один минимальный следующий шаг: "
                "эталонный вход с заранее известным ожидаемым результатом и что нужно зафиксировать.")

    tasks: List[Dict[str, str]] = []
    target_counts = {"logic": max(1, int(round(count * 0.50))),
                     "web": max(1, int(round(count * 0.30)))}
    target_counts["calculate"] = count - target_counts["logic"] - target_counts["web"]
    made = Counter()
    seen = set()
    attempts = 0
    while len(tasks) < count and attempts < count * 80:
        attempts += 1
        # Quotas keep the corpus logic-first even when one prompt family has
        # more syntactic variations than the others.
        choices = [k for k, n in target_counts.items() if made[k] < n]
        if not choices:
            break
        kind = choices[rng.randrange(len(choices))]
        if kind == "web":
            topic = web_topics[rng.randrange(len(web_topics))]
            angle = research_angles[rng.randrange(len(research_angles))]
            m_person = re.match(r"^кто такой\s+(.+)$", topic, flags=re.IGNORECASE)
            if m_person:
                short = m_person.group(1)
                natural = [f"Кто такой {short}?", f"Расскажи, кто такой {short}.",
                           f"{short} — кто это и чем известен?", f"Какие главные факты о {short}?"]
            else:
                short = topic
                natural = [f"Что такое {topic}?", f"Расскажи коротко про {topic}.",
                           f"{topic}: что это такое?", f"Какие главные факты про {topic}?"]
            if rng.random() < 0.5:
                user = web_forms[rng.randrange(4)].format(topic=topic, angle=angle)
            else:
                qform = natural[rng.randrange(len(natural))]
                user = web_forms[4 + rng.randrange(len(web_forms) - 4)].format(qform=qform, angle=angle)
            user += " Контекст финального ответа: " + web_contexts[rng.randrange(len(web_contexts))] + "."
        elif kind == "calculate":
            a, b = rng.randint(12, 240), rng.randint(2, 120)
            variant = rng.randrange(5)
            if variant == 0:
                op = ["+", "-", "*"][rng.randrange(3)]
                user = (f"Вычисли {a} {op} {b}. Вызови calculate и не пиши финальный ответ "
                        "до tool response.")
            elif variant == 1:
                c = rng.randint(2, 30)
                user = (f"Посчитай точно: ({a} + {b}) * {c}. Сначала calculate, ответ только "
                        "после tool response.")
            elif variant == 2:
                c = rng.randint(2, 12)
                user = (f"Сколько будет {a} / {c}? Используй calculate и не отвечай до tool response.")
            elif variant == 3:
                c = rng.randint(2, 60)
                user = (f"Вычисли: {a} - {b} + {c}. Вызови calculate, финальный ответ после tool response.")
            else:
                x = rng.choice([4, 9, 16, 25, 36, 49, 64, 81, 100, 144, 225, 400, 625])
                user = (f"Чему равен sqrt из {x}? Сначала вызови calculate, потом ответь.")
        else:
            user = normal_form(rng.randrange(max(128, count * 5)), rng.randrange(max(128, count * 5)))
            if user not in greetings:
                user += " Контекст: " + logic_contexts[rng.randrange(len(logic_contexts))] + "."
                user += " Ответь кратко, по существу, не более 60 слов."
        key = kind + "\0" + " ".join(user.lower().split())
        if key not in seen:
            seen.add(key)
            made[kind] += 1
            tasks.append({"user": user, "kind": kind})
    if len(tasks) < count:
        raise RuntimeError(f"Curriculum создал лишь {len(tasks)} уникальных задач из {count}")
    return tasks


def _teacher_curriculum_prompts(count: int, seed: int) -> List[str]:
    """Совместимая text-only проекция curriculum для лёгких тестов."""
    return [x["user"] for x in _teacher_curriculum_tasks(count, seed)]


def _teacher_answer_quality(raw: str) -> Tuple[bool, str]:
    """Fail-closed фильтр: не исправляет teacher, только исключает плохую пару."""
    plain = raw.replace("<|im_end|>", " ").replace("<|endoftext|>", " ").strip()
    low = " ".join(plain.lower().split())
    if len(low) < 8:
        return False, "too_short"
    # Corpus is Russian. CJK fragments in a Russian target are a known Qwen
    # quantization/template failure mode, not content to distill into student.
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]", plain):
        return False, "wrong_language"
    refusal = (
        "извините", "к сожалению", "не могу", "не способен", "не умею", "не имею возможности",
        "не могу предоставить", "как языковая модель", "запрещенные вещества", "противоречит законам",
    )
    if any(x in low for x in refusal):
        return False, "refusal"
    # Repeated 3-grams are a cheap, deterministic detector of degeneration.
    words = re.findall(r"[\wё]+", low)
    if len(words) >= 12:
        grams = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        if grams and max(grams.values()) >= 3:
            return False, "repetition"
    if "<tool_call>" in raw:
        call_text = plain.replace("</tool_call>", "</tool_call>")
        if last_tool_call(call_text, end_of_turn=True) is None:
            return False, "invalid_tool"
    return True, "ok"


def _llamacpp_chat_completion(url: str, system: str, user: str, max_new: int,
                              stop: Optional[Sequence[str]] = None) -> Tuple[str, bool]:
    """Один request к локальному OpenAI-совместимому llama-server.

    Сервер возвращает уже завершённый assistant turn. Мы не дописываем EOT,
    tool-теги или JSON; `finish_reason=length` считается обрывом и отклоняется.
    """
    payload = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.0, "top_p": 1.0, "max_tokens": int(max_new), "stream": False,
    }
    if stop:
        payload["stop"] = list(stop)
    req = urllib.request.Request(
        url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"llama-server request failed: {exc}") from exc
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("llama-server returned no choices")
    choice = choices[0]
    message = choice.get("message", {})
    raw = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(raw, str):
        raw = ""
    return raw, str(choice.get("finish_reason", "stop")) != "length"



def _teacher_tool_stop(teacher_name: str) -> List[str]:
    """Output boundary for known chat templates; never edits model JSON.

    Qwen's ChatML teacher commonly ends a valid inline JSON call at its first
    newline and may otherwise continue with fabricated tool output. Llama-3
    Saiga emits its JSON *after* an initial newline, so it must stop only at
    the literal closing tag. Unknown teachers use the conservative latter mode.
    """
    if "qwen" in teacher_name.lower():
        return ["\n", "</tool_call>"]
    return ["</tool_call>"]

def teacher_data_main(cfg: Config) -> None:
    """Создаёт reviewable teacher corpus; student в этом режиме не загружается."""
    teacher_name = cfg.teacher_data_teacher
    if "0.5b" in teacher_name.lower() or "0_5b" in teacher_name.lower():
        raise SystemExit("[СТОП] Qwen-0.5B нельзя использовать teacher.")
    if cfg.teacher_data_samples < 32 or cfg.teacher_data_max_new < 16:
        raise SystemExit("--teacher-data-samples >= 32 и --teacher-data-max-new >= 16 обязательны")
    if not (0.5 <= cfg.teacher_data_min_accept <= 1.0):
        raise SystemExit("--teacher-data-min-accept должен быть в диапазоне 0.5..1.0")
    if not cfg.out:
        raise SystemExit("Для --mode teacher-data укажите --out DATA_DIR")
    out_dir = Path(cfg.out).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"[СТОП] {out_dir} уже не пуста; укажите новую --out. Ничего не перезаписываю.")
    backend = cfg.teacher_data_backend
    if backend not in ("transformers", "llamacpp"):
        raise SystemExit("--teacher-data-backend: transformers или llamacpp")
    if cfg.teacher_data_parallel < 1:
        raise SystemExit("--teacher-data-parallel должен быть >= 1")

    random.seed(cfg.teacher_data_seed)
    torch.manual_seed(cfg.teacher_data_seed)
    teacher = tok = None
    device = None
    if backend == "transformers":
        device = _df_device(cfg.teacher_device)
        if device.type != "cuda":
            raise SystemExit("[СТОП] teacher-data намеренно требует GPU; CPU teacher здесь не поддерживается.")
        if cfg.teacher_data_parallel != 1:
            raise SystemExit("transformers backend работает последовательно; для параллельной генерации используйте --teacher-data-backend llamacpp")
        print(f"[TEACHER-DATA] transformers teacher={teacher_name} device={device} dtype={cfg.teacher_dtype}")
        tok = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name, torch_dtype=_df_dtype(device, cfg.teacher_dtype), device_map=None
        ).to(device).eval()
        for par in teacher.parameters():
            par.requires_grad_(False)
    else:
        print(f"[TEACHER-DATA] llama-server={cfg.teacher_data_url}; label={teacher_name}; parallel={cfg.teacher_data_parallel}")

    # Extra candidates make rejection meaningful; no rejected answer is rewritten.
    requested = cfg.teacher_data_samples
    candidates = _teacher_curriculum_tasks(max(requested + 32, int(requested * 1.35)), cfg.teacher_data_seed)
    system = tools_system_text()
    eos = {IM_END, ENDOFTEXT}
    tool_stop = _teacher_tool_stop(teacher_name)
    accepted: List[Dict[str, str]] = []
    web_pending: List[Dict[str, Any]] = []
    rejected = Counter()
    rejected_preview: List[Dict[str, str]] = []

    def reject_example(user: str, raw: str, reason: str) -> None:
        rejected[reason] += 1
        if len(rejected_preview) < 32:
            rejected_preview.append({"user": user, "assistant": raw, "reason": reason})

    def accept_example(number: int, user: str, answer: str, kind: str) -> None:
        accepted.append({"user": user, "assistant": answer, "kind": kind})
        if number == 1 or number % 16 == 0 or len(accepted) >= requested:
            print(f"[TEACHER-DATA] accepted={len(accepted)}/{requested}; candidates={number}; rejected={sum(rejected.values())}")

    def transformer_completion(system_text: str, user_text: str) -> Tuple[str, bool]:
        assert teacher is not None and tok is not None and device is not None
        ids = _df_teacher_input(tok, system_text, user_text).to(device)
        with torch.inference_mode():
            out = teacher.generate(ids, max_new_tokens=cfg.teacher_data_max_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        answer_ids = [int(x) for x in out[0, ids.size(1):].detach().cpu().tolist()]
        return tok.decode(answer_ids, skip_special_tokens=False), bool(answer_ids) and answer_ids[-1] in eos

    def get_completion(system_text: str, user_text: str,
                       stop: Optional[Sequence[str]] = None) -> Tuple[str, bool]:
        if backend == "transformers":
            # Native transformers generation has no server stop API; this backend
            # remains sequential and is not the fast ROCm/GGUF recipe.
            return transformer_completion(system_text, user_text)
        return _llamacpp_chat_completion(cfg.teacher_data_url, system_text, user_text,
                                         cfg.teacher_data_max_new, stop=stop)

    def canonical_tool_block(raw: str) -> str:
        """Only adds the transcript delimiter; model JSON/arguments stay byte-for-byte."""
        clean = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        close = clean.find("</tool_call>")
        return clean[:close + len("</tool_call>")] if close >= 0 else clean + "</tool_call>"

    def tool_result_is_empty(result: str) -> bool:
        try:
            content = str(json.loads(result).get("content", "")).strip().lower()
        except Exception:
            content = ""
        return not content or content == "не нашёл" or content.startswith("ошибка")

    def process_initial(number: int, task: Dict[str, str], raw: str, complete: bool) -> None:
        user, kind = task["user"], task["kind"]
        if not complete:
            reject_example(user, raw, "truncated")
            return
        clean_raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        if kind not in ("web", "calculate"):
            if "<tool_call>" in raw:
                # Logic targets must be self-contained. Tool curriculum is below.
                reject_example(user, raw, "unexpected_tool")
                return
            good, why = _teacher_answer_quality(raw)
            if not good:
                reject_example(user, raw, why)
                return
            accept_example(number, user, raw, "logic")
            return

        # Research/calculate task: tool name and exact arguments are selected by teacher.
        call = last_tool_call(clean_raw, end_of_turn=True)
        if call is None:
            reject_example(user, raw, "missing_" + kind + "_tool")
            return
        name, args = call
        expected_name = "web_search" if kind == "web" else "calculate"
        if name != expected_name:
            reject_example(user, raw, "wrong_" + kind + "_tool")
            return
        # Preserve the teacher JSON exactly; only serialize the already valid
        # finished call into the training transcript grammar.
        web_pending.append({"number": number, "user": user, "kind": kind, "name": name, "args": args,
                            "tool_block": canonical_tool_block(raw)})

    if backend == "transformers":
        for number, task in enumerate(candidates, 1):
            raw, complete = get_completion(
                system, task["user"], stop=tool_stop if task["kind"] in ("web", "calculate") else None,
            )
            process_initial(number, task, raw, complete)
    else:
        # llama-server keeps quantized weights once and batches independent
        # first-turn generations across --parallel slots.
        failures = 0
        with ThreadPoolExecutor(max_workers=cfg.teacher_data_parallel) as pool:
            pending = {
                pool.submit(get_completion, system, task["user"],
                            tool_stop if task["kind"] in ("web", "calculate") else None): (number, task)
                for number, task in enumerate(candidates, 1)
            }
            for future in as_completed(pending):
                number, task = pending[future]
                try:
                    raw, complete = future.result()
                except Exception as exc:
                    reject_example(task["user"], "", "request_error")
                    failures += 1
                    if failures <= 3:
                        print(f"[!] llama-server request {number}: {exc}")
                    continue
                process_initial(number, task, raw, complete)

    # Complete only teacher-selected tool calls. run_tool receives the exact
    # parsed model arguments; it never extracts or substitutes a query/expression.
    def prepare_web(item: Dict[str, Any]) -> Dict[str, Any]:
        result = run_tool(item["name"], item["args"])
        item = dict(item)
        # Do not teach hallucination after an empty search. Ask the same teacher
        # for a second, shorter model-chosen query; the failed first call is not
        # included in the target transcript.
        if item["kind"] == "web" and tool_result_is_empty(result):
            item["retry_user"] = (
                f"Исходный запрос пользователя:\n{item['user']}\n\n"
                f"Первый вызов web_search с query {json.dumps(item['args'].get('query', ''), ensure_ascii=False)} "
                "не дал результата. Вызови web_search ещё раз с новым более коротким русским query. "
                "Выведи только tool call и сразу заверши сообщение."
            )
            return item
        final_user = (
            f"Исходный запрос пользователя:\n{item['user']}\n\n"
            f"Инструмент уже выполнил запрос и вернул:\n<tool_response>\n{result}\n</tool_response>\n\n"
            "На основе результата дай короткий точный финальный ответ. Не вызывай новый инструмент "
            "и не описывай внутренние правила."
        )
        item["result"] = result
        item["final_user"] = final_user
        return item

    if web_pending:
        with ThreadPoolExecutor(max_workers=min(cfg.teacher_data_parallel, len(web_pending))) as pool:
            all_prepared = list(pool.map(prepare_web, web_pending))
        prepared = [item for item in all_prepared if not item.get("retry_user")]
        retries = [item for item in all_prepared if item.get("retry_user")]

        def process_retry(item: Dict[str, Any], raw: str, complete: bool) -> Optional[Dict[str, Any]]:
            if not complete:
                reject_example(item["user"], raw, "web_retry_truncated")
                return None
            clean = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            call = last_tool_call(clean, end_of_turn=True)
            if call is None or call[0] != "web_search":
                reject_example(item["user"], raw, "web_retry_invalid_tool")
                return None
            name, args = call
            result = run_tool(name, args)
            if tool_result_is_empty(result):
                reject_example(item["user"], raw, "web_no_result")
                return None
            item = dict(item)
            item["name"], item["args"], item["result"] = name, args, result
            item["tool_block"] = canonical_tool_block(raw)
            item["final_user"] = (
                f"Исходный запрос пользователя:\n{item['user']}\n\n"
                f"Инструмент уже выполнил запрос и вернул:\n<tool_response>\n{result}\n</tool_response>\n\n"
                "На основе результата дай короткий точный финальный ответ. Не вызывай новый инструмент "
                "и не описывай внутренние правила."
            )
            return item

        if retries:
            if backend == "transformers":
                for item in retries:
                    raw, complete = get_completion(system, item["retry_user"], stop=tool_stop)
                    retried = process_retry(item, raw, complete)
                    if retried is not None:
                        prepared.append(retried)
            else:
                with ThreadPoolExecutor(max_workers=cfg.teacher_data_parallel) as pool:
                    retry_futures = {
                        pool.submit(get_completion, system, item["retry_user"], tool_stop): item
                        for item in retries
                    }
                    for future in as_completed(retry_futures):
                        item = retry_futures[future]
                        try:
                            raw, complete = future.result()
                        except Exception as exc:
                            reject_example(item["user"], "", "web_retry_request_error")
                            print(f"[!] llama-server retry {item['number']}: {exc}")
                            continue
                        retried = process_retry(item, raw, complete)
                        if retried is not None:
                            prepared.append(retried)

        final_system = (system + "\nИнструмент уже выполнен и его результат передан в сообщении пользователя. "
                        "Сформулируй только финальный ответ по этому результату.")
        evidence_system = (
            system + "\nПеред финальным ответом проверь достаточность фактического tool response. "
            "Если он прямо содержит нужный для исходного запроса факт, выведи ровно "
            "<evidence_sufficient> и ничего больше. Если факта недостаточно, сам вызови "
            "web_search с новым коротким русским query и выведи только валидный tool call. "
            "Не угадывай факт, не пиши финальный ответ и не описывай проверку."
        )

        def evidence_user(item: Dict[str, Any]) -> str:
            return (
                f"Исходный запрос пользователя:\n{item['user']}\n\n"
                f"web_search уже вернул:\n<tool_response>\n{item['result']}\n</tool_response>\n\n"
                "Достаточно ли этого источника именно для ответа на исходный запрос?"
            )

        def validate_web_evidence(item: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str, str]:
            """Teacher, not code, decides whether retrieved evidence answers the task.

            A chosen retry query is parsed/executed under the same strict protocol;
            only the final sufficient call survives into the student transcript.
            """
            current = dict(item)
            last_raw = ""
            for _ in range(2):
                try:
                    raw, complete = get_completion(evidence_system, evidence_user(current), stop=tool_stop)
                except Exception as exc:
                    return None, "web_evidence_request_error", str(exc)
                last_raw = raw
                if not complete:
                    return None, "web_evidence_truncated", raw
                clean = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                if clean == "<evidence_sufficient>":
                    return current, "", ""
                call = last_tool_call(clean, end_of_turn=True)
                if call is None or call[0] != "web_search":
                    return None, "web_evidence_invalid", raw
                name, args = call
                result = run_tool(name, args)
                if tool_result_is_empty(result):
                    detail = json.dumps({"query": args.get("query", ""), "result": result}, ensure_ascii=False)
                    return None, "web_evidence_no_result", detail
                # The prior insufficient call stays out of the target trajectory.
                current = dict(current)
                current["name"], current["args"], current["result"] = name, args, result
                current["tool_block"] = canonical_tool_block(raw)
                current["final_user"] = (
                    f"Исходный запрос пользователя:\n{current['user']}\n\n"
                    f"Инструмент уже выполнил запрос и вернул:\n<tool_response>\n{result}\n</tool_response>\n\n"
                    "На основе результата дай короткий точный финальный ответ. Не вызывай новый инструмент "
                    "и не описывай внутренние правила."
                )
            return None, "web_evidence_retry_exhausted", last_raw

        # A nonempty result can still be irrelevant (for example a generic lead
        # for a request about a date). The teacher gets one model-driven chance
        # to replace it before final synthesis; code never derives the query.
        web_ready = [item for item in prepared if item.get("kind") == "web"]
        other_ready = [item for item in prepared if item.get("kind") != "web"]
        validated_web: List[Dict[str, Any]] = []
        if backend == "transformers":
            for item in web_ready:
                validated, reason, detail = validate_web_evidence(item)
                if validated is None:
                    reject_example(item["user"], detail, reason)
                else:
                    validated_web.append(validated)
        else:
            with ThreadPoolExecutor(max_workers=cfg.teacher_data_parallel) as pool:
                checks = {pool.submit(validate_web_evidence, item): item for item in web_ready}
                for future in as_completed(checks):
                    item = checks[future]
                    try:
                        validated, reason, detail = future.result()
                    except Exception as exc:
                        reject_example(item["user"], str(exc), "web_evidence_request_error")
                        continue
                    if validated is None:
                        reject_example(item["user"], detail, reason)
                    else:
                        validated_web.append(validated)
        prepared = other_ready + validated_web

        def process_final(item: Dict[str, Any], raw: str, complete: bool) -> None:
            if not complete:
                reject_example(item["user"], raw, "web_final_truncated")
                return
            if "<tool_call>" in raw:
                reject_example(item["user"], raw, "web_final_tool")
                return
            good, why = _teacher_answer_quality(raw)
            if not good:
                reject_example(item["user"], raw, "web_final_" + why)
                return
            trajectory = (item["tool_block"] + "\n<tool_response>\n" + item["result"] +
                          "\n</tool_response>\n" + raw.strip())
            accept_example(item["number"], item["user"], trajectory, str(item["kind"]))

        if backend == "transformers":
            for item in prepared:
                raw, complete = get_completion(final_system, item["final_user"])
                process_final(item, raw, complete)
        else:
            with ThreadPoolExecutor(max_workers=cfg.teacher_data_parallel) as pool:
                finals = {pool.submit(get_completion, final_system, item["final_user"]): item for item in prepared}
                for future in as_completed(finals):
                    item = finals[future]
                    try:
                        raw, complete = future.result()
                    except Exception as exc:
                        reject_example(item["user"], "", "web_final_request_error")
                        print(f"[!] llama-server final request {item['number']}: {exc}")
                        continue
                    process_final(item, raw, complete)

    if teacher is not None:
        del teacher
        torch.cuda.empty_cache()
    # Keep both research and exact-calculation trajectories as real minorities
    # rather than letting faster logic requests crowd them out by completion order.
    web_rows = [row for row in accepted if row.get("kind") == "web"]
    calc_rows = [row for row in accepted if row.get("kind") == "calculate"]
    logic_rows = [row for row in accepted if row.get("kind") == "logic"]
    required_web = max(4, requested // 10)
    required_calc = max(4, requested // 10)
    selected_web = web_rows[:max(required_web, requested // 5)]
    selected_calc = calc_rows[:max(required_calc, requested // 5)]
    selected_logic = logic_rows[:max(0, requested - len(selected_web) - len(selected_calc))]
    accepted = selected_logic + selected_web + selected_calc
    if len(accepted) < requested:
        remainder = web_rows[len(selected_web):] + calc_rows[len(selected_calc):] + logic_rows[len(selected_logic):]
        accepted += remainder[:requested - len(accepted)]
    if len(web_rows) < required_web:
        rejected["insufficient_web"] += required_web - len(web_rows)
    if len(calc_rows) < required_calc:
        rejected["insufficient_calculate"] += required_calc - len(calc_rows)

    acceptance = len(accepted) / max(1, len(accepted) + sum(rejected.values()))
    gate_failed = len(accepted) < requested or acceptance < cfg.teacher_data_min_accept
    accepted_override = False
    if gate_failed and cfg.teacher_data_use_accepted and len(accepted) >= requested:
        # Explicit operator choice: corpus still contains only rows that passed
        # all per-row checks. We record the aggregate shortfall; never inject a
        # rejected row and never change the default fail-closed policy.
        accepted_override = True
        print(
            f"[!] Явный --teacher-data-use-accepted: сохраняю {len(accepted)} уже accepted rows "
            f"при raw rate={acceptance:.1%}; rejected rows не будут использованы."
        )
    if gate_failed and not accepted_override:
        diag_dir = out_dir.with_name(out_dir.name + ".failed")
        if diag_dir.exists() and any(diag_dir.iterdir()):
            diag_dir = out_dir.with_name(out_dir.name + f".failed_{datetime.now():%Y%m%d_%H%M%S}")
        diag_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "teacher": teacher_name, "backend": backend, "requested": requested,
            "accepted": len(accepted), "acceptance_rate": round(acceptance, 6),
            "rejected": dict(rejected), "max_new": cfg.teacher_data_max_new,
        }
        (diag_dir / "teacher_data_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (diag_dir / "teacher_data_rejected.jsonl").open("w", encoding="utf-8") as f:
            for row in rejected_preview:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"Teacher quality gate failed: accepted={len(accepted)}/{requested}, "
            f"rate={acceptance:.1%}, rejected={dict(rejected)}. Student training не начато. "
            f"Диагностика: {diag_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.teacher_data_seed)
    rng.shuffle(accepted)
    preview_rows: List[Dict[str, str]] = []
    for kind, n_show in (("logic", 4), ("web", 4), ("calculate", 4)):
        preview_rows.extend([row for row in accepted if row.get("kind") == kind][:n_show])
    val_n = max(16, min(256, len(accepted) // 20))
    val, train = accepted[:val_n], accepted[val_n:]
    def write_rows(path: Path, rows: List[Dict[str, str]], split: str) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                item = {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": row["user"]},
                        {"role": "assistant", "content": row["assistant"]},
                    ],
                    "format": "chat", "source": f"teacher-data/{teacher_name}/{split}/{row.get('kind', 'logic')}",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_rows(out_dir / "train.jsonl", train, "train")
    write_rows(out_dir / "val.jsonl", val, "val")
    report = {
        "teacher": teacher_name, "backend": backend, "parallel": cfg.teacher_data_parallel,
        "requested": requested, "accepted": len(accepted),
        "acceptance_rate": round(acceptance, 6), "rejected": dict(rejected),
        "quality_gate": {
            "threshold": cfg.teacher_data_min_accept,
            "passed": not gate_failed,
            "explicit_filtered_override": accepted_override,
            "note": ("Сохранены только per-row accepted rows по явному флагу; "
                     "aggregate rate ниже default threshold." if accepted_override else ""),
        },
        "by_kind": dict(Counter(str(row.get("kind", "logic")) for row in accepted)), 
        "max_new": cfg.teacher_data_max_new, "seed": cfg.teacher_data_seed,
        "protocol": "teacher-selected tool name/query; raw JSON unchanged; no runtime repair",
    }
    (out_dir / "teacher_data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Readable samples are intentionally stored for review before any student run.
    with (out_dir / "teacher_data_preview.md").open("w", encoding="utf-8") as f:
        f.write(f"# Teacher data preview\n\nTeacher: `{teacher_name}`\n\n")
        for i, row in enumerate(preview_rows, 1):
            f.write(f"## {i}. User ({row.get('kind', 'logic')})\n\n{row['user']}\n\n## Assistant\n\n{row['assistant']}\n\n")
    print(f"\n[УСПЕХ] teacher-data: train={len(train)}, val={len(val)}, rate={acceptance:.1%} -> {out_dir}")
    print(f"[i] Сначала проверьте {out_dir/'teacher_data_preview.md'} и несколько JSONL строк. Student ещё не обучался.")


def teacher_probe_main(cfg: Config) -> None:
    """Проверяет нового local GGUF teacher до запуска дорогой сборки corpus.

    Probe не создаёт обучающих строк и не меняет student. Он проверяет русский
    ответ, model-selected calculate/web call и final answer после реального
    tool response. Это намеренно отдельный gate: смена teacher не должна
    начинаться сразу с 128 сэмплов.
    """
    if not cfg.out:
        raise SystemExit("Для --mode teacher-probe укажите --out ПАПКА_ПРОВЕРКИ")
    if cfg.teacher_data_backend != "llamacpp":
        raise SystemExit("teacher-probe требует --teacher-data-backend llamacpp и local llama-server")
    if cfg.teacher_data_parallel < 1:
        raise SystemExit("--teacher-data-parallel должен быть >= 1")
    if cfg.teacher_data_max_new < 32:
        raise SystemExit("Для teacher-probe укажите --teacher-data-max-new >= 32")
    out_dir = Path(cfg.out).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"[СТОП] {out_dir} уже не пуста; укажите новую --out. Ничего не перезаписываю.")
    out_dir.mkdir(parents=True, exist_ok=False)

    wanted_probe = {"logic": 2, "web": 6, "calculate": 2}
    by_kind: Dict[str, List[Dict[str, str]]] = {"logic": [], "web": [], "calculate": []}
    for task in _teacher_curriculum_tasks(64, cfg.teacher_data_seed):
        if len(by_kind[task["kind"]]) < wanted_probe[task["kind"]]:
            by_kind[task["kind"]].append(task)
    tests = by_kind["logic"] + by_kind["web"] + by_kind["calculate"]
    if len(tests) != sum(wanted_probe.values()):
        raise RuntimeError("Внутренняя ошибка: probe не собрал нужные curriculum-задачи")

    system = tools_system_text()
    tool_stop = _teacher_tool_stop(cfg.teacher_data_teacher)
    final_system = (system + "\nИнструмент уже выполнен и его результат передан в сообщении пользователя. "
                    "Сформулируй только финальный ответ по этому результату.")
    evidence_system = (
        system + "\nПеред финальным ответом проверь достаточность фактического tool response. "
        "Если он прямо содержит нужный для исходного запроса факт, выведи ровно "
        "<evidence_sufficient> и ничего больше. Если факта недостаточно, сам вызови "
        "web_search с новым коротким русским query и выведи только валидный tool call. "
        "Не угадывай факт, не пиши финальный ответ и не описывай проверку."
    )

    def complete(system_text: str, user_text: str, stop: Optional[Sequence[str]] = None) -> Tuple[str, bool]:
        return _llamacpp_chat_completion(cfg.teacher_data_url, system_text, user_text,
                                         cfg.teacher_data_max_new, stop=stop)

    def empty_result(result: str) -> bool:
        try:
            content = str(json.loads(result).get("content", "")).strip().lower()
        except Exception:
            content = ""
        return not content or content == "не нашёл" or content.startswith("ошибка")

    def probe_evidence_user(user: str, result: str) -> str:
        return (
            f"Исходный запрос пользователя:\n{user}\n\n"
            f"web_search уже вернул:\n<tool_response>\n{result}\n</tool_response>\n\n"
            "Достаточно ли этого источника именно для ответа на исходный запрос?"
        )

    def validate_probe_web(user: str, name: str, args: Dict[str, Any], result: str):
        current_name, current_args, current_result = name, args, result
        checks: List[str] = []
        for _ in range(2):
            try:
                raw, complete_turn = complete(
                    evidence_system, probe_evidence_user(user, current_result), tool_stop
                )
            except Exception as exc:
                return None, checks, "evidence_request_error", str(exc)
            checks.append(raw)
            if not complete_turn:
                return None, checks, "evidence_truncated", raw
            clean = raw.replace("<|im_end|>", " ").replace("<|endoftext|>", "").strip()
            if clean == "<evidence_sufficient>":
                return (current_name, current_args, current_result), checks, "", ""
            call = last_tool_call(clean, end_of_turn=True)
            if call is None or call[0] != "web_search":
                return None, checks, "evidence_invalid", raw
            current_name, current_args = call
            current_result = run_tool(current_name, current_args)
            if empty_result(current_result):
                detail = json.dumps({"query": current_args.get("query", ""), "result": current_result}, ensure_ascii=False)
                return None, checks, "evidence_no_result", detail
        return None, checks, "evidence_retry_exhausted", checks[-1] if checks else ""

    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(cfg.teacher_data_parallel, len(tests))) as pool:
        futures = {
            pool.submit(complete, system, task["user"], tool_stop
                        if task["kind"] in ("web", "calculate") else None): task
            for task in tests
        }
        for future in as_completed(futures):
            task = futures[future]
            record: Dict[str, Any] = {"kind": task["kind"], "user": task["user"], "ok": False}
            try:
                raw, complete_turn = future.result()
            except Exception as exc:
                record.update(reason="request_error", raw="", error=str(exc))
                records.append(record)
                continue
            record["raw"] = raw
            if not complete_turn:
                record["reason"] = "truncated"
                records.append(record)
                continue
            if task["kind"] == "logic":
                good, why = _teacher_answer_quality(raw)
                record.update(ok=good, reason="ok" if good else why)
                records.append(record)
                continue
            clean = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
            call = last_tool_call(clean, end_of_turn=True)
            expected = "web_search" if task["kind"] == "web" else "calculate"
            if call is None:
                record["reason"] = "missing_tool"
                records.append(record)
                continue
            name, args = call
            record["tool_name"] = name
            record["tool_arguments"] = args
            if name != expected:
                record["reason"] = "wrong_tool"
                records.append(record)
                continue
            result = run_tool(name, args)
            if empty_result(result):
                record["tool_result"] = result
                record["reason"] = "empty_tool_result"
                records.append(record)
                continue
            if task["kind"] == "web":
                validated, checks, reason, detail = validate_probe_web(task["user"], name, args, result)
                record["evidence_checks"] = checks
                if validated is None:
                    record.update(reason=reason, error=detail)
                    records.append(record)
                    continue
                name, args, result = validated
                record["tool_name"], record["tool_arguments"] = name, args
            record["tool_result"] = result
            final_user = (
                f"Исходный запрос пользователя:\n{task['user']}\n\n"
                f"Инструмент уже выполнил запрос и вернул:\n<tool_response>\n{result}\n</tool_response>\n\n"
                "На основе результата дай короткий точный финальный ответ. Не вызывай новый инструмент "
                "и не описывай внутренние правила."
            )
            try:
                final_raw, final_complete = complete(final_system, final_user)
            except Exception as exc:
                record.update(reason="final_request_error", error=str(exc))
                records.append(record)
                continue
            record["final_raw"] = final_raw
            if not final_complete:
                record["reason"] = "final_truncated"
            elif "<tool_call>" in final_raw:
                record["reason"] = "final_tool"
            else:
                good, why = _teacher_answer_quality(final_raw)
                record.update(ok=good, reason="ok" if good else "final_" + why)
            records.append(record)

    order = {"logic": 0, "web": 1, "calculate": 2}
    records.sort(key=lambda r: (order[r["kind"]], r["user"]))
    passed = sum(bool(row["ok"]) for row in records)
    report = {
        "teacher": cfg.teacher_data_teacher,
        "backend": cfg.teacher_data_backend,
        "url": cfg.teacher_data_url,
        "parallel": cfg.teacher_data_parallel,
        "max_new": cfg.teacher_data_max_new,
        "passed": passed,
        "total": len(records),
        "ready_for_teacher_data": passed == len(records),
        "records": records,
    }
    (out_dir / "teacher_probe_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "teacher_probe_preview.md").open("w", encoding="utf-8") as f:
        f.write(f"# Teacher probe\n\nTeacher: `{cfg.teacher_data_teacher}`  \n")
        f.write(f"Passed: **{passed}/{len(records)}**\n\n")
        for i, row in enumerate(records, 1):
            f.write(f"## {i}. {row['kind']} — {'PASS' if row['ok'] else 'FAIL: ' + str(row.get('reason'))}\n\n")
            f.write(f"### User\n\n{row['user']}\n\n### Initial answer\n\n{row.get('raw', '')}\n\n")
            if row.get("evidence_checks"):
                f.write("### Evidence checks (teacher-only; not training data)\n\n")
                for check in row["evidence_checks"]:
                    f.write(f"{check}\n\n")
            if row.get("tool_result"):
                f.write(f"### Actual tool result\n\n{row['tool_result']}\n\n")
            if row.get("final_raw"):
                f.write(f"### Final answer\n\n{row['final_raw']}\n\n")
            if row.get("error"):
                f.write(f"### Error\n\n{row['error']}\n\n")
    if passed != len(records):
        raise RuntimeError(
            f"Teacher probe failed: {passed}/{len(records)}. Student training и corpus generation не начаты. "
            f"Проверьте {out_dir/'teacher_probe_preview.md'}"
        )
    print(f"[УСПЕХ] Teacher probe: {passed}/{len(records)} -> {out_dir}")
    print("[i] Можно запускать только reviewable --mode teacher-data; student всё ещё не обучался.")


def _bridge_bad_teacher_answer(raw: str) -> bool:
    """Не переносим в student явные refusal-штампы, которые уже испортили DF-пилоты."""
    plain = raw.replace("<|im_end|>", " ").replace("<|endoftext|>", " ").strip().lower()
    if len(plain) < 2:
        return True
    bad = ("не могу помочь", "не могу ответить", "не имею возможности", "как языковая модель")
    return any(x in plain for x in bad)


def bridge_main(cfg: Config) -> None:
    """Teacher-generated hard-target warm-up перед любым on-policy DF.

    Teacher сначала создаёт фиксированные ответы в новом universal system context
    и полностью выгружается с GPU. Затем student учится только CE на этих ответах.
    Это не меняет compact-архитектуру и не применяет KL к собственным rollout.
    """
    if not cfg.student_run:
        raise SystemExit("Для --mode bridge укажите --student-run RUN")
    if "0.5b" in cfg.teacher.lower() or "0_5b" in cfg.teacher.lower():
        raise SystemExit("[СТОП] Qwen-0.5B нельзя использовать teacher.")
    if cfg.bridge_samples < 16 or cfg.bridge_steps <= 0:
        raise SystemExit("--bridge-samples >= 16 и --bridge-steps > 0 обязательны")
    if cfg.bridge_batch_size < 1 or cfg.bridge_max_new < 8:
        raise SystemExit("--bridge-batch-size >= 1 и --bridge-max-new >= 8 обязательны")
    if not cfg.out:
        raise SystemExit("Для --mode bridge укажите новую --out RUN")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    student_device = pick_device(cfg)
    teacher_device = _df_device(cfg.teacher_device)
    source = Path(cfg.student_run).expanduser().resolve()
    target = Path(cfg.out).expanduser().resolve()
    if target.exists():
        raise SystemExit(f"[СТОП] {target} уже существует; укажите новую --out")
    source_dir, payload, source_cfg = load_run(str(source))
    tok_dir = source_dir / "tokenizer"
    student_tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True) if tok_dir.exists() else load_tokenizer(source_cfg)
    pack_path = source_dir / "compact_vocab.pt"
    if not pack_path.exists():
        pack_path = source_dir / "vocab_pack.pt"
    source_pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    old_to_new = get_forward_map(source_pack, len(student_tok))

    # Phase 1: only the frozen teacher occupies GPU. Answers are fixed before
    # student training; the student never learns from its own degenerate rollout.
    teacher_tok = AutoTokenizer.from_pretrained(cfg.teacher, use_fast=True)
    probe = "Проверка token alignment: русский текст 2 + 2."
    if student_tok.encode(probe, add_special_tokens=False) != teacher_tok.encode(probe, add_special_tokens=False):
        raise RuntimeError("Tokenizers teacher/student несовместимы; bridge требует общий Qwen tokenizer.")
    print(f"[BRIDGE] frozen teacher={cfg.teacher} device={teacher_device} dtype={cfg.teacher_dtype}")
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher, torch_dtype=_df_dtype(teacher_device, cfg.teacher_dtype), device_map=None
    ).to(teacher_device).eval()
    for par in teacher.parameters():
        par.requires_grad_(False)
    questions = _df_questions(teacher, teacher_tok, teacher_device, cfg.bridge_samples, cfg.seed)
    system = tools_system_text()
    records: List[Dict[str, Any]] = []
    skipped = 0
    eos_old = {IM_END, ENDOFTEXT}
    for i, question in enumerate(questions, 1):
        ids = _df_teacher_input(teacher_tok, system, question).to(teacher_device)
        with torch.inference_mode():
            out = teacher.generate(ids, max_new_tokens=cfg.bridge_max_new, do_sample=False,
                                   pad_token_id=teacher_tok.eos_token_id, eos_token_id=teacher_tok.eos_token_id)
        answer_old = [int(x) for x in out[0, ids.size(1):].detach().cpu().tolist()]
        raw = teacher_tok.decode(answer_old, skip_special_tokens=False)
        # A max_new cutoff can leave an arbitrary partial answer; do not teach it.
        if not answer_old or answer_old[-1] not in eos_old or _bridge_bad_teacher_answer(raw):
            skipped += 1
            continue
        raw_for_protocol = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        if "<tool_call>" in raw and last_tool_call(raw_for_protocol, end_of_turn=True) is None:
            skipped += 1
            continue
        prompt_old = student_tok.encode(build_prompt_text([
            {"role": "system", "content": system}, {"role": "user", "content": question},
        ]), add_special_tokens=False)
        if len(prompt_old) + len(answer_old) > int(source_cfg.max_seq_len):
            skipped += 1
            continue
        if any(x < 0 or x >= len(old_to_new) for x in answer_old):
            skipped += 1
            continue
        records.append({"user": question, "assistant": raw, "prompt_old": prompt_old, "answer_old": answer_old})
        if i == 1 or i % 16 == 0 or i == len(questions):
            print(f"[BRIDGE] teacher pairs {len(records)}/{i}, skipped={skipped}")
    del teacher
    if teacher_device.type == "cuda":
        torch.cuda.empty_cache()
    if len(records) < max(16, min(cfg.bridge_samples // 2, 64)):
        raise RuntimeError(f"Teacher дал лишь {len(records)} пригодных ответов из {len(questions)}; bridge отменён")

    _df_copy_scaffold(source_dir, target)
    with (target / "teacher_bridge_pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"user": r["user"], "assistant": r["assistant"]}, ensure_ascii=False) + "\n")
    (target / "bridge_config.json").write_text(json.dumps({
        "teacher": cfg.teacher, "samples_requested": cfg.bridge_samples, "pairs_kept": len(records),
        "steps": cfg.bridge_steps, "batch_size": cfg.bridge_batch_size, "max_new": cfg.bridge_max_new,
        "lr": cfg.bridge_lr, "lr_base": cfg.bridge_lr_base, "source": str(source),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Phase 2: student-only CE. No teacher logits, hidden projection, or
    # on-policy student rollout is present in this phase.
    student, loaded_tok, pack = load_model_from_run(source_dir, payload, source_cfg, student_device)
    student.train()
    old_to_new = get_forward_map(pack, len(loaded_tok))
    examples: List[Tuple[List[int], List[int]]] = []
    for r in records:
        old = r["prompt_old"] + r["answer_old"]
        x = [int(old_to_new[t]) for t in old]
        y = [-100] * len(r["prompt_old"]) + x[len(r["prompt_old"]):]
        examples.append((x, y))
    rng = random.Random(cfg.seed)
    opt = make_optimizer(student, source_cfg)
    opt.param_groups[0]["lr"] = cfg.bridge_lr_base
    for group in opt.param_groups[1:]:
        group["lr"] = cfg.bridge_lr
    scaler = torch.amp.GradScaler("cuda", enabled=(student_device.type == "cuda"))
    metrics = target / "bridge_metrics.jsonl"
    start_time, best = time.time(), float("inf")

    def next_batch() -> Tuple[torch.Tensor, torch.Tensor]:
        chosen = [examples[rng.randrange(len(examples))] for _ in range(cfg.bridge_batch_size)]
        n = max(len(x) for x, _ in chosen)
        bx = torch.zeros((len(chosen), n), dtype=torch.long, device=student_device)
        by = torch.full((len(chosen), n), -100, dtype=torch.long, device=student_device)
        for j, (x, y) in enumerate(chosen):
            bx[j, :len(x)] = torch.tensor(x, dtype=torch.long, device=student_device)
            by[j, :len(y)] = torch.tensor(y, dtype=torch.long, device=student_device)
        return bx, by

    for step in range(1, cfg.bridge_steps + 1):
        x, y = next_batch()
        with _df_autocast(student_device):
            ce = _batch_loss(student, x, y)
        if not torch.isfinite(ce):
            print(f"[!] bridge step {step}: non-finite CE, пропуск")
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(ce).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        val = float(ce.detach())
        with metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "ce": round(val, 6), "seconds": round(time.time() - start_time, 2)},
                               ensure_ascii=False) + "\n")
        if val < best:
            best = val
            save_model_state(student, target / "checkpoints" / "best.pt", torch.float16)
        if step == 1 or step % 20 == 0 or step == cfg.bridge_steps:
            print(f"[BRIDGE {step:>4}/{cfg.bridge_steps}] ce={val:.4f} | {(time.time()-start_time)/60:.1f} min")
        if step % max(1, cfg.bridge_save_every) == 0:
            save_model_state(student, target / "checkpoints" / "last_model.pt", torch.float16)
        del x, y

    save_model_state(student, target / "checkpoints" / "final.pt", torch.float16)
    (target / "bridge_report.md").write_text(
        f"# Teacher-generated bridge\n\n- source: `{source}`\n- teacher: `{cfg.teacher}`\n"
        f"- generated pairs: {len(records)} of {len(questions)} questions (skipped {skipped})\n"
        f"- CE steps: {cfg.bridge_steps}; batch: {cfg.bridge_batch_size}\n- best CE: {best:.6f}\n\n"
        "Frozen teacher generated fixed targets first, then was released. Student training used response-only CE; "
        "there was no on-policy rollout, hidden loss, or output repair.\n",
        encoding="utf-8")
    print(f"\n[УСПЕХ] BRIDGE: {cfg.bridge_steps} CE-шагов за {(time.time()-start_time)/60:.1f} мин")
    print(f"[i] Сначала проверьте чат: python build_neuroset.py --mode chat --run {target} --tools")


def distill_main(cfg: Config) -> None:
    """White-box DF-OPD. Teacher frozen; Compact architecture не меняется."""
    if not cfg.student_run:
        raise SystemExit("Для --mode distill укажите --student-run RUN")
    if "0.5b" in cfg.teacher.lower() or "0_5b" in cfg.teacher.lower():
        raise SystemExit("[СТОП] Qwen-0.5B нельзя использовать teacher. Укажите модель >=1.5B.")
    if cfg.df_steps <= 0 or cfg.df_questions < 8:
        raise SystemExit("--df-steps > 0 и --df-questions >= 8 обязательны")
    if not cfg.out:
        raise SystemExit("Для --mode distill укажите новую --out RUN")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    student_device = pick_device(cfg)
    teacher_device = _df_device(cfg.teacher_device)
    source = Path(cfg.student_run).expanduser().resolve()
    target = Path(cfg.out).expanduser().resolve()
    if target.exists():
        raise SystemExit(f"[СТОП] {target} уже существует; укажите новую --out")

    source_dir, payload, source_cfg = load_run(str(source))
    student, student_tok, pack = load_model_from_run(source_dir, payload, source_cfg, student_device)
    student.train()
    old_to_new = get_forward_map(pack, len(student_tok))
    new_to_old = get_inverse_map(pack, int(pack["compact_weights"].shape[0]))
    new_to_old = torch.as_tensor(new_to_old, dtype=torch.long).cpu()
    kept_old = new_to_old.tolist()

    print(f"[DF-OPD] student={source_dir} V={len(kept_old):,} device={student_device}")
    print(f"[DF-OPD] frozen teacher={cfg.teacher} device={teacher_device} dtype={cfg.teacher_dtype}")
    teacher_tok = AutoTokenizer.from_pretrained(cfg.teacher, use_fast=True)
    probe = "Проверка token alignment: русский текст 2 + 2."
    if student_tok.encode(probe, add_special_tokens=False) != teacher_tok.encode(probe, add_special_tokens=False):
        raise RuntimeError("Tokenizers teacher/student несовместимы; DF-KL требует общий Qwen tokenizer.")
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher, torch_dtype=_df_dtype(teacher_device, cfg.teacher_dtype),
                                                    device_map=None).to(teacher_device).eval()
    for par in teacher.parameters():
        par.requires_grad_(False)

    # A rolling teacher-generated curriculum: regenerate prompts instead of cycling
    # the same finite list all night. Seed varies per block, and all batches are logged.
    questions = _df_questions(teacher, teacher_tok, teacher_device, cfg.df_questions, cfg.seed)
    question_block = 0
    _df_copy_scaffold(source_dir, target)
    question_log = target / "df_opd_questions.jsonl"
    with question_log.open("w", encoding="utf-8") as qf:
        for q in questions:
            qf.write(json.dumps({"block": question_block, "question": q}, ensure_ascii=False) + "\n")

    with torch.inference_mode():
        test = teacher(torch.tensor([[teacher_tok.encode("тест", add_special_tokens=False)[0]]], device=teacher_device),
                       output_hidden_states=True, use_cache=False)
        teacher_dim = int(test.hidden_states[-1].shape[-1])
    del test
    projector = DFHiddenProjector(student.D_model, teacher_dim).to(student_device)
    opt = make_optimizer(student, source_cfg)
    opt.param_groups[0]["lr"] = cfg.df_lr_base
    for group in opt.param_groups[1:]:
        group["lr"] = cfg.df_lr
    opt.add_param_group({"params": projector.parameters(), "lr": cfg.df_lr,
                         "weight_decay": 0.0, "name": "df_hidden_projector"})
    scaler = torch.amp.GradScaler("cuda", enabled=(student_device.type == "cuda"))

    # Small, short anchor batches: teacher lives on GPU too, so anchor never uses
    # the original batch=4/seq=512 that caused the prior OOM.
    anchor_iter = None
    if cfg.df_anchor_dir and cfg.df_anchor_weight > 0:
        anchor_cfg = Config(**{k: v for k, v in vars(source_cfg).items() if k in {f.name for f in fields(Config)}})
        anchor_cfg.data_dir = cfg.df_anchor_dir
        anchor_cfg.train_file = anchor_cfg.val_file = ""
        anchor_cfg.max_samples = 8000
        anchor_cfg.max_tokens = 2_000_000
        anchor_cfg.max_seq_len = min(256, int(source_cfg.max_seq_len))
        anchor_data = prepare_datasets(anchor_cfg, student_tok, print)
        anchor_data["train"].set_map(old_to_new)
        anchor_loader = DataLoader(anchor_data["train"], batch_size=1, shuffle=True,
                                   collate_fn=collate_fn, pin_memory=(student_device.type == "cuda"))
        anchor_iter = iter(anchor_loader)
        print(f"[DF-OPD] anchor={cfg.df_anchor_dir} weight={cfg.df_anchor_weight} batch=1 seq={anchor_cfg.max_seq_len}")

    eos = {int(old_to_new[x]) for x in (IM_END, ENDOFTEXT) if 0 <= x < len(old_to_new)}
    # Full old-token -> compact-token map. В KL нельзя просто взять logits
    # kept-токенов: вероятность всех отброшенных BPE должна суммироваться в их
    # compact fallback, иначе teacher превращается в неверное условное
    # распределение на малом подмножестве vocabulary.
    teacher_vocab = int(getattr(teacher.config, "vocab_size", len(old_to_new)))
    # У Qwen2.5-1.5B config может резервировать несколько сотен tail-ID,
    # которых нет в fast tokenizer student (и которые не возникают при его
    # tokenization). Distill только общее tokenizer-пространство и
    # перенормируем mass; несуществующие для student tail-ID не учим.
    common_vocab = min(len(old_to_new), teacher_vocab)
    if common_vocab < 1000:
        raise RuntimeError("Слишком маленькое пересечение teacher/student vocabulary")
    if teacher_vocab != len(old_to_new):
        print(f"[DF-OPD] tokenizer overlap: {common_vocab:,} из teacher V={teacher_vocab:,}; "
              "reserved tail-ID исключены из KL")
    old_to_compact_teacher = torch.as_tensor(old_to_new[:common_vocab], dtype=torch.long,
                                             device=teacher_device)
    if int(old_to_compact_teacher.max()) >= int(pack["compact_weights"].shape[0]):
        raise RuntimeError("old_to_new содержит ID вне compact vocabulary")
    system = tools_system_text()  # всегда один и тот же system context
    max_prompt = max(32, int(source_cfg.max_seq_len) - cfg.df_max_new - 1)
    metrics = target / "df_opd_metrics.jsonl"
    start_time, best = time.time(), float("inf")

    for step in range(1, cfg.df_steps + 1):
        # Refresh the prompt distribution every full pass. This prevents the
        # finite initial batch from becoming the student's entire curriculum.
        if step > 1 and (step - 1) % len(questions) == 0:
            question_block += 1
            questions = _df_questions(teacher, teacher_tok, teacher_device,
                                      cfg.df_questions, cfg.seed + question_block * 1009)
            with question_log.open("a", encoding="utf-8") as qf:
                for q in questions:
                    qf.write(json.dumps({"block": question_block, "question": q}, ensure_ascii=False) + "\n")
            print(f"[DF-OPD] curriculum refresh: block={question_block}, prompts={len(questions)}")
        question = questions[(step - 1) % len(questions)]
        prompt_old = student_tok.encode(build_prompt_text([
            {"role": "system", "content": system}, {"role": "user", "content": question},
        ]), add_special_tokens=False)
        if len(prompt_old) > max_prompt:
            raise RuntimeError(f"System+question={len(prompt_old)} tokens; доступно {max_prompt}. "
                               "Уменьшите --df-max-new, не обрезая system prompt.")
        prompt_new = [int(old_to_new[x]) for x in prompt_old]
        rollout = _df_rollout(student, prompt_new, cfg.df_max_new, cfg.df_rollout_temp,
                               cfg.df_top_p, eos, student_device)
        if not rollout:
            continue
        full_new = prompt_new + rollout
        full_old = prompt_old + [int(new_to_old[x]) for x in rollout]
        student_ids = torch.tensor([full_new], dtype=torch.long, device=student_device)
        teacher_ids = torch.tensor([full_old], dtype=torch.long, device=teacher_device)
        pos0, pos1 = len(prompt_new) - 1, len(prompt_new) - 1 + len(rollout)
        T = max(0.2, cfg.df_kd_temperature)

        # Сначала строим teacher distribution над ВСЕМ его vocabulary, затем
        # scatter-add в compact ID. Простое index_select(kept tokens) неверно:
        # оно теряет массу выкинутых BPE и обучает student ложному distribution.
        with torch.inference_mode():
            tout = teacher(teacher_ids, output_hidden_states=True, use_cache=False)
            # Сначала full softmax, затем берём совместимое tokenizer-пространство
            # и нормируем его заново: Qwen teacher config содержит reserved tail-ID.
            full_probs = F.softmax(tout.logits[:, pos0:pos1, :].float() / T, dim=-1)[..., :common_vocab]
            full_probs = full_probs / full_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            target_prob = torch.zeros((1, pos1 - pos0, int(pack["compact_weights"].shape[0])),
                                      dtype=torch.float32, device=teacher_device)
            scatter_idx = old_to_compact_teacher.view(1, 1, -1).expand(1, pos1 - pos0, -1)
            target_prob.scatter_add_(2, scatter_idx, full_probs)
            target_prob = target_prob.to(student_device)
            t_hidden = tout.hidden_states[-1][:, pos0:pos1, :].float().to(student_device)
        del tout, full_probs, teacher_ids
        if student_device.type == "cuda" and teacher_device == student_device:
            torch.cuda.empty_cache()

        student.train()
        with _df_autocast(student_device):
            s_logits, s_hidden = student(student_ids, return_hidden=True)
        s_logits = s_logits[:, :-1, :].float()
        s_hidden = s_hidden[:, :-1, :]
        target_prob = target_prob.clamp_min(1e-12)
        # reduction="batchmean" делит лишь на batch, но не на число rollout
        # токенов. Для B=1 это раздувает KL примерно в L раз и быстро разрушает
        # student. Усредняем token-KL по позициям явно.
        kd = F.kl_div(F.log_softmax(s_logits[:, pos0:pos1, :] / T, dim=-1), target_prob,
                      reduction="none").sum(dim=-1).mean() * (T * T)
        hidden = 1.0 - F.cosine_similarity(
            F.normalize(projector(s_hidden[:, pos0:pos1, :]), dim=-1),
            F.normalize(t_hidden, dim=-1), dim=-1,
        ).mean()

        anchor = torch.zeros((), device=student_device)
        # Маленький anchor на каждом шаге держит базовый русский язык, пока
        # on-policy rollout дрейфует по teacher-траекториям.
        if anchor_iter is not None and cfg.df_anchor_weight > 0:
            try:
                ax, ay = next(anchor_iter)
            except StopIteration:
                anchor_iter = iter(anchor_loader)
                ax, ay = next(anchor_iter)
            ax, ay = ax.to(student_device, non_blocking=True), ay.to(student_device, non_blocking=True)
            with _df_autocast(student_device):
                anchor = _batch_loss(student, ax, ay)
        loss = kd + cfg.df_hidden_weight * hidden + cfg.df_anchor_weight * anchor
        if not torch.isfinite(loss):
            print(f"[!] step {step}: non-finite loss, пропуск")
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(projector.parameters()), 1.0)
        scaler.step(opt)
        scaler.update()
        del target_prob, t_hidden, s_logits, s_hidden, student_ids

        record = {"step": step, "question": question, "loss": round(float(loss.detach()), 6),
                  "kd": round(float(kd.detach()), 6), "hidden": round(float(hidden.detach()), 6),
                  "anchor": round(float(anchor.detach()), 6), "seconds": round(time.time() - start_time, 2)}
        with metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if float(loss) < best:
            best = float(loss)
            save_model_state(student, target / "checkpoints" / "best.pt", torch.float16)
        if step == 1 or step % 20 == 0 or step == cfg.df_steps:
            print(f"[{step:>4}/{cfg.df_steps}] loss={float(loss):.4f} kd={float(kd):.4f} "
                  f"hidden={float(hidden):.4f} anchor={float(anchor):.4f} "
                  f"| {(time.time()-start_time)/60:.1f} min")
        if step % max(1, cfg.df_save_every) == 0:
            save_model_state(student, target / "checkpoints" / "last_model.pt", torch.float16)
            torch.save({"projector": projector.state_dict(), "optimizer": opt.state_dict(), "step": step},
                       target / "checkpoints" / "df_opd_state.pt")

    save_model_state(student, target / "checkpoints" / "final.pt", torch.float16)
    torch.save(projector.state_dict(), target / "checkpoints" / "hidden_projector.pt")
    (target / "df_opd_report.md").write_text(
        f"# DF-OPD run\n\n- student: `{source}`\n- teacher: `{cfg.teacher}`\n"
        f"- rolling teacher-generated curriculum; final batch size: {len(questions)}\n- steps: {cfg.df_steps}\n"
        f"- best loss: {best:.6f}\n- hidden weight: {cfg.df_hidden_weight}\n"
        f"- anchor weight: {cfg.df_anchor_weight}\n\n"
        "Teacher frozen. hidden_projector.pt нужен только для дальнейшей distillation, не для chat.\n",
        encoding="utf-8")
    print(f"\n[УСПЕХ] DF-OPD: {cfg.df_steps} шагов за {(time.time()-start_time)/60:.1f} мин")
    print(f"[i] Чат: python build_neuroset.py --mode chat --run {target} --tools")


def main():
    args = parser().parse_args()
    cfg = config_from_args(args)

    if cfg.mode in ("chat", "bench"):
        if not cfg.run and not cfg.resume:
            cfg.run = str(_ask("Папка рана (например runs/night)", ""))
        if not cfg.run and not cfg.resume:
            print("Укажите --run ПАПКА")
            raise SystemExit(1)
        if cfg.mode == "chat":
            chat_main(cfg)
        else:
            bench_main(cfg)
        return

    if cfg.mode == "teacher":
        teacher_main(cfg)
        return

    if cfg.mode == "teacher-probe":
        teacher_probe_main(cfg)
        return

    if cfg.mode == "teacher-data":
        teacher_data_main(cfg)
        return

    if cfg.mode == "bridge":
        bridge_main(cfg)
        return

    if cfg.mode == "distill":
        # DF-OPD имеет свой цикл и использует --student-run как исходную модель.
        distill_main(cfg)
        return

    if cfg.resume:
        cfg = apply_resume(cfg, args)
        if not cfg.yes and sys.stdin.isatty():
            extra = _ask("Сколько ещё часов учить (0 = до конца исходного бюджета)",
                         cfg.extra_hours, float)
            cfg.extra_hours = float(extra or 0)
    elif cfg.mode == "train" and not cfg.yes and sys.stdin.isatty():
        cfg = interactive_wizard(cfg)
    elif not cfg.yes and not sys.stdin.isatty():
        print("[i] Неинтерактивный запуск: использую значения по умолчанию. "
              "Добавьте --yes, чтобы убрать это сообщение.")

    validate_config(cfg)
    if cfg.mode == "plan":
        cfg.dry_run = True
        print("[i] Режим plan: 10 шагов, замер скорости, без сохранения модели")
    train_main(cfg)


if __name__ == "__main__":
    main()
