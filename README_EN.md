# Grimoire — Compact Dual Dynamic LLM

[![Model on HF](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/embrelgrim/grimoire)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

An experimental, compact Russian-language model featuring a recurrent weight factory and a tool-calling protocol for `calculate` / `web_search`.
The core architecture implementation, training pipeline, and interactive chat interface are consolidated into a single file, `build_neuroset.py`; evaluation scripts are organized separately.

**Status:** Research prototype, not a reliable general-purpose assistant.
The model can generate tool calls, but it may select malformed queries, hallucinate or distort retrieved search results, and fail to address the original prompt. The presence of a source in the context window does not guarantee the factual accuracy of the final answer.

> **Evaluation Methodology Correction.** Previously reported BPB values for our model and scores such as `8/8`, `76/91`, and `77/91` do not substantiate the performance claims made in earlier iterations of this README. BPB was computed with future lookahead leakage through the factory and lossy token mapping; response verification previously checked for target facts across the entire execution trace, including raw tool outputs. `bench_results.json` is preserved as an archival artifact with appropriate caveats, not as an active benchmark ranking. Re-run results are presented below; they do not substantiate prior claims of semantic accuracy or architectural superiority.

## Model Summary

| Property | Value |
|---|---|
| Primary Checkpoint | [`embrelgrim/grimoire`](https://huggingface.co/embrelgrim/grimoire), local path `runs/assistant8` |
| Parameters | 35,849,352 |
| Dimensions | `d_model=896`, 14 heads × 64 |
| Base Block | Layer index 6 from `Qwen/Qwen2.5-0.5B-Instruct`, GQA with 2 KV heads |
| Recurrent Depth | 8 steps reusing the base layer |
| Compact Vocabulary | 11,264 tokens, tied embeddings / LM head |
| Target Training Set | 65,490 samples / 40,030,505 tokens |
| Reported Target Training Time | ~8 hours on an 8 GB AMD Radeon RX 6600; excludes Qwen pretraining |
| Larger Experimental Variant | `assistant_big6`, approximately 129M parameters |

Dataset size and training duration reflect the original experimental run rather than an independent reproduction. This project builds upon pretrained Qwen components; it is not an end-to-end model trained from scratch in eight hours. The previously cited 99.84% vocabulary coverage applies strictly to the vocabulary construction corpus, not across arbitrary inputs.

## Quickstart

Install PyTorch for your specific compute platform, followed by the required dependencies. On AMD hardware, a compatible PyTorch/ROCm environment is required; standard `torch` wheels do not guarantee support for consumer GPUs. Although the script sets `HSA_OVERRIDE_GFX_VERSION`, this does not replace proper driver and ROCm runtime compatibility.

```bash
pip install transformers numpy huggingface_hub
hf download embrelgrim/grimoire \
  --local-dir runs/assistant8 \
  --include "checkpoints/*.pt" "compact_vocab.pt" "config.json" "tokenizer/*"

python build_neuroset.py --mode chat --run runs/assistant8
python build_neuroset.py --mode chat --run runs/assistant8 \
  --prompt "сколько будет 47 * 13?"
python build_neuroset.py --mode bench --run runs/assistant8
```

Execute all commands from the repository root. Compatibility across arbitrary combinations of `torch` and `transformers` versions requires verification; a pinned environment configuration is not yet published. For reproducible runs, record your `pip freeze`, ROCm version, GPU architecture, Git commit hash, and checkpoint revision.

Load `.pt` files only from trusted sources: current model loaders use `torch.load(..., weights_only=False)` and can execute arbitrary Python deserialization payloads.

## Architecture

1. Compact token embeddings are summed with `0.1 × byte-morph patcher` representations (derived from the first 8 bytes of each token).
2. The weight factory takes the hidden context, the recurrent GRU depth state, and a depth-step embedding as inputs, generating per-head attention and MLP weights on the fly.
3. At each recurrence step, both the shared Qwen base layer and the dynamically generated block are computed, followed by a gated residual update.
4. Output hidden states pass through RMSNorm and the tied LM head.

`W_o` and `W_out` in the generated block are initialized to zero. This applies strictly to the generated branch, and does not imply that the complete recurrent step behaves as an identity operation.

Increasing the recurrent depth adds computation, but does not inherently guarantee improved reasoning capabilities. Core weights remain shared across iterations. At `d_model=896`, each additional recurrent step introduces 897 parameters: a depth embedding and a scalar gate. Computational cost and KV-cache overhead scale linearly with depth.

### Parameter Breakdown

| Component | Parameters |
|---|---:|
| `base_layer` | 14,912,384 |
| `embed_tokens`, tied with `lm_head` | 10,092,544 |
| `factory`, including depth embeddings | 10,682,368 |
| `morph_patcher` | 158,464 |
| `gen_layer` RMSNorm | 1,792 |
| `g_init` | 896 |
| `depth_scale` | 8 |
| `norm` | 896 |
| **Total** | **35,849,352** |

### Causality and KV Cache

During target token training and generation, the factory operates with an anchor (`anchor_lens`) to prevent lookahead access to future response tokens. When an external tool completes, a new segment initializes with an updated anchor. Calling `model(x)` directly without an anchor does not provide a valid autoregressive likelihood over all positions of `x`, as the factory can observe downstream tokens.

`prefill_cached` and `step_cached` routines are implemented. Bitwise equivalence with an un-cached full forward pass requires environment-specific verification. In particular, configurations using `factory_prefix>0` need explicit testing, as cached and uncached forward passes handle prefix boundaries differently. The cache implementation and network architecture remain unchanged in this revision.

### Compact Vocabulary Limitations

Unmapped source tokens are projected to their longest matching dictionary prefix or ID 0, rather than an invertible byte sequence. This induces information loss, particularly on rare lexical items, named entities, and out-of-distribution inputs. The byte-morph patcher cannot recover token information destroyed during this mapping step.

The published checkpoint preserves compatibility with this dictionary. Migrating to an invertible subword or byte-level tokenizer represents a separate task that may require full retraining.

## Data and Training

The repository does not bundle automated dataset preparation pipelines. You must supply a directory containing `train.jsonl` and `val.jsonl`, or pass explicit paths via `--train-file` / `--val-file`.

```bash
# Example training run (reference only, not an exact reproduction recipe)
python build_neuroset.py --preset qwen36m --data-dir YOUR_DATA_DIR \
  --out runs/base --hours 5 --max-seq-len 768 \
  --lr 2e-4 --lr-base 4e-5 --warmup 400 --yes

# Continuation using an exported compact vocabulary
python build_neuroset.py --preset qwen36m --data-dir YOUR_NEXT_DATA_DIR \
  --out runs/next --init-ckpt runs/base/checkpoints/best.pt \
  --vocab-pack runs/base/compact_vocab.pt \
  --hours 1 --max-seq-len 768 --lr 4e-5 --lr-base 1e-5 --warmup 200 --yes
```

Available presets include `qwen36m`, `night-scratch`, `night-scratch-pro`, and `smoke`. The previously reported validation loss of ~2.55 is an internal metric tied to this specific training protocol, not a cross-model comparable Bits-Per-Byte (BPB) figure.

## Chat and Tool Execution

Multi-turn dialogue history is not preserved by default; the model is not trained for robust multi-turn conversations. Experimental context stitching can be enabled via:

```bash
python build_neuroset.py --mode chat --run runs/assistant8 --chat-context
```

- `calculate`: Evaluates arithmetic expressions using an isolated AST parser without Python's native `eval`. Correct tool execution does not guarantee that the model formulated the right expression or accurately conveyed the calculation in its final reply.
- `web_search`: Queries a local Wikipedia store if present, falling back to live network retrieval. The previously described local `kb/` store is not bundled with this repository; absent this corpus, retrieval behavior may diverge from initial experiments.

The model emits tool calls formatted as JSON; the runtime environment parses and executes supported calls. Grounded synthesis remains fragile. For example, in an archival run querying the capital of France, the retrieval tool correctly returned Paris, but the model generated a response claiming France was "located in Germany." This constitutes a hallucination error, not an accurate grounded response.

## Evaluation: Revisions and Unproven Claims

### Archival Results

`bench_results.json` contains historical outputs flagged as `legacy_invalid_for_quality_comparison`. They must not be interpreted as an objective ranking:

- BPB calculations for our models contained future leakage via the factory context and frontier anchors.
- Lossy token mapping evaluated truncated token streams while normalization denominators used raw source byte counts.
- Instruction following and tool benchmark scores credited target facts found inside raw `<tool_response>` blocks.
- Inference latency was originally described as FP16, whereas published loaders default to float32.

Rerun measurements are detailed below.

### Rerun Benchmark Results

The following results were captured from console logs after repairing the evaluation harness. This run does not constitute an independent third-party reproduction. Evaluation environment: AMD Radeon RX 6600, checkpoint `runs/assistant8/checkpoints/final.pt`. Exact dependency versions and weight checksums were not recorded in this trace.

**Executed:** 6 unit tests, `eval_det.py`, `eval_ood.py`, `bench_suite.py instr`, and `bench_suite.py speed`. All 6 unit tests passed; these tests validate harness functionality, not model accuracy.

#### Fixed Tool Responses (Oracle Context) — `eval_det.py`

| Test | Result | Verification Scope |
|---|---:|---|
| WEB query | 27/39 | Target string marker present in `web_search` arguments |
| WEB answer | 22/39 | Fact marker identified in final generated text, excluding tool trace |
| CALC | 7/8 | Expected numerical token present in final output |
| CALIB | 5/5 | No spurious tool calls on 5 conversational control prompts; basic output sanity checks |
| Lexical overlap | 0.42 | Mean ratio of evaluated response tokens present in reference tool output |
| Total heuristic checks | 61/91 | Aggregate heuristic passes, **not an accuracy rate over independent tasks** |

In earlier revisions, WEB answer was reported at 37/39; isolating facts from the raw tool trace lowered the score to 22/39. This reflects a methodology correction, not weight degradation. Arithmetic passed 7 of 8 test cases; on `sqrt of 144`, the model routed the prompt to web search instead of the calculator.

Passing heuristic checks (`[OK]`) does not imply overall answer correctness:

- "The Moon is an annual lake on Earth: world football cup" passes because the token "Moon" is present.
- A query on *War and Peace* cites Tolstoy correctly, but fabricates a publication date of 1930.
- The malformed query "Горубинская впадины" passes query extraction checks via the generic substring "впадин".

Accordingly, 22/39 cannot be cited as grounded synthesis accuracy, nor can 27/39 serve as an accurate metric for entity extraction. Validating these capabilities requires rigorous semantic annotation and assertion testing.

#### Live Tool Execution — `eval_ood.py`

| Test | Result | Interpretation |
|---|---:|---|
| CALC | 7/8 | Target numerical token matches in final response |
| WEB | 2/12 | Marker found in final response given a valid tool response |
| GEN | 8/8 | Basic formatting, minimum length, and non-degeneration heuristics only |
| CALIB | 5/5 | Zero tool invocations on control prompts alongside basic text sanity checks |
| Harness reported total | 22/33 (67%) | Aggregated heuristic pass rate, **not 67% task accuracy** |

A score of GEN=8/8 does not demonstrate instruction adherence: an open-ended request to name a coffee shop triggered an extraneous search call followed by a clarification prompt, while an explanation of friendship produced unrelated text. The WEB run conflates generation errors with retrieval failures: queries regarding Lake Baikal, Japan, and other topics returned "not found" from the retriever. Root causes cannot be determined from logs alone. Because WEB test sets diverge across test scripts, 22/39 and 2/12 cannot be directly compared to isolate the impact of replacing oracle contexts with live search.

#### System-Level Comparisons

Applied evaluations target **end-to-end assistant performance**, rather than static parametric memory alone. Grimoire is evaluated alongside its native `calculate` and `web_search` integration: tool selection, argument generation, and context summarization represent core features of the system. Disabling tools to enforce parity with bare models is not required here.

While end-to-end system comparisons are practically informative, they do not substitute for controlled architectural ablations in identical environments. Successful calculator invocations reflect system-level orchestration, not an innate capacity of the neural network to perform arithmetic internally.

#### Eight Instruction Smoke Tests

| Model | Automated Marker/Format Checks |
|---|---:|
| ours_36m (assistant8), with tools | 8/8 |
| Veyra-30M-Instruct | 0/8 |
| Veyra2-Mango-30M-Instruct | 1/8 |
| Stentor-30M-Instruct | 0/8 |
| Tensor-2-40m-instruct | 1/8 |
| TinyStories-Instruct-33M | 2/8 |
| TinyStories-45M | 1/8 |
| ru-babylm-urop-shivan | 0/8 |
| ru-babylm-ft | 2/8 |

**These figures represent internal diagnostic scores, not an objective quality ranking.** An 8/8 score persists under corrected parsers due to lenient heuristic rules: a response on the ozone layer passes a question on water because it mentions "oxygen"; an answer on Pushkin containing an incorrect birth year passes based on the name alone; creative writing and advice queries are assessed primarily by formatting length checks. Baseline models exhibit similar false-positive approvals. A qualitative semantic analysis of all saved responses under a uniform rubric is detailed below; Grimoire does not reliably solve all eight prompts.

#### Qualitative Assessment of Generated Outputs

Following benchmark execution, all 157 responses across the three generated JSON files were reviewed, including 72 outputs from nine comparative models in the instruction smoke test. This constitutes **an LLM-assisted evaluation using explicit rubrics**, not an independent human study or an environment re-run.

- **Pass:** The instruction is fulfilled coherently without factual failure modes or degenerative repetitions.
- **Partial:** The response contains relevant, useful elements, but remains incomplete, semantically distorted, or hallucinated. Does not count as fully solved.
- **Fail:** The primary response is factually incorrect, absent, or unaligned with the prompt.

For WEB tasks, grounding against the retrieved context is required. Extraneous claims outside the source do not automatically make a response false, but preclude a strict pass for grounded synthesis. Subjective boundaries apply to open-ended tasks; independent verification of every factual claim was not conducted. Evaluations assess the full generated sequence, not merely the opening sentence.

| Grimoire Test Set | Pass | Partial | Fail |
|---|---:|---:|---:|
| `eval_det`: WEB, fixed tool responses | 3 | 10 | 26 |
| `eval_det`: CALC | 7 | 0 | 1 |
| `eval_det`: CALIB | 5 | 0 | 0 |
| `eval_ood`: WEB, live tools | 0 | 2 | 10 |
| `eval_ood`: CALC | 7 | 0 | 1 |
| `eval_ood`: GEN | 1 | 1 | 6 |
| `eval_ood`: CALIB | 5 | 0 | 0 |

In fixed WEB evaluations, unambiguous passes occurred exclusively on Pushkin, Mars, and Python prompts. All seven queries on world capitals omitted the capital city name in the final generation, despite its presence in the tool context. This indicates persistent failures in selective extraction from retrieved context: the model frequently mirrors opening definitional lines while discarding the specific requested attribute.

Under live WEB conditions, two partial passes emerged (Paris and Pushkin), both pairing factual elements with geographical fabrications or distorted biographical details. In six runs, the retriever returned "not found", despite five queries being well-formed. This highlights external retrieval failure modes; an explicit refusal is preferable to hallucination, but leaves the user task unfulfilled.

**Arithmetic performance (7/8) and conversational calibration across five social control prompts are verified.** Primary failure modes in the current checkpoint center on target attribute extraction, semantic drift during summarization, and open-ended instruction following.

Qualitative assessment of the eight instruction test prompts across all systems:

| Evaluated System | Pass | Partial | Fail |
|---|---:|---:|---:|
| Grimoire 35.85M + integrated tools | 1 | 4 | 3 |
| Veyra-30M-Instruct | 0 | 0 | 8 |
| Veyra2-Mango-30M-Instruct | 0 | 2 | 6 |
| Stentor-30M-Instruct | 0 | 0 | 8 |
| Tensor-2-40m-instruct | 0 | 2 | 6 |
| TinyStories-Instruct-33M | 0 | 5 | 3 |
| TinyStories-45M | 0 | 1 | 7 |
| ru-babylm-urop-shivan | 0 | 0 | 8 |
| ru-babylm-ft | 0 | 0 | 8 |

Across this limited test suite, Grimoire achieved a full pass strictly on arithmetic via external tool offloading. This capability was absent in competing baseline configurations, though a single test sample cannot justify generalized claims of architectural superiority. Partial successes among baselines were also identified: for instance, TinyStories generated the candidate shop name `Sugar Land` before collapsing into an infinite generation loop.

Overlapping prompts across test files do not constitute independent trials. Aggregate sums across disparate categories are not reported as a unified accuracy metric.

#### Generation Speed

Standard speed measurement protocol: 50 new tokens generated, 3 warmup iterations, averaged across 10 evaluation runs on an AMD Radeon RX 6600, using default float32 precision; KV-caching enabled for Grimoire. Timings include initial prompt prefill. Evaluated models use distinct tokenizers and prompt formats.

| Model | tok/s |
|---|---:|
| ru-babylm-ft | 325.93 |
| ru-babylm-urop-shivan | 319.04 |
| TinyStories-Instruct-33M | 178.07 |
| Veyra-30M-Instruct | 146.08 |
| **ours_36m (assistant8)** | **93.81** |
| TinyStories-45M | 93.72 |
| Veyra2-Mango-30M-Instruct | 82.85 |
| Tensor-2-40m-instruct | 68.06 |
| Stentor-30M-Instruct | 64.18 |

Small throughput differentials (such as 93.81 vs. 93.72 tok/s) are within margin of error; run-to-run variance is not published. These metrics capture local generation exclusively and do not account for end-to-end network retrieval latency.

Raw test runs are archived in `results/eval_det_v2.json`, `results/eval_ood_v2.json`, and `results/bench_results_v2.json`.

### Current Verification Methods

`eval_utils.py` strips tool traces prior to scoring final generations. Arithmetic verification requires exact numerical token matches rather than substring searches. These checks remain heuristic: they cannot confirm factual consistency and may pass contradictory statements.

`bench_suite.py instr` provides eight smoke-test prompts, not a comprehensive instruction benchmark. Automated scores indicate surface pattern matches, not holistic reasoning capability. Qualitative findings are detailed above. Comparisons reflect integrated assistant systems: Grimoire with its runtime tools against text-only baselines. Tool access represents an engineered feature, not grounds for benchmark exclusion. However, this test does not isolate architectural contributions from tool augmentation or dataset composition.

`eval_det.py` injects standardized tool outputs regardless of generated queries (oracle-context setup). It separately inspects arguments in the query and target facts in the final generation. Across 39 WEB queries, this yields 78 checks; combined with 8 CALC and 5 CALIB prompts, it produces 91 evaluations, **not 91 independent tasks**. `lexical_overlap` tracks surface n-gram overlap, not factual faithfulness. A passing substring check does not establish overall sentence validity.

Comparable BPB computation for our model is **disabled**, rather than patched via masking heuristics. Valid BPB restoration requires a strictly causal protocol paired with an invertible tokenizer. The Hugging Face BPB evaluation is retained as a legacy diagnostic with non-overlapping windows: leading window tokens are discarded, while denominators reflect total byte counts. Comprehensive comparisons require a standardized evaluation protocol tracking exact evaluated bytes.

Throughput is benchmarked in float32 using default loaders. Because tokenizers vary in vocabulary density, raw tok/s does not directly translate to equivalent textual throughput across models.

### Benchmark Commands

```bash
# After environment configuration and checkpoint retrieval:
python bench_suite.py instr --output bench_results_v2.json --force
python bench_suite.py speed --output bench_results_v2.json --force
python bench_suite.py hf --output bench_results_v2.json --force

# 'ours' indicates comparable BPB is disabled; 'all' runs remaining suites
python bench_suite.py ours

python eval_det.py --run runs/assistant8 --show --json-out eval_det_v2.json
python eval_ood.py --run runs/assistant8 --show --json-out eval_ood_v2.json
python -m unittest discover -s tests -v
```

New runs output to `bench_results_v2.json`. Omitting `--force` appends runs matching the v2 schema; when evaluating alternate checkpoints or runtime configurations, specify a unique output destination. Benchmark scripts reference `night_glm9q` and `assistant_big6`: interim research checkpoints that may be published at a later stage.

## Scaling and Limitations

The ~129M variant integrates a base decoder layer from Qwen2.5-1.5B with an expanded vocabulary. Its existence demonstrates parameter scaling of the architecture, but does not prove improved semantic performance or justify the efficiency of the dynamic weight factory. Controlled ablations—freezing weights, removing the dynamic factory, varying recurrent depths under fixed compute budgets, and evaluating on standardized out-of-domain benchmarks—remain necessary.

Core technical limitations include a lossy vocabulary, brittle tool query formulation, factual distortion during context summarization, lack of conversational history, and unverified out-of-distribution generalization beyond constrained demonstration prompts.

## Repository Structure

- `build_neuroset.py` — Core architecture, training routines, text generation, and tool orchestrators.
- `bench_suite.py` — Comparative smoke testing and generation speed profiling.
- `eval_det.py`, `eval_ood.py` — Deterministic and out-of-distribution tool-use test runners.
- `eval_utils.py` — Generation post-processing, trace separation, and evaluation heuristics.
- `tests/test_eval_utils.py` — Zero-GPU unit test suite for the evaluation harness.
- `bench_results.json` — Archival benchmark results with associated caveats.
- `bench_corpus.txt`, `bench_corpus_en.txt` — Reference corpora for language modeling diagnostics.

`runs/`, training datasets, and local retrieval stores represent external runtime artifacts not tracked by this Git repository. Checkpoints must be downloaded directly from Hugging Face. Checkpoint loaders look for `final.pt`, followed by `best.pt`, `last_model.pt`, and `last.pt`; explicitly verify the active weight file when reproducing results.

## License and Acknowledgments

Source code is licensed under Apache-2.0; see `LICENSE` and `NOTICE`. Base model components derive from Qwen2.5. Comparative baseline models remain the intellectual property of their respective creators; examine weight and dataset licenses independently. Wikipedia extracts are subject to their respective Creative Commons attribution and distribution terms; the software license does not supersede third-party data licenses.