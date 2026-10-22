# Nurture English models · trained research release

Three candidates are trained, exported and evaluated locally, and available in the website's explicit-download model lab. This is **not an autonomous safety-filter release**: precision and independently validated benign-false-positive gates remain unmet. No private tabs, streams or customer content were used for training.

## Candidates

| Model | Training | Quantized artifact | Role |
| --- | --- | --- | --- |
| Character-context logistic regression | Word, neighboring word, word-pair and character 2–4 gram features hashed into 65,536 dimensions; three regularization candidates selected on validation | 65,536 signed INT8 weights plus scale/bias | Fastest default; pure JavaScript inference |
| BERT Tiny | Google 2-layer/128-hidden encoder, 4,369,666 parameters; eight epochs, best validation checkpoint at epoch 4 | 4,488,172-byte ONNX | Compact comparison; not an accuracy upgrade over the character model |
| BERT Mini | Google 4-layer/256-hidden encoder, 11,105,282 parameters; eight epochs, best validation checkpoint at epoch 2 | 11,351,057-byte ONNX | Modest contextual benchmark improvement at greater compute cost |

Neural models use binary subword span labels (`O` / `TOXIC`), not generative rewriting. A subword is positive when at least half its covered characters are annotated toxic. AdamW uses learning rate 0.0001, weight decay 0.01, batch size 64, 10% warmup and gradient clipping at 1.0. Training uses seed 1729 and the local M4 MPS backend; floating-point kernels may not be bitwise reproducible across platforms.

Training, evaluation and serving use 128-position chunks with 32-WordPiece overlap and average overlapping subword scores. No tail is silently truncated. Checkpoints and thresholds use validation only, with thresholds reselected independently after quantization and frozen before held-out evaluation: 0.30 (character), 0.425 (Tiny INT8), 0.40 (Mini INT8).

## Data

Pinned upstream **SemEval-2021 Task 5 Toxic Spans**, CC0. After lowercase/whitespace-normalized exact deduplication: **7,917 train**, **689 validation** (official trial), **2,000 test**. Test takes deduplication priority over validation and train, without label-based decisions. Source revision, hashes, dropped counts and split checksums are in `docs/reports/dataset.json`. Raw data stays in ignored `ml/data/`, not on Pages.

This is toxic-comment-domain span annotation, not representative English browsing. **Empty gold spans are not benign labels**: an entire toxic comment can have no attributable span. Author IDs are absent, so author/source-disjoint evaluation cannot be verified. Exact deduplication does not establish semantic or near-duplicate independence. Dialect fairness and production false-positive rates are not established.

The purpose-written `ml/challenge.json` separately probes benign language, identity mentions, quotation/negation and direct insults. It is not training or threshold-selection data. Its single-author provisional labels are not independently adjudicated or a representative benchmark. Predictions are in `docs/reports/challenge.json`; this diagnostic does not establish a production benign false-positive rate.

## Evidence

See [measured results](docs/reports/RESULTS.md), [native evaluation](docs/reports/evaluation.json), [full browser evaluation](docs/reports/browser-evaluation.json) and [browser benchmarks](docs/reports/browser-benchmark.json). All 2,000 held-out comments run through the actual served browser worker, not just a Python surrogate.

Reports distinguish macro per-document character F1, micro character precision/recall, exact character-set agreement, empty-gold prediction rates and length slices. Confidence intervals bootstrap whole documents 1,000 times rather than treating dependent characters as independent samples. They describe this dataset, not deployment guarantees. Expanding subwords to displayed whole words has separate metrics; it cannot silently inherit subword accuracy.

Speed reports name hardware, OS, browser/runtime versions, thread count, warmups and repetitions. Warm workloads cover 126, 128 and 1,024 useful input tokens, excluding overlap and padding from throughput. Actual variable-length held-out throughput is separate. Native inference-only and browser end-to-end processing are not interchangeable; localhost initialization is not Internet download performance. Repeated timing sentences are not an accuracy test.

## Efficiency

- INT8 MatMul and embedding/Gather weights reduce neural files roughly fourfold; quantized accuracy is independently evaluated. Smaller is not uniformly faster on native CPU, and no universal speedup is claimed.
- Share one vocabulary and pinned, self-hosted WASM runtime. The character model does not initialize WASM. No inference text is uploaded.
- Run off the main thread with a fixed 128-position sequence dimension, bounded batches of eight and single-thread CPU/WASM. No GPU or cross-origin isolation is required. Batch sweeps are published; small noisy differences are not advertised as wins.
- Bound inputs to 50,000 characters and handle all chunks. Stop/unload terminates the worker and clears its output. Explicit model consent is revocable; public weights may remain in normal browser caches.
- The 150 MB additional-memory target is **not proven**. Whole-Chrome RSS sampling is a noisy process-tree proxy, not per-model allocation. Runtime overhead exceeds weight size; no mobile-memory claim is made.

## Release gates

The proposed automatic-filter gate requires at least 99% precision, meaningful recall, independently low benign false positives and bounded performance. These candidates **do not pass**. They remain opt-in review tools, not child-safety guarantees or substitutes for platform moderation. Unknown candidates are highlighted for review. Only the user's explicit lexicon supplies replacement words, marked `*`. Model scores are not calibrated harm probabilities.

The extension remains deterministic: it does not silently load models or broaden permissions. Stream preview remains unfiltered. OCR, region tracking, delayed visual overlays and speech filtering are separate, unimplemented workstreams. Text-model evaluation does not demonstrate stream censorship.

## Balanced-data follow-up

A second local experiment adds pinned, human-rated zero-toxicity Civil Comments examples without increasing model architecture sizes. Its native evaluation substantially reduces rated-negative false positives; details, sampling caveats and the measured recall/F1 tradeoff are in [ml/EXPERIMENTS.md](ml/EXPERIMENTS.md) and [the comparison report](docs/reports/balanced-evaluation.json). These candidates are **not deployed yet**: full browser evaluation, diagnostics and efficiency verification remain required before promotion. The automatic-filter gate is still unmet and the overall improvement work remains active.

## Reproduce

Reference: Apple M4, 24 GiB RAM, Python 3.12 and Node.js. Packages are pinned in `ml/requirements.lock` and `package-lock.json`.

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r ml/requirements.lock
npm ci
sh scripts/train-models.sh
```

This downloads pinned public revisions, verifies dataset hashes, trains all candidates, selects on validation, exports, checks PyTorch/ONNX parity, evaluates FP32/INT8, checks tokenization and runs unit tests. No paid cloud jobs are launched. The dataset contains offensive language. MPS is preferred when available, otherwise CPU is used.

Serve `docs/` in another terminal with `python3 -m http.server 8000 --directory docs`. Set `CHROME_PATH` for a non-default Chrome location. Stop other heavy jobs; run measurements sequentially:

```sh
npm run build
npm run test:browser
node scripts/benchmark-browser.mjs
node scripts/evaluate-browser.mjs
.venv/bin/python ml/report_browser.py
node scripts/evaluate-challenge.mjs
node scripts/manifest-models.mjs
.venv/bin/python ml/render_report.py
```

Browser tests use real CPU/WASM inference; capture denial is mocked, not cross-platform real-capture evidence. Tokenizer conformance covers token IDs, Unicode offsets and hashed features on 695 validation/diagnostic fixtures. Export checks compare FP32 ONNX against PyTorch; full quantized accuracy is evaluated separately.

Pages publishes committed artifacts from `main` → `/docs`, without a custom Actions workflow. Artifact hashes are in `docs/reports/artifacts.json`. No claim of signed model releases is made.

## Upstream sources and licenses

- Dataset: https://github.com/ipavlopoulos/toxic_spans — pinned SemEval-2021 subset. See the ACL2022 README for empty-span interpretation.
- Tiny: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
- Mini: https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
- Quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Browser runtime: https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html

BERT-derived weights retain upstream Apache-2.0; ONNX Runtime is MIT; the dataset is CC0. Notices and full upstream licenses are in `licenses/`; these do not assign a new project-wide license.
