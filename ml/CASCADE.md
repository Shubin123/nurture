# Experimental coarse-to-fine text cascade

Repository: https://github.com/Shubin123/nurture

Follow-up: [kernel search and supervised cosine-VQ repair](KERNEL_SEARCH.md).
The original pilot below remains a historical negative result.

Status: research-only; not a served model, not an automatic replacement policy.
This work covers the shared residual-VQ encoder and span-classification experiment,
not the later generator or autoregressive roadmap stages. No browser, extension,
speech, consent flow, or existing model artifact is changed.

## Repository inventory

- `ml/prepare.py`, `ml/prepare_benign.py`, and `ml/common.py`: pinned, deduplicated,
  checksum-verified SemEval Toxic Spans and supplemental Civil Comments zero-rated
  negatives. Raw examples remain ignored under `ml/data/`. Empty annotated spans
  are not a general benignness label. Supplemental negatives are source-specific.
- `ml/train_linear.py`: 65,536 hashed character/context features; INT8 JavaScript
  classifier. `ml/train_neural.py`: BERT Tiny (2 layers, width 128) and Mini
  (4 layers, width 256), original and balanced variants; AdamW span classification.
  Labels are binary `O`/`TOXIC`, assigned by majority annotated character coverage.
- `ml/export_neural.py`, `ml/evaluate.py`, `ml/evaluate_balanced.py`: ONNX INT8
  export, validation selection, held-out character metrics. Neural text is chunked
  at 128 positions with 32-WordPiece overlap and merged by exact offsets.
- `ml/browser-runtime.js`, `ml/browser-worker.js`, `scripts/build-browser.mjs`:
  local WordPiece/character inference, self-hosted ONNX Runtime WASM, worker build
  into `docs/model-worker.js`. The opt-in lab exposes balanced character and Mini.
- `scripts/benchmark-browser.mjs`, `scripts/evaluate-browser.mjs`, and conformance
  tests: actual worker tokenization, inference, aggregation, latency, throughput,
  responsiveness and memory. These end-to-end browser timings are **not** equivalent
  to native model-only timing. Existing release reports and selections are untouched.

## Architecture and relation to NCM

Reference: Hermann Kumbong et al., *Next-Concept Modeling: Fast, Parallel
Coarse-to-Fine Text Generation*, arXiv:2610.14162v1, submitted October 9, 2026.
See https://arxiv.org/html/2610.14162v1, especially §3.3 and Appendix C.1.

`ml/cascade.py` implements a randomly initialized 64-wide, one-block, four-head
bidirectional transformer; mask-weighted adaptive pooling into 2/4/8 slots
(optionally 2/4/8/16); 64-entry per-level codebooks; nearest-code residual
quantization; linear interpolation back to token positions; and cumulative
quantized representations. Three/four small LayerNorm + linear heads predict
span classes without a raw-token bypass. Training uses token reconstruction,
straight-through gradients, codebook MSE, commitment weight 0.25 and 50% nested
dropout of the finest levels. Per-epoch assignment histograms, active code counts,
perplexity and training reconstruction accuracy expose collapse.

This is deliberately **not a paper reproduction**. The paper's example tokenizers
use zero transformer encoder blocks, wider latents, many more scales, full-token
finest resolution, transformer decoders and EMA codebooks. This compact variant
uses a small contextual encoder, a pointwise reconstruction decoder, only three
or four compressed scales and gradient-updated codebooks. Those are consequential
departures, particularly for exact lexical toxic-span recovery. The paper's
generative speedups do not imply classification speedups here.

After unsupervised pretraining, the encoder/codebooks are frozen. Each label
fraction trains only its heads from the same initialization on a nested subset of
training **documents**, keeping the unsupervised data budget fixed. Temperature
scaling uses a 41-point positive log-temperature grid on half of validation
documents. Gates use the other half, including all chunks of each document in
only one partition. Test documents never select temperatures or gates.

An early gate requires the minimum token confidence in a chunk to exceed its
threshold. The lowest qualifying grid threshold must produce at least 32 exits
and at most 2% empirical chunk exact-label errors among currently active validation
chunks. Threshold 1 disables an early gate, even with saturated probabilities;
the final level always exits. This is not a statistical safety guarantee or a
class-specific precision gate. Special tokens participate in inference confidence,
but not classification metrics. Calibration needs larger, independent validation
sets before any deployment claim.

Inference actually removes exited chunks from later quantizers/heads. The shared
transformer still processes the entire input first, so encoder cost cannot be
saved. Every long document retains the existing overlapping-chunk/offset merge;
reported exit fractions are per **chunk**, not per complete document.

## Reproduction

Use the existing pinned `ml/requirements.lock` environment and prepared datasets.
The committed 30,522-entry WordPiece vocabulary is loaded locally; these commands
do not download models or send input text to a service.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
npm test

.venv/bin/python ml/train_cascade.py --experimental-cascade \
  --profile balanced --limit 1024 --pretrain-epochs 3 --head-epochs 6 \
  --fractions 1 0.25 0.1 --output cascade-adamw-pilot
.venv/bin/python ml/train_cascade.py --experimental-cascade --optimizer muon \
  --profile balanced --limit 1024 --pretrain-epochs 3 --head-epochs 6 \
  --fractions 1 0.25 0.1 --output cascade-muon-pilot
.venv/bin/python ml/evaluate_cascade_reference.py \
  --report docs/reports/cascade-adamw-pilot.json
```

Omit `--limit` for all data; use new `--output` names so pilot evidence is preserved.
`--seed`, `--levels`, `--device`, `--learning-rate`, and `--muon-rate` are explicit.
Checkpoints stay ignored under `ml/runs/<output>/`; aggregate JSON goes to
`docs/reports/<output>.json`, with vocabulary, source and dataset-manifest hashes.
CPU is the reproducibility default; MPS is optional and not bitwise equivalent.
Native runs set four PyTorch CPU threads.
Do not run heavy experiments/benchmarks concurrently when comparing wall clocks.

Muon uses only two-dimensional transformer hidden matrices, Nesterov momentum
0.95, five Newton–Schulz iterations and starting LR 0.02. Embeddings, codebooks,
decoder, heads, biases and norms use AdamW at 0.0003 during pretraining. The
baseline uses AdamW everywhere, identical seed/data/batch/epoch budgets. Frozen
encoder head fitting uses AdamW at 0.001 for both arms. These untuned defaults are
not equalized optimization effort. SOAP/Shampoo are not implemented.

## Measured pilot (October 26, 2026)

Reports: `docs/reports/cascade-adamw-pilot.json`,
`docs/reports/cascade-muon-pilot.json`, and `docs/reports/cascade-reference.json`.
One seed (1729), balanced profile, independently shuffled/capped 1,024 documents
per split, three pretraining epochs, six head epochs per label fraction, CPU.
This is a feasibility pilot, not full-data training or a release qualification.
The model has 3,992,192 training parameters and 2,008,134 inference parameters
when the reconstruction decoder is excluded; no quantized artifact is exported.

| Model / labelled fraction | Sampled-test macro character F1 | Micro character precision | Recall | Micro character F1 | Early exits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cascade AdamW / 100%, 25%, 10% | 0.7832 | 0 | 0 | 0 | 0% |
| Cascade Muon / 100%, 25%, 10% | 0.7832 | 0 | 0 | 0 | 0% |
| Served balanced character reference | 0.8550 | 0.5015 | 0.7209 | 0.5915 | n/a |
| Served balanced Mini INT8 reference (native) | 0.8703 | 0.4912 | 0.7567 | 0.5957 | n/a |

Both cascade arms predict only `O` on this test sample at every label fraction.
They score 99.3692% token accuracy because only 432 of 68,486 evaluated chunk
tokens are positive. Their seemingly reasonable macro character F1 comes from
802 empty-gold documents out of 1,024; all 1,874 positive characters are missed.
Temperature scaling does not repair this representation/class-imbalance failure.
No gate qualifies; every chunk reaches level three, so there is no demonstrated
compute saving, even relative to the same cascade forced to its final head.
Native speed-ratio differences with zero exits are measurement noise, not a gain.

Codebook diagnostics catch collapse. AdamW's third pretraining epoch uses exactly
one code per level (perplexity 1). Muon's third epoch uses 6/18/37 codes, but its
coarse-level perplexity is only about 1.19: raw active-code count alone would hide
near-collapse. Third-epoch training reconstruction token accuracy is only about
5.76% (AdamW) / 5.91% (Muon). Neither yields useful toxic-span features. Higher
Muon utilization does not establish better filtering or faster time to useful
accuracy. Label-efficiency conclusions cannot be drawn when every head fails.

The recorded CPU pretraining wall times are 26.41 seconds (AdamW) and 53.54 seconds
(Muon). With all 1,024 labelled training documents, first attainment of the default
95% final-head validation token-accuracy target is 26.86 / 54.07 seconds including
pretraining, respectively. These single-run wall clocks exclude process startup
and data preparation; they are not time to a useful positive-class accuracy or a
general optimizer ranking. Native full-label model-only rates are approximately
242k / 180k processed labelled chunk tokens/s; these include overlap and exclude
tokenization/merging. Do not compare them with the website's useful-token worker
benchmark. No browser cascade measurement exists, and no speedup is established.

Early smoke testing also caught an API compatibility trap: the pinned Transformers
version accepts but ignores the old `vocab_file` keyword in this constructor,
silently making a five-special-token vocabulary. The loader now passes the full
explicit vocabulary and regression-tests ordinary-word encoding. Those invalid
smoke measurements were discarded, not included in these reports.

**Decision: reject promotion and browser export for this pilot.** The current
served models remain untouched. This negative result does not disprove NCM or
coarse-to-fine filtering; it rejects this compressed, undertrained configuration
on this bounded protocol. Full-data, multi-seed work and browser cascade timings
remain unperformed. SOAP/Shampoo and later generator stages are also unperformed.

## Interpretation and next gates

Token accuracy is majority-class sensitive; always inspect positive precision,
recall/F1, per-class support and character metrics. Token metrics and native
processed-token throughput count overlapping chunks; character metrics merge
duplicates per document. Native timing includes the encoder, quantizers, heads
and active-batch selection, but excludes tokenization, host/device transfers and
offset aggregation. The forced-final comparison is a short diagnostic, not a
stable throughput benchmark or a browser result. The 95% accuracy target records
first attainment on final-head validation tokens, including pretraining time;
it can be reached by predicting only `O` and is not useful filter quality.

The reference evaluator compares already-trained served classifiers on exactly
the sampled test documents at their frozen release thresholds. It is a quality
reference, not an equal-training-budget comparison. The test set has been used
for prior repository research: neither this pilot nor repeated follow-ups are a
fresh independent release evaluation. A single seed and a capped sample cannot
establish label efficiency, fairness, or generalization.

Browser export is intentionally withheld unless positive-class quality and a
credible native saving warrant it. A future export needs a separately runnable
encoder plus per-level quantizer/head artifacts and worker-side active chunk
routing; exporting every level in one ONNX graph would not demonstrate early-exit
compute savings. Then extend the existing worker benchmark and conformance tests
before any promotion. No browser throughput is claimed for this experiment.

Next research steps, in order: improve reconstruction and code usage on train /
validation only; evaluate lexical bottleneck resolution and class-balanced head
losses; repeat matched-budget multi-seed runs on full data; establish real exit
savings before export. Do not proceed to a code generator or autoregressive
quality baseline on the strength of an uninformative majority-class classifier.
