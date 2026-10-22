# Balanced-context follow-up experiment

The initial three-model research release remains archived. The balanced character and Mini variants have now passed the full browser research verification below and are selected as explicit opt-in review tools, not production safety filters. Identity mentions and quoted insults remain important failure modes.

## Follow-up protocol

- `prepare_benign.py` pins Google Civil Comments at revision `f2970eb3a55777454c94069077cc8d9b5866312d` (CC0). Downloaded parquet shards and raw examples stay ignored.
- Select posts with all seven human-rated toxicity sub-scores exactly zero, 20–4000 characters long. Keep deterministic lowest normalized SHA-256 samples: 10,000 train, 2,000 validation, 5,000 test. Exclude all existing Toxic Spans rows and cross-split selected duplicates. Only the first official train shard is used. Hashes and sampling details are in `docs/reports/balanced-dataset.json`.
- These are source-specific rated-negative labels, not independent proof of universal benignness. Preserve that distinction when reporting false-positive rates.
- The original 2,000-comment span test has been inspected in the first experiment and is reused for comparison, not a fresh independent final test for this follow-up. The supplemental 5,000 rated-negative test comments are frozen new controls and must not tune candidates.
- Mix these negatives with the original span splits. Character-context training uses the same architecture and regularization candidates. Mini uses the same compact architecture, learning rate 0.00005 and six epochs. No raw tab data is used.
- Select checkpoints/thresholds on validation using the mean of original-span macro F1 and rated-negative acceptance. This gives the subsets equal weight rather than letting the larger negative count dominate. The new negative test labels must not tune models or thresholds.
- Keep candidates under ignored `ml/runs/candidate-models/`, separate from served release assets. Do not replace the deployed models until the new accuracy/false-positive comparison, quantization, browser throughput and diagnostics are verified.

## Commands

```sh
.venv/bin/python ml/prepare_benign.py
.venv/bin/python ml/train_linear.py --profile balanced
.venv/bin/python ml/train_neural.py --profile balanced --size mini --epochs 6
.venv/bin/python ml/export_neural.py --profile balanced --size mini
.venv/bin/python ml/evaluate_balanced.py
```

The initial release's reproduction commands retain their `original` defaults. Profiles do not overwrite original checkpoints or reports. GPU training and heavy performance benchmarks should not run concurrently. After training, revalidate the process/job state, evaluate frozen candidates, inspect both toxic-span recall and rated-negative false positives, then decide from evidence whether to promote, refine or reject the candidates. Do not mark the overall goal complete solely because model files exist or one timing test passes.

## Native results

Training and quantized native evaluation completed. `docs/reports/balanced-evaluation.json` records the comparison on 2,000 reused span-test comments and 5,000 newly frozen rated-negative controls:

| Model | Span-test macro character F1 | Rated-negative post false-positive rate |
| --- | ---: | ---: |
| Original character | 0.6419 | 12.14% |
| Balanced character | 0.6501 | 4.06% |
| Original Mini | 0.6609 | 24.04% |
| Balanced Mini | 0.6475 | 3.12% |

The character candidate improves both measures in this comparison. Mini reduces false positives substantially with a modest span-F1 tradeoff. File sizes remain 65,536 bytes of character weights and 11,351,057 bytes for Mini's quantized ONNX model. These rates still do not meet the proposed automatic-filter gate. The mixed-data aggregate F1 must not be advertised as generic browsing accuracy.

## Browser verification and promotion

Both models ran all 7,000 test comments through the actual compiled worker, with candidate assets intercepted from local private export paths and SHA-256 bound to reports. Browser span F1 / negative-post FPR are 0.6501 / 4.06% (character) and 0.6402 / 3.88% (Mini). Mini's browser/native difference is reported, not hidden. Both models flag all six quotation/negation probes; identity probes flag 2/10 and 1/10 respectively. Small authored diagnostics do not validate fairness.

An ASCII fast path avoids UTF-8 allocation when hashing English features; non-ASCII uses the original UTF-8 algorithm. All 14,000 before/after predictions and displayed-word expansions agree exactly. On the reference M4, the character model's 1,024-token warm workload measures approximately 418,000 useful tokens/s versus 106,000 before this optimization. Mini measures approximately 3,822/s; no Mini speedup is claimed. Warm 128-token p95 worker latency is 0.4 ms / 46.7 ms respectively. These are local CPU measurements, not mobile guarantees or Internet download timings.

`node scripts/verify-fast-path.mjs` reproduces the equivalence audit using the ignored pre-optimization and optimized prediction files. To recreate the baseline, evaluate the same frozen candidate weights with the earlier compiled worker before applying the ASCII fast path; do not substitute the original unbalanced weights. Public reports retain only aggregate outcomes, runtime fingerprints and timings, not raw corpus text.

Whole-Chrome RSS and worker heap/backing-storage sampling are published, but neither proves a complete native/WASM allocation bound. The 150 MB gate remains unproven. Automatic-filter gates also remain unmet: both negative false-positive rates exceed 0.5%, and quoted language is not solved. No OCR, audio or video filtering is implemented.

```sh
.venv/bin/python ml/browser_fixtures.py --profile balanced
node tests/tokenizer-conformance.mjs balanced
npm run build
MODEL_PROFILE=balanced node scripts/evaluate-browser.mjs
.venv/bin/python ml/report_browser.py --profile balanced
MODEL_PROFILE=balanced node scripts/evaluate-challenge.mjs
MODEL_PROFILE=balanced node scripts/benchmark-browser.mjs
MODEL_PROFILE=balanced node scripts/worker-memory.mjs
node scripts/promote-candidates.mjs
node scripts/manifest-models.mjs
node scripts/verify-artifacts.mjs
.venv/bin/python ml/render_report.py
npm test
npm run test:browser
```

Promotion checks validation-objective improvement, frozen selection, browser/native consistency, diagnostic presence, latency, file size and asset/runtime fingerprints before copying to versioned `docs/models/linear-balanced/` and `docs/models/bert-mini-balanced/` paths. It does not assert safety or total-memory gates. `release.json` records both as false. The initial models and unprefixed browser reports preserve their original fingerprints; current release evidence has `balanced-` prefixes. Default deployed-profile harnesses read the release manifest, request actual served assets without interception and write `release-*` reports. Publication stays branch-based Pages without custom Actions workflows or agent commit attribution.

Next research requires independently adjudicated quotations/context, a fresh span holdout and representative dialect/domain validation before any automatic-policy claim. Accuracy, privacy and consent take priority over throughput alone.
