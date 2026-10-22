# Balanced-context follow-up experiment

The first research release is committed and deployed. It includes three genuinely trained models, native FP32/INT8 evaluation, full held-out browser evaluation, throughput and explicit unmet production gates. Its diagnostics flag identity mentions and quoted insults too often. The goal remains active while improving useful context handling rather than calling raw throughput sufficient.

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

## Native results and remaining work

Training and quantized native evaluation completed. `docs/reports/balanced-evaluation.json` records the comparison on 2,000 reused span-test comments and 5,000 newly frozen rated-negative controls:

| Model | Span-test macro character F1 | Rated-negative post false-positive rate |
| --- | ---: | ---: |
| Original character | 0.6419 | 12.14% |
| Balanced character | 0.6501 | 4.06% |
| Original Mini | 0.6609 | 24.04% |
| Balanced Mini | 0.6475 | 3.12% |

The character candidate improves both measures in this comparison. Mini reduces false positives substantially with a modest span-F1 tradeoff. File sizes remain 65,536 bytes of character weights and 11,351,057 bytes for Mini's quantized ONNX model. These rates still do not meet the proposed automatic-filter gate. The mixed-data aggregate F1 must not be advertised as generic browsing accuracy.

**Next critical path:** run the candidate weights/configurations through the actual browser worker on the mixed test set, repeat the benign/identity/quotation diagnostics, benchmark latency/throughput without concurrent training, and only then decide whether to promote. Candidate exports remain in `ml/runs/candidate-models/linear-balanced/` and `ml/runs/candidate-models/bert-mini-balanced/`. The deployed website still serves the original research models. Test runner `tests/browser.mjs` supports `BASE_URL` and has passed against the public Pages site.

The first release's asset manifest and benchmark fingerprints still refer to the original deployed models. Do not overwrite their reports or relabel their timing as measurements of the balanced candidates. If promoting, preserve the original evidence, regenerate the deployment manifest and corresponding browser reports, rerun all tests, and publish via branch-based Pages without a custom Actions workflow or agent commit attribution.
