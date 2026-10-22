# Measured model results

Local measurements, not deployment or safety guarantees.

Hardware: Apple M4, 24 GiB RAM. Headless Chrome 155.0.8059.85, CPU-only, one WASM thread. Native ONNX Runtime uses two threads.

Balanced training: 17,917 train, 2,689 validation. Test: 2,000 previously inspected Toxic Spans comments plus 5,000 new frozen rated-negative Civil Comments controls. Checkpoints and thresholds use validation only. This is not a fresh final span holdout.

| Model | Weight/model bytes | Span-test character F1 | Negative post FPR | p95 ms / 128 useful tokens | Useful tokens/s / 1,024-token document |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear-balanced | 65,536 | 0.6501 | 4.06% | 0.4 | 417,618 |
| bert-mini-balanced | 11,351,057 | 0.6402 | 3.88% | 46.7 | 3,822 |

## Interpretation

- F1 is macro per-document character-span F1, not classification accuracy. Whole-word display expansion has separate metrics.
- The character model is the throughput baseline. Balanced Mini lowers negative false positives relative to original Mini but loses span F1 and costs substantially more compute. It is not an accuracy upgrade over the balanced character model on this test.
- Warm speed tests use a repeated English sentence, five warmups and 50 runs. Actual variable-length held-out throughput is in browser-evaluation.json. Useful tokens exclude overlap and padding.
- 126 useful WordPieces plus two special tokens fit one neural chunk; 128 useful WordPieces require two overlapping chunks.
- Native/browser dynamic quantization and batching differ slightly. Quantization reduces weights about fourfold, but is not uniformly faster on native CPU.
- Localhost initialization is not a real Internet cold-download measurement. Neural models additionally need the shared approximately 14 MiB WASM runtime and vocabulary.
- Whole-Chrome RSS sampling is a noisy process-tree proxy and can double-count shared pages. The 150 MB additional-model-memory target is not proven; no memory pass is claimed.
- Empty gold spans are not benign labels: entire toxic comments can lack localized annotations. The small authored challenge is diagnostic, not independent fairness or production false-positive validation.
- The 99% precision / low benign-false-positive release gate is NOT met. Models stay opt-in and review-only; only user-approved rules supply replacements.

## Evidence

- balanced-evaluation.json: frozen native original/balanced comparison. Browser Mini differs slightly because quantized batching/backends differ; browser results govern the UI.
- balanced-browser-evaluation.json: all 7,000 comments evaluated through the compiled browser worker with exact candidate asset hashes.
- release-browser-evaluation.json and served-candidate-equivalence.json: a second full 7,000-comment browser pass using actual served versioned assets without candidate request interception; all prediction/display decisions agree.
- balanced-browser-benchmark.json: worker/roundtrip latency, batch sweeps, responsiveness and process-tree RSS proxy.
- balanced-pre-ascii-benchmark.json and hash-fast-path-equivalence.json: before/after ASCII hashing measurements and zero prediction/display differences across 14,000 model-document pairs. Timing is reference-machine evidence, not a universal speedup.
- balanced-challenge.json: both models falsely flag all six quotation/negation diagnostics. Benign identity mentions flag 2/10 (character) and 1/10 (Mini). These small authored probes are not fairness validation.
- tokenizer-conformance-balanced.json: token IDs, Unicode offsets and hashed features agree on 2,695 validation/diagnostic fixtures.
- balanced-worker-memory.json: experimental worker heap/backing-storage samples, not complete native/process allocation or proof of the 150 MB memory gate.
- release.json: selected versioned review-only assets. Original three models and unprefixed evaluation/browser reports remain archived comparison evidence; their worker fingerprints refer to the initial runtime.
- dataset.json, frozen-selection.json, *-training.json, *-export.json and artifacts.json: provenance, selection, training and integrity records.
