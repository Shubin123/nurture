# Third-party materials

- **BERT checkpoints and WordPiece vocabulary:** Google BERT Miniatures, Apache-2.0 as listed in upstream model cards. Fine-tuned Tiny/Mini derivative ONNX weights retain that license; see `BERT-APACHE-2.0`. Revisions are in training reports. Reference: Turc, Chang, Lee and Toutanova, *Well-Read Students Learn Better: On the Importance of Pre-training Compact Models* (2019).
- **Benchmark:** SemEval-2021 Task 5 Toxic Spans, upstream CC0-1.0; see `TOXIC-SPANS-CC0`. Raw comments are downloaded to ignored local storage, not bundled on Pages. Reference: Pavlopoulos, Sorensen, Laugier and Androutsopoulos, *SemEval-2021 Task 5: Toxic Spans Detection* (2021). Source revision and hashes are in `docs/reports/dataset.json`.
- **ONNX Runtime Web:** Microsoft, MIT; see `ONNX-RUNTIME-LICENSE`, mirrored in `docs/vendor/`. Bundled JavaScript license notices are preserved in `docs/model-worker.js.LEGAL.txt`. Runtime version is pinned in `package-lock.json` and the asset manifest.

These upstream notices do not assign a new license to the entire project.
