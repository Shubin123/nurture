# Text cascade: kernel-space search and representation repair

Repository: https://github.com/Shubin123/nurture

This follow-up interprets **kernel space** as (a) exact nearest-code lookup
implementations and (b) the representation/search configuration around that
lookup. It is not an OS-kernel change or an RBF/SVM kernel experiment. All
experiments remain text-only, local, optional and unserved. Existing release
models, browser workers, consent controls and speech paths are unchanged.

## Research direction

Primary sources consulted:

- Yu et al., *Vector-quantized Image Modeling with Improved VQGAN*, ICLR 2022,
  https://arxiv.org/abs/2110.04627 and the paper's low-dimensional, normalized
  code lookup design. We reuse that **quantization idea on text**, not its vision
  model, datasets, generation claims or numerical results.
- Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter
  Optimization*, https://arxiv.org/abs/1603.06560. The implementation here is
  one deterministic successive-halving bracket, not full multi-bracket Hyperband
  or Bayesian optimization. Official scikit-learn implementation reference:
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html.
- PyTorch `cdist` documentation,
  https://docs.pytorch.org/docs/2.14/generated/torch.cdist.html, for the independent
  non-matrix-multiplication distance path used as a correctness reference.
- Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple*,
  https://arxiv.org/abs/2309.15505, and Zhai et al., *VP-VAE: Rethinking Vector
  Quantization via Adaptive Vector Perturbation*, February 19, 2026,
  https://arxiv.org/abs/2602.17133. These motivate future collapse-resistant
  comparisons but are **not implemented or measured here**. Their reported
  image/audio advantages do not establish a text-filtering advantage.
- The original toxic-span shared-task report,
  https://aclanthology.org/2021.semeval-1.6/, establishes task context, not a
  contemporary leaderboard victory. There is no independently established SOTA
  comparison in this follow-up.

No external implementation was copied or added as a dependency.

## 1. Exact lookup implementation search

`ml/vq_search.py` offers four implementations for the same nearest-code problem:

- `reference`: squared Euclidean norm expansion (or normalized dot for cosine).
- `dot`: maximize `query @ code.T - ||code||² / 2`, omitting the query norm and
  the factor of two because neither changes the Euclidean argmin. Cosine uses
  normalized dot products instead.
- `tiled`: scan 256-code blocks, retaining the best score/index per query and
  preserving the first index on exact ties across blocks. Score-block storage
  scales with the tile width, not the full codebook.
- `cdist`: direct, non-GEMM distances, used for independent float64 confirmation.

Lookup is detached from autograd; the straight-through representation and
codebook/commitment losses still receive gradients. No approximate-neighbor
index, codebook norm cache, network service or persistent state is involved.
Float16 inputs are searched in float32; float64 reference inputs retain float64.
Zero cosine queries explicitly select index zero, avoiding spurious tie-breaking
from square-root rounding in the direct-distance implementation. This edge case
has a regression test; the full predefined protocol was rerun after its repair,
without changing candidates, budgets or calibration rules based on test results.
The rerun produced the identical selected-checkpoint SHA-256 and quality metrics.
Algebraic equivalence is not a promise of bitwise identical nearest indices for
every possible floating-point near tie or extreme-magnitude input.

`ml/benchmark_vq_search.py` selects the lowest-median agreeing kernel on tuning
inputs, then confirms on **fresh random inputs and codebooks**. Seven randomized
timing rounds, 30 calls per round, five warmups per implementation, one CPU
thread. Every measured implementation agreed with the direct float64 reference
on 100% of tested tuning and confirmation indices.

| Queries × codes × width | Selected on tuning | Fresh-input reference / selected time |
| --- | --- | ---: |
| 16 × 64 × 64 | dot | 1.20× |
| 64 × 64 × 64 | dot | 1.22× |
| 1,024 × 128 × 16 | tiled | 1.20× |
| 1,024 × 256 × 64 | tiled | 1.24× |
| 1,024 × 4,096 × 64 | tiled | 1.53× |

The largest shape's analytical score block is 1 MiB instead of the full
16 MiB score matrix. This is **not** a measured total-memory/RSS reduction:
inputs, outputs, reductions, framework allocations and overlapping temporaries
are additional. Development reruns varied; raw per-round values from the final
run are published rather than implying a stable hardware-independent gain.
These are bounded lookup-kernel wins, not end-to-end, browser,
quality, or SOTA speed claims. No runtime auto-tuner is installed in the website.
The representation search below fixes `dot`, independently of these timing ranks.

## 2. Validation-only representation search

`ml/search_cascade.py` implements a reusable successive-halving scheduler with
strictly increasing cumulative epoch budgets, deterministic tie-breaking and
optimizer-state continuation only for promoted candidates. Search ranking is
positive-token F1, **not majority-class accuracy**. All candidates share seed
1729, 4,096 original-profile training documents, batch size 32, AdamW LR 0.001,
and the same seeded document/chunk order. Four configurations are predeclared:

| Candidate | Slots | Codes per level | Code width | Lookup metric |
| --- | --- | ---: | ---: | --- |
| coarse-euclidean | 2 / 4 / 8 | 64 | 64 | Euclidean |
| fine-euclidean | 8 / 32 / 128 | 64 | 64 | Euclidean |
| compressed-cosine | 8 / 16 / 32 | 128 | 16 | Cosine |
| fine-cosine | 8 / 32 / 128 | 128 | 16 | Cosine |

The original-profile validation set is partitioned by document into 344 selection,
172 temperature-calibration, and 173 gate-selection documents. All chunks from
one document stay in one partition. Search never loads the test split. The
selected checkpoint and calibration are hashed before the separate evaluator
opens test data. Checkpoints remain ignored under `ml/runs/`.

This is a deliberate **supervised representation repair**, not the earlier
frozen-encoder label-efficiency experiment. The full encoder, projections,
codebooks and heads train jointly with weighted cross-entropy plus VQ loss.
Positive loss weight is `min(20, sqrt(negative_tokens / positive_tokens))`,
computed from training only; head weights are 0.1/0.2/0.7. Codebooks initialize
from an identical seeded training batch, never from validation or test. There is
no raw-token bypass, reconstruction loss, pretrained teacher or new external data.
Cosine candidates use learned 64→16→64 projections and normalized code vectors.

Budgets are 2 / 6 / 12 epochs, promoting half the candidates at each rung. Actual
work is **22 candidate-epochs vs 48** for training the full grid to 12 epochs,
a 54.2% reduction in this declared resource count, not a measured wall-clock
speedup versus a separately executed full grid. Successive halving can discard
late bloomers, and configuration dimensions/counts differ: this is a search, not
a causal one-factor ablation or a multi-seed optimizer study.

Final-head selection F1 at epoch 2: coarse Euclidean 0.1754, fine Euclidean
0.2759, compressed cosine 0.1454, fine cosine 0.3716. Fine cosine reaches 0.4319
at epoch 6 and 0.4258 at epoch 12. The predefined final-budget selection chooses
**fine-cosine at epoch 12**; the test set is not used to choose the earlier epoch.
In its final training increment, active codes are 75/128/128 and perplexities
16.84/49.58/91.03, versus near-total collapse in the prior frozen-encoder pilot.
These numbers do not isolate whether normalization, data initialization, joint
supervision or finer resolution caused the improvement.

## 3. Frozen test evaluation: progress, but no SOTA advantage

`ml/evaluate_cascade_search.py` evaluates the frozen selected model on all 2,000
original toxic-span test documents, alongside the currently served balanced
references at their existing frozen thresholds:

| Model | Macro character F1 | Micro character precision | Recall | Micro character F1 |
| --- | ---: | ---: | ---: | ---: |
| Selected fine-cosine cascade | 0.5315 | 0.3425 | 0.6996 | 0.4599 |
| Served balanced character | 0.6501 | 0.5663 | 0.6986 | 0.6255 |
| Served balanced Mini INT8, native | 0.6475 | 0.5394 | 0.7241 | 0.6183 |

The new cascade has positive-token precision/recall/F1 of
0.3097 / 0.6758 / 0.4247. It no longer predicts only `O`, but remains substantially
worse than the served baselines. The old pilot used a different profile/sample
and objective: its zero recall is a historical failure, not a matched-budget
test baseline for this run.

No early gate passes the existing empirical 2%-error / minimum-32-exits constraint;
both chunk and document exit fractions are 0% / 0% / 100%. Single-pass native
forced-final/cascade time ratio is 0.978, not a speedup; the two paths execute all
levels and their timing difference is not meaningful cascade acceleration.
Changing dot lookup to reference lookup in the full selected model produces
maximum probability difference **0.0** across this held-out workload.

The references have more/different training data and their thresholds were
selected under different protocols. The old test set has also been reused in
repository research; this is not a fresh independent holdout or a contemporary
leaderboard comparison. Single-seed results cannot establish generalization,
fairness, label efficiency or safe replacement precision. No browser export or
promotion is justified. Existing release artifacts remain unchanged.

## Reproduction and evidence

Use the existing pinned environment and checksum-verified local datasets. Run
heavy commands sequentially, not concurrently:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
npm test
.venv/bin/python ml/benchmark_vq_search.py --repeats 30 --rounds 7
.venv/bin/python ml/search_cascade.py --experimental-cascade
.venv/bin/python ml/evaluate_cascade_search.py
```

Reports: `docs/reports/vq-kernel-search.json`,
`docs/reports/cascade-kernel-space-selection.json`, and
`docs/reports/cascade-kernel-space-evaluation.json`. They include raw timing
rounds, rung history, document-partition hashes, calibration, source/checkpoint
hashes, overlap-aware character metrics and frozen-reference fingerprints.
The original pilot reports remain historical snapshots tied to their recorded
source hashes; they are not regenerated by this follow-up.

Next useful work should use train/validation only: separate each repair in a
matched-budget ablation; compare a fixed scalar quantizer against cosine VQ;
evaluate multi-seed positive-class quality before spending on larger codebooks;
and investigate useful per-level precision rather than relaxing gates just to
manufacture exits. Browser measurements and the later generator roadmap remain
blocked on a model that merits them, not on a missing deployment button.
