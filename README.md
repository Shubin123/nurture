# Nurture

Consent-first blue-team prototype for local English word replacement. Each changed word gets a small `*`.

- **Website:** https://shubin123.github.io/nurture/ — working rule-based text demo and explicit browser stream picker. Shared video is an **unfiltered preview**, not censored output.
- **Extension:** [installation instructions](extension/README.md) — explicitly enable replacements on an authorized current tab, with stop/restore controls.
- **Models:** [MODEL.md](MODEL.md) — trained character-context, BERT Tiny and BERT Mini candidates with INT8 exports, full browser evaluation and throughput measurements. They remain opt-in review tools: automatic-filter precision gates are not met.
- **Results:** [measured comparisons](docs/reports/RESULTS.md), [held-out browser evaluation](docs/reports/browser-evaluation.json) and [browser throughput](docs/reports/browser-benchmark.json).
- **Active improvement experiment:** [balanced-data follow-up](ml/EXPERIMENTS.md) reduces rated-negative false positives in native evaluation. Candidate weights remain local until browser validation and efficiency checks finish.
- **Plan:** [PLAN.md](PLAN.md) — milestones and trust boundaries. The earlier software-protection proposal is preserved in [PROTECTION.md](PROTECTION.md).

## Development

Serve locally with `python3 -m http.server 8000 --directory docs`, then open `http://localhost:8000`. Committed static artifacts run without installation. For development, run `npm ci`, `npm test`, `npm run build` and `npm run test:browser` (with the server running). The extension's shared rule engine is copied from `docs/filter.js` with `npm run sync-filter` after edits.

Full training and evaluation commands are in [MODEL.md](MODEL.md). Python dependencies and JS packages are pinned. Raw data/checkpoints remain ignored; public quantized weights, self-hosted WASM, licenses and aggregate reports are committed. The character model is the throughput baseline; contextual models cost more memory and compute, and are not guaranteed to improve your filtering policy.

The website sends no form text or shared streams to a server, uses no analytics and stores no settings. Hosting receives normal asset requests; public model/runtime files may be cached by the browser. Models load only after explicit download consent, and inference stays local in a worker. Screen sharing requires browser permission. This prototype does not guarantee complete filtering and does not censor video, images or audio.

## Publication

Public repository and public GitHub Pages, published from `main` → `/docs`. No custom Actions workflow. Commits use the owner's configured identity without agent attribution.
