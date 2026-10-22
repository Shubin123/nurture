# Nurture

Consent-first blue-team prototype for local English word replacement. Each changed word gets a small `*`.

- **Website:** https://shubin123.github.io/nurture/ — working rule-based text demo and explicit browser stream picker. Shared video is an **unfiltered preview**, not censored output.
- **Extension:** [installation instructions](extension/README.md) — explicitly enable replacements on an authorized current tab, with stop/restore controls.
- **Models:** [MODEL.md](MODEL.md) — balanced character-context and BERT Mini INT8 are the selected opt-in review tools. Original three research models remain archived. Automatic-filter precision gates are not met.
- **Results:** [measured comparisons](docs/reports/RESULTS.md), [7,000-comment browser evaluation](docs/reports/balanced-browser-evaluation.json) and [browser throughput](docs/reports/balanced-browser-benchmark.json).
- **Balanced-data experiment:** [protocol and limitations](ml/EXPERIMENTS.md). Browser negative-post false positives are 4.06% (character) and 3.88% (Mini); quotation errors persist. The character baseline processes approximately 418,000 useful tokens/s on the reference M4 warm workload, not a mobile or Internet-speed guarantee.
- **Plan:** [PLAN.md](PLAN.md) — milestones and trust boundaries. The earlier software-protection proposal is preserved in [PROTECTION.md](PROTECTION.md).

## Development

Serve locally with `python3 -m http.server 8000 --directory docs`, then open `http://localhost:8000`. Committed static artifacts run without installation. For development, run `npm ci`, `npm test`, `npm run build` and `npm run test:browser` (with the server running). The extension's shared rule engine is copied from `docs/filter.js` with `npm run sync-filter` after edits.

Full training and evaluation commands are in [MODEL.md](MODEL.md). Python dependencies and JS packages are pinned. Raw data/checkpoints remain ignored; public quantized weights, self-hosted WASM, licenses and aggregate reports are committed. The character model is the throughput baseline; contextual models cost more memory and compute, and are not guaranteed to improve your filtering policy.

The website sends no form text or shared streams to a server, uses no analytics and stores no settings. Hosting receives normal asset requests; public model/runtime files may be cached by the browser. Models load only after explicit download consent, and inference stays local in a worker. Screen sharing requires browser permission. This prototype does not guarantee complete filtering and does not censor video, images or audio.

## Publication

Public repository and public GitHub Pages, published from `main` → `/docs`. No custom Actions workflow. Commits use the owner's configured identity without agent attribution.
