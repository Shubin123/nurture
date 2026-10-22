# Nurture

Consent-first blue-team prototype for local English word replacement. Each changed word gets a small `*`.

- **Website:** https://shubin123.github.io/nurture/ — working rule-based text demo and explicit browser stream picker. Shared video is an **unfiltered preview**, not censored output.
- **Extension:** [installation instructions](extension/README.md) — explicitly enable replacements on an authorized current tab, with stop/restore controls.
- **Model:** [MODEL.md](MODEL.md) — English data, training, browser efficiency and evaluation roadmap. No trained weights or measured model results yet.
- **Plan:** [PLAN.md](PLAN.md) — milestones and trust boundaries. The earlier software-protection proposal is preserved in [PROTECTION.md](PROTECTION.md).

## Development

Serve locally with `python3 -m http.server 8000 --directory docs`, then open `http://localhost:8000`. Run `npm test` with a current Node.js. There are no dependencies. The shared engine in `docs/filter.js` is copied to `extension/filter.js` with `npm run sync-filter` after edits.

The website sends no form text or shared stream data to a server, uses no analytics and stores no settings. Hosting still receives ordinary page requests. Screen sharing requires browser permission; do not share private content without authorization. This prototype does not guarantee complete filtering and does not censor video, images or audio.

## Publication

Public repository and public GitHub Pages, published from `main` → `/docs`. No custom Actions workflow. Commits use the owner's configured identity without agent attribution.
