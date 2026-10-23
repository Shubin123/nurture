# Nurture · Consent-first content filtering plan

Updated October 22, 2026. Reframes the original protection proposal around voluntary English content filtering. The earlier licensing and anti-redistribution strategy remains in [PROTECTION.md](PROTECTION.md); it is not the current product's primary scope.

## Product contract

- Users explicitly choose each shared source in the browser picker. No silent capture or access to another person's tabs. Authorization is required for other people's private content.
- The current website offers an unfiltered local display-stream preview and a separate working English text-replacement demo. These are distinct capabilities, clearly labeled.
- The companion Chromium extension modifies ordinary text only on tabs explicitly enabled by the user. Each replacement receives a small superscript `*`. A stop-and-restore control is available on the page and in the popup.
- No page content or captures are uploaded, recorded, retained for training or logged. No model is claimed to be trained or accurate without evaluation.
- The filter is a user preference tool, not an access-control mechanism, child-safety guarantee or substitute for platform moderation.

## Implementation status

| Capability | Status |
| --- | --- |
| English whole-word replacement with explicit rules | Implemented; deterministic baseline, not ML |
| Small marker next to replacements | Implemented in demo and extension |
| Browser source picker, consent, local preview, stop | Implemented; requires manual browser/platform verification |
| Another tab's ordinary DOM text replacement | Extension prototype; user must explicitly activate on that tab |
| Context-sensitive English models | Tiny/Mini comparison plus balanced Mini fine-tuning, INT8 export and full browser evaluation; balanced Mini selected as opt-in review only, automatic-filter gates unmet |
| Lightweight learned model | Balanced character-context model selected; 64 KiB INT8 weights, ASCII hashing optimization, full browser evaluation and benchmarks included |
| Shared-tab speech filtering | Implemented: local Whisper tiny.en (8-bit, CPU/WASM worker), configurable 3–8 s audio/video delay, mute or beep on rule and word-list matches, filtered captions, late-detection reporting; end-to-end test with synthesized speech. Real-content accuracy unmeasured |
| Cited word-list sources | CMU Offensive/Profane Word List vendored with citation and checksum; manifest-driven additions plus session-only local `.txt` lists; phrase/hyphen/possessive matching; exceptions list |
| Stream OCR, pixel overlays | Not implemented |

## Proposed milestones

1. **Prototype review:** exercise allow/deny/cancel sharing, browser stop, consent withdrawal, extension activation, dynamic pages and restoration. Confirm the English lexicon and replacement policy with the user. Treat unexpected page behavior as a release blocker for that site.
2. **Reliable DOM filtering:** test supported Chromium versions, large pages, page frameworks, navigation and accessibility. Add worker/batched processing where measurements require it. Keep no broad host access by default. Do not claim complete filtering for frames, shadow DOM or split-node words.
3. **Dataset preparation — completed for the research experiment:** pinned CC0 Toxic Spans data, normalized disjoint splits and checksums. No tab data collected. Author IDs are unavailable, so author/source independence is not established. See [MODEL.md](MODEL.md).
4. **Model experiment — completed:** trained a character-context baseline and fine-tuned two compact span classifiers; compared native FP32/INT8 and actual browser outputs. Published precision, recall, character F1, confidence intervals, diagnostics and limitations. Independent dialect/fairness validation remains open.
5. **Browser validation — research release completed, production gate open:** quantized models, self-hosted CPU/WASM worker, batched overlapping chunks, tokenization conformance, measured latency/throughput and a process-tree memory proxy. No GPU is required. Models remain explicitly opt-in review tools because automatic-filter accuracy and memory gates are not established.
6. **Stream filtering:** speech is implemented as an explicit opt-in (above). Next: measure word-level miss and late rates on real, consented recordings across accents, music and crosstalk; evaluate larger Whisper models against the latency budget. Separately validate OCR, region tracking and delayed overlay output; do not relabel shared pixels as filtered.

## Trust boundaries and release gates

Use DOM text nodes and `textContent`, not HTML interpolation, for user rules. Leave inputs, passwords and editable regions alone. Original matched words are kept only in the extension's page session for restoration; explain that DOM changes remain visible to site scripts. Stop must disconnect mutation observers and release capture tracks. Handle permission denial without retry loops or hidden prompts.

Run engine regressions, confirm the two engine copies match, inspect extension permissions and check JavaScript syntax before publication. Manual browser tests must cover unsupported capture, permission cancellation, extension restrictions, dynamic content and reload recovery. The prototype may flash original content before replacement and is not a preventive security boundary.

`MODEL.md` and `docs/reports/RESULTS.md` distinguish achieved research deliverables from unmet production gates. Training uses a pinned public CC0 benchmark, not captured tabs. Do not silently promote the research models to automatic enforcement based on throughput or aggregate F1 alone.

## Hosting

Keep the repository and GitHub Pages public, as requested. Publish from `main` → `/docs` without a custom Actions workflow. Do not publish captures, private browsing data, credentials or sensitive datasets. No agent attribution in commits or pushes.
