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
| Context-sensitive English model | Training/evaluation roadmap only |
| Stream OCR, pixel overlays, speech filtering | Not implemented |

## Proposed milestones

1. **Prototype review:** exercise allow/deny/cancel sharing, browser stop, consent withdrawal, extension activation, dynamic pages and restoration. Confirm the English lexicon and replacement policy with the user. Treat unexpected page behavior as a release blocker for that site.
2. **Reliable DOM filtering:** test supported Chromium versions, large pages, page frameworks, navigation and accessibility. Add worker/batched processing where measurements require it. Keep no broad host access by default. Do not claim complete filtering for frames, shadow DOM or split-node words.
3. **Dataset preparation:** approve category definitions, licensing, consent, annotation scheme and source-separated splits. No implicit training capture. See [MODEL.md](MODEL.md).
4. **Model experiment:** compare compact span classification with deterministic and character baselines; evaluate context, benign text and dialect slices. Train only after authorized data exists. Publish metrics and failures.
5. **Browser validation:** quantize/export the candidate, measure warm and cold performance and memory on named hardware. Optional GPU must have a tested CPU fallback. Ship only if the model meets predefined accuracy and efficiency gates.
6. **Optional stream filtering:** separately validate OCR, region tracking and delayed overlay output. Do not relabel the existing preview as filtered. Audio remains a separate explicit opt-in milestone.

## Trust boundaries and release gates

Use DOM text nodes and `textContent`, not HTML interpolation, for user rules. Leave inputs, passwords and editable regions alone. Original matched words are kept only in the extension's page session for restoration; explain that DOM changes remain visible to site scripts. Stop must disconnect mutation observers and release capture tracks. Handle permission denial without retry loops or hidden prompts.

Run engine regressions, confirm the two engine copies match, inspect extension permissions and check JavaScript syntax before publication. Manual browser tests must cover unsupported capture, permission cancellation, extension restrictions, dynamic content and reload recovery. The prototype may flash original content before replacement and is not a preventive security boundary.

Model launch gates, performance targets and data requirements are proposed in `MODEL.md`; none are reported as achieved. No training should start until the dataset and evaluation protocol are approved.

## Hosting

Keep the repository and GitHub Pages public, as requested. Publish from `main` → `/docs` without a custom Actions workflow. Do not publish captures, private browsing data, credentials or sensitive datasets. No agent attribution in commits or pushes.
