# English browser model: training and evaluation plan

Status: proposed, no trained model, dataset or performance results yet.

## Goal

Help a consenting user identify English text spans they choose to soften, on their own authorized browsing surface. Keep transformations explicit: a deterministic replacement from the user's approved lexicon, followed by a small `*`. Never let a generative model silently rewrite surrounding meaning. A visual stream is not page text: OCR and audio recognition are separate systems.

## Baseline first

The included whole-word engine is the reproducible, zero-model baseline. It supports user-defined English terms, case-insensitive matching and visible replacement markers. It does not detect context, inflections, obfuscation, speech or text rendered in images. Log no page text. Maintain explicit regression fixtures rather than collecting shared streams.

## Data and consent

- Define the categories with the user before selecting data; there is no universal implicit censorship policy. Start with a narrow English vocabulary and distinguish criticism, quoted language and targeted abuse.
- Use licensed or purpose-written examples. Record provenance and reuse permission. No scraping private tabs, collecting account identifiers or training on screen captures by default.
- Annotate exact spans, category, context and whether replacement is appropriate. Use two annotators plus adjudication; measure agreement. Keep benign near-matches, negations, quotation, dialect variation and reclaimed language in the evaluation set.
- Split by source and author before augmentation; deduplicate across train, validation and held-out test sets. Keep an additional adversarial set. Redact personal information and document deletion procedures before collection.

## Candidate and training procedure

1. Version an annotation schema and dataset manifest with hashes, licenses and split membership. Do not publish sensitive source text into this public repository.
2. Select a license-compatible compact English encoder with a span-labeling head (BIO labels), compared against both the included dictionary and a lightweight character-based baseline. Do not choose by model size alone.
3. Fine-tune on consented examples with fixed seeds and recorded hyperparameters. Tune thresholds only on validation data. Use per-category precision/recall and exact-span F1, including benign and dialect slices. Reject ambiguous predictions or let the user review them rather than silently changing meaning.
4. For accepted spans, apply only explicit user-approved replacements. Preserve punctuation and attach a marker. If a contextual model flags a span without an approved substitute, show a review suggestion rather than inventing one.
5. Export the candidate to a browser-compatible runtime format. Evaluate quantization against unquantized accuracy; test CPU/WASM fallback and optional WebGPU. Pin runtime/model versions and hashes before release. Model downloads need clear consent, size disclosure and no tab-text upload.
6. Ship only if the candidate materially improves held-out performance over rules while meeting measured browser constraints. Otherwise keep the deterministic baseline and publish the failed experiment honestly.

## Proposed acceptance targets—not measured claims

| Measure | Initial release gate |
| --- | --- |
| Automatic replacement precision | At least 99% on held-out approved categories; report confidence intervals and sample counts |
| Benign-example false positive rate | At most 0.5%, with slice breakdowns; abstain where data is insufficient |
| Recall | Report overall and per category; agree thresholds before the pilot, no promise of complete censorship |
| Warm inference | p95 at most 100 ms per 128-token chunk on named reference desktop hardware |
| Model download | At most 30 MB compressed as an initial engineering budget |
| Peak additional memory | At most 150 MB on the measured reference browsers |
| UI responsiveness | Run inference off the main thread; measure long tasks and time to interactive |
| Cold start / compatibility | Report download, initialization, unsupported-browser behavior and CPU fallback separately |

Publish model card, licenses, dataset summary, evaluation script, reference hardware, browser versions, accuracy/latency results and known failures. These targets require revision if the evidence does not support them; there is no current claim that any candidate meets them.

## Stream workstream

The site currently requests a user-selected display stream with audio disabled and previews it locally. It neither records nor analyzes it. Browser permission is required each time; the user can stop with the app control or browser control. No access to another user's tabs is implied. Their private content requires their own authorization and a separate sharing mechanism.

For visual censorship, prototype downsampled OCR in a worker, timestamped bounding boxes, a delayed processed canvas and explicit overlays. Measure OCR misses, tracking jitter, frame processing latency and CPU/battery use on licensed English test clips. Do not call the preview censored until a separate evaluation shows the overlays actually cover the intended regions. No protection against first-frame leaks can be promised without a measured buffer.

For speech, a later explicit audio opt-in would require a local recognizer, word timing, buffering and muted/bleeped output. Audio capture is browser/platform dependent. Do not play raw audio alongside a claimed filtered output. Keep this out of the initial milestone.

## Technical references

- Screen Capture API: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia
- Temporary extension permission: https://developer.chrome.com/docs/extensions/develop/concepts/activeTab
- Browser inference and WebGPU: https://huggingface.co/docs/transformers.js/guides/webgpu

Recheck supported runtimes and browser behavior at implementation time. The model work is a roadmap, not completed training.
