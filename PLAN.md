# Nurture · Software protection plan

Status: proposed implementation roadmap · October 22, 2026

## Objective and boundaries

Protect paid value, reduce casual license sharing and commercial redistribution, and preserve a reliable experience for legitimate customers. Treat every client as untrusted. No client-side control promises uncrackable software. This plan is a design proposal, not an implemented security system.

Assumptions: a commercial application with user accounts and optional online premium services. Streaming controls apply only if protected media is part of the product. Initial schedule is a six-week planning estimate, subject to product scope and staffing.

## Protection layers

| Layer | Proposed controls | Target abuse | Limits and safeguards |
| --- | --- | --- | --- |
| Licensing | Signed licenses, activation, entitlement API, revocation | Keygen-style abuse and unmanaged redistribution | Online check-in strengthens enforcement; never embed signing secrets in clients. |
| Device identity | Pseudonymous device registration and optional device binding | License sharing, VM cloning, seat abuse | Fingerprints can be spoofed; provide device resets and avoid raw hardware identifiers. |
| Server-side control | Remote premium APIs, model weights, billing and critical business logic | Offline copies of core paid services | Requires connectivity; budget latency and design an outage experience. |
| Tamper resistance | Selective obfuscation, integrity checks, signed updates and anti-debugging review | Casual reverse engineering and patching | Cost-raising only; avoid invasive drivers and unreliable lockouts. |
| Streaming / content | Platform DRM where supported, forensic watermarking and concurrent-stream limits | Redistribution, credential sharing and leak tracing | Conditional workstream; watermarking helps trace leaks, not prevent capture. |
| Detection | Minimal activation events, anomaly rules and consumption analysis | Abnormal seats, suspicious activations and account abuse | Minimize data, limit retention and review signals before enforcement. |
| Enforcement | Evidence preservation, web monitoring and reviewed provider notices | Commercial redistribution and discoverability | Prioritize material harm; notices and escalation require authorized legal review. |

## Architecture and trust boundaries

1. A billing webhook is authenticated, deduplicated and reconciled into the entitlement service. A client must never grant its own paid status.
2. The account service authenticates the customer; activation atomically reserves an allowed device seat. A device-bound key is preferred over an invasive fingerprint where feasible.
3. A separate signing service issues a short-lived entitlement lease containing subject, product, features, device reference, issuer, audience, expiry and key ID. Use a maintained cryptographic library and an explicitly allowed signature algorithm.
4. Clients verify signatures with public keys. Premium servers independently validate identity, entitlement, expiry, quotas and revocation. Never trust client UI checks for authorization.
5. Proposed default: renew leases every 24 hours, with a documented 72-hour offline allowance for local features after the last successful renewal. These are product-policy proposals, not security guarantees. Remote operations require current server authorization; offline revocation cannot take effect until expiry or reconnection.
6. Rotate signing keys with an overlap period, audited access and a tested compromise runbook. Keep signing material in managed key storage, separate from source code and release artifacts.

## Proposed API contract

| Endpoint | Responsibility | Required safeguards |
| --- | --- | --- |
| `POST /v1/activations` | Register device and reserve seat | Authenticated account, idempotency key, rate limits, atomic seat cap |
| `POST /v1/leases/renew` | Issue refreshed signed entitlement | Device proof, current billing state, revoked-device check |
| `GET /v1/entitlements` | Return current account features | Account-scoped authorization; no shared-cache leakage |
| `DELETE /v1/activations/{id}` | Release an owned seat | Ownership verification, audit trail, reset-abuse limits |
| `POST /v1/admin/revocations` | Revoke account, device or license | Privileged role, strong authentication, reason and audit record |

Use TLS, bounded request sizes, replay-resistant device challenges, structured error responses and explicit API versioning. Do not log complete licenses, tokens or hardware identifiers.

## Delivery roadmap

| Phase | Owner role | Deliverables | Exit criteria |
| --- | --- | --- | --- |
| Week 1 · Define | Product + security | Asset inventory, abuse model, connectivity policy, privacy review and baseline metrics | Approved threat model, offline policy and data inventory |
| Weeks 2–3 · Authorize | Backend + billing | Entitlement store, webhook reconciliation, signing, activation and revocation APIs | Tampered/expired/revoked leases rejected; concurrent activation cannot exceed seats; webhook retries do not duplicate grants |
| Week 4 · Integrate | Client + platform | Lease verification, renewal, account device dashboard and recovery UX | Offline boundaries, clock changes, reset flows and service outages tested; supported OS/device matrix documented |
| Week 5 · Observe | Security + operations | Minimal telemetry, anomaly rules, alert review and signing-key runbooks | Retention verified, false-positive baseline measured, key rotation and recovery rehearsed |
| Week 6 · Harden and pilot | Release + support | Selective hardening, staged rollout, support playbook and optional media assessment | Update signature verified, rollback exercised, pilot reviewed and release approved |

## Privacy and customer experience

- Collect only account ID, pseudonymous device reference, application version, event time and the minimal usage counters needed for enforcement. Review IP handling separately; do not collect persistent raw hardware inventories by default.
- Proposed retention: 30 days for raw security events, 90 days for aggregates, with documented exceptions for active incidents. Approve the policy before collection starts.
- Publish what is collected and why. Provide device removal, account recovery, accessible error messages and an appeal route.
- Start anomaly detection in observation mode. Do not permanently ban accounts on one signal; require corroboration and human review for consequential actions.
- On entitlement-service outages, preserve bounded local access where a valid lease permits it. Do not silently make premium server APIs fail-open.

## Validation and launch gates

- Security: invalid signatures, wrong audience/product/device, expired leases, replay, revocation, privilege boundaries, webhook forgery and concurrent seat races have automated coverage.
- Reliability: exercise service timeout, partial outage, key rotation, billing delay, renewal backoff, offline expiry and recovery. Set availability and latency targets after measuring the baseline.
- Privacy: inspect emitted events, verify deletion and retention, restrict operator access and complete the privacy review.
- Customer impact: test legitimate hardware changes, reinstall, travel, accessibility tools and device replacement; support can recover access without bypassing audit controls.
- Release: staged rollout with an entitlement enforcement kill switch controlled server-side, documented rollback and on-call ownership. Kill switches must not bypass server authentication or billing authorization.

## Measures of success

Establish a pre-enforcement baseline, then review weekly: activation success rate, entitlement API errors and latency, renewal failure rate, seat-reset requests, reviewed abuse incidents, false-positive rate, support burden and estimated revenue protected. Agree numerical targets before pilot; do not count blocked attempts alone as success.

## Operations and enforcement

Triage alerts → preserve minimal evidence → verify ownership and impact → review customer history → choose reversible mitigation → notify and offer appeal → escalate commercial redistribution to authorized counsel/providers where appropriate. Keep an audit record and revisit rules after false positives. Legal procedures differ by jurisdiction; this plan does not supply legal advice or automated notice templates.

## Open decisions

- Which features must remain usable fully offline, and for how long?
- What are the seat limits, transfer allowance and refund/revocation rules?
- Which platforms and media formats need support?
- Who owns key management, incident response, privacy review and customer appeals?
- What is the acceptable operating budget and measured latency envelope?

## Hosting and privacy

The GitHub repository and static website are public, as approved by the owner. GitHub Pages publishes from the `main` branch and `/docs` directory without a custom Actions workflow. No secrets or production security implementation belong in this documentation repository.
