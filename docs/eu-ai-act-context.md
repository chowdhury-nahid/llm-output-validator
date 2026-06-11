# Where Deterministic Verification Fits Under the EU AI Act

The EU AI Act (Regulation (EU) 2024/1689) is the first comprehensive AI law with
extraterritorial reach: it applies to any AI system whose output is used in the EU,
regardless of where the provider is established. Its high-risk obligations become
enforceable on 2 August 2026. For teams shipping LLM-powered features into
regulated domains — tax, finance, legal, healthcare — the Act changes what
"testing an AI feature" means.

This document explains how a deterministic verification layer like
`llm-output-validator` relates to the Act. It is an engineering perspective, not
legal advice.

## The key distinction: AI system vs. verification infrastructure

Article 3(1) defines an AI system as a machine-based system that *infers* how to
generate outputs from its inputs. An LLM answering tax questions is squarely
inside that definition.

A deterministic verification layer is not. It applies fixed, rule-bound checks —
schema contracts, citation grounding against a closed corpus, numeric boundary
tables, temporal consistency rules — with no inference step. The same input always
produces the same verdict. That places it **outside the Act's definition of an AI
system entirely**: it is compliance infrastructure that sits *next to* the
regulated system, not another regulated system.

This matters practically. Every verification capability you implement
deterministically is a capability that does not itself accumulate regulatory
obligations.

## Three articles this library maps to

### Article 14 — Human oversight

High-risk AI systems must be designed so that natural persons can effectively
oversee them, including the ability to "correctly interpret the high-risk AI
system's output" and to "decide not to use" it in any particular situation.

A human reviewer cannot meaningfully oversee raw LLM output at scale — plausible
text and hallucinated text look identical. A verification report
(pass / warn / fail per check, with machine-readable detail) converts oversight
from "read everything and hope" into "review what the contract flagged." The
`VerificationReport` produced by this library is exactly that artifact: a
structured basis for the human decision the Act requires.

### Article 15 — Accuracy, robustness and cybersecurity

High-risk systems must achieve "an appropriate level of accuracy, robustness and
cybersecurity" and perform "consistently in those respects throughout their
lifecycle." Two of this library's patterns map directly:

- **Golden regression checks** (Pattern 8) demonstrate consistency over the
  lifecycle: known-good inputs are re-verified on every run, so silent model or
  pipeline drift becomes a visible, dated test failure.
- **Prompt injection resilience** (Pattern 7) addresses the Act's explicit
  concern with "attempts by unauthorised third parties to alter [the system's]
  use, outputs or performance by exploiting system vulnerabilities."

### Article 12 — Record-keeping

High-risk systems must support automatic logging of events relevant to
identifying risks. A verification layer that emits a structured, timestamped
report for every LLM response — what was checked, what passed, what failed and
why — is a natural source for exactly those records. The JSON exporter in this
library produces audit-ready output for that purpose.

## Why deterministic, not LLM-as-judge

It is tempting to verify an LLM with another LLM. Under the Act that approach has
a structural cost: the judge is itself an AI system under Article 3(1), with its
own accuracy obligations, its own failure modes, and no way to prove consistency.
A deterministic layer gives you something an LLM judge cannot: **reproducible
evidence**. When a regulator, auditor, or enterprise customer asks "how do you
know the output was checked?", a rule-bound report with fixed semantics is an
answer; "a second model agreed" is a new question.

## Scope honesty

This library does not make a system compliant. Classification under the Act
(prohibited / high-risk / limited-risk / minimal-risk), conformity assessment,
technical documentation, and transparency disclosures are organisational and
legal obligations that no library can discharge. What a deterministic
verification layer provides is the *engineering substrate* several of those
obligations assume: structured oversight artifacts, lifecycle consistency
evidence, and event records.

## References

- [Regulation (EU) 2024/1689 — full text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)
- [Article 3: Definitions](https://artificialintelligenceact.eu/article/3/)
- [Article 12: Record-keeping](https://artificialintelligenceact.eu/article/12/)
- [Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/)
- [Article 15: Accuracy, robustness and cybersecurity](https://artificialintelligenceact.eu/article/15/)
