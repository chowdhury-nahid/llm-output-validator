# EU AI Act Risk Classification Guide for Enterprise AI in Tax and Finance

**Regulation (EU) 2024/1689 — Practitioner reference for engineering and compliance teams**

_This document is a practitioner's guide, not legal advice. It reflects the state of
the law as of June 2026. The Digital Omnibus proposal (Commission, November 2025) is
under trilogue and is not treated as enacted. Seek qualified legal counsel before
finalising any classification position._

---

## 1 — Legal Basis, Territorial Scope, and Enforcement Timeline

### 1.1 What the Act is and who it applies to

Regulation (EU) 2024/1689 — the EU AI Act — entered into force on 1 August 2024
following publication in the Official Journal of the European Union on 12 July 2024.
It is directly applicable law across all EU Member States without national
transposition.

The territorial scope, set out in Article 2(1), extends to:

- AI systems **placed on the EU market** or **put into service in the EU**, regardless
  of where the provider is established
- Providers and deployers **established outside the EU** where their AI systems'
  outputs are used within the EU

This extraterritorial reach is the defining characteristic for global technology
organisations. A company headquartered in the United States, United Kingdom, or
anywhere outside the EU is nonetheless subject to the Act if its AI system is
used by EU-based clients or users. There is no sector carve-out for financial
services or professional software.

### 1.2 Dual roles: provider and deployer

Article 3 creates distinct roles that carry different obligations. Many organisations
in enterprise software occupy both simultaneously:

**Provider (Article 3(3)):** The organisation that develops an AI system and places
it on the market under its own name. Provider status triggers the heavier set of
obligations: technical documentation, conformity assessment (if high-risk), quality
management system, post-market monitoring, and EU database registration.

**Deployer (Article 3(4)):** An organisation that uses an AI system under its own
authority — including, critically, an organisation that integrates an upstream
large language model into its own product. Deployer obligations include human
oversight measures, input data governance, user disclosure, and potentially a
fundamental rights impact assessment under Article 27.

**The reclassification risk:** Article 25(1) provides that where a deployer
substantially modifies a high-risk AI system or alters its intended purpose beyond
what the original provider contemplated, that deployer becomes a provider for
compliance purposes. Engineering teams integrating LLMs into products should
document their integration architecture specifically to address this question.

### 1.3 Enforcement timeline

| Date | Milestone | Status (June 2026) |
|---|---|---|
| 1 Aug 2024 | Entry into force. 24-month compliance clock began. | ✓ Past |
| 2 Feb 2025 | Art. 5 prohibited practices enforceable. Art. 4 AI literacy obligation live. | ✓ Active — live exposure |
| 2 Aug 2025 | GPAI obligations (Arts. 53–55) fully applicable. Art. 99 penalty framework operative — fines up to €35M / 7% global turnover are enforceable **now**. | ✓ Active — live exposure |
| 2 Aug 2026 | Full high-risk AI obligations enforceable. Arts. 8–15, Art. 26 deployer obligations, Art. 50 transparency, Art. 71 EU database registration. | ⚑ Primary deadline |
| 2 Aug 2027 | AI in harmonised-legislation products (medical devices, machinery). Not relevant to most enterprise software. | Pending |

**The enforcement gap most organisations miss:** the Article 99 penalty framework
became operative on 2 August 2025 — not 2026. Fines for violations of GPAI
obligations and prohibited practices are live today, not on the horizon.

### 1.4 Penalty structure

| Violation category | Maximum fine |
|---|---|
| Prohibited AI practices (Art. 5) | €35M or 7% of worldwide annual turnover |
| High-risk / transparency violations (Arts. 16, 26, 50) | €15M or 3% of worldwide annual turnover |
| Incorrect information to authorities | €7.5M or 1% of worldwide annual turnover |

For SMEs, Article 99(6) applies the lower of the absolute figure and the percentage —
a mitigation factor, not an exemption. Penalty exposure scales with revenue.

### 1.5 The Digital Omnibus proposal — why not to plan around it

The European Commission's Digital Omnibus package (November 2025) includes a
proposal to extend certain Annex III high-risk deadlines from 2 August 2026 to as
late as December 2027, conditional on harmonised technical standards not being
finalised. As of June 2026, this proposal is under trilogue negotiation and has not
been enacted.

**Treating a non-enacted proposal as a de facto extension is the highest-probability
compliance failure mode.** The compliance infrastructure required — risk management
systems, technical documentation, human oversight protocols, logging — takes months
to build correctly. If the Omnibus is subsequently enacted, it provides additional
runway. It is a contingency, not a plan.

---

## 2 — Classifying Your AI System: Definitions and the Mandatory Sequence

### 2.1 What counts as an AI system under the Act

Article 3(1) defines an AI system as:

> "a machine-based system that is designed to operate with varying levels of autonomy
> and that may exhibit adaptiveness after deployment, and that, for explicit or
> implicit objectives, infers, from the input it receives, how to generate outputs
> such as predictions, content, recommendations, or decisions that can influence
> physical or virtual environments."

The operative word is **infers**. A system that derives outputs from inputs through
a learning process — an LLM, a recommendation engine, a classification model — is
an AI system. The scope is deliberately broad.

**What is not an AI system:**

- Rule-based software that applies fixed logic to structured inputs (a statutory
  calculation engine applying published formulae, a document classifier using
  hardcoded rules)
- Deterministic verification infrastructure that checks AI outputs against
  predefined contracts (schema validators, citation grounding checks, numeric
  boundary rules)
- Static data products (calendars, rate tables, entity registries) with no
  inference layer

This exclusion is practically significant. Where a team builds deterministic
controls _around_ an AI system — to verify its outputs, log its decisions, or
constrain its downstream effect — those controls are compliance infrastructure, not
regulated AI. They do not themselves accumulate obligations under the Act. The
engineering corollary: every capability you implement deterministically is a
capability outside the Act's classification scope.

### 2.2 The mandatory classification sequence

The Act's risk framework is a strict sequence. There is no shortcut to a lower
tier — each step must be completed in order:

```
Step 1 — Article 5: Prohibited practice?
          │
          ├─ YES → Stop. Do not deploy. This use case is unlawful.
          │
          └─ NO  ↓

Step 2 — Annex III / Article 6: High-risk AI system?
          │
          ├─ YES → Full Chapter III obligations apply (Arts. 8–17, 19, 72).
          │        Conformity assessment. EU database registration.
          │
          ├─ ART. 6(3) EXCEPTION → Document the exclusion (Art. 6(4)).
          │                         Register under Art. 49(2).
          │
          └─ NO  ↓

Step 3 — Article 50: Transparency obligations?
          │
          ├─ YES → Disclose AI interaction (Art. 50(1)) and/or
          │        mark AI-generated content (Art. 50(2)/(4)).
          │
          └─ NO  ↓

Step 4 — Minimal risk
          No specific Act obligations beyond Art. 4 AI literacy
          and Art. 99 penalty framework (already live).
```

**Each AI system in your product inventory must be assessed separately.** The
classification of one system does not carry over to another, even within the same
product suite. A conversational assistant and a calculation engine that share an
infrastructure layer receive independent classifications.

### 2.3 Conducting your AI system inventory

Before classification, you need an inventory. For each item, answer three questions:

1. Does it meet the Article 3(1) definition? (Does it *infer* outputs, or does it
   apply fixed rules?)
2. Is it user-facing, or is it an internal tool? (Affects Art. 50 scope and
   the Art. 6(3) exception analysis)
3. Can its output reach a regulatory authority or materially influence a decision
   with legal effect, directly or indirectly?

Question 3 is the determining question for enterprise tax and financial AI. An
output that feeds a mandatory regulatory filing — even indirectly, through a
human review step — sits in different territory from an output that informs an
internal discussion. The quality and independence of the human review at the point
of filing is the architectural question the Act cares about, not merely whether a
human is nominally present.

---

## 3 — Step 1: Article 5 Prohibited Practices Elimination Test

Article 5(1) lists eight categories of AI practice that are **unconditionally
prohibited** from 2 February 2025. This test must be completed first, for every
system. A prohibited system cannot be made compliant by adding oversight or
documentation — it cannot be deployed.

| Art. 5(1) | Prohibited practice | Applies to enterprise tax/financial AI? |
|---|---|---|
| (a) | Subliminal or manipulative techniques to distort behaviour | No — systems operate on regulatory data, not behavioural influence |
| (b) | Exploitation of vulnerabilities (age, disability, economic precarity) | No — users are qualified professionals in enterprise settings |
| (c) | Social scoring by public authorities | No — no evaluation of individuals based on social behaviour |
| (d) | Criminal risk prediction for law enforcement | No — no predictive policing or criminality profiling |
| (e) | Untargeted facial recognition scraping | No — no biometric data ingestion |
| (f) | Emotion recognition in workplace or education | No — no inference from user emotional state |
| (g) | Biometric categorisation on sensitive attributes | No — no biometric processing |
| (h) | Real-time remote biometric identification | No — no identification or surveillance function |

**For enterprise tax and financial AI, all eight sub-paragraphs are typically
inapplicable.** The structural reason is that this category of AI operates on
regulatory reference data, financial inputs, and structured domain knowledge — not
on natural person behaviour, biometrics, or social attributes. Article 5 was
designed for surveillance, social control, and manipulative consumer AI; those
purposes are not present in tax or financial workflow automation.

### 3.1 The one case to watch carefully

Sub-paragraph (b) — exploitation of vulnerabilities — is the most likely candidate
for misapplication in financial contexts. It does not apply where:

- Users are trained professionals (accountants, tax lawyers, compliance officers)
  making deliberate, informed use of the system
- The AI system provides information or analysis, not persuasion targeted at
  financial precarity

It could apply to a consumer-facing AI product that uses financial stress signals
to influence purchasing or borrowing decisions. Enterprise B2B tools serving
professional users are outside this scope.

### 3.2 What to document

Even where the Article 5 conclusion is unambiguous, it should be documented. Write
one paragraph per sub-paragraph explaining why it does not apply, with reference
to the system's design purpose and user base. This is the foundation of your
classification record under Article 6(4), and it is the first page a regulator
will ask for.

---

## 4 — Step 2: Annex III High-Risk Analysis for Enterprise Tax and Financial AI

### 4.1 The eight Annex III categories

Under Article 6(2), AI systems listed in Annex III are presumed high-risk. There
are eight categories. Seven of them are inapplicable to virtually all enterprise
tax and financial AI for structural reasons:

| Annex III Category | Scope | Applies to enterprise tax/financial AI? |
|---|---|---|
| 1 — Biometrics | Remote identification, categorisation, emotion recognition | No — no biometric processing in standard tax/financial workflows |
| 2 — Critical infrastructure | Safety components of energy, transport, water, digital infrastructure | No — financial software is not a safety component of physical infrastructure |
| 3 — Education and vocational training | Access to education, learning assessment, student monitoring | No — enterprise tax tools serve professional users, not educational contexts |
| 4 — Employment and worker management | Recruitment, promotion, termination, performance monitoring of individuals | Peripheral — see Section 4.2 |
| 5 — Essential private and public services | Benefits eligibility, credit scoring, insurance pricing, emergency dispatch | **Critical for tax AI — see Section 4.3** |
| 6 — Law enforcement | Crime prediction, criminal profiling, polygraphs, evidence reliability | No — enterprise tax tools do not serve law enforcement functions |
| 7 — Migration and border management | Asylum assessment, border surveillance, travel document verification | No — no immigration or border function |
| 8 — Administration of justice and democracy | Judicial decisions, electoral influence | No — tax tools do not make judicial decisions |

### 4.2 Category 4 — Employment: the internal tool boundary

Category 4 covers AI used for decisions about workers: recruitment, task
allocation based on individual behaviour, performance monitoring, termination.
This category is relevant where an organisation deploys an AI tool internally
to assess its own staff.

**The boundary that matters:** Category 4 applies to AI that assesses *natural
persons in their capacity as workers* — their behaviour, performance, or
suitability. An internal AI tool that assesses *code objects, document quality,
or risk signals in business artefacts* — and produces advisory output for human
review — is outside this category, even if its outputs are used to allocate work.

The engineering design that preserves this boundary: no author identity in the
assessment input, no per-person scoring, and a mandatory human decision step
that is structural (not cosmetic). The Art. 6(3)(c) exception — systems that
detect patterns for human review without replacing human assessment — provides a
secondary defence.

### 4.3 Category 5 — The critical classification question for tax AI

Category 5 is the most contested category for enterprise tax and financial AI.
It covers:

- AI used by public authorities to evaluate eligibility for public assistance
- AI assessing creditworthiness or establishing credit scores for **natural persons**
  (explicit carve-out for fraud detection)
- AI for risk assessment and pricing in life and health insurance
- AI evaluating and classifying emergency calls

**The central argument against high-risk classification for B2B enterprise tax AI:**

Every concrete sub-item in Annex III Category 5 explicitly targets **natural
persons**. Creditworthiness scoring applies to natural persons; benefit eligibility
applies to natural persons; emergency dispatch and insurance pricing concern natural
persons.

Enterprise tax tools typically serve multinational corporations — legal entities,
not natural persons. The client impact chain runs through legal entities. The
European Parliament's explanatory materials confirm that Category 5 does not extend
to AI used in a "sole business-to-business environment without foreseeable impact
on natural persons."

**The strongest counterarguments to understand (to rebut them):**

1. *Regulatory submission nexus:* Mandatory filings submitted to tax authorities
   on the basis of AI-generated figures could be argued to "materially influence"
   the enterprise's legal relationship with a public body — a relationship with
   "essential" characteristics. The response: the Act's enumerated sub-items
   control over the category heading; the natural persons scope gate is textual, not
   interpretive.

2. *Expansive reading of "essential public services":* Some commentators argue
   Category 5's title is broader than its sub-items. The response: regulatory
   interpretation follows the legislative text; the sub-items are the operative
   scope.

3. *AI influence on regulatory outcomes:* Even if clients are legal entities, errors
   in AI-generated filing data could affect the enterprise's compliance standing.
   The response: this conflates the AI's advisory role with decisional authority;
   where qualified tax professionals review and take responsibility for filings, the
   AI output is an input to human judgment, not a substitute for it.

**Classification position:** For B2B enterprise tax AI whose clients are legal
entities, the primary ground for exclusion from Category 5 is textual — the
natural persons scope gate. The Art. 6(3)(a) narrow procedural task exception
(applying fixed regulatory formulae to structured inputs) and Art. 6(3)(b)
improvement of completed human activity exception provide secondary grounds for
calculation and filing tools specifically.

### 4.4 The Art. 6(3) exception pathway — costs and requirements

If a system falls within an Annex III category but the provider claims it is not
high-risk under Art. 6(3), two mandatory obligations attach under Art. 6(4):

1. The provider must **document the exception assessment** before the system is
   placed on the market or put into service
2. The provider is subject to **registration obligations** under Article 49(2)

The documentation obligation applies even where the primary classification position
is scope exclusion (the natural persons argument) rather than a claimed exception.
A written, pre-deployment classification record — documenting the scope exclusion
reasoning for any Annex III-adjacent features — is appropriate risk management. It
costs relatively little and provides immediate evidence of good-faith engagement
with the Act if a national authority ever inquires.

### 4.5 Summary classification table

The following table applies the four-step sequence to the common feature types in
enterprise tax and financial AI:

| Feature type | Art. 5 | Annex III best-fit | High-risk? | Art. 50 obligations |
|---|---|---|---|---|
| Conversational AI assistant (research Q&A) | Pass | Cat. 5 peripheral — legal entity clients | No | Art. 50(1) chatbot disclosure at first interaction |
| Scenario and computation engine (tax calculations) | Pass | Cat. 5 contested — deterministic rule application, legal entity clients | No (Art. 6(3)(a) secondary) | None unless chat interface added |
| Regulatory filing calculation engine | Pass | Cat. 5 contested — strongest case; BEPS arithmetic on structured inputs | No (Art. 6(3)(a) strongest) | None unless output published publicly |
| Document generation (memos, reports, forms) | Pass | None clearly applicable | No | Art. 50(2) machine-readable marking applies to provider; Art. 50(4) if published publicly |
| Internal LLM-assisted development tool (e.g. PR risk review) | Pass | Cat. 4 peripheral — assesses code objects not workers; Art. 6(3)(c) applies | No | None — not user-facing or externally deployed |
| Deterministic verification layer | N/A — not an AI system under Art. 3(1) | N/A | N/A | None |

---

## 5 — Step 3: Article 50 Transparency Obligations

Article 50 applies as a **parallel track independent of risk classification**. A
system that passes the Article 5 test and falls outside all Annex III categories
may still carry Article 50 obligations. All obligations in this article become
enforceable on 2 August 2026.

### 5.1 Article 50(1) — Conversational AI disclosure

Providers must ensure that AI systems intended to interact directly with natural
persons are designed so those persons are informed that they are interacting with
an AI system, before the interaction begins.

**Exception:** disclosure is not required if the AI nature of the interaction is
"obvious from the point of view of a natural person who is reasonably well-informed,
observant and circumspect, taking into account the circumstances and the context
of use." This exception is interpreted narrowly — a branded product name alone
does not typically suffice.

**What this means in practice:**

- Any conversational AI assistant (chat interface, Q&A tool) that interacts
  directly with users must display a clear disclosure before the first user input
- The disclosure must be explicit — "You are interacting with an AI assistant"
  or equivalent — not implied by UI design
- It must appear at first interaction, not buried in terms of service

**Cost of non-compliance:** up to €15M or 3% of global annual turnover under
Article 99. The fix is technically trivial — a persistent label and a pre-load
message. There is no engineering reason to leave this unresolved.

### 5.2 Article 50(2) — Machine-readable marking of AI-generated content

Providers of AI systems that generate synthetic content (audio, images, video,
text) must ensure outputs are marked in a machine-readable format as
artificially generated or manipulated.

**Scope for enterprise AI:** this obligation attaches at the **provider level** —
the upstream LLM provider is the primary obligor for marking its outputs. However,
as a downstream integrated system provider, the organisation building on top of
the LLM carries a secondary obligation to ensure the marking mechanism is
preserved and surfaced appropriately.

**Practical consequence:** if your product exports AI-generated documents (reports,
memos, form outputs) to users who may use them in downstream contexts, confirm
with your LLM vendor whether their outputs include machine-readable AI provenance
markers. If not, this is a vendor obligation gap to document and track.

### 5.3 Article 50(4) — Text disclosure for AI-generated public information

Where a deployer uses a text-generating AI system to produce information intended
to inform the public on matters of public interest, the deployer must disclose
that the text is AI-generated — unless a human reviewer has assumed editorial
responsibility for the content.

**Scope for enterprise B2B AI:** in a pure internal enterprise workflow, this
obligation has limited reach. It is triggered by **publication** of AI-generated
content to a public audience on matters of public interest. Enterprise tax memos
used internally or transmitted to clients under professional review do not
typically cross this threshold. The obligation activates if AI-generated content
is published externally — a blog post, a regulatory commentary, a public bulletin
— without a named human editorial reviewer taking responsibility.

**Practical action:** establish a policy on whether AI-generated content can be
published externally, and under what human editorial process. The policy protects
both the organisation and its users.

### 5.4 Article 50(3) — Emotion recognition and biometric categorisation

Providers and deployers of emotion recognition systems or biometric
categorisation systems must inform users. Not applicable to standard enterprise
tax and financial AI.

---

## 6 — GPAI Obligations: If You Use an Upstream Large Language Model

For organisations that integrate a third-party frontier LLM (GPT-4, Claude,
Gemini, or equivalent) into their product, a separate and earlier obligation set
applies under Chapter V of the Act. These obligations became enforceable on
**2 August 2025** — they are not part of the August 2026 deadline.

### 6.1 Your role: downstream integrated system provider

Under Article 25, an organisation that integrates a GPAI model into its own
AI system and places that system on the EU market is a **downstream integrated
system provider**. This creates obligations that are distinct from and additional
to those of the GPAI model provider (the LLM vendor). The LLM vendor's Chapter V
compliance does not discharge the downstream provider's system-level obligations.

**Obligations you carry as downstream provider (independent of the LLM vendor):**

| Obligation | Legal basis |
|---|---|
| Technical documentation for your AI system (not the model) | Art. 18, Annex IV |
| Instructions for use — capabilities, limitations, known failure modes | Art. 13 |
| Human oversight mechanisms built into the product | Art. 14 |
| Accuracy, robustness, cybersecurity at system level | Art. 15 |
| Post-market monitoring (if any feature is confirmed high-risk) | Art. 72 |

### 6.2 What to demand from your LLM vendor

The LLM vendor carries Article 53 obligations as a GPAI model provider. These
are their obligations — but your ability to discharge your own downstream
documentation depends on receiving evidence that they are being met. The
following checklist covers what you should request contractually:

| Document | Legal basis | Priority |
|---|---|---|
| Annex XI technical documentation (architecture, training, evaluation, known limitations) | Art. 53(1)(a) | Obtain immediately — live obligation since Aug 2025 |
| Annex XII downstream integration guide (capabilities, limitations, acceptable use, known failure modes) | Art. 53(1)(b) | Obtain immediately — basis for your own instructions-for-use |
| Copyright compliance policy (CDSM Directive compliance, opt-out mechanism) | Art. 53(1)(c) | Obtain immediately |
| Training data summary URL (publicly filed with AI Office) | Art. 53(1)(d) | Verify and retain |
| 10-year documentation retention confirmation | Art. 53(1)(e) | Confirm in contract |
| Systemic risk designation status under Art. 51 | Art. 51(1) | Confirm and retain |
| Adversarial testing / red-team results summary (if systemic risk model) | Art. 55(1)(a) | Request evidence of completion |
| EU authorised representative details (if vendor is non-EU) | Art. 54 | Verify representative is appointed |

### 6.3 The fine-tuning boundary

Commission Guidelines (July 2025) established when downstream modification of a
GPAI model elevates a downstream actor from integrated system provider to GPAI
model provider — triggering full Chapter V obligations.

The indicative threshold is compute-based: the modification compute must exceed
one-third of the compute used to train the original model. In practice, this
threshold is not reached by:

- System prompt engineering
- Few-shot prompting and prompt templates
- RAG pipelines over a domain knowledge base
- Standard instruction fine-tuning runs

If you have conducted any full fine-tuning runs against a proprietary corpus,
the compute should be measured and documented — even if below the threshold —
to demonstrate the question has been assessed.

---

## 7 — Compliance Action Checklist

The following actions are organised by urgency. The first two columns address
**live obligations** (enforceable now). The third addresses the August 2026
deadline.

### Quick wins — no engineering required (do now)

| Action | Legal basis | Owner |
|---|---|---|
| Send formal written request for Annex XII documentation to your LLM vendor | Art. 53(1)(b) | Legal / CTO |
| Confirm LLM vendor's systemic risk designation and Art. 55 compliance status | Art. 51, Art. 55 | Legal |
| Confirm EU authorised representative has been appointed by vendor | Art. 54 | Legal |
| Implement AI literacy programme for all staff who interact with AI systems | Art. 4 (live since Feb 2025) | People / Leadership |

### One-sprint tasks — before 2 August 2026

| Action | Legal basis | Owner |
|---|---|---|
| Add persistent AI disclosure to any conversational AI interface (pre-load message, first interaction) | Art. 50(1) | Product / Engineering |
| Add Art. 50(2) AI-generated content labelling to any exported document outputs | Art. 50(2) | Engineering |
| Produce written classification memo for any Annex III-adjacent features (2–4 pages; document the natural persons scope exclusion and Art. 6(3) analysis) | Art. 6(4) | Legal (lead), CTO (approval) |

### Structural programme — gate on classification memo output

| Action | Legal basis | Scope trigger |
|---|---|---|
| Technical documentation (Annex IV) for each AI system | Art. 11, 18 | Required for all AI systems; critical if any feature confirmed high-risk |
| Quality management system (QMS) | Art. 17 | Required if any feature confirmed high-risk |
| AI event logging and audit trail infrastructure | Art. 12, 19 | Required for high-risk; best practice regardless |
| Instructions for use (capability and limitation disclosures) for enterprise clients | Art. 13 | Required for all AI systems placed on the market |
| Post-market monitoring plan | Art. 72 | Required if any feature confirmed high-risk |

---

## 8 — References

| Article | Subject | Link |
|---|---|---|
| Full text | Regulation (EU) 2024/1689 | [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) |
| Art. 3 | Definitions | [artificialintelligenceact.eu/article/3](https://artificialintelligenceact.eu/article/3/) |
| Art. 5 | Prohibited practices | [artificialintelligenceact.eu/article/5](https://artificialintelligenceact.eu/article/5/) |
| Art. 6 | Classification rules for high-risk AI | [artificialintelligenceact.eu/article/6](https://artificialintelligenceact.eu/article/6/) |
| Annex III | High-risk AI system categories | [artificialintelligenceact.eu/annex/3](https://artificialintelligenceact.eu/annex/3/) |
| Art. 12 | Record-keeping / logging | [artificialintelligenceact.eu/article/12](https://artificialintelligenceact.eu/article/12/) |
| Art. 13 | Transparency and instructions for use | [artificialintelligenceact.eu/article/13](https://artificialintelligenceact.eu/article/13/) |
| Art. 14 | Human oversight | [artificialintelligenceact.eu/article/14](https://artificialintelligenceact.eu/article/14/) |
| Art. 15 | Accuracy, robustness, cybersecurity | [artificialintelligenceact.eu/article/15](https://artificialintelligenceact.eu/article/15/) |
| Art. 25 | Responsibilities along the AI value chain | [artificialintelligenceact.eu/article/25](https://artificialintelligenceact.eu/article/25/) |
| Art. 50 | Transparency obligations | [artificialintelligenceact.eu/article/50](https://artificialintelligenceact.eu/article/50/) |
| Art. 53 | GPAI model provider obligations | [artificialintelligenceact.eu/article/53](https://artificialintelligenceact.eu/article/53/) |
| Art. 99 | Penalties | [artificialintelligenceact.eu/article/99](https://artificialintelligenceact.eu/article/99/) |
| Recital 53 | Narrow procedural task exception guidance | [artificialintelligenceact.eu/recital/53](https://artificialintelligenceact.eu/recital/53/) |
