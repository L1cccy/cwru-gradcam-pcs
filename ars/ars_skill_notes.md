# ARS (Academic Research Skills) Skill Notes

## 1. What can ARS help with?

ARS helps automate the academic research-to-publication workflow:

- **Research** (Stage 1): Systematic literature search, source verification, cross-source synthesis, risk of bias assessment, meta-analysis, and APA 7.0 report compilation
- **Writing** (Stage 2): Paper drafting with structured outlines, argument construction, bilingual abstracts, citation compliance, and LaTeX/DOCX/PDF formatting
- **Integrity verification** (Stages 2.5 & 4.5): 100% reference and citation verification, claim-evidence alignment audit, AI research failure mode checklist
- **Peer review** (Stage 3): Multi-perspective review simulation (5 reviewers: EIC, methodology, domain, perspective, Devil's Advocate), editorial decision, and revision roadmap
- **Revision** (Stage 4): Targeted revision with patch protocol, revision coaching via Socratic dialogue
- **Re-review** (Stage 3'): Verification review checking whether revisions addressed reviewer concerns
- **Finalization** (Stage 5): Format conversion (MD/LaTeX/DOCX/PDF), process summary documentation

## 2. What can ARS not do?

- ARS cannot **invent results or citations** — it has integrity gates that block fabricated or unverifiable content
- ARS cannot **make scientific judgments** — the human researcher must decide the research question, dataset choice, split design, experimental evidence, and interpretation
- ARS cannot **supply research creativity** — it structures and accelerates workflow but does not replace the researcher's domain expertise
- ARS cannot **validate physical meaning** — mechanism validation and physical interpretation remain the student's responsibility
- ARS cannot **guarantee correctness** — it checks citation existence and claim-evidence alignment, but cannot verify whether a method is appropriate for a given problem

## 3. What are its integrity gates?

ARS has multiple mandatory integrity checkpoints:

- **Stage 2.5 (Pre-review integrity)**: 5-phase verification — references, citation context, statistical data, originality, claims. Also runs the 7-mode AI Research Failure Mode Checklist (Mode 1-7: citation hallucination, implementation bugs, hallucinated results, shortcut reliance, bug-as-insight, methodology fabrication, pipeline-level frame-lock)
- **Stage 4.5 (Final integrity)**: Full re-verification from scratch (not just re-checking Stage 2.5 findings). Must PASS with zero issues to proceed
- **Claim audit (opt-in, v3.8)**: L3 claim-faithfulness audit at Stage 4→5 transition, checking sampled citations for claim-reference alignment and negative-constraint compliance
- **Process Summary (Stage 6)**: Final process record documenting collaboration quality, AI self-reflection report, and failure-mode audit log

Blocking rules:
- Integrity FAIL at Stage 2.5/4.5 blocks pipeline progression
- SUSPECTED or INSUFFICIENT EVIDENCE in Modes 1/3/5/6 of the failure mode checklist blocks progression with no `--no-block` escape hatch
- Critical-severity issues in Devil's Advocate review make Accept impossible

## 4. How does it reduce hallucinated citations or unsupported claims?

- **Citation verification**: Every citation must be verified via DOI or WebSearch — references that cannot be confirmed are flagged as FAIL, not marked as "uncertain"
- **Vibe citing anti-pattern**: Explicit prohibition against mixing elements from 2-3 real papers into a fabricated reference — every reference must be independently verified
- **Claim-evidence audit (v3.8)**: Sampled citations are audited for actual alignment between the claim made and the source content
- **Integrity gates block progression**: Stage 2.5 and 4.5 are mandatory blocking gates — if any unverifiable reference or unsupported claim is found, the pipeline stops
- **Material Passport**: All artifacts are versioned, hashed, and auditable, creating a traceable chain from evidence to claim
- **Stage 6 process summary**: The AI self-reflection report includes the full failure-mode audit log, ensuring transparency

## 5. What must the human researcher still decide?

The human researcher must supply and decide:

1. **Research question**: What problem to study, what bottleneck to address
2. **Dataset choice**: Which datasets to use (CWRU + additional CWRU-like data)
3. **Split design**: How to segment data and split train/validation/test to prevent leakage
4. **Method selection**: Which method family, architecture, and parameters to use
5. **Experimental evidence**: Running experiments, collecting results
6. **Scientific interpretation**: Whether results support the hypothesis, what limitations exist
7. **Claim boundaries**: How strongly to state each claim, what caveats to include
8. **Verified citations**: Ensuring cited sources actually exist and support the argument
9. **Physical mechanism validation**: Verifying that the method works for the right physical reasons
10. **Conclusion discipline**: Accepting negative results, avoiding overclaiming

> Rule: "No evidence, no claim." — the human decides what counts as evidence and what claim it supports.
