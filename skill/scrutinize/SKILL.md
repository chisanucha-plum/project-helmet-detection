---
name: scrutinize
description: "Use this skill whenever the user asks to: review, audit, sanity-check, get second opinion, scrutinize, verify, validate, or deep-dive on a plan, PR, diff, design doc, or proposed code change. Trigger proactively on '/scrutinize' and when users want outsider-perspective verification that code actually does what it claims. Even if they just say 'double-check this' or 'does this look right', activate this skill."
---

# Scrutinize

Outsider-perspective end-to-end review that questions intent first, then verifies the change actually does what it claims by tracing real code paths.

## Critical rules

- **No rubber-stamps** - "LGTM" is not valid output. Always show what you traced and checked
- **Cite or it didn't happen** - Every claim references specific path, file, or line
- **Distinguish claim from verification** - Separate "PR says X" from "I verified X"
- **One simpler-alternative pass is mandatory** - Always ask if the change is necessary
- **Lead with structural problems** - Don't pad with style nits when there's a real issue
- **No flattery, no hedging** - State findings directly
- **Question scope first** - Skip only if user explicitly says "don't question scope"

## Workflow

Run these steps in order. Do not skip ahead.

### Operating stance

- **Outsider** - Forget who wrote it and why they think it's right. Read the artifact cold
- **End-to-end, not diff-local** - Follow the call graph through real code paths, not just the diff
- **Actionable, concise, with rationale** - Every finding states what to change, why, and what evidence led there

### 1. Intent — What is this actually trying to do?

- State the goal in one sentence. If you cannot, the artifact is underspecified — say so and stop
- Ask: **Is there a simpler, smaller, or more elegant way?** Consider:
  - Doing nothing (is the problem real/load-bearing?)
  - Using existing codebase features instead of adding new surface
  - A smaller change solving 90% with 10% of the risk
  - Solving at different layer (config vs code, framework vs app, build vs runtime)
- If better alternative exists, name it explicitly with rationale before line-by-line review

### 2. Trace — Walk the actual code path

- For each behavior the change claims, trace path end-to-end through real code (not just diff lines):
  - Entry point → call sites → branches taken → state mutated → exit/return/side effect
  - Include unchanged code on either side of diff. Bugs hide at seams
- For plan/design doc: trace proposed flow against existing system. Where does it touch reality?
- Note every surprise (unexpected branch, dead code reached, unknown state). Surprises are signal

### 3. Verify — Does it actually do what it claims?

For each claim:
- **Does the traced code path actually produce that behavior?** Walk explicitly: "Claims X. Path: A → B → C. At C, [observation]. Therefore [holds/doesn't hold]"
- **What inputs/states would break it?** Edge cases, concurrent callers, error paths, partial failures, retries, empty/null/unicode/huge inputs, ordering assumptions
- **What does it silently change?** Performance, error semantics, observability, contract for other callers, on-disk/on-wire format
- **How is it tested?** Do tests exercise the traced path, or pass while skipping it (mocks hiding bugs, asserts on intermediate state, happy path only)?

### 4. Report

Output one section per finding. Order by severity (blocker → major → nit):

**Per finding:**
- **Finding** — One sentence, specific. Cite file:line when applicable
- **Why it matters** — The consequence, not the principle
- **Evidence** — The trace step or input that exposes it
- **Suggested change** — Concrete, minimal

**Close with verdict:** ship / fix-then-ship / rework / reject — with single biggest reason

## Bundled resources

None - This skill contains all necessary review methodology inline.