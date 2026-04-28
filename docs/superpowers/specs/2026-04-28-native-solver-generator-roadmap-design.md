# Design: Native Solver And Generator Roadmap Consolidation

**Status:** Implemented design, retained temporarily for consolidation review
**Date:** 2026-04-28
**Scope:** Native solver, generator, runtime specialization, compact state, and related planning documents.

## Problem

Native solver/generator direction is spread across many Markdown files: root-level refactor and progress notes, native compiler/generator plans, solver heuristic notes, performance specs, and implementation checklists. Some describe current reality, some are historical, and many are exploratory. The result is a planning surface that is hard for a human or agent to trust.

The cleanup goal is not to preserve old plans. The goal is to use old docs as raw information, then replace the active planning surface with a roadmap that reflects current code, current measurements, and the direction we choose now.

## Documentation Shape

`native/PLAN.md` becomes the canonical roadmap for native solver/generator/compiler direction. It should be the first and usually only planning document needed before working on native solver or generator optimization.

Supporting Markdown should remain only when it is factual and stable:

- command and navigation docs, such as `CPP_PORT.md` or `native/README.md`, if they still earn their keep;
- short factual appendices only when they reduce clutter in the roadmap.

Aspirational, exploratory, high-concept, or implementation-plan Markdown should be mined for useful facts, ideas, constraints, and measurements, then removed from the active planning surface. Old docs are inputs, not authorities.

## Roadmap Structure

The new `native/PLAN.md` should be a practical plan of action with these sections:

1. **Goal:** native solver/generator should be fast, deterministic, and easy to measure, with JS as the semantic oracle and the native interpreter as a first-class implementation and fallback.
2. **Target architecture:** define the state and execution vocabulary used by the roadmap.
3. **Current reality:** summarize only current facts: implemented paths, bridge/scaffold paths, useful coverage numbers, and visible bottlenecks.
4. **Execution roadmap:** ordered milestones from documentation consolidation through compact turn coverage, solver graph overhead, generator throughput, and heuristic work.
5. **Gates:** correctness and performance commands that must stay central.
6. **Temporary cleanup checklist:** files to mine and remove during consolidation. This section should shrink or disappear after cleanup.

## Target Architecture

JS remains the semantic oracle. Native behavior is ultimately judged against the existing JS compiler/runtime corpus.

The native interpreter remains first-class. It is fast enough to matter, directly comparable to JS, useful for debugging, and the native fallback/reference for generated paths.

Solver and generator graph state should be compact. The durable concept is `CompactState`: settled board occupancy plus complete RNG state. The roadmap should decide whether the current solver-local `SearchNodeState` name is temporary or preferred, but the architecture should not carry two overlapping concepts.

Hot turns should move toward specialized compact execution first. The long-term shape is:

```text
compact_state + input -> compact_state + TurnResult
```

Generated per-game compact turns should replace bridge/materialization paths where measurement shows the value, starting with hot solver/generator cases.

Specialized rulegroups remain useful. They improve the native interpreter path and are a migration aid, but they are not the deepest solver/generator destination. The deeper target is compact whole-turn execution for settled solver/generator nodes.

The generator should ride the solver path. It should use the same compact state and turn machinery unless measurement proves that a separate generator model is necessary.

Heuristic work is measurement-gated. The current wincondition heuristic remains the baseline until selection, reporting, timing, and compact/full scorer parity are in place.

## Execution Roadmap

1. **Consolidate planning docs.** Rewrite `native/PLAN.md` as the canonical roadmap, using old docs only as source material. Remove native aspirational/planning/high-concept docs after mining them.
2. **Lock vocabulary and state boundaries.** Decide names and responsibilities for `FullState`, compact solver/generator state, board occupancy, scratch, and turn results.
3. **Make compact-state execution coverage visible.** Track native compact kernels, interpreter bridge cases, unsupported/fallback reasons, and focus-corpus coverage.
4. **Expand specialized compact turns by hot path.** Replace bridge calls with native compact kernels for rule shapes and games that matter in solver/generator benchmarks, with parity and speed gates.
5. **Reduce solver graph overhead.** Keep measuring step, clone/materialization, hash, visited, node storage, heuristic, and unattributed time. Prefer compact storage/materialization reductions before deeper heuristic changes.
6. **Bring generator optimization onto the same measured path.** Use fixed-seed benchmarks and improve candidate evaluation, dedupe, top-K, and solver integration through shared compact/specialized runtime machinery.
7. **Improve heuristics after measurement scaffolding.** Add heuristic selection and reporting, lock baseline scoring, then try allocation/reachability/count heuristics behind named modes.
8. **Keep correctness gates central.** Every milestone should name the required JS/native parity, compact-turn, solver, generator, and performance gates.

## Cleanup Policy

Keep docs that describe current commands, APIs, or factual repo navigation.

Remove native aspirational/planning/high-concept docs after mining them for useful facts, constraints, measurements, and ideas.

Do not preserve old planning docs just because they contain a past decision. Re-decide important direction in the new roadmap based on current code and current goals.

Avoid long appendices. If an appendix remains, it should be factual, short, and actively useful.

Likely pre-existing files to mine and remove or simplify include historical root native notes, native compiler/generator/solver planning notes, and old native-related specs, plans, and reviews under this directory tree.

`CPP_PORT.md` and `native/README.md` should be reviewed as command/navigation docs, not roadmap docs.

Third-party/vendor Markdown and non-native user-facing docs are outside this cleanup scope.

## Gates For The Consolidation Work

Documentation-only consolidation should at minimum pass:

```sh
git diff --check
```

If the cleanup edits Makefiles, scripts, or source references, run the relevant native smoke gate as well.

The roadmap itself should name the ongoing engineering gates, including:

- JS simulation corpus;
- native simulation corpus;
- compact turn oracle/simulation tests;
- compact turn coverage;
- solver smoke and parity tests;
- generator smoke and fixed-seed benchmark;
- solver focus performance report.

## Risks

The main risk is accidentally treating old exploratory docs as binding commitments. The mitigation is to extract evidence and useful options, then write the roadmap from current code reality.

Another risk is making the roadmap too large. The mitigation is to keep `native/PLAN.md` strategic and actionable, and delete the temporary cleanup checklist once the consolidation is complete.

## Open Decisions For The Roadmap

- Whether the compact solver/generator state should be named `CompactState` in code, replacing or aliasing the current `SearchNodeState`.
- Whether any heuristic appendix is worth keeping after the roadmap extracts the measurement-gated heuristic milestone.
- Which command/navigation docs remain after `native/PLAN.md` becomes canonical.

## Self-Review

- Completeness scan: no incomplete sections.
- Consistency check: old docs are treated as source material, not binding decisions.
- Scope check: limited to native solver/generator/compiler planning docs; third-party/vendor and non-native user-facing docs stay out of scope.
- Ambiguity check: cleanup wording refers to pre-existing/old native planning docs, not this design artifact during review.
