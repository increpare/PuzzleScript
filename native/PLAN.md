# PuzzleScript Native Roadmap

## Goal

The native work exists to make PuzzleScript solving and generation fast, deterministic, measurable, and maintainable.

JavaScript remains the semantic oracle. The native interpreter remains first-class: it is useful, fast, directly comparable to JS, and the fallback/reference for generated native paths.

The solver/generator optimization target is compact, settled graph state plus generated per-game turn execution where measurement proves the value.

## Principles

- Current code reality beats old plans.
- Old exploratory docs are source material, not authority.
- Correctness gates come before speed claims.
- Keep the native interpreter healthy.
- Optimize measured hot paths, especially solver/generator workloads.
- Prefer shared solver/generator state and turn machinery until measurement proves a split is needed.
- Keep planning docs few, current, and actionable.

## Target Architecture

- The JS compiler/runtime corpus is the semantic oracle.
- The native interpreter is the native oracle/fallback and should remain performant.
- `FullState` is the full mutable native runtime state for interpreter, player, API, and broad debugging paths.
- Compact solver/generator graph state represents settled board occupancy plus complete RNG state. Current solver code names this concept `SearchNodeState`; the roadmap should decide whether code should call it `CompactState` or keep `SearchNodeState`.
- `BoardOccupancy` is the runtime object-major occupancy mirror used to bridge full runtime state and compact state work.
- Specialized rulegroups are useful for the interpreted native path and as migration machinery.
- Specialized compact turns are the deeper solver/generator target.
- Long-term hot path shape: `compact_state + input -> compact_state + TurnResult`.
- Generator candidate validation should use the same compact/specialized solver machinery unless measurement proves otherwise.
- Heuristics are selectable, reportable, and measurement-gated before becoming defaults.

## Current Reality

- The native compiler/runtime/player command exists as `puzzlescript_cpp`.
- The native interpreter, native compiler, SDL player integration, and C API exist and remain active surfaces.
- `FullState`, `BoardOccupancy`, `TurnResult`, and solver `SearchNodeState` exist in current code.
- Specialized rulegroup, specialized full-turn, and specialized compact-turn backend structures exist.
- Compact turn ABI, oracle, simulation, and coverage machinery exist.
- The compact turn bridge makes broad callable coverage possible while native compact kernels grow.
- Solver output has timing buckets for load, step, clone/materialization, hash, queue/frontier, visited lookup/insert, node storage, heuristic, solved/timeout checks, reconstruct, compact state bytes, and unattributed cost.
- Generator output includes aggregate `solver_totals`; fixed-seed generator benchmark machinery exists.
- Large old planning docs were mined as evidence and removed. Exact coverage and benchmark numbers should be regenerated with the commands below instead of copied from old notes.

## Execution Roadmap

### 1. Finish Documentation Consolidation

Purpose: keep native direction discoverable and trustworthy.

First actions: keep this file as the canonical roadmap; keep root/native README files factual; remove or update any reference that points at deleted planning notes.

Exit signal: native planning work starts from `native/PLAN.md`, while `CPP_PORT.md` and `native/README.md` stay command/navigation references.

### 2. Lock Vocabulary And State Boundaries

Purpose: remove overlapping names and unclear state ownership.

First actions: decide `CompactState` vs `SearchNodeState`; document exactly what belongs in compact graph state; audit materialization paths between `FullState`, `BoardOccupancy`, and solver graph nodes.

Exit signal: roadmap and code vocabulary describe one compact solver/generator state concept, including board occupancy, RNG, scratch, and turn-result boundaries.

### 3. Make Compact Execution Coverage Decision-Grade

Purpose: make compact execution progress visible enough to guide engineering time.

First actions: run `make compact_turn_coverage`; track native compact kernels, interpreter bridge cases, callable compact backends, unsupported/fallback reasons, and focus-corpus coverage.

Exit signal: coverage reports explain whether a solver/generator workload uses native compact kernels, bridge fallback, or unsupported paths.

### 4. Expand Specialized Compact Turns By Hot Path

Purpose: replace bridge/materialization work only where it matters.

First actions: rank bridge-backed cases by solver/generator benchmark impact; add native compact kernels for the highest-value rule shapes; verify each addition with compact oracle and simulation gates.

Exit signal: selected hot games run meaningful portions of `compact_state + input -> compact_state + TurnResult` without materializing `FullState`, and the change shows a measured benefit.

### 5. Reduce Solver Graph Overhead

Purpose: make search spend less time outside actual game stepping.

First actions: use solver focus reports to inspect step, clone/materialization, hash, visited, node storage, heuristic, and unattributed time; split materialization timing from clone timing if it becomes ambiguous; prefer compact storage/materialization reductions before deeper heuristic changes.

Exit signal: solver focus runs show lower graph overhead with no parity or determinism regression.

### 6. Bring Generator Optimization Onto The Same Measured Path

Purpose: make generation faster through the same solver/runtime improvements instead of a separate model.

First actions: run fixed-seed generator benchmarks; keep candidate evaluation, dedupe, top-K maintenance, and solver integration on shared compact/specialized machinery; report aggregate solver totals for every speed claim.

Exit signal: generator throughput improves on fixed inputs, with matching solver identities where relevant and no solver correctness regression.

### 7. Add Heuristic Improvements Behind Reporting And Gates

Purpose: improve search guidance without hiding behavior changes inside default modes.

First actions: add heuristic selection and reporting; keep the current wincondition heuristic as the baseline; compare compact/full scorer parity before trying allocation, reachability, count, or region heuristics.

Exit signal: each new heuristic is named, timed, selectable, and promoted only after measured benefit on the target corpus.

## Correctness Gates

Documentation-only cleanup starts with:

```sh
git diff --check
```

Broader gates should be selected based on the files changed. Native behavior changes should draw from:

```sh
node src/tests/run_tests_node.js --sim-only
make simulation_tests_cpp
make compact_turn_oracle_smoke
make compact_turn_simulation_tests
make solver_smoke_tests
make solver_parity_smoke
make solver_compact_parity_smoke
make generator_smoke_tests
```

## Performance Gates

Speed claims need before/after runs with fixed inputs, matching target identities where relevant, and no correctness regression.

```sh
make compact_turn_coverage
make solver_focus_compare
make solver_focus_perf_report SOLVER_FOCUS_RUNS=3
make generator_benchmark
```

## Documentation Cleanup

This consolidation is complete when historical native planning notes are removed, live references point here, and only factual command/navigation docs remain beside this roadmap.

Canonical native docs:

- `native/PLAN.md`: current solver/generator/compiler roadmap.
- `CPP_PORT.md`: root-level quick command and navigation reference.
- `native/README.md`: native-local command and navigation reference.

The current roadmap consolidation design spec may remain temporarily as review context. Once this roadmap is accepted, remove that spec too if the repository should have exactly one native planning source of truth.

## Open Decisions

- Should the compact solver/generator state be named `CompactState` in code, replacing or aliasing the current `SearchNodeState`?
- Does any solver heuristic note remain useful as a short factual appendix, or should all heuristic direction live in the roadmap?
- Should command/navigation content live in both `CPP_PORT.md` and `native/README.md`, or should one point to the other?
