# Refactor Implementation Checklist

This checklist turns `Refactor.md` into a sequence of small, testable changes. Each section should land as one or more focused commits. Prefer boring mechanical commits with clear test output over clever broad rewrites.

The target terminology is:

- `FullState` / `CompactState`
- `InterpretedTurn` / `SpecializedTurn`
- `InterpretedRulegroups` / `SpecializedRulegroups`
- `AgainPolicy::Yield` / `AgainPolicy::Drain`

The main optimization target remains:

```text
CompactState + SpecializedTurn + SpecializedRulegroups + AgainPolicy::Drain
```

## Ground Rules

- [x] Do not mix large renames with semantic changes unless the semantic change is necessary to keep behavior identical.
- [x] Stage only files touched by the current checklist item.
- [x] Commit after each coherent green slice.
- [ ] Keep compatibility aliases only while they are actively helping a staged migration.
- [ ] Remove aliases before declaring a rename track complete, unless a compatibility decision explicitly keeps them.
- [ ] Keep `PS_INPUT_TICK` unchanged as a PuzzleScript input value.
- [ ] Keep complete RNG state in `CompactState` identity.
- [ ] Keep transient movement words out of solver/generator `CompactState`.
- [ ] Treat `restart` as a terminal failed edge for solver/generator compact-state work.
- [ ] Treat `again` as drained before solver/generator graph insertion.

## Baseline Snapshot

- [x] Record current branch and dirty-state context with `git status --short`.
- [x] Confirm whether local `ProgressReport.md` changes are user-owned before touching it.
- [x] Run `make build` if the build is not known-good.
- [x] Run `make solver_smoke_tests`.
- [x] Run `make solver_compact_parity`.
- [ ] Run `make compact_tick_coverage` and record callable/native/bridge counts.
- [ ] Run `make compact_tick_simulation_tests` if iteration time is acceptable.
- [ ] Save baseline notes in the first implementation commit message or a short doc update.

## Phase 1: Rename Solver Compact State

Goal: replace the misleading solver-local name `CompactSolverState` with `CompactState` without changing behavior.

- [x] Rename `struct CompactSolverState` to `CompactState`.
- [x] Update solver node storage from `CompactSolverState compact` to `CompactState compact`.
- [x] Update function signatures that accept or return the solver compact state.
- [x] Keep `compactStateKey` name unchanged.
- [x] Rename `compactStateFromSession` only if the `FullState` rename has already landed; otherwise leave it for the state rename phase.
- [x] Rename local variables only where it improves clarity and does not obscure the diff.
- [x] Confirm `CompactState` still contains object occupancy bitsets and complete RNG state.
- [x] Confirm `CompactState` still does not contain movement words.
- [x] Confirm equality includes object bitsets and full RNG state.
- [x] Confirm `byteSize()` includes object bitsets and full RNG state.
- [x] Confirm materialization still zero-initializes movement words.
- [x] Confirm compact tick view still passes null/zero movement words from solver compact state.

Acceptance:

- [x] `rg "CompactSolverState" native/src/solver native/src/compiler ProgressReport.md Refactor.md RefactorImplementation.md` only finds historical notes or checklist items.
- [x] `make build`
- [x] `make solver_smoke_tests`
- [x] `make solver_compact_parity`
- [ ] Commit: `Rename CompactSolverState to CompactState`

## Phase 2: Introduce Turn Vocabulary Without Removing Old APIs

Goal: add the new turn concepts while keeping old `step`/`tick` names as forwarding shims.

- [x] Add `enum class AgainPolicy { Yield, Drain };`.
- [x] Add or rename toward `TurnOptions`.
- [x] Keep existing option fields: playable undo behavior and audio emission.
- [x] Add `AgainPolicy againPolicy = AgainPolicy::Yield` to the interpreted/full-state turn options.
- [x] Introduce a single interpreted turn implementation that accepts `ps_input`, including `PS_INPUT_TICK`.
- [x] Make `PS_INPUT_TICK` flow through the same interpreted turn API as player inputs.
- [x] Preserve title-screen and message-screen behavior for `PS_INPUT_ACTION`.
- [x] Preserve `PS_INPUT_TICK` behavior as direction mask `0`.
- [x] Preserve playable undo behavior: player inputs can push undo; `PS_INPUT_TICK` should not accidentally gain player-input undo behavior.
- [x] Preserve audio behavior.
- [x] Preserve restart/checkpoint behavior.
- [x] Preserve win/transition behavior.
- [x] Preserve the existing `again` scheduling behavior for `AgainPolicy::Yield`.
- [x] Implement `AgainPolicy::Drain` by applying the input once, then applying `PS_INPUT_TICK` until `pendingAgain` is false or the existing maximum again iteration limit is reached.
- [x] Define how multiple drained turn results merge: preserve any `changed`, `won`, `restarted`, or `transitioned` signal from the chain.
- [x] Keep `step`, `tick`, `interpreterStep`, and `interpreterTick` as wrappers during this phase.
- [ ] Mark wrappers as transitional in comments if helpful.

Acceptance:

- [x] `make build`
- [x] `make simulation_tests_cpp`
- [x] `make solver_smoke_tests`
- [x] `make solver_parity_smoke`
- [x] `make solver_compact_parity`
- [ ] Commit: `Introduce interpreted turn and again policy`

## Phase 3: Move Solver And Generator To Drain Policy

Goal: make solver/generator call the unified turn path with `AgainPolicy::Drain` instead of calling one turn and then manually settling `again`.

- [ ] Replace solver `step(...)` plus `settlePendingAgain(...)` call sites with the unified interpreted turn using `AgainPolicy::Drain`.
- [ ] Replace generator solver-loop `step(...)` plus `settlePendingAgain(...)` call sites with the unified interpreted turn using `AgainPolicy::Drain`.
- [ ] Preserve solver semantics for terminal edges.
- [ ] Preserve solver treatment of `restart` as a failed/terminal edge.
- [ ] Preserve compact tick oracle behavior: compact result compared to interpreted drained result where solver expects drained states.
- [ ] Check that normal player/runtime call sites still use `AgainPolicy::Yield`.
- [ ] Keep public C API behavior unchanged in this phase.

Acceptance:

- [ ] `make build`
- [ ] `make solver_smoke_tests`
- [ ] `make solver_parity_smoke`
- [ ] `make solver_compact_parity`
- [ ] `make generator_smoke_tests`
- [ ] Commit: `Use drained turns in solver and generator`

## Phase 4: Rename Compact Tick Architecture To Compact Turn

Goal: stop using `tick` as an architecture term for compact execution.

- [ ] Rename internal compact tick counters to compact turn counters.
- [ ] Rename JSON fields:
  - [ ] `compiled_compact_tick_attached` -> `specialized_compact_turn_attached`
  - [ ] `compact_tick_attempts` -> `compact_turn_attempts`
  - [ ] `compact_tick_hits` -> `compact_turn_hits`
  - [ ] `compact_tick_fallbacks` -> `compact_turn_fallbacks`
  - [ ] `compact_tick_unsupported` -> `compact_turn_unsupported`
  - [ ] `compact_tick_oracle_checks` -> `compact_turn_oracle_checks`
  - [ ] `compact_tick_oracle_failures` -> `compact_turn_oracle_failures`
- [ ] Update benchmark comparison scripts to read the new fields.
- [ ] Add temporary fallback reads for old JSON fields only if needed for comparing older benchmark files.
- [ ] Rename Make targets:
  - [ ] `compact_tick_oracle_smoke` -> `compact_turn_oracle_smoke`
  - [ ] `compact_tick_simulation_tests` -> `compact_turn_simulation_tests`
  - [ ] `compact_tick_coverage` -> `compact_turn_coverage`
- [ ] Add temporary Make aliases from old target names to new target names if useful for muscle memory.
- [ ] Rename CLI flags:
  - [ ] `--compact-tick-oracle` -> `--compact-turn-oracle`
  - [ ] `--require-compact-tick-oracle-checks` -> `--require-compact-turn-oracle-checks`
- [ ] Add temporary CLI flag aliases if tests/scripts still need to migrate in separate commits.
- [ ] Update human-readable output text.
- [ ] Update docs and checklist references.

Acceptance:

- [ ] `make build`
- [ ] `make compact_turn_oracle_smoke`
- [ ] `make compact_turn_simulation_tests`
- [ ] `make compact_turn_coverage`
- [ ] `make solver_compact_parity`
- [ ] `rg "compact_tick" native/src src Makefile Refactor.md RefactorImplementation.md ProgressReport.md` only finds intentional compatibility aliases or historical notes.
- [ ] Commit: `Rename compact tick tooling to compact turn`

## Phase 5: Rename Specialized Compact Backend Types

Goal: make type names match architecture names.

- [ ] Rename `CompiledCompactTickStateView` -> `CompactStateView`.
- [ ] Rename `CompiledCompactTickApplyOutcome` -> `SpecializedCompactTurnOutcome` or `CompactTurnOutcome`.
- [ ] Rename `CompiledCompactTickStepFn` -> `SpecializedCompactTurnFn`.
- [ ] Rename `CompiledCompactTickBackend` -> `SpecializedCompactTurnBackend`.
- [ ] Rename `compiledCompactTickInterpreterBridge` -> `compactStateInterpretedTurnBridge`.
- [ ] Rename generated compact backend symbols from `compact_tick_*` to `specialized_compact_turn_*`.
- [ ] Keep bridge semantics unchanged: compact boundary, materialize full state, run interpreted turn, copy compact state back.
- [ ] Ensure bridge copies complete RNG state both directions.
- [ ] Ensure bridge does not require compact-state movement words.
- [ ] Update backend lookup function names if they are not public ABI.
- [ ] If lookup symbols are public or linker-sensitive, add temporary forwarding symbols.

Acceptance:

- [ ] `make build`
- [ ] `make compact_turn_oracle_smoke`
- [ ] `make compact_turn_simulation_tests`
- [ ] `make compact_turn_coverage`
- [ ] `make solver_smoke_tests`
- [ ] Commit: `Rename specialized compact turn backend types`

## Phase 6: Rename Rulegroup Specialization Types

Goal: replace `CompiledRules` terminology where it means generated per-game rulegroup kernels.

- [ ] Rename `CompiledRulesBackend` -> `SpecializedRulegroupsBackend`.
- [ ] Rename `CompiledRuleGroupFn` -> `SpecializedRulegroupFn`.
- [ ] Rename `CompiledRuleApplyOutcome` -> `SpecializedRulegroupOutcome`.
- [ ] Rename `CompiledTickRuleGroupsFn` -> `SpecializedRulegroupsForInterpretedTurnFn`.
- [ ] Rename `CompiledTickRuleGroupsOutcome` -> `SpecializedRulegroupsForInterpretedTurnOutcome`.
- [ ] Rename runtime counters from compiled rule hits to specialized rulegroup hits.
- [ ] Update generated C++ emission to use the new names.
- [ ] Keep behavior unchanged: these kernels still operate on the full runtime state.
- [ ] Decide whether the command-line subcommand `compile-rules` stays temporarily or gains a new alias such as `specialize-rulegroups`.
- [ ] If `compile-rules` remains, document that it is a compatibility command name.

Acceptance:

- [ ] `make build`
- [ ] `make simulation_tests_cpp`
- [ ] `make compiled_rules_simulation_suite_coverage` or its renamed equivalent if already changed.
- [ ] `make solver_smoke_tests SPECIALIZE=true`
- [ ] `make generator_smoke_tests SPECIALIZE=true`
- [ ] Commit: `Rename compiled rules to specialized rulegroups`

## Phase 7: Rename Full Runtime State Carefully

Goal: decide and execute the large `Session` rename only after smaller axes are stable.

Decision checkpoint:

- [ ] Re-read `Refactor.md` and the `FullState` caveat.
- [ ] Decide whether this phase uses `FullState` or defers to a larger `RuntimeState` split.
- [ ] If choosing `FullState`, document why it is still the immediate rename target.
- [ ] If choosing `RuntimeState`, update `Refactor.md` first and stop this checklist to avoid implementing the wrong name.

If proceeding with `FullState`:

- [ ] Rename C++ `Session` -> `FullState`.
- [ ] Rename `PreparedSession` -> `PreparedFullState`.
- [ ] Rename helper functions:
  - [ ] `hashSession64` -> `hashFullState64`
  - [ ] `hashSession128` -> `hashFullState128`
  - [ ] `hashSession*NoAlloc` -> `hashFullState*NoAlloc`
  - [ ] `sessionStateKey` -> `fullStateKey`
  - [ ] `compactStateFromSession` -> `compactStateFromFullState`
  - [ ] `materializeCompactStateIntoSession` -> `materializeCompactStateIntoFullState`
- [ ] Update generated C++ emission from `Session&` to `FullState&`.
- [ ] Update comments and docs that use `Session` as architecture vocabulary.
- [ ] Leave `ps_session_*` C API names untouched until the public API phase.

Acceptance:

- [ ] `make build`
- [ ] `make simulation_tests_cpp`
- [ ] `make solver_smoke_tests`
- [ ] `make solver_parity_smoke`
- [ ] `make solver_compact_parity`
- [ ] `make compact_turn_simulation_tests`
- [ ] Commit: `Rename internal Session to FullState`

## Phase 8: Rename Public C API

Goal: align the external C API with the architecture terms once internals are stable.

- [ ] Rename opaque C type `ps_session` -> `ps_full_state`.
- [ ] Rename constructor/destructor:
  - [ ] `ps_session_create` -> `ps_full_state_create`
  - [ ] `ps_session_create_with_loaded_level_seed` -> `ps_full_state_create_with_loaded_level_seed`
  - [ ] `ps_session_clone` -> `ps_full_state_clone`
  - [ ] `ps_session_destroy` -> `ps_full_state_destroy`
- [ ] Rename turn functions:
  - [ ] `ps_session_step` -> `ps_full_state_turn`
  - [ ] `ps_session_tick` -> remove or alias to `ps_full_state_turn(state, PS_INPUT_TICK)`
- [ ] Rename status/accessor functions consistently to `ps_full_state_*`.
- [ ] Update CLI and SDL player call sites.
- [ ] Update tests and helper scripts.
- [ ] Add compatibility wrappers only if downstream consumers require them.
- [ ] If wrappers remain, mark them deprecated in comments and docs.

Acceptance:

- [ ] `make build`
- [ ] `make simulation_tests_cpp`
- [ ] `make solver_smoke_tests`
- [ ] Run/player compile target if separate from `make build`.
- [ ] `rg "ps_session" native/src src Makefile` only finds intentional compatibility wrappers or historical notes.
- [ ] Commit: `Rename public session API to full state API`

## Phase 9: Final Hygiene Pass

Goal: remove leftover ambiguity after the mechanical work is done.

- [ ] Remove temporary C++ aliases.
- [ ] Remove temporary Make aliases unless intentionally kept.
- [ ] Remove temporary CLI flag aliases unless intentionally kept.
- [ ] Update `ProgressReport.md`.
- [ ] Update `native/src/compiler/PLAN.md`.
- [ ] Update `native/src/compiler/IMPLEMENTATION_CHECKLIST.md`.
- [ ] Update benchmark scripts and output labels.
- [ ] Update help text.
- [ ] Update commit-era docs that now mislead more than they help.
- [ ] Keep historical notes only when they are clearly marked historical.

Search hygiene:

- [ ] `rg "\bSession\b" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [ ] `rg "CompactSolverState" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [ ] `rg "GenericTurn|GenericRulegroups|Generic turn|generic turn" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [ ] `rg "compiled tick|compact_tick|CompiledCompactTick|CompiledTick" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [ ] `rg "\btick\b|Tick" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [ ] Review each remaining hit and classify it as:
  - [ ] PuzzleScript input/event terminology
  - [ ] compatibility wrapper
  - [ ] historical note
  - [ ] bug to rename

Acceptance:

- [ ] `make build`
- [ ] `make simulation_tests_cpp`
- [ ] `make solver_smoke_tests`
- [ ] `make solver_parity_smoke`
- [ ] `make solver_compact_parity`
- [ ] `make generator_smoke_tests`
- [ ] `make compact_turn_oracle_smoke`
- [ ] `make compact_turn_simulation_tests`
- [ ] `make compact_turn_coverage`
- [ ] Commit: `Finish turn terminology refactor cleanup`

## Performance Sanity

This refactor should mostly preserve performance, except where `AgainPolicy::Drain` removes duplicated settle-again plumbing or specialized compact turns become monomorphized.

- [ ] Run `make solver_focus_compare` before starting behavior-sensitive phases.
- [ ] Run `make solver_focus_compare` after Phase 3.
- [ ] Run `make solver_focus_compare` after compact-turn backend renames.
- [ ] Confirm target identities remain the same.
- [ ] Confirm `median_generated` remains the same.
- [ ] Investigate any large `median_elapsed_ms` regression.
- [ ] Keep benchmark JSON field compatibility in comparison scripts until old benchmark files are no longer useful.

## Commit Discipline

- [ ] Use commit titles that describe the completed slice.
- [ ] Include metrics in commit titles when a slice changes coverage or performance.
- [ ] Prefer examples:
  - [ ] `Rename CompactSolverState to CompactState`
  - [ ] `Introduce interpreted turn and again policy`
  - [ ] `Use drained turns in solver and generator`
  - [ ] `Rename compact tick tooling to compact turn`
  - [ ] `Rename specialized compact turn backend types`
  - [ ] `Rename compiled rules to specialized rulegroups`
- [ ] Never include unrelated user-owned working tree changes in these commits.
