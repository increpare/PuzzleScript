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
- [x] Keep compatibility aliases only while they are actively helping a staged migration.
- [x] Remove aliases before declaring a rename track complete, unless a compatibility decision explicitly keeps them.
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
- [x] Run `make compact_turn_coverage` and record callable/native/bridge counts.
- [x] Run `make compact_turn_simulation_tests` if iteration time is acceptable.
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
- [x] Confirm compact turn view still passes null/zero movement words from solver compact state.

Acceptance:

- [x] `rg "CompactSolverState" native/src/solver native/src/compiler ProgressReport.md Refactor.md RefactorImplementation.md` only finds historical notes or checklist items.
- [x] `make build`
- [x] `make solver_smoke_tests`
- [x] `make solver_compact_parity`
- [x] Commit: `Rename CompactSolverState to CompactState`

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
- [x] Commit: `Introduce interpreted turn and again policy`

## Phase 3: Move Solver And Generator To Drain Policy

Goal: make solver/generator call the unified turn path with `AgainPolicy::Drain` instead of calling one turn and then manually settling `again`.

- [x] Replace solver `step(...)` plus `settlePendingAgain(...)` call sites with the unified interpreted turn using `AgainPolicy::Drain`.
- [x] Replace generator solver-loop `step(...)` plus `settlePendingAgain(...)` call sites with the unified interpreted turn using `AgainPolicy::Drain`.
- [x] Preserve solver semantics for terminal edges.
- [x] Preserve solver treatment of `restart` as a failed/terminal edge.
- [x] Preserve compact turn oracle behavior: compact result compared to interpreted drained result where solver expects drained states.
- [x] Check that normal player/runtime call sites still use `AgainPolicy::Yield`.
- [x] Keep public C API behavior unchanged in this phase.

Acceptance:

- [x] `make build`
- [x] `make solver_smoke_tests`
- [x] `make solver_parity_smoke`
- [x] `make solver_compact_parity`
- [x] `make generator_smoke_tests`
- [x] Commit: `Use drained turns in solver and generator`

## Phase 4: Rename Compact Tick Architecture To Compact Turn

Goal: stop using `tick` as an architecture term for compact execution.

- [x] Rename internal compact tick counters to compact turn counters.
- [x] Rename JSON fields:
  - [x] `compiled_compact_tick_attached` -> `specialized_compact_turn_attached`
  - [x] `compact_tick_attempts` -> `compact_turn_attempts`
  - [x] `compact_tick_hits` -> `compact_turn_hits`
  - [x] `compact_tick_fallbacks` -> `compact_turn_fallbacks`
  - [x] `compact_tick_unsupported` -> `compact_turn_unsupported`
  - [x] `compact_tick_oracle_checks` -> `compact_turn_oracle_checks`
  - [x] `compact_tick_oracle_failures` -> `compact_turn_oracle_failures`
- [x] Update benchmark comparison scripts to read the new fields.
- [x] Add temporary fallback reads for old JSON fields only if needed for comparing older benchmark files.
- [x] Rename Make targets:
  - [x] `compact_tick_oracle_smoke` -> `compact_turn_oracle_smoke`
  - [x] `compact_tick_simulation_tests` -> `compact_turn_simulation_tests`
  - [x] `compact_tick_coverage` -> `compact_turn_coverage`
- [x] Add temporary Make aliases from old target names to new target names if useful for muscle memory.
- [x] Rename CLI flags:
  - [x] `--compact-tick-oracle` -> `--compact-turn-oracle`
  - [x] `--require-compact-tick-oracle-checks` -> `--require-compact-turn-oracle-checks`
- [x] Add temporary CLI flag aliases if tests/scripts still need to migrate in separate commits.
- [x] Update human-readable output text.
- [x] Update docs and checklist references.

Acceptance:

- [x] `make build`
- [x] `make compact_turn_oracle_smoke`
- [x] `make compact_turn_simulation_tests`
- [x] `make compact_turn_coverage`
- [x] `make solver_compact_parity`
- [x] `rg "compact_tick" native/src src Makefile Refactor.md RefactorImplementation.md ProgressReport.md` only finds intentional compatibility aliases or historical notes.
- [x] Commit: `Rename compact tick tooling to compact turn`

## Phase 5: Rename Specialized Compact Backend Types

Goal: make type names match architecture names.

- [x] Rename `CompiledCompactTickStateView` -> `CompactStateView`.
- [x] Rename `CompiledCompactTickApplyOutcome` -> `SpecializedCompactTurnOutcome`.
- [x] Rename `CompiledCompactTickStepFn` -> `SpecializedCompactTurnFn`.
- [x] Rename `CompiledCompactTickBackend` -> `SpecializedCompactTurnBackend`.
- [x] Rename `compiledCompactTickInterpreterBridge` -> `compactStateInterpretedTurnBridge`.
- [x] Rename generated compact backend symbols from `compact_tick_*` to `specialized_compact_turn_*`.
- [x] Keep bridge semantics unchanged: compact boundary, materialize full state, run interpreted turn, copy compact state back.
- [x] Ensure bridge copies complete RNG state both directions.
- [x] Ensure bridge does not require compact-state movement words.
- [x] Update backend lookup function names if they are not public ABI.
- [x] If lookup symbols are public or linker-sensitive, add temporary forwarding symbols.
- [x] Remove unused `CompiledCompactTick*` type aliases and the old inline bridge helper after source call sites migrate.

Acceptance:

- [x] `make build`
- [x] `make compact_turn_oracle_smoke`
- [x] `make compact_turn_simulation_tests`
- [x] `make compact_turn_coverage`
- [x] `make solver_smoke_tests`
- [x] `make solver_compact_parity`
- [x] Commit: `Rename specialized compact turn backend types`

## Phase 6: Rename Rulegroup Specialization Types

Goal: replace `CompiledRules` terminology where it means generated per-game rulegroup kernels.

- [x] Rename `CompiledRulesBackend` -> `SpecializedRulegroupsBackend`.
- [x] Rename `CompiledRuleGroupFn` -> `SpecializedRulegroupFn`.
- [x] Rename `CompiledRuleApplyOutcome` -> `SpecializedRulegroupOutcome`.
- [x] Rename `CompiledTickRuleGroupsFn` -> `SpecializedRulegroupsForInterpretedTurnFn`.
- [x] Rename `CompiledTickRuleGroupsOutcome` -> `SpecializedRulegroupsForInterpretedTurnOutcome`.
- [x] Rename runtime counters from compiled rule hits to specialized rulegroup hits.
- [x] Update generated C++ emission to use the new names.
- [x] Keep behavior unchanged: these kernels still operate on the full runtime state.
- [x] Decide whether the command-line subcommand `compile-rules` stays temporarily or gains a new alias such as `specialize-rulegroups`.
- [x] If `compile-rules` remains, document that it is a compatibility command name.
- [x] Remove unused `CompiledRule*` / `CompiledRulesBackend` compatibility type aliases after source call sites migrate.
- [x] Rename active full-state generated turn backend types from `CompiledTick*` to `SpecializedFullTurn*`.
- [x] Generated registries export `ps_specialized_full_turn_find_backend`, with `ps_compiled_tick_find_backend` kept as a compatibility symbol.
- [x] Rename interpreted/full-state helper entrypoints toward specialized rulegroups:
  - [x] `interpretedTurnWithCompiledRuleGroups` -> `interpretedTurnWithSpecializedRulegroups`, with compatibility wrapper.
  - [x] `interpreterStepWithCompiledRuleGroups` -> `interpretedStepWithSpecializedRulegroups`, with compatibility wrapper.
  - [x] `interpreterTickWithCompiledRuleGroups` -> `interpretedTickWithSpecializedRulegroups`, with compatibility wrapper.
- [x] Update generated full-turn C++ to call the specialized-rulegroup helper names.
- [x] Expose primary `specialized_full_turn_*` counters while keeping `compiled_tick_*` aliases.
- [x] Expose primary `specialized_full_turn` coverage JSON while keeping `compiled_tick` aliases.
- [x] Add a coverage-shape smoke target that checks current and compatibility coverage keys.

Acceptance:

- [x] `make build`
- [x] `make simulation_tests_cpp`
- [x] `make compiled_rules_simulation_suite_coverage` or its renamed equivalent if already changed.
- [x] `make solver_smoke_tests SPECIALIZE=true`
- [x] `make generator_smoke_tests SPECIALIZE=true`
- [x] Commit the rulegroup/full-turn rename track in smaller slices:
  - `Rename preparedFullState field 469 simulations 7 smoke`
  - `Remove stale compiled alias types 7 specialized smoke`
  - `Rename full-turn backend 7 specialized smoke 18 oracle`
  - `Expose specialized full-turn counters 7 smoke`
  - `Report specialized full-turn coverage 5 sources 7 smoke`
  - `Guard coverage JSON shape 5 sources`
  - `Rename interpreted rulegroup turn helpers 7 smoke`

## Phase 7: Rename Full Runtime State Carefully

Goal: decide and execute the large `Session` rename only after smaller axes are stable.

Decision checkpoint:

- [x] Re-read `Refactor.md` and the `FullState` caveat.
- [x] Decide whether this phase uses `FullState` or defers to a larger `RuntimeState` split.
- [x] If choosing `FullState`, document why it is still the immediate rename target.
- [ ] If choosing `RuntimeState`, update `Refactor.md` first and stop this checklist to avoid implementing the wrong name.

If proceeding with `FullState`:

- [x] Rename the primary C++ state struct `Session` -> `FullState`, with a temporary `Session` alias.
- [x] Finish migrating internal call sites from the `Session` alias to `FullState`.
  - [x] Search heuristic helpers use `FullState`.
  - [x] Solver compact-node and compact-materialization helpers use `FullState`.
  - [x] Generator solver helpers use `FullState`.
  - [x] Compact interpreted bridge helpers use `FullState`.
  - [x] Internal runtime declarations use `FullState`.
  - [x] `createFullState*` constructors exist; `createSession*` remains as a temporary compatibility wrapper.
  - [x] Runtime lifecycle, hash/serialization, and turn wrapper definitions use `FullState`.
  - [x] Runtime core implementation uses `FullState`; `Session` remains only as a temporary compatibility surface.
  - [x] Runtime C API internals store and inspect `FullState`; public `ps_session_*` names are unchanged.
  - [x] Public C API still uses `ps_session_*` by design until Phase 8.
- [x] Rename the primary prepared-state struct `PreparedSession` -> `PreparedFullState`, with a temporary `PreparedSession` alias.
- [x] Rename the internal `preparedSession` field/identifier to `preparedFullState`; serialized `prepared_session` keys remain unchanged.
- [x] Rename helper functions:
  - [x] `hashSession64` -> `hashFullState64`, with temporary wrapper.
  - [x] `hashSession128` -> `hashFullState128`, with temporary wrapper.
  - [x] `hashSession*NoAlloc` -> `hashFullState*NoAlloc`, with temporary wrapper.
  - [x] `sessionStateKey` -> `fullStateKey`, with temporary wrapper.
  - [x] `compactStateFromSession` -> `compactStateFromFullState`; temporary wrapper removed.
  - [x] `materializeCompactStateIntoSession` -> `materializeCompactStateIntoFullState`; temporary wrapper removed.
- [x] Update generated C++ emission from `Session&` to `FullState&`.
- [x] Update comments and docs that use `Session` as architecture vocabulary.
  - [x] Solver heuristic notes refer to normal `FullState` scoring.
  - [x] `ProgressReport.md` uses `CompactState`, `FullState`, and compact turn terminology.
- [x] Leave `ps_session_*` C API names untouched until the public API phase.

Decision note: proceed with `FullState` now because it names the contrast with
`CompactState` clearly for solver/generator work. The larger `RuntimeState`
split remains attractive, but it should be a semantic ownership refactor rather
than a prerequisite for getting compact specialized turns into shape.

Acceptance:

- [x] `make build`
- [x] `make simulation_tests_cpp`
- [x] `make solver_smoke_tests`
- [x] `make solver_parity_smoke`
- [x] `make solver_compact_parity`
- [x] `make compact_turn_simulation_tests`
- [x] Commit the internal state rename track in smaller slices, ending with
  `Remove internal FullState compatibility aliases`.

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

- [x] Remove temporary C++ aliases.
- [ ] Remove temporary Make aliases unless intentionally kept.
- [ ] Remove temporary CLI flag aliases unless intentionally kept.
- [ ] Update `ProgressReport.md`.
- [x] Update `native/src/compiler/PLAN.md`.
- [x] Update `native/src/compiler/IMPLEMENTATION_CHECKLIST.md`.
- [ ] Update benchmark scripts and output labels.
- [ ] Update help text.
- [ ] Update commit-era docs that now mislead more than they help.
- [ ] Keep historical notes only when they are clearly marked historical.

Search hygiene:

- [x] `rg "\bSession\b" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [x] `rg "CompactSolverState" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [x] `rg "GenericTurn|GenericRulegroups|Generic turn|generic turn" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [x] `rg "compiled tick|compact_tick|CompiledCompactTick|CompiledTick" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
  - Remaining Make target hits are compatibility aliases for old muscle memory.
  - Remaining JS hits are old benchmark/JSON fallback readers.
  - Remaining C API hits are public compatibility names deferred to Phase 8.
  - Remaining generated-symbol hits are linker compatibility shims.
  - Remaining `*WithCompiledRuleGroups` hits are compatibility wrappers for old
    generated C++.
- [x] `rg "\btick\b|Tick" native/src src Makefile ProgressReport.md Refactor.md RefactorImplementation.md`
- [x] Review each remaining hit and classify it as:
  - [x] PuzzleScript input/event terminology
    - `PS_INPUT_TICK`, string input `"tick"`, realtime gameplay, SDL timing,
      JS UI timing, and PuzzleScript source fixtures keep using `tick`.
  - [x] compatibility wrapper
    - Public `step`/`tick`, `ps_session_tick`, legacy `compact_tick` and
      `compiled_tick` targets/symbols stay until Phase 8 compatibility cleanup.
  - [x] historical note
    - Refactor-plan entries may mention old names when they document the rename
      itself or explicitly describe forwarding shims.
  - [x] bug to rename
    - Stale compiler-plan/checklist wording was updated to full-turn/turn
      wording.

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
  - [ ] `Rename compact turn tooling to compact turn`
  - [ ] `Rename specialized compact turn backend types`
  - [ ] `Rename compiled rules to specialized rulegroups`
- [ ] Never include unrelated user-owned working tree changes in these commits.
