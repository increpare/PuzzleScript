# FullState / CompactState Refactor Plan

## Goal

Make the native runtime terminology precise enough that the architecture says what is actually happening.

The core split is not "compiled" versus "interpreted", because all native C++ paths are compiled. The useful split is whether a path is game-agnostic or generated per game, and whether it operates on the full runtime state or the compact solver/generator state.

The target vocabulary is:

- `FullState` vs `CompactState`
- `GenericTurn` vs `SpecializedTurn`
- `GenericRulegroups` vs `SpecializedRulegroups`
- `AgainPolicy::LeavePending` vs `AgainPolicy::ResolveImmediately`

The runtime should also move toward one `turn(state, input, options)` operation. `PS_INPUT_TICK` remains just another `ps_input`, not a reason to keep separate step/tick architecture names.

## Core Vocabulary

`FullState` is the full runtime state currently represented by the C++ `Session` type. It owns the board, movement words, caches, command/runtime scratch, dirty flags, undo/restart/checkpoint state, audio/UI state, RNG, and other runtime machinery.

`CompactState` is the solver/generator state representation: object occupancy bitsets plus RNG. It represents settled graph-search states and should not contain transient movement words after a turn has fully resolved.

`GenericTurn` is the game-agnostic C++ turn driver. It owns the normal turn sequencing: input handling, player movement seeding, early rulegroups, movement resolution, late rulegroups, commands, win checks, restart/checkpoint behavior, and `again` scheduling.

`SpecializedTurn` is generated per-game turn sequencing. It replaces the generic turn driver for a game-specific path.

`GenericRulegroups` are interpreted rulegroup matching and application through runtime data structures.

`SpecializedRulegroups` are generated per-game rulegroup kernels.

`AgainPolicy::LeavePending` is normal play behavior: apply one input and leave `again` pending so the player/runtime can advance it as subsequent `PS_INPUT_TICK` turns.

`AgainPolicy::ResolveImmediately` is solver/generator behavior: apply the requested input, then repeatedly apply `PS_INPUT_TICK` until `again` is exhausted, so graph nodes are settled states.

## Current Runtime Paths

`FullState + GenericTurn + GenericRulegroups` is the native interpreter and correctness oracle.

`FullState + GenericTurn + SpecializedRulegroups` is the current specialized-rulegroup runtime. The generic turn driver still owns sequencing, but rulegroup evaluation can dispatch to generated kernels.

`CompactState boundary -> FullState + GenericTurn + GenericRulegroups` is the current compact generic bridge. It accepts compact state at the boundary, materializes full state, runs the generic interpreter path, and copies compact state back out.

`CompactState + SpecializedTurn + SpecializedRulegroups` is the desired solver/generator runtime. It should execute a whole settled turn directly on compact state, without materializing full state.

`FullState + SpecializedTurn + SpecializedRulegroups` is possible, but it is not the main target. It may be useful as a migration step or gameplay optimization, but the solver/generator goal is specialized compact execution.

## Turn Unification

The architecture should stop treating step and tick as separate runtime concepts.

`PS_INPUT_TICK` already exists in `ps_input`:

```c
typedef enum ps_input {
    PS_INPUT_UP = 0,
    PS_INPUT_LEFT = 1,
    PS_INPUT_DOWN = 2,
    PS_INPUT_RIGHT = 3,
    PS_INPUT_ACTION = 4,
    PS_INPUT_TICK = 5
} ps_input;
```

That means realtime/tick behavior should be represented as:

```text
turn(state, PS_INPUT_TICK, options)
```

rather than as a separate tick function.

The meaningful distinction is `again` policy:

```text
turn(state, input, AgainPolicy::LeavePending)
turn(state, input, AgainPolicy::ResolveImmediately)
```

Normal play should use `LeavePending`. Solver and generator should use `ResolveImmediately`.

## Rename Plan

Rename runtime state terms:

- `Session` -> `FullState`
- `PreparedSession` -> `PreparedFullState`
- `CompactSolverState` -> `CompactState`
- `ps_session_*` -> `ps_full_state_*`

Rename turn concepts:

- `step` / `tick` architecture names -> `turn`
- `interpreterStep` / `interpreterTick` -> `genericTurn`
- `interpreterStepWithCompiledRuleLoops` / `interpreterTickWithCompiledRuleLoops` -> a single full-state generic turn function with optional specialized rulegroup callbacks
- `RuntimeStepOptions` -> `TurnOptions`
- add `AgainPolicy::{LeavePending, ResolveImmediately}`

Rename specialization concepts:

- `CompiledRulesBackend` -> `SpecializedRulegroupsBackend`
- `CompiledRuleGroupFn` -> `SpecializedRulegroupFn`
- `CompiledTickBackend` -> avoid as a central architecture term; use explicit full-state specialized-turn naming only if that path becomes deliberate
- `CompiledCompactTickBackend` -> `SpecializedCompactTurnBackend`
- `CompiledCompactTickStateView` -> `CompactStateView`
- `compiledCompactTickInterpreterBridge` -> `compactStateGenericTurnBridge`

Rename tool-facing compact terminology:

- `compact_tick_*` Make targets and JSON fields -> `compact_turn_*`
- `compiled_tick_*` architecture wording -> `specialized_turn_*`
- keep `PS_INPUT_TICK` unchanged, because it is a PuzzleScript input value rather than an architecture term

The migration should be layered. Temporary aliases are acceptable while keeping the tree buildable, but the final cleanup should remove aliases unless there is an explicit compatibility requirement.

## Test Plan

Because the first step is documentation-only:

- Run `git diff --check`.
- Confirm `Refactor.md` renders as Markdown.
- Do not run build/tests unless accidental code changes appear.

For the later implementation rename, run:

- `make build`
- `make simulation_tests_cpp`
- `make solver_smoke_tests`
- `make solver_parity_smoke`
- `make solver_compact_parity`
- `make generator_smoke_tests`
- renamed compact-turn coverage/oracle targets after they exist

Search hygiene for the final rename:

```sh
rg "\bSession\b|CompactSolverState|compiled tick|compact_tick|CompiledCompactTick|CompiledTick" native/src src Makefile ProgressReport.md Refactor.md
```

Remaining hits should be intentional legacy references, generated compatibility shims, or PuzzleScript's `PS_INPUT_TICK`.

## Assumptions

This document is a docs-only change. It should not change runtime behavior.

The terminology cleanup is allowed to be breaking once implementation begins.

`PS_INPUT_TICK` remains part of `ps_input`.

"Tick" should refer to PuzzleScript input/event behavior, not the architecture.

The main optimization goal remains `CompactState + SpecializedTurn + SpecializedRulegroups`.

`FullState + SpecializedTurn + SpecializedRulegroups` is optional and should not be treated as a required milestone.
