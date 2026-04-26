# FullState / CompactState Refactor Plan

## Goal

Make the native runtime terminology precise enough that the architecture says what is actually happening.

The core split is not "compiled" versus "interpreted", because all native C++ paths are compiled. The useful split is whether a path is game-agnostic or generated per game, and whether it operates on the full runtime state or the compact solver/generator state.

The target vocabulary is:

- `FullState` vs `CompactState`
- `InterpretedTurn` vs `SpecializedTurn`
- `InterpretedRulegroups` vs `SpecializedRulegroups`
- `AgainPolicy::Yield` vs `AgainPolicy::Drain`

The runtime should also move toward one `turn(state, input, options)` operation. `PS_INPUT_TICK` remains just another `ps_input`, not a reason to keep separate step/tick architecture names.

## Core Vocabulary

`FullState` is the full runtime state currently represented by the C++ `Session` type. It owns the board, movement words, caches, command/runtime scratch, dirty flags, undo/restart/checkpoint state, audio/UI state, RNG, and other runtime machinery.

There is a real naming caveat: today's `Session` is not just "more state" than `CompactState`; it is also lifecycle scaffolding. The immediate rename target is `FullState` for consistency with `CompactState`, but a larger later split may be cleaner: immutable `Game` data plus mutable `RuntimeState`, with `CompactState` as the solver/generator sibling of `RuntimeState`.

`CompactState` is the solver/generator state representation: object occupancy bitsets plus RNG. It represents settled graph-search states and should not contain transient movement words after a turn has fully resolved.

`InterpretedTurn` is the game-agnostic C++ turn driver. It owns the normal turn sequencing: input handling, player movement seeding, early rulegroups, movement resolution, late rulegroups, commands, win checks, restart/checkpoint behavior, and `again` scheduling.

`SpecializedTurn` is generated per-game turn sequencing. It replaces the interpreted turn driver for a game-specific path.

`InterpretedRulegroups` are interpreted rulegroup matching and application through runtime data structures.

`SpecializedRulegroups` are generated per-game rulegroup kernels.

`AgainPolicy::Yield` is normal play behavior: apply one input and leave `again` pending so the player/runtime can advance it as subsequent `PS_INPUT_TICK` turns.

`AgainPolicy::Drain` is solver/generator behavior: apply the requested input, then repeatedly apply `PS_INPUT_TICK` until `again` is exhausted, so graph nodes are settled states.

## Current Runtime Paths

`FullState + InterpretedTurn + InterpretedRulegroups` is the native interpreter and correctness oracle.

`FullState + InterpretedTurn + SpecializedRulegroups` is the current specialized-rulegroup runtime. The interpreted turn driver still owns sequencing, but rulegroup evaluation can dispatch to generated kernels.

`CompactState boundary -> FullState + InterpretedTurn + InterpretedRulegroups` is the current compact interpreted bridge. It accepts compact state at the boundary, materializes full state, runs the interpreter path, and copies compact state back out.

`CompactState + SpecializedTurn + SpecializedRulegroups` is the desired solver/generator runtime. It should execute a whole settled turn directly on compact state, without materializing full state.

`FullState + SpecializedTurn + SpecializedRulegroups` is possible, but it is not the main target. It may be useful as a migration step or gameplay optimization, but the solver/generator goal is specialized compact execution.

The useful turn/rulegroup matrix is:

| Turn driver | InterpretedRulegroups | SpecializedRulegroups |
| --- | --- | --- |
| `InterpretedTurn` | pure interpreter / oracle | current hybrid optimization |
| `SpecializedTurn` | not a coherent target | full per-game codegen |

So `SpecializedTurn` normally implies `SpecializedRulegroups`. The rulegroup axis is most important for the interpreted turn driver, where the runtime can either walk rule data structures or call generated rulegroup kernels.

## CompactState Identity

`CompactState` should be precise enough to reconstruct the future of the level.

It includes:

- object occupancy bitsets
- complete RNG state, not just an RNG seed or identity

It excludes:

- transient movement words after a turn has settled
- visual viewport state such as flickscreen/zoomscreen
- undo/audio/UI/debug scratch

Solver and generator graph nodes should be settled states. In that mode, `again` is drained before insertion, so there should be no pending-again bit in normal compact identity. A compact turn that cannot drain `again` must report failure/fallback rather than returning a half-settled solver node.

Current level index is not part of the first solver compact-state target because solver runs a particular level as its state space. Multi-level solving can add an explicit level identity field later if it becomes a real use case.

Checkpoint/restart state is not part of the first solver compact-state target. For solver/generator purposes, `restart` is treated as a terminal failed edge rather than as a mechanic that rewinds to a checkpoint.

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
turn(state, input, AgainPolicy::Yield)
turn(state, input, AgainPolicy::Drain)
```

Normal play should use `Yield`. Solver and generator should use `Drain`.

For ergonomics, interpreted paths may carry the policy in `TurnOptions`. For specialized compact turns, the policy should be monomorphized where performance matters, either with template parameters or separate entrypoints. Solver/generator hot paths should not branch on again policy inside the inner loop.

The return value also needs to be part of the architecture. A turn returns a structured result, not just a boolean:

```text
TurnResult {
    changed
    won
    restarted
    transitioned
    commands / command effects
    optional heuristic hint for solver/generator
}
```

Normal play may need audio/UI command effects. Solver/generator can use a reduced result shape, but it must still preserve semantic flags such as `won`, `restarted`, and `changed`.

## Rename Plan

Rename runtime state terms:

- `Session` -> `FullState`
- `PreparedSession` -> `PreparedFullState`
- `CompactSolverState` -> `CompactState`
- `ps_session_*` -> `ps_full_state_*`

Rename turn concepts:

- `step` / `tick` architecture names -> `turn`
- `interpreterStep` / `interpreterTick` -> `interpretedTurn`
- `interpreterStepWithCompiledRuleGroups` / `interpreterTickWithCompiledRuleGroups` legacy wrappers -> `interpretedStepWithSpecializedRulegroups` / `interpretedTickWithSpecializedRulegroups`
- `RuntimeStepOptions` -> `TurnOptions`
- add `AgainPolicy::{Yield, Drain}`

Rename specialization concepts:

- `CompiledRulesBackend` -> `SpecializedRulegroupsBackend`
- `CompiledRuleGroupFn` -> `SpecializedRulegroupFn`
- legacy `CompiledTickBackend` references -> `SpecializedFullTurnBackend`
- legacy `CompiledCompactTickBackend` references -> `SpecializedCompactTurnBackend`
- legacy `CompiledCompactTickStateView` references -> `CompactStateView`
- `compiledCompactTickInterpreterBridge` -> `compactStateInterpretedTurnBridge`

Rename tool-facing compact terminology:

- legacy `compact_tick_*` Make targets and JSON fields -> `compact_turn_*`
- `compiled_tick_*` architecture wording -> `specialized_turn_*`
- keep `PS_INPUT_TICK` unchanged, because it is a PuzzleScript input value rather than an architecture term

The migration should be layered:

1. Introduce new names with temporary aliases and keep tests green.
2. Flip call sites and generated-code emission.
3. Rename public tools/API names.
4. Remove aliases unless there is an explicit compatibility requirement.

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
rg "\bSession\b|CompactSolverState|compiled tick|compact_tick|CompiledCompactTick|CompiledTick|Tick" native/src src Makefile ProgressReport.md Refactor.md
```

Remaining hits should be intentional legacy references, generated compatibility shims, or PuzzleScript's `PS_INPUT_TICK`.

## Assumptions

This document is a docs-only change. It should not change runtime behavior.

The terminology cleanup is allowed to be breaking once implementation begins.

`PS_INPUT_TICK` remains part of `ps_input`.

"Tick" should refer to PuzzleScript input/event behavior, not the architecture.

The main optimization goal remains `CompactState + SpecializedTurn + SpecializedRulegroups + AgainPolicy::Drain`.

`FullState + SpecializedTurn + SpecializedRulegroups` is optional and should not be treated as a required milestone.
