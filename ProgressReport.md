# CompactSolverState Progress Report

## Purpose

`CompactSolverState` is the solver/generator-facing representation of a level state. It is meant to make graph search cheaper than storing a whole runtime `Session` at every visited node.

The intended semantic state is the end-of-turn board occupancy: which objects occupy which cells, plus only the extra data that can change future deterministic outcomes. Solver nodes are created after a full player input has resolved, including all rule loops and `again` processing. That means transient movement state should not be part of the graph node identity.

The level generator benefits through the solver pipeline: when generated levels are validated, scored, or solved, they can use the same compact node storage and compact tick path.

## Current Status

The compact tick infrastructure is wired through the native build and the testdata simulation corpus.

- `make compact_tick_simulation_tests` currently runs all `src/tests/resources/testdata.js` simulation cases through the compact tick oracle path.
- The last measured run passed `469/469` direct simulation cases with `16554` compact oracle checks and `0` compact oracle failures.
- `make compact_tick_coverage` reports `452/452` unique testdata sources with callable compact backends.
- Of those, `79/452` currently use native compact kernels and `373/452` use the interpreter bridge backend.

So the current achievement is full callable coverage, not full native compact codegen coverage. The interpreter bridge is deliberately part of the coverage story: it lets every game exercise the compact ABI while native kernels are added incrementally.

## Key Locations

Solver-side compact state and graph search live mostly in:

- `native/src/solver/main.cpp`
  - `CompactSolverState`
  - `compactStateFromSession`
  - `compactStateKey`
  - `compactStateWithTiming`
  - `materializeCompactStateIntoSession`
  - `tryCompiledCompactTick`
  - `runSearch`

Runtime compact tick ABI and fallback support live in:

- `native/src/runtime/compiled_rules.hpp`
  - `CompiledCompactTickStateView`
  - `CompiledCompactTickApplyOutcome`
  - `CompiledCompactTickBackend`
  - `compiledCompactTickInterpreterBridge`
- `native/src/runtime/compiled_rules.cpp`
  - compact bridge materialization and copy-back helpers
- `native/src/runtime/c_api.cpp`
  - compact tick oracle checks for the simulation harness

Generated compact backend code is emitted from:

- `native/src/cli/main.cpp`
  - compact tick support classification
  - generated `compact_tick_step_*` functions
  - native-vs-bridge coverage metadata

Useful Make targets are:

- `make compact_tick_simulation_tests`
- `make compact_tick_coverage`
- `make solver_smoke_tests`
- `make solver_parity_smoke`
- `make generator_smoke_tests`
- `make solver_focus_compare`

## Solver Flow

At a high level, the solver is moving toward this shape:

1. Start with an interpreter-created initial `Session`.
2. Convert it to `CompactSolverState` with `compactStateFromSession`.
3. Hash the compact state with `compactStateKey`.
4. Store compact states in the search graph instead of full sessions where possible.
5. For each candidate input, call `tryCompiledCompactTick`.
6. If compact tick handles the game/input, use the returned compact child state directly.
7. If compact tick does not handle it, materialize a `Session` with `materializeCompactStateIntoSession` and run the interpreter path.
8. Compare, insert, and prioritize states in the solver frontier.

The long-term target is:

```text
tick(object_bits, input) -> {changed, won, restarted, next_object_bits, heuristic}
```

The current implementation is not there yet. It has the ABI and oracle scaffolding, plus a small set of native compact kernels, while most games still pass through the bridge.

## Interpreter Fallback

The interpreter remains the correctness oracle and fallback.

When a game cannot yet be represented by native compact code, generated metadata marks it as bridge-supported. The compact backend can then call `compiledCompactTickInterpreterBridge`, which reconstructs a temporary `Session`, runs the normal interpreter step, and copies the resulting compact state back out through `CompiledCompactTickStateView`.

This fallback is useful for two reasons:

- It keeps all compact tick entrypoints callable while native support grows.
- It gives the oracle path a direct way to compare compact behavior against the reference interpreter.

For solver performance, bridge-backed games are not the destination. They mainly protect correctness and keep the ABI honest while native compact kernels are filled in.

## Important Design Issue: movementWords

`CompactSolverState` currently contains `movementWords`. That is probably wrong for solver graph identity.

Movement words are useful inside a turn while rules and movement resolution are still running. Solver graph nodes, however, are end-of-turn states. Once an input has fully resolved, there should be no outstanding movement to remember. Including movement data in equality, hashing, storage, or materialization risks making the compact state larger and more semantically confusing than it needs to be.

The likely fix is:

- Remove `movementWords` from `CompactSolverState`.
- Remove movement words from compact solver equality, byte-size accounting, and hashing.
- Materialize solver fallback sessions with zeroed `liveMovements`.
- Keep movement buffers, if needed, inside `CompiledCompactTickStateView` or bridge scratch state as a transitional runtime detail rather than as part of node identity.

This should be handled before treating compact solver storage as architecturally settled.

## Randomness

Random state is currently present in `CompactSolverState`, because random rules can make two visually identical boards have different futures.

For the immediate solver/generator focus group, random and `randomDir` games are being excluded. Under that policy, random state does not need to participate in compact solver node identity. There are two reasonable next steps:

- Gate compact solver node storage to deterministic games and remove RNG from `CompactSolverState`.
- Or keep RNG fields only for broader simulation/oracle coverage, but keep deterministic solver/generator paths explicitly free of random-state hashing.

The first option is simpler for solver performance. The second option is safer for eventually supporting random games in compact simulation tests.

## Missing Functionality

The main missing pieces are native behavior coverage and solver-oriented cleanup.

- Remove `movementWords` from compact solver node identity.
- Decide whether solver compact state excludes RNG entirely when random games are not admitted.
- Continue replacing interpreter bridge cases with native compact kernels.
- Add native compact support for remaining PuzzleScript features: complex movement creation, ellipsis patterns, commands, `again`, `cancel`, `restart`, `checkpoint`, `win`, rigid groups, random/randomDir if they return to scope, late rules, and more complex multi-cell/multi-row patterns.
- Move more solver heuristics toward compact or generated code so the solver avoids repeated materialization.
- Keep generator integration focused on the same compact solver path instead of creating a separate representation.
- Measure whether compact state storage and exact equality reduce graph overhead on nontrivial solver focus games.

## Near-Term Checklist

1. Remove `movementWords` from solver node identity.
2. Re-run `make compact_tick_simulation_tests`.
3. Re-run compact solver parity/smoke targets.
4. Confirm `make compact_tick_coverage` still reports `452/452` callable compact backends.
5. Measure `make solver_focus_compare` before and after the compact state cleanup.
6. Resume native compact kernel coverage work, using `compact_tick_coverage` as the progress counter.
