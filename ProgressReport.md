# CompactState Progress Report

## Purpose

`CompactState` is the solver/generator-facing representation of a level state. It is meant to make graph search cheaper than storing a whole runtime `FullState` at every visited node.

The intended semantic state is the end-of-turn board occupancy: which objects occupy which cells, plus complete RNG state so future turns can be replayed deterministically. Solver nodes are created after a full player input has resolved, including rule loops and `again` draining. That means transient movement state is not part of solver graph node identity.

The level generator benefits through the solver pipeline: when generated levels are validated, scored, or solved, they can use the same compact node storage and compact turn path.

## Current Status

The compact turn infrastructure is wired through the native build and the testdata simulation corpus.

- `make compact_turn_simulation_tests` runs all `src/tests/resources/testdata.js` simulation cases through the compact turn oracle path.
- The current compact turn callable coverage is `452/452` unique testdata sources.
- `make compact_turn_coverage` currently reports `80/452` native compact kernels and `372/452` interpreter bridge backends with `COMPILED_RULES_MAX_ROWS=99`.
- `make solver_compact_parity_smoke` passes the smoke corpus through normal and compact solver node storage.
- `make solver_compact_parity` supports targeted checks with `SOLVER_COMPACT_PARITY_GAME` and `SOLVER_COMPACT_PARITY_LEVEL`.

So the current achievement is full callable compact-turn coverage, not full native compact codegen coverage. The interpreter bridge is deliberately part of the coverage story: it lets every game exercise the compact ABI while native kernels are added incrementally.

The current public/runtime naming is:

- `FullState` for the full mutable runtime state.
- `CompactState` for solver/generator graph nodes.
- `ps_full_state_*` for the public C API.
- `ps_session_*`, `compiled_tick_*`, and `compact_tick_*` remain only as compatibility aliases for older callers, generated-symbol compatibility, old benchmark JSON, or Makefile muscle memory.

## Key Locations

Solver-side compact state and graph search live mostly in:

- `native/src/solver/main.cpp`
  - `CompactState`
  - `compactStateFromFullState`
  - `compactStateKey`
  - `compactStateWithTiming`
  - `materializeCompactStateIntoFullState`
  - `trySpecializedCompactTurn`
  - `runSearch`

Runtime compact turn ABI and fallback support live in:

- `native/src/runtime/compiled_rules.hpp`
  - `CompactStateView`
  - `SpecializedCompactTurnOutcome`
  - `SpecializedCompactTurnBackend`
  - `compactStateInterpretedTurnBridge`
- `native/src/runtime/compiled_rules.cpp`
  - compact bridge materialization and copy-back helpers
- `native/src/runtime/c_api.cpp`
  - compact turn oracle checks for the simulation harness

Generated compact backend code is emitted from:

- `native/src/cli/main.cpp`
  - compact turn support classification
  - generated `specialized_compact_turn_step_*` functions
  - native-vs-bridge coverage metadata

Useful Make targets are:

- `make compact_turn_simulation_tests`
- `make compact_turn_coverage`
- `make solver_compact_parity_smoke`
- `make solver_compact_parity`
- `make solver_smoke_tests`
- `make solver_parity_smoke`
- `make generator_smoke_tests`
- `make solver_focus_compare`

Targeted compact parity examples:

```sh
make solver_compact_parity \
  SOLVER_COMPACT_PARITY_GAME='a distant sunset.txt' \
  SOLVER_COMPACT_PARITY_LEVEL=37 \
  SOLVER_COMPACT_PARITY_TIMEOUT_MS=1000
```

## Solver Flow

At a high level, the solver is moving toward this shape:

1. Start with an interpreter-created initial `FullState`.
2. Convert it to `CompactState` with `compactStateFromFullState`.
3. Hash the compact state with `compactStateKey`.
4. Store compact states in the search graph instead of full states where possible.
5. For each candidate input, call `trySpecializedCompactTurn`.
6. If the specialized compact turn handles the game/input, use the returned compact child state directly.
7. If it does not handle the game/input, materialize a `FullState` with `materializeCompactStateIntoFullState` and run the interpreted turn path.
8. Compare, insert, and prioritize states in the solver frontier.

The long-term target is:

```text
turn(compact_state, input) -> {changed, won, restarted, next_compact_state, heuristic}
```

The current implementation is not there yet. It has the ABI and oracle scaffolding, plus a small set of native compact kernels, while most games still pass through the bridge.

## Interpreter Fallback

The interpreter remains the correctness oracle and fallback.

When a game cannot yet be represented by native compact code, generated metadata marks it as bridge-supported. The compact backend can then call `compactStateInterpretedTurnBridge`, which reconstructs a temporary `FullState`, runs the interpreted turn path, and copies the resulting compact state back out through `CompactStateView`.

This fallback is useful for two reasons:

- It keeps all compact turn entrypoints callable while native support grows.
- It gives the oracle path a direct way to compare compact behavior against the reference interpreter.

For solver performance, bridge-backed games are not the destination. They mainly protect correctness and keep the ABI honest while native compact kernels are filled in.

## Movement Words

`CompactState` does not contain movement words. That is intentional for solver graph identity because solver nodes are end-of-turn states, and after an input fully resolves there is no outstanding movement to remember.

- `CompactState` has object occupancy bitsets plus RNG state.
- `compactStateKey` skips movement words entirely; equality and `byteSize()` cover object bitsets and RNG.
- `materializeCompactStateIntoFullState` zero-initialises `liveMovements`.
- `trySpecializedCompactTurn` passes `nullptr`/`0` for movement fields in `CompactStateView`. The bridge already copes: when movement words are missing it zeroes `liveMovements` on materialise and skips copy-back.

The C API oracle (`CompactOracleState` in `runtime/c_api.cpp`) is independent and still tracks movement words for runtime parity checks against the interpreter. That can stay as it is for now because its scope is intra-turn comparison, not solver node identity.

## Randomness

Random state is present in `CompactState`, because random rules can make two visually identical boards have different futures.

It should stay there. Even if the immediate solver/generator focus group excludes random and `randomDir` games for iteration stability, that exclusion is a benchmark-selection policy, not a compact-state simplification. The compact representation should remain semantically complete enough that a compact state can be materialized back into a valid runtime state.

So RNG fields should remain in equality, hashing, materialization, and oracle comparison unless we introduce a separate explicitly deterministic-only representation. If that ever happens, it should be a distinct type or mode, not an implicit weakening of `CompactState`.

## Missing Functionality

The main missing pieces are native behavior coverage and solver-oriented cleanup.

- Keep RNG in compact solver node identity, while continuing to exclude random games from focus-group mining when that improves iteration stability.
- Continue replacing interpreter bridge cases with native compact kernels.
- Add native compact support for remaining PuzzleScript features: complex movement creation, ellipsis patterns, commands, `again`, `cancel`, `restart`, `checkpoint`, `win`, rigid groups, random/randomDir if they return to scope, late rules, and more complex multi-cell/multi-row patterns.
- Move more solver heuristics toward compact or generated code so the solver avoids repeated materialization.
- Keep generator integration focused on the same compact solver path instead of creating a separate representation.
- Measure whether compact state storage and exact equality reduce graph overhead on nontrivial solver focus games.

## Near-Term Checklist

1. Resume native compact kernel coverage work, using `compact_turn_coverage` as the progress counter. Largest remaining buckets include bridge-backed early and late rule features.
2. Extend native compact turn support past the current direct-cell cases so multi-cell late and early object replacements can run without the bridge.
3. Measure `make solver_focus_compare` to track real solver impact of native coverage gains.
4. Re-run targeted compact parity for any suspicious case before using the full corpus gate.
5. Re-run `make compact_turn_simulation_tests`, `make compact_turn_coverage`, and `make solver_compact_parity` after meaningful kernel extensions.
