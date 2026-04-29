# Compact Turn Clean Codegen Plan

> **Intent:** Rebuild native compact turn code generation from PuzzleScript semantics, not from game-shape recognizers. The compiler should grow by implementing language features directly and letting testdata failures expose missing semantics.

## Status

This is the active compact-turn implementation plan as of 2026-04-29. It
supersedes the earlier shape-specific compact-turn prototype notes in
`native/src/compiler/IMPLEMENTATION_CHECKLIST.md`.

The runtime ontology has also settled since the first compact-state experiments:
`PersistentLevelState` owns a cell-major persistent board plus RNG. The only
object-major structure that should remain in the normal turn path is the
derived `Scratch::objectCellBits` / `Scratch::objectCellCounts` rule-scan index.

## Summary

The previous compact turn prototype drifted into shape-specific helpers such as simple push, push chain, target clear, and movement-only variants. That direction is too complex and too brittle: it encourages recognizing individual game families instead of compiling PuzzleScript.

This plan resets compact turn code generation around a strict mode split:

```text
interpreter mode: always use the interpreter / bridge
compiler mode: always use generated compact native code
```

There should not be an expanding matrix of native eligibility checks and fallback reasons. If compiler mode is selected, the generator emits code for the current implementation level. At first much of the corpus will fail. The job is to work through failures until the compiler becomes correct.

The primary progress metric is **test case pass count in compiler mode**, not fallback coverage. Coverage can still describe what happened, but it is not the goal.

## Non-Goals

- Do not add per-game or per-shape recognizers.
- Do not reintroduce helpers like `emitCompactSimplePushMovement`, `emitCompactPushChainMovement`, or `compactLooksLikeSimplePushRules`.
- Do not maintain a live-service-style feature fallback matrix.
- Do not silently fall back from compiler mode to interpreter mode on unsupported features.
- Do not optimize before the generic semantic compiler is correct.
- Do not treat backend support/fallback percentages as the main success metric.

## Design Principles

- **Compile PuzzleScript features, not game shapes.**
- **Let tests fail loudly in compiler mode.**
- **Keep interpreter mode as an explicit switch for correctness comparison and normal safe execution.**
- **Use the simulation corpus as the bring-up queue.**
- **Measure progress by compiler-mode testdata passes.**
- **Prefer one generic code path per semantic phase.**
- **Use templates only to make emitted C++ readable, not to hide bespoke special cases.**

## Reference Implementation

`src/js/engine.js` is a primary reference, not incidental background. It already contains working generated JavaScript for most of the rule interpreter machinery. The native generator should study and translate those generator structures before inventing new ones.

Important reference points:

- `generate_moveEntitiesAtIndex`
- `generate_repositionEntitiesAtCell`
- `Rule.prototype.generateCellRowMatchesFunction`
- `CellPattern.prototype.generateMatchFunction`
- `CellPattern.prototype.generateReplaceFunction`
- `generateMatchCellRow`
- `generateMatchCellRowWildCard`
- `Rule.prototype.generateApplyAt`
- `applyRandomRuleGroup`
- `applyRuleGroup`
- `applyRules`
- `generate_resolveMovements`
- `processInput`
- `processCommandQueue`
- `checkWin`

The first implementation pass should map these JS generator phases to native compact codegen phases, noting where native compact state layout changes the emitted accessors but not the semantics.

## Runtime Modes

### Interpreter Mode

Interpreter mode is the stable path. It may use the existing interpreter bridge for compact state execution. This is the mode to use when we want known-correct behavior while the compiler is incomplete.

### Compiler Mode

Compiler mode means generated compact code is mandatory. The backend should not decline individual games based on detected feature combinations. Unsupported semantics should fail through compile errors, runtime assertions, oracle mismatches, or simulation failures.

The generated code can still contain explicit TODO traps for unimplemented semantic constructs, but those traps are part of compiler bring-up, not a fallback system.

## Phase 0: Sane Runtime Ontology

Before compact compiler work resumes, remove duplicated state vocabularies. The codebase should have one name for each concept:

- `GameInformation`: immutable compiled game data.
- `GameSession`: runtime session container.
- `MetaGameState`: session/metagame flow only.
- `PersistentLevelState`: turn-persistent within-level state only.
- `LevelDimensions`: immutable per-level width/height context.
- `Scratch`: temporary turn execution buffers.
- `TurnResult`: turn effects and summary.

Cleanup order:

1. Remove alias/legacy type layers such as `PreparedFullState`; the type is `MetaGameState`.
2. Keep one persistent board authority: `PersistentLevelState::board.objects`.
   `LevelTemplate` is compiled/metagame level data, and scratch boards or
   object-cell indexes are derived execution buffers.
3. Collapse solver-local `SearchNodeState` into the cleaned `PersistentLevelState` or a thin alias.
4. Quarantine `CompactStateView` as interpreter-bridge plumbing only, then replace it in compiler mode.
5. Merge duplicated compact materialization paths into one runtime helper.
6. Move compiler/codegen responsibilities out of `native/src/cli/main.cpp`.
7. Make `native/src/runtime/core.hpp` a beautiful first-contact file: split
   mask primitives, compiled-game IR, session types, and runtime API into
   focused headers so `core.hpp` can read as a small front door rather than a
   warehouse.

This phase is now the #1 priority. New compact codegen should not build on duplicated infrastructure.

## Turn Core Boundary

The compiler-mode turn boundary should follow the four-part runtime state model instead of introducing parallel compact state structures.

Persistent within-level state contains only data that changes from turn to turn:

```cpp
struct PersistentBoardState {
    // Cell-major object masks: objects[tileIndex * strideObject + word].
    MaskVector objects;
};

struct PersistentLevelState {
    PersistentBoardState board;
    RandomState rng;
};
```

If heuristics or profiling tools want an object-major view, they must derive it
from this board or maintain an explicitly scratch-only cache. They should not
introduce a second persistent state representation.

Immutable per-level dimensions are context:

```cpp
struct LevelDimensions {
    int32_t width;
    int32_t height;
};
```

`layerCount` is not part of `LevelDimensions`; all levels in a game share the same collision-layer structure, and the specialized compiler knows the layer count from compiled game data. Message levels are metagame/session flow and should not enter solver or turn-core simulation.

The generic compiled turn shape is:

```cpp
TurnResult take_turn_compiled(
    const LevelDimensions& dimensions,
    PersistentLevelState& levelState,
    Scratch& scratch,
    ps_input input,
    const RuntimeStepOptions& options);
```

For fully specialized generated code, `LevelDimensions` may be compiled into the generated function for a known level. `Scratch` is still explicit: movements, dirty masks, replacement buffers, rigid masks, and other temporary execution storage are not persistent state.

`GameInformation` and `MetaGameState` do not belong in the compiled turn-core boundary. They are orchestration inputs for session/interpreter paths:

```cpp
TurnResult take_turn_interpreted(
    const GameInformation& game,
    const MetaGameState& meta,
    const LevelDimensions& dimensions,
    PersistentLevelState& levelState,
    Scratch& scratch,
    ps_input input,
    const RuntimeStepOptions& options);
```

`CompactStateView` is bridge-era adapter plumbing only. It can remain while the interpreter bridge needs raw pointers, but compiler-mode codegen should not be designed around it.

## Compiler Architecture

### Phase 1: Move Compact Codegen Out Of The CLI Blob

**Goal:** Create a dedicated compact codegen module with small, named semantic emitters.

**Progress:**
- [x] Created `native/src/compiler/compact_turn_codegen.hpp`.
- [x] Created `native/src/compiler/compact_turn_codegen.cpp`.
- [x] Moved per-source compact backend emission out of `native/src/cli/main.cpp`.
- [ ] Move mode selection and future semantic compact emitters into the new
  module.

Files:
- Create: `native/src/compiler/compact_turn_codegen.hpp`
- Create: `native/src/compiler/compact_turn_codegen.cpp`
- Modify: `native/src/cli/main.cpp`

Expected shape:

```cpp
struct CompactCodegenOptions {
    bool interpreterMode = false;
};

std::string generateCompactTurnBackend(
    const GameInformation& compiledGame,
    size_t sourceIndex,
    CompactCodegenOptions options);
```

The CLI should choose interpreter mode or compiler mode at a coarse level only.

### Phase 2: Emit A Semantic Turn Skeleton

**Goal:** Generate a full compact turn function with all major PuzzleScript phases present, even if many phases initially contain TODO traps.

Generated structure:

```cpp
TurnResult specialized_compact_turn_source_N(
    const LevelDimensions& dimensions,
    PersistentLevelState& levelState,
    Scratch& scratch,
    ps_input input,
    const RuntimeStepOptions& options
) {
    // validate level dimensions / state storage
    // decode input direction
    // seed input movements
    // apply early rulegroups
    // resolve movement
    // apply late rulegroups
    // process commands / again policy
    // evaluate win conditions
    // canonicalize / return result
}
```

Acceptance:
- Compiler mode emits this function for every source.
- Interpreter mode emits the bridge call.
- There is no per-source native eligibility gate.
- Compiler-mode code does not take `Game&` or `CompactStateView` as its semantic turn boundary.

### Phase 3: Compact State Access Layer

**Goal:** Establish generic helpers for compact state reads/writes used by every generated phase.

Features:
- object bit query
- object bit set/clear
- movement bit query/set/clear
- collision layer helpers
- cell coordinate iteration
- row scan helpers for directional rules

Acceptance:
- Generated code no longer hand-writes object-major bit indexing in every semantic emitter.
- Tests may still fail because semantics are incomplete.

### Phase 4: Input Movement Seeding

**Goal:** Implement generic player input movement seeding.

Semantics:
- map input to movement mask
- find all player objects from compiled player mask
- mark player layer movements
- respect action/tick input behavior

Acceptance:
- Simple movement-only games start producing meaningful compact native transitions in compiler mode.
- Oracle mismatches become the driver for missing edge cases.

### Phase 5: Rule Matching Without Movement Resolution

**Goal:** Compile generic rule matching and replacements before solving push/movement.

Features:
- one-row rules
- multi-cell directional scans
- object-present constraints
- object-missing constraints
- movement-present constraints
- movement-missing constraints
- object set/clear replacements
- movement set/clear replacements
- rulegroup loop-until-stable behavior

Deferred until later:
- ellipsis
- random
- rigid
- commands with control flow
- loop points

Acceptance:
- No-movement transformation games should pass in compiler mode.
- Failures should be semantic mismatches, not missing shape recognizers.

### Phase 6: Generic Movement Resolution

**Goal:** Implement PuzzleScript movement resolution as a generic algorithm over collision layers and movement masks.

Features:
- blocked movement
- chain movement / pushing as a consequence of movement propagation
- collision layer occupancy rules
- movement mask clearing
- changed flag computation

Acceptance:
- Sokoban-like games pass because the generic movement algorithm works, not because Sokoban was recognized.

### Phase 7: Late Rules, Win Conditions, And Terminal Events

**Goal:** Complete the standard turn envelope.

Features:
- late rulegroups
- win conditions
- restart/reset/cancel terminal treatment for solver
- `again` according to `RuntimeStepOptions::againPolicy`
- deterministic RNG state threading where needed

Acceptance:
- Compact compiler mode can run a meaningful subset of the simulation corpus end-to-end.

### Phase 8: Expand Language Coverage

Implement remaining PuzzleScript semantics one feature at a time:

- ellipsis
- random rules and random replacements
- rigid rules
- loop points
- commands
- sfx/message/checkpoint effects
- aggregate masks and properties where current lowering exposes them

Each feature should be implemented in the generic semantic emitters, not in a game-shape side path.

## CLI / Harness Controls

Add a coarse switch for compact turn execution mode.

Suggested names:

```text
--compact-turn-mode=interpreter
--compact-turn-mode=compiler
```

Default can remain interpreter mode until compiler mode has substantial corpus coverage.

Compiler-focused targets should force compiler mode and expect failures early in development:

```text
make compact_turn_codegen_bringup
make compact_turn_codegen_one SOURCE=...
```

The existing oracle/simulation targets can keep using interpreter mode unless explicitly testing compiler mode.

## Test Strategy

Use tests as the feature backlog.

1. Start with one tiny fixture in compiler mode.
2. Run oracle against interpreter.
3. Fix the first semantic mismatch.
4. Add the next fixture.
5. Sort or group failing cases by feature only to make the work easier to batch.
6. Scale from solver smoke fixtures to selected testdata.
7. Then run the full simulation corpus.

Recommended progress report:

```text
compiler_mode_testdata: passed=N failed=M total=T
first_failure: path/to/game.txt case=...
suspected_feature: movement resolution / ellipsis / random / ...
```

This should replace “native fallback coverage” as the main planning signal.

Useful gates:

```bash
git diff --check
make build
make compact_turn_coverage
make compact_turn_oracle_smoke
make compact_turn_simulation_tests
```

For compiler bring-up, add new failing-friendly targets rather than weakening stable gates.

## Acceptance Criteria

- Compact codegen is organized in a dedicated module, not embedded as a long CLI string-concatenation block.
- There are exactly two high-level modes: interpreter and compiler.
- Compiler mode does not silently fall back per feature or per game.
- No shape-specific recognizers or emitters exist.
- Generic semantic phases drive the generated code.
- `src/js/engine.js` is used as the semantic codegen model.
- Compiler-mode testdata pass count is used as the roadmap for missing compiler semantics.
