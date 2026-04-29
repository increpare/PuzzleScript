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
- [x] Move mode selection and future semantic compact emitters into the new
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

**Progress:**
- [x] Added coarse compact turn codegen mode plumbing.
- [x] Compiler mode emits a native compact-turn skeleton with the semantic
  phase outline.
- [x] Add compiler-mode bring-up targets that expect current failures.
- [x] Split compiler mode into an ABI wrapper plus an inner turn-core function
  with the planned `LevelDimensions` / `PersistentLevelState` / `Scratch`
  boundary.
- [ ] Replace the skeleton TODO phases with generated code, one phase at a
  time.

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

**Progress:**
- [x] Emit source-local cell-major access helpers for object and movement
  storage.
- [x] Emit state preparation/validation for dimensions, persistent board
  storage, and scratch movement storage.
- [x] Add collision layer helpers.
- [x] Add row scan helpers for directional rules.
- [x] Read and write packed 5-bit movement fields across `MaskWord`
  boundaries, matching the interpreter mask helpers.

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

**Progress:**
- [x] Emit generated input-to-direction mapping.
- [x] Emit player-mask and object-layer tables from compiled game data.
- [x] Seed player collision-layer movement bits in `Scratch::liveMovements`.
- [x] Turn seeded movement into handled transitions once movement resolution
  exists.

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
- [x] one-row rules
- [x] multi-cell directional scans
- [x] object-present constraints
- [x] object-missing constraints
- [x] movement-present constraints
- [x] movement-missing constraints
- [x] object set/clear replacements
- [x] movement set/clear replacements
- [x] rulegroup loop-until-stable behavior
- [x] deterministic multi-row rule tuples

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

**Progress:**
- [x] Align generated compact tile indexing with the JS/native column-major
  level layout.
- [x] Emit a generic collision-layer movement pass over
  `Scratch::liveMovements`.
- [x] Resolve movement for collision layers whose 5-bit movement field spans
  two mask words.
- [ ] Handle blocked movement side effects and rigid/cancel interactions.
- [ ] Return handled compiler-mode transitions once win/terminal semantics are
  available.

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

**Progress:**
- [x] Emit generic win-condition evaluation for compiler mode.
- [x] Return handled compiler-mode turns for games with no early/late
  rulegroups.
- [x] Apply late rulegroups through the same deterministic one-row rulegroup
  emitter used for early rules.
- [x] Handle `require_player_movement` cancellation for compiler-mode turns.
- [ ] Handle restart/reset/cancel terminal treatment for solver.
- [ ] Handle `again` policy.
- [x] Thread deterministic RNG for random rule-group selection.
- [x] Queue simple output commands (`message`, `again`) for compiler-mode
  result parity.

Features:
- late rulegroups
- win conditions
- restart/reset/cancel terminal treatment for solver
- `again` according to `RuntimeStepOptions::againPolicy`
- deterministic RNG state threading where needed

Acceptance:
- Compact compiler mode can run a meaningful subset of the simulation corpus end-to-end.

### Phase 8: Expand Language Coverage

Implement PuzzleScript semantics one feature at a time.

Recently added compiler coverage:

- ellipsis rows
- random rule groups
- rigid rule retry for unresolved movement
- random replacements

Remaining coverage:

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
make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=...
make compact_turn_codegen_selected_tests
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

Use the ranked frontier helper before choosing the next `testdata.js` case:

```bash
make compact_turn_codegen_frontier
```

This sorts cases by rough source/input/rule/level size so compiler bring-up
does not accidentally chase large games just because they appear early in the
file. Differential compact-oracle checks remain the authority once a case is
chosen.

Current compiler-mode testdata foothold:

```text
make compact_turn_codegen_testdata_one
  default case: 1, "sokoban no win condition"
  result: passes with compact oracle checks

make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=2
  case: "sokoban with win condition"
  result: passes with compact oracle checks

make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=3
  result: passes with compact oracle checks

make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=4
  result: passes with compact oracle checks

make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=5
  case: "by your side"
  result: passes with compact oracle checks
  added coverage: deterministic multi-row rules and require_player_movement cancellation

make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=6
  result: passes with compact oracle checks
  added coverage: ellipsis row matching with broad oracle exercise

Additional ranked-frontier probes:
  case 7, "ellipsisPropagationBug1": passes
  case 8, "ellipsisPropagationBug2": passes
  case 9, "undo test": passes
  case 13, "rigid body test": passes
  case 16, "annoying edge case": passes
  case 23, "beginloop/endloop with mutual recursion": passes
  case 26, "simple sokobond test": passes
  case 27, "loop length 1": passes
    added coverage: generated group dispatcher honors startloop/endloop loop-point propagation
  case 36, "rule application hat test": passes
  case 37, "ortho test 1": passes
  case 38, "ortho test 2": passes
  case 40, "don't mask movements if no movements happening": passes
  case 42, "Remove movements from empty layers after rule application": passes
  case 43, "movement matching - correctly matching different objects same cell moving in different directions": passes
  case 44, "movement matching - ellipsis bug - forgot to include one case in above": passes
  case 45, "ellipsis bug: rule matches two candidates, first replacement invalidates second": passes
  case 48, "random movement determinism test": passes
  case 49, "random instances of properties": passes
  case 108, "multiple patterns not checking for modifications": passes
  case 111, "Make synonyms of properties work. #215": passes
  case 112, "Make synonyms of properties work. #230": passes
  case 114, "Make synonyms of properties work. #243": passes
  case 116, "Failed rigid groups shouldn't block late rule execution. #254": passes
  case 119, "Laser movement check (#264)": passes
  case 121, "Rigid weirdness test (#369)": passes
  case 122, "Synonym confusion": passes
  case 125, "Reserved keywords are too greedy (#419)": passes
  case 126, "Removing background tiles breaks \"no X\" wincondition (#534)": passes
  case 131, "undoing reset undoes two steps, not one #453": passes
  case 134, "Win condition test \"NO X\"": passes
  case 135, "Win condition test \"SOME X\"": passes
  case 140, "Test for trigger message at same time as cancel": passes
  case 143, "third test for #492 movement not getting correctly cleared from tile": passes
  case 144, "fourth test for #492 movement not getting correctly cleared from tile": passes
  case 145, "fifth test for #492 movement not getting correctly cleared from tile": passes
  case 146, "random rules - report by caeth": passes
    added coverage: generic random rule-group candidate selection and persistent RNG threading
  case 147, "right [ vertical Player | perpendicular Player ] -> [ perpendicular Player | ] produces error #682": passes
  case 148, "right [ horizontal TestObject1 | perpendicular TestObject1 ] -> [ perpendicular TestObject1 | ] produces an error #498": passes
  case 149, "[ orthogonal a | moving a ] -> [ moving a | orthogonal a ] produces an error #496": passes
  case 150, "1st alternative test for: right [ vertical Player | perpendicular Player ] -> [ perpendicular Player | ] produces error #682": passes
  case 151, "2nd alternative test for: right [ vertical Player | perpendicular Player ] -> [ perpendicular Player | ] produces error #682": passes
  case 152, "super tricky (related to #469) right [ vertical playerortarget | vertical player ] -> [ playerortarget | playerortarget ]": passes
  case 153, "right [ vertical playerortarget | vertical player ] -> [ vertical player | vertical playerortarget ]": passes
  case 155, "gallery game: at the hedges of time": passes
    added coverage: movement resolution for packed 5-bit layer fields that cross `MaskWord` boundaries
  case 322, "\"right [ Player ] -> [ up Player ]\" gets compiled to down #755": passes
    added coverage: horizontal row-match traversal order matches interpreter row-major scan
  case 363, "rigid applies to movements even if the movements are chagned by subsequent non-rigid rules in other groups": passes
  case 364, "rigid applies to movements even if the objects are changed to different objects in the same layer": passes
  case 367, "rigid applies to movements even if the initial movement applied is an ACTION that is later changed to a movement": passes
  case 374, "misc rigid test": passes
  case 382, "Audio Test 2": passes
  case 398, "Autowin": passes
  case 399, "Autowin2": passes
  case 415, "trigger delete newmetadata.flickscreen": passes
  case 416, "trigger delete newmetadata.zoomscreen": passes
  case 417, "parser rigid in strange place highlighting test": passes
  case 418, "background_color transparent code path": passes
  case 419, "Levels can not contain glyphs that resemble section names #976": passes
  case 421, "Missing/Skipping Rules? Objects disappear for no reason? #1046": passes

Selected compiler-mode testdata progress: 203/469 known passing.

Full-prefix compiler-mode sweep, 2026-04-29:
  command:
    per-case `make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=N COMPILED_RULES_BUILD_JOBS=4`
  result before interruption:
    attempted 188/469
    passed 177
    failed 11
    interrupted on case 189 after a pathological generated-source compile
  failed semantic oracle cases:
    31 "collapse simple"
    32 "collapse long"
    57 "Flying Kick"
    62 "Cute Train"
    85 "Sok7"
    87 "Color Chained"
    88 "Drop Swap"
    89 "Drop Swap 2"
    90 "Drop Swap 3"
    155 "gallery game: at the hedges of time"
    188 "gallery: marble shot"
  compile-time blocker:
    case 189 generates 200 groups / 4739 rules and did not finish compiling
    within a useful interactive window.
  artifacts:
    build/compiled-rules/testdata-compact-compiler-per-case-sweep.tsv
    build/compiled-rules/testdata-compact-compiler-per-case-sweep.log

Again-probe progress update, 2026-04-29:
  result against the previous attempted prefix:
    attempted 188/469
    passed 186
    remaining semantic oracle failure: 155 "gallery game: at the hedges of time"
    case 189 remains the next compile-time frontier
  fixed from the previous failed set:
    31 "collapse simple"
    32 "collapse long"
    57 "Flying Kick"
    62 "Cute Train"
    85 "Sok7"
    87 "Color Chained"
    88 "Drop Swap"
    89 "Drop Swap 2"
    90 "Drop Swap 3"
    188 "gallery: marble shot"
  semantic fix:
    Generated compact turns now mirror the interpreter's `again` scheduling
    probe: after a modifying turn with `again`, run one tick in dont-modify
    probe mode, restore board/movement state, preserve RNG advancement, and
    schedule another tick only if the probe would change state.

Cross-word movement progress update, 2026-04-29:
  selected old-failure replay:
    cases: 31 32 57 62 85 87 88 89 90 155 188
    result: 11/11 pass
  prefix status:
    attempted 188/469
    passed 187
    case 189 remains the next compile-time frontier
  fixed semantic oracle failure:
    155 "gallery game: at the hedges of time"
  semantic fix:
    Generated compact movement helpers now mirror the interpreter's packed
    movement-bit access when a layer's 5-bit direction field spans two
    `MaskWord`s. This keeps rule replacements, player seeding, rigid masks,
    and movement resolution on one generic movement accessor instead of
    assuming every layer fits in one word.

Codegen-size iteration note, 2026-04-29:
  fast selected old-failure replay:
    cases: 31 32 57 62 85 87 88 89 90 188
  slow compile sentinels, run deliberately at the end of an iteration:
    155 "gallery game: at the hedges of time"
    189 "gallery: cyber-lasso"
  rationale:
    case 155 is semantically passing but compiles slowly enough to nuke the
    inner loop; case 189 remains the generated-source size frontier. Keep both
    out of the fast selected replay, then return to them when evaluating source
    size, sharding, constant deduplication, and other compile-time work.

Codegen-size cleanup update, 2026-04-29:
  selected replay after direct canonical mask references:
    cases: 31 32 57 62 85 87 88 89 90 188
    result: 10/10 pass
  case 189 generated-source probe:
    before direct references: 1,256,987 lines, 73 MB, 81,663 pointer aliases
    after direct references: 1,175,324 lines, 62 MB, 0 pointer aliases
  cleanup:
    Pattern functions still stay specialized, but generated mask uses now
    reference canonical `compact_turn_mask_data_*` arrays directly instead of
    emitting one pointer alias per pattern field.

Aggregate/transition progress update, 2026-04-29:
  ranked-frontier replay:
    cases: 323 365 366 368 369 370 371 372 373 375 379 381 385 391 422 423 424 425
    result: 18/18 pass
  added targeted replay:
    case 424 "aggregate player allowed C #1032": passes
  semantic fixes:
    Generated compact movement now mirrors the interpreter's atomic aggregate
    player movement rule, so multi-layer aggregate players do not split when a
    constituent layer is blocked.
    Generated compact turn results now mark win transitions when another level
    exists and include board modification/transition in `result.changed`.

Frontier expansion update, 2026-04-29:
  ranked-frontier replay:
    cases: 136 123 129 110 17 130 137 14 251 138 139 387 397 396 376 19 141 142 28 384
    result: 20/20 pass
  added coverage:
    `SOME/NO X ON Y` win conditions, property clearing from `no property` RHS,
    restart/cancel/message command interactions, additional loop/rigid cases,
    double-ellipsis fixtures, and aggregate win-condition variants.

Second frontier expansion update, 2026-04-29:
  ranked-frontier replay:
    cases: 132 321 386 260 361 380 12 413 118 320 197 263 305 427 428 430 434 433 429 451
    result: 20/20 pass
  added coverage:
    More loop and late-rule fixtures, random property concretization, broader
    rigid disablement checks, gallery-sized small games, and the first wave of
    `#1067` `[ player no wall ]` fixtures.

Third frontier expansion update, 2026-04-29:
  ranked-frontier replay:
    cases: 431 432 440 445 452 439 446 426 435 436 437 441 438 447 442 444 453 454 448 450
    result: 20/20 pass
  added coverage:
    More `#1067` `[ player no wall ]` fixtures covering one-rule
    object-absence pattern variants.

Fourth frontier expansion update, 2026-04-29:
  ranked-frontier replay:
    cases: 443 449 420 383 127 124 346 458 457 461 462 464 466 459 465 468 463 467 460 456
    result: 20/20 pass
  added coverage:
    Remaining small `#1067` object-absence fixtures, overlapping-object
    movement, rule direction inference, glyph parsing with `=`, audio-trigger
    rules, and simple level-change recording fixtures.

Fifth frontier expansion update, 2026-04-29:
  ranked-frontier replay:
    cases: 455 133 326 47 173 194 99 388 345 30 261 100 11 243 10 362 304 95 22 105
    result: 20/20 pass
  added coverage:
    Level-change recording, undo/realtime, many-layer and many-object
    fixtures, multi-word dictionary names, restart, mirror rules, push/pull,
    rigid preexisting movement behavior, and broader loop coverage.

Sixth frontier expansion update, 2026-04-30:
  ranked-frontier replay:
    cases: 317 51 29 269 120 72 169 20 410 389 390 46 297 309 41 257 96 238 24 128
    result: 20/20 pass
  added coverage:
    Sokoban-family nested fixtures, modality and mirror-loop cases, undo and
    transition-heavy probes, aggregate win-condition variants, many-object and
    Unicode parsing fixtures, double-ellipsis magnetism, and broader late-rule
    coverage.

Executable selected-pass target:
  make compact_turn_codegen_selected_tests
  cases: COMPACT_TURN_CODEGEN_SELECTED_CASES in Makefile

Additional ranked-frontier command probe:
  case 93, "again + message combo": passes
  added coverage: simple `message` and `again` command queueing

Known next unsupported ranked-frontier feature:
  run `make compact_turn_codegen_frontier` and probe the next smallest
  unrecorded case

Known ranked-frontier unsupported feature:
  The current failures are real compact-vs-interpreter state mismatches, not
  feature-filter misses. Case 189 also shows generated C++ size/compile-time
  pressure for very large rule sets.
```

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
