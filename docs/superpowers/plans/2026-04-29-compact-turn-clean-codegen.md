# Compact Turn Clean Codegen Plan

> **Intent:** Rebuild native compact turn code generation from PuzzleScript semantics, not from game-shape recognizers. The compiler should grow by implementing language features directly and letting testdata failures expose missing semantics.

## Status

This is the active compact-turn implementation plan as of 2026-04-30. It
supersedes the earlier shape-specific compact-turn prototype notes in
`native/src/compiler/IMPLEMENTATION_CHECKLIST.md`.

The runtime ontology has also settled since the first compact-state experiments:
`PersistentLevelState` owns a cell-major persistent board plus RNG. The only
object-major structure that should remain in the normal turn path is the
derived `Scratch::objectCellBits` / `Scratch::objectCellCounts` rule-scan index.

Current state: compiler-mode compact turns pass the full `testdata.js`
simulation corpus (`469/469`) with zero compact-oracle failures. The active
work has moved from semantic bring-up to solver integration, generated-source
scale, and runtime performance against the solver focus benchmark.

Post-board-migration baseline on 2026-04-30:

- Solver focus comparison uses the same 50 targets for interpreted and compiled
  compact modes; both solve 50/50.
- Compiled compact median elapsed is `193 ms` vs interpreted `258 ms`
  (`0.748x`, 25.2% faster).
- Median step time is `111.6 ms` vs interpreted `170.2 ms` (`0.656x`, 34.4%
  faster); median clone time is `1.4 ms` vs `16.6 ms` (`0.084x`).
- Native compact turn coverage is 50/50 focus targets with no bridge,
  fallback, or unsupported compact turns.
- The current speed goal (`<=0.500x` elapsed) is not met. Slowest step targets
  include `the_saga_of_the_candy_scroll.txt#58`, `Vexatious Match 3.txt#2`,
  and `paint everything everywhere.txt` levels; the next performance work
  should inspect generated native compact phase timings and mask rebuild work
  on those targets.

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

Progress: item 2 is complete for board authority. `Scratch::interpreterBoard`
and scratch-to-persistent sync helpers have been removed from `native/src`; the
persistent board remains cell-major.

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
- [x] Replace the original skeleton TODO phases with generated semantic code
  for the full `testdata.js` simulation corpus.

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
- [x] Handle blocked movement side effects and rigid/cancel interactions for
  the `testdata.js` simulation corpus.
- [x] Return handled compiler-mode transitions once win/terminal semantics are
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
- [x] Audit restart/reset/cancel terminal treatment through the solver-specific
  compact path.
- [x] Handle `again` policy.
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

Remaining work after full-corpus correctness:

- solver focus performance: reduce compact turn setup/mask-cache rebuild cost,
  then remeasure against the interpreter baseline on like-for-like targets
- generated-source scale: keep reducing giant generated sources and raise or
  remove the solver focus line-budget misses without returning to shape
  recognizers
- solver graph overhead: investigate hash/heuristic costs now that clone and
  turn execution have moved sharply in the right direction
- benchmark/reporting polish: keep compiler-mode metrics separate from
  interpreter-bridge coverage, and report native/not-attached buckets without
  reviving feature fallback language

Any future semantic feature should be implemented in the generic semantic
emitters, not in a game-shape side path.

## CLI / Harness Controls

Add a coarse switch for compact turn execution mode.

Suggested names:

```text
--compact-turn-mode=interpreter
--compact-turn-mode=compiler
```

Default can remain interpreter mode for stable runtime behavior until the
compiler-mode solver benchmarks are consistently better and the remaining
line-budget misses are resolved.

Compiler-focused targets force compiler mode. Earlier in bring-up they were
allowed to expose failures; now the full-corpus and selected targets are
regression guards:

```text
make compact_turn_codegen_bringup
make compact_turn_codegen_testdata_one COMPACT_TURN_CODEGEN_TESTDATA_CASE=...
make compact_turn_codegen_selected_tests
make compact_turn_codegen_simulation_tests COMPILED_RULES_BUILD_JOBS=4
make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1
```

The existing oracle/simulation targets can keep using interpreter mode unless
explicitly testing compiler mode. Compiler-mode coverage and benchmark targets
should remain named separately so bridge/interpreter metrics do not get mixed
with native compiler metrics.

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

Historical selected compiler-mode checkpoint before later frontier expansion:
390/469 known passing.

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

Seventh frontier expansion update, 2026-04-30:
  ranked-frontier replay:
    cases: 77 246 278 262 179 392 393 25 347 294 219 228 275 394 395 303 232 113 286 193
    result: 20/20 pass
  added coverage:
    Multi-level romance and gallery fixtures, no-rule/menu-like execution,
    slide/pull and scale mechanics, double-ellipsis stress cases, level-change
    recording, late-rule synonym handling, and additional push-heavy puzzle
    games.

Eighth frontier expansion update, 2026-04-30:
  ranked-frontier replay:
    cases: 330 335 104 272 56 377 296 115 349 252 274 80 98 293 109 332 339 341 267 318
    result: 20/20 pass
  added coverage:
    More object/layer stress fixtures, 2048-style rule expansion, one-player
    unlimited rigidbody coverage, rasteriser/push puzzle variants, gallery
    games with heavier late-rule use, and generated-rule-count pressure up to
    hundreds of lowered rules.

Ninth frontier expansion update, 2026-04-30:
  ranked-frontier replay:
    cases: 354 97 21 245 39 78 247 277 310 67 336 273 15 295 154 287 271 409 209 229
    result: 20/20 pass
  added coverage:
    Larger gallery and parser-stress fixtures, 200-object coverage, late
    beginloop/endloop replay, many-rigidbody pressure, slot-machine/random
    mechanics, and generated-rule-count pressure up to 478 lowered rules.

Tenth frontier expansion update, 2026-04-30:
  ranked-frontier replay:
    cases: 76 82 315 192 412 357 73 183 378 356 167 338 270 79 327 221 159 160 202 186
    result: 20/20 pass
  added coverage:
    More gallery-sized games, late-rule-heavy Sokoban variants, unlimited
    rigidbody and parallel-player fixtures, larger path/auto-hurdle games, and
    generated-rule-count pressure up to 115 lowered rules.

Eleventh frontier expansion update, 2026-04-30:
  ranked-frontier replay:
    cases: 312 71 268 210 18 360 54 168 175 334 92 205 103 174 50 344 68 352 244 253
    result: 20/20 pass
  old-failure bookkeeping replay:
    cases: 31 32 57 62 85 87 88 89 90 188
    result: 10/10 pass
  added coverage:
    More gallery and input-heavy games, larger late-rule sets, many-level
    fixtures, 10-layer/50-object pressure, and the previously fixed
    `again`/movement mismatch cases now live in the selected executable target.

Twelfth frontier expansion update, 2026-04-30:
  ranked-frontier replay:
    cases: 83 285 265 199 350 70 74 227 163 107 216 266 201 264 231 258 276 55 191 225
    result: 20/20 pass
  added coverage:
    More input-heavy and gallery games, broad late-rule replay, many-level
    fixtures, and generated-rule-count pressure up to 463 lowered rules.

Thirteenth frontier expansion update, 2026-04-30:
  unselected-gap replay:
    cases: 33 34 35 52 53 58 59 60 61 63 64 65 66 69 75 81 84 86 91 94
    result: 20/20 pass
  added coverage:
    Early dense rule fixtures, late-group-heavy games, and a generated-rule-count
    pressure case with 501 lowered rules.

Fourteenth frontier expansion update, 2026-04-30:
  unselected-gap replay:
    cases: 101 102 106 117 156 157 158 161 162 164 165 166 170 171 172 176 177 178 180 181
    result: 20/20 pass
  promoted coverage:
    18/20 added to the selected executable target, including several dense
    generated-rule fixtures up to 1123 lowered rules.
  held out for compile-pressure queue:
    cases 164 and 181 passed, but compile 3611 and 4961 lowered rules
    respectively, so they stay out of the selected target until the end queue.

Fifteenth frontier expansion update, 2026-04-30:
  unselected-gap replay:
    cases: 182 184 185 187 190 195 196 198 200 203 204 206 207 208 211 212 213 214 215 217
    result: 20/20 pass
  promoted coverage:
    19/20 added to the selected executable target, including sharded generated
    kernels and generated-rule pressure up to 1459 lowered rules.
  held out for compile-pressure queue:
    case 213 passed, but compiles 5024 lowered rules, so it stays out of the
    selected target until the end queue.

Sixteenth frontier expansion update, 2026-04-30:
  unselected-gap replay:
    passing cases: 218 220 222 224 226 233 234 235 236 237 239 240 241 248 250 254 255 256 259 279 280 281 282
    compile-pressure cases: 223 230 242 249
  promoted coverage:
    23 cases added to the selected executable target; selected coverage is now
    413/469. `make compact_turn_codegen_selected_tests
    COMPILED_RULES_BUILD_JOBS=4` passes with `passed=413`.
  held out for compile-pressure queue:
    case 223 emitted 14246 lowered rules; case 230 emitted 3758; case 242
    stalled in source compile despite 1164 lowered rules; case 249 stalled in
    source compile with 343 generated groups. Keep all four out of the selected
    target until the end queue.

Seventeenth frontier expansion update, 2026-04-30:
  unselected-gap replay:
    cases: 283 284 288 289 290 291 292 298 299 301 302 307 308 311 313 314 316 319 328 329
    result: 20/20 pass
  promoted coverage:
    20 cases added to the selected executable target; selected coverage is now
    433/469. `make compact_turn_codegen_selected_tests
    COMPILED_RULES_BUILD_JOBS=4` passes with `passed=433`.
  held out for compile-pressure queue:
    preflight marks cases 324, 325, and 403 as end-queue candidates. Cases
    300, 306, and 402 are borderline by generated-rule count and should be
    probed separately.

Eighteenth frontier expansion update, 2026-04-30:
  unselected-gap replay:
    cases: 331 333 337 340 342 343 348 351 353 355 358 359 400 401 404 405 406 407 408 411 469
    result: 21/21 pass
  promoted coverage:
    21 cases added to the selected executable target; selected coverage is now
    454/469. `make compact_turn_codegen_selected_tests
    COMPILED_RULES_BUILD_JOBS=4` passes with `passed=454`. The remaining
    non-parked probes are cases 300, 306, and 402.
  held out for compile-pressure queue:
    cases 155, 164, 181, 189, 213, 223, 230, 242, 249, 324, 325, and 403
    stay out of the selected target until the end queue.

Nineteenth frontier expansion update, 2026-04-30:
  borderline probe replay:
    cases: 300 306 402
    result: 2/3 promoted
  promoted coverage:
    cases 306 and 402 passed in compiler mode and were added to the selected
    executable target; selected coverage is now 456/469. `make
    compact_turn_codegen_selected_tests COMPILED_RULES_BUILD_JOBS=4` passes
    with `passed=456`.
  held out for compile-pressure queue:
    case 300 emitted 1899 lowered rules but stalled in generated-source compile
    past the iteration budget, so it joins the end queue. The compile-pressure
    queue is now 155, 164, 181, 189, 213, 223, 230, 242, 249, 300, 324, 325,
    and 403.

Twentieth frontier expansion update, 2026-04-30:
  compile-pressure fix:
    Large non-random generated rule groups now split their group-apply dispatch
    into fixed-size helper chunks. This preserves the same consecutive-failure
    and loop semantics, but avoids producing a single massive group function.
  end-queue replay:
    passing cases: 155 164 181 189 230 242 249 300 324 325
    still held out: 213 223 403
  promoted coverage:
    10 cases added to the selected executable target; selected coverage is now
    466/469. The remaining cases need another compile-size pass rather than a
    semantics/fallback change.
    Guard: `make compact_turn_codegen_selected_tests COMPILED_RULES_BUILD_JOBS=4`
    passes with `passed=466`.

Twenty-first frontier expansion update, 2026-04-30:
  compile-size fix:
    Per-pattern generated `_matches` / `_apply` functions were replaced with
    source-local generic pattern matcher/applier helpers. Row/rule code now
    passes deduplicated mask constants into those helpers, preserving generic
    compiler semantics while cutting emitted function volume.
  end-queue replay:
    passing cases: 213 223 403
  promoted coverage:
    All `testdata.js` simulation cases are now in the selected executable
    target: 469/469. Individual held-out probes passed with compact oracle
    parity before promotion.
    Guard: `make compact_turn_codegen_selected_tests COMPILED_RULES_BUILD_JOBS=4`
    passes with `passed=469`.

Compiler-mode coverage tooling update, 2026-04-30:
  coverage signal:
    The default `compact_turn_coverage` target still reports the interpreter
    bridge-oriented compact path. Added `make compact_turn_codegen_coverage`
    for compiler-mode coverage, using `--compact-turn-mode=compiler` and
    asserting zero interpreter bridge backends.
  guard:
    `make compact_turn_codegen_coverage` reports
    `native_compact_kernels: 452/452 (100.0%)` and
    `interpreter_bridge_backends: 0/452 (0.0%)`.

Full-corpus compiler-mode simulation update, 2026-04-30:
  full-corpus guard:
    Added `make compact_turn_codegen_simulation_tests`, which compiles
    `testdata.js` once as a sharded compiler-mode compact backend and then
    runs the full simulation corpus through that generated binary.
  guard:
    `make compact_turn_codegen_simulation_tests COMPILED_RULES_BUILD_JOBS=4`
    passes with `cpp_simulation_tests_direct passed=469 failed=0 total=469`
    and `compact_turn_oracle_failures=0`.

Solver compact compiler-mode audit, 2026-04-30:
  The solver treats compact restart edges as terminal and does not store the
  child state. Compact win/transition handling is compatible with the solver's
  solved check because compiler-mode transitions also report `won`.
  `make compact_turn_codegen_bringup COMPILED_RULES_BUILD_JOBS=4` reports
  attempts=18 hits=18 fallbacks=0 with no compact oracle failures.

Generated-source cleanup, 2026-04-30:
  Removed an unused rigid group-number lookup table from compact generated
  sources and shortened emitted mask-word literals by dropping a redundant
  nested cast. Case 403 still compiles and passes oracle replay after the
  cleanup; its generated source is about 50.1 MB, so larger wins still need to
  come from reducing emitted rule bodies rather than polishing constants.

Generated mask arena cleanup, 2026-04-30:
  Per-phase rule masks now emit as one constexpr mask arena plus pointer
  offsets instead of one named array per unique mask. Case 403 still compiles
  and passes compact oracle replay; generated source dropped to about 43 MB /
  784,204 lines.

Generated symbol-name cleanup, 2026-04-30:
  Generated-internal compact rule/row/group helper names now use short
  source-local prefixes while preserving the public backend ABI names. Case 403
  still passes compact oracle replay; generated source dropped again to about
  41 MB / 42,784,588 bytes.

Deterministic rule wrapper cleanup, 2026-04-30:
  Deterministic rule `_apply` functions now inline rule-level collect,
  still-match, and apply-tuple forwarding instead of emitting separate wrapper
  functions. Random groups still emit tuple helpers because random candidate
  selection needs them. Case 403 still passes compact oracle replay; generated
  source dropped to about 34 MB / 653,032 lines.

One-row deterministic apply cleanup, 2026-04-30:
  One-row deterministic rules now iterate their single match row directly
  instead of emitting the generic multi-row tuple-index loop. This preserves
  the first-match/no-recheck behavior and the subsequent still-match checks.
  Case 403 still passes compact oracle replay; generated source dropped to
  about 30 MB / 537,292 lines.

One-row deterministic row-helper cleanup, 2026-04-30:
  One-row deterministic rules now inline row still-match and replacement
  application code into the rule apply function instead of emitting separate
  row helper functions. Random groups and multi-row deterministic rules still
  keep row helpers where candidate selection or tuple assembly needs them.
  Case 403 still passes compact oracle replay; generated source dropped to
  29,386,948 bytes / 498,712 lines.

Generated row-helper interning, 2026-04-30:
  Compact row match/apply/collect helpers are now interned by emitted function
  body, so duplicated lowered row semantics share one generated helper instead
  of emitting the same scan/replacement body under many rule-local names. This
  is a codegen infrastructure cleanup, not a game-shape recognizer.
  Focused giant-case results:
  - case 223 `vertebrae`: 57,369,593 bytes / 980,324 lines to
    52,547,375 bytes / 917,205 lines.
  - case 213 `gallery: season finale`: 37,791,520 bytes / 656,218 lines to
    25,604,333 bytes / 474,405 lines.
  - case 403 `Crate Assembler`: 29,386,948 bytes / 498,712 lines to
    18,467,682 bytes / 329,116 lines.
  Full generated `testdata.js` corpus dropped from 467,843,800 bytes /
  8,166,556 lines to 383,808,060 bytes / 6,975,910 lines. Full compiler-mode
  simulation guard still passes 469/469 with no compact oracle failures.

Generated rule-apply interning, 2026-04-30:
  Compact rule `_apply` helpers are now interned by emitted function body after
  row-helper names have been canonicalized, so identical deterministic dispatch
  bodies share one generated helper. This is another generic codegen-size pass,
  not a feature/fallback gate.
  Focused giant-case results:
  - case 223 `vertebrae`: 52,547,375 bytes / 917,205 lines to
    47,814,855 bytes / 856,731 lines.
  - case 213 `gallery: season finale`: 25,604,333 bytes / 474,405 lines to
    25,092,133 bytes / 470,189 lines.
  - case 403 `Crate Assembler`: 18,467,682 bytes / 329,116 lines to
    6,598,468 bytes / 147,132 lines.
  Full generated `testdata.js` corpus dropped from 383,808,060 bytes /
  6,975,910 lines to 365,412,548 bytes / 6,768,295 lines. Full compiler-mode
  simulation guard still passes 469/469 with no compact oracle failures.

Generated command-queue interning, 2026-04-30:
  Compact command queue helpers now go through the same function interner as
  match/apply helpers, with canonical helper names threaded into deterministic
  and random rule dispatch. This removes duplicated command-output plumbing
  without changing command semantics.
  Focused giant-case results:
  - case 223 `vertebrae`: 47,814,855 bytes / 856,731 lines to
    47,055,143 bytes / 831,204 lines.
  - case 403 `Crate Assembler`: 6,598,468 bytes / 147,132 lines to
    6,580,389 bytes / 146,534 lines.
  Full generated `testdata.js` corpus dropped from 365,412,548 bytes /
  6,768,295 lines to 360,577,475 bytes / 6,604,937 lines. Full compiler-mode
  simulation guard still passes 469/469 with no compact oracle failures.

Full-corpus post-cleanup guard, 2026-04-30:
  `make compact_turn_codegen_simulation_tests COMPILED_RULES_BUILD_JOBS=4`
  passes after the generated-source reductions with
  `cpp_simulation_tests_direct passed=469 failed=0 total=469` and
  `compact_turn_oracle_failures=0`.

Runtime/reporting cleanup, 2026-04-30:
  Renamed runtime generated-backend support metadata from fallback-specific
  naming to neutral `statusReason`, and added solver JSON/profile aliases for
  `compact_turn_unhandled` while keeping `compact_turn_fallbacks` for existing
  scripts. `make compact_turn_codegen_bringup COMPILED_RULES_BUILD_JOBS=4`
  now reports `unhandled=0`.

Solver compact compiler profiling, 2026-04-30:
  Added explicit solver focus targets for compiler-mode compact turns:
  `make solver_focus_compact_codegen_compare` and
  `make solver_focus_compact_codegen_perf_report`. These build the specialized
  solver with `--compact-turn-mode=compiler` and run the solver with
  `--compact-node-storage`, so the comparison is interpreter solver versus the
  generated compact turn pipeline rather than the compact interpreter bridge.
  The benchmark comparison report now prints compact-turn native/bridge hit
  buckets plus runtime phase counters for setup, early rules, movement, late
  rules, win, canonicalize, and bridge materialization/copyback. A smoke
  profile against `src/tests/solver_smoke_tests` confirmed
  `compact_turn_native=1`, `bridge=0`, and `compact_turn_native_hit`.
  Generated compiler-mode compact turns now feed those native phase counters
  directly; probe turns used by `again` are charged to the caller's
  canonicalization bucket so the report does not double-count their inner
  phases.

Compact compiler optimization pass, 2026-04-30:
  Simple non-random, single-row compact rules now store matched start tiles in
  `Scratch::singleRowMatchScratch` instead of building nested match vectors.
  On `make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1`, the
  all-target median moved from the earlier compiler-mode compact profile of
  roughly `314ms -> 500ms` to `304ms -> 325ms`. The native compact subset is
  now printed explicitly; the fresh run reports `targets=45`,
  `elapsed_ms=298.0->321.0`, `step_ms=157.3->228.2`, and median generated
  compact early-rule time `135.497ms`. This is still slower than the
  interpreter, but the first measured bottleneck was cut substantially.

Compact rule scan prefilter pass, 2026-04-30:
  Generated compact rules now emit row/column/board mask prefilters backed by
  `Scratch` mask caches, and generated turns avoid copying the start board
  unless rollback/probe semantics need it (`rigid`, `cancel`, `restart`,
  `require_player_movement`, or `again`). On
  `make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1`, the
  latest all-target median is `elapsed_ms=304.0->250.0`, `step_ms=159.9->105.0`;
  the native compact subset is `elapsed_ms=298.0->240.0`,
  `step_ms=152.6->101.6`, with median generated early-rule time `25.011ms`.
  Full compiler-mode simulation still passes 469/469 with zero compact oracle
  failures.

Selective compact mask-cache preparation, 2026-04-30:
  Generated compact sources now compute which scratch mask-cache families their
  emitted rules can actually read, then prepare only those object/movement
  row/column/board caches. Dirty flags stay conservative for any later
  interpreter boundary, but compiler-mode turns no longer rebuild unused mask
  families up front. On
  `make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1`, the
  all-target median is `elapsed_ms=304.0->254.0`, `step_ms=160.7->102.1`;
  the native compact subset is `elapsed_ms=298.0->243.0`,
  `step_ms=153.1->97.6`, with median setup time `17.835ms` and early-rule time
  `25.150ms`. Full compiler-mode simulation still passes 469/469 with zero
  compact oracle failures.

Compact-turn-only line-budget fix, 2026-04-30:
  The generated-line budget probe now honors `--compact-turn-only`. Before
  this, solver-focus compact compiler builds could skip sources by measuring
  full rulegroup-plus-compact output even though the actual binary only emits
  compact turn kernels. The focus corpus now emits all 34 input sources under
  the 200k-line/source budget, and
  `make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1` reports
  `compiled_usage: compact_turn_native=50` with no not-attached bucket. The
  latest like-for-like benchmark is all-target native compact:
  `elapsed_ms=304.0->254.0`, `step_ms=162.9->102.8`, with median generated
  early-rule time `32.059ms`.

Solver heuristic scratch reuse, 2026-04-30:
  The compact solver win-condition heuristic now reuses the existing
  `HeuristicScratch` distance-field buffers instead of allocating fresh vectors
  on each state evaluation. This keeps the compact solver path aligned with
  the interpreter heuristic's allocation discipline. A fresh
  `make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1` run
  reports all 50 focus targets on native compact kernels with
  `elapsed_ms=303.0->257.0`, `step_ms=162.3->106.3`, median setup time
  `17.745ms`, median early-rule time `32.360ms`, and median heuristic time
  `8.4ms->11.6ms`.

Solver key hashing cleanup, 2026-04-30:
  Solver state-key construction now packs RNG byte state into 64-bit words via
  a shared `appendStateKeyBytes` helper instead of mixing one byte at a time in
  the compact path. This also removes the duplicated byte-packing loop from the
  full-state key helper. A fresh one-run focus report shows the hash bucket no
  longer favoring the interpreter path (`hash_ms=32.8->31.2`) and the compiled
  compact median at `elapsed_ms=262.0->194.0`; treat the elapsed number as a
  noisy one-run benchmark, but the key-building cleanup is structurally better.

Rejected anchored-row scan experiment, 2026-04-30:
  An object-cell-index anchored row-scan prototype was measured and backed out
  rather than committed. Even after collapsing the generated scanner into a
  shared helper, the `paint everything everywhere` probe grew from roughly
  122k generated lines to 158k, and the focus benchmark moved step time in the
  wrong direction on the dense-scan regressions. Do not revive this path unless
  it can be proven as a generic scratch index with clear net wins and no
  source-size tax.

Again-probe telemetry, 2026-04-30:
  A Drain-mode shortcut that blindly scheduled `again` ticks without probing
  was measured and rejected. It made the focus benchmark faster, but full
  compact codegen simulation parity dropped to `447/469`: no-change `again`
  probes are semantic and must be rolled back rather than committed as real
  ticks. The committed follow-up splits `compact_turn_again_probe_calls` and
  `compact_turn_again_probe_ns` out of the runtime counters and benchmark
  detail report so this work is visible without mislabeling it as canonicalize
  time.

  One-run focus result after instrumentation:
  - Interpreted vs compiler-mode compact: `elapsed_ms=265.0->194.0`
    (`0.732x`), `step_ms=167.7->113.7` (`0.678x`).
  - Median `again_probe_ms` is zero because many focus targets never schedule
    `again`, but the detailed table now surfaces the real hotspots:
    `paint everything everywhere.txt#29` spends `198.1ms` in probes,
    `#11` spends `183.1ms`, `#25` spends `178.0ms`, and `#27` spends
    `171.6ms`.
  - Next safe optimization here is not "skip probes"; it is to reduce the cost
    of executing a semantically rollbackable tick, or to make generated rule
    scanning cheaper inside both normal turns and probes.

Executable selected-pass target:
  make compact_turn_codegen_selected_tests
  cases: COMPACT_TURN_CODEGEN_SELECTED_CASES in Makefile

Current next frontier:
  `testdata.js` compiler-mode correctness is green both per-case and as a
  combined generated corpus. The solver focus benchmark now shows the compiled
  compact pipeline winning overall median elapsed time, but not yet at the
  long-term 0.5x goal.

  The next architectural cleanup is to move the interpreter onto the same
  cell-major `PersistentLevelState::board.objects` authority as compiler-mode
  compact turns. See
  `docs/superpowers/plans/2026-04-30-compact-interpreter-board-migration.md`.

  After that migration, the immediate compiler/search optimization worklist is:
  1. Reduce `compact_turn_setup_ms`, currently dominated by rebuilding
     scratch row/column/board masks from arbitrary solver materializations.
  2. Optimize dense generated row scans now that all solver-focus targets are
     attached to compiler-mode compact kernels. The current slowest step
     regressions are `paint everything everywhere`, `Vexatious Match 3`,
     `the_saga_of_the_candy_scroll`, and `karamell`.
  3. Investigate graph-side costs that became more visible after clone/turn
     improvements, especially hash and heuristic time on compact node storage.
  4. Keep the benchmark suite oriented around like-for-like comparisons:
     interpreter solver baseline vs compiler-mode compact solver on the same
     focus targets, with native/not-attached buckets called out explicitly.
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
make compact_turn_codegen_coverage
make compact_turn_codegen_simulation_tests COMPILED_RULES_BUILD_JOBS=4
make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1
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
