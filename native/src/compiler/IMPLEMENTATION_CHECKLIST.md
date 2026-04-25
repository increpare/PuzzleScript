# Whole-Game Compilation Implementation Checklist

This checklist turns `PLAN.md` into day-to-day engineering work. It should stay
specific enough that a person or agent can pick the next unchecked item, make a
small commit, and know how to prove it.

The north star is a generated per-game execution path: conceptually,
`tick(state, input) -> state`. The interpreter remains the reference
implementation until the generated path has earned every piece of behavior.

## How To Use This File

- Treat each checkbox as a commit-sized or review-sized task unless it says
  otherwise.
- Keep the generated path boring and truthful: handle only the cases it really
  implements, and fall back to the interpreter for everything else.
- After each behavior-moving change, run the smallest relevant parity test first,
  then a broader one before committing.
- Update this checklist when a task is completed, split, invalidated, or made
  more precise by new evidence.

Status markers:

- `[x]` done and validated.
- `[ ]` not started.
- `[?]` needs a design decision or measurement before implementation.

## Always-On Guardrails

- [ ] Preserve `interpreterStep` and `interpreterTick` as the behavior oracle.
  Generated code may call them directly, or decline handling so dispatch falls
  through to them.

- [ ] Preserve debug behavior. If a `PS_DEBUG_*` mode would lose information on
  the generated path, dispatch must choose the interpreter until trace parity is
  deliberately implemented.

- [ ] Preserve solver/generator semantics for `RuntimeStepOptions`:
  `playableUndo=false`, `emitAudio=false`, no accidental auto-settling of
  `pendingAgain`, and no hidden undo snapshot growth.

- [ ] Keep one-game specialization ergonomic. Production iteration should use a
  single generated game source where practical; corpus-wide generation remains a
  proving rig.

- [ ] Keep fallback cheap enough to be useful. A generated tick function should
  be allowed to bail out without rebuilding a whole second world around the
  interpreter.

- [ ] Prefer explicit coverage and counters to confidence by inspection. If a
  milestone changes behavior or dispatch shape, add a way to observe whether it
  is being used.

## Current Groundwork

- [x] Compiled rule-group kernels exist and attach by source hash.

- [x] `compile-rules --coverage-json` reports aggregate and per-source coverage.

- [x] `make compiled_rules_simulation_suite_coverage` writes a reusable
  simulation-suite coverage JSON file.

- [x] With `COMPILED_RULES_MAX_ROWS=99`, compiled rules cover 452/452 unique
  simulation-suite source texts.

- [x] Solver and generator can opt into generated rule code with
  `SPECIALIZE=true`.

- [x] Public `step` / `tick` dispatch can try a `CompiledTickBackend`.

- [x] `interpreterStep` / `interpreterTick` name the reference engine path.

- [x] Generated sources export a compiled tick backend that currently delegates
  to `interpreterStep` / `interpreterTick`.

## Baseline Measurements

- [x] Record a clean no-specialization baseline for solver smoke.

  Command:

  ```sh
  /usr/bin/time -p make solver_smoke_tests SPECIALIZE=false
  ```

  Done means: elapsed time and pass count are noted in the relevant commit or
  PR description.

- [x] Record a clean specialization baseline for solver smoke.

  Command:

  ```sh
  /usr/bin/time -p make solver_smoke_tests SPECIALIZE=true
  ```

  Done means: elapsed time, generated source reuse behavior, and pass count are
  noted.

- [x] Record a generator smoke baseline.

  Command:

  ```sh
  /usr/bin/time -p make generator_smoke_tests SPECIALIZE=true
  ```

  Done means: elapsed time and pass/fail behavior are noted.

- [x] Record compiled-rule simulation coverage at both the fast default and the
  high-row proving setting.

  Commands:

  ```sh
  make compiled_rules_simulation_suite_coverage COMPILED_RULES_MAX_ROWS=1
  make compiled_rules_simulation_suite_coverage COMPILED_RULES_MAX_ROWS=99
  ```

  Done means: coverage counts are noted, including remaining miss reasons at
  `COMPILED_RULES_MAX_ROWS=1`.

- [x] Decide whether to add a small checked-in benchmark notes file, or keep
  measurement notes in commit messages and PR descriptions only.

## Generated Tick Observability

- [x] Add runtime counters for compiled tick attempts, hits, and fallbacks.

  Intent: make it obvious whether a solver/generator run is actually entering
  generated tick dispatch.

  Acceptance criteria:

  - Counters are available through the existing runtime-counter snapshot path.
  - Existing counter callers remain source-compatible.
  - The profile output can distinguish compiled-rule hits from compiled-tick
    hits.

  Validation:

  ```sh
  make build
  build/native/puzzlescript_cpp test simulation-corpus src/tests/resources/testdata.js --case-index 1 --repeat 3 --profile-timers --jobs 1
  make solver_smoke_tests SPECIALIZE=true
  ```

- [x] Add a focused generated-tick dispatch smoke test.

  Intent: prove the generated tick backend is found by source hash and called,
  even while it still delegates to the interpreter.

  Acceptance criteria:

  - Test fails if `ps_compiled_tick_find_backend` is not linked or not found.
  - Test proves both `step` and `tick` dispatch can be attempted.
  - Test does not depend on timing.

  Suggested shape:

  - Generate C++ for `src/demo/sokoban_basic.txt`.
  - Link a specialized test binary or use an existing specialized smoke path.
  - Assert compiled tick attempt/hit counters changed.

- [ ] Add a debug-gating smoke test.

  Intent: ensure generated tick dispatch does not bypass debug traces by
  accident.

  Acceptance criteria:

  - At least one `PS_DEBUG_RULES`, `PS_DEBUG_MOVES`, or `PS_DEBUG_AGAIN` run
    proves dispatch chooses the interpreter.
  - Counter output or test plumbing makes that choice visible.

## Generated Tick Owns Rule-Group Dispatch

- [x] Move early rule-group loop selection into generated tick code.

  Intent: generated tick should iterate the game-specific early rule groups and
  call compiled group kernels directly, instead of entering the generic
  `applyRuleGroups` loop first.

  Implementation notes:

  - Start with games whose early groups are all covered by compiled rules.
  - Preserve loop-point behavior exactly.
  - Preserve banned rigid-group behavior by falling back if unsupported.
  - Keep late rules on the interpreter path for the first commit if that makes
    the change smaller.

  Acceptance criteria:

  - Generated tick handles at least one simple non-rigid game without calling
    generic rule-group dispatch.
  - It falls back for games or states outside the supported slice.
  - Simulation parity remains green.

  Validation:

  ```sh
  make build
  make simulation_tests_cpp
  make solver_smoke_tests SPECIALIZE=true
  ```

- [ ] Move late rule-group loop selection into generated tick code.

  Intent: generated tick owns both early and late rule traversal for supported
  games.

  Acceptance criteria:

  - Late rule groups call compiled kernels directly.
  - Late loop points behave identically to the interpreter.
  - Unsupported late groups force a clean fallback.

  Validation:

  ```sh
  make simulation_tests_cpp
  make solver_parity_smoke SPECIALIZE=true
  ```

- [x] Generate fixed loop-point tables.

  Intent: avoid dynamic loop-point lookup for generated rule traversal.

  Acceptance criteria:

  - Generated C++ contains static per-game loop-point decisions.
  - Behavior matches `lookupLoopPoint`.
  - Infinite-loop guard behavior remains capped at the interpreter limit.

- [x] Generate direct group coverage predicates.

  Intent: a generated tick function should know at compile time whether every
  group it needs is compiled.

  Acceptance criteria:

  - No runtime scan is needed to decide rule-loop eligibility.
  - The generated backend declines handling if any required group is missing.
  - Coverage JSON and generated-code eligibility agree.

- [x] Remove redundant rule backend lookup inside generated tick.

  Intent: once generated tick owns rule traversal, it should call local generated
  functions directly rather than re-finding the same backend.

  Acceptance criteria:

  - Generated tick source compiles as a one-game unit.
  - Direct calls do not break sharded/corpus generation.
  - `CompiledRulesBackend` remains available for interpreter fallback.

## Turn Skeleton Specialization

- [x] Extract an interpreter-readable turn skeleton.

  Intent: identify the exact ordered phases currently inside `executeTurn` so
  the generated implementation can mirror them without copying mystery.

  Acceptance criteria:

  - The phase list is documented in comments or helper names.
  - No behavior changes are made in the extraction commit.
  - The generated checklist can refer to named phases.

  Current phases to preserve:

  - Clear audio state.
  - Prepare create/destroy sound masks.
  - Create or reference turn-start undo snapshot.
  - Handle `require_player_movement` starting positions.
  - Clear movement state.
  - Seed player movement.
  - Apply early rules.
  - Resolve movements.
  - Retry rigid pass if needed.
  - Apply late rules.
  - Compute modification status.
  - Process cancel / dontModify / output commands.
  - Process restart / checkpoint / win / again.
  - Fill `ps_step_result`.
  - Rebuild masks before returning.

- [ ] Generate a supported-game predicate for the turn skeleton.

  Intent: generated tick should handle only games whose required skeleton pieces
  have been implemented.

  Acceptance criteria:

  - Predicate is generated from game features, not handwritten per fixture.
  - Unsupported commands or metadata choose fallback.
  - The reason for fallback can be observed in debug/profile mode.

- [ ] Implement generated no-input `tick` for the simplest supported games.

  Intent: make `tick(session, options)` do real generated work for games without
  player-input complications.

  Acceptance criteria:

  - Generated tick executes early rules, movement resolution by fallback or
    generated helper, late rules, and result assembly for a narrow slice.
  - `step(session, PS_INPUT_TICK, options)` and `tick(session, options)` agree.
  - Unsupported cases delegate to `interpreterTick`.

- [ ] Implement generated directional/action `step` entry.

  Intent: move real input processing into generated code after no-input tick is
  stable.

  Acceptance criteria:

  - Title-screen and message-mode behavior remains interpreter-handled until
    explicitly specialized.
  - Directional movement seeding matches interpreter behavior.
  - Action input works for supported games or falls back cleanly.

## Command Handling

- [ ] Generate command queue shape for supported games.

  Intent: avoid dynamic string command handling where the command set is known.

  Acceptance criteria:

  - Generated code can represent `again`, `cancel`, `checkpoint`, `restart`,
    `win`, `message`, and output commands without string scans on the hot path.
  - Interpreter fallback can still consume the existing `CommandState`.
  - Unknown or unsupported command forms choose fallback.

- [ ] Specialize `cancel`.

  Acceptance criteria:

  - State restoration matches interpreter behavior.
  - Audio behavior respects `emitAudio`.
  - `dontModify` result behavior is identical.

- [ ] Specialize `restart`.

  Acceptance criteria:

  - Restart target restoration matches the interpreter.
  - `run_rules_on_level_start` behavior is preserved or forces fallback.
  - `recordRestartUndo` behavior is preserved.

- [ ] Specialize `checkpoint`.

  Acceptance criteria:

  - Restart snapshot contents match interpreter behavior.
  - Checkpoint after win is handled identically.

- [ ] Specialize `win` and explicit win command.

  Acceptance criteria:

  - Explicit `win` command and evaluated win conditions produce identical
    `won` / `transitioned` results.
  - Level transition behavior remains correct for final and non-final levels.

- [ ] Specialize `again` scheduling.

  Acceptance criteria:

  - `pendingAgain` is set only when interpreter would set it.
  - The would-again-change probe either runs generated code safely or delegates
    to the interpreter.
  - Solver/generator external settling remains unchanged.

- [ ] Specialize message and output commands.

  Acceptance criteria:

  - Message text, text mode, and output command side effects match interpreter
    behavior.
  - Unsupported output behavior falls back.

## Movement Specialization

- [ ] Add movement-resolution coverage classification.

  Intent: know which games can use generated movement before writing a large
  mover.

  Acceptance criteria:

  - Coverage output can classify movement features: layer count, aggregate
    player mask, rigid groups, movement sounds, failure sounds, and metadata
    affecting movement.
  - Classification is available per source.

- [ ] Generate fixed layer loops for movement masks.

  Intent: replace dynamic layer-count loops in the hot movement path for
  supported games.

  Acceptance criteria:

  - Generated code uses compile-time layer count and movement-word count.
  - Behavior matches interpreter movement resolution for non-rigid fixtures.
  - Unsupported aggregate/rigid cases fall back.

- [ ] Specialize player movement seeding.

  Acceptance criteria:

  - Direction masks match interpreter constants.
  - Aggregate player movement is either correctly handled or falls back.
  - `require_player_movement` behavior remains correct.

- [ ] Specialize movement success/failure sounds.

  Acceptance criteria:

  - `emitAudio=false` keeps all audio work out of solver/generator hot paths.
  - `emitAudio=true` matches event seeds and ordering.
  - Debug audio paths either match or force interpreter dispatch.

- [ ] Specialize rigid retry bookkeeping.

  Acceptance criteria:

  - Rigid retry limit matches interpreter behavior.
  - Banned groups are updated identically.
  - Existing rigid simulation cases pass under specialized builds.

- [ ] Expand movement specialization to aggregate player cases.

  Acceptance criteria:

  - Aggregate movement remains atomic across player layers.
  - Mixed aggregate/non-aggregate games pass parity.

## Win Conditions And Level Transitions

- [ ] Generate fixed win-condition checks.

  Acceptance criteria:

  - `some`, `no`, `all`, and aggregate combinations match interpreter behavior.
  - Masks are generated or referenced without dynamic name lookup.
  - Empty or unusual levels match interpreter results.

- [ ] Specialize level advancement.

  Acceptance criteria:

  - `transitioned` and `won` fields match interpreter behavior.
  - Message levels and title/text mode are either handled or fall back.
  - Prepared-session state remains consistent after transition.

- [ ] Specialize `run_rules_on_level_start`.

  Acceptance criteria:

  - Games without the metadata pay no generated overhead.
  - Games with the metadata run the correct rule path after load/restart.
  - Undo/audio options are preserved.

## Game Constants And Generated Data

- [ ] Emit per-game constants for strides, dimensions, object counts, layer
  counts, and word counts.

  Acceptance criteria:

  - Generated code no longer loads these values through `Game` for supported
    hot paths.
  - Constants match the loaded runtime game.

- [ ] Emit mask tables used by generated tick.

  Acceptance criteria:

  - Rule masks, layer masks, player masks, win masks, and sound masks needed by
    generated code are available as generated constants or stable references.
  - No generated constant duplicates large data without a measured reason.

- [ ] Emit compact rule metadata for generated traversal.

  Acceptance criteria:

  - Group order, late/early partitioning, loop points, random flags, rigid
    mapping, and command capability are represented without runtime scanning.

- [ ] Emit level metadata needed by generated transition logic.

  Acceptance criteria:

  - Level dimensions, message flags, target level indices, and restart data are
    available to generated code.
  - Loaded-level seeds remain interpreter-compatible.

## Specialized State Layout

- [?] Decide the first specialized state boundary.

  Options:

  - Keep using `Session` and specialize only code first.
  - Introduce a compact generated state alongside `Session`.
  - Introduce a compact state only for solver/generator, leaving player/API
    paths on `Session`.

  Recommended default: keep using `Session` until generated tick has meaningful
  behavior, then introduce compact solver/generator state.

- [ ] Add state-layout notes once the boundary is chosen.

  Acceptance criteria:

  - The chosen state owns live objects, live movements, random state, pending
    again, current level, and restart/checkpoint data explicitly.
  - Conversion to/from `Session` is defined for fallback and testing.

- [ ] Prototype compact solver/generator state for one simple game.

  Acceptance criteria:

  - State clone cost is lower than `Session` clone cost.
  - Hashing and serialization for tests remain available.
  - Interpreter fallback remains possible through a conversion path.

- [ ] Replace solver child cloning for supported games.

  Acceptance criteria:

  - Solver parity remains green.
  - Memory use and nodes/second improve on at least one benchmark.

## Smaller Tick-Game ABI

- [?] Decide whether the first compact ABI is C++-only or C-facing.

  Recommended default: C++-only until state layout stabilizes, then wrap in C.

- [ ] Define a compact tick result.

  Acceptance criteria:

  - Represents changed, won, transitioned, pending again, and optional audio.
  - Does not require heap allocation on solver/generator hot paths.

- [ ] Define compact input representation.

  Acceptance criteria:

  - Covers up, down, left, right, action, and no-input tick.
  - Maps exactly to existing `ps_input` values or has a lossless adapter.

- [ ] Add generated ABI smoke test.

  Acceptance criteria:

  - One generated game can be stepped without going through public `Session`
    dispatch.
  - Result and serialized final state match interpreter behavior.

## Solver Integration

- [ ] Add a solver flag or internal capability path for compiled tick.

  Intent: distinguish "compiled rules are linked" from "whole tick is actually
  handling solver steps".

  Acceptance criteria:

  - Solver can report compiled tick attempts/hits/fallbacks.
  - `SPECIALIZE=true` remains the user-facing Makefile entry.
  - Unsupported games still solve through the interpreter.

- [ ] Add solver parity target focused on compiled tick.

  Acceptance criteria:

  - Runs a small representative game set.
  - Fails if compiled tick silently falls back for every case when it should not.
  - Still verifies final solver results.

- [ ] Benchmark one-game solver hot loop.

  Acceptance criteria:

  - Compare `SPECIALIZE=false`, compiled rules only, and compiled tick handling.
  - Report build time separately from solve time.
  - Include at least one Sokoban-like and one rule-heavy fixture.

## Generator Integration

- [ ] Add generator capability reporting for compiled tick.

  Acceptance criteria:

  - Generator can report whether generated tick handled candidate steps.
  - Fallback behavior is visible in quiet logs or profiling output.

- [ ] Add generator parity/smoke target focused on compiled tick.

  Acceptance criteria:

  - Existing `generator_smoke_tests` remains green.
  - A compiled-tick-specific path proves generated dispatch is exercised.

- [ ] Benchmark generator candidate evaluation.

  Acceptance criteria:

  - Compare generated tick against interpreter path on the same preset and seed.
  - Separate generation/build time from candidate evaluation throughput.

## Build And Codegen Ergonomics

- [ ] Keep generated source reuse robust as tick code grows.

  Acceptance criteria:

  - Stamps include any options that change generated tick output.
  - Reuse never hides stale generated code after CLI changes.
  - `COMPILED_RULES_REUSE_SINGLE_CPP=false` still forces regeneration.

- [ ] Keep sharded corpus generation working.

  Acceptance criteria:

  - `--emit-cpp-dir` emits both rule and tick backend accessors per source.
  - `registry.cpp` links all generated rule and tick backends.
  - Corpus generation remains useful for coverage and smoke tests.

- [ ] Split generated code when compile time demands it.

  Acceptance criteria:

  - One-game generation remains simple.
  - Corpus generation can shard large tick helpers if needed.
  - Link time is measured before and after any split.

- [ ] Consider moving codegen out of `native/src/cli/main.cpp`.

  Intent: whole-game codegen will outgrow the CLI file.

  Acceptance criteria:

  - New module has a narrow interface: source/game/options in, generated C++
    out.
  - Behavior is unchanged in the extraction commit.
  - Tests and `compile-rules` output are unchanged except for intentional
    formatting.

## Coverage And Eligibility Reporting

- [x] Extend coverage JSON from compiled rules to compiled tick eligibility.

  Acceptance criteria:

  - Per source, report whether rule loops, commands, movement, win conditions,
    level transitions, and state layout are generated or fallback-only.
  - Aggregate counts are easy to read from a one-line Node or jq command.

- [x] Add human-readable miss reasons for compiled tick.

  Acceptance criteria:

  - Reasons are stable strings suitable for tracking over time.
  - Examples: `unsupported_command`, `movement_rigid`, `message_level`,
    `debug_trace`, `state_layout`.

- [x] Track "fully generated tick" count separately from "compiled rules"
  coverage.

  Acceptance criteria:

  - Coverage can say "rules compile" and "whole tick handles" independently.
  - The distinction is reflected in docs and CLI output.

## Correctness Test Matrix

- [ ] Keep the core simulation suite green after every behavior move.

  Command:

  ```sh
  make simulation_tests_cpp
  ```

- [ ] Keep solver smoke green after every generated dispatch change.

  Command:

  ```sh
  make solver_smoke_tests SPECIALIZE=true
  ```

- [ ] Keep solver determinism green after random or again changes.

  Command:

  ```sh
  make solver_determinism_tests SPECIALIZE=true
  ```

- [ ] Keep solver parity green after state, movement, command, or win changes.

  Command:

  ```sh
  make solver_parity_smoke SPECIALIZE=true
  ```

- [ ] Keep generator smoke green after solver-facing step changes.

  Command:

  ```sh
  make generator_smoke_tests SPECIALIZE=true
  ```

- [ ] Add focused fixtures as bugs are found.

  Acceptance criteria:

  - Every generated-path correctness bug gets a minimal regression fixture.
  - The fixture fails before the fix and passes after it.

## Performance Test Matrix

- [ ] Track generated-code size for one-game builds.

  Acceptance criteria:

  - Record generated `.cpp` size and object size for a simple game and a
    rule-heavy game.
  - Avoid optimizing for corpus-wide code size unless it affects iteration.

- [ ] Track compile and link time separately.

  Acceptance criteria:

  - Use `/usr/bin/time -p`.
  - Note whether generated sources were reused.
  - Note whether Ninja or another CMake generator was used.

- [ ] Track solver throughput.

  Acceptance criteria:

  - Measure nodes/second or equivalent existing solver output.
  - Compare against non-specialized and compiled-rule-only baselines.

- [ ] Track generator throughput.

  Acceptance criteria:

  - Measure candidate evaluations or existing generator throughput output.
  - Use fixed seed/preset when possible.

## Documentation Tasks

- [ ] Update `PLAN.md` when an architecture decision changes.

- [ ] Update this checklist when a milestone is completed.

- [ ] Add a short user-facing note once compiled tick does real work.

  Acceptance criteria:

  - Explain how to enable it through `SPECIALIZE=true`.
  - Explain fallback behavior.
  - Avoid implying C/LLVM/assembly support before it exists.

- [ ] Document `COMPILED_RULES_MAX_ROWS` as a build-time/code-size knob.

  Acceptance criteria:

  - Docs make clear that high-row support exists, but the default is conservative
    for iteration.

## Open Decisions To Revisit

- [?] Should generated tick and compiled rules remain under the `compile-rules`
  command, or should the CLI grow a clearer `compile-game` command once whole
  tick generation is real?

- [?] Should coverage files use one combined schema or separate
  compiled-rules/compiled-tick schemas?

- [?] How much trace parity is enough before generated tick may run under
  `PS_DEBUG_*` flags?

- [?] When compact state exists, should interpreter fallback convert compact
  state to `Session`, or should unsupported cases be rejected before entering
  compact-state execution?

- [?] Which benchmark should be the "north-star" solver benchmark for this
  project?

## First Recommended Implementation Slice

This is the next practical sequence after this checklist exists:

1. Add compiled tick counters.
2. Add a dispatch smoke test proving generated tick is called.
3. Add compiled tick eligibility reporting to coverage JSON.
4. Generate early rule-loop ownership for the simplest fully covered games.
5. Run simulation, solver smoke, solver parity smoke, and generator smoke.
6. Record the first real compiled-tick throughput comparison.

This slice is deliberately narrow. It makes generated tick observable before it
tries to be clever, then moves one hot behavior into generated code with the
interpreter still close enough to catch us.
