# Generator And Shared Runtime Optimization Plan

## Current State
- The generator cleanup pass is complete: solver statuses are typed, solver metadata is cached per game, dedupe eviction is deterministic, top-K maintenance avoids full resorting per solve, expanded solver sessions are released, and the shared wincondition heuristic uses distance fields.
- Generator JSON now includes aggregate solver totals, and a fixed-seed preset benchmark script/Makefile target exists.
- `rule_plan_v1` already exists in both JS and native emitted IR.
- Rule-plan parity tooling already exists: `make rule_plan_parity_tests` and the `puzzlescript_rule_plan_parity` CTest both run `src/tests/run_rule_plan_parity.js` against native `--emit-ir-json`.
- C++ lowering already emits row object masks, row movement masks, `ruleMovementMask`, ellipsis counts, anchor metadata, and replacement metadata.
- C++ runtime already uses movement-aware rule skipping, anchored row scans with a density cutoff, deterministic sort/unique for anchored candidates, and ellipsis line prefilters.
- JS runtime already uses row object/movement mask prefilters and has basic live `rulePlanMetadata`, but it does not yet consume the full rule-plan metadata surface.

## Principles
- Keep JS/C++ parity as the hard gate. Prefer emitting or comparing metadata first, then consuming it in runtime code.
- Optimize work avoided before micro-tuning: fewer candidate cells, fewer pattern tests, fewer row scans, fewer replacement branches, fewer solver allocations.
- Measure generator changes with fixed seeds and fixed presets so speedups are not confused with different sample streams.
- Preserve PuzzleScript semantics and diagnostics. Any normalization that could change warnings/errors stays behind explicit parity checks.

## Generator Benchmarks
- Add a repeatable generator benchmark target over the current presets:
  - `src/tests/generator_presets/sokoban_room_scatter.gen`
  - `src/tests/generator_presets/sokoban_transform_pairs.gen`
- Use fixed game, fixed seed, fixed sample count, fixed jobs, and fixed solver timeout.
- Record:
  - samples attempted/sec
  - valid generation/sec
  - dedupe rate
  - solve rate
  - solver expanded/generated/unique/duplicate totals
  - timeout/exhausted counts
  - top score and top solution length distribution
- Aggregate solver totals are emitted in `solver_totals`.
- Keep a small smoke benchmark suitable for local iteration and a longer benchmark for before/after claims.

## Rule Plan Contract
- Treat `rule_plan_v1` as a versioned contract. Any structural change should either remain backward-compatible or bump the version.
- Keep rule-plan parity part of the normal native/CMake test surface, not only a Makefile target.
- Keep canonical serialization stable: sorted ids, deterministic group/rule order after known impossible-rule filtering, and minimal optional fields.
- Add targeted fixtures for:
  - property and any-object anchors
  - movement-only rules
  - single and double ellipsis
  - random entity and random direction replacements
  - rigid movement replacements
  - rules with commands/audio

## Shared Runtime Work
1. Consume existing `rule_plan_v1` metadata in JS where it is currently descriptive only.
2. Port C++ anchored row scan behavior to JS:
   - concrete object anchors
   - property/any-object anchors
   - anchor density cutoff
   - deterministic sort/unique only when needed
3. Move ellipsis plan data out of ad hoc runtime recomputation:
   - concrete cell counts
   - ellipsis positions
   - min concrete suffix counts
   - line-level object/movement prefilters
4. Use replacement metadata in both runtimes:
   - simple direct clear/set masks
   - touches objects/movements/random/rigid/audio
   - random choice lists
   - layer clear masks
5. Add a simple deterministic row-rule execution path:
   - one row
   - no random
   - no rigid
   - no ellipsis
   - no commands/debug side effects
   - direct replacement masks only
6. Consider stride-specialized bit operations only after the metadata-driven paths are stable:
   - JS generated code for `STRIDE_OBJ` / `STRIDE_MOV` 1 or 2
   - C++ templated kernels selected by object/movement word width

## Differential Debugging
- Add a mode that reports the first rule-plan/runtime divergence with:
  - rule id, group id, row index, and source line
  - candidate starts and selected random candidate
  - board object and movement masks
  - RNG state
  - command queue and audio events
- Make artifacts easy to diff: source, JS IR, native IR, canonical rule plan, and first runtime trace mismatch.

## Test Gates
- Always keep these gates available and passing before claiming runtime speedups:
  - `make simulation_tests_cpp`
  - `make compilation_tests_cpp`
  - `make simulation_tests_cpp_js_parity`
  - `make rule_plan_parity_tests`
- For generator changes, also run:
  - `ctest --test-dir build -R puzzlescript_generator_smoke --output-on-failure`
  - `make generator_benchmark` or a smaller smoke variant with reduced samples/runs.

## Near-Term Order
1. Done: add generator aggregate solver metrics and fixed-seed benchmark scripts.
2. Done: wire rule-plan parity into CMake/CTest.
3. Port C++ anchor selection to JS using existing rule-plan fields.
4. Move ellipsis scratch/precompute data into explicit rule-plan fields and consume them in both runtimes.
5. Add simple direct replacement execution in C++ first, guarded by parity tests.
6. Mirror the simple deterministic row-rule path in JS.
