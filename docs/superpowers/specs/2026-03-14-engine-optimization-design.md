# PuzzleScript Engine Optimization Pass

## Goal

Reduce total execution time of the PuzzleScript engine across both compilation and runtime paths. The test suite (`node src/tests/run_tests_node.js`) serves as the benchmark — it exercises compilation, rule application, movement resolution, undo, and restart across 900+ tests.

Runtime performance is slightly more important than compile time, but both matter since games compile on load.

## Constraints

- All existing tests must continue to pass after every change.
- No restrictions on optimization approach — algorithmic changes, data structure changes, generated code changes, caching, rewriting hot functions are all acceptable.

## Approach: Profile-Guided Structural Optimization

Use V8 profiling to identify where CPU time is spent, then do targeted structural analysis of hot areas to find the best fix.

### Phase 1 — Baseline & Profile

1. Run the test suite without profiling to establish a baseline wall-clock time (average of 3 runs).
2. Run with `node --prof src/tests/run_tests_node.js` to generate a V8 profiling tick file.
3. Process with `node --prof-process` to produce a human-readable breakdown.
4. Identify the top hotspots by CPU ticks — both built-in functions and application code.

### Phase 2 — Analyze & Optimize (Iterative)

For each hotspot, in order of impact:

1. **Read** the hot code path and understand what it does.
2. **Diagnose** the root cause: unnecessary allocations, redundant computation, poor branching, cache-unfriendly access patterns, unnecessary function calls, etc.
3. **Implement** the optimization.
4. **Verify** all tests still pass.
5. **Measure** the improvement (average of 3 runs, same as baseline).

Each optimization is atomic — verify correctness before moving to the next.

**Stopping criteria:** Continue until the top remaining hotspot accounts for less than 5% of total ticks, or until three consecutive optimization attempts yield less than 2% cumulative improvement, whichever comes first.

### Phase 3 — Validate

1. Run final V8 profiling pass to confirm hotspots have shifted.
2. Compare total test suite time (average of 3 runs): before vs after.
3. Confirm all tests pass.

## Key Engine Files

| File | Role | Likely Hot Paths |
|------|------|-----------------|
| `src/js/engine.js` | Game loop, rule application, movement resolution | `processInput()`, `applyRules()`, `applyRuleGroup()`, generated `resolveMovements` (via `generate_resolveMovements()`) |
| `src/js/compiler.js` | Parsing, compilation, code generation | `compile()`, `rulesToArray()`, `rulesToMask()`, `collapseRules()` |
| `src/js/parser.js` | Tokenization infrastructure | Token processing loops |
| `src/js/level.js` | Level state with bitvectors | State cloning, backup/restore |
| `src/js/bitvec.js` | Bitwise operations | `ior()`, `iand()`, `iclear()`, `clone()` |

Note: The engine uses runtime code generation for hot paths (e.g., `generate_resolveMovements()` produces the actual movement resolution function). Optimizing these paths requires modifying the generator, not the generated output.

## Success Criteria

- At least 10% reduction in total test suite execution time.
- All 900+ tests continue to pass.
- No functional regressions.
