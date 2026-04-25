# Strict Native Solver Optimization Execution Plan

## Summary

Optimize only the strict apples-to-apples solver metric: one `250ms` budget per playable level, `--jobs 1`, no multi-budget fallback headline numbers.

Agent prep is complete:
- Curie: no-undo/no-audio solver runtime step design.
- Euclid: compact state, no-allocation hash, flat visited table design.
- Nietzsche: pippable-level profiling harness and commit gates.

Current known baseline after fallback undo:
- `HEAD`: `104bd631`
- Strict portfolio: about `241/540`
- Strict weighted A*: measured about `261/540`
- Biggest timing buckets: `step`, `clone`, `hash`
- Most useful targets: pippable levels like `slidings.txt#11/#13/#15/#17`, `dropswap.txt#3/#12`, `coincounter.txt#7`, `cakemonsters.txt#38`.

## Task List And Commit Order

1. **Commit 1: Profiling Harness And Strict Baselines**
   - Add `mine_solver_near_threshold.js` and `run_solver_level_benchmark.js`.
   - Add make targets: `solver_mine_pippable`, `solver_benchmark_targets`, optional `solver_sample_target`.
   - Produce a pippable manifest from strict runs only.
   - Switch benchmark defaults to `weighted-astar` only if a fresh run confirms it beats `portfolio` under the same `250ms` budget.
   - Commit only harness/default changes that are benchmark-neutral or improve solved count.
   - Commit message must include strict solved count, target-suite count, and generated/sec.

2. **Commit 2: No-Allocation Hash**
   - Replace solver hashing path with a streaming/no-allocation dual hash.
   - Preserve current runtime hash semantics first; do not change search behavior.
   - Add a hash-equivalence test comparing old `hashSession128` and new no-allocation hash across smoke/replay states.
   - Commit only if `hash_ms` and target-suite wall time improve without solved-count regression.
   - Discard if the speedup is lost in noise or any hash parity risk appears.

3. **Commit 3: Solver Runtime Step Options**
   - Add internal C++ `RuntimeStepOptions` with defaults preserving existing playable behavior.
   - Solver uses `playableUndo=false` and `emitAudio=false`.
   - Preserve rule semantics: `again`, `win`, `cancel`, `restart`, `checkpoint`, messages, and level transitions.
   - Add solver-mode assertion/test that generated child states do not retain undo stack history.
   - Commit only if clone/step time improves and strict solved count does not regress.

4. **Commit 4: Flat Visited Table**
   - Replace `std::unordered_map<StateKey, depth>` in solver with an open-addressed flat table.
   - Add stale-pop skipping using node key/depth so improved shallower paths do not leave expensive obsolete frontier work.
   - Commit only if target-suite generated/sec improves by at least 2% and strict solved count is stable or better.

5. **Commit 5: Algorithmic Strict-Budget Tuning**
   - Compare strict schedules, not multi-budget fallback:
     - `weighted-astar`
     - `greedy`
     - `bfs`
     - `greedy 25ms + weighted-astar 225ms`
     - `greedy 50ms + weighted-astar 200ms`
     - priority variants: `depth + h`, `depth + 2h`, greedy with depth tie-break.
   - Use pippable target suite first, then full corpus.
   - Commit only schedule/priority changes that improve strict `250ms` solved count and do not meaningfully reduce throughput.

6. **Commit 6: Compact Solver State Store**
   - Replace retained full `Session` nodes with compact solver state storage.
   - Keep scratch sessions per search for materialization and stepping.
   - Persist live objects, movements, random state, pending-again/current-level flags, and checkpoint/restart state needed for correctness.
   - This is the highest-risk/highest-reward change; implement only after prior commits establish strong baselines.
   - Commit only after full correctness suite and strict benchmark pass.

## Benchmark And Discard Rules

- Every candidate patch gets:
  - `make build_solver`
  - `make solver_smoke_tests`
  - `make solver_determinism_tests`
  - `make solver_parity_smoke`
  - pippable target benchmark
  - full strict benchmark when the target benchmark improves

- Full metric command:
  `build/native/puzzlescript_solver src/tests/solver_tests --timeout-ms 250 --jobs 1 --strategy weighted-astar --no-solutions --quiet --json`

- Discard ruthlessly when:
  - strict solved count decreases,
  - target-suite median wall time regresses by more than 2%,
  - generated/sec regresses by more than 2% without a solved-count gain,
  - correctness/parity changes are not fully explained,
  - a change only helps long 10s levels and does not help pippable levels.

## Assumptions

- The multi-budget fallback commit stays undone and is not part of the benchmark.
- The first implementation priority is profilability and pippable-level improvement, not solving very long levels.
- Internal runtime helpers are allowed if public C API and `puzzlescript_cpp` behavior stay unchanged.
- Commits should be small, metric-bearing, and reversible.
