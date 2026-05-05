# Solver heuristics / optimizations TODO

## Performance optimizations

- [x] **1. cellHasBlockingObject recomputes constant `allowed` mask per call (lines 1032-1050)**
  Done: added `getAllowedMask(condition)` helper backed by a `Map` keyed by condition reference. Both `cellHasBlockingObject` and `cellHasStaticBlockingObject` now look up a precomputed `Int32Array(STRIDE_OBJ)` instead of rebuilding it per tile.

- [x] **2. Backing arrays should be typed**
  Done: `heuristicDistances` and `obstacleDistances` are now `Float64Array` (preserves `Infinity` semantics). `obstacleQueue` is `Int32Array`. `conditionDistances[i]` uses `Float64Array`.

- [x] **3. MinHeap swap allocates an array per swap**
  Done: rewrote `push`/`pop` to use the standard "hole" sift technique — values are written along the path and the moving item is placed once at the end. No destructuring swaps, no per-call `less` function calls (comparison inlined).

- [x] **4. Heuristic dispatch is a string switch on every call**
  Done: `HEURISTIC_FUNCTIONS` map resolved once at construction; `heuristic()` calls the captured `selectedHeuristic` reference.

- [ ] **5. Tiebreak heuristics duplicate base work**
  Skipped: invasive refactor; would require restructuring every tiebreak heuristic to share an "all-on context" object. Worth doing if profiling points here.

- [x] **6. allOnDeadlockHeuristic and allOnMinMatchingHeuristic re-resolve the condition**
  Done for `allOnDeadlockHeuristic`: now computes `unsatisfied`/`targets` once and calls a shared `allOnMatchingScore(unsatisfied, targets)` helper. `allOnMinMatchingHeuristic` left as-is (still calls both `winconditionDistanceHeuristic` and `allOnMatchingHeuristic` because it needs both scores anyway).

- [x] **7. noOnPlayerHeuristic calls noOnEscapeHeuristic then re-computes collectOverlapTiles**
  Done: extracted `computeNoOnEscapeScore(condition, offenders)` so the offender list is collected once.

- [ ] **8. Per-node performance.now() overhead**
  Skipped: would distort the timing breakdown the user depends on. Could gate behind an env flag if profiling itself becomes a bottleneck.

- [x] **9. Date.now() checked twice per action**
  Done: removed the inner per-action `Date.now()` checks in both `runMode` and `runAdaptivePortfolio`. The outer per-node check still bounds total wall time.

- [x] **10. Hoist constant arrays inside per-call functions**
  Done: `pushAccessPenalty`'s `directions` array is now a module-level `PUSH_ACCESS_DIRECTIONS` constant.

- [x] **11. minAssignmentDistance memo key is a string**
  Done: numeric key `(sourceIndex << targets.length) | usedMask` (sources ≤ 10, targets ≤ 20 by gating, fits comfortably in 31 bits).

- [x] **12. discoverGames does extra statSync per file**
  Done: switched to `readdirSync(dir, { withFileTypes: true })` and removed the redundant `fs.existsSync` in `runCorpus`.

- [x] **13. obstacleDistanceField branches per cell on blockerMask ternary**
  Done: hoisted `isBlocked` closure (one of two functions chosen by `blockerMask` at the top of the BFS). Also hoisted `level.height`/`level.width` reads.

## Correctness / suspicious behavior — left as observations (need design call)

- [ ] **A. some-on heuristics return only 16 when targetCount === 0**
  Suggest bumping to a much larger penalty (e.g., 64 or `Infinity`) so the search doesn't think it's nearly done. Want to confirm intent before changing — the current value may be deliberate to keep heuristic bounded.

- [ ] **B. winconditionDistanceHeuristic does redundant matchesMask after distance field**
  `matchesMask(filter2, ...)` could be replaced with `heuristicDistances[tile] === 0`. Easy win, low risk; left because tweaking the wincondition heuristic affects everything downstream.

- [ ] **C. winconditionDistanceHeuristic is inadmissible**
  Documentation suggestion only.

- [ ] **D. bestManhattan returns 64 when targets is empty**
  Same family of issue as A — may want a larger sentinel.

- [ ] **E. noOnPlayerHeuristic adds player-distance directly (not divided)**
  Inconsistent with other tiebreak heuristics; verify this was deliberate.

- [ ] **F. flagsForHash omits some snapshot fields (`hasUsedCheckpoint`, `titleSelected`, etc.)**
  Potential unsound dedup. Needs verification that these flags don't affect reachability.

- [ ] **G. installZobristHash silently no-ops if level not loaded yet**
  Not a bug given current call ordering, but fragile — could add an explicit assertion.

- [ ] **H. priorityForPortfolioMode doesn't use astarWeight option**
  Add CLI plumbing if you ever want to tune portfolio weights.

## Smaller cleanups

- [x] **cloneLevelState redundant `if (value.diff)` branch**
  Done: collapsed the two identical `Int32Array` constructions into one.

- [ ] **cloneLevelData simplification**
  Skipped: the existing form is already explicit; the suggested `new Int32Array(source || 0)` is equivalent but slightly cryptic.

- [ ] **arraysEqual / int32ArraysEqual near-duplicates**
  Left: the typed-array version is genuinely faster on hot paths; merging would lose that.

- [ ] **nextZobristSeed allocates {seed, value} per call**
  Skipped: only runs at table construction (one-time per wordCount, cached globally).

- [ ] **solverActionsForGame rebuilt per runMode invocation**
  Skipped: trivial, called once per `solveLevel`.

- [x] **someOnRowColPenalty unused `targetCount` binding**
  Done.

- [ ] **someOnLineDistancePenalty similar pattern**
  Already uses `targetCount` legitimately (passed to `nearestTargetLineDistance`); not actually redundant on review.

- [ ] **cellHasBlockingObject / cellHasStaticBlockingObject 95% identical**
  Now share `getAllowedMask`; remaining difference (mask intersection vs. blockerMask) is small. Not worth merging further.
