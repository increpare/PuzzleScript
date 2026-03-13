# PuzzleScript Engine Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve at least 10% reduction in test suite execution time through profile-guided optimization of the PuzzleScript engine's compile and runtime paths.

**Architecture:** Profile the full test suite using V8's `--prof` to identify CPU hotspots, then systematically optimize the top hotspots in order of impact. Each optimization is atomic — verify all tests pass before proceeding to the next.

**Tech Stack:** Node.js, V8 profiler (`--prof` / `--prof-process`), PuzzleScript engine (vanilla JS with runtime code generation)

---

## Chunk 1: Baseline & Profiling

### Task 1: Establish Baseline Timing

**Files:**
- Read: `src/tests/run_tests_node.js`

- [ ] **Step 1: Run the test suite 3 times and record wall-clock times**

Run each separately and note the "Total: N tests in X.XXs" output:

```bash
cd C:/Users/Anwender/Documents/GitHub/PuzzleScript
node src/tests/run_tests_node.js
node src/tests/run_tests_node.js
node src/tests/run_tests_node.js
```

Record the three times and compute the average. This is the baseline.

- [ ] **Step 2: Run with --breakdown to get function-level timing**

```bash
node src/tests/run_tests_node.js --breakdown
```

Record the compile/processInput/undo/restart breakdown. This tells us the rough split between compile-time and runtime.

- [ ] **Step 3: Commit baseline numbers as a comment at the top of this plan file**

Edit this plan file to add baseline numbers at the top under a "## Baseline" heading.

### Task 2: Generate V8 Profile

**Files:**
- Read: V8 profiler output (isolate-*.log)

- [ ] **Step 1: Run with V8 profiler**

```bash
cd C:/Users/Anwender/Documents/GitHub/PuzzleScript
node --prof src/tests/run_tests_node.js
```

This generates an `isolate-*.log` file in the current directory.

- [ ] **Step 2: Process the profile**

```bash
node --prof-process isolate-*.log > profile.txt
```

- [ ] **Step 3: Read and analyze the profile output**

Read `profile.txt`. Focus on:
1. The `[JavaScript]` section — which JS functions have the most ticks
2. The `[C++]` section — which V8 builtins are hot (indicates allocation, GC, etc.)
3. The `[Summary]` section — overall time distribution

Identify the top 5-10 hotspots by tick count. Record them in this plan file under a "## Profile Results" heading.

- [ ] **Step 4: Clean up profiler artifacts**

```bash
rm isolate-*.log
```

---

## Chunk 2: Optimize Hotspots (Iterative)

Based on profiling data, apply optimizations in order of impact. The specific optimizations below are based on code analysis of known inefficiency patterns in the engine. After profiling, reorder or skip these based on actual hotspot data.

### Task 3: Optimize `processInput` Direction Parsing

**Files:**
- Modify: `src/js/engine.js:2648-2665`

The `processInput` function uses `parseInt('00001', 2)` etc. for direction constants. `parseInt` is unnecessary overhead for constant values.

- [ ] **Step 1: Replace parseInt calls with literal constants**

In `src/js/engine.js`, in the `processInput` function (around line 2648), replace:
```javascript
case 0: // up
    dir = parseInt('00001', 2);
    break;
case 1: // left
    dir = parseInt('00100', 2);
    break;
case 2: // down
    dir = parseInt('00010', 2);
    break;
case 3: // right
    dir = parseInt('01000', 2);
    break;
case 4: // action
    dir = parseInt('10000', 2);
    break;
```

With:
```javascript
case 0: // up
    dir = 0b00001;
    break;
case 1: // left
    dir = 0b00100;
    break;
case 2: // down
    dir = 0b00010;
    break;
case 3: // right
    dir = 0b01000;
    break;
case 4: // action
    dir = 0b10000;
    break;
```

- [ ] **Step 2: Run tests to verify correctness**

```bash
node src/tests/run_tests_node.js
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/js/engine.js
git commit -m "perf: replace parseInt with binary literals in processInput"
```

### Task 4: Reduce Allocations in processInput's Rigid Loop

**Files:**
- Modify: `src/js/engine.js:2684-2732`

Each call to `processInput` creates `new Int32Array(level.objects)` for the start state, then potentially creates more on each rigid body rollback via `new Int32Array(startState.objects)`. Also uses `.concat([])` to clone arrays.

- [ ] **Step 1: Analyze allocation patterns**

Read `src/js/engine.js` lines 2680-2735. Identify:
- `startState.objects = new Int32Array(level.objects)` — initial backup
- `startState.movements = new Int32Array(level.movements)` — initial backup
- `level.objects = new Int32Array(startState.objects)` — rollback copy
- `level.movements = new Int32Array(startState.movements)` — rollback copy
- `.concat([])` used to clone `rigidGroupIndexMask` and `rigidMovementAppliedMask`

- [ ] **Step 2: Replace rollback `new Int32Array` with `.set()` to copy in-place**

The rollback on lines ~2727-2732 creates new typed arrays every iteration. Instead, copy data in-place using `.set()`:

Replace:
```javascript
level.objects = new Int32Array(startState.objects);
level.movements = new Int32Array(startState.movements);
```

With:
```javascript
level.objects.set(startState.objects);
level.movements.set(startState.movements);
```

This avoids allocating new typed arrays on each rigid loop iteration.

- [ ] **Step 3: Replace `.concat([])` with element-wise copy for rigid mask rollback**

The rigid mask arrays contain BitVec objects and need to be restored. Replace:
```javascript
level.rigidGroupIndexMask = startState.rigidGroupIndexMask.concat([]);
level.rigidMovementAppliedMask = startState.rigidMovementAppliedMask.concat([]);
```

With a loop that copies data back into existing BitVec objects (avoiding new array + new BitVec allocation):
```javascript
for (let j = 0; j < startState.rigidGroupIndexMask.length; j++) {
    startState.rigidGroupIndexMask[j].cloneInto(level.rigidGroupIndexMask[j]);
    startState.rigidMovementAppliedMask[j].cloneInto(level.rigidMovementAppliedMask[j]);
}
```

Also apply the same pattern to the initial backup of these masks (lines ~2687-2688).

- [ ] **Step 4: Run tests to verify correctness**

```bash
node src/tests/run_tests_node.js
```

Expected: All tests pass.

- [ ] **Step 5: Measure improvement**

```bash
node src/tests/run_tests_node.js
node src/tests/run_tests_node.js
node src/tests/run_tests_node.js
```

Average the three times and compare to baseline.

- [ ] **Step 6: Commit**

```bash
git add src/js/engine.js
git commit -m "perf: reduce typed array allocations in processInput rigid loop"
```

### Task 5: Optimize Level Backup/Restore

**Files:**
- Modify: `src/js/engine.js` (backupLevel function, ~line 689)

The `backupLevel` function at line 689 creates `new Int32Array(level.objects)` for every turn. This is a hot allocation path.

- [ ] **Step 1: Read the backupLevel function and undo system**

Read `src/js/engine.js` lines 689-730 to understand how backups are stored and used by DoUndo.

- [ ] **Step 2: Consider pooling or reusing backup arrays**

If the backup array is always the same size within a level, we could maintain a pool of pre-allocated Int32Arrays and reuse them. Analyze whether backup sizes change during gameplay.

- [ ] **Step 3: Implement the optimization**

The exact approach depends on what profiling reveals about allocation cost vs. the undo stack's usage pattern. If `backupLevel` is a top hotspot, implement array pooling. If not, skip this task.

- [ ] **Step 4: Run tests**

```bash
node src/tests/run_tests_node.js
```

- [ ] **Step 5: Measure and commit if improvement is meaningful**

### Task 6: Optimize `calculateRowColMasks`

**Files:**
- Modify: `src/js/engine.js` (calculateRowColMasks)

This is called at the start of every `processInput`. It recalculates row/column content caches for the entire level. If most of the level hasn't changed, this is wasted work.

- [ ] **Step 1: Read calculateRowColMasks**

Find and read the `calculateRowColMasks` function to understand what it computes and how.

- [ ] **Step 2: Profile whether this is actually hot**

Check the V8 profile output. If this function doesn't appear in the top hotspots, skip this task.

- [ ] **Step 3: If hot, consider incremental updates**

Instead of recalculating all masks from scratch, track which cells changed and only update affected rows/columns. This requires modifying `setCell` to mark dirty rows/columns.

- [ ] **Step 4: Implement, test, measure, commit**

### Task 7: Profile-Guided Optimizations (Dynamic)

This task covers additional optimizations discovered from the actual V8 profile data. The specific changes will be determined after Task 2 completes.

- [ ] **Step 1: Review profile data and identify remaining top hotspots**

After Tasks 3-6, re-profile to see what's now dominant.

- [ ] **Step 2: For each remaining hotspot, analyze and optimize**

Follow the pattern: read → diagnose → implement → verify → measure.

Common optimization patterns to look for:
- **Unnecessary object creation** in hot loops (replace with pre-allocated objects)
- **String concatenation** in generated code paths (use template caching)
- **Redundant Map/cache lookups** (cache results in local variables)
- **Megamorphic call sites** (V8 deoptimization from polymorphic dispatches)
- **Excessive `.length` property access** in tight loops (cache in local variable)
- **indexOf on arrays** (replace with Set for large collections)
- **BitVec method calls** in hot loops (inline the operations where profiling shows overhead)

- [ ] **Step 3: Repeat until stopping criteria met**

Stop when:
- Top remaining hotspot < 5% of total ticks, OR
- Three consecutive attempts yield < 2% cumulative improvement

---

## Chunk 3: Validation

### Task 8: Final Validation

**Files:**
- Read: `src/tests/run_tests_node.js`

- [ ] **Step 1: Run final profiling pass**

```bash
cd C:/Users/Anwender/Documents/GitHub/PuzzleScript
node --prof src/tests/run_tests_node.js
node --prof-process isolate-*.log > profile_final.txt
```

Compare `profile_final.txt` hotspots to the original `profile.txt`. Verify that the original hotspots have been reduced.

- [ ] **Step 2: Run final timing comparison**

```bash
node src/tests/run_tests_node.js
node src/tests/run_tests_node.js
node src/tests/run_tests_node.js
```

Average the three times. Compare to the baseline average recorded in Task 1. The improvement should be at least 10%.

- [ ] **Step 3: Run full test suite one more time to confirm all tests pass**

```bash
node src/tests/run_tests_node.js
```

Expected: 0 failures, 0 errors.

- [ ] **Step 4: Clean up profiler artifacts**

```bash
rm isolate-*.log profile.txt profile_final.txt
```

- [ ] **Step 5: Commit final state**

If any uncommitted optimization changes remain, commit them.

- [ ] **Step 6: Update this plan with final results**

Edit this plan file to add a "## Results" section with:
- Baseline time
- Final time
- Percentage improvement
- Summary of optimizations applied
