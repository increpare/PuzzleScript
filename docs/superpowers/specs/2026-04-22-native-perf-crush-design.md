# Native engine performance: crush-it plan

Status: design approved, pending implementation plan.
Date: 2026-04-22
Branch context: `cpp`, baseline commit `3a554bfb`.

## 1. Goals, scope, and success metrics

### Primary goal
Reduce C++ trace suite wall time from the current ~143s to well under the JS baseline of ~11s, targeting the 1–3s range ("crush JS"). Per-step simulation latency should land in the 1M–10M steps/sec range for median games.

### Scope

In scope:
- Everything in `native/src/` (`core.cpp`, `core.hpp`, `json.*`, hot paths in `cli_main.cpp`).
- Build configuration (`CMakeLists.txt` at the repo root and under `native/`).
- Wire format — JSON IR may be replaced late in Phase 1 with a flat-binary IR; JSON retained as a fallback/debug loader.

Out of scope:
- Porting compilation from JS to C++ (the future "D" end-state — tracked separately).
- Any observable engine semantics change.
- Editor / `play.html` / graphics / audio code paths — engine core only.
- Binary compatibility between phases: struct layouts and wire format change freely.

### Hard invariants (never regress)
- All 469 trace fixtures pass bit-for-bit identical to today.
- `prepared_session_checks_passed` stays 469/469.
- No change to observable engine semantics (RNG, movement ordering, rule firing order).

### Success metrics

Measured via `check-trace-sweep --profile-timers` (same command that produced the baseline `profile_stats.txt`).

| Gate              | `fast_replay_ms` | `wall_ms` | `trace_json_parse_ms + ir_miss_ms` |
| ----------------- | ---------------- | --------- | ---------------------------------- |
| Baseline today    | 114,731          | 143,341   | 28,089                             |
| Phase 1 exit      | ≤30,000          | ≤40,000   | ≤1,000                             |
| Phase 2 exit      | ≤6,000           | ≤10,000   | ≤1,000                             |
| Phase 3 exit      | ≤2,000           | ≤3,000    | ≤1,000                             |
| Project "crush"   | —                | ≤3,000    | —                                  |

"Done" for a phase means: all 469 fixtures still pass, the gate numbers are met on median-of-5 runs, and code review is signed off.

## 2. Game size assumptions (for fast-path sizing)

Measured distribution of games in scope:

- Median: 3 layers, 13 objects, 44 rules.
- ~85% of games have ≤32 objects.
- ~95% of games have ≤64 objects.
- ~90% of games have ≤10 layers.
- Wide tail — engine must still handle outliers correctly, just not optimally.

This justifies the fast-path boundary at `wordCount ∈ {1, 2}` (with `MaskWord = uint64_t`, i.e. 64 or 128 bits of object mask). Larger games take a slower dynamic path but remain correct.

## 3. Phase 1 — Data layout redesign

Theme: get allocations out of the hot path; get data cache-friendly. No algorithmic change — same semantics, packed differently.

### 3.1 Build configuration
- Default `CMAKE_BUILD_TYPE=Release` in `native/CMakeLists.txt` when the user has not set one. Honor explicit Debug.
- Release flags: `-O3 -DNDEBUG -march=native` (equivalent `-mcpu=apple-m1` on Apple Silicon), `-flto`.
- `-fno-exceptions -fno-rtti` if feasible across the codebase (profile shows ~1% in variant/exception guards; we do not throw across API boundaries).
- Explicitly disable libc++ hardening on Release: `_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE`. Eliminates the `__annotate_contiguous_container` / `__annotate_shrink` / `abi:nqe210106` instrumentation visible in the baseline profile.

### 3.2 `Mask` — the core type replacing `BitVector`

`BitVector = std::vector<int32_t>` is retired. New design:

```cpp
using MaskWord = uint64_t;
struct MaskRef { const MaskWord* data; };   // read-only view
struct MaskMut { MaskWord* data; };          // read-write view
```

Word count is implicit in context (stored once on `Game`) and not carried per-mask.

All per-game bitmasks (patterns, replacements, rules, player/aggregate/layer masks) live in one flat arena:

```cpp
struct Game {
  uint32_t wordCount;               // (objectCount + 63) / 64
  uint32_t movementWordCount;       // likewise for movement strides
  std::vector<MaskWord> maskArena;  // all bitsets concatenated, contiguous
  // Patterns, Rules, Replacements store uint32_t offsets into maskArena.
};
```

Per-session transient masks (`rowMasks`, `columnMasks`, `boardMask`, `rowMovementMasks`, `columnMovementMasks`, `boardMovementMask`, `rigidGroupIndexMasks`, etc.) move to `uint64_t` and live in a single `Session::sessionArena` sized at `loadLevel` time.

Why this shape:
- Every mask read in the hot loop is a known-stride pointer walk; the compiler unrolls `for (w=0; w<W; w++)` when W is 1 or 2 (≥85% of games).
- No heap allocation or indirection per mask.
- Cache-coherent: all of a rule's masks sit within a few cache lines.
- Uniform layout enables template specialization on `W∈{1,2}` later (see 4.4).

### 3.3 Flatten `Game` data

Today `Game` uses `std::map<std::string, BitVector>` (`glyphDict`, `objectMasks`, `aggregateMasks`) and `std::vector<std::vector<Rule>>` / `std::vector<std::vector<Pattern>>`. Phase 1 flattens:

- Name-keyed maps → sorted `std::vector<std::pair<std::string, uint32_t>>` keyed at load; runtime lookups use `std::lower_bound` or (if the profile demands) a small FNV-1a hash table. Used only at prepare/load, never in the hot step loop — correctness dominates speed here.
- `std::vector<std::vector<Rule>>` → flat `std::vector<Rule>` plus `std::vector<RuleGroup>` where `RuleGroup = {first, last}`.
- `Pattern::patterns` (`std::vector<std::vector<Pattern>>`) → one flat arena with `{row_offset, row_count, col_offset, col_count}` descriptors.

`Rule` and `Pattern` become POD-ish structs of offsets and ints — no owned heap data.

### 3.4 Scratch buffers on `Session`

`rebuildMasks` is the #1 hot function partly because each call repeatedly `vector::assign`s fresh buffers. Phase 1:
- Move all scratch buffers used by `step` and rule evaluation into `Session`. Allocate once in `createSession`, resize on `loadLevel`.
- `rebuildMasks` becomes `memset` of inline `uint64_t*` buffers. (`rebuildMasks` is deleted in Phase 2; this is a tactical step that removes the allocator from the hot loop meanwhile.)

### 3.5 JSON parser replacement

Current parser (`json.cpp`) uses `std::variant<..., std::map<std::string, Value>>` — visible in the profile.

- Replace with **simdjson** (single-header amalgamation in-tree).
- Gate behind a `PS_USE_SIMDJSON` define during integration; delete the old parser after one phase of green CI.
- Combined IR parse cost (26s `ir_miss_ms` + 2s `trace_json_parse_ms`) drops below 1s.

Optional late-Phase-1 follow-up: emit a **flat binary IR** from JS alongside the JSON. C++ mmaps and uses in place, driving load cost to effectively zero. May slip to Phase 2 if schedule is tight.

### Phase 1 exit criteria
All 469 fixtures green. `fast_replay_ms` ≤30,000. `trace_json_parse_ms + ir_miss_ms` ≤1,000. No heap allocations observed in `step()` under a debug allocator.

## 4. Phase 2 — Algorithmic improvements

Theme: do less work. Phase 1 made existing work fast; Phase 2 avoids the work entirely where possible.

### 4.1 Incremental row/column/board masks (the big win)

Today `rebuildMasks` scans the entire live level and rebuilds `rowMasks`, `columnMasks`, `boardMask`, `rowMovementMasks`, `columnMovementMasks`, `boardMovementMask` from scratch — inside the rule-group iteration loop. For an N-cell level each call is `O(N × W)`.

Phase 2 maintains these incrementally:
- Initial full build at `loadLevel` (one-time `O(N × W)`).
- `applyReplacementAt(cell, replacement)` computes the delta from the replacement (clears + sets) and `^=`s / `|=`s it into the affected row, column, and board-word entries in `O(W)`. The replacement is the only mutator of cell contents.
- Movement masks update similarly when `setCellMovements` fires.
- Rigid masks follow the same pattern.

Key invariant: at the top of every rule-group iteration the row/col/board masks are already correct. `rebuildMasks` becomes a no-op, then is deleted.

Expected impact: this is the single biggest lever in the plan. The baseline profile shows `rebuildMasks` plus its vector overhead at >60% of runtime.

### 4.2 Rule pre-match filtering

Each `Rule` already has a `ruleMask` (union of objects the rule can touch). With the incremental `boardMask` correct at all times:
- Before evaluating a rule: `if (!anyBitsInCommon(boardMask, rule.ruleMask)) continue;` — this prunes the majority of rules in the majority of iterations.
- Similarly test movement `ruleMaskMovements` against `boardMovementMask`.
- For directional rules, pre-compute direction-specific exclusions.

One-line addition per rule evaluation; cascades with 4.1 to make idle rule-group iterations near-free.

### 4.3 Cell-candidate index per rule (optional within Phase 2)

Gated: only pursued if 4.1 + 4.2 do not hit the Phase 2 gate.

- Maintain per-object-mask posting lists: `cellsContaining[objectMaskId] → sorted list of tile indices`. Updated incrementally in `applyReplacementAt`.
- For each rule, its first pattern row has required object bits; candidate starts are the intersection of the posting lists for those bits.
- Falls back to full scan when the first-row pattern is degenerate.

### 4.4 Fast-path specialization on word count

With Phase 1's uniform `MaskWord` arena, dispatch once per session (or per rule group) into a template specialized on `W`:

```cpp
template <int W> void applyRuleGroupN(Session&, RuleGroup);
// dispatch: switch (game->wordCount) { case 1: applyRuleGroupN<1>(...); case 2: ... default: applyRuleGroupNDyn(...); }
```

W=1 (~85% of games) and W=2 (~95% cumulative) get specialized, with inner `matchesPatternAt` and `applyReplacementAt` reduced to a handful of inline ops. Dynamic fallback covers the wide tail. The specializations are produced from the *same* template as the dynamic path — no hand-written 1-word fast path, to avoid divergence bugs.

### 4.5 Minor but worthwhile

- `getCellObjects` / `getCellMovements` currently allocate and return a `BitVector` by value (visible in the baseline hot stack). Replace with `MaskRef` pointing into the live buffer.
- `anyBitsSet`, `anyBitsInCommon` become `constexpr`-friendly inline functions over `MaskWord*`.
- Rule command dispatch: replace `std::vector<RuleCommand>` with packed enum codes covering the known command set (`cancel`, `restart`, `win`, `again`, `sfx0..9`, etc.).

### Phase 2 exit criteria
All 469 fixtures green. `fast_replay_ms` ≤6,000. `rebuildMasks` absent from top-20 profile entries. `step()` doing `O(firing rules × affected cells × W)` work, not `O(total rules × board)`.

## 5. Phase 3 — Optional, measurement-gated

Pursued only if Phase 1+2 have not hit the ≤3,000 `wall_ms` "crush it" ceiling.

### 5.1 Rule bytecode

Compile each rule once at game-load into a tiny bytecode sequence; evaluate with a switch-dispatched interpreter:

```
op MATCH_CELL     cell_offset, pattern_mask_offset
op MATCH_MOVEMENT cell_offset, movement_mask_offset
op APPLY_REPLACE  cell_offset, replacement_offset
op COMMAND        command_code, arg
op COMMIT
```

Dense tagged-dispatch over 8–16 opcodes, hot ops inlined. Expected additional gain on top of Phase 2: ~1.5–2×.

### 5.2 Codegen (if bytecode isn't enough)

In order of increasing complexity:
- **Template-expanded rule specialization** (same technique as 4.4) — default codegen path.
- **Generated C++ per-game**: emit a `.cpp` from the JS IR exporter and build a shared library. Useful for benchmarking, not for production deploy.
- **Real JIT** via libgccjit / asmjit: deferred unless everything else has been exhausted. Not planned in this spec; door left open for a future project.

### 5.3 Parallel test runner

Helps test-suite wall time, not per-game performance.
- 469 fixtures are independent — run in a thread or process pool.
- On 8-core Apple Silicon expect near-linear ~8× speedup on test iteration.
- Land this **first** in Phase 3 regardless of what else Phase 3 takes — it tightens the dev loop for everything that follows.

### 5.4 SIMD for mask operations (speculative)

NEON `uint64x2_t` handles W=2 masks in one op. Only pursued if the profile after 5.1/5.2 still shows mask ops dominant. Probably a wash after template specialization.

### Phase 3 decision gate
Do not start Phase 3 unless Phase 1+2 `wall_ms > 3000`. First item is always 5.3 (parallel runner). 5.1 next. 5.2/5.4 only if specific profile signals call for them.

## 6. Testing, measurement, risks

### 6.1 Correctness gate (every merge)
- All 469 trace fixtures pass bit-identical to today. `check-trace-sweep` must report `trace_replay_passed=469 trace_replay_failed=0 prepared_session_checks_passed=469 prepared_session_checks_failed=0`.
- No new compiler warnings at `-Wall -Wextra -Wpedantic`.
- CI runs the full fixture suite, not a sample.
- Undo/restart parity: these tests were fragile in recent history and Phase 2's incremental-mask work touches this area directly.

### 6.2 Development-time invariants
During Phase 2, a **debug-only** `verifyMaskInvariants(session)` full-rebuilds all row/col/board masks and compares them against the incrementally-maintained copies. Called at the end of every `executeTurn` under `#ifndef NDEBUG`. Similar check for posting lists if 4.3 ships.

### 6.3 Measurement protocol
Every phase is gated on numbers from `./run_profile.sh` (the wrapper that produced the baseline `profile_stats.txt`).

- Primary metrics: `wall_ms`, `fast_replay_ms`, `ir_miss_ms`, `trace_json_parse_ms`.
- **Median of 5 runs** (not mean, not best-of-N — median resists warmup and thermal spikes).
- Quiet laptop, on AC power, no heavy background processes, same shell session.
- Before-and-after `native_trace_suite_profile` lines go into the commit message of the phase-completion commit.

### 6.4 Performance discipline — mandatory regression gate

**The rule:** every PR that claims a performance improvement includes measured before/after numbers. Every PR, improvement-claiming or not, is measured against the previous phase's gate. A PR that regresses the measured metric is **reverted**, not iterated on, unless it is an explicit documented "enabling" PR in a stacked series.

**Regression thresholds (per PR):**
- `fast_replay_ms` or `wall_ms` regression >2% median-over-5: reverted.
- Regression 0–2%: noted in PR description with reason; allowed only if the PR's explicit purpose is correctness, cleanup, or enabling a named follow-up.
- Improvement-claiming PRs within ±2% (noise): the perf claim is struck, the PR is retitled as a refactor, merged only if independently justified.

**"Enabling" PRs:**
1. Title begins "Enabling PR — no perf claim, regression allowed up to X%".
2. Name the follow-up PR that realizes the gain.
3. Reverted if the follow-up does not land within the same phase.

**Phase-exit gate:**
The phase-exit squash PR must hit the numbers in §1's table. If it does not, the phase branch is held (not merged) until it does, or this spec is revised and re-approved before merging.

**No "the change is obviously good" exemptions.** If the profiler does not see it, it did not happen. This is the rule precisely because AI-proposed optimizations have a known failure mode of looking clever on paper and making things slower in practice.

**Mechanical help:**
- `scripts/perf_check.sh` runs the benchmark 5× and prints median plus each metric's delta vs. a saved baseline.
- Baseline file `perf_baseline.json` checked into the repo; updated only on phase-exit merge.
- Pre-merge reviewer checklist item: "`perf_check.sh` output pasted into the PR?"

### 6.5 Risks and mitigations

| Risk                                                           | Impact                      | Mitigation                                                                                                                |
| -------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Incremental mask drift (4.1)                                   | Silent correctness bug      | Debug-build full-rebuild comparison after every turn (6.2).                                                               |
| Arena-offset bugs in flattened `Game` (3.3)                    | Crash or corruption         | Debug-build bounds assertion on every `MaskRef` construction; fuzz against existing fixtures.                             |
| simdjson integration friction (3.5)                            | Schedule slip               | Keep old `json.cpp` behind `PS_USE_SIMDJSON` until parity is proven; delete after one phase of green CI.                  |
| Template-specialized paths diverge from dynamic path (4.4/5.2) | Divergent bugs hard to find | Produce W=1/W=2 specializations from the *same* template as the dynamic path. No hand-written fast path.                  |
| Build config change breaks someone's Debug workflow            | Developer friction          | Default to Release only when `CMAKE_BUILD_TYPE` is unset; honor explicit Debug.                                           |
| "Crush it" ceiling not met after Phase 3                       | Spec failure                | Profile reviewed explicitly at Phase 2 exit; re-scope before starting Phase 3 if trajectory is wrong.                     |

### 6.6 Branch and merge strategy
- One feature branch per phase: `perf/phase1-data-layout`, `perf/phase2-incremental`, optional `perf/phase3-codegen`.
- Within a phase, multiple small PRs against the phase branch, merged to master via a final squash PR at phase exit.
- Phase-exit PR commit message contains before/after profile numbers.

### 6.7 Explicitly out of scope
- Porting compilation JS → C++ (the "D" end-state — separate future project).
- Any engine semantics change.
- Editor / `play.html` / graphics / audio code paths.
- Multi-threading within a single game step (5.3's test-suite parallelism is a separate concern).
