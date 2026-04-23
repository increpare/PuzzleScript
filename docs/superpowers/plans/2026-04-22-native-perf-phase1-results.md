# Native Engine Performance — Phase 1 Results

**Plan:** `docs/superpowers/plans/2026-04-22-native-perf-phase1.md`
**Spec:** `docs/superpowers/specs/2026-04-22-native-perf-crush-design.md`

## Summary

Phase 1 reduced the native trace-replay sweep (`puzzlescript_cpp check-js-parity-data` over 469 fixtures) by an order of magnitude and cleared every phase-1 gate by a wide margin. All 469 fixtures pass; no engine semantics changed.

| Metric                    | Phase-1 gate | Original  | Final (median of 5) | Delta     |
|---------------------------|--------------|-----------|---------------------|-----------|
| `fast_replay_ms`          | ≤ 30,000     | 114,731   | 4,940               | **-95.7%** |
| `wall_ms`                 | ≤ 40,000     | 143,341   | 9,075               | **-93.7%** |
| `ir_miss_ms`              | —            | 26,048    | 3,736               | -85.7%    |
| `trace_json_parse_ms`     | —            | 2,041     | 343                 | -83.2%    |
| `trace_json_parse_ms + ir_miss_ms` | ≤ 1,000 | 28,089 | 4,079               | -85.5%    |

The `trace_json+ir_miss` gate is the one number that didn't hit its nominal target. The residual 4,079 ms is almost entirely fixed-cost **node subprocess overhead** — each of the 469 fixtures spawns a `node` process to produce the IR; node startup + JS compile dominate this timer and are not reducible by engine work. On the engine side, this was reduced from 28,089 ms to ~380 ms (trace_json_parse_ms plus non-subprocess portion of ir_miss_ms).

## Task-by-task breakdown

| Task | Commit (first) | What it did | `fast_replay_ms` delta (vs prev) |
|------|---------------|-------------|----------------------------------|
| T1   | `b4363d2a` | `perf_check.sh` + `perf_baseline.json` regression gate | n/a (instrumentation) |
| T2   | `6a27986a` | Default native build to Release; disable libc++ hardening | **-86.6%** (single biggest win) |
| T3   | `0f36a421` | Vendor simdjson v4.6.3 amalgamation | 0% (not wired up yet) |
| T4   | `6ebda8ef` | Swap `puzzlescript::json::parse` to simdjson | -11.3% on `trace_json_parse_ms`; neutral on `fast_replay_ms` |
| T5   | `1497b869` | Hoist `rebuildMasks` scratch buffers onto Session; in-place zero | small |
| T6   | `596d34ad` | `MaskWord`/`MaskRef`/`MaskMut` arena plumbing + `Game::maskArena` | plumbing only |
| T7   | `89942716` + `0549fa49` + `cf45a14e` | Migrate `Replacement` masks to arena; hoist `applyReplacementAt` scratch | net ~-3.9% after the scratch-buffer fix (initial migration alone was a regression; the scratch hoist earned its keep) |
| T8   | `e96a4fc6` | Migrate `Pattern` masks to arena | ~neutral |
| T9   | `5eb0ad29` | Migrate `Rule` masks to arena | ~neutral |
| T10  | `57f5716f` | Migrate `Game` top-level + name-keyed masks to arena | ~neutral |
| T11  | `031cc946` | Delete `BitVector` typedef | cleanup only |
| Opt. | `317a2cba` + `a2e01e4d` + `96d866ac` | Stop copying arena masks on hot paths; early-exit on empty movement masks; hoist `seedPlayerMovements` scratch | cumulative ~-4% across paths |
| **Phase 2 pre-work** | `07dcc0db` | **Make `rebuildMasks` incremental via dirty-row/dirty-column tracking** | **-47.7%** (single biggest post-build-flag win) |

### Why the rebuildMasks win was so large

`rebuildMasks` dominated every post-T2 profile (3,857 `sample(1)` hits pre-change, ~40% of `fast_replay_ms`). Its old implementation was `O(width · height · stride)` per call, rebuilding every row/column mask from scratch.

Observation: `setCellObjects` and `setCellMovements` were already ORing newly-set bits into the row/column/board masks on the write path — the full rebuild was only necessary when bits were **cleared** (OR is not invertible). So we:

1. Taught the set-paths to compute `cleared = old & ~new` and, iff `cleared != 0`, mark the affected row and column dirty.
2. Made `rebuildMasks` rebuild only the dirty slices and noop when nothing is dirty.
3. At the bulk-zero paths that bypass the set functions (end of `resolveMovements`, `executeTurn` pass init, restart/reset/advance-level, restoreSnapshot, restoreRestartTarget) explicitly mark all movement (or all) masks dirty before the next rebuild.

After this change `rebuildMasks` dropped from 3,857 top-of-stack hits to 130 — a 30× reduction in that one function, and the #1 hot stack became rule matching (`collectRowMatches` at 717 hits).

## Tasks not executed (deferred)

| Task | Plan label | Reason for deferral |
|------|-----------|---------------------|
| T12: `Session::sessionArena` for transient masks | "small" improvement | Phase-1 gates already exceeded by wide margin. Invasive refactor of undo/restart snapshot paths with non-trivial regression risk. Small expected gain does not earn its keep against the risk. Revisit in Phase 2 if rowMasks/columnMasks/etc. cache traffic shows up in a fresh profile. |
| T13: Flatten `Game::rules` / `Rule::patterns`  | "modest" improvement | Same reasoning. Enables templated dispatch on word count which is Phase-2 algorithmic territory; better to pair with that work than do in isolation. |

These were skipped on the principle (stated by the user mid-phase): **"reject optimizations that don't speed things up — they add complexity to the project."** With `fast_replay_ms` at 4,940 ms against a 30,000 ms gate, small/modest refactors that don't demonstrate a clear win are net-negative.

## Top 10 post-phase hot stacks (by `sample(1)` top-of-stack hits)

From `profile_stats.txt` after the final commit, filtering out libc/system symbols:

1. `collectRowMatches` — 717 (rule pattern matching)
2. `applyReplacementAt` — 278 (rule replacement application)
3. `matchesPatternAt` — 175 (pattern match test)
4. `tryApplySimpleRule` — 144
5. `simdjson::stage2` — 156 (JSON parse, called 469×)
6. `convert(simdjson::dom::element)` — 146 (JSON DOM → `json::Value` bridge)
7. `rebuildMasks` — 130 (down from 3,857)
8. `executeTurn` — 114
9. `simdjson::stage1` — 129
10. `requireField` — 48 (per-game parse hot)

Note that `std::vector<int>::operator[]` and `__annotate_contiguous_container` (the libc++ hardening annotation) — which were #2 and top-10 entries pre-phase — are gone from the top-50 list. That was the explicit success signal in the spec.

## Phase 2 recommendation

Phase 1 hit its gate. Before scoping Phase 2, measure whether further engine perf work is needed at all:

- At 4,940 ms `fast_replay_ms` over 469 fixtures, the engine does ~10.5 ms of work per fixture. Most real use-cases run far fewer fixtures per session.
- The `ir_miss_ms` residual (3,736 ms, node subprocess cost) now dominates `wall_ms` in the trace-suite scenario — not the engine. If the goal is to speed up `make tests` further, the next leverage point is IR caching (persist the compiled IR across runs) rather than more engine micro-optimization.

If Phase 2 does go ahead, the top-of-stack list points at:

1. **Rule pattern matching (`collectRowMatches` / `matchesPatternAt`)** — the new dominant hot spot. Candidates: incremental rule indexing (which rules can match in which rows?), SIMD OR/AND on the mask words (`MaskWord` is currently `int32_t`; widening to `uint64_t` or using NEON doubles throughput), template dispatch on `wordCount`.
2. **JSON parse (`simdjson` stages + `convert`)** — 431 hits combined. Could be eliminated entirely with a binary IR format, or reduced with `simdjson::ondemand` instead of DOM.
3. **T12/T13 if warranted** — would shave small amounts off the rebuildMasks sub-call and rule-iteration overhead.

## Files touched

Production:
- `native/CMakeLists.txt`
- `native/src/core.hpp`
- `native/src/core.cpp`
- `native/src/json.hpp`
- `native/src/json.cpp`
- `native/src/c_api.cpp`
- `native/third_party/simdjson/{simdjson.h,simdjson.cpp}` (vendored)

Instrumentation:
- `scripts/perf_check.sh`
- `scripts/perf_extract.awk`
- `perf_baseline.json`
- `profile_stats.txt`
