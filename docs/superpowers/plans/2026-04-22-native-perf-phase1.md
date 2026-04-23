/# Native Engine Performance — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut native trace-suite `fast_replay_ms` from 114,731 → ≤30,000 and `trace_json_parse_ms + ir_miss_ms` from 28,089 → ≤1,000 by fixing build config and moving all per-game/per-session bitmasks into contiguous arenas, with zero semantic change to the engine.

**Architecture:** Three themes — (1) turn on optimizer, disable libc++ hardening; (2) replace the `std::variant + std::map` JSON parser with simdjson; (3) replace `BitVector = std::vector<int32_t>` with a single contiguous `MaskWord` arena per `Game` (and per `Session` for transient masks) so bit operations become stride-known inline loops with no heap traffic.

**Tech Stack:** C++17, CMake, libc++ (Apple clang), simdjson single-header amalgamation, QUnit-based trace fixture suite.

**Spec:** `docs/superpowers/specs/2026-04-22-native-perf-crush-design.md`.

**Follow-on plans (written after this phase completes):**
- Phase 2 — algorithmic improvements (incremental masks, rule filtering)
- Phase 3 — optional, measurement-gated (bytecode, parallel runner)

---

## File Structure

**New files:**
- `scripts/perf_check.sh` — run profile 5×, print median and deltas vs baseline.
- `scripts/perf_extract.awk` — extract profile metrics from a pass1.stderr file.
- `perf_baseline.json` — tracked baseline numbers; updated only at phase-exit merges.
- `native/third_party/simdjson/simdjson.h` and `simdjson.cpp` — vendored amalgamation.

**Modified files:**
- `native/CMakeLists.txt` — default Release build type, Release flags, hardening disabled, simdjson source, libc++ hardening toggle.
- `native/src/core.hpp` — `MaskWord`, `MaskRef`, `MaskMut` typedefs; `Game::maskArena`, `Game::wordCount`, `Game::movementWordCount`; `Session::sessionArena`; arena offsets on `Replacement` / `Pattern` / `Rule` / `WinCondition` / `SoundMaskEntry`; flattened rule groups; sorted name→offset vectors replacing `std::map` members.
- `native/src/core.cpp` — parse functions write into the arena and return offsets; accessors returning `MaskRef` instead of `const BitVector&`; hoisted scratch buffers on `Session`; `rebuildMasks` writes to arena-backed words.
- `native/src/json.cpp`, `native/src/json.hpp` — replaced with a thin wrapper over simdjson's DOM API that preserves the existing `json::Value` interface (minimising blast radius).

**Deleted files (at phase exit, after one PR of green CI):**
- The now-unused `BitVector`, `std::map<std::string, BitVector>` fields inside `Game` and related structs.

---

## Task 1: Baseline Measurement Scaffolding

**Files:**
- Create: `scripts/perf_check.sh`
- Create: `scripts/perf_extract.awk`
- Create: `perf_baseline.json`

**Purpose:** Without a reliable measurement tool, the regression gate in the spec (§5.7 / §6.4) is not enforceable. This task makes every subsequent task measurable.

- [ ] **Step 1: Create `scripts/perf_extract.awk`**

```awk
# Extracts profile metrics from a pass1.stderr file.
# Expects a line matching: native_trace_suite_profile simulation_fixtures=469 ... wall_ms=N ...
/^native_trace_suite_profile/ {
  for (i = 1; i <= NF; i++) {
    split($i, kv, "=")
    if (kv[1] == "wall_ms"             ) wall = kv[2]
    if (kv[1] == "fast_replay_ms"      ) fast = kv[2]
    if (kv[1] == "ir_miss_ms"          ) irmiss = kv[2]
    if (kv[1] == "trace_json_parse_ms" ) jsonp = kv[2]
  }
}
END {
  printf "{\"wall_ms\":%s,\"fast_replay_ms\":%s,\"ir_miss_ms\":%s,\"trace_json_parse_ms\":%s}\n", wall, fast, irmiss, jsonp
}
```

- [ ] **Step 2: Create `scripts/perf_check.sh`**

```bash
#!/usr/bin/env bash
# Run the native profile 5 times, take the median of each metric, compare
# against perf_baseline.json, and print a diff table. Exit 0 unconditionally
# (gatekeeping is done by humans reading the output, not this script).

set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PUZZLESCRIPT_CPP="${PUZZLESCRIPT_CPP:-$ROOT/build/native/native/puzzlescript_cpp}"
MANIFEST="${PROFILE_MANIFEST:-$ROOT/build/native/coverage-fixtures/fixtures.json}"
BASELINE="$ROOT/perf_baseline.json"
RUNS="${PERF_RUNS:-5}"

if [[ ! -x "$PUZZLESCRIPT_CPP" ]]; then echo "missing $PUZZLESCRIPT_CPP — build native first" >&2; exit 2; fi
if [[ ! -f "$MANIFEST" ]]; then echo "missing $MANIFEST — run: make coverage-fixtures" >&2; exit 2; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for i in $(seq 1 "$RUNS"); do
  "$PUZZLESCRIPT_CPP" check-js-parity-data "$MANIFEST" --profile-timers >"$TMP/run$i.stdout" 2>"$TMP/run$i.stderr"
  awk -f "$ROOT/scripts/perf_extract.awk" "$TMP/run$i.stderr" > "$TMP/run$i.json"
done

python3 - "$TMP" "$RUNS" "$BASELINE" <<'PY'
import json, sys, os, statistics
tmp, runs, baseline_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
metrics = ["wall_ms","fast_replay_ms","ir_miss_ms","trace_json_parse_ms"]
samples = {m: [] for m in metrics}
for i in range(1, runs+1):
    with open(os.path.join(tmp, f"run{i}.json")) as f:
        d = json.load(f)
    for m in metrics: samples[m].append(int(d[m]))
median = {m: statistics.median(samples[m]) for m in metrics}
print(f"runs: {runs}")
for m in metrics:
    print(f"  {m}: samples={samples[m]} median={median[m]}")

if os.path.exists(baseline_path):
    with open(baseline_path) as f: baseline = json.load(f)
    print("\ndelta vs baseline:")
    for m in metrics:
        b = int(baseline[m]); c = median[m]
        pct = 100.0 * (c - b) / b if b else 0.0
        marker = "  OK" if c <= b * 1.02 else "REGR"
        print(f"  [{marker}] {m}: baseline={b} current={c} delta={c-b:+d} ({pct:+.1f}%)")
else:
    print(f"\n(no baseline at {baseline_path}; write current numbers there to establish one)")
PY
```

- [ ] **Step 3: Make the script executable**

Run: `chmod +x scripts/perf_check.sh`

- [ ] **Step 4: Create `perf_baseline.json` with today's numbers**

From the committed `profile_stats.txt` (commit `3a554bfb`):

```json
{
  "wall_ms": 143341,
  "fast_replay_ms": 114731,
  "ir_miss_ms": 26048,
  "trace_json_parse_ms": 2041
}
```

- [ ] **Step 5: Smoke-test the script**

Run: `make build-native coverage-fixtures && scripts/perf_check.sh`

Expected: prints a "runs:" block with 5 sample arrays, a median line per metric, and a "delta vs baseline" block where all four metrics land near 0% (within ±10% — single-run noise). No `[REGR]` markers beyond noise-level deltas, but if the machine is thermally loaded and it reports `[REGR]` on first run, that's informational, not a gate.

- [ ] **Step 6: Commit**

```bash
git add scripts/perf_check.sh scripts/perf_extract.awk perf_baseline.json
git commit -m "Add perf_check.sh + perf_baseline.json regression gate scaffolding"
```

---

## Task 2: Release Build Defaults + Disable libc++ Hardening

**Files:**
- Modify: `native/CMakeLists.txt` (top of file)

**Purpose:** The current build has `CMAKE_BUILD_TYPE` empty, meaning no `-O3`, no `-DNDEBUG`, and libc++ hardening live. This is the single largest "free" win in Phase 1.

- [ ] **Step 1: Edit `native/CMakeLists.txt` — add default build type and Release flags**

Insert the following block immediately after the `set(CMAKE_CXX_STANDARD ...)` / `set(CMAKE_CXX_EXTENSIONS OFF)` lines:

```cmake
# --- Performance defaults ----------------------------------------------------
# Default to Release when the user hasn't set anything. Honor explicit Debug.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Release, Debug, RelWithDebInfo, MinSizeRel)" FORCE)
  message(STATUS "Defaulting CMAKE_BUILD_TYPE to Release (native perf build).")
endif()

# Release-build flags: -O3, LTO, native codegen, NDEBUG. Tuned for the
# performance target in docs/superpowers/specs/2026-04-22-native-perf-crush-design.md.
set(PS_RELEASE_EXTRA_FLAGS
    -O3
    -DNDEBUG
    -march=native
    -fno-exceptions
    -fno-rtti
    -flto)

# libc++ safe-mode instrumentation (seen as __annotate_contiguous_container in
# baseline profiles) is a large tax; turn it off on Release only.
set(PS_RELEASE_EXTRA_DEFINES
    _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_NONE)

add_compile_options($<$<CONFIG:Release>:${PS_RELEASE_EXTRA_FLAGS}>)
add_compile_definitions($<$<CONFIG:Release>:${PS_RELEASE_EXTRA_DEFINES}>)
add_link_options($<$<CONFIG:Release>:-flto>)
```

- [ ] **Step 2: Nuke and reconfigure build directory**

Some CMake caches persist the old empty `CMAKE_BUILD_TYPE`; a clean regen is safer than fighting the cache.

Run:
```
rm -rf build/native
make build-native
```

Expected output includes: `-- Defaulting CMAKE_BUILD_TYPE to Release (native perf build).`

- [ ] **Step 3: Confirm the build is actually Release**

Run:
```
grep '^CMAKE_BUILD_TYPE' build/native/CMakeCache.txt
```

Expected: `CMAKE_BUILD_TYPE:STRING=Release`

- [ ] **Step 4: Run the full fixture suite**

Run: `make tests`

Expected: suite completes, `trace_replay_checked=469 trace_replay_passed=469 trace_replay_failed=0`. Wall time is now a small fraction of the baseline; first-glance confirmation.

- [ ] **Step 5: Run `perf_check.sh` and record numbers**

Run: `scripts/perf_check.sh`

Expected: significant speedups against the baseline across all four metrics. Specifically I expect `fast_replay_ms` roughly in the 20,000–40,000 range (vs. 114,731). If `fast_replay_ms` is still >50,000, `-O3` likely isn't being applied — investigate before proceeding.

- [ ] **Step 6: Update `perf_baseline.json` to the new Release numbers**

Replace `perf_baseline.json` with the median numbers printed by `perf_check.sh` in Step 5. Keep the file format:

```json
{
  "wall_ms": <median wall_ms>,
  "fast_replay_ms": <median fast_replay_ms>,
  "ir_miss_ms": <median ir_miss_ms>,
  "trace_json_parse_ms": <median trace_json_parse_ms>
}
```

- [ ] **Step 7: Commit**

```bash
git add native/CMakeLists.txt perf_baseline.json
git commit -m "Default native build to Release; disable libc++ hardening

fast_replay_ms: <baseline> -> <new>
wall_ms:        <baseline> -> <new>"
```

---

## Task 3: Vendor simdjson and Gate Old Parser Behind a Toggle

**Files:**
- Create: `native/third_party/simdjson/simdjson.h`
- Create: `native/third_party/simdjson/simdjson.cpp`
- Modify: `native/CMakeLists.txt`

**Purpose:** Get simdjson present in the build without using it yet, so its compile-time cost is bounded before we start rewiring the parser.

- [ ] **Step 1: Download the simdjson single-header amalgamation**

Fetch the amalgamation from the simdjson release tarball into `native/third_party/simdjson/`. The two files needed are `simdjson.h` and `simdjson.cpp`. Use the latest 3.x tag.

Run (adjust version if a newer release exists):
```
mkdir -p native/third_party/simdjson
curl -L https://github.com/simdjson/simdjson/releases/download/v3.11.3/singleheader.zip -o /tmp/simdjson.zip
unzip -o /tmp/simdjson.zip 'simdjson.h' 'simdjson.cpp' -d native/third_party/simdjson/
```

- [ ] **Step 2: Add simdjson to the CMake build**

Edit `native/CMakeLists.txt`. In the `PUZZLESCRIPT_NATIVE_SOURCES` list, append `third_party/simdjson/simdjson.cpp`. After `target_include_directories(puzzlescript_native ...)`, add:

```cmake
target_include_directories(puzzlescript_native
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/simdjson)

# simdjson is performance-critical but its own sources use RTTI/exceptions
# paths we disabled project-wide. Silence the warnings from -fno-rtti on its
# SIMD dispatch; simdjson does not use dynamic_cast on its own classes.
set_source_files_properties(
  third_party/simdjson/simdjson.cpp
  PROPERTIES COMPILE_FLAGS "-Wno-everything")
```

- [ ] **Step 3: Add a no-op include of simdjson in `core.cpp` to confirm it builds**

At the top of `native/src/core.cpp`, add (after the other includes):

```cpp
#include "simdjson.h" // vendored; will replace puzzlescript::json in Task 4
```

- [ ] **Step 4: Build and run tests**

Run: `make build-native && make tests`

Expected: build succeeds, no new warnings in `puzzlescript_native` sources, all 469 fixtures pass.

- [ ] **Step 5: Run perf_check.sh**

Run: `scripts/perf_check.sh`

Expected: metrics within ±2% of baseline (no simdjson code has been called yet — this is a pure "does adding the dependency regress the build" check).

- [ ] **Step 6: Commit**

```bash
git add native/third_party/simdjson/ native/CMakeLists.txt native/src/core.cpp
git commit -m "Vendor simdjson v3.11.3 amalgamation; not yet wired up"
```

---

## Task 4: Replace `puzzlescript::json` Internals With simdjson

**Files:**
- Modify: `native/src/json.hpp`
- Modify: `native/src/json.cpp`
- Test: `make tests` (the 469 fixtures are the correctness test)

**Purpose:** Keep the `puzzlescript::json::Value` API stable (so call sites in `core.cpp` don't change) but swap the backing parser. This drops `trace_json_parse_ms + ir_miss_ms` from ~28s to <1s.

- [ ] **Step 1: Keep `json.hpp`'s public surface unchanged**

Do not modify `native/src/json.hpp`. The `Value` class, its accessors, and `parse(std::string_view)` stay as-is. Callers in `core.cpp` continue to work verbatim.

- [ ] **Step 2: Rewrite `json.cpp` to build `Value` via simdjson**

Replace the body of `native/src/json.cpp` entirely. Keep the constructors and trivial accessors (`Value::Value`, `kind`, `isBool`, etc.) byte-for-byte identical to the current implementation. Replace the `Parser` class and the top-level `parse()` function with an adapter that walks simdjson's DOM and produces the same `Value` tree:

```cpp
#include "json.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "simdjson.h"

namespace puzzlescript::json {

namespace {

Value convert(simdjson::dom::element el) {
    using simdjson::dom::element_type;
    switch (el.type()) {
        case element_type::NULL_VALUE:
            return Value();
        case element_type::BOOL:
            return Value(static_cast<bool>(el.get_bool()));
        case element_type::INT64:
            return Value(static_cast<int64_t>(el.get_int64()));
        case element_type::UINT64:
            return Value(static_cast<int64_t>(static_cast<int64_t>(el.get_uint64())));
        case element_type::DOUBLE:
            return Value(static_cast<double>(el.get_double()));
        case element_type::STRING: {
            std::string_view sv = el.get_string();
            return Value(std::string(sv));
        }
        case element_type::ARRAY: {
            Value::Array out;
            simdjson::dom::array arr = el.get_array();
            out.reserve(arr.size());
            for (auto child : arr) {
                out.push_back(convert(child));
            }
            return Value(std::move(out));
        }
        case element_type::OBJECT: {
            Value::Object out;
            simdjson::dom::object obj = el.get_object();
            for (auto field : obj) {
                std::string_view key = field.key;
                out.emplace(std::string(key), convert(field.value));
            }
            return Value(std::move(out));
        }
    }
    throw ParseError("unhandled simdjson element_type");
}

} // namespace

Value parse(std::string_view input) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(input.data(), input.size());
    simdjson::dom::element root;
    auto err = parser.parse(padded).get(root);
    if (err) {
        throw ParseError(std::string("simdjson: ") + simdjson::error_message(err));
    }
    return convert(root);
}

// ----- Value implementation (unchanged semantics) -------------------------

// [Copy the existing Value ctors and accessors from the pre-simdjson json.cpp
//  verbatim. They operate on the std::variant member and have no parser ties.]

} // namespace puzzlescript::json
```

The comment block must be replaced with the actual copied-in code — do not leave a placeholder. Lift the existing `Value` methods from the current `json.cpp` below the namespace brace.

- [ ] **Step 3: Build and run the fixture suite**

Run: `make build-native && make tests`

Expected: all 469 fixtures still pass. If a fixture breaks, simdjson is producing a different numeric type (e.g. a number that overflows int64 coming through `UINT64`), or object-key order matters somewhere — investigate the specific failing fixture.

- [ ] **Step 4: Run `perf_check.sh`**

Run: `scripts/perf_check.sh`

Expected: `trace_json_parse_ms` and `ir_miss_ms` both collapse to a small fraction of baseline (target: sum <1,000ms). `fast_replay_ms` essentially unchanged.

- [ ] **Step 5: Update `perf_baseline.json`**

Record the new medians.

- [ ] **Step 6: Commit**

```bash
git add native/src/json.cpp perf_baseline.json
git commit -m "Back puzzlescript::json::parse with simdjson

ir_miss_ms:          <before> -> <after>
trace_json_parse_ms: <before> -> <after>"
```

---

## Task 5: Hoist `rebuildMasks` Scratch Buffers Onto `Session`

**Files:**
- Modify: `native/src/core.hpp` (add fields to `Session`)
- Modify: `native/src/core.cpp` (change `rebuildMasks` to use hoisted buffers)

**Purpose:** `rebuildMasks` is the #1 hot function. Before we change the data layout, get its per-call `vector::assign` calls out of the hot path by reusing `Session`-owned buffers. This is a tactical win; `rebuildMasks` itself gets deleted entirely in Phase 2.

- [ ] **Step 1: Identify the current `rebuildMasks` implementation**

Open `native/src/core.cpp`, find the function `rebuildMasks`. Identify every local `std::vector<int32_t>` (or equivalent) that it creates or resizes per call. These are the candidates to hoist.

(If there are none and the function directly writes to `session.rowMasks` etc., skip to Step 4 — the scratch-buffer hoisting is instead about making sure `session.rowMasks`/`columnMasks`/`boardMask` are sized once at `loadLevel` time and only zeroed per-call, not `assign`ed to a new size.)

- [ ] **Step 2: Add any needed scratch fields on `Session`**

Edit `native/src/core.hpp`. Inside `struct Session`, append (immediately before the existing `bool canUndo` line):

```cpp
// Reusable scratch buffers for the hot path. Sized at loadLevel and never
// reallocated thereafter; rebuildMasks and step() treat these as write-only
// working storage. Keeping them here (rather than recreating per-call) avoids
// malloc/free traffic visible in baseline profiles.
std::vector<int32_t> rebuildMasksScratchRow;
std::vector<int32_t> rebuildMasksScratchCol;
std::vector<int32_t> rebuildMasksScratchBoard;
std::vector<int32_t> rebuildMasksScratchRowMov;
std::vector<int32_t> rebuildMasksScratchColMov;
std::vector<int32_t> rebuildMasksScratchBoardMov;
```

Only keep the ones `rebuildMasks` actually needs (from Step 1's analysis). Delete the others.

- [ ] **Step 3: Size the scratch buffers in `loadLevel`**

In `native/src/core.cpp`, find `loadLevel` (or whichever function initialises `session.rowMasks` etc. after a level is loaded). Immediately after those existing `assign`/`resize` calls, resize every scratch buffer added in Step 2 to the same size, and zero-fill:

```cpp
session.rebuildMasksScratchRow.assign(session.rowMasks.size(), 0);
session.rebuildMasksScratchCol.assign(session.columnMasks.size(), 0);
// ...one line per scratch buffer added in Step 2
```

- [ ] **Step 4: Change `rebuildMasks` to `std::fill` the existing buffers instead of reallocating**

Inside `rebuildMasks`, replace any `vec.assign(n, 0)` that resets one of the Session-owned mask buffers with `std::fill(vec.begin(), vec.end(), 0)`. If a genuine scratch vector was being constructed locally, replace it with the hoisted `session.rebuildMasksScratch*` buffer + a `std::fill`.

- [ ] **Step 5: Run the fixture suite**

Run: `make build-native && make tests`

Expected: 469/469 pass.

- [ ] **Step 6: Run `perf_check.sh`**

Expected: `fast_replay_ms` down by 5–15% vs. the post-Task-4 baseline. If worse or flat, something is sizing the buffers wrong — check `loadLevel` and undo/restart paths (`restart`, `undo` must also resize the scratch buffers if they swap in a different level).

- [ ] **Step 7: Update `perf_baseline.json` and commit**

```bash
git add native/src/core.hpp native/src/core.cpp perf_baseline.json
git commit -m "Hoist rebuildMasks scratch buffers onto Session

fast_replay_ms: <before> -> <after>"
```

---

## Task 6: Introduce `MaskWord`, `MaskRef`, `MaskMut`, and the `Game` Arena Plumbing

**Files:**
- Modify: `native/src/core.hpp`
- Modify: `native/src/core.cpp` (add helpers, no call-site migration yet)

**Purpose:** Land the arena types and helper functions without changing existing call sites. This lets subsequent tasks migrate one struct at a time without a single enormous commit.

- [ ] **Step 1: Add the `MaskWord` typedefs and arena fields to `Game` in `core.hpp`**

Near the top of `namespace puzzlescript` in `native/src/core.hpp`, add:

```cpp
// ---- Mask representation (Phase 1) -----------------------------------------
// Every per-game bitmask (pattern masks, replacement masks, rule masks, layer
// masks, glyph masks, aggregate masks, etc.) is stored as a run of `wordCount`
// consecutive MaskWords inside Game::maskArena. Structs that owned a BitVector
// now store a uint32_t offset into the arena.
//
// MaskWord is kept as int32_t in Phase 1 to match the existing IR layout; the
// Phase 2 plan switches it to uint64_t after the arena is in place.
using MaskWord = int32_t;

struct MaskRef { const MaskWord* data; };
struct MaskMut { MaskWord* data; };

// Offset into Game::maskArena (in words, not bytes). `kNullMaskOffset` means
// "no mask assigned" (used for fields that are optional or vary per pattern).
using MaskOffset = uint32_t;
inline constexpr MaskOffset kNullMaskOffset = static_cast<uint32_t>(-1);
```

- [ ] **Step 2: Add `maskArena`, `wordCount`, `movementWordCount` fields to `Game`**

Inside `struct Game` in `core.hpp`, near the top of the members (just below `schemaVersion`, `strideObject`, `strideMovement`):

```cpp
uint32_t wordCount = 0;          // = (objectCount + 31) / 32 with int32 words
uint32_t movementWordCount = 0;  // = wordCount * directions-per-cell factor
std::vector<MaskWord> maskArena; // all per-game bitmasks concatenated
```

- [ ] **Step 3: Add helper functions in `core.cpp` (inside the anonymous namespace)**

In `native/src/core.cpp`, inside the top anonymous namespace (near the other helpers like `parseIntVector`), add:

```cpp
// Append `words` into `game.maskArena` and return the offset of the first
// element. Used during IR parsing to populate the arena.
MaskOffset storeMaskWords(Game& game, const std::vector<int32_t>& words) {
    MaskOffset offset = static_cast<MaskOffset>(game.maskArena.size());
    game.maskArena.insert(game.maskArena.end(), words.begin(), words.end());
    return offset;
}

// Append `wordCount` zero words and return the offset. Used for fields that
// are absent in the IR and need an all-zero mask at the arena's width.
MaskOffset storeZeroMask(Game& game) {
    MaskOffset offset = static_cast<MaskOffset>(game.maskArena.size());
    game.maskArena.insert(game.maskArena.end(), game.wordCount, 0);
    return offset;
}

MaskRef maskAt(const Game& game, MaskOffset offset) {
    return MaskRef{ game.maskArena.data() + offset };
}

MaskMut maskAt(Game& game, MaskOffset offset) {
    return MaskMut{ game.maskArena.data() + offset };
}

bool anyBitsSet(MaskRef m, uint32_t wordCount) {
    for (uint32_t w = 0; w < wordCount; ++w) {
        if (m.data[w] != 0) return true;
    }
    return false;
}

bool anyBitsInCommon(MaskRef a, MaskRef b, uint32_t wordCount) {
    for (uint32_t w = 0; w < wordCount; ++w) {
        if ((a.data[w] & b.data[w]) != 0) return true;
    }
    return false;
}
```

- [ ] **Step 4: Set `wordCount` at game-load time**

In `native/src/core.cpp`, find where `Game::strideObject` is populated during IR parsing (in the top-level parser, probably `loadGameFromJson` or equivalent). After it's set, compute and assign `wordCount`:

```cpp
game.wordCount = static_cast<uint32_t>(game.strideObject);  // strideObject is already (objectCount+31)/32 in current IR
game.movementWordCount = static_cast<uint32_t>(game.strideMovement);
game.maskArena.reserve(1024);  // generous initial reserve; ~most games fit
```

(Exact field name for the per-game stride may differ — use whatever the current IR loader populates. If `strideObject` is in 32-bit words already, no transformation is needed. Confirm the invariant by asserting one of the existing per-game masks, e.g. `game.playerMask.size() == game.wordCount`, after all fields are parsed.)

- [ ] **Step 5: Run the fixture suite — nothing should change**

Run: `make build-native && make tests`

Expected: 469/469 pass. This task added types and helpers but doesn't change any semantics.

- [ ] **Step 6: Run `perf_check.sh`**

Expected: flat against baseline (±2%). The arena and helpers are unused.

- [ ] **Step 7: Commit**

```bash
git add native/src/core.hpp native/src/core.cpp
git commit -m "Add MaskWord/MaskRef/MaskMut types and Game mask arena plumbing

Enabling PR — no perf claim. Subsequent tasks migrate Replacement,
Pattern, Rule, and Game top-level masks onto the arena."
```

---

## Task 7: Migrate `Replacement` Masks to Arena Offsets

**Files:**
- Modify: `native/src/core.hpp` (`struct Replacement`)
- Modify: `native/src/core.cpp` (parser + every consumer of a `Replacement` mask)

**Purpose:** First of four migrations. `Replacement` is the smallest/simplest — good place to validate the pattern before tackling larger structs.

- [ ] **Step 1: Change `Replacement` in `core.hpp` to hold offsets**

Replace the existing `struct Replacement` body with:

```cpp
struct Replacement {
    MaskOffset objectsClear       = kNullMaskOffset;
    MaskOffset objectsSet         = kNullMaskOffset;
    MaskOffset movementsClear     = kNullMaskOffset;
    MaskOffset movementsSet       = kNullMaskOffset;
    MaskOffset movementsLayerMask = kNullMaskOffset;
    MaskOffset randomEntityMask   = kNullMaskOffset;
    MaskOffset randomDirMask      = kNullMaskOffset;
};
```

- [ ] **Step 2: Change the `Replacement` parse function to store into the arena**

Find `parseReplacement` (or the lines currently doing `replacement.objectsClear = parseIntVector(requireField(object, "objects_clear"));`). Replace each such line with:

```cpp
replacement.objectsClear       = storeMaskWords(game, parseIntVector(requireField(object, "objects_clear")));
replacement.objectsSet         = storeMaskWords(game, parseIntVector(requireField(object, "objects_set")));
replacement.movementsClear     = storeMaskWords(game, parseIntVector(requireField(object, "movements_clear")));
replacement.movementsSet       = storeMaskWords(game, parseIntVector(requireField(object, "movements_set")));
replacement.movementsLayerMask = storeMaskWords(game, parseIntVector(requireField(object, "movements_layer_mask")));
replacement.randomEntityMask   = storeMaskWords(game, parseIntVector(requireField(object, "random_entity_mask")));
replacement.randomDirMask      = storeMaskWords(game, parseIntVector(requireField(object, "random_dir_mask")));
```

(The parse function needs access to `Game&` — thread it through as a parameter if it doesn't already have it.)

- [ ] **Step 3: Update every consumer of `Replacement.*` mask fields**

Search `native/src/core.cpp` for every read of `replacement.objectsClear`, `replacement.objectsSet`, etc. Every call site needs to change from using `const BitVector&` to `MaskRef` and a word count:

Pattern to replace:
```cpp
const BitVector& m = replacement.objectsClear;
for (size_t w = 0; w < m.size(); ++w) { ... m[w] ... }
```

Becomes:
```cpp
MaskRef m = maskAt(*session.game, replacement.objectsClear);
for (uint32_t w = 0; w < session.game->wordCount; ++w) { ... m.data[w] ... }
```

For movement masks, use `movementWordCount` instead.

- [ ] **Step 4: Handle the "random" masks (which may legally have different widths)**

`randomEntityMask` and `randomDirMask` may have a different word count than `wordCount`. Either:
- Store their width alongside the offset in the arena (add a header word: `arena[offset-1] = wordCount`), OR
- Keep `Replacement::randomEntityMaskWidth` / `randomDirMaskWidth` as separate `uint32_t` fields.

Recommend the second — explicit fields:

```cpp
struct Replacement {
    // ... existing offset fields ...
    uint32_t randomEntityMaskWidth = 0;
    uint32_t randomDirMaskWidth    = 0;
};
```

Set them in the parse function from the parsed vector's size before `storeMaskWords`.

- [ ] **Step 5: Run the fixture suite**

Run: `make build-native && make tests`

Expected: 469/469 pass. If any fixture fails, the typical cause is a consumer that was iterating from 0 to `someBitVector.size()` but now has `session.game->wordCount` — and for a movement mask it should be `movementWordCount`. Walk through each failing fixture's command path.

- [ ] **Step 6: Run `perf_check.sh`**

Expected: `fast_replay_ms` either flat or slightly improved (we removed some allocations but most allocations came from the Pattern/Rule structures, which haven't been migrated yet).

- [ ] **Step 7: Commit**

```bash
git add native/src/core.hpp native/src/core.cpp perf_baseline.json
git commit -m "Migrate Replacement masks onto Game::maskArena

fast_replay_ms: <before> -> <after>"
```

---

## Task 8: Migrate `Pattern` Masks to Arena Offsets

**Files:**
- Modify: `native/src/core.hpp` (`struct Pattern`)
- Modify: `native/src/core.cpp` (parser + every consumer of a `Pattern` mask)

**Purpose:** `Pattern` is read in `matchesPatternAt` — a Phase-2 hot path — and each Pattern currently owns 4+ heap-allocated BitVectors plus a vector-of-BitVectors for `anyObjectsPresent`. This is the biggest single allocation win in Phase 1.

- [ ] **Step 1: Change `Pattern` in `core.hpp`**

Replace the existing `struct Pattern` body with:

```cpp
struct Pattern {
    enum class Kind { Ellipsis, CellPattern };
    Kind kind = Kind::CellPattern;

    MaskOffset objectsPresent   = kNullMaskOffset;
    MaskOffset objectsMissing   = kNullMaskOffset;
    MaskOffset movementsPresent = kNullMaskOffset;
    MaskOffset movementsMissing = kNullMaskOffset;

    // anyObjectsPresent is a variable-length list; store a run of offsets in
    // Game::anyObjectOffsets with (first, count) locator here.
    uint32_t anyObjectsFirst = 0;
    uint32_t anyObjectsCount = 0;

    std::optional<Replacement> replacement;
};
```

- [ ] **Step 2: Add `anyObjectOffsets` to `Game`**

In `core.hpp`, add to `struct Game`:

```cpp
std::vector<MaskOffset> anyObjectOffsets;  // referenced by Pattern::anyObjectsFirst/Count
```

- [ ] **Step 3: Update the `Pattern` parse function**

Replace lines like:
```cpp
pattern.objectsPresent = parseIntVector(requireField(object, "objects_present"));
```

With:
```cpp
pattern.objectsPresent = storeMaskWords(game, parseIntVector(requireField(object, "objects_present")));
pattern.objectsMissing = storeMaskWords(game, parseIntVector(requireField(object, "objects_missing")));
pattern.movementsPresent = storeMaskWords(game, parseIntVector(requireField(object, "movements_present")));
pattern.movementsMissing = storeMaskWords(game, parseIntVector(requireField(object, "movements_missing")));

pattern.anyObjectsFirst = static_cast<uint32_t>(game.anyObjectOffsets.size());
for (const auto& anyMask : requireField(object, "any_objects_present").asArray()) {
    game.anyObjectOffsets.push_back(storeMaskWords(game, parseIntVector(anyMask)));
}
pattern.anyObjectsCount = static_cast<uint32_t>(game.anyObjectOffsets.size()) - pattern.anyObjectsFirst;
```

- [ ] **Step 4: Update every consumer of `Pattern` mask fields**

Search `native/src/core.cpp` for `pattern.objectsPresent`, `pattern.objectsMissing`, etc. Every call site reading one of these fields now resolves via `maskAt(*game, pattern.objectsPresent).data[w]`, iterating 0 to `game.wordCount` (or `movementWordCount` for the movement fields).

For `anyObjectsPresent`, iterate:
```cpp
for (uint32_t i = 0; i < pattern.anyObjectsCount; ++i) {
    MaskRef any = maskAt(*session.game, session.game->anyObjectOffsets[pattern.anyObjectsFirst + i]);
    // ... use any.data[w] for w in [0, wordCount)
}
```

- [ ] **Step 5: Run the fixture suite**

Run: `make build-native && make tests`

Expected: 469/469 pass.

- [ ] **Step 6: Run `perf_check.sh`**

Expected: visible `fast_replay_ms` improvement (patterns are read in tight loops during rule evaluation, and removing their per-pattern allocations and indirections helps).

- [ ] **Step 7: Update `perf_baseline.json` and commit**

```bash
git add native/src/core.hpp native/src/core.cpp perf_baseline.json
git commit -m "Migrate Pattern masks onto Game::maskArena

fast_replay_ms: <before> -> <after>"
```

---

## Task 9: Migrate `Rule` Masks to Arena Offsets

**Files:**
- Modify: `native/src/core.hpp` (`struct Rule`)
- Modify: `native/src/core.cpp` (parser + every consumer of a `Rule` mask)

- [ ] **Step 1: Change `Rule` in `core.hpp`**

Replace mask-bearing members on `Rule`:

```cpp
struct Rule {
    int32_t direction = 0;
    bool hasReplacements = false;
    int32_t lineNumber = 0;
    std::vector<int32_t> ellipsisCount;
    int32_t groupNumber = 0;
    bool rigid = false;
    std::vector<RuleCommand> commands;
    bool isRandom = false;

    // Each cellRowMasks[i] is a run of wordCount MaskWords at arena offset
    // cellRowMaskOffsets[i]. Stored as a flat run in
    // Game::cellRowMaskOffsets with (first, count) pointing into it.
    uint32_t cellRowMasksFirst = 0;
    uint32_t cellRowMasksCount = 0;
    uint32_t cellRowMasksMovementsFirst = 0;
    uint32_t cellRowMasksMovementsCount = 0;

    MaskOffset ruleMask = kNullMaskOffset;

    std::vector<std::vector<Pattern>> patterns;  // flattened in Task 13
};
```

Add to `struct Game`:
```cpp
std::vector<MaskOffset> cellRowMaskOffsets;
std::vector<MaskOffset> cellRowMaskMovementsOffsets;
```

- [ ] **Step 2: Update `Rule` parse function**

Replace:
```cpp
for (const auto& rowMask : requireField(object, "cell_row_masks").asArray()) {
    rule.cellRowMasks.push_back(parseIntVector(rowMask));
}
```

With:
```cpp
rule.cellRowMasksFirst = static_cast<uint32_t>(game.cellRowMaskOffsets.size());
for (const auto& rowMask : requireField(object, "cell_row_masks").asArray()) {
    game.cellRowMaskOffsets.push_back(storeMaskWords(game, parseIntVector(rowMask)));
}
rule.cellRowMasksCount = static_cast<uint32_t>(game.cellRowMaskOffsets.size()) - rule.cellRowMasksFirst;
```

Mirror the same for `cellRowMasksMovements`. For `ruleMask`:

```cpp
rule.ruleMask = storeMaskWords(game, parseIntVector(requireField(object, "rule_mask")));
```

- [ ] **Step 3: Update every consumer**

Search for `rule.cellRowMasks`, `rule.cellRowMasksMovements`, `rule.ruleMask`. Every consumer resolves via `maskAt(*game, game->cellRowMaskOffsets[rule.cellRowMasksFirst + i])` patterns analogous to Task 8.

- [ ] **Step 4: Fixture suite**

Run: `make build-native && make tests`

Expected: 469/469 pass.

- [ ] **Step 5: `perf_check.sh` and commit**

```bash
git add native/src/core.hpp native/src/core.cpp perf_baseline.json
git commit -m "Migrate Rule masks onto Game::maskArena

fast_replay_ms: <before> -> <after>"
```

---

## Task 10: Migrate Game Top-Level Masks and Name-Keyed Mask Maps to the Arena

**Files:**
- Modify: `native/src/core.hpp` (`struct Game`)
- Modify: `native/src/core.cpp` (parser + consumers)

**Purpose:** `playerMask`, `layerMasks`, `objectMasks`, `aggregateMasks`, `glyphDict` all still own `BitVector` values directly. Migrate them to arena offsets, and convert the `std::map<std::string, BitVector>` members to sorted vectors of `{name, MaskOffset}` (touches only load-time code paths, so speed is not the goal — cache/allocation hygiene is).

- [ ] **Step 1: Change `Game` members in `core.hpp`**

Replace:

```cpp
std::vector<BitVector> layerMasks;
std::map<std::string, BitVector> objectMasks;
std::map<std::string, BitVector> aggregateMasks;
std::map<std::string, BitVector> glyphDict;
bool playerMaskAggregate = false;
BitVector playerMask;
// ...
std::vector<SoundMaskEntry> sfxCreationMasks;
std::vector<SoundMaskEntry> sfxDestructionMasks;
```

With:

```cpp
std::vector<MaskOffset> layerMaskOffsets;  // size == layerCount

// Name-keyed mask tables. Sorted by name at load so lookups are binary search.
struct NamedMaskEntry { std::string name; MaskOffset offset; };
std::vector<NamedMaskEntry> objectMaskTable;
std::vector<NamedMaskEntry> aggregateMaskTable;
std::vector<NamedMaskEntry> glyphMaskTable;

bool playerMaskAggregate = false;
MaskOffset playerMask = kNullMaskOffset;

// ...
std::vector<SoundMaskEntry> sfxCreationMasks;  // SoundMaskEntry also migrated in Step 2
std::vector<SoundMaskEntry> sfxDestructionMasks;
```

- [ ] **Step 2: Migrate `SoundMaskEntry` too**

```cpp
struct SoundMaskEntry {
    MaskOffset objectMask    = kNullMaskOffset;
    MaskOffset directionMask = kNullMaskOffset;
    uint32_t directionMaskWidth = 0;  // may differ from wordCount
    int32_t seed = 0;
};
```

- [ ] **Step 3: Update all the parse functions**

Each place that currently writes `game.playerMask = parseIntVector(...)` becomes `game.playerMask = storeMaskWords(game, parseIntVector(...))`. Each `game.objectMasks[name] = parseIntVector(...)` becomes a `NamedMaskEntry` push followed by (after the full parse loop) a single `std::sort` by name. Example:

```cpp
for (const auto& [name, value] : ...objectMasks in IR...) {
    game.objectMaskTable.push_back({ name, storeMaskWords(game, parseIntVector(value)) });
}
std::sort(game.objectMaskTable.begin(), game.objectMaskTable.end(),
          [](auto& a, auto& b){ return a.name < b.name; });
```

And add a lookup helper in the anonymous namespace:

```cpp
MaskOffset lookupMask(const std::vector<Game::NamedMaskEntry>& table,
                      std::string_view name) {
    auto it = std::lower_bound(table.begin(), table.end(), name,
        [](const Game::NamedMaskEntry& e, std::string_view n) { return e.name < n; });
    if (it == table.end() || it->name != name) return kNullMaskOffset;
    return it->offset;
}
```

- [ ] **Step 4: Update every consumer**

For every `game.playerMask`, `game.layerMasks[i]`, `game.objectMasks[name]`, etc., switch to the arena-based access. Use `lookupMask(game.objectMaskTable, name)` for the name-keyed ones.

- [ ] **Step 5: Fixture suite and perf_check**

Run: `make build-native && make tests && scripts/perf_check.sh`

Expected: 469/469 pass; small additional `fast_replay_ms` improvement.

- [ ] **Step 6: Commit**

```bash
git add native/src/core.hpp native/src/core.cpp perf_baseline.json
git commit -m "Migrate Game top-level masks and name-keyed maps to arena

fast_replay_ms: <before> -> <after>"
```

---

## Task 11: Delete the Now-Unused `BitVector` Type

**Files:**
- Modify: `native/src/core.hpp`
- Modify: `native/src/core.cpp` (any residual references)

**Purpose:** After Tasks 7–10, no struct field should hold a `BitVector` (= `std::vector<int32_t>`). Remove the typedef and confirm the migration is complete. If anything still references `BitVector`, that indicates an unmigrated hot-path call site.

- [ ] **Step 1: Delete the `using BitVector = std::vector<int32_t>;` line in `core.hpp`**

Remove line 28 (or wherever it now lives).

- [ ] **Step 2: Build**

Run: `make build-native`

Expected: Every compile error identifies a site that still uses the legacy type. Fix each one by migrating it to `MaskRef` / `MaskMut` / `MaskOffset` as in Tasks 7–10.

- [ ] **Step 3: Fixture suite**

Run: `make tests`

Expected: 469/469 pass.

- [ ] **Step 4: Commit**

```bash
git add native/src/core.hpp native/src/core.cpp
git commit -m "Delete BitVector typedef; all masks live in Game::maskArena"
```

---

## Task 12: Introduce `Session::sessionArena` for Transient Masks

**Files:**
- Modify: `native/src/core.hpp` (`struct Session`)
- Modify: `native/src/core.cpp` (accessors + consumers)

**Purpose:** Session-level transient masks (`rowMasks`, `columnMasks`, `boardMask`, `rowMovementMasks`, `columnMovementMasks`, `boardMovementMask`, `rigidGroupIndexMasks`, `rigidMovementAppliedMasks`, `pendingCreateMask`, `pendingDestroyMask`) currently live as independent `std::vector<int32_t>` members. Move them into one arena so they are contiguous and allocation-free after `loadLevel`.

- [ ] **Step 1: Change `Session` in `core.hpp`**

Replace the independent vector members with offsets into a single arena:

```cpp
struct Session {
    // ... existing members ...

    std::vector<MaskWord> sessionArena;

    // Offsets (in words) into sessionArena. Populated at loadLevel, stable
    // thereafter — these are never moved once allocated.
    uint32_t rowMasksOffset = 0;
    uint32_t columnMasksOffset = 0;
    uint32_t boardMaskOffset = 0;
    uint32_t rowMovementMasksOffset = 0;
    uint32_t columnMovementMasksOffset = 0;
    uint32_t boardMovementMaskOffset = 0;
    uint32_t rigidGroupIndexMasksOffset = 0;
    uint32_t rigidMovementAppliedMasksOffset = 0;
    uint32_t pendingCreateMaskOffset = 0;
    uint32_t pendingDestroyMaskOffset = 0;

    uint32_t rowMasksLen = 0;   // in words
    uint32_t columnMasksLen = 0;
    uint32_t boardMaskLen = 0;
    uint32_t rowMovementMasksLen = 0;
    uint32_t columnMovementMasksLen = 0;
    uint32_t boardMovementMaskLen = 0;
    uint32_t rigidGroupIndexMasksLen = 0;
    uint32_t rigidMovementAppliedMasksLen = 0;
    uint32_t pendingCreateMaskLen = 0;
    uint32_t pendingDestroyMaskLen = 0;

    // ...
};
```

Delete the old `std::vector<int32_t> rowMasks;` etc. lines.

- [ ] **Step 2: Size the arena in `loadLevel`**

Compute required total size from level dimensions + game word counts, then:
```cpp
session.sessionArena.assign(totalWords, 0);
session.rowMasksOffset = 0;
session.rowMasksLen    = height * wordCount;
session.columnMasksOffset = session.rowMasksOffset + session.rowMasksLen;
session.columnMasksLen    = width  * wordCount;
// ... same pattern for every offset, in declaration order
```

- [ ] **Step 3: Add helper accessors**

In `core.cpp` anonymous namespace:

```cpp
inline MaskMut sessionMasks(Session& s, uint32_t offset) {
    return MaskMut{ s.sessionArena.data() + offset };
}
inline MaskRef sessionMasks(const Session& s, uint32_t offset) {
    return MaskRef{ s.sessionArena.data() + offset };
}
```

- [ ] **Step 4: Update every consumer**

Every `session.rowMasks[i]` becomes `session.sessionArena[session.rowMasksOffset + i]` — or, cleaner, use accessors. Update similarly for every vector replaced in Step 1.

- [ ] **Step 5: Update `rebuildMasks`**

The scratch buffers introduced in Task 5 go away — `rebuildMasks` now writes directly into `session.sessionArena` runs via `std::fill` + loop, with the target locations computed via the offsets.

- [ ] **Step 6: Update undo/restart paths**

`UndoSnapshot` captures transient state. Replace its `std::vector<int32_t> liveMovements; rigidGroupIndexMasks; rigidMovementAppliedMasks;` with a single `std::vector<MaskWord> sessionArenaSnapshot;` captured as a whole-arena copy. Save/restore is now one `std::copy` per snapshot.

- [ ] **Step 7: Run the fixture suite**

Run: `make build-native && make tests`

Expected: 469/469 pass. Undo/restart fixtures are especially sensitive — any snapshot mismatch shows up immediately.

- [ ] **Step 8: `perf_check.sh`**

Expected: small `fast_replay_ms` improvement from reduced cache traffic (one contiguous region instead of 10 vectors).

- [ ] **Step 9: Commit**

```bash
git add native/src/core.hpp native/src/core.cpp perf_baseline.json
git commit -m "Move Session transient masks into a single sessionArena

fast_replay_ms: <before> -> <after>"
```

---

## Task 13: Flatten `Game::rules` / `Game::lateRules`

**Files:**
- Modify: `native/src/core.hpp` (`struct Game`, `struct Rule`)
- Modify: `native/src/core.cpp` (parser, every `applyRuleGroup` / `applyRuleGroups` site)

**Purpose:** Replace `std::vector<std::vector<Rule>>` with `std::vector<Rule>` + `std::vector<RuleGroup>`. Also flatten `Rule::patterns` from `std::vector<std::vector<Pattern>>` into a flat arena + per-rule locator. Cache-line contiguous rule iteration; enables Phase 2's template dispatch on word count.

- [ ] **Step 1: Change `Game` in `core.hpp`**

```cpp
struct RuleGroup {
    uint32_t first;   // index into Game::rules
    uint32_t count;
};

struct Game {
    // ...
    std::vector<Rule> rules;
    std::vector<RuleGroup> ruleGroups;

    std::vector<Rule> lateRules;
    std::vector<RuleGroup> lateRuleGroups;

    std::vector<Pattern> patternArena;  // flat store of all Pattern rows×cols
    // ...
};
```

Change `Rule::patterns` from `std::vector<std::vector<Pattern>>` to:

```cpp
struct Rule {
    // ... existing fields ...
    uint32_t patternsRowsFirst = 0;   // index into Game::patternRowLocators
    uint32_t patternsRowsCount = 0;
};

// In Game:
struct PatternRowLocator { uint32_t first; uint32_t count; };
std::vector<PatternRowLocator> patternRowLocators;  // indexes into patternArena
```

- [ ] **Step 2: Update the rules parser**

When the parser reads each nested rule group, flatten into the new layout:

```cpp
for (const auto& group : requireField(object, "rules").asArray()) {
    RuleGroup g{ static_cast<uint32_t>(game.rules.size()), 0 };
    for (const auto& ruleValue : group.asArray()) {
        Rule r = parseRule(game, ruleValue);  // parseRule now takes Game&
        game.rules.push_back(std::move(r));
    }
    g.count = static_cast<uint32_t>(game.rules.size()) - g.first;
    game.ruleGroups.push_back(g);
}
```

Inside `parseRule`, replace the `rule.patterns.push_back(rowVec)` pattern with:

```cpp
rule.patternsRowsFirst = static_cast<uint32_t>(game.patternRowLocators.size());
for (const auto& rowValue : requireField(object, "patterns").asArray()) {
    PatternRowLocator locator{ static_cast<uint32_t>(game.patternArena.size()), 0 };
    for (const auto& patternValue : rowValue.asArray()) {
        game.patternArena.push_back(parsePattern(game, patternValue));
    }
    locator.count = static_cast<uint32_t>(game.patternArena.size()) - locator.first;
    game.patternRowLocators.push_back(locator);
}
rule.patternsRowsCount = static_cast<uint32_t>(game.patternRowLocators.size()) - rule.patternsRowsFirst;
```

- [ ] **Step 3: Update every consumer**

Search for iterators over `game.rules` (which is now flat) and `rule.patterns` (which now resolves via the locator arrays). Iteration patterns:

```cpp
// was: for (const auto& group : game->rules) { for (const auto& rule : group) { ... } }
for (const auto& group : game->ruleGroups) {
    for (uint32_t i = 0; i < group.count; ++i) {
        const Rule& rule = game->rules[group.first + i];
        // ...
    }
}

// was: for (const auto& row : rule.patterns) { for (const auto& pat : row) { ... } }
for (uint32_t r = 0; r < rule.patternsRowsCount; ++r) {
    PatternRowLocator loc = game->patternRowLocators[rule.patternsRowsFirst + r];
    for (uint32_t c = 0; c < loc.count; ++c) {
        const Pattern& pat = game->patternArena[loc.first + c];
        // ...
    }
}
```

Mirror for `lateRules` / `lateRuleGroups`.

- [ ] **Step 4: Fixture suite**

Run: `make build-native && make tests`

Expected: 469/469 pass.

- [ ] **Step 5: `perf_check.sh`**

Expected: modest `fast_replay_ms` improvement from cache-coherent rule iteration.

- [ ] **Step 6: Commit**

```bash
git add native/src/core.hpp native/src/core.cpp perf_baseline.json
git commit -m "Flatten Game::rules and Rule::patterns to contiguous arenas

fast_replay_ms: <before> -> <after>"
```

---

## Task 14: Phase 1 Exit — Full Measurement and Gate

**Files:**
- Modify: `perf_baseline.json`
- Create: `docs/superpowers/plans/2026-04-22-native-perf-phase1-results.md`

**Purpose:** Verify the Phase 1 gate from the spec (§1 success metrics) is hit, record the final numbers, and decide whether Phase 2 is worth writing.

- [ ] **Step 1: Regenerate `profile_stats.txt` for the canonical record**

Run: `bash src/tests/profile_native_trace_suite.sh`

Expected: fresh `profile_stats.txt` at repo root. Review the hot-stack section — `rebuildMasks` is still present (deleted in Phase 2, not Phase 1); `std::vector<int>::operator[]` and the libc++ hardening annotations should both be gone from the hot list.

- [ ] **Step 2: Run `perf_check.sh` for the phase-exit numbers**

Run: `scripts/perf_check.sh`

Expected gates (from spec §1):
- `fast_replay_ms` ≤ 30,000
- `wall_ms` ≤ 40,000
- `trace_json_parse_ms + ir_miss_ms` ≤ 1,000

If `fast_replay_ms` > 30,000: review the hot stacks for anything obviously still on the table (unmigrated mask access? arena offsets still going through an allocating accessor?). Do not move on to Phase 2 without hitting the gate.

- [ ] **Step 3: Write the results note**

Create `docs/superpowers/plans/2026-04-22-native-perf-phase1-results.md` with the before/after numbers for all four metrics, the relevant commit hashes, and a short paragraph on whether Phase 2 is needed to reach the "crush it" ceiling. Include the top 10 entries from the new hot-stacks section — these inform Phase 2 task selection.

- [ ] **Step 4: Final commit**

```bash
git add perf_baseline.json profile_stats.txt docs/superpowers/plans/2026-04-22-native-perf-phase1-results.md
git commit -m "Phase 1 exit: native perf gate hit

fast_replay_ms:      114,731 -> <final>
wall_ms:             143,341 -> <final>
trace_json+ir_miss:   28,089 -> <final>

See docs/superpowers/plans/2026-04-22-native-perf-phase1-results.md
for detailed before/after and Phase 2 recommendation."
```

---

## Self-Review (plan author)

### Spec coverage

| Spec section                                      | Plan task(s)     |
| ------------------------------------------------- | ---------------- |
| §1 Success metrics + regression gate scaffolding  | Task 1           |
| §3.1 Build configuration                          | Task 2           |
| §3.2 MaskWord / MaskRef / MaskMut                 | Task 6           |
| §3.2 Game::maskArena                              | Task 6           |
| §3.2 Replacement / Pattern / Rule migration       | Tasks 7, 8, 9    |
| §3.2 Game top-level + name-keyed masks            | Task 10          |
| §3.2 Delete BitVector                             | Task 11          |
| §3.2 Session arena                                | Task 12          |
| §3.3 Flatten Game maps                            | Task 10          |
| §3.3 Flatten rules nesting + patterns             | Task 13          |
| §3.4 Scratch buffers on Session                   | Task 5           |
| §3.5 simdjson replacement                         | Tasks 3, 4       |
| §3.5 Flat-binary IR (optional late-Phase-1)       | **Deferred.** Tracked as Phase 2 starter if measurements after Phase 1 don't yet hit the "crush" ceiling. Load cost is already <1s after Task 4, so the value is marginal; confirming in Task 14. |
| §1 Phase 1 exit criteria                          | Task 14          |

### Placeholder scan

- No "TBD"/"TODO" inside step code blocks.
- Task 4 Step 2 has a comment pointing at code that must be copied from the existing `json.cpp` rather than re-authored — the existing `Value` ctor/accessor bodies are not re-quoted to keep the plan concise; the engineer must copy them verbatim. This is a documented copy, not a placeholder.
- Task 5 Step 1 notes that the scratch fields should only include the ones `rebuildMasks` actually uses — this is a "read the current code and keep what applies" instruction, not a placeholder. Fine.
- Task 6 Step 4 notes the exact field name for the per-game stride may differ; the engineer is told to use the current IR loader's field name and verify with an assertion. This is a documented investigation step, not a placeholder.

### Type/name consistency

- `MaskWord` (int32_t in Phase 1) / `MaskRef` / `MaskMut` / `MaskOffset` / `kNullMaskOffset` names used consistently Tasks 6–13.
- `storeMaskWords(game, ...)` / `maskAt(game, offset)` / `anyBitsSet(MaskRef, wordCount)` / `anyBitsInCommon(MaskRef, MaskRef, wordCount)` appear with the same signatures across tasks.
- `sessionArena` / `rebuildMasksScratch*` naming consistent.
- `RuleGroup { uint32_t first; uint32_t count; }` and `PatternRowLocator { uint32_t first; uint32_t count; }` use the same field-name convention.
- `perf_check.sh` / `perf_baseline.json` / `scripts/perf_extract.awk` paths used consistently.

### Known gaps

- Task 6 Step 4 invariant check (`game.playerMask.size() == game.wordCount`) cannot run after Task 10 since `playerMask` is already a `MaskOffset` by then. This is a Task-6-only sanity check, documented as such.
- The `Replacement::randomEntityMaskWidth` / `randomDirMaskWidth` fields added in Task 7 are required because those two masks can have a different width than `wordCount`. Phase 2 may be able to collapse them into `wordCount` once IR format is understood better; left as-is for now.
