# FullState split (MetaGameState, BoardOccupancy, Scratch, TurnResult) — Phased Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the native runtime so `FullState` is nested composition (`Game` + `MetaGameState` + `BoardOccupancy` + `Scratch`), with **canonical object-major compact occupancy** for the board, **modal/progression state** in meta (including restart snapshots in compact form), **turn-visible effects** in a `TurnResult`-style envelope, and **solver nodes** copying only occupancy+RNG — without changing PuzzleScript semantics (JS corpus + native compact/simulation gates stay green).

**Architecture:** (1) **Carve types** out of `core.hpp` into focused headers; (2) **migrate `preparedFullState` → `MetaGameState`** (mechanical rename + field moves from `FullState` where needed); (3) **lift occupancy to `BoardOccupancy`** and route all object cell reads/writes through a **single access layer** until `LevelTemplate::objects` is no longer authoritative; (4) **bucket derived+ephemeral vectors** into `Scratch` and fix **clone/fork** in the solver to avoid memcpy of scratch; (5) **route audio/UI** through return structs while keeping `ps_step_result` compatible at C boundaries; (6) **rename solver-local `CompactState`** → e.g. `SearchNodeState` after runtime types stabilize.

**Tech Stack:** C++17, native Makefile/CMake build under `native/`, QUnit corpus via `src/tests/run_tests_node.js`, native targets `compact_turn_simulation_tests`, solver parity Makefile targets.

**Spec:** `docs/superpowers/specs/2026-04-28-fullstate-compact-occupancy-design.md`.

---

## File structure (touch map)

| Area | Primary files |
|------|----------------|
| Types / `FullState` layout | `native/src/runtime/core.hpp`, **new** `native/src/runtime/session_types.hpp` (or split further if files exceed ~800 LOC) |
| Interpreter / masks / step | `native/src/runtime/core.cpp` |
| Compact bridge / oracle | `native/src/runtime/compiled_rules.hpp`, `native/src/runtime/compiled_rules.cpp` |
| C API | `native/src/runtime/c_api.cpp` |
| Solver search nodes | `native/src/solver/main.cpp`, possibly `native/src/search/search_common.hpp` |
| Code generation | `native/src/cli/main.cpp` (generated `FullState& session` emission) |
| Tests / harness | `Makefile` (already defines compact/sim targets); `src/tests/run_tests_node.js` unchanged for JS parity |

---

## Phase A — Types + mechanical rename (compile-first, minimal behavior risk)

### Task A1: `MetaGameState` alias + rename member (`preparedFullState` → `meta`)

**Files:**
- Modify: `native/src/runtime/core.hpp`
- Modify: all `native/**/*.cpp`, `native/**/*.hpp` that reference `preparedFullState`

**Purpose:** Establish the **`MetaGameState`** name without moving struct definitions yet (avoids circular includes with `MaskVector`). Later phases can move **`MetaGameState`** into its own header once dependencies are untangled.

- [ ] **Step 1: Typedef alias**

In `core.hpp`, immediately after `struct PreparedFullState { ... };`, add:

```cpp
using MetaGameState = PreparedFullState;
```

- [ ] **Step 2: Rename `FullState` member**

Replace `PreparedFullState preparedFullState` with `MetaGameState meta` inside `struct FullState`.

- [ ] **Step 3: Repo-wide mechanical rename**

Run ripgrep-driven replace in **`native/` only**:

```bash
rg -l 'preparedFullState' native --glob '*.cpp' --glob '*.hpp'
```

Replace identifier `preparedFullState` → `meta`. **Preserve** external JSON keys if the CLI emits stable benchmark JSON — inspect before replacing:

```bash
rg 'preparedFullState|prepared_full' native/src/cli/main.cpp
```

- [ ] **Step 4: Build**

From repo root, use the project’s usual native target (see root `Makefile`; often `make build`):

```bash
cd /Users/stephenlavelle/Documents/GitHub/PuzzleScript && make build
```

Expected: successful link.

- [ ] **Step 5: Commit**

```bash
git add native/src/runtime/core.hpp native/
git commit -m "refactor(native): rename preparedFullState to meta (MetaGameState alias)"
```

---

### Task A2: Introduce `BoardOccupancy` shell + RNG placement decision

**Files:**
- Modify: `native/src/runtime/core.hpp`
- Modify: `native/src/runtime/core.cpp` (constructors / zero-init only)

**Purpose:** Reserve `BoardOccupancy` as `{ std::vector<uint64_t> objectBits; RandomState rng; }` (exact RNG type reuse `FullState::RandomState`). **Phase A:** duplicate RNG **temporarily**: keep existing `FullState::randomState` and sync in **one place** (`createFullState` / loadLevel) until Task B removes duplication.

- [ ] **Step 1: Define struct** (in `core.hpp` near `FullState`)

```cpp
struct BoardOccupancy {
    std::vector<uint64_t> objectBits;
    RandomState rng;  // reuse nested RandomState from FullState scope — lift RandomState to namespace scope first if needed
};
```

If `RandomState` is nested inside `FullState`, either move `struct RandomState` to namespace scope in `core.hpp` **above** `BoardOccupancy`, or duplicate minimal RNG fields inside `BoardOccupancy` and migrate later (**prefer moving `RandomState` out once**).

- [ ] **Step 2: Add member** `BoardOccupancy occupancy;` to `FullState`.

- [ ] **Step 3: Ensure sizing hooks**

Find `createFullState`, `loadLevel`, `materializeCompactStateIntoFullState`, `compactStateFromFullState` and add **`occupancy.objectBits.resize(...)`** alongside **`liveLevel.objects`** sizing (same dimensions). Fill zeros.

- [ ] **Step 4: Build + smoke**

```bash
cd /Users/stephenlavelle/Documents/GitHub/PuzzleScript && make build
```

- [ ] **Step 5: Commit**

```bash
git commit -am "refactor(native): add BoardOccupancy shell alongside liveLevel"
```

---

### Task A3: Introduce `Scratch` shell (subset move)

**Files:**
- Modify: `native/src/runtime/core.hpp`
- Modify: `native/src/runtime/core.cpp` (member access paths)

**Purpose:** Group scratch vectors under `struct Scratch { ... };` **without** changing algorithms yet. Pick **one** cohesive bundle first (e.g. all `replacement*` scratch vectors from `FullState`), fix compile.

- [ ] **Step 1: Define `Scratch`** containing moved fields from `FullState` lines ~397–411 (`replacementObjectsClearScratch`, …).

- [ ] **Step 2: Replace uses** `session.replacementObjectsClearScratch` → `session.scratch.replacementObjectsClearScratch` via scoped ripgrep replace.

- [ ] **Step 3: Build + native simulation smoke**

```bash
make compact_turn_oracle_smoke
```

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git commit -am "refactor(native): nest replacement scratch vectors under Scratch"
```

---

## Phase B — Occupancy access layer + invariant checks

### Task B1: Centralize compact ↔ cell-major helpers

**Files:**
- Create: `native/src/runtime/occupancy_access.hpp`
- Create: `native/src/runtime/occupancy_access.cpp` (if non-inline)
- Modify: `native/src/solver/main.cpp` — eventually delegate conversion to shared helpers (avoid drift).

**Purpose:** Single implementation for sizing `objectBits`, indexing `(objectId, tileIndex)`, and converting **to/from** `liveLevel.objects` stride layout (`compactStateFromFullState` / `materializeCompactStateIntoFullState` logic lifted from `solver/main.cpp`).

- [ ] **Step 1: Implement**

Functions (names illustrative):

- `size_t occupancyWordCount(int32_t tileCount);`
- `void resizeOccupancy(BoardOccupancy&, const Game&, int32_t tileCount);`
- `void syncOccupancyFromLiveLevel(FullState&);` — fills `occupancy.objectBits` from `liveLevel.objects`
- `void syncLiveLevelFromOccupancy(FullState&);` — fills `liveLevel.objects` from `occupancy.objectBits`

Use existing loops from `compactStateFromFullState` / inverse in `materializeCompactStateIntoFullState` as **golden reference**.

- [ ] **Step 2: Dev-only assert**

After hot mutations in `setCellObjects`-style paths (locate in `core.cpp`), `#ifndef NDEBUG` call both directions and **memcmp** cell-major vs derived up to tileCount×stride.

- [ ] **Step 3: Tests**

```bash
make compact_turn_simulation_tests
```

Expected: PASS (same as baseline).

- [ ] **Step 4: Commit**

```bash
git add native/src/runtime/occupancy_access.cpp native/src/runtime/occupancy_access.hpp native/Makefile.native # if include paths changed
git commit -m "feat(native): shared occupancy sync helpers + debug invariant"
```

---

### Task B2: Dual authoritative phase ends — pick compact writes first on critical paths

**Purpose:** Flip writes rule/replacement paths to update **`occupancy.objectBits` first**, then derive `liveLevel.objects`, OR vice versa — **pick one ordering** documented in `occupancy_access.hpp` header comment. Complete migration for:

- Player movement / object mutations in `core.cpp`
- Rule application entry points (`applyReplacementAt`, etc.)

Track progress with checklist in PR description until **grep** shows no direct `liveLevel.objects[tile * stride + w] |=` outside accessors.

- [ ] **Step 1:** Identify mutation sites:

```bash
rg 'liveLevel\.objects' native/src/runtime/core.cpp native/src/runtime/compiled_rules.cpp
```

- [ ] **Step 2:** Migrate highest-frequency paths first (measure with counters already in engine if present).

- [ ] **Step 3:** Gate

```bash
make compact_turn_simulation_tests
node src/tests/run_tests_node.js --sim-only
```

- [ ] **Step 4: Commit** per cohesive subdirectory (core vs compiled_rules).

---

## Phase C — Generated rule codegen (`cli/main.cpp`)

### Task C1: Redirect emitted code to accessors

**Files:**
- Modify: `native/src/cli/main.cpp` (string templates emitting `session.liveLevel.objects`)

**Purpose:** Generated C++ must call **`setCellObjects` / readers** (whatever the Phase B API stabilizes on) instead of raw vector indexing.

- [ ] **Step 1:** Search emission patterns:

```bash
rg 'liveLevel\.objects' native/src/cli/main.cpp | head -50
```

- [ ] **Step 2:** For each generator branch, emit accessor calls. Rebuild generated fixtures / re-run codegen pipeline per repo convention (`node compile.js` does not apply to native — native codegen runs when building games).

- [ ] **Step 3:**

```bash
make compiled_rules_simulation_suite_coverage
```
(or the narrowest native rule suite Makefile exposes)

- [ ] **Step 4: Commit**

```bash
git commit -am "refactor(codegen): emit occupancy accessors instead of raw liveLevel.objects"
```

---

## Phase D — Meta restart snapshots + strip authoritative cell-major

### Task D1: Restart snapshot compact layout

**Files:**
- Modify: `native/src/runtime/core.hpp` (`RestartSnapshot`)
- Modify: `native/src/runtime/core.cpp` (`restart`, `restoreRestartTarget`, undo snapshots)

**Purpose:** Replace `RestartSnapshot::objects` MaskVector with **`std::vector<uint64_t> restartObjectBits`** (same packing as `BoardOccupancy`) **or** nested `BoardOccupancy`. Migrate `restoreRestartTarget` and undo paths.

- [ ] **Step 1:** Structural change + migrate callers.

- [ ] **Step 2:**

```bash
node src/tests/run_tests_node.js
```
(use a substring filter only if targeted restart regressions exist in test names)

```bash
make compact_turn_simulation_tests
```

- [ ] **Step 3: Commit**

---

### Task D2: Remove `LevelTemplate::objects` authority

**Purpose:** After grep confirms no reads of authoritative cell-major except serializers:

- Delete or reserve-empty `liveLevel.objects` during level play path.
- Keep export/test serialization paths converting **from compact**.

Verification:

```bash
rg 'liveLevel\.objects' native/src/runtime/
make compact_turn_simulation_tests
```

---

## Phase E — TurnResult envelope + audio queues

### Task E1: Introduce `TurnResult` / extend `ps_step_result`

**Files:**
- Modify: `native/src/runtime/core.hpp`, `native/src/runtime/core.cpp`
- Modify: `native/src/runtime/c_api.cpp`
- Modify: `native/include/puzzlescript/puzzlescript.h` if C struct fields needed

**Purpose:** Move **`lastAudioEvents` / `lastUiAudioEvents`** off `FullState` into per-turn outputs; C API copies into caller-owned buffers **before return** (preserve ABI).

- [ ] **Step 1:** Define `struct TurnResult { ps_step_result core; std::vector<ps_audio_event> audio; ... };`

- [ ] **Step 2:** Implement `turn()` returning `TurnResult`; shim `step()`/`ps_step_result` wrappers extracting `.core`.

- [ ] **Step 3:**

```bash
make build
node src/tests/run_tests_node.js --verbose  # spot-check audio-heavy games if filter exists
```

---

## Phase F — Solver fork + naming

### Task F1: Solver scratch reuse

**Files:**
- Modify: `native/src/solver/main.cpp`

**Purpose:** Stop constructing **`FullState(*initial)` three times** when scratch buckets can be reset between edges. Pass **`Scratch&`** reset or allocate once per worker.

- [ ] **Step 1:** Profile current `cloneNs` timing fields in solver JSON output (baseline).

- [ ] **Step 2:** Implement reset-on-edge.

- [ ] **Step 3:**

```bash
make solver_compact_parity_smoke
```

---

### Task F2: Rename solver `CompactState` → `SearchNodeState`

**Files:**
- Modify: `native/src/solver/main.cpp`, any headers referencing solver compact struct

**Purpose:** Avoid name collision with runtime **compact occupancy** terminology.

- [ ] **Step 1:** Rename struct + functions `compactStateFromFullState` clarity (`searchNodeFromFullState` optional).

- [ ] **Step 2:** Build + solver smoke.

---

## Verification matrix (run each phase exit)

| Gate | Command |
|------|---------|
| JS engine corpus | `node src/tests/run_tests_node.js` |
| Native compact oracle | `make compact_turn_simulation_tests` |
| Solver parity | `make solver_compact_parity_smoke` |
| Full native smoke | `make smoke` or `make build` + targeted binaries per Makefile |

---

## Self-review (plan vs spec)

| Spec § | Covered by |
|--------|------------|
| Canonical compact occupancy | Tasks B*, C*, D2 |
| MetaGameState + restart in meta | Task A1 (rename path), D1 |
| Scratch clone semantics | A3, F1 |
| TurnResult | E1 |
| Solver copies occupancy only | F1; naming F2 |
| Derived live level | B access layer + docs in code comments |

**Placeholder scan:** No "TBD" tasks; open decisions from spec §12 resolved during Phase A/B (**RandomState** lifted to namespace scope in Task A2; **UndoSnapshot** compact addressed alongside D1; **simd/backend** stays on `FullState` root until Scratch migration proves otherwise).

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-04-28-fullstate-split-implementation-plan.md`.

**Execution options:**

1. **Subagent-driven** — one task per agent, review between tasks (recommended for blast radius).
2. **Inline** — batched tasks with human checkpoints after each phase A–F.

Which approach do you want for execution?
