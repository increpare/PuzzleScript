# Design: FullState split — MetaGameState, canonical compact occupancy, scratch, TurnResult

**Status:** Draft for implementation planning  
**Date:** 2026-04-28  
**Scope:** Runtime-wide refactor (`native/`). Interactive play, interpreter, generated rules, C API, solver/generator integration.

## 1. Problem

`FullState` today bundles immutable game reference, modal/progression state, canonical level occupancy (cell-major `liveLevel.objects`), movement buffers, derived masks, rule scratch, undo, audio queues, and flags. That forces **large deep copies** on solver branches and blurs **what belongs in a search node** vs **what is execution ephemera**.

## 2. Goals

- **Single canonical occupancy layout:** object-major compact bitsets (same *semantic* shape as today’s solver `CompactState::objectBits` in `native/src/solver/main.cpp`), used **everywhere** the engine reads/writes the board — not a parallel cell-major `LevelTemplate::objects` as the source of truth.
- **Clear separation:**
  - **`MetaGameState`:** mode, progression, seeds, serialized level, flickscreen data, **restart/checkpoint board snapshots**, and other fields **not** mutated directly by rules but **updated from `TurnResult` / transition logic**. Not part of solver node identity; **not copied per search node**.
  - **`BoardOccupancy` (persistent board):** compact occupancy + **RNG state** used by random rules (must remain in node identity when randomness matters).
  - **`Scratch`:** movements, row/column/board masks, `objectCellBits` (if retained), dirty flags, replacement/ellipsis temporaries, optional cached **level view** for hot paths.
- **Turn API:** `turn(game, meta, occupancy, scratch, input, options) -> TurnResult` (names TBD). **Caller-visible side effects** (audio/UI events, step flags) live in **`TurnResult`**, not as long-lived fields on core state. Intra-turn state (`pendingAgain` across internal passes) remains inside the turn implementation until commit.
- **`FullState`:** nested composition only — e.g. `{ shared_ptr<const Game> game; MetaGameState meta; BoardOccupancy occupancy; Scratch scratch; }` — **still one C++ type** passed as `FullState&` to generated code during migration, with shims as needed.

## 3. Non-goals (initial milestone)

- Parallel search with **shared** scratch pools (possible later).
- Changing **hash key semantics** for random games (RNG stays in occupancy side until an explicit deterministic-only mode exists).
- Rewriting `Game` / compiler beyond what is required to compile and test.

## 4. MetaGameState

**Responsibility:** Everything that is **not** “which objects occupy which cells” and **not** scratch — the slice currently approximated by `PreparedFullState` in `native/src/runtime/core.hpp` (lines 127–148): level index and target, title/text modes, selections, winning, messages, `loadedLevelSeed`, prepared-level RNG fields (to be **deduplicated** with `FullState::RandomState` where redundant), `oldFlickscreenDat`, **level template metadata**, `serializedLevel`, and **`RestartSnapshot`** / checkpoint target boards.

**Rules:** Rule application does **not** mutate `MetaGameState` directly. **`turn()`** applies **`TurnResult`** (and internal helpers) to update meta.

**Restart / checkpoint:** `RestartSnapshot` (today: `width`, `height`, cell-major `objects`, `oldFlickscreenDat`) moves under **meta**. Board bytes should migrate to the **same compact occupancy layout** as the main board. These fields are **irrelevant to typical solvers/generators**; search keeps a **single fixed meta** for the duration of a level solve.

**Solver/generator:** Do **not** copy `MetaGameState` per frontier node. Keep **one** meta + **one** `Game`; nodes store **only** `BoardOccupancy` (and possibly rename today’s `CompactState` to avoid confusion with the runtime type).

## 5. BoardOccupancy (canonical)

- **Layout:** Object-major cell bitsets: for each object id, one bit per cell (packed in `uint64_t` words), sized from `objectCount × cellWordCount`, matching the conversion logic in `compactStateFromFullState` / `materializeCompactStateIntoFullState` in `native/src/solver/main.cpp`.
- **RNG:** `FullState::RandomState` (or merged equivalent) lives with occupancy for **runtime identity**; it remains in **search keys** when random rules matter.
- **Not in occupancy:** Title screen, message mode, level index, restart snapshots, etc.

## 6. Derived “live level”

**Concept:** Static fields (`width`, `height`, `layerCount`, `isMessage`, `message`, `lineNumber`) come from **`Game::levels[meta.currentLevelIndex]`** (or the active template meta points at). **Cell contents** come **only** from **`BoardOccupancy`**.

**Implementation:** Prefer a **scratch-resident `LevelView`** (or accessors) built at turn start / after level transition — avoid a second persistent copy of `LevelTemplate::objects` once migration is complete.

**Edge cases:** Message levels may ignore occupancy; level transitions reset occupancy from load/restart snapshot via **meta + `TurnResult`**, not from the previous board alone.

## 7. Scratch

- **Movement:** `liveMovements` and movement-derived masks (cell-major is acceptable).
- **Derived object masks:** row/column/board, dirty vectors, replacement/ellipsis buffers, pending SFX masks if still needed during a turn.
- **Clone semantics:** When forking a search branch, **copy occupancy (and RNG)** only; **reuse or resize** scratch, **do not** memcpy large scratch vectors from parent.

## 8. TurnResult

Hold or extend what is today surfaced through **`ps_step_result`** plus **audio/UI event lists** currently stored on `FullState` (`lastAudioEvents`, `lastUiAudioEvents` in `core.cpp`). **Apply** meta transitions from the result; **do not** keep duplicate long-lived queues on session if the caller consumes them on return.

## 9. Migration strategy (phased)

1. Introduce **`MetaGameState`**, **`BoardOccupancy`**, **`Scratch`** (and `TurnResult` envelope) alongside existing fields; add **accessors** for cell occupancy in compact form.
2. Route **new** reads/writes through accessors; keep **`liveLevel.objects`** as a **deprecated cache** optional until parity tests pass.
3. Migrate **interpreter** paths in `core.cpp`, then **generated** rule code in `native/src/cli/main.cpp`, then **compact bridge** / C API.
4. Remove cell-major **authoritative** occupancy from `LevelTemplate` / `liveLevel`; **restart** meta uses compact layout.
5. Tighten **C API** and naming (`CompactState` in solver vs runtime `BoardOccupancy`).

## 10. Testing / correctness

- `node src/tests/run_tests_node.js` (JS corpus) unchanged for project-wide behavior.
- Native: `make compact_turn_simulation_tests`, solver parity targets (`solver_compact_parity`, smoke), any existing native perf smoke.
- Each phase: compare **TurnResult + occupancy hash** (or full serialization) against baseline.

## 11. Risks

- **Blast radius:** Generated `FullState&` code and large `core.cpp` surface.
- **Locality:** Object-major vs cell-major hot paths; mitigate with accessors and optional tile-side caches in scratch if profiling requires.
- **RNG duplication:** Clean up `PreparedFullState` vs `FullState::RandomState` during meta split.

## 12. Open decisions

- Exact **names** (`BoardOccupancy` vs reusing `CompactState` at runtime — avoid name clash with solver struct).
- Whether **`UndoSnapshot`** stores compact occupancy only + meta pointer or full meta snapshot for playable undo.
- **`simd` / backend** field: meta vs scratch vs `FullState` root.

---

## Self-review (2026-04-28)

- [x] No unresolved “TBD” in critical paths; open decisions listed explicitly.
- [x] Consistent: solver nodes = occupancy + RNG; meta fixed per level search; restart in meta, solver-irrelevant.
- [x] Scope: single design; implementation can still be phased.
- [x] Ambiguity: restart snapshot **migrates to compact layout** (explicit).
