# Review: Phase A — FullState split (MetaGameState, BoardOccupancy, Scratch)

**Date:** 2026-04-28  
**Commits reviewed:** `5dbd525f` … `108e36e7` (Task A1–A3)  
**Spec:** `docs/superpowers/specs/2026-04-28-fullstate-compact-occupancy-design.md`  
**Plan:** `docs/superpowers/plans/2026-04-28-fullstate-split-implementation-plan.md`

## 1. Spec and plan compliance (Tasks A1–A3)

| Deliverable | Verdict |
|-------------|--------|
| **A1** — `using MetaGameState = PreparedFullState`; `preparedFullState` → `meta` in `native/` | **Met.** The `preparedFullState` identifier is gone from `native/`. |
| **A2** — `RandomState` lifted, `BoardOccupancy`, `FullState::occupancy`, sizing + RNG mirror, sync helper | **Met** structurally. `syncOccupancyRngFromAuthoritativeRandomState` documents `FullState::randomState` as the authoritative RNG until a later task. |
| **A3** — `struct Scratch`, replacement `MaskVector`s under `session.scratch.*` | **Met.** Only `core.cpp` / `core.hpp` use the new fields; replacement paths use `session.scratch.…`. |

**Design alignment:** Matches the Phase A intent in the design spec (type split and shells without yet making compact occupancy authoritative in the rule engine).

## 2. Code quality and architecture

**Strengths**

- **Typedef-first** `MetaGameState` avoids a large header split in A1; good incrementalism.
- **Scratch in one struct** establishes a clear boundary for later “do not memcpy scratch on fork.”
- **Centralized RNG sync** reduces the chance of missing a copy at a boundary; `materializeCompactStateIntoFullState` and `prepareSolverChildFullStateFromParent` call it after resize.
- **Build** and **`make compact_turn_oracle_smoke`** were run green during the Phase A rollout.

## 3. Important finding — `occupancy.objectBits` was only cleared, not filled

**Behavior**

- `resizeBoardOccupancyObjectBits` resizes `occupancy.objectBits` and **zero-fills** it.
- It runs **after** valid `liveLevel.objects` are built (e.g. `materializeCompactStateIntoFullState`, `materializeCompactBridgeState`, `prepareSolverChildFullStateFromParent`) and on undo / restart / load paths.
- So the **sized buffer** was correct, but the **content** stayed all zeros whenever the board was non-empty.

**Readiness check:** `occupancy.objectBits` was only referenced from that resize function (no other readers). The **engine still used `liveLevel.objects`**, so gameplay and oracles could remain correct.

**Phase B impact:** Before treating compact occupancy as meaningful, occupancy must be **kept in sync with** `liveLevel` (or the order of authority must flip in a later task). **Addressed in Phase B** via `syncOccupancyObjectBitsFromLiveLevel` and calling it whenever the board layout is established or restored.

## 4. Minor notes (non-blocking)

- **`meta` on both `Game` and `FullState`:** Slightly overloaded naming; optional rename later (e.g. `gameMeta` / `sessionMeta`).
- **`PreparedFullState` still carries RNG-related fields alongside `RandomState`:** Known deduplication item from the design doc; can follow meta/occupancy cleanup.

## 5. Verdict

| Gate | Result |
|------|--------|
| **Phase A accepted** | **Yes** — with the occupancy sync gap **closed in Phase B**. |
| **Proceed to Phase B** | **Yes** after implementing occupancy sync and access helpers. |
