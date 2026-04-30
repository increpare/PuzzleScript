# Compact Interpreter Board Migration Plan

## Summary

Move the native interpreter to use `PersistentLevelState::board.objects` as the
authoritative cell-major board. `Scratch::interpreterBoard` should stop being
the interpreter's live board authority and either become a temporary migration
mirror or disappear once call sites are converted.

This is the next architectural cleanup before deeper compiler optimization. It
should make interpreter, compact compiler, solver, C API, and oracle paths agree
on one persistent board representation, reducing sync/materialization work and
making future benchmarks compare interpreter dispatch against generated code
rather than different state layouts.

## Key Changes

- Retarget runtime board helpers so interpreter reads and writes go through
  `PersistentLevelState::board.objects`.
  - `getCellObjectsPtr`, `setCellObjects`, `setCellObjectsFromWords`,
    `compiledRuleCellObjects`, serialization, restart/checkpoint, win checks,
    and rule matching should all observe the same persistent board.
  - Keep object-cell indexes, row/column/board masks, live movements, rigid
    masks, replacement scratch, and dirty flags in `Scratch`.

- Replace sync-style APIs with board-authority APIs.
  - `setInterpreterBoardObjectsFromCellMajor` should become a compatibility
    loader that writes `session.levelState.board.objects`.
  - `copyInterpreterBoardObjectsAsCellMajor` should read
    `session.levelState.board.objects`.
  - `syncPersistentLevelStateFromScratch` and `syncPersistentBoardFromScratch`
    should become no-ops or be deleted once all callers no longer need them.

- Migrate solver and compact bridge plumbing.
  - `persistentLevelStateFromFullState` should copy from
    `session.levelState.board.objects`, not `scratch.interpreterBoard.objects`.
  - `materializePersistentLevelStateIntoFullState` should install the compact
    board by assigning `session.levelState.board.objects`.
  - Compact bridge copy-back should no longer copy from interpreter scratch.

- Remove `Scratch::interpreterBoard` after the migration is complete.
  - During the transition, it may remain only as a compatibility mirror for
    unconverted call sites.
  - The final state should have exactly one persistent board authority:
    `PersistentLevelState::board.objects`.

## Implementation Order

1. Add or retarget the lowest-level object-cell access helpers to operate on
   `session.levelState.board.objects`; preserve all existing dirty-mask and
   object-cell-index invalidation behavior.
2. Convert interpreter rule matching, replacements, movement resolution, win
   checks, restart/checkpoint, serialization, and debug/oracle state reads to
   the helpers.
3. Convert solver and compact bridge materialization/copy-back paths to direct
   `PersistentLevelState` board assignment.
4. Run broad parity, then delete remaining `scratch.interpreterBoard` reads and
   the sync compatibility helpers.
5. Rerun solver focus benchmarks comparing interpreter-on-compact-board against
   compiled compact kernels.

## Progress

- 2026-04-30: Retargeted the low-level runtime/search/solver board helpers so
  live object reads, writes, snapshots, restart/checkpoint storage, compact
  oracle export, and solver keys use `PersistentLevelState::board.objects`.
  `Scratch::interpreterBoard.objects` remains only as a compatibility mirror
  maintained by the legacy loader/clear helpers.

## Test Plan

Run after each conversion slice:

```bash
git diff --check
make build
make compact_turn_oracle_smoke
make solver_smoke_tests
make solver_compact_parity_smoke
```

Run before removing compatibility helpers:

```bash
make simulation_tests_cpp
make compact_turn_codegen_simulation_tests COMPILED_RULES_BUILD_JOBS=4
make compact_turn_simulation_tests
make solver_focus_compact_codegen_perf_report SOLVER_FOCUS_RUNS=1
```

Acceptance criteria:

- `PersistentLevelState::board.objects` is the only authoritative live board.
- Interpreter and compiler-mode compact turns pass existing simulation/oracle
  guards.
- Solver compact storage no longer materializes through
  `Scratch::interpreterBoard`.
- `Scratch::interpreterBoard` is deleted or documented as a temporary mirror
  with no remaining authority over gameplay state.

## Assumptions

- The persistent board remains cell-major:
  `objects[tileIndex * strideObject + word]`.
- Object-major `Scratch::objectCellBits` and `Scratch::objectCellCounts` remain
  scratch-only derived indexes.
- Message/title/metagame state remains outside turn-core board storage.
- This migration should not add new compiler shape recognizers or change
  PuzzleScript semantics.
