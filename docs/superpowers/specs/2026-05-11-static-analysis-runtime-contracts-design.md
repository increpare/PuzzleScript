# Static Analysis Runtime Contracts Design

## Purpose

Add a runtime contract check for static-analysis claims by replaying the existing JavaScript simulation corpus and asserting that proved static objects stay put at every turn boundary.

The goal is to test semantic consequences of analyser tags, not to re-derive tags from runtime behavior. If the analyser says an object is `static`, then normal gameplay input and rule execution must not change that object's per-cell occupancy. The first slice checks this contract over the already curated `src/tests/resources/testdata.js` simulation tests.

## First Slice

The first implementation checks object-level `static` tags only.

For each simulation test:

1. Analyze the PuzzleScript source with `analyzeSource`.
2. Collect `ps_tagged.objects` whose `tags.static` value is true.
3. Compile and load the simulation target level with the test's existing random seed.
4. Drain any initial `again` chain.
5. Snapshot each static object's occupancy across `level.n_tiles`.
6. Replay the recorded inputs exactly like the existing simulation runner.
7. After each input plus `again` drain, assert every static object snapshot is unchanged.
8. Compare the final serialized level against the simulation test's expected output as a replay-parity guard.

`undo` and `restart` are snapshot reset points. After either command runs and any `again` chain drains, the checker rebuilds static object snapshots from the current level. Static objects are promised not to move because of gameplay rules and input; undo and restart intentionally restore or reload board state.

## Components

Add a new Node test script:

```text
src/tests/run_static_analysis_runtime_contracts_node.js
```

The script should reuse the same browser-shimmed JavaScript engine/testdata environment used by `src/tests/run_tests_node.js`, rather than inventing a separate simulation path.

The script owns three focused helpers:

- `staticContractForSource(source, sourcePath)`: runs `analyzeSource`, validates status, and returns static object display names.
- `snapshotStaticObjects(objectNames)`: maps analyser display names to runtime object names and records one occupancy vector per object.
- `runSimulationWithStaticChecks(testName, dataarray)`: mirrors the existing `runTest` compile/replay loop, checking static snapshots at every boundary.

The runtime object-name lookup should follow existing static-analysis runtime helpers where possible: compare lower-case runtime names and `state.original_case_names` display names.

## Data Flow

Each simulation case in `global.testdata` has the existing shape:

```text
[source, inputs, expectedSerializedLevel, targetLevel?, randomSeed?, expectedSounds?]
```

The contract runner reads the same fields:

- `source` goes to both `analyzeSource` and `compile`.
- `inputs` drive `processInput`, `DoUndo`, `DoRestart`, and `tick` handling.
- `expectedSerializedLevel` is checked to confirm the contract runner replayed the test the same way as the normal simulation suite.
- `targetLevel` and `randomSeed` are passed to `compile` exactly as in `runTest`.
- `expectedSounds`, when present, is checked like `runTest`.

The checker should preserve the simulation suite's `again` behavior: drain `againing` after compile and after every input before checking boundary invariants.

## Failure Reporting

Failures should identify the contract breach without making the user dig through raw engine state.

A static-object occupancy failure should include:

- simulation test name
- input index and token
- object display name
- first changed cell index
- before/after presence at that cell

Other failures should be strict:

- static analysis status is not `ok`
- static object display name cannot be mapped to a runtime object
- replay throws
- final serialized level differs from expected output, which means the contract runner did not faithfully replay the simulation case
- sound output differs from expected output when `expectedSounds` is present

The runner should return a non-zero exit status on any failure.

## Test Selection

The runner should support:

```sh
node src/tests/run_static_analysis_runtime_contracts_node.js
node src/tests/run_static_analysis_runtime_contracts_node.js --filter "sokoban with win condition"
```

The filter is a debugging convenience and matches simulation test names. The default run should cover every `testdata.js` simulation case.

Avoid skips by default. If a legacy simulation case cannot be analysed or replayed in this environment, add a tiny explicit skip list with comments explaining why each skipped test is outside this contract harness. The intended steady state is no skips.

## Make Targets

Add a direct target for iteration:

```make
static_analysis_runtime_contracts:
	$(NODE) src/tests/run_static_analysis_runtime_contracts_node.js
```

Wire the same script into `make static_analysis_tests`, because this checks analyser claims using the simulation corpus. It should not replace `simulation_tests_js`; the normal simulation suite remains the exact replay regression test.

## Reporting

At the end of a successful run, print a concise summary:

```text
static_analysis_runtime_contracts: ok (N cases, M with static objects, K object-boundary checks)
```

`N` counts simulation cases run after filtering, `M` counts cases with at least one static object, and `K` counts object snapshot comparisons at turn boundaries.

## Out Of Scope

This design does not add:

- random generated 5x5 boards
- `count_invariant`, `temporary`, layer-static, mergeability, action-noop, program-flow, or winflow runtime contracts
- mid-turn movement-buffer inspection
- solver or optimizer behavior checks
- new static-analysis semantics

The harness should be written so later claim families can reuse its compile/replay loop, but the first implementation should stay focused on static object occupancy.

## Success Criteria

- The new runner checks all ordinary simulation tests by default.
- Every proved static object is asserted unchanged after each normal input/tick turn boundary.
- `undo` and `restart` rebuild snapshots instead of comparing against the previous board.
- Final serialized-level checks still match the simulation corpus expectations, confirming replay parity with `runTest`.
- Sound-output checks still match simulation corpus expectations when expected sounds are listed.
- `make static_analysis_tests` runs the contract runner.
- Failures name the simulation case, input, object, and changed cell.
