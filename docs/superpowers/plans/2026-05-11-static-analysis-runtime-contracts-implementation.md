# Static Analysis Runtime Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runtime contract runner that replays the existing JavaScript simulation corpus and asserts proved static objects keep identical per-cell occupancy at every gameplay turn boundary.

**Architecture:** Create one focused Node script that loads the existing browser-shimmed PuzzleScript runtime with testdata, analyzes each simulation source, snapshots static object occupancy, replays the same inputs as `runTest`, and verifies both static invariants and replay parity. Wire the script into a direct make target and into `make static_analysis_tests`.

**Tech Stack:** Node.js CommonJS, existing PuzzleScript JS runtime, `src/tests/js_oracle/lib/puzzlescript_node_env.js`, `src/tests/ps_static_analysis.js`, `src/tests/resources/testdata.js`, Makefile targets.

---

## File Structure

- Create `src/tests/run_static_analysis_runtime_contracts_node.js`
  - Owns the static-object runtime contract runner.
  - Loads the existing engine and simulation corpus via `loadPuzzleScript({ includeTests: true })`.
  - Imports `analyzeSource` from `src/tests/ps_static_analysis.js`.
  - Exports helpers for focused smoke/unit checks.
  - Runs as a CLI with `--filter` support.
- Modify `Makefile`
  - Add `static_analysis_runtime_contracts` to `.PHONY`.
  - Add a direct target that runs the new script.
  - Add the new script to `static_analysis_tests`.

No production engine or analyser semantics change in this plan.

The runner treats `undo`, `restart`, and detected board identity changes such as level/message transitions as snapshot reset boundaries. Static object occupancy is checked across normal gameplay turns within a loaded board, not across intentional level reloads or transitions.

---

### Task 1: Add The Runtime Contract Runner

**Files:**
- Create: `src/tests/run_static_analysis_runtime_contracts_node.js`

- [ ] **Step 1: Run the missing-runner smoke check**

Run:

```sh
node -e "require('./src/tests/run_static_analysis_runtime_contracts_node.js')"
```

Expected: FAIL with a module-not-found error for `./src/tests/run_static_analysis_runtime_contracts_node.js`.

- [ ] **Step 2: Create `src/tests/run_static_analysis_runtime_contracts_node.js`**

Create the file with this complete content:

```js
#!/usr/bin/env node
'use strict';

const assert = require('assert');

const { analyzeSource } = require('./ps_static_analysis');
const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

let runtimeLoaded = false;

function parseArgs(argv) {
    const options = { filter: null, help: false };
    for (let index = 2; index < argv.length; index++) {
        const arg = argv[index];
        if (arg === '--help' || arg === '-h') {
            options.help = true;
        } else if (arg === '--filter') {
            assert.ok(index + 1 < argv.length, '--filter requires a value');
            options.filter = argv[++index];
        } else if (!arg.startsWith('-') && options.filter === null) {
            options.filter = arg;
        } else {
            throw new Error(`Unexpected argument: ${arg}`);
        }
    }
    return options;
}

function usage() {
    return [
        'Usage: node src/tests/run_static_analysis_runtime_contracts_node.js [--filter NAME]',
        '',
        'Replays src/tests/resources/testdata.js and checks static object occupancy invariants.',
    ].join('\n');
}

function ensureRuntimeLoaded() {
    if (!runtimeLoaded) {
        loadPuzzleScript({ includeTests: true, messageSink: [] });
        runtimeLoaded = true;
    }
}

function resetParserErrors() {
    if (typeof resetParserErrorState === 'function') {
        resetParserErrorState();
    } else {
        errorStrings = [];
        errorCount = 0;
    }
}

function strippedErrorSummary() {
    return (errorStrings || []).map(stripHTMLTags).join('\n');
}

function drainAgain() {
    while (againing) {
        againing = false;
        processInput(-1);
    }
}

function staticContractForSource(source, sourcePath) {
    const report = analyzeSource(source, { sourcePath });
    if (report.status !== 'ok') {
        throw new Error(`${sourcePath}: static analysis status ${report.status}`);
    }
    return ((report.ps_tagged && report.ps_tagged.objects) || [])
        .filter(object => object.tags && object.tags.static === true)
        .map(object => object.name);
}

function engineObjectName(displayName) {
    const target = String(displayName).toLowerCase();
    const match = Object.keys(state.objects || {}).find(name =>
        name === target
        || (state.original_case_names && state.original_case_names[name] === displayName)
    );
    if (!match) {
        const available = Object.keys(state.objects || {}).sort().join(', ');
        throw new Error(`runtime object not found for static analyser object ${JSON.stringify(displayName)}; available: ${available}`);
    }
    return match;
}

function canSnapshotBoard() {
    return level && Number.isInteger(level.n_tiles);
}

function boardIdentity() {
    if (!canSnapshotBoard()) {
        return {
            available: false,
            textMode: Boolean(textMode),
            titleScreen: Boolean(titleScreen),
        };
    }
    return {
        available: true,
        curlevel: typeof curlevel === 'number' ? curlevel : null,
        curlevelTarget: typeof curlevelTarget === 'number' ? curlevelTarget : null,
        width: level.width,
        height: level.height,
        nTiles: level.n_tiles,
        textMode: Boolean(textMode),
        titleScreen: Boolean(titleScreen),
    };
}

function sameBoardIdentity(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
}

function objectOccupancySnapshot(displayName) {
    if (!canSnapshotBoard()) {
        throw new Error(`cannot snapshot ${displayName}: no active board level`);
    }
    const runtimeName = engineObjectName(displayName);
    const object = state.objects[runtimeName];
    const cells = [];
    for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
        cells.push(level.getCell(cellIndex).get(object.id) ? 1 : 0);
    }
    return cells;
}

function snapshotStaticObjects(objectNames) {
    const snapshots = new Map();
    if (!canSnapshotBoard()) return snapshots;
    for (const objectName of objectNames) {
        snapshots.set(objectName, objectOccupancySnapshot(objectName));
    }
    return snapshots;
}

function firstSnapshotDifference(beforeSnapshots, objectNames) {
    for (const objectName of objectNames) {
        const before = beforeSnapshots.get(objectName) || [];
        const after = objectOccupancySnapshot(objectName);
        const length = Math.max(before.length, after.length);
        for (let cellIndex = 0; cellIndex < length; cellIndex++) {
            const beforeValue = before[cellIndex] || 0;
            const afterValue = after[cellIndex] || 0;
            if (beforeValue !== afterValue) {
                return {
                    objectName,
                    cellIndex,
                    before: beforeValue,
                    after: afterValue,
                };
            }
        }
    }
    return null;
}

function executeInputToken(inputToken) {
    if (inputToken === 'undo') {
        DoUndo(false, true);
        return { resetsSnapshot: true };
    }
    if (inputToken === 'restart') {
        DoRestart();
        return { resetsSnapshot: true };
    }
    if (inputToken === 'tick') {
        processInput(-1);
        return { resetsSnapshot: false };
    }
    processInput(inputToken);
    return { resetsSnapshot: false };
}

function compileSimulationSource(source, targetLevel, randomSeed) {
    levelString = source;
    resetParserErrors();
    compile(['loadLevel', targetLevel], source, randomSeed);
    assert.strictEqual(errorCount, 0, strippedErrorSummary());
    drainAgain();
}

function tokenLabel(inputToken) {
    return typeof inputToken === 'string' ? inputToken : String(inputToken);
}

function assertFinalReplayParity(testName, expectedSerializedLevel, expectedSounds) {
    const actualSerializedLevel = convertLevelToString();
    if (actualSerializedLevel !== expectedSerializedLevel) {
        throw new Error(`${testName}: final serialized level differs from simulation expectation`);
    }

    if (expectedSounds !== null) {
        const actualSounds = soundHistory.join(';');
        const expectedSoundText = expectedSounds.join(';');
        if (actualSounds !== expectedSoundText) {
            throw new Error(`${testName}: sound output expected ${JSON.stringify(expectedSoundText)}, got ${JSON.stringify(actualSounds)}`);
        }
    }
}

function runSimulationWithStaticChecks(testName, dataarray) {
    const source = dataarray[0];
    const inputs = dataarray[1];
    const expectedSerializedLevel = dataarray[2];
    const targetLevel = dataarray[3] === undefined ? 0 : dataarray[3];
    const randomSeed = dataarray[4] === undefined ? null : dataarray[4];
    const expectedSounds = dataarray[5] === undefined ? null : dataarray[5];
    const staticObjects = staticContractForSource(source, `testdata:${testName}`);

    const previousUnitTesting = unitTesting;
    const previousLazyFunctionGeneration = lazyFunctionGeneration;
    unitTesting = true;
    lazyFunctionGeneration = false;

    let objectBoundaryChecks = 0;
    try {
        compileSimulationSource(source, targetLevel, randomSeed);

        let currentIdentity = boardIdentity();
        let snapshots = snapshotStaticObjects(staticObjects);

        for (let inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
            const inputToken = inputs[inputIndex];
            const result = executeInputToken(inputToken);
            drainAgain();

            const nextIdentity = boardIdentity();
            const resetBoundary =
                result.resetsSnapshot
                || !sameBoardIdentity(currentIdentity, nextIdentity)
                || !canSnapshotBoard();

            if (resetBoundary) {
                currentIdentity = nextIdentity;
                snapshots = snapshotStaticObjects(staticObjects);
                continue;
            }

            const diff = firstSnapshotDifference(snapshots, staticObjects);
            if (diff) {
                throw new Error([
                    `${testName}: static object occupancy changed`,
                    `  input ${inputIndex}: ${tokenLabel(inputToken)}`,
                    `  object: ${diff.objectName}`,
                    `  cell: ${diff.cellIndex}`,
                    `  before: ${diff.before}`,
                    `  after: ${diff.after}`,
                ].join('\n'));
            }

            objectBoundaryChecks += staticObjects.length;
            currentIdentity = nextIdentity;
        }

        assertFinalReplayParity(testName, expectedSerializedLevel, expectedSounds);

        return {
            staticObjectCount: staticObjects.length,
            objectBoundaryChecks,
        };
    } finally {
        unitTesting = previousUnitTesting;
        lazyFunctionGeneration = previousLazyFunctionGeneration;
    }
}

function testMatchesFilter(testName, filter) {
    return !filter || testName.toLowerCase().includes(filter.toLowerCase());
}

function runAll(options = {}) {
    ensureRuntimeLoaded();
    assert.ok(Array.isArray(global.testdata), 'global.testdata should be loaded');

    const failures = [];
    let caseCount = 0;
    let casesWithStaticObjects = 0;
    let objectBoundaryChecks = 0;

    for (const entry of global.testdata) {
        const testName = entry[0];
        const dataarray = entry[1];
        if (!testMatchesFilter(testName, options.filter || null)) continue;

        caseCount++;
        try {
            const result = runSimulationWithStaticChecks(testName, dataarray);
            if (result.staticObjectCount > 0) {
                casesWithStaticObjects++;
            }
            objectBoundaryChecks += result.objectBoundaryChecks;
        } catch (error) {
            failures.push(`ERROR: ${testName}: ${error.message}`);
        }
    }

    if (caseCount === 0) {
        throw new Error(options.filter
            ? `No simulation tests matched filter ${JSON.stringify(options.filter)}`
            : 'No simulation tests were loaded');
    }

    return {
        ok: failures.length === 0,
        caseCount,
        casesWithStaticObjects,
        objectBoundaryChecks,
        failures,
    };
}

function main() {
    const options = parseArgs(process.argv);
    if (options.help) {
        console.log(usage());
        return 0;
    }

    const result = runAll(options);
    if (!result.ok) {
        console.error('static_analysis_runtime_contracts: failed');
        for (const failure of result.failures) {
            console.error(failure);
        }
        return 1;
    }

    console.log(
        `static_analysis_runtime_contracts: ok (${result.caseCount} cases, ${result.casesWithStaticObjects} with static objects, ${result.objectBoundaryChecks} object-boundary checks)`
    );
    return 0;
}

if (require.main === module) {
    try {
        process.exitCode = main();
    } catch (error) {
        console.error(error.stack || error.message);
        process.exitCode = 1;
    }
}

module.exports = {
    boardIdentity,
    engineObjectName,
    firstSnapshotDifference,
    parseArgs,
    runAll,
    runSimulationWithStaticChecks,
    snapshotStaticObjects,
    staticContractForSource,
};
```

- [ ] **Step 3: Run the missing-runner smoke check again**

Run:

```sh
node -e "const runner = require('./src/tests/run_static_analysis_runtime_contracts_node.js'); console.log(typeof runner.runAll)"
```

Expected: PASS and prints:

```text
function
```

- [ ] **Step 4: Run one filtered simulation contract**

Run:

```sh
node src/tests/run_static_analysis_runtime_contracts_node.js --filter "sokoban with win condition"
```

Expected: PASS and output starts with:

```text
static_analysis_runtime_contracts: ok (1 cases,
```

- [ ] **Step 5: Run the full contract runner**

Run:

```sh
node src/tests/run_static_analysis_runtime_contracts_node.js
```

Expected: PASS and output starts with:

```text
static_analysis_runtime_contracts: ok (
```

If this fails with `static object occupancy changed`, keep the runner strict and inspect the named source/input/object/cell. The intended result of this task is a passing static-object contract over the current simulation corpus.

- [ ] **Step 6: Commit the runner**

Run:

```sh
git add src/tests/run_static_analysis_runtime_contracts_node.js
git commit -m "test: add static analysis runtime contract runner"
```

Expected: commit succeeds and mentions `src/tests/run_static_analysis_runtime_contracts_node.js`.

---

### Task 2: Wire The Runner Into Make Targets

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Run the missing make target**

Run:

```sh
make static_analysis_runtime_contracts
```

Expected: FAIL with:

```text
No rule to make target `static_analysis_runtime_contracts'
```

The punctuation around the target name can vary by make version.

- [ ] **Step 2: Add the target to `.PHONY`**

In `Makefile`, add `static_analysis_runtime_contracts` immediately after `static_analysis_tests` in the `.PHONY` list. The beginning of the `.PHONY` line should become:

```make
.PHONY: help build build_32 build_solver build_generator generator solver run ctest tests js_parity_tests tests_js static_analysis_tests static_analysis_runtime_contracts static_analysis_explorer simulation_tests_js simulation_tests_js_profile simulation_tests_js_profile_breakdown compilation_tests_js performance_testpage \
```

- [ ] **Step 3: Add the direct make target**

In `Makefile`, add this target immediately after `static_analysis_tests`:

```make
static_analysis_runtime_contracts:
	$(NODE) src/tests/run_static_analysis_runtime_contracts_node.js
```

- [ ] **Step 4: Wire the runner into `static_analysis_tests`**

In `Makefile`, add the new runner after `src/tests/ps_static_analysis_node.js` so the target reads:

```make
static_analysis_tests:
	$(NODE) src/tests/ps_static_analysis_node.js
	$(NODE) src/tests/run_static_analysis_runtime_contracts_node.js
	$(NODE) src/tests/static_analysis_testdata_runner.js
	$(NODE) src/tests/static_analysis_testdata_runner_node.js
	$(NODE) src/tests/static_analysis_explorer_node.js
	$(NODE) src/tests/solver_static_opt_node.js
	$(NODE) src/tests/compare_solver_static_opt_runs_node.js
```

- [ ] **Step 5: Run the direct make target**

Run:

```sh
make static_analysis_runtime_contracts
```

Expected: PASS and output starts with:

```text
node src/tests/run_static_analysis_runtime_contracts_node.js
static_analysis_runtime_contracts: ok (
```

- [ ] **Step 6: Run the full static-analysis target**

Run:

```sh
make static_analysis_tests
```

Expected: PASS. Output includes:

```text
ps_static_analysis_node: ok
static_analysis_runtime_contracts: ok (
static_analysis_testdata_runner: ok
static_analysis_testdata_runner_node: ok
static_analysis_explorer_node: ok
solver_static_opt_node: ok
compare_solver_static_opt_runs_node: ok
```

- [ ] **Step 7: Commit the make wiring**

Run:

```sh
git add Makefile
git commit -m "test: run static analysis runtime contracts"
```

Expected: commit succeeds and mentions `Makefile`.

---

### Task 3: Final Verification And Review

**Files:**
- No new edits expected.

- [ ] **Step 1: Check git status**

Run:

```sh
git status --short
```

Expected: no output.

- [ ] **Step 2: Re-run focused filter for debugging ergonomics**

Run:

```sh
node src/tests/run_static_analysis_runtime_contracts_node.js --filter "sokoban"
```

Expected: PASS and output starts with:

```text
static_analysis_runtime_contracts: ok (
```

The case count should be greater than zero.

- [ ] **Step 3: Re-run the full verification target**

Run:

```sh
make static_analysis_tests
```

Expected: PASS with the runtime contract summary included.

- [ ] **Step 4: Inspect the resulting commits**

Run:

```sh
git log --oneline -3
```

Expected: the most recent commits include:

```text
test: run static analysis runtime contracts
test: add static analysis runtime contract runner
```
