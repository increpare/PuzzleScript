# Compiler Monster Garden Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, bounded compiler mutation garden that turns existing PuzzleScript fixtures into minimized regression-ready crash specimens.

**Architecture:** `garden.js` is a pure Node helper for corpus, mutation, signatures, shrinking, and artifacts. `worker.js` is a one-job child that loads the real compiler the same way `run_tests_node.js` does. `run.js` is the parent: timeouts, classification, shrinking, reporting. Do not modify `src/js/compiler.js` or the editor.

**Tech Stack:** Node.js built-ins, the existing browser-style JavaScript sources, and the existing fixture arrays.

**Worktree:** `.worktrees/compiler-monster-garden`

**Spec:** `docs/superpowers/specs/2026-08-14-compiler-monster-garden-design.md`

---

## File map

- Create: `src/tests/monster_garden/garden.js` — deterministic core
- Modify: `src/tests/monster_garden/tests.js` — already started; later tasks append tests
- Create: `src/tests/monster_garden/worker.js` — one-shot compiler child
- Create: `src/tests/monster_garden/run.js` — CLI parent
- Modify: `.gitignore` — ignore `.build/monster_garden/`
- Modify: `DEVELOPMENT.md` — how to run the garden

Do not create a new test framework. Follow the existing `assert` + `test()` style in `tests.js`.

---

### Task 1: Garden core

**Files:**
- Create: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js` (already written)

The unit tests already exist and import `./garden`. Implement the smallest module that makes them pass. Do not add worker or CLI process code yet.

- [ ] **Step 1: Confirm the tests fail because the module is missing**

Run: `node src/tests/monster_garden/tests.js`

Expected: FAIL with `Cannot find module './garden'`

- [ ] **Step 2: Implement `garden.js`**

Create `src/tests/monster_garden/garden.js` with exactly this source:

```js
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

function Random(seed) {
    this._state = seed >>> 0;
}

Random.prototype.next = function() {
    this._state = (this._state + 0x6D2B79F5) >>> 0;
    let t = this._state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

Random.prototype.integer = function(n) {
    if (!(n > 0)) {
        throw new Error('n must be positive');
    }
    return Math.floor(this.next() * n);
};

Random.prototype.pick = function(array) {
    return array[this.integer(array.length)];
};

function evaluateDataFile(filePath) {
    const context = {};
    vm.runInNewContext(fs.readFileSync(filePath, 'utf8'), context);
    return context;
}

function loadCorpus(resourceDir) {
    const testdata = evaluateDataFile(path.join(resourceDir, 'testdata.js')).testdata;
    const errors = evaluateDataFile(path.join(resourceDir, 'errormessage_testdata.js')).errormessage_testdata;
    const corpus = [];
    for (let i = 0; i < testdata.length; i++) {
        const payload = testdata[i][1];
        corpus.push({
            name: testdata[i][0],
            fixtureIndex: i,
            kind: 'simulation',
            source: payload[0],
            inputs: payload[1] || [],
            level: payload[3] !== undefined ? payload[3] : 0,
            randomSeed: payload[4] !== undefined ? payload[4] : null
        });
    }
    for (let i = 0; i < errors.length; i++) {
        const payload = errors[i][1];
        corpus.push({
            name: errors[i][0],
            fixtureIndex: i,
            kind: 'compiler-message',
            source: payload[0],
            inputs: [],
            level: 0,
            randomSeed: null
        });
    }
    return corpus;
}

const SECTION_NAMES = [
    'OBJECTS', 'LEGEND', 'SOUNDS', 'COLLISIONLAYERS', 'RULES', 'WINCONDITIONS', 'LEVELS'
];

function findSection(source, name) {
    const lines = source.split('\n');
    let start = -1;
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].trim().toUpperCase() === name) {
            start = i;
            break;
        }
    }
    if (start < 0) {
        return null;
    }
    let end = lines.length;
    for (let i = start + 1; i < lines.length; i++) {
        if (SECTION_NAMES.indexOf(lines[i].trim().toUpperCase()) >= 0) {
            end = i;
            break;
        }
    }
    return { name: name, start: start, end: end, lines: lines };
}

function replaceRange(source, start, end, insert) {
    return source.slice(0, start) + insert + source.slice(end);
}

function marksIn(text, regex) {
    const marks = [];
    const re = new RegExp(regex.source, regex.flags.indexOf('g') >= 0 ? regex.flags : regex.flags + 'g');
    let match;
    while ((match = re.exec(text))) {
        marks.push({ index: match.index, text: match[0] });
    }
    return marks;
}

function mutateSection(source, sectionName, fn) {
    const section = findSection(source, sectionName);
    if (!section) {
        return null;
    }
    const body = section.lines.slice(section.start, section.end).join('\n');
    const next = fn(body);
    if (!next) {
        return null;
    }
    const lines = section.lines.slice();
    const replacement = next.source.split('\n');
    lines.splice(section.start, section.end - section.start, ...replacement);
    return { source: lines.join('\n'), detail: next.detail };
}

function deleteRulePunctuation(source, rng) {
    return mutateSection(source, 'RULES', function(body) {
        const marks = marksIn(body, /->|[\[\]|]/);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, ''),
            detail: 'deleted ' + mark.text
        };
    });
}

function duplicateRulePunctuation(source, rng) {
    return mutateSection(source, 'RULES', function(body) {
        const marks = marksIn(body, /->|[\[\]|]/);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, mark.text + mark.text),
            detail: 'duplicated ' + mark.text
        };
    });
}

function swapLegendOperator(source, rng) {
    return mutateSection(source, 'LEGEND', function(body) {
        const marks = marksIn(body, /\s+(and|or)\s+/i);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        const swapped = /and/i.test(mark.text)
            ? mark.text.replace(/and/i, 'or')
            : mark.text.replace(/or/i, 'and');
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, swapped),
            detail: 'swapped ' + mark.text.trim()
        };
    });
}

function invalidViewport(source, rng) {
    const marks = marksIn(source, /\b(flickscreen|zoomscreen)\s+-?\d+x-?\d+/i);
    const replacement = rng.integer(2) === 0 ? '0x0' : '-1x3';
    if (marks.length > 0) {
        const mark = marks[rng.integer(marks.length)];
        const next = mark.text.replace(/-?\d+x-?\d+/, replacement);
        return {
            source: replaceRange(source, mark.index, mark.index + mark.text.length, next),
            detail: 'viewport ' + next
        };
    }
    return {
        source: 'flickscreen ' + replacement + '\n' + source,
        detail: 'injected flickscreen ' + replacement
    };
}

function duplicateRuleCommand(source, rng) {
    return mutateSection(source, 'RULES', function(body) {
        const marks = marksIn(body, /\b(again|cancel|restart|checkpoint|win)\b/i);
        if (marks.length === 0) {
            return null;
        }
        const mark = marks[rng.integer(marks.length)];
        return {
            source: replaceRange(body, mark.index, mark.index + mark.text.length, mark.text + ' ' + mark.text),
            detail: 'duplicated command ' + mark.text
        };
    });
}

function legendCycle(source) {
    return mutateSection(source, 'LEGEND', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\nGardenCycleA = GardenCycleB\nGardenCycleB = GardenCycleA\n',
            detail: 'added GardenCycleA/GardenCycleB synonym cycle'
        };
    });
}

function swapLineRanges(lines, first, second) {
    const lo = first.start < second.start ? first : second;
    const hi = first.start < second.start ? second : first;
    return lines.slice(0, lo.start)
        .concat(lines.slice(hi.start, hi.end))
        .concat(lines.slice(lo.end, hi.start))
        .concat(lines.slice(lo.start, lo.end))
        .concat(lines.slice(hi.end))
        .join('\n');
}

function swapSections(source, rng) {
    const lines = source.split('\n');
    const found = [];
    for (let i = 0; i < SECTION_NAMES.length; i++) {
        const section = findSection(source, SECTION_NAMES[i]);
        if (section) {
            found.push(section);
        }
    }
    const candidates = [];
    for (let i = 0; i < found.length; i++) {
        for (let j = i + 1; j < found.length; j++) {
            const swapped = swapLineRanges(lines, found[i], found[j]);
            if (swapped !== source) {
                candidates.push({
                    source: swapped,
                    detail: 'swapped ' + found[i].name + ' and ' + found[j].name
                });
            }
        }
    }
    if (candidates.length === 0) {
        return null;
    }
    return candidates[rng.integer(candidates.length)];
}

function oddWhitespace(source, rng) {
    const indexes = [];
    for (let i = 1; i < source.length - 1; i++) {
        if (source[i] === ' ' && /[A-Za-z0-9]/.test(source[i - 1]) && /[A-Za-z0-9]/.test(source[i + 1])) {
            indexes.push(i);
        }
    }
    if (indexes.length === 0) {
        return null;
    }
    const index = indexes[rng.integer(indexes.length)];
    const insert = rng.integer(2) === 0 ? '\t' : '\u00a0';
    return {
        source: replaceRange(source, index, index + 1, insert),
        detail: 'replaced a token-boundary space'
    };
}

function unterminatedComment(source, rng) {
    const lines = source.split('\n');
    const index = rng.integer(lines.length);
    lines[index] = '(' + lines[index];
    return {
        source: lines.join('\n'),
        detail: 'unterminated comment on line ' + (index + 1)
    };
}

const mutators = [
    { name: 'delete-rule-punctuation', apply: deleteRulePunctuation },
    { name: 'duplicate-rule-punctuation', apply: duplicateRulePunctuation },
    { name: 'swap-legend-operator', apply: swapLegendOperator },
    { name: 'invalid-viewport', apply: invalidViewport },
    { name: 'duplicate-rule-command', apply: duplicateRuleCommand },
    { name: 'legend-cycle', apply: legendCycle },
    { name: 'swap-sections', apply: swapSections },
    { name: 'odd-whitespace', apply: oddWhitespace },
    { name: 'unterminated-comment', apply: unterminatedComment }
];

function mutateFixture(fixture, rng, mutatorNames, options) {
    const allowed = mutators.filter(function(mutator) {
        return !mutatorNames || mutatorNames.indexOf(mutator.name) >= 0;
    });
    if (allowed.length === 0) {
        throw new Error('inapplicable mutation: no mutators selected');
    }
    const maxAttempts = options && options.maxAttempts ? options.maxAttempts : 8;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        const mutator = allowed[rng.integer(allowed.length)];
        const applied = mutator.apply(fixture.source, rng);
        if (applied) {
            return {
                mutator: mutator.name,
                fixtureName: fixture.name,
                fixtureIndex: fixture.fixtureIndex,
                kind: fixture.kind,
                source: applied.source,
                detail: applied.detail,
                inputs: fixture.inputs,
                level: fixture.level,
                randomSeed: fixture.randomSeed,
                attempt: attempt
            };
        }
    }
    throw new Error('inapplicable mutation after ' + maxAttempts + ' attempts');
}

function needValue(argv, i, name) {
    if (i + 1 >= argv.length) {
        throw new Error('Missing value for ' + name);
    }
    return argv[i + 1];
}

function needPositiveInt(argv, i, name) {
    const value = Number(needValue(argv, i, name));
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(name + ' must be a positive integer');
    }
    return value;
}

function parseArguments(argv, options) {
    const now = options && options.now ? options.now : Date.now;
    const result = {
        seed: typeof now === 'function' ? now() : now,
        count: 100,
        timeoutMs: 2000,
        shrink: true,
        replay: true,
        maxInputs: 8,
        shrinkBudget: 200,
        maxAttempts: 8,
        output: '.build/monster_garden',
        fixture: null,
        mutators: null,
        listMutators: false
    };
    const names = mutators.map(function(mutator) { return mutator.name; });
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        switch (arg) {
            case '--seed': {
                const value = Number(needValue(argv, i, 'seed'));
                if (!Number.isInteger(value) || value < 0) {
                    throw new Error('seed must be a non-negative integer');
                }
                result.seed = value;
                i++;
                break;
            }
            case '--count':
                result.count = needPositiveInt(argv, i, 'count');
                i++;
                break;
            case '--timeout-ms':
                result.timeoutMs = needPositiveInt(argv, i, 'timeout-ms');
                i++;
                break;
            case '--fixture':
                result.fixture = needValue(argv, i, 'fixture');
                i++;
                break;
            case '--mutator': {
                const selected = needValue(argv, i, 'mutator').split(',').map(function(name) {
                    return name.trim();
                }).filter(Boolean);
                for (let j = 0; j < selected.length; j++) {
                    if (names.indexOf(selected[j]) < 0) {
                        throw new Error('Unknown mutator: ' + selected[j]);
                    }
                }
                result.mutators = selected;
                i++;
                break;
            }
            case '--output':
                result.output = needValue(argv, i, 'output');
                i++;
                break;
            case '--no-shrink':
                result.shrink = false;
                break;
            case '--no-replay':
                result.replay = false;
                break;
            case '--max-inputs':
                result.maxInputs = needPositiveInt(argv, i, 'max-inputs');
                i++;
                break;
            case '--shrink-budget':
                result.shrinkBudget = needPositiveInt(argv, i, 'shrink-budget');
                i++;
                break;
            case '--max-attempts':
                result.maxAttempts = needPositiveInt(argv, i, 'max-attempts');
                i++;
                break;
            case '--list-mutators':
                result.listMutators = true;
                break;
            default:
                throw new Error('Unknown option: ' + arg);
        }
    }
    return result;
}

function failureSignature(result) {
    if (result.kind === 'crash' && result.error) {
        const message = String(result.error.message || '').split('\n')[0];
        return 'crash:' + result.error.name + ':' + message;
    }
    if (result.kind === 'invariant') {
        return 'invariant:' + String(result.detail || '');
    }
    return String(result.kind);
}

module.exports = {
    Random: Random,
    loadCorpus: loadCorpus,
    mutators: mutators,
    mutateFixture: mutateFixture,
    parseArguments: parseArguments,
    failureSignature: failureSignature
};
```

- [ ] **Step 3: Run the garden unit tests**

Run: `node src/tests/monster_garden/tests.js`

Expected: `7/7 monster garden tests passed` and exit 0.

If a named mutator does not change `SAMPLE`, inspect that mutator only. Do not weaken the test.

- [ ] **Step 4: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js docs/superpowers/specs/2026-08-14-compiler-monster-garden-design.md docs/superpowers/plans/2026-08-14-compiler-monster-garden.md
git commit -m "$(cat <<'EOF'
Add deterministic monster garden core.

EOF
)"
```

---

### Task 2: Classification, shrinking, and artifact helpers

**Files:**
- Modify: `src/tests/monster_garden/tests.js`
- Modify: `src/tests/monster_garden/garden.js`

- [ ] **Step 1: Append failing tests before the `async function main` line**

```js
test('level invariants accept a well-formed level and name the first broken field', function() {
    const good = {
        width: 2,
        height: 3,
        n_tiles: 6,
        objects: { length: 12 },
        movements: { length: 6 }
    };
    assert.strictEqual(garden.checkLevelInvariants(good, 2, 1), null);
    assert(/missing/.test(garden.checkLevelInvariants(null, 2, 1)));
    assert(/dimensions/.test(garden.checkLevelInvariants({ width: 0, height: 3, n_tiles: 0, objects: { length: 0 } }, 2, 1)));
    assert(/n_tiles/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 5, objects: { length: 10 }
    }, 2, 1)));
    assert(/objects/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 6, objects: { length: 5 }
    }, 2, 1)));
    assert(/movements/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 6, objects: { length: 12 }, movements: { length: 2 }
    }, 2, 1)));
});

test('only crashes, timeouts, invariants, nondeterminism, and replay divergence are interesting', function() {
    assert.strictEqual(garden.isInteresting({ kind: 'ok' }), false);
    assert.strictEqual(garden.isInteresting({ kind: 'compiler-error' }), false);
    assert.strictEqual(garden.isInteresting({ kind: 'crash' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'timeout' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'invariant' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'nondeterministic' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'replay-divergence' }), true);
});

test('line shrinking keeps a deletion only when the signature stays the same', function() {
    const source = 'keep\nnoise\nkeep\n';
    const result = garden.shrinkSource(source, function(candidate) {
        return candidate.indexOf('keep') >= 0 && candidate.indexOf('noise') < 0;
    }, 20);
    assert.strictEqual(result.source, 'keep\nkeep\n');
    assert(result.steps > 0);
    assert(result.steps <= 20);
});

test('artifact names and regression fixtures are copy-pasteable and path-safe', function() {
    const name = garden.artifactDirName('crash:TypeError:bad thing / \\ : *', 99, 3);
    assert.strictEqual(name, 'crash-TypeError-bad-thing-s99_0003');
    assert(!/[\/\\:*]/.test(name));
    const snippet = garden.formatRegression('monster garden 99 3', 'title "X"\nline\n');
    assert.strictEqual(
        snippet,
        '[\n    "monster garden 99 3",\n    ["title \\"X\\"\\nline\\n", [], ""]\n],\n'
    );
});
```

- [ ] **Step 2: Run tests and confirm the new helpers are missing**

Run: `node src/tests/monster_garden/tests.js`

Expected: FAIL with `checkLevelInvariants is not a function` (or the first missing export).

- [ ] **Step 3: Add the helpers to `garden.js` and export them**

Append these functions before `module.exports`, then add them to the export object:

```js
function checkLevelInvariants(level, strideObj, strideMov) {
    if (!level || typeof level !== 'object') {
        return 'level is missing';
    }
    if (!(level.width > 0) || !(level.height > 0)) {
        return 'level dimensions are invalid';
    }
    if (level.n_tiles !== level.width * level.height) {
        return 'n_tiles does not match width*height';
    }
    const expectedObjects = level.n_tiles * strideObj;
    if (!level.objects || level.objects.length !== expectedObjects) {
        return 'objects length is ' + (level.objects && level.objects.length) + ' expected ' + expectedObjects;
    }
    if (level.movements && level.movements.length !== level.n_tiles * strideMov) {
        return 'movements length is ' + level.movements.length + ' expected ' + (level.n_tiles * strideMov);
    }
    return null;
}

function isInteresting(result) {
    return result.kind === 'crash'
        || result.kind === 'timeout'
        || result.kind === 'invariant'
        || result.kind === 'nondeterministic'
        || result.kind === 'replay-divergence';
}

function shrinkSource(source, keep, budget) {
    let current = source.split('\n');
    let remaining = budget;
    let changed = true;
    while (changed && remaining > 0) {
        changed = false;
        let i = 0;
        while (i < current.length && remaining > 0) {
            const candidate = current.slice(0, i).concat(current.slice(i + 1));
            remaining--;
            if (keep(candidate.join('\n'))) {
                current = candidate;
                changed = true;
            } else {
                i++;
            }
        }
    }
    return { source: current.join('\n'), steps: budget - remaining };
}

function artifactDirName(signature, seed, index) {
    const safe = String(signature)
        .replace(/[^A-Za-z0-9._-]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 80);
    return (safe || 'monster') + '-s' + seed + '_' + String(index).padStart(4, '0');
}

function formatRegression(name, source) {
    return '[\n    ' + JSON.stringify(name) + ',\n    [' + JSON.stringify(source) + ', [], ""]\n],\n';
}

function writeArtifacts(outputDir, dirName, files) {
    const tmp = path.join(outputDir, dirName + '.tmp');
    const dest = path.join(outputDir, dirName);
    fs.rmSync(tmp, { recursive: true, force: true });
    fs.mkdirSync(tmp, { recursive: true });
    const names = Object.keys(files);
    for (let i = 0; i < names.length; i++) {
        fs.writeFileSync(path.join(tmp, names[i]), files[names[i]]);
    }
    fs.rmSync(dest, { recursive: true, force: true });
    fs.renameSync(tmp, dest);
    return dest;
}
```

Export `checkLevelInvariants`, `isInteresting`, `shrinkSource`, `artifactDirName`, `formatRegression`, and `writeArtifacts`.

If the artifact-name test fails because of a slightly different sanitization, change `artifactDirName` until the expected string matches. Do not change the test.

- [ ] **Step 4: Re-run the garden tests**

Run: `node src/tests/monster_garden/tests.js`

Expected: `11/11 monster garden tests passed`

- [ ] **Step 5: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "$(cat <<'EOF'
Add garden classification, shrinking, and artifact helpers.

EOF
)"
```

---

### Task 3: Isolated compiler worker

**Files:**
- Create: `src/tests/monster_garden/worker.js`
- Modify: `src/tests/monster_garden/tests.js`

The worker copies the browser shims and concatenated-source loading from `src/tests/run_tests_node.js`. It does **not** load `testdata.js` or `errormessage_testdata.js`.

- [ ] **Step 1: Append failing worker tests**

```js
function runWorkerSync(job, timeoutMs) {
    return spawnSync(process.execPath, [path.join(__dirname, 'worker.js')], {
        input: JSON.stringify(job),
        encoding: 'utf8',
        timeout: timeoutMs || 20000
    });
}

function workerResult(job) {
    const child = runWorkerSync(job);
    assert.strictEqual(child.error, undefined, child.stderr);
    const line = child.stdout.trim().split('\n').pop();
    return JSON.parse(line);
}

test('the worker compiles a valid sample and returns a stable ok fingerprint', function() {
    const job = {
        source: SAMPLE,
        inputs: [0, 3],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8
    };
    const first = workerResult(job);
    const second = workerResult(job);
    assert.strictEqual(first.kind, 'ok', JSON.stringify(first));
    assert.strictEqual(first.error, null);
    assert.strictEqual(typeof first.fingerprint, 'string');
    assert(first.fingerprint.indexOf('\n') >= 0);
    assert.strictEqual(first.errorCount, 0);
    assert.deepStrictEqual(first, second);
});

test('the worker treats compile diagnostics as compiler-error, not a crash', function() {
    const result = workerResult({
        source: 'title No Background\n=======\\nOBJECTS\\n=======\\n\\nPlayer\\nwhite\\n\\n=======\\nLEGEND\\n=======\\nP = Player\\n',
        inputs: [0],
        level: 0,
        randomSeed: null,
        replay: true,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'compiler-error', JSON.stringify(result));
    assert.strictEqual(result.error, null);
    assert(result.errorCount > 0);
    assert.strictEqual(result.fingerprint, 'compiler-error:' + result.errorCount);
});

test('the worker reports a crash when execution throws', function() {
    const result = workerResult({
        source: SAMPLE,
        inputs: ['not-a-command'],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'crash', JSON.stringify(result));
    assert.strictEqual(typeof result.error.name, 'string');
    assert.strictEqual(typeof result.error.message, 'string');
});
```

If `not-a-command` does not throw, keep the test but change the worker so unknown non-numeric, non-`undo`/`restart`/`tick` inputs throw `Error('Unknown input: ...')`. That is garden policy, not a compiler change.

- [ ] **Step 2: Run tests and confirm the worker is missing**

Run: `node src/tests/monster_garden/tests.js`

Expected: FAIL (`Cannot find module` or `Unexpected end of JSON input` from an empty spawn).

- [ ] **Step 3: Implement `worker.js`**

Create `src/tests/monster_garden/worker.js` with this source. The shim block is the same idea as `run_tests_node.js`; the execution policy is new.

```js
#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const garden = require('./garden');

const srcDir = path.join(__dirname, '..', '..');

const _storage = {};
global.localStorage = {
    getItem(key) { return _storage.hasOwnProperty(key) ? _storage[key] : null; },
    setItem(key, value) { _storage[key] = String(value); },
    removeItem(key) { delete _storage[key]; }
};

global.document = {
    URL: 'test://',
    body: {
        classList: { contains() { return false; } },
        addEventListener() {},
        removeEventListener() {}
    },
    createElement() {
        return { style: {}, innerHTML: '', textContent: '', getContext() { return null; } };
    },
    getElementById() { return null; }
};

global.window = global;
global.lastDownTarget = null;
global.canvas = null;
global.input = global.document.createElement('TEXTAREA');
global.canvasResize = function() {};
global.redraw = function() {};
global.forceRegenImages = function() {};
global.consolePrintFromRule = function() {};
global.consolePrint = function() {};
global.console_print_raw = function() {};
global.consoleError = function() {};
global.consoleCacheDump = function() {};
global.addToDebugTimeline = function() {};
global.killAudioButton = function() {};
global.showAudioButton = function() {};
global.regenSpriteImages = function() {};
global.jumpToLine = function() {};
global.printLevel = function() {};
global.playSound = function() {};
global.levelString = '';
global.editor = { getValue() { return global.levelString; } };
global.PuzzleScriptTestAssertions = { push() {}, equal() {} };
global.UnitTestingThrow = function(error) { throw error; };

const sourceFiles = [
    'js/storagewrapper.js',
    'js/bitvec.js',
    'js/level.js',
    'js/languageConstants.js',
    'js/globalVariables.js',
    'js/debug.js',
    'js/font.js',
    'js/rng.js',
    'js/riffwave.js',
    'js/sfxr.js',
    'js/codemirror/stringstream.js',
    'js/colorhelpers.js',
    'js/colors.js',
    'js/engine.js',
    'js/parser.js',
    'js/compiler.js',
    'js/soundbar.js'
];

let allCode = '';
for (let i = 0; i < sourceFiles.length; i++) {
    allCode += '\n// ---- ' + sourceFiles[i] + ' ----\n';
    allCode += fs.readFileSync(path.join(srcDir, sourceFiles[i]), 'utf8') + '\n';
}
vm.runInThisContext(allCode, { filename: 'monster_garden_worker.js' });

function emit(result) {
    process.stdout.write(JSON.stringify(result) + '\n');
}

function drainAgain() {
    while (global.againing) {
        global.againing = false;
        global.processInput(-1);
    }
}

function applyInputs(inputs) {
    for (let i = 0; i < inputs.length; i++) {
        const value = inputs[i];
        if (value === 'undo') {
            global.DoUndo(false, true);
        } else if (value === 'restart') {
            global.DoRestart();
        } else if (value === 'tick') {
            global.processInput(-1);
        } else if (typeof value === 'number') {
            global.processInput(value);
        } else {
            throw new Error('Unknown input: ' + value);
        }
        drainAgain();
    }
}

function fingerprintAfter(errorCount) {
    return String(errorCount) + '\n' + global.convertLevelToString();
}

function runOnce(job) {
    global.unitTesting = true;
    global.lazyFunctionGeneration = false;
    global.IDE = false;
    if (typeof global.resetParserErrorState === 'function') {
        global.resetParserErrorState();
    } else {
        global.errorStrings = [];
        global.errorCount = 0;
    }
    global.compile(['loadLevel', job.level], job.source, job.randomSeed);
    const errorCount = global.errorCount;
    if (errorCount > 0) {
        return {
            kind: 'compiler-error',
            error: null,
            fingerprint: 'compiler-error:' + errorCount,
            detail: '',
            errorCount: errorCount
        };
    }
    const broken = garden.checkLevelInvariants(global.level, global.STRIDE_OBJ, global.STRIDE_MOV);
    if (broken) {
        return {
            kind: 'invariant',
            error: null,
            fingerprint: fingerprintAfter(errorCount),
            detail: broken,
            errorCount: errorCount
        };
    }
    const prefix = (job.inputs || []).slice(0, job.maxInputs);
    applyInputs(prefix);
    const afterExec = garden.checkLevelInvariants(global.level, global.STRIDE_OBJ, global.STRIDE_MOV);
    if (afterExec) {
        return {
            kind: 'invariant',
            error: null,
            fingerprint: fingerprintAfter(errorCount),
            detail: afterExec,
            errorCount: errorCount
        };
    }
    return {
        kind: 'ok',
        error: null,
        fingerprint: fingerprintAfter(errorCount),
        detail: '',
        errorCount: errorCount,
        prefixLength: prefix.length
    };
}

function runJob(job) {
    try {
        const first = runOnce(job);
        if (first.kind !== 'ok') {
            return first;
        }
        const prefix = (job.inputs || []).slice(0, job.maxInputs);
        if (job.replay && prefix.length > 0) {
            for (let i = 0; i < prefix.length; i++) {
                global.DoUndo(false, true);
            }
            applyInputs(prefix);
            const replayed = fingerprintAfter(global.errorCount);
            if (replayed !== first.fingerprint) {
                return {
                    kind: 'replay-divergence',
                    error: null,
                    fingerprint: first.fingerprint,
                    detail: replayed,
                    errorCount: first.errorCount
                };
            }
        }
        const second = runOnce(job);
        if (second.kind !== 'ok') {
            return second;
        }
        if (second.fingerprint !== first.fingerprint) {
            return {
                kind: 'nondeterministic',
                error: null,
                fingerprint: first.fingerprint,
                detail: second.fingerprint,
                errorCount: first.errorCount
            };
        }
        return first;
    } catch (error) {
        return {
            kind: 'crash',
            error: { name: error.name, message: String(error.message || error) },
            fingerprint: '',
            detail: '',
            errorCount: typeof global.errorCount === 'number' ? global.errorCount : 0
        };
    }
}

function readStdin() {
    return new Promise(function(resolve, reject) {
        let data = '';
        process.stdin.setEncoding('utf8');
        process.stdin.on('data', function(chunk) { data += chunk; });
        process.stdin.on('end', function() { resolve(data); });
        process.stdin.on('error', reject);
    });
}

readStdin().then(function(raw) {
    emit(runJob(JSON.parse(raw)));
}).catch(function(error) {
    emit({
        kind: 'crash',
        error: { name: error.name, message: String(error.message || error) },
        fingerprint: '',
        detail: '',
        errorCount: 0
    });
});
```

- [ ] **Step 4: Re-run the garden tests**

Run: `node src/tests/monster_garden/tests.js`

Expected: `14/14 monster garden tests passed`

If the valid-sample test gets `compiler-error`, print the worker JSON and fix `SAMPLE` usage or worker setup. Do not treat diagnostics as `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/tests/monster_garden/worker.js src/tests/monster_garden/tests.js
git commit -m "$(cat <<'EOF'
Add one-job compiler worker for the monster garden.

EOF
)"
```

---

### Task 4: Parent runner, timeout, and shrinking

**Files:**
- Create: `src/tests/monster_garden/run.js`
- Modify: `src/tests/monster_garden/tests.js`

- [ ] **Step 1: Append failing parent tests**

```js
test('the parent classifies a hung child as timeout', function() {
    const hung = path.join(os.tmpdir(), 'monster-garden-hang.js');
    fs.writeFileSync(hung, 'setTimeout(function() {}, 100000);\n');
    return garden.runChild(process.execPath, [hung], '', 80).then(function(result) {
        assert.strictEqual(result.kind, 'timeout');
    });
});

test('run.js --list-mutators prints every mutator and exits 0', function() {
    const child = spawnSync(process.execPath, [path.join(__dirname, 'run.js'), '--list-mutators'], {
        encoding: 'utf8'
    });
    assert.strictEqual(child.status, 0, child.stderr);
    garden.mutators.forEach(function(mutator) {
        assert(child.stdout.indexOf(mutator.name) >= 0, mutator.name);
    });
});

test('run.js rejects malformed options with a nonzero exit', function() {
    const child = spawnSync(process.execPath, [path.join(__dirname, 'run.js'), '--count', '0'], {
        encoding: 'utf8'
    });
    assert.notStrictEqual(child.status, 0);
    assert(/count/.test(child.stderr));
});

test('a one-mutant CLI run is deterministic and writes no artifacts for healthy output', function() {
    const output = fs.mkdtempSync(path.join(os.tmpdir(), 'monster-garden-'));
    const args = [
        path.join(__dirname, 'run.js'),
        '--seed', '12345',
        '--count', '1',
        '--no-shrink',
        '--no-replay',
        '--timeout-ms', '20000',
        '--output', output
    ];
    const first = spawnSync(process.execPath, args, { encoding: 'utf8' });
    const second = spawnSync(process.execPath, args, { encoding: 'utf8' });
    assert.strictEqual(first.status, 0, first.stderr + first.stdout);
    assert.strictEqual(second.status, 0, second.stderr + second.stdout);
    assert.strictEqual(first.stdout, second.stdout);
    assert.strictEqual(fs.readdirSync(output).length, 0);
});
```

`runChild` lives on the `garden` export so the timeout test does not spawn the compiler. `run.js` must call that same function.

- [ ] **Step 2: Run tests and confirm the new API is missing**

Run: `node src/tests/monster_garden/tests.js`

Expected: FAIL with `runChild is not a function` or `Cannot find module './run.js'`.

- [ ] **Step 3: Add `runChild` to `garden.js` and export it**

```js
const { spawn } = require('child_process');

function runChild(command, args, stdin, timeoutMs) {
    return new Promise(function(resolve) {
        const child = spawn(command, args, { stdio: ['pipe', 'pipe', 'pipe'] });
        let stdout = '';
        let stderr = '';
        let timedOut = false;
        const timer = setTimeout(function() {
            timedOut = true;
            child.kill('SIGKILL');
        }, timeoutMs);
        child.stdout.on('data', function(chunk) { stdout += chunk; });
        child.stderr.on('data', function(chunk) { stderr += chunk; });
        child.on('close', function() {
            clearTimeout(timer);
            if (timedOut) {
                resolve({
                    kind: 'timeout',
                    error: null,
                    fingerprint: '',
                    detail: 'timeout',
                    errorCount: 0
                });
                return;
            }
            try {
                resolve(JSON.parse(stdout.trim().split('\n').pop()));
            } catch (error) {
                resolve({
                    kind: 'crash',
                    error: { name: 'ChildOutputError', message: (stdout || stderr || error.message).split('\n')[0] },
                    fingerprint: '',
                    detail: stderr,
                    errorCount: 0
                });
            }
        });
        child.stdin.write(stdin);
        child.stdin.end();
    });
}
```

- [ ] **Step 4: Implement `run.js`**

Create `src/tests/monster_garden/run.js`:

```js
#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const garden = require('./garden');

const workerPath = path.join(__dirname, 'worker.js');
const resourceDir = path.join(__dirname, '..', 'resources');

function filterCorpus(corpus, fixture) {
    if (!fixture) {
        return corpus;
    }
    const needle = fixture.toLowerCase();
    return corpus.filter(function(item) {
        return item.name.toLowerCase().indexOf(needle) >= 0;
    });
}

function allowedMutators(option) {
    return option || garden.mutators.map(function(mutator) { return mutator.name; });
}

async function evaluateMutant(mutant, options) {
    const job = {
        source: mutant.source,
        inputs: mutant.inputs,
        level: mutant.level,
        randomSeed: mutant.randomSeed,
        replay: options.replay,
        maxInputs: options.maxInputs
    };
    const result = await garden.runChild(
        process.execPath,
        [workerPath],
        JSON.stringify(job),
        options.timeoutMs
    );
    return result;
}

async function shrinkMutant(mutant, result, options) {
    const signature = garden.failureSignature(result);
    if (!options.shrink) {
        return { source: mutant.source, steps: 0, signature: signature };
    }
    let current = mutant.source.split('\n');
    let steps = 0;
    let remaining = options.shrinkBudget;
    let changed = true;
    while (changed && remaining > 0) {
        changed = false;
        let i = 0;
        while (i < current.length && remaining > 0) {
            const candidateSource = current.slice(0, i).concat(current.slice(i + 1)).join('\n');
            remaining--;
            steps++;
            const next = await evaluateMutant({
                source: candidateSource,
                inputs: mutant.inputs,
                level: mutant.level,
                randomSeed: mutant.randomSeed
            }, options);
            if (garden.failureSignature(next) === signature) {
                current = candidateSource.split('\n');
                changed = true;
            } else {
                i++;
            }
        }
    }
    return { source: current.join('\n'), steps: steps, signature: signature };
}

async function main() {
    let options;
    try {
        options = garden.parseArguments(process.argv.slice(2));
    } catch (error) {
        process.stderr.write(error.message + '\n');
        process.exitCode = 1;
        return;
    }
    if (options.listMutators) {
        garden.mutators.forEach(function(mutator) {
            process.stdout.write(mutator.name + '\n');
        });
        return;
    }
    const corpus = filterCorpus(garden.loadCorpus(resourceDir), options.fixture);
    if (corpus.length === 0) {
        process.stderr.write('No fixtures matched --fixture\n');
        process.exitCode = 1;
        return;
    }
    const rng = new garden.Random(options.seed);
    const counts = {
        ok: 0,
        'compiler-error': 0,
        crash: 0,
        timeout: 0,
        invariant: 0,
        nondeterministic: 0,
        'replay-divergence': 0,
        skipped: 0
    };
    let artifactIndex = 0;
    for (let i = 0; i < options.count; i++) {
        const fixture = corpus[rng.integer(corpus.length)];
        let mutant;
        try {
            mutant = garden.mutateFixture(fixture, rng, allowedMutators(options.mutators), {
                maxAttempts: options.maxAttempts
            });
        } catch (error) {
            counts.skipped++;
            continue;
        }
        const result = await evaluateMutant(mutant, options);
        counts[result.kind] = (counts[result.kind] || 0) + 1;
        process.stdout.write(
            '#' + (i + 1) + ' ' + result.kind + ' ' + mutant.mutator + ' ' + mutant.fixtureName + '\n'
        );
        if (!garden.isInteresting(result)) {
            continue;
        }
        const minimized = await shrinkMutant(mutant, result, options);
        artifactIndex++;
        const dirName = garden.artifactDirName(minimized.signature, options.seed, artifactIndex);
        fs.mkdirSync(options.output, { recursive: true });
        garden.writeArtifacts(options.output, dirName, {
            'original.txt': mutant.source,
            'minimized.txt': minimized.source,
            'report.json': JSON.stringify({
                seed: options.seed,
                fixtureName: mutant.fixtureName,
                fixtureIndex: mutant.fixtureIndex,
                mutator: mutant.mutator,
                detail: mutant.detail,
                result: result,
                signature: minimized.signature,
                shrinkSteps: minimized.steps
            }, null, 2) + '\n',
            'regression.js': garden.formatRegression(
                'monster garden ' + options.seed + ' ' + artifactIndex,
                minimized.source
            )
        });
    }
    process.stdout.write(JSON.stringify(counts) + '\n');
}

main().catch(function(error) {
    process.stderr.write(error.stack + '\n');
    process.exitCode = 1;
});
```

- [ ] **Step 5: Re-run the garden tests**

Run: `node src/tests/monster_garden/tests.js`

Expected: `18/18 monster garden tests passed`

The one-mutant CLI test can take a few seconds because it boots the compiler twice. If it finds an interesting case and writes artifacts, the empty-directory assertion will fail. In that case keep seed `12345` and set `--mutator odd-whitespace` or another mutator that still yields `ok`/`compiler-error` for that seed. Do not delete the determinism assertion.

- [ ] **Step 6: Commit**

```bash
git add src/tests/monster_garden/run.js src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "$(cat <<'EOF'
Add monster garden parent runner with timeouts and shrinking.

EOF
)"
```

---

### Task 5: Docs, ignore rule, and smoke verification

**Files:**
- Modify: `.gitignore`
- Modify: `DEVELOPMENT.md`
- Modify: `src/tests/monster_garden/tests.js` only if a smoke assertion is still missing

- [ ] **Step 1: Ignore garden output**

Append this line to `.gitignore` if it is not already present:

```
.build/monster_garden/
```

- [ ] **Step 2: Document the commands**

Append this section to `DEVELOPMENT.md`:

```
## Compiler monster garden

Optional local fuzzer for compiler and runtime crashes. It is not part of the
normal test suite.

    node src/tests/monster_garden/tests.js
    node src/tests/monster_garden/run.js --list-mutators
    node src/tests/monster_garden/run.js --seed 12345 --count 20

A fixed seed reproduces the same mutants. Interesting cases are written under
`.build/monster_garden/` as `original.txt`, `minimized.txt`, `report.json`, and
`regression.js`. Paste `regression.js` into `testdata.js` or
`errormessage_testdata.js` only after you have confirmed it is a real defect.
```

- [ ] **Step 3: Run the garden unit tests**

Run: `node src/tests/monster_garden/tests.js`

Expected: all garden tests passed.

- [ ] **Step 4: Run the existing Node suite**

Run: `node src/tests/run_tests_node.js`

Expected: the same pass count as before this branch. The garden must not have changed compiler or fixture files.

- [ ] **Step 5: Smoke the CLI**

Run:

```
node src/tests/monster_garden/run.js --list-mutators
node src/tests/monster_garden/run.js --seed 12345 --count 5 --no-replay --timeout-ms 20000
```

Expected: mutator names print; the five-mutant run exits 0 and prints a JSON tally whose keys are the outcome kinds.

- [ ] **Step 6: Review the diff**

Confirm the only production-adjacent edits are `.gitignore` and `DEVELOPMENT.md`. `src/js/`, `src/editor.html`, `src/tests/resources/`, and `src/tests/run_tests_node.js` must be untouched.

- [ ] **Step 7: Commit**

```bash
git add .gitignore DEVELOPMENT.md
git commit -m "$(cat <<'EOF'
Document the compiler monster garden and ignore its artifacts.

EOF
)"
```

---

## Self-review

**Spec coverage**

| Spec requirement | Task |
| --- | --- |
| Seeded corpus + mutators + CLI parse | Task 1 |
| Inapplicable retry | Task 1 |
| Invariants, interestingness, shrink, artifact names | Task 2 |
| Worker JSON job/result, compiler-error vs crash | Task 3 |
| Fingerprint, replay, nondeterminism | Task 3 |
| Parent timeout, shrinking, artifacts, tally | Task 4 |
| Docs, ignore rule, no compiler edits | Task 5 |

**Placeholder scan:** none.

**Type consistency:** `kind` strings, `failureSignature`, `parseArguments` field names, and artifact filenames are the same in the spec, tests, and implementation snippets.
