# Garden Equivalence Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the garden to run semantics-preserving mutations and report `equivalence-break` when a still-valid game quietly behaves differently from the unmutated original.

**Architecture:** Every trial already runs the unmutated fixture as a baseline with identical inputs, and both runs return a `fingerprint`. Comparison is a pure function in `garden.js` over two values already in hand — no new worker code and no extra child process. Mutators gain an optional `equivalence` field (`'full'` or `'board'`); only those mutators are compared.

**Tech Stack:** Node.js built-ins only. `garden.js`, `run.js` and `tests.js` under `src/tests/monster_garden/`. Do not modify `src/js/compiler.js` or any engine source.

**Worktree:** `.worktrees/compiler-monster-garden`

**Spec:** `docs/superpowers/specs/2026-08-14-garden-issue-mined-mutators-design.md` (Group B, the oracle, shrinking, testing)

## Global Constraints

- Node built-ins only. No new dependencies.
- ES5-style function syntax with `const`/`let`, matching the existing files. No arrow functions in `garden.js` or `run.js`; the existing code uses `function() {}` throughout.
- `'use strict';` at the top of every file already; keep it.
- Run `node src/tests/monster_garden/tests.js` after every change. All tests must pass.
- Run `node src/tests/run_tests_node.js` before the final commit of each task that touches `garden.js`. It must not regress.
- Do not modify `src/js/compiler.js`, `src/js/engine.js`, `src/js/parser.js` or `src/js/debug.js`.
- The test at `src/tests/monster_garden/tests.js:136` asserts the exact list of mutator names in order. **Every task that adds a mutator must append its name to that list**, or the suite fails.

---

## File map

- Modify: `src/tests/monster_garden/garden.js` — `sectionBlocks`, `normaliseBoardNames`, `compareEquivalence`, `shrinkEquivalencePair`, the 8 new mutators, `equivalence-break` in `isInteresting`/`KNOWN_RESULT_KINDS`, randomness exclusion in `mutateFixture`, `equivalenceContext` propagation.
- Modify: `src/tests/monster_garden/run.js` — counts key, equivalence check after `attributeMonster`, shrink dispatch, `baseline.txt` artifact.
- Modify: `src/tests/monster_garden/tests.js` — `RICH_SAMPLE` fixture, unit tests, end-to-end test.

No new files. `garden.js` is already the home for every pure, testable helper; this follows that pattern rather than introducing a module the rest of the tool would have to learn about.

---

### Task 1: `sectionBlocks` helper

Several new mutators need the OBJECTS section split into per-object blocks. This task adds only the helper.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Produces: `sectionBlocks(body)` returning `{ header: string[], blocks: string[][] }`. `body` is the string a `mutateSection` callback receives, which begins with the section name line and its `====` underline. `header` is those leading lines; `blocks` is each run of non-blank lines after them.

- [ ] **Step 1: Write the failing test.** Add to `src/tests/monster_garden/tests.js`, immediately after the `corpusIndex is unique` test (around line 115):

```js
test('sectionBlocks splits a section body into header and object blocks', function() {
    const body = [
        'OBJECTS',
        '========',
        '',
        'Background',
        'black',
        '',
        'Player',
        'white',
        'white'
    ].join('\n');
    const parsed = garden.sectionBlocks(body);
    assert.deepStrictEqual(parsed.header, ['OBJECTS', '========']);
    assert.deepStrictEqual(parsed.blocks, [
        ['Background', 'black'],
        ['Player', 'white', 'white']
    ]);
});

test('sectionBlocks returns no blocks for a section that is only a header', function() {
    const parsed = garden.sectionBlocks('SOUNDS\n=========\n\n');
    assert.deepStrictEqual(parsed.header, ['SOUNDS', '=========']);
    assert.deepStrictEqual(parsed.blocks, []);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node src/tests/monster_garden/tests.js`
Expected: two `F` marks and `TypeError: garden.sectionBlocks is not a function`.

- [ ] **Step 3: Implement `sectionBlocks`.** Add to `src/tests/monster_garden/garden.js` directly after the `mutateSection` function (after line 130):

```js
function sectionBlocks(body) {
    const lines = body.split('\n');
    let i = 0;
    while (i < lines.length) {
        const trimmed = lines[i].trim();
        if (/^=+$/.test(trimmed) || SECTION_NAMES.indexOf(trimmed.toUpperCase()) >= 0) {
            i++;
            continue;
        }
        break;
    }
    const header = lines.slice(0, i);
    const blocks = [];
    let current = [];
    for (; i < lines.length; i++) {
        if (lines[i].trim() === '') {
            if (current.length) {
                blocks.push(current);
                current = [];
            }
            continue;
        }
        current.push(lines[i]);
    }
    if (current.length) {
        blocks.push(current);
    }
    return { header: header, blocks: blocks };
}
```

- [ ] **Step 4: Export it.** In the `module.exports` block at the end of `src/tests/monster_garden/garden.js`, add after `loadCorpus: loadCorpus,`:

```js
    sectionBlocks: sectionBlocks,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all tests pass, count increased by 2.

- [ ] **Step 6: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add sectionBlocks helper for splitting section bodies."
```

---

### Task 2: `normaliseBoardNames` and `compareEquivalence`

The oracle itself, as a pure function. No wiring yet.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `normaliseBoardNames(board, renames)` → string. `renames` maps new name to old name.
  - `compareEquivalence(mutator, baseline, mutantResult, context)` → `null` or `{ detail: string }`. `mutator` is an entry from the `mutators` array. `baseline` and `mutantResult` are worker results with `kind` and `fingerprint`. `context` is the mutant's `equivalenceContext` or `null`.

**Background the implementer needs.** `convertLevelToString` in `src/js/debug.js:16` renders each cell as the space-joined, alphabetically sorted list of object *names* present, followed by `:` and an index, but only the first time a given combination appears; later cells with the same combination emit just the index. So a cell is either `Player Wall:3,` or `3,`. A consistent rename therefore changes the board text even though the game is identical, which is why `board`-level mutators may supply a normaliser. The index numbering depends on the partition of cells, not on names, so it survives renaming untouched.

- [ ] **Step 1: Write the failing tests.** Add to `src/tests/monster_garden/tests.js` after the `sectionBlocks` tests:

```js
test('normaliseBoardNames maps names back and re-sorts each cell', function() {
    const board = 'CrateRenamed Player:0,1,\nBackground:1,0,\n';
    const normalised = garden.normaliseBoardNames(board, { CrateRenamed: 'Crate' });
    assert.strictEqual(normalised, 'Crate Player:0,1,\nBackground:1,0,\n');
});

test('normaliseBoardNames re-sorts when the new name sorts differently', function() {
    const board = 'Player Zebra:0,\n';
    const normalised = garden.normaliseBoardNames(board, { Zebra: 'Crate' });
    assert.strictEqual(normalised, 'Crate Player:0,\n');
});

test('normaliseBoardNames leaves empty cells alone', function() {
    assert.strictEqual(garden.normaliseBoardNames(':0,0,\n', { A: 'B' }), ':0,0,\n');
});

test('compareEquivalence ignores mutators that do not declare equivalence', function() {
    const plain = { name: 'plain', apply: function() { return null; } };
    const result = garden.compareEquivalence(
        plain,
        { kind: 'ok', fingerprint: 'a' },
        { kind: 'ok', fingerprint: 'b' },
        null
    );
    assert.strictEqual(result, null);
});

test('compareEquivalence makes no claim when the baseline is unhealthy', function() {
    const mutator = { name: 'm', equivalence: 'full', apply: function() { return null; } };
    const result = garden.compareEquivalence(
        mutator,
        { kind: 'crash', fingerprint: 'a' },
        { kind: 'ok', fingerprint: 'b' },
        null
    );
    assert.strictEqual(result, null);
});

test('compareEquivalence reports a full-fingerprint divergence', function() {
    const mutator = { name: 'm', equivalence: 'full', apply: function() { return null; } };
    assert.strictEqual(garden.compareEquivalence(
        mutator, { kind: 'ok', fingerprint: 'a' }, { kind: 'ok', fingerprint: 'a' }, null
    ), null);
    const broken = garden.compareEquivalence(
        mutator, { kind: 'ok', fingerprint: 'a' }, { kind: 'ok', fingerprint: 'b' }, null
    );
    assert(broken);
    assert.strictEqual(broken.detail, 'fingerprint differs');
});

test('compareEquivalence compares only the board for board-level mutators', function() {
    const mutator = { name: 'm', equivalence: 'board', apply: function() { return null; } };
    const baseline = { kind: 'ok', fingerprint: JSON.stringify({ board: 'Crate:0,\n', curlevel: 0 }) };
    const sameBoard = { kind: 'ok', fingerprint: JSON.stringify({ board: 'Crate:0,\n', curlevel: 9 }) };
    assert.strictEqual(garden.compareEquivalence(mutator, baseline, sameBoard, null), null);
    const otherBoard = { kind: 'ok', fingerprint: JSON.stringify({ board: 'Player:0,\n', curlevel: 0 }) };
    const broken = garden.compareEquivalence(mutator, baseline, otherBoard, null);
    assert(broken);
    assert.strictEqual(broken.detail, 'board differs');
});

test('compareEquivalence applies the rename normaliser before comparing boards', function() {
    const mutator = {
        name: 'm',
        equivalence: 'board',
        normalise: garden.normaliseBoardNames,
        apply: function() { return null; }
    };
    const baseline = { kind: 'ok', fingerprint: JSON.stringify({ board: 'Crate Player:0,\n' }) };
    const renamed = { kind: 'ok', fingerprint: JSON.stringify({ board: 'CrateRenamed Player:0,\n' }) };
    const context = { renames: { CrateRenamed: 'Crate' } };
    assert.strictEqual(garden.compareEquivalence(mutator, baseline, renamed, context), null);
});

test('compareEquivalence treats a compiler error on a valid transformation as a break', function() {
    const mutator = { name: 'm', equivalence: 'full', apply: function() { return null; } };
    const broken = garden.compareEquivalence(
        mutator, { kind: 'ok', fingerprint: 'a' }, { kind: 'compiler-error', fingerprint: 'x' }, null
    );
    assert(broken);
    assert(/compiler-error/.test(broken.detail));
});

test('compareEquivalence does not treat a new warning or a crash as a break', function() {
    const mutator = { name: 'm', equivalence: 'full', apply: function() { return null; } };
    assert.strictEqual(garden.compareEquivalence(
        mutator, { kind: 'ok', fingerprint: 'a' }, { kind: 'compiler-warning', fingerprint: 'x' }, null
    ), null);
    assert.strictEqual(garden.compareEquivalence(
        mutator, { kind: 'ok', fingerprint: 'a' }, { kind: 'crash', fingerprint: 'x' }, null
    ), null);
});

test('compareEquivalence makes no claim when a board is missing from a fingerprint', function() {
    const mutator = { name: 'm', equivalence: 'board', apply: function() { return null; } };
    const baseline = { kind: 'ok', fingerprint: JSON.stringify({ levelCount: 2 }) };
    const mutant = { kind: 'ok', fingerprint: JSON.stringify({ board: 'Crate:0,\n' }) };
    assert.strictEqual(garden.compareEquivalence(mutator, baseline, mutant, null), null);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: failures with `garden.normaliseBoardNames is not a function`.

- [ ] **Step 3: Implement both functions.** Add to `src/tests/monster_garden/garden.js` directly before `function isInteresting(` (before line 1259):

```js
function normaliseBoardNames(board, renames) {
    if (!renames) {
        return String(board);
    }
    return String(board).replace(/([^,:\n]*):/g, function(match, names) {
        if (names === '') {
            return match;
        }
        const mapped = names.split(' ').map(function(name) {
            return Object.prototype.hasOwnProperty.call(renames, name) ? renames[name] : name;
        });
        mapped.sort();
        return mapped.join(' ') + ':';
    });
}

function fingerprintBoard(result) {
    if (!result || typeof result.fingerprint !== 'string') {
        return null;
    }
    let parsed;
    try {
        parsed = JSON.parse(result.fingerprint);
    } catch (error) {
        return null;
    }
    if (!parsed || typeof parsed.board !== 'string') {
        return null;
    }
    return parsed.board;
}

// A crash, timeout or invariant is already interesting on its own, so this only
// speaks about the two cases the other classifiers cannot see: a clean run that
// answers differently, and a valid transformation that stops compiling. A new
// warning is not a break, because mutators that add declarations can legitimately
// provoke one.
function compareEquivalence(mutator, baseline, mutantResult, context) {
    if (!mutator || !mutator.equivalence || !baseline || !mutantResult) {
        return null;
    }
    if (baseline.kind !== 'ok') {
        return null;
    }
    if (mutantResult.kind === 'compiler-error') {
        return { detail: 'semantics-preserving mutation produced compiler-error' };
    }
    if (mutantResult.kind !== 'ok') {
        return null;
    }
    if (mutator.equivalence === 'full') {
        if (mutantResult.fingerprint !== baseline.fingerprint) {
            return { detail: 'fingerprint differs' };
        }
        return null;
    }
    const baselineBoard = fingerprintBoard(baseline);
    let mutantBoard = fingerprintBoard(mutantResult);
    if (baselineBoard === null || mutantBoard === null) {
        return null;
    }
    if (mutator.normalise) {
        mutantBoard = mutator.normalise(mutantBoard, context && context.renames);
    }
    if (mutantBoard !== baselineBoard) {
        return { detail: 'board differs' };
    }
    return null;
}
```

- [ ] **Step 4: Export both.** In `module.exports`, add after `isInteresting: isInteresting,`:

```js
    normaliseBoardNames: normaliseBoardNames,
    compareEquivalence: compareEquivalence,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the garden equivalence comparator and board normaliser."
```

---

### Task 3: Register `equivalence-break` as an interesting kind

**Files:**
- Modify: `src/tests/monster_garden/garden.js:1259-1270` (`isInteresting`), `garden.js:1392-1395` (`KNOWN_RESULT_KINDS`), `garden.js:1606` (summary line)
- Modify: `src/tests/monster_garden/run.js:119` (counts)
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Produces: the string `'equivalence-break'` as a recognised result kind and tally key.

- [ ] **Step 1: Write the failing test.** Add to `src/tests/monster_garden/tests.js` after the `compareEquivalence` tests:

```js
test('equivalence-break is a known, interesting result kind', function() {
    assert(garden.KNOWN_RESULT_KINDS.indexOf('equivalence-break') >= 0);
    assert.strictEqual(garden.isInteresting({ kind: 'equivalence-break' }), true);
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node src/tests/monster_garden/tests.js`
Expected: one `F`, `AssertionError` on the `indexOf` assertion.

- [ ] **Step 3: Add the kind in four places.**

In `src/tests/monster_garden/garden.js`, `isInteresting` currently ends:

```js
        || result.kind === 'semantic-mismatch';
```

Change it to:

```js
        || result.kind === 'semantic-mismatch'
        || result.kind === 'equivalence-break';
```

In the same file, `KNOWN_RESULT_KINDS` becomes:

```js
const KNOWN_RESULT_KINDS = [
    'ok', 'compiler-error', 'compiler-warning', 'crash',
    'invariant', 'nondeterministic', 'replay-divergence', 'semantic-mismatch',
    'equivalence-break'
];
```

At `garden.js:1606` the summary line ends with:

```js
        ' semantic-mismatch=' + (counts['semantic-mismatch'] || 0);
```

Change it to:

```js
        ' semantic-mismatch=' + (counts['semantic-mismatch'] || 0) +
        ' equivalence-break=' + (counts['equivalence-break'] || 0);
```

In `src/tests/monster_garden/run.js`, the counts object at line 119 contains `'semantic-mismatch': 0,`. Add directly after it:

```js
        'equivalence-break': 0,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/run.js src/tests/monster_garden/tests.js
git commit -m "Register equivalence-break as a garden result kind."
```

---

### Task 4: Randomness exclusion and `equivalenceContext` propagation

Games using `random` or `randomdir` draw from object sets whose iteration order depends on bit assignment, so a reordering mutator can legitimately change a draw. Without this guard the semantics half emits a steady drip of false monsters.

**Files:**
- Modify: `src/tests/monster_garden/garden.js:965-996` (`mutateFixture`)
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `mutator.equivalence` from Task 2's contract.
- Produces: `mutateFixture` result gains `equivalenceContext` (the mutator's, or `null`). Mutators declaring `equivalence` are filtered out for fixtures whose source uses randomness.

The randomness guard is tested here against a synthetic mutator, so this task's suite is green on its own. Task 8 adds the equivalent test against the real mutators once they exist.

- [ ] **Step 1: Write the failing test.** Add to `src/tests/monster_garden/tests.js` after the `equivalence-break` test:

```js
test('semantics-preserving mutators are skipped on fixtures that use randomness', function() {
    const randomFixture = {
        name: 'randomish',
        source: 'title T\n\n======\nRULES\n======\n\n[ Player ] -> [ randomDir Player ]\n',
        inputs: [],
        level: 0,
        randomSeed: null
    };
    const plainFixture = Object.assign({}, randomFixture, {
        source: 'title T\n\n======\nRULES\n======\n\n[ Player ] -> [ > Player ]\n'
    });
    const preserving = {
        name: 'fake-preserving',
        equivalence: 'full',
        apply: function(source) { return { source: source + 'author A\n', detail: 'edited' }; }
    };
    const saved = garden.mutators.slice();
    garden.mutators.length = 0;
    garden.mutators.push(preserving);
    try {
        assert.throws(function() {
            garden.mutateFixture(randomFixture, new garden.Random(7), null, { maxAttempts: 4 });
        }, /inapplicable/, 'a random fixture must offer no semantics-preserving mutator');
        const ok = garden.mutateFixture(plainFixture, new garden.Random(7), null, { maxAttempts: 4 });
        assert.strictEqual(ok.mutator, 'fake-preserving');
    } finally {
        garden.mutators.length = 0;
        for (let i = 0; i < saved.length; i++) {
            garden.mutators.push(saved[i]);
        }
    }
});

test('mutateFixture carries equivalenceContext through to the mutant', function() {
    const fixture = { name: 'f', source: 'title T\n', inputs: [], level: 0, randomSeed: null };
    const fake = [{
        name: 'fake-preserving',
        equivalence: 'board',
        apply: function(source) {
            return {
                source: source + 'author A\n',
                detail: 'renamed',
                equivalenceContext: { renames: { New: 'Old' } }
            };
        }
    }];
    const saved = garden.mutators.slice();
    garden.mutators.length = 0;
    garden.mutators.push(fake[0]);
    try {
        const mutant = garden.mutateFixture(fixture, new garden.Random(3), null, { maxAttempts: 2 });
        assert.deepStrictEqual(mutant.equivalenceContext, { renames: { New: 'Old' } });
    } finally {
        garden.mutators.length = 0;
        for (let i = 0; i < saved.length; i++) {
            garden.mutators.push(saved[i]);
        }
    }
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: both fail — the first because `mutateFixture` does not yet filter on randomness, the second on `equivalenceContext` being `undefined`.

- [ ] **Step 3: Implement.** In `src/tests/monster_garden/garden.js`, add above `function mutateFixture(`:

```js
const RANDOMNESS_RE = /\brandom(dir)?\b/i;
```

Replace the opening of `mutateFixture`:

```js
function mutateFixture(fixture, rng, mutatorNames, options) {
    const allowed = mutators.filter(function(mutator) {
        return !mutatorNames || mutatorNames.indexOf(mutator.name) >= 0;
    });
```

with:

```js
function mutateFixture(fixture, rng, mutatorNames, options) {
    // Randomness draws from object sets whose iteration order depends on bit
    // assignment, so a reordering mutation can legitimately change a draw.
    // Equivalence cannot be asserted for those fixtures.
    const usesRandomness = RANDOMNESS_RE.test(fixture.source || '');
    const allowed = mutators.filter(function(mutator) {
        if (mutatorNames && mutatorNames.indexOf(mutator.name) < 0) {
            return false;
        }
        if (mutator.equivalence && usesRandomness) {
            return false;
        }
        return true;
    });
```

Then in the returned object inside the same function, after `detail: applied.detail,`, add:

```js
                equivalenceContext: applied.equivalenceContext || null,
```

- [ ] **Step 4: Run the tests**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Exclude semantics-preserving mutators from random fixtures."
```

---

### Task 5: `shrinkEquivalencePair`

An equivalence-break is a relation between two sources. Deleting a line from the mutant but not the baseline manufactures a divergence for the wrong reason, so the shrinker must delete from both.

**Files:**
- Modify: `src/tests/monster_garden/garden.js` (new function beside `shrinkInteresting`, around line 1372)
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `compareEquivalence` from Task 2.
- Produces: `shrinkEquivalencePair(mutant, baselineSource, mutator, options)` → `Promise<{ source, baselineSource, steps, skipped }>`. `options` needs `shrink` (boolean), `shrinkBudget` (number) and `evaluate(job)` (async, as `shrinkInteresting` already uses). `skipped` is `true` when the mutant is not line-aligned with the baseline.

- [ ] **Step 1: Write the failing test.** Add to `src/tests/monster_garden/tests.js` after the `mutateFixture` tests:

```js
test('shrinkEquivalencePair deletes the same line from both sources', async function() {
    const mutator = { name: 'm', equivalence: 'full', apply: function() { return null; } };
    // Line "junk" is irrelevant to the divergence; line "keep" causes it.
    const evaluate = async function(job) {
        const diverges = job.source.indexOf('keep') >= 0;
        if (job.source.indexOf('MUTANT') >= 0) {
            return { kind: 'ok', fingerprint: diverges ? 'mutant' : 'same' };
        }
        return { kind: 'ok', fingerprint: 'same' };
    };
    const mutant = {
        source: 'MUTANT\njunk\nkeep\n',
        inputs: [],
        level: 0,
        randomSeed: null
    };
    const result = await garden.shrinkEquivalencePair(
        mutant,
        'BASE\njunk\nkeep\n',
        mutator,
        { shrink: true, shrinkBudget: 50, evaluate: evaluate }
    );
    assert.strictEqual(result.skipped, false);
    assert.strictEqual(result.source.indexOf('junk'), -1);
    assert.strictEqual(result.baselineSource.indexOf('junk'), -1);
    assert(result.source.indexOf('keep') >= 0);
    assert(result.baselineSource.indexOf('keep') >= 0);
});

test('shrinkEquivalencePair skips when the mutant is not line-aligned', async function() {
    const mutator = { name: 'm', equivalence: 'full', apply: function() { return null; } };
    const mutant = { source: 'a\nb\nc\n', inputs: [], level: 0, randomSeed: null };
    const result = await garden.shrinkEquivalencePair(
        mutant,
        'a\nb\n',
        mutator,
        { shrink: true, shrinkBudget: 50, evaluate: async function() { throw new Error('must not evaluate'); } }
    );
    assert.strictEqual(result.skipped, true);
    assert.strictEqual(result.source, 'a\nb\nc\n');
    assert.strictEqual(result.steps, 0);
});

test('shrinkEquivalencePair honours the shrink option being off', async function() {
    const mutator = { name: 'm', equivalence: 'full', apply: function() { return null; } };
    const mutant = { source: 'a\nb\n', inputs: [], level: 0, randomSeed: null };
    const result = await garden.shrinkEquivalencePair(
        mutant,
        'a\nb\n',
        mutator,
        { shrink: false, shrinkBudget: 50, evaluate: async function() { throw new Error('must not evaluate'); } }
    );
    assert.strictEqual(result.skipped, true);
    assert.strictEqual(result.steps, 0);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: `garden.shrinkEquivalencePair is not a function`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` directly after `shrinkInteresting` (after line 1372):

```js
// An equivalence-break is a relation between two sources, so a line must leave
// both or neither. That is only meaningful while the mutant stays line-aligned
// with the fixture; when it is not, shrinking is skipped rather than done wrongly.
async function shrinkEquivalencePair(mutant, baselineSource, mutator, options) {
    const unshrunk = {
        source: mutant.source,
        baselineSource: baselineSource,
        steps: 0,
        skipped: true
    };
    if (!options.shrink) {
        return unshrunk;
    }
    let mutantLines = mutant.source.split('\n');
    let baseLines = baselineSource.split('\n');
    if (mutantLines.length !== baseLines.length) {
        return unshrunk;
    }
    let steps = 0;
    let remaining = options.shrinkBudget;
    let changed = true;
    while (changed && remaining > 0) {
        changed = false;
        let i = 0;
        while (i < mutantLines.length && remaining > 0) {
            if (i === mutantLines.length - 1 && mutantLines[i] === '') {
                break;
            }
            const candidateMutant = mutantLines.slice(0, i).concat(mutantLines.slice(i + 1));
            const candidateBase = baseLines.slice(0, i).concat(baseLines.slice(i + 1));
            remaining--;
            steps++;
            const mutantResult = await options.evaluate({
                source: candidateMutant.join('\n'),
                inputs: mutant.inputs,
                level: mutant.level,
                randomSeed: mutant.randomSeed
            });
            const baselineResult = await options.evaluate({
                source: candidateBase.join('\n'),
                inputs: mutant.inputs,
                level: mutant.level,
                randomSeed: mutant.randomSeed
            });
            const stillBroken = compareEquivalence(
                mutator,
                baselineResult,
                mutantResult,
                mutant.equivalenceContext
            );
            if (stillBroken) {
                mutantLines = candidateMutant;
                baseLines = candidateBase;
                changed = true;
            } else {
                i++;
            }
        }
    }
    return {
        source: mutantLines.join('\n'),
        baselineSource: baseLines.join('\n'),
        steps: steps,
        skipped: false
    };
}
```

- [ ] **Step 4: Export it.** In `module.exports`, add after `shrinkInteresting: shrinkInteresting,`:

```js
    shrinkEquivalencePair: shrinkEquivalencePair,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Shrink equivalence-break monsters as a source pair."
```

---

### Task 6: Wire the oracle into `run.js`

**Files:**
- Modify: `src/tests/monster_garden/run.js:193` (after `attributeMonster`), `run.js:208` (shrink dispatch), `run.js:212-244` (artifacts)
- Test: `src/tests/monster_garden/tests.js` (end-to-end coverage arrives in Task 9)

**Interfaces:**
- Consumes: `garden.compareEquivalence`, `garden.shrinkEquivalencePair`, `mutant.equivalenceContext`.
- Produces: `equivalence-break` tallies and artifacts containing `baseline.txt`.

- [ ] **Step 1: Add a mutator lookup.** In `src/tests/monster_garden/run.js`, add near the other top-level helpers, after the `evaluateMutant` function (after line 62):

```js
function mutatorByName(name) {
    for (let i = 0; i < garden.mutators.length; i++) {
        if (garden.mutators[i].name === name) {
            return garden.mutators[i];
        }
    }
    return null;
}
```

- [ ] **Step 2: Compare after attribution.** Replace line 193:

```js
        const attributed = garden.attributeMonster(baseline, result);
```

with:

```js
        let attributed = garden.attributeMonster(baseline, result);
        const mutatorDef = mutatorByName(mutant.mutator);
        const equivalence = garden.compareEquivalence(
            mutatorDef,
            baseline,
            result,
            mutant.equivalenceContext
        );
        if (equivalence) {
            result.kind = 'equivalence-break';
            result.equivalenceDetail = equivalence.detail;
            attributed = { save: true, tally: 'equivalence-break', baseline: false };
        }
```

Note the `const` on line 193 becomes `let`.

- [ ] **Step 3: Dispatch shrinking.** Replace line 208:

```js
        const minimized = await shrinkMutant(mutant, result, options);
```

with:

```js
        let minimized;
        let minimizedBaseline = fixture.source;
        if (equivalence) {
            const paired = await garden.shrinkEquivalencePair(
                mutant,
                fixture.source,
                mutatorDef,
                Object.assign({}, options, {
                    evaluate: function(job) { return evaluateMutant(job, options); }
                })
            );
            minimizedBaseline = paired.baselineSource;
            minimized = {
                source: paired.source,
                steps: paired.steps,
                signature: garden.failureSignature(result),
                result: result,
                shrinkSkipped: paired.skipped
            };
        } else {
            minimized = await shrinkMutant(mutant, result, options);
        }
```

- [ ] **Step 4: Record both sources in the artifact.** In the `garden.writeArtifacts` call, add a key after `'minimized.txt': minimized.source,`:

```js
            'baseline.txt': minimizedBaseline,
```

and inside the `report.json` object, after `shrinkSteps: minimized.steps`, add:

```js
,
                shrinkSkipped: minimized.shrinkSkipped === true,
                equivalence: mutatorDef && mutatorDef.equivalence ? mutatorDef.equivalence : null,
                equivalenceDetail: result.equivalenceDetail || null
```

- [ ] **Step 5: Verify the tool still runs**

Run: `node src/tests/monster_garden/run.js --seed 4242 --count 12 --output /tmp/garden-wiring-check`
Expected: exit 0, a tally line that now includes `equivalence-break=0`.

Run: `node src/tests/monster_garden/tests.js`
Expected: same failures as before (Task 4's randomness test only), nothing new.

- [ ] **Step 6: Commit**

```bash
git add src/tests/monster_garden/run.js
git commit -m "Report and shrink equivalence-break monsters in the garden runner."
```

---

### Task 7: `RICH_SAMPLE` test fixture

The existing `SAMPLE` has an empty SOUNDS section and no win conditions, so several new mutators cannot apply to it. This adds a richer fixture used by Tasks 8 and 9.

**Files:**
- Modify: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Produces: module-level `RICH_SAMPLE` string, and a third fallback source in the mutator-coverage loop at `tests.js:190-204`.

- [ ] **Step 1: Add the fixture.** In `src/tests/monster_garden/tests.js`, add directly after the `SAMPLE` constant (after line 70):

```js
const RICH_SAMPLE = `title Garden Rich Sample

========
OBJECTS
========

Background
black

Player
white

Crate
brown

Target
yellow

=======
LEGEND
=======

. = Background
P = Player
C = Crate
T = Target
Pushable = Crate

=========
SOUNDS
=========

Crate move 12345
Player move 67890

================
COLLISIONLAYERS
================

Background
Target
Player, Crate

======
RULES
======

[ > Player | Pushable ] -> [ > Player | > Pushable ]

==============
WINCONDITIONS
==============

all Crate on Target
no Player on Target

=======
LEVELS
=======

.T.
PC.
`;
```

- [ ] **Step 2: Add it as a third fallback.** In the coverage loop at `tests.js:190-204`, the body currently retries with `winSource`. Replace the loop body's retry block:

```js
        if (!mutatorChangedJob(result, source, fixture)) {
            source = winSource;
            fixture = Object.assign({ source: source }, fixtureBase);
            result = mutator.apply(source, new garden.Random(100 + i), fixture);
        }
```

with:

```js
        if (!mutatorChangedJob(result, source, fixture)) {
            source = winSource;
            fixture = Object.assign({ source: source }, fixtureBase);
            result = mutator.apply(source, new garden.Random(100 + i), fixture);
        }
        if (!mutatorChangedJob(result, source, fixture)) {
            source = RICH_SAMPLE;
            fixture = Object.assign({ source: source }, fixtureBase);
            result = mutator.apply(source, new garden.Random(100 + i), fixture);
        }
```

- [ ] **Step 3: Verify the fixture compiles.** Write it to a file and run one garden trial against it to prove it is a valid game:

```bash
node -e "
const fs=require('fs');
const t=fs.readFileSync('src/tests/monster_garden/tests.js','utf8');
const m=/const RICH_SAMPLE = \`([\s\S]*?)\`;/.exec(t);
fs.writeFileSync('/tmp/rich.txt', m[1]);
console.log('extracted', m[1].length, 'chars');
"
echo '{"source":'"$(node -p "JSON.stringify(require('fs').readFileSync('/tmp/rich.txt','utf8'))")"',"inputs":[],"level":0,"randomSeed":null,"replay":false,"maxInputs":8}' | node src/tests/monster_garden/worker.js
```

Expected: JSON on stdout with `"kind":"ok"`. If it reports `compiler-error`, fix `RICH_SAMPLE` until it does not — every later task depends on this being a valid game.

- [ ] **Step 4: Run the tests**

Run: `node src/tests/monster_garden/tests.js`
Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add src/tests/monster_garden/tests.js
git commit -m "Add a richer garden test fixture with sounds and win conditions."
```

---

### Task 8: The eight semantics-preserving mutators

**Files:**
- Modify: `src/tests/monster_garden/garden.js` (eight functions plus registry entries)
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `sectionBlocks` (Task 1), `normaliseBoardNames` (Task 2), `mutateSection`, `mutateArrowRule`, `findSection` (existing).
- Produces: eight registry entries carrying `equivalence` and, for `rename-object`, `normalise`.

- [ ] **Step 1: Write the failing tests.** Add to `src/tests/monster_garden/tests.js` after the coverage test (after line 205):

```js
test('semantics-preserving mutators declare an equivalence level', function() {
    const expected = {
        'rename-object': 'board',
        'reorder-objects': 'board',
        'reorder-winconditions': 'full',
        'reorder-sounds': 'full',
        'inline-legend-synonym': 'full',
        'add-legend-alias': 'full',
        'add-unreachable-rule': 'full',
        'comment-reflow': 'full'
    };
    const names = Object.keys(expected);
    for (let i = 0; i < names.length; i++) {
        const mutator = garden.mutators.filter(function(m) { return m.name === names[i]; })[0];
        assert(mutator, names[i] + ' should be registered');
        assert.strictEqual(mutator.equivalence, expected[names[i]], names[i] + ' equivalence level');
    }
    const rename = garden.mutators.filter(function(m) { return m.name === 'rename-object'; })[0];
    assert.strictEqual(typeof rename.normalise, 'function');
});

test('rename-object renames consistently and never touches Background or Player', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'rename-object'; })[0];
    for (let seed = 0; seed < 25; seed++) {
        const result = mutator.apply(RICH_SAMPLE, new garden.Random(seed), {});
        if (!result) {
            continue;
        }
        assert(/\bBackground\b/.test(result.source), 'Background must survive seed ' + seed);
        assert(/\bPlayer\b/.test(result.source), 'Player must survive seed ' + seed);
        assert(!/\bBackgroundRenamed\b/.test(result.source));
        assert(!/\bPlayerRenamed\b/.test(result.source));
        const renames = result.equivalenceContext.renames;
        const newNames = Object.keys(renames);
        assert.strictEqual(newNames.length, 1);
        // The old name is gone everywhere and the new name appears in OBJECTS,
        // LEGEND, COLLISIONLAYERS and RULES alike.
        const oldName = renames[newNames[0]];
        assert(!new RegExp('\\b' + oldName + '\\b').test(result.source), 'old name gone, seed ' + seed);
    }
});

test('rename-object output normalises back to the original board', function() {
    const board = 'Crate Player:0,Target:1,\n';
    const renamed = 'CrateRenamed Player:0,Target:1,\n';
    assert.strictEqual(garden.normaliseBoardNames(renamed, { CrateRenamed: 'Crate' }), board);
});

test('reorder-objects permutes blocks without changing the object set', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'reorder-objects'; })[0];
    const result = mutator.apply(RICH_SAMPLE, new garden.Random(11), {});
    assert(result);
    assert.notStrictEqual(result.source, RICH_SAMPLE);
    const names = ['Background', 'Player', 'Crate', 'Target'];
    for (let i = 0; i < names.length; i++) {
        assert(new RegExp('^' + names[i] + '$', 'm').test(result.source), names[i] + ' should survive');
    }
});

test('reorder-winconditions and reorder-sounds keep the line count', function() {
    const names = ['reorder-winconditions', 'reorder-sounds'];
    for (let i = 0; i < names.length; i++) {
        const mutator = garden.mutators.filter(function(m) { return m.name === names[i]; })[0];
        const result = mutator.apply(RICH_SAMPLE, new garden.Random(20 + i), {});
        assert(result, names[i] + ' should apply to RICH_SAMPLE');
        assert.strictEqual(
            result.source.split('\n').length,
            RICH_SAMPLE.split('\n').length,
            names[i] + ' must stay line-aligned'
        );
    }
});

test('inline-legend-synonym stays line-aligned and removes the alias', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'inline-legend-synonym'; })[0];
    const result = mutator.apply(RICH_SAMPLE, new garden.Random(5), {});
    assert(result);
    assert.strictEqual(result.source.split('\n').length, RICH_SAMPLE.split('\n').length);
    assert(!/\bPushable\b/.test(result.source), 'the alias should be gone');
    assert(/\[ > Player \| Crate \]/.test(result.source), 'uses should be inlined');
});

test('comment-reflow stays line-aligned and puts the comment inside the brackets', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'comment-reflow'; })[0];
    const result = mutator.apply(RICH_SAMPLE, new garden.Random(9), {});
    assert(result);
    assert.strictEqual(result.source.split('\n').length, RICH_SAMPLE.split('\n').length);
    assert(/\[ \(garden\)/.test(result.source));
});

test('add-unreachable-rule declares its object in OBJECTS, layers and RULES', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'add-unreachable-rule'; })[0];
    const result = mutator.apply(RICH_SAMPLE, new garden.Random(2), {});
    assert(result);
    assert.strictEqual((result.source.match(/GardenGhost/g) || []).length, 3);
    assert(/\[ GardenGhost \] -> \[ \]/.test(result.source));
});

test('add-legend-alias defines the alias and routes a rule reference through it', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'add-legend-alias'; })[0];
    const result = mutator.apply(RICH_SAMPLE, new garden.Random(13), {});
    assert(result);
    assert(/^GardenAlias = \w+$/m.test(result.source));
    const rulesIndex = result.source.indexOf('RULES');
    assert(result.source.indexOf('GardenAlias', rulesIndex) > rulesIndex, 'alias should appear in RULES');
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: failures because none of the eight names are registered.

- [ ] **Step 3: Implement the mutators.** Add to `src/tests/monster_garden/garden.js` directly before `const mutators = [` (before line 900):

```js
// Background and Player are structurally required by the engine, so renaming
// either breaks the game and every resulting hit would be noise.
const UNRENAMEABLE = ['background', 'player'];

function objectNamesIn(source) {
    const section = findSection(source, 'OBJECTS');
    if (!section) {
        return [];
    }
    const body = section.lines.slice(section.start, section.end).join('\n');
    const parsed = sectionBlocks(body);
    const names = [];
    for (let i = 0; i < parsed.blocks.length; i++) {
        const first = parsed.blocks[i][0].trim().split(/\s+/)[0];
        if (/^[A-Za-z][A-Za-z0-9_]*$/.test(first)) {
            names.push(first);
        }
    }
    return names;
}

function renameObject(source, rng) {
    const names = objectNamesIn(source).filter(function(name) {
        return UNRENAMEABLE.indexOf(name.toLowerCase()) < 0;
    });
    if (names.length === 0) {
        return null;
    }
    const oldName = names[rng.integer(names.length)];
    const newName = oldName + 'Renamed';
    if (new RegExp('\\b' + newName + '\\b', 'i').test(source)) {
        return null;
    }
    const next = source.replace(new RegExp('\\b' + oldName + '\\b', 'g'), newName);
    if (next === source) {
        return null;
    }
    const renames = {};
    renames[newName] = oldName;
    return {
        source: next,
        detail: 'renamed ' + oldName + ' to ' + newName,
        equivalenceContext: { renames: renames }
    };
}

function reorderObjects(source, rng) {
    return mutateSection(source, 'OBJECTS', function(body) {
        const parsed = sectionBlocks(body);
        if (parsed.blocks.length < 2) {
            return null;
        }
        const blocks = parsed.blocks.slice();
        const a = rng.integer(blocks.length);
        let b = rng.integer(blocks.length);
        if (a === b) {
            b = (b + 1) % blocks.length;
        }
        const swap = blocks[a];
        blocks[a] = blocks[b];
        blocks[b] = swap;
        let out = parsed.header.slice();
        for (let i = 0; i < blocks.length; i++) {
            out.push('');
            out = out.concat(blocks[i]);
        }
        out.push('');
        return {
            source: out.join('\n'),
            detail: 'swapped object blocks ' + a + ' and ' + b
        };
    });
}

function reorderSectionLines(sectionName, label) {
    return function(source, rng) {
        return mutateSection(source, sectionName, function(body) {
            const lines = body.split('\n');
            const indexes = [];
            for (let i = 0; i < lines.length; i++) {
                const trimmed = lines[i].trim();
                if (trimmed === '' || /^=+$/.test(trimmed)) {
                    continue;
                }
                if (SECTION_NAMES.indexOf(trimmed.toUpperCase()) >= 0) {
                    continue;
                }
                indexes.push(i);
            }
            if (indexes.length < 2) {
                return null;
            }
            const first = rng.integer(indexes.length);
            let second = rng.integer(indexes.length);
            if (first === second) {
                second = (second + 1) % indexes.length;
            }
            const a = indexes[first];
            const b = indexes[second];
            const next = lines.slice();
            const swap = next[a];
            next[a] = next[b];
            next[b] = swap;
            return {
                source: next.join('\n'),
                detail: 'swapped ' + label + ' lines ' + a + ' and ' + b
            };
        });
    };
}

function inlineLegendSynonym(source, rng) {
    const section = findSection(source, 'LEGEND');
    if (!section) {
        return null;
    }
    const lines = section.lines.slice();
    const candidates = [];
    for (let i = section.start; i < section.end; i++) {
        const match = /^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([A-Za-z][A-Za-z0-9_]*)\s*$/.exec(lines[i]);
        if (match && match[1].length > 1) {
            candidates.push({ index: i, alias: match[1], target: match[2] });
        }
    }
    if (candidates.length === 0) {
        return null;
    }
    const pick = candidates[rng.integer(candidates.length)];
    // Blanking rather than deleting keeps the mutant line-aligned with the
    // fixture, which is what lets shrinkEquivalencePair reduce the pair.
    lines[pick.index] = '';
    const next = lines.join('\n').replace(new RegExp('\\b' + pick.alias + '\\b', 'g'), pick.target);
    if (next === source) {
        return null;
    }
    return {
        source: next,
        detail: 'inlined legend synonym ' + pick.alias + ' to ' + pick.target
    };
}

function addLegendAlias(source, rng) {
    const alias = 'GardenAlias';
    if (new RegExp('\\b' + alias + '\\b', 'i').test(source)) {
        return null;
    }
    const names = objectNamesIn(source).filter(function(name) {
        return UNRENAMEABLE.indexOf(name.toLowerCase()) < 0;
    });
    if (names.length === 0) {
        return null;
    }
    const target = names[rng.integer(names.length)];
    const withAlias = mutateSection(source, 'LEGEND', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\n' + alias + ' = ' + target + '\n',
            detail: ''
        };
    });
    if (!withAlias) {
        return null;
    }
    const routed = mutateSection(withAlias.source, 'RULES', function(body) {
        const re = new RegExp('\\b' + target + '\\b');
        if (!re.test(body)) {
            return null;
        }
        return { source: body.replace(re, alias), detail: '' };
    });
    if (!routed) {
        return null;
    }
    return {
        source: routed.source,
        detail: 'aliased ' + target + ' as ' + alias
    };
}

function addUnreachableRule(source) {
    const name = 'GardenGhost';
    if (new RegExp('\\b' + name + '\\b', 'i').test(source)) {
        return null;
    }
    const withObject = mutateSection(source, 'OBJECTS', function(body) {
        return {
            source: body.replace(/\s*$/, '') + '\n\n' + name + '\ntransparent\n',
            detail: ''
        };
    });
    if (!withObject) {
        return null;
    }
    const withLayer = mutateSection(withObject.source, 'COLLISIONLAYERS', function(body) {
        return { source: body.replace(/\s*$/, '') + '\n' + name + '\n', detail: '' };
    });
    if (!withLayer) {
        return null;
    }
    const withRule = mutateSection(withLayer.source, 'RULES', function(body) {
        return { source: body.replace(/\s*$/, '') + '\n[ ' + name + ' ] -> [ ]\n', detail: '' };
    });
    if (!withRule) {
        return null;
    }
    return {
        source: withRule.source,
        detail: 'added an unreachable rule for ' + name
    };
}

// A comment inside the brackets is legal and must be invisible. It is also the
// shape reported in issue #1128, "bad error if parenthetical inside rule".
function commentReflow(source, rng) {
    return mutateArrowRule(source, rng, function(line) {
        const bracket = line.indexOf('[');
        if (bracket < 0) {
            return null;
        }
        return {
            line: line.slice(0, bracket + 1) + ' (garden)' + line.slice(bracket + 1),
            detail: 'inserted a comment inside a rule'
        };
    });
}
```

- [ ] **Step 4: Register them.** In the `mutators` array in `src/tests/monster_garden/garden.js`, add after the final entry `{ name: 'prefix-chop', apply: prefixChop }`:

```js
,
    {
        name: 'rename-object',
        apply: renameObject,
        equivalence: 'board',
        normalise: normaliseBoardNames
    },
    { name: 'reorder-objects', apply: reorderObjects, equivalence: 'board' },
    {
        name: 'reorder-winconditions',
        apply: reorderSectionLines('WINCONDITIONS', 'wincondition'),
        equivalence: 'full'
    },
    {
        name: 'reorder-sounds',
        apply: reorderSectionLines('SOUNDS', 'sound'),
        equivalence: 'full'
    },
    { name: 'inline-legend-synonym', apply: inlineLegendSynonym, equivalence: 'full' },
    { name: 'add-legend-alias', apply: addLegendAlias, equivalence: 'full' },
    { name: 'add-unreachable-rule', apply: addUnreachableRule, equivalence: 'full' },
    { name: 'comment-reflow', apply: commentReflow, equivalence: 'full' }
```

`normaliseBoardNames` sits later in the file than the `mutators` array, which is fine: it is a `function` declaration, so it is hoisted and already bound when the array literal evaluates. Leave it where Task 2 put it.

- [ ] **Step 5: Update the mutator name list.** In `src/tests/monster_garden/tests.js`, the `expected` array at line 137 must gain the eight names in registry order, after `'prefix-chop'`:

```js
        'prefix-chop',
        'rename-object',
        'reorder-objects',
        'reorder-winconditions',
        'reorder-sounds',
        'inline-legend-synonym',
        'add-legend-alias',
        'add-unreachable-rule',
        'comment-reflow'
```

- [ ] **Step 6: Add the real-mutator randomness test.** Task 4 tested the guard against a synthetic mutator. Now that the real ones exist, add to `src/tests/monster_garden/tests.js`:

```js
test('the real semantics-preserving mutators are all skipped on random fixtures', function() {
    const randomFixture = {
        name: 'randomish',
        source: RICH_SAMPLE.replace('[ > Player | Pushable ]', '[ > Player | randomDir Pushable ]'),
        inputs: [],
        level: 0,
        randomSeed: null
    };
    const names = garden.mutators
        .filter(function(mutator) { return mutator.equivalence; })
        .map(function(mutator) { return mutator.name; });
    assert.strictEqual(names.length, 8);
    assert.throws(function() {
        garden.mutateFixture(randomFixture, new garden.Random(7), names, { maxAttempts: 6 });
    }, /inapplicable/);
});
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 8: Confirm the main suite has not regressed**

Run: `node src/tests/run_tests_node.js`
Expected: the same pass count as before this plan started. Record the number in the commit message.

- [ ] **Step 9: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add eight semantics-preserving garden mutators."
```

---

### Task 9: The silence test

The payoff. A run restricted to semantics-preserving mutators should find nothing, because the compiler is expected to be correct under these transformations. Any hit is a real bug.

**Files:**
- Modify: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the test.** Add to `src/tests/monster_garden/tests.js` before the `main()` function at the end:

```js
test('a semantics-preserving campaign finds no equivalence breaks', async function() {
    const output = fs.mkdtempSync(path.join(os.tmpdir(), 'garden-equivalence-'));
    const names = garden.mutators
        .filter(function(mutator) { return mutator.equivalence; })
        .map(function(mutator) { return mutator.name; })
        .join(',');
    const result = spawnSync(process.execPath, [
        path.join(__dirname, 'run.js'),
        '--seed', '31337',
        '--count', '40',
        '--mutator', names,
        '--timeout-ms', '8000',
        '--output', output
    ], { encoding: 'utf8' });
    assert.strictEqual(result.status, 0, result.stderr);
    const match = /equivalence-break=(\d+)/.exec(result.stdout);
    assert(match, 'summary should report an equivalence-break count:\n' + result.stdout);
    assert.strictEqual(
        match[1],
        '0',
        'semantics-preserving mutators must not break equivalence.\n' +
        'A non-zero count is a real compiler bug, not a test failure to paper over.\n' +
        'Artifacts are in ' + output + '\n' + result.stdout
    );
    fs.rmSync(output, { recursive: true, force: true });
});
```

- [ ] **Step 2: Run it**

Run: `node src/tests/monster_garden/tests.js`
Expected: passes.

**If it fails, do not weaken the test.** A non-zero count means one of two things: a genuine compiler bug, which is the whole point and should be reported to the user with the artifact directory; or a mutator that is not actually semantics-preserving, which is a bug in this plan's code. Diagnose which by reading `baseline.txt` and `minimized.txt` in the artifact directory and deciding whether the two programs really should behave identically. Report the finding rather than adjusting the assertion.

- [ ] **Step 3: Run a longer campaign by hand as a smoke check**

Run: `node src/tests/monster_garden/run.js --seed 99 --count 300 --mutator rename-object,reorder-objects,reorder-winconditions,reorder-sounds,inline-legend-synonym,add-legend-alias,add-unreachable-rule,comment-reflow --output /tmp/garden-silence`
Expected: exit 0 and `equivalence-break=0`. Report the tally line to the user either way.

- [ ] **Step 4: Confirm no runtime regression**

Run: `node src/tests/run_tests_node.js`
Expected: unchanged pass count.

- [ ] **Step 5: Commit**

```bash
git add src/tests/monster_garden/tests.js
git commit -m "Assert that semantics-preserving garden mutators find nothing."
```

---

## Documentation

- [ ] **Update the garden design doc.** In `docs/superpowers/specs/2026-08-14-compiler-monster-garden-design.md`, the tally list under "Command-line experience" reads:

```
`nondeterministic`, `replay-divergence`, `semantic-mismatch`, `baseline`, and
`skipped`.
```

Change it to:

```
`nondeterministic`, `replay-divergence`, `semantic-mismatch`, `equivalence-break`,
`baseline`, and `skipped`.
```

- [ ] **Commit**

```bash
git add docs/superpowers/specs/2026-08-14-compiler-monster-garden-design.md
git commit -m "Document the equivalence-break tally."
```
