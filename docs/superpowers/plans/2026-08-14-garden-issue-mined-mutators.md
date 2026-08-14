# Garden Issue-Mined Mutators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eight mutators drawn from families of bugs that recur in this project's closed issues, targeting language regions the existing 43 mutators do not reach.

**Architecture:** Each mutator is a small named function in `garden.js` returning `{ source, detail }` or `null`, registered in the `mutators` array — the pattern already established. Two shared helpers (`mutateRuleCell`, `cellObjectNames`) are added because five of the eight need to operate on an individual cell inside a rule's brackets, which no existing helper does.

**Tech Stack:** Node.js built-ins only. `garden.js` and `tests.js` under `src/tests/monster_garden/`.

**Worktree:** `.worktrees/compiler-monster-garden`

**Spec:** `docs/superpowers/specs/2026-08-14-garden-issue-mined-mutators-design.md` (Group A)

**Ordering:** Independent of `2026-08-14-garden-equivalence-oracle.md` and can land before or after it. The only interaction is `multi-fault` excluding mutators that declare `equivalence`; if the oracle plan has not landed, no mutator declares it and the filter is simply a no-op.

## Global Constraints

- Node built-ins only. No new dependencies.
- ES5-style function syntax with `const`/`let`, matching the existing files. No arrow functions.
- Run `node src/tests/monster_garden/tests.js` after every change. All tests must pass.
- Run `node src/tests/run_tests_node.js` before the final commit. It must not regress.
- Do not modify `src/js/compiler.js`, `src/js/engine.js`, `src/js/parser.js` or `src/js/debug.js`.
- The test at `src/tests/monster_garden/tests.js:136` asserts the exact list of mutator names in order. **Every task that adds a mutator must append its name to that list**, or the suite fails.
- These mutators produce *invalid or damaging* programs on purpose. They must never declare an `equivalence` field.

---

## File map

- Modify: `src/tests/monster_garden/garden.js` — `cellObjectNames`, `mutateRuleCell`, eight mutator functions, eight registry entries.
- Modify: `src/tests/monster_garden/tests.js` — per-mutator tests plus the name-list update.

No new files, following the existing structure.

---

### Task 1: `cellObjectNames` and `mutateRuleCell` helpers

Five of the eight mutators need to reach an individual cell inside a rule's brackets. `mutateArrowRule` gives a whole line; nothing reaches a cell.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Produces:
  - `cellObjectNames(text)` → `string[]`. Identifier-shaped tokens that are not direction or matching qualifiers.
  - `mutateRuleCell(source, rng, fn)` → `{ source, detail }` or `null`. `fn(cellText, rng)` returns `{ text, detail }` or `null`, where `text` replaces that one cell's contents.

- [ ] **Step 1: Write the failing tests.** Add to `src/tests/monster_garden/tests.js` after the `corpusIndex is unique` test (around line 115):

```js
test('cellObjectNames returns object tokens and drops qualifiers', function() {
    assert.deepStrictEqual(garden.cellObjectNames(' > Player | Wall '), ['Player', 'Wall']);
    assert.deepStrictEqual(garden.cellObjectNames(' no moving Crate '), ['Crate']);
    assert.deepStrictEqual(garden.cellObjectNames('  '), []);
});

test('mutateRuleCell replaces exactly one cell in one rule', function() {
    const result = garden.mutateRuleCell(SAMPLE, new garden.Random(4), function(cellText) {
        return { text: ' Replaced ', detail: 'replaced a cell' };
    });
    assert(result);
    assert.strictEqual(result.detail, 'replaced a cell');
    assert(/\[[^\]]*Replaced[^\]]*\]/.test(result.source));
    assert.strictEqual(
        result.source.split('\n').length,
        SAMPLE.split('\n').length,
        'cell edits stay on one line'
    );
});

test('mutateRuleCell reports inapplicable when the callback declines', function() {
    const result = garden.mutateRuleCell(SAMPLE, new garden.Random(4), function() {
        return null;
    });
    assert.strictEqual(result, null);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: `garden.cellObjectNames is not a function`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` directly after `mutateArrowRule` (after line 458):

```js
const CELL_QUALIFIERS = [
    'no', 'moving', 'stationary', 'perpendicular', 'parallel',
    'horizontal', 'vertical', 'orthogonal', 'up', 'down', 'left', 'right',
    'action', 'randomdir', 'random'
];

function cellObjectNames(text) {
    return String(text).trim().split(/\s+/).filter(function(token) {
        if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(token)) {
            return false;
        }
        return CELL_QUALIFIERS.indexOf(token.toLowerCase()) < 0;
    });
}

function mutateRuleCell(source, rng, fn) {
    return mutateArrowRule(source, rng, function(line, ruleRng) {
        const match = /\[([^\]]*)\]/.exec(line);
        if (!match) {
            return null;
        }
        const cells = match[1].split('|');
        const index = ruleRng.integer(cells.length);
        const next = fn(cells[index], ruleRng);
        if (!next) {
            return null;
        }
        cells[index] = next.text;
        return {
            line: line.slice(0, match.index) + '[' + cells.join('|') + ']' +
                line.slice(match.index + match[0].length),
            detail: next.detail
        };
    });
}
```

- [ ] **Step 4: Export them.** In `module.exports`, add after `loadCorpus: loadCorpus,`:

```js
    cellObjectNames: cellObjectNames,
    mutateRuleCell: mutateRuleCell,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add rule-cell helpers for garden mutators."
```

---

### Task 2: `no-x-with-x`

Issues #1169, #1136, #1071, #762 all report trouble when an object is negated in a cell that also contains it. `injectNo` prepends a hardcoded `no Player` to the first cell and never creates that contradiction.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `mutateRuleCell`, `cellObjectNames` from Task 1.
- Produces: registry entry `{ name: 'no-x-with-x', apply: noXWithX }`.

- [ ] **Step 1: Write the failing test.** Add to `src/tests/monster_garden/tests.js` after the coverage test (after line 205):

```js
test('no-x-with-x negates an object that is present in the same cell', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'no-x-with-x'; })[0];
    assert(mutator, 'no-x-with-x should be registered');
    let sawDoubled = false;
    let sawSingle = false;
    for (let seed = 0; seed < 30; seed++) {
        const result = mutator.apply(SAMPLE, new garden.Random(seed), {});
        if (!result) {
            continue;
        }
        const cell = /\[([^\]]*)\]/.exec(result.source)[1];
        const negated = /no\s+([A-Za-z][A-Za-z0-9_]*)/.exec(cell);
        assert(negated, 'seed ' + seed + ' should negate something: ' + cell);
        assert(
            new RegExp('\\b' + negated[1] + '\\b').test(cell.replace(/no\s+[A-Za-z][A-Za-z0-9_]*/g, '')),
            'the negated object should also be present unnegated, seed ' + seed
        );
        if (/no\s+(\w+)\s+no\s+\1/.test(cell)) {
            sawDoubled = true;
        } else {
            sawSingle = true;
        }
    }
    assert(sawSingle, 'expected at least one single negation across seeds');
    assert(sawDoubled, 'expected at least one doubled negation across seeds');
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node src/tests/monster_garden/tests.js`
Expected: `no-x-with-x should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
// Issues #1169, #1136, #1071, #762: an object negated in a cell that also
// contains it. injectNo never produces this contradiction.
function noXWithX(source, rng) {
    return mutateRuleCell(source, rng, function(cellText, cellRng) {
        const objects = cellObjectNames(cellText);
        if (objects.length === 0) {
            return null;
        }
        const object = objects[cellRng.integer(objects.length)];
        const twice = cellRng.integer(2) === 0;
        const addition = twice ? ' no ' + object + ' no ' + object : ' no ' + object;
        return {
            text: cellText.replace(/\s*$/, '') + addition + ' ',
            detail: 'negated ' + object + ' beside itself' + (twice ? ' twice' : '')
        };
    });
}
```

- [ ] **Step 4: Register it** in the `mutators` array, after the last existing entry:

```js
,
    { name: 'no-x-with-x', apply: noXWithX }
```

- [ ] **Step 5: Update the name list** in `src/tests/monster_garden/tests.js` at line 137 — append `'no-x-with-x'` after the last entry.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the no-x-with-x garden mutator (issues #1169, #1136, #1071, #762)."
```

---

### Task 3: `relative-direction-cell`

Issues #682, #498, #496 and #941 all report spurious errors from relative direction qualifiers on individual cell objects. `directionPrefixSalad` prepends one fixed string to the whole rule and never touches a cell.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `mutateRuleCell`, `cellObjectNames`.
- Produces: registry entry `{ name: 'relative-direction-cell', apply: relativeDirectionCell }`.

Combining a cell qualifier with a conflicting rule-level prefix is deliberately left to `multi-fault` (Task 9) composing this mutator with `direction-prefix-salad`, rather than duplicating that logic here.

- [ ] **Step 1: Write the failing test.** Add to `src/tests/monster_garden/tests.js`:

```js
test('relative-direction-cell qualifies a cell object with a relative direction', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'relative-direction-cell'; })[0];
    assert(mutator, 'relative-direction-cell should be registered');
    const seen = {};
    for (let seed = 0; seed < 40; seed++) {
        const result = mutator.apply(SAMPLE, new garden.Random(seed), {});
        if (!result) {
            continue;
        }
        const match = /(perpendicular|parallel|vertical|horizontal|orthogonal|moving|stationary)\s+[A-Za-z]/
            .exec(result.source);
        assert(match, 'seed ' + seed + ' should add a qualifier');
        seen[match[1]] = true;
        assert.strictEqual(result.source.split('\n').length, SAMPLE.split('\n').length);
    }
    assert(Object.keys(seen).length >= 3, 'expected several distinct qualifiers across seeds');
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node src/tests/monster_garden/tests.js`
Expected: `relative-direction-cell should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
const RELATIVE_DIRECTIONS = [
    'perpendicular', 'parallel', 'vertical', 'horizontal',
    'orthogonal', 'moving', 'stationary'
];

// Issues #682, #498, #496, #941: relative qualifiers on individual cell objects.
function relativeDirectionCell(source, rng) {
    return mutateRuleCell(source, rng, function(cellText, cellRng) {
        const objects = cellObjectNames(cellText);
        if (objects.length === 0) {
            return null;
        }
        const object = objects[cellRng.integer(objects.length)];
        const qualifier = RELATIVE_DIRECTIONS[cellRng.integer(RELATIVE_DIRECTIONS.length)];
        const next = cellText.replace(new RegExp('\\b' + object + '\\b'), qualifier + ' ' + object);
        if (next === cellText) {
            return null;
        }
        return { text: next, detail: 'qualified ' + object + ' with ' + qualifier };
    });
}
```

- [ ] **Step 4: Register it** after `{ name: 'no-x-with-x', apply: noXWithX }`:

```js
,
    { name: 'relative-direction-cell', apply: relativeDirectionCell }
```

- [ ] **Step 5: Update the name list** in `tests.js` — append `'relative-direction-cell'`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the relative-direction-cell garden mutator (issues #682, #498, #496, #941)."
```

---

### Task 4: `same-layer-cell`

Issues #735, #605 and #734 report exceptions and missing diagnostics when a rule cell demands two objects that can never overlap. `layerDoubleBook` edits the COLLISIONLAYERS section and never a rule cell.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `mutateRuleCell`, `findSection`.
- Produces: registry entry `{ name: 'same-layer-cell', apply: sameLayerCell }`.

- [ ] **Step 1: Write the failing test.** `SAMPLE` has the line `Player, Wall` in COLLISIONLAYERS, which is the pair this test relies on.

```js
test('same-layer-cell puts two objects from one collision layer in one cell', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'same-layer-cell'; })[0];
    assert(mutator, 'same-layer-cell should be registered');
    const result = mutator.apply(SAMPLE, new garden.Random(6), {});
    assert(result);
    const cells = /\[([^\]]*)\]/.exec(result.source)[1].split('|');
    const crowded = cells.filter(function(cell) {
        return /\bPlayer\b/.test(cell) && /\bWall\b/.test(cell);
    });
    assert.strictEqual(crowded.length, 1, 'exactly one cell should hold both: ' + cells.join(' | '));
    assert(/same-layer/.test(result.detail));
});

test('same-layer-cell declines when no layer holds two objects', function() {
    const singleLayer = SAMPLE.replace('Background\nPlayer, Wall', 'Background\nPlayer\nWall');
    const mutator = garden.mutators.filter(function(m) { return m.name === 'same-layer-cell'; })[0];
    assert.strictEqual(mutator.apply(singleLayer, new garden.Random(6), {}), null);
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: `same-layer-cell should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
// Issues #735, #605, #734: a rule cell demanding two objects that can never
// overlap. layerDoubleBook only edits the COLLISIONLAYERS section.
function sameLayerCell(source, rng) {
    const section = findSection(source, 'COLLISIONLAYERS');
    if (!section) {
        return null;
    }
    const pairs = [];
    for (let i = section.start; i < section.end; i++) {
        const trimmed = section.lines[i].trim();
        if (trimmed === '' || /^=+$/.test(trimmed) || trimmed.toUpperCase() === 'COLLISIONLAYERS') {
            continue;
        }
        const parts = trimmed.split(',').map(function(part) {
            return part.trim();
        }).filter(function(part) {
            return /^[A-Za-z][A-Za-z0-9_]*$/.test(part);
        });
        for (let a = 0; a < parts.length; a++) {
            for (let b = a + 1; b < parts.length; b++) {
                pairs.push([parts[a], parts[b]]);
            }
        }
    }
    if (pairs.length === 0) {
        return null;
    }
    const pair = pairs[rng.integer(pairs.length)];
    return mutateRuleCell(source, rng, function() {
        return {
            text: ' ' + pair[0] + ' ' + pair[1] + ' ',
            detail: 'same-layer ' + pair[0] + ' and ' + pair[1] + ' in one cell'
        };
    });
}
```

- [ ] **Step 4: Register it** after the `relative-direction-cell` entry:

```js
,
    { name: 'same-layer-cell', apply: sameLayerCell }
```

- [ ] **Step 5: Update the name list** in `tests.js` — append `'same-layer-cell'`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the same-layer-cell garden mutator (issues #735, #605, #734)."
```

---

### Task 5: `property-in-concrete-slot`

Issues #929, #495, #812 and #824 report exceptions and missing diagnostics when a property or aggregate name appears where a concrete object is required. `background-as-aggregate` and `sound-on-property` are two fixed instances; this generalises them.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `findSection`, `mutateSection`, `cellObjectNames`.
- Produces: registry entry `{ name: 'property-in-concrete-slot', apply: propertyInConcreteSlot }`.

`SAMPLE` defines `Obstacle = Player or Wall` in LEGEND, which is the property this test relies on.

- [ ] **Step 1: Write the failing test.**

```js
test('property-in-concrete-slot substitutes an or-property for a concrete object', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'property-in-concrete-slot'; })[0];
    assert(mutator, 'property-in-concrete-slot should be registered');
    let applied = 0;
    for (let seed = 0; seed < 30; seed++) {
        const result = mutator.apply(SAMPLE, new garden.Random(seed), {});
        if (!result) {
            continue;
        }
        applied++;
        assert.notStrictEqual(result.source, SAMPLE);
        assert(/Obstacle/.test(result.source));
        assert(/replaced .* with property Obstacle/.test(result.detail), result.detail);
    }
    assert(applied > 0, 'expected the mutator to apply for at least one seed');
});

test('property-in-concrete-slot declines when the legend defines no or-property', function() {
    const noProperty = SAMPLE.replace('Obstacle = Player or Wall\n', '');
    const mutator = garden.mutators.filter(function(m) { return m.name === 'property-in-concrete-slot'; })[0];
    for (let seed = 0; seed < 10; seed++) {
        assert.strictEqual(mutator.apply(noProperty, new garden.Random(seed), {}), null);
    }
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: `property-in-concrete-slot should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
// Issues #929, #495, #812, #824: a property or aggregate where a concrete
// object is required. Generalises background-as-aggregate and sound-on-property.
function propertyInConcreteSlot(source, rng) {
    const section = findSection(source, 'LEGEND');
    if (!section) {
        return null;
    }
    const properties = [];
    for (let i = section.start; i < section.end; i++) {
        const match = /^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$/.exec(section.lines[i]);
        if (match && match[1].length > 1 && /\bor\b/i.test(match[2])) {
            properties.push(match[1]);
        }
    }
    if (properties.length === 0) {
        return null;
    }
    const property = properties[rng.integer(properties.length)];
    const targets = ['RULES', 'WINCONDITIONS', 'SOUNDS', 'COLLISIONLAYERS'];
    for (let i = targets.length - 1; i > 0; i--) {
        const j = rng.integer(i + 1);
        const swap = targets[i];
        targets[i] = targets[j];
        targets[j] = swap;
    }
    for (let i = 0; i < targets.length; i++) {
        const applied = mutateSection(source, targets[i], function(body) {
            const names = cellObjectNames(body.replace(/[\[\]|,>]/g, ' ')).filter(function(name) {
                if (name.toLowerCase() === property.toLowerCase()) {
                    return false;
                }
                return SECTION_NAMES.indexOf(name.toUpperCase()) < 0;
            });
            if (names.length === 0) {
                return null;
            }
            const victim = names[rng.integer(names.length)];
            const re = new RegExp('\\b' + victim + '\\b');
            if (!re.test(body)) {
                return null;
            }
            return {
                source: body.replace(re, property),
                detail: 'replaced ' + victim + ' with property ' + property + ' in ' + targets[i]
            };
        });
        if (applied) {
            return applied;
        }
    }
    return null;
}
```

- [ ] **Step 4: Register it** after the `same-layer-cell` entry:

```js
,
    { name: 'property-in-concrete-slot', apply: propertyInConcreteSlot }
```

- [ ] **Step 5: Update the name list** in `tests.js` — append `'property-in-concrete-slot'`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the property-in-concrete-slot garden mutator (issues #929, #495, #812, #824)."
```

---

### Task 6: `rigid-prefix`

Issues #952, #1118 and #869 concern rigid bodies. The word `rigid` currently appears only inside `directionPrefixSalad`'s fixed prefix string, always alongside three other qualifiers, so rigid behaviour is never exercised on its own.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `mutateArrowRule`.
- Produces: registry entry `{ name: 'rigid-prefix', apply: rigidPrefix }`.

- [ ] **Step 1: Write the failing test.**

```js
test('rigid-prefix makes a rule rigid, sometimes late rigid', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'rigid-prefix'; })[0];
    assert(mutator, 'rigid-prefix should be registered');
    let sawPlain = false;
    let sawLate = false;
    for (let seed = 0; seed < 20; seed++) {
        const result = mutator.apply(SAMPLE, new garden.Random(seed), {});
        assert(result, 'seed ' + seed + ' should apply');
        assert(/\brigid\b/.test(result.source));
        if (/^late rigid /m.test(result.source)) {
            sawLate = true;
        } else if (/^rigid /m.test(result.source)) {
            sawPlain = true;
        }
    }
    assert(sawPlain, 'expected a bare rigid prefix across seeds');
    assert(sawLate, 'expected a late rigid prefix across seeds');
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node src/tests/monster_garden/tests.js`
Expected: `rigid-prefix should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
// Issues #952, #1118, #869: rigid bodies. directionPrefixSalad only ever emits
// rigid alongside three other qualifiers, so it never isolates rigid behaviour.
function rigidPrefix(source, rng) {
    return mutateArrowRule(source, rng, function(line, ruleRng) {
        const prefix = ruleRng.integer(2) === 0 ? 'rigid ' : 'late rigid ';
        return { line: prefix + line, detail: 'prefixed ' + prefix.trim() };
    });
}
```

- [ ] **Step 4: Register it** after the `property-in-concrete-slot` entry:

```js
,
    { name: 'rigid-prefix', apply: rigidPrefix }
```

- [ ] **Step 5: Update the name list** in `tests.js` — append `'rigid-prefix'`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the rigid-prefix garden mutator (issues #952, #1118, #869)."
```

---

### Task 7: `sprite-matrix-resize`

Issues #973 and #927 report that sprites of sizes other than 5x5 are inconsistently accepted and rendered. `spriteMatrixNoise` flips pixels within a matrix and never changes its shape.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `mutateSection`.
- Produces: registry entry `{ name: 'sprite-matrix-resize', apply: spriteMatrixResize }`.

`SAMPLE` has no sprite matrices, so this test builds one.

- [ ] **Step 1: Write the failing test.**

```js
const MATRIX_SAMPLE = SAMPLE.replace('Player\nwhite\n', [
    'Player',
    'white black',
    '00000',
    '01110',
    '01110',
    '01110',
    '00000',
    ''
].join('\n'));

test('sprite-matrix-resize changes the shape of a sprite matrix', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'sprite-matrix-resize'; })[0];
    assert(mutator, 'sprite-matrix-resize should be registered');
    const shapes = {};
    for (let seed = 0; seed < 30; seed++) {
        const result = mutator.apply(MATRIX_SAMPLE, new garden.Random(seed), {});
        if (!result) {
            continue;
        }
        assert.notStrictEqual(result.source, MATRIX_SAMPLE);
        const rows = result.source.split('\n').filter(function(line) {
            return /^[01.]{2,}$/.test(line.trim());
        });
        const widths = rows.map(function(row) { return row.trim().length; });
        const ragged = widths.some(function(width) { return width !== 5; });
        shapes[rows.length !== 5 || ragged ? 'changed' : 'same'] = true;
    }
    assert(shapes.changed, 'expected at least one seed to change the matrix shape');
});

test('sprite-matrix-resize declines when there is no sprite matrix', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'sprite-matrix-resize'; })[0];
    for (let seed = 0; seed < 10; seed++) {
        assert.strictEqual(mutator.apply(SAMPLE, new garden.Random(seed), {}), null);
    }
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node src/tests/monster_garden/tests.js`
Expected: `sprite-matrix-resize should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
// Issues #973, #927: sprites that are not 5x5. spriteMatrixNoise only flips
// pixels and never changes the shape.
function spriteMatrixResize(source, rng) {
    return mutateSection(source, 'OBJECTS', function(body) {
        const lines = body.split('\n');
        const indexes = [];
        for (let i = 0; i < lines.length; i++) {
            if (/^[0-9.]{2,}$/.test(lines[i].trim())) {
                indexes.push(i);
            }
        }
        if (indexes.length === 0) {
            return null;
        }
        const index = indexes[rng.integer(indexes.length)];
        const next = lines.slice();
        const mode = rng.integer(3);
        let detail;
        if (mode === 0) {
            next.splice(index, 1);
            detail = 'dropped a sprite matrix row';
        } else if (mode === 1) {
            next.splice(index, 0, next[index]);
            detail = 'duplicated a sprite matrix row';
        } else {
            next[index] = next[index].replace(/\s*$/, '') + '0';
            detail = 'widened a sprite matrix row';
        }
        return { source: next.join('\n'), detail: detail };
    });
}
```

- [ ] **Step 4: Register it** after the `rigid-prefix` entry:

```js
,
    { name: 'sprite-matrix-resize', apply: spriteMatrixResize }
```

- [ ] **Step 5: Update the name list** in `tests.js` — append `'sprite-matrix-resize'`.

- [ ] **Step 6: Handle the coverage loop.** The test at `tests.js:136` walks every mutator against `SAMPLE` and then `winSource`. Neither has a sprite matrix, so `sprite-matrix-resize` will fail that test. Add `MATRIX_SAMPLE` as a further fallback in the loop, after the `winSource` retry:

```js
        if (!mutatorChangedJob(result, source, fixture)) {
            source = MATRIX_SAMPLE;
            fixture = Object.assign({ source: source }, fixtureBase);
            result = mutator.apply(source, new garden.Random(100 + i), fixture);
        }
```

`MATRIX_SAMPLE` must be declared before the coverage test that uses it; move its `const` up to sit beside `SAMPLE` near line 70.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the sprite-matrix-resize garden mutator (issues #973, #927)."
```

---

### Task 8: `restart-again-message`

Issues #774, #981 and #341 report hangs and crashes from restart, again and message interacting. The garden has `inject-again-loop` and `message-sandwich` separately and never combines them, and never puts a restart in the input tape alongside them.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: `mutateArrowRule`.
- Produces: registry entry `{ name: 'restart-again-message', apply: restartAgainMessage }`. This mutator returns an `inputs` array in addition to `source`, which `mutateFixture` already propagates.

- [ ] **Step 1: Write the failing test.**

```js
test('restart-again-message combines a turn command, a message and a tape restart', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'restart-again-message'; })[0];
    assert(mutator, 'restart-again-message should be registered');
    const fixture = { source: SAMPLE, inputs: [0, 3], level: 0, randomSeed: null };
    const seen = {};
    for (let seed = 0; seed < 30; seed++) {
        const result = mutator.apply(SAMPLE, new garden.Random(seed), fixture);
        assert(result, 'seed ' + seed + ' should apply');
        const command = /(again|restart|checkpoint|win|cancel)\s+message garden/.exec(result.source);
        assert(command, 'seed ' + seed + ' should append a command and a message');
        seen[command[1]] = true;
        assert(Array.isArray(result.inputs));
        assert.strictEqual(result.inputs.length, 3);
        assert(result.inputs.indexOf('restart') >= 0, 'restart should be spliced into the tape');
    }
    assert(Object.keys(seen).length >= 3, 'expected several distinct turn commands across seeds');
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node src/tests/monster_garden/tests.js`
Expected: `restart-again-message should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
const TURN_COMMANDS = ['again', 'restart', 'checkpoint', 'win', 'cancel'];

// Issues #774, #981, #341: restart, again and message interacting. The garden
// has inject-again-loop and message-sandwich but never combines them, and never
// puts a restart in the tape alongside them.
function restartAgainMessage(source, rng, fixture) {
    const command = TURN_COMMANDS[rng.integer(TURN_COMMANDS.length)];
    const applied = mutateArrowRule(source, rng, function(line) {
        return {
            line: line + ' ' + command + ' message garden',
            detail: 'appended ' + command + ' and a message'
        };
    });
    if (!applied) {
        return null;
    }
    const inputs = (fixture && fixture.inputs ? fixture.inputs : []).slice();
    inputs.splice(rng.integer(inputs.length + 1), 0, 'restart');
    return {
        source: applied.source,
        detail: applied.detail + ' with a restart in the tape',
        inputs: inputs
    };
}
```

- [ ] **Step 4: Register it** after the `sprite-matrix-resize` entry:

```js
,
    { name: 'restart-again-message', apply: restartAgainMessage }
```

- [ ] **Step 5: Update the name list** in `tests.js` — append `'restart-again-message'`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the restart-again-message garden mutator (issues #774, #981, #341)."
```

---

### Task 9: `multi-fault`

Issues #1012, #1002 and #980 concern behaviour that only appears once several errors exist at once: the "Too many errors/warnings; noping out" path, the abort-versus-warn wording, and a realtime throttle failure. Every existing mutator injects exactly one fault, so none of that is reachable.

**Files:**
- Modify: `src/tests/monster_garden/garden.js`
- Test: `src/tests/monster_garden/tests.js`

**Interfaces:**
- Consumes: the `mutators` array itself, read at call time.
- Produces: registry entry `{ name: 'multi-fault', apply: multiFault }`.

**Known limitation to preserve:** `multiFault` composes source-editing mutators only. Mutators that change nothing but `inputs`, `level` or `randomSeed` (`nudge-input`, `off-by-one-level`, `seed-poison`, `prefix-chop`) are skipped, because detecting their effect would mean reimplementing `mutationChangedJob` inside the loop for little gain. Do not "fix" this without a reason.

- [ ] **Step 1: Write the failing test.**

```js
test('multi-fault stacks several damaging mutators and never a preserving one', function() {
    const mutator = garden.mutators.filter(function(m) { return m.name === 'multi-fault'; })[0];
    assert(mutator, 'multi-fault should be registered');
    const preserving = garden.mutators
        .filter(function(m) { return m.equivalence; })
        .map(function(m) { return m.name; });
    let applied = 0;
    for (let seed = 0; seed < 30; seed++) {
        const fixture = { source: SAMPLE, inputs: [0, 3], level: 0, randomSeed: null };
        const result = mutator.apply(SAMPLE, new garden.Random(seed), fixture);
        if (!result) {
            continue;
        }
        applied++;
        assert.notStrictEqual(result.source, SAMPLE);
        const names = /applied (.+)$/.exec(result.detail)[1].split(' + ');
        assert(names.length >= 2, 'multi-fault should stack at least two: ' + result.detail);
        for (let i = 0; i < names.length; i++) {
            assert.strictEqual(names[i], names[i].trim());
            assert(preserving.indexOf(names[i]) < 0, names[i] + ' preserves semantics and must not be stacked');
            assert.notStrictEqual(names[i], 'multi-fault', 'multi-fault must not recurse');
        }
    }
    assert(applied > 0, 'expected multi-fault to apply for at least one seed');
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node src/tests/monster_garden/tests.js`
Expected: `multi-fault should be registered`.

- [ ] **Step 3: Implement.** Add to `src/tests/monster_garden/garden.js` before `const mutators = [`:

```js
// Issues #1012, #1002, #980: behaviour that only appears with several errors at
// once. Every other mutator injects exactly one fault, so the error-count
// thresholds are otherwise unreachable. Mutators that declare equivalence are
// excluded: mixing a damaging mutation into a preserving one leaves a program
// that is neither, with nothing meaningful to assert about it.
function multiFault(source, rng, fixture) {
    const pool = mutators.filter(function(mutator) {
        return !mutator.equivalence && mutator.name !== 'multi-fault';
    });
    if (pool.length === 0) {
        return null;
    }
    const wanted = 2 + rng.integer(3);
    const names = [];
    let current = source;
    let inputs = fixture && fixture.inputs ? fixture.inputs : undefined;
    for (let attempt = 0; attempt < wanted * 4 && names.length < wanted; attempt++) {
        const mutator = pool[rng.integer(pool.length)];
        const job = Object.assign({}, fixture, { source: current, inputs: inputs });
        let applied;
        try {
            applied = mutator.apply(current, rng, job);
        } catch (error) {
            continue;
        }
        if (!applied || applied.source === current) {
            continue;
        }
        current = applied.source;
        if (applied.inputs) {
            inputs = applied.inputs;
        }
        names.push(mutator.name);
    }
    if (names.length < 2) {
        return null;
    }
    const result = { source: current, detail: 'applied ' + names.join(' + ') };
    if (inputs) {
        result.inputs = inputs;
    }
    return result;
}
```

- [ ] **Step 4: Register it** after the `restart-again-message` entry:

```js
,
    { name: 'multi-fault', apply: multiFault }
```

- [ ] **Step 5: Update the name list** in `tests.js` — append `'multi-fault'`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/tests/monster_garden/garden.js src/tests/monster_garden/tests.js
git commit -m "Add the multi-fault garden mutator (issues #1012, #1002, #980)."
```

---

### Task 10: Campaign smoke check and documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-08-14-compiler-monster-garden-design.md`

- [ ] **Step 1: Run a campaign restricted to the new mutators**

```bash
node src/tests/monster_garden/run.js \
  --seed 20260814 --count 200 \
  --mutator no-x-with-x,relative-direction-cell,same-layer-cell,property-in-concrete-slot,rigid-prefix,sprite-matrix-resize,restart-again-message,multi-fault \
  --output /tmp/garden-issue-mined
```

Expected: exit 0. Unlike the semantics-preserving campaign, findings here are *expected* and are the point — report the tally line and any saved artifact directories to the user. A `crash`, `timeout` or `invariant` count above zero is a result, not a failure.

- [ ] **Step 2: Confirm no runtime regression**

Run: `node src/tests/run_tests_node.js`
Expected: unchanged pass count from before this plan. Record the number.

Run: `node src/tests/monster_garden/tests.js`
Expected: all pass.

- [ ] **Step 3: Document the new mutators.** In `docs/superpowers/specs/2026-08-14-compiler-monster-garden-design.md`, the "Mutation strategy" section lists example mutators. Add after the existing bullet list:

```markdown
Issue-mined mutators target regions where this project's own bug history
clusters, and each carries the issue numbers it came from in a comment beside
its definition:

- `no-x-with-x`, `relative-direction-cell`, `same-layer-cell`
- `property-in-concrete-slot`, `rigid-prefix`, `sprite-matrix-resize`
- `restart-again-message`, `multi-fault`

See `2026-08-14-garden-issue-mined-mutators-design.md`.
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-14-compiler-monster-garden-design.md
git commit -m "Document the issue-mined garden mutators."
```
