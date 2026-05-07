# Static Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-preserving, tag-first static analyzer for compiled PuzzleScript games.

**Architecture:** Reuse the existing JS compiler runtime through `src/canonicalize.js`, then add a focused analyzer module that turns compiled state into `ps_tagged` and derives fact families from that representation. The CLI stays thin; tests exercise module functions with small fixture games before corpus runs.

**Tech Stack:** Node.js CommonJS, existing PuzzleScript JS compiler/runtime, `assert`-based Node tests, JSON reports.

---

## File Structure

- Modify: `src/canonicalize.js`
  - Export `compileSemanticSource()` so analyzer code can inspect compiled state without duplicating the VM runtime loader.
- Modify: `src/tests/canonicalizer_node.js`
  - Regression test for the new compile helper.
- Create: `src/tests/ps_static_analysis.js`
  - Analyzer library: build `ps_tagged`, tag rules/groups/objects/layers/game, derive fact families, analyze files/paths.
- Create: `src/tests/run_ps_static_analysis.js`
  - CLI wrapper for files and directories.
- Create: `src/tests/ps_static_analysis_node.js`
  - Fixture tests for representation, tags, mergeability, action, invariants, transients, and filters.

Implementation notes:

- Use compiled state as the source of truth. Do not canonicalize or rename objects.
- Use compiled rule objects from `compileSemanticSource()`, specifically their `lhs`, `rhs`, `commands`, `lineNumber`, `late`, `rigid`, `groupNumber`, and `randomRule` fields.
- Do not build `ps_tagged` from engine `Rule`/`CellPattern` replacement masks; those include implicit layer clearing and are not a faithful hand-checkable syntax view.
- Represent unqualified terms as `{kind:"present", movement:null}`. Explicit `stationary` remains `movement:"stationary"`.

---

### Task 1: Export Semantic Compile Helper

**Files:**
- Modify: `src/canonicalize.js`
- Modify: `src/tests/canonicalizer_node.js`

- [ ] **Step 1: Write failing helper test**

Update the import block in `src/tests/canonicalizer_node.js`:

```js
const {
    buildComparisonHashes,
    canonicalizeSource,
    compileSemanticSource,
    hashCanonical,
} = require('../canonicalize');
```

Add this after `baseGame` is declared:

```js
const compiledBase = compileSemanticSource(baseGame);
assert.strictEqual(compiledBase.errorCount, 0, 'compileSemanticSource should compile valid source');
assert.ok(compiledBase.state, 'compileSemanticSource should return compiled state');
assert.ok(compiledBase.state.objects.hero, 'compiled state should expose normalized compiler object names');
assert.strictEqual(compiledBase.state.original_case_names.hero, 'Hero', 'compiled state should retain original object casing');
assert.ok(Array.isArray(compiledBase.state.rules), 'compiled state should expose processed rules');
assert.ok(compiledBase.state.rules.some(rule => rule.lineNumber), 'compiled rules should retain source line numbers');
assert.ok(Array.isArray(compiledBase.state.winconditions), 'compiled state should expose processed win conditions');
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/canonicalizer_node.js
```

Expected: FAIL with `compileSemanticSource is not a function`.

- [ ] **Step 3: Implement helper**

Add this function in `src/canonicalize.js` after `canonicalizeSource()`:

```js
function compileSemanticSource(source, options = {}) {
    const includeWinConditions = options.includeWinConditions !== false;
    const throwOnError = options.throwOnError !== false;
    const compiled = getRuntime().compileSemantic(source, includeWinConditions);
    if (throwOnError && (compiled.errorCount > 0 || compiled.state === null || compiled.state.invalid)) {
        const message = compiled.errorStrings.join('\n');
        throw new Error(`Unable to compile PuzzleScript source.\n${message}`);
    }
    return compiled;
}
```

Update the export block:

```js
module.exports = {
    buildComparisonHashes,
    canonicalizeFile,
    canonicalizeSource,
    compileSemanticSource,
    hashCanonical,
    stableStringify,
};
```

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
node src/tests/canonicalizer_node.js
```

Expected: `canonicalizer_node: ok`.

- [ ] **Step 5: Commit**

```bash
git add src/canonicalize.js src/tests/canonicalizer_node.js
git commit -m "test: expose semantic compile helper"
```

---

### Task 2: Scaffold Analyzer Module, Test, And CLI

**Files:**
- Create: `src/tests/ps_static_analysis.js`
- Create: `src/tests/run_ps_static_analysis.js`
- Create: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Write failing skeleton test**

Create `src/tests/ps_static_analysis_node.js`:

```js
#!/usr/bin/env node
'use strict';

const assert = require('assert');

const { analyzeSource } = require('./ps_static_analysis');

const SIMPLE_GAME = `
title Static Analysis Simple

========
OBJECTS
========

Background
black

Player
white

Goal
yellow

${'======='}
LEGEND
${'======='}

. = Background
P = Player
G = Goal

================
COLLISIONLAYERS
================

Background
Player, Goal

=====
RULES
=====

[ > Player ] -> [ > Player ]

=============
WINCONDITIONS
=============

Some Player on Goal

======
LEVELS
======

P.G
...
`;

const report = analyzeSource(SIMPLE_GAME, { sourcePath: 'simple.txt' });
assert.strictEqual(report.schema, 'ps-static-analysis-v1');
assert.strictEqual(report.status, 'ok');
assert.strictEqual(report.source.path, 'simple.txt');
assert.ok(report.ps_tagged, 'report should include ps_tagged by default');
assert.ok(report.facts.mergeability, 'report should include mergeability facts');
assert.ok(report.facts.movement_action, 'report should include movement_action facts');
assert.ok(report.facts.count_layer_invariants, 'report should include count_layer_invariants facts');
assert.ok(report.facts.transient_boundary, 'report should include transient_boundary facts');

console.log('ps_static_analysis_node: ok');
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL with `Cannot find module './ps_static_analysis'`.

- [ ] **Step 3: Implement minimal module**

Create `src/tests/ps_static_analysis.js`:

```js
#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { compileSemanticSource } = require('../canonicalize');

const SCHEMA = 'ps-static-analysis-v1';

function emptyFacts() {
    return {
        mergeability: [],
        movement_action: [],
        count_layer_invariants: [],
        transient_boundary: [],
    };
}

function analyzeSource(source, options = {}) {
    const sourcePath = options.sourcePath || '<memory>';
    const compiled = compileSemanticSource(source, {
        includeWinConditions: true,
        throwOnError: false,
    });

    if (compiled.errorCount > 0 || compiled.state === null || compiled.state.invalid) {
        return {
            schema: SCHEMA,
            source: { path: sourcePath },
            status: 'compile_error',
            errors: compiled.errorStrings.slice(),
            ps_tagged: null,
            facts: emptyFacts(),
            summary: { proved: 0, candidate: 0, rejected: 0 },
        };
    }

    const psTagged = { game: { tags: {} } };
    const report = {
        schema: SCHEMA,
        source: { path: sourcePath },
        status: 'ok',
        ps_tagged: psTagged,
        facts: emptyFacts(),
        summary: { proved: 0, candidate: 0, rejected: 0 },
    };

    if (options.includePsTagged === false) {
        delete report.ps_tagged;
    }
    return report;
}

function analyzeFile(filePath, options = {}) {
    const resolved = path.resolve(filePath);
    const source = fs.readFileSync(resolved, 'utf8');
    return analyzeSource(source, Object.assign({}, options, { sourcePath: filePath }));
}

module.exports = {
    SCHEMA,
    analyzeFile,
    analyzeSource,
};
```

- [ ] **Step 4: Implement minimal CLI**

Create `src/tests/run_ps_static_analysis.js`:

```js
#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { analyzeFile } = require('./ps_static_analysis');

function usage() {
    console.error('Usage: node src/tests/run_ps_static_analysis.js <file.txt> [--out PATH] [--no-ps-tagged]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    usage();
}

const inputPath = args[0];
const options = { includePsTagged: true };
let outPath = null;

for (let index = 1; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--no-ps-tagged') {
        options.includePsTagged = false;
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

const report = analyzeFile(inputPath, options);
const json = `${JSON.stringify(report, null, 2)}\n`;
if (outPath) {
    fs.writeFileSync(outPath, json);
} else {
    process.stdout.write(json);
}
```

- [ ] **Step 5: Run checks**

Run:

```bash
node --check src/tests/ps_static_analysis.js
node --check src/tests/run_ps_static_analysis.js
node --check src/tests/ps_static_analysis_node.js
node src/tests/ps_static_analysis_node.js
```

Expected: syntax checks pass and test prints `ps_static_analysis_node: ok`.

- [ ] **Step 6: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/run_ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "test: scaffold PuzzleScript static analyzer"
```

---

### Task 3: Build Whole-Game `ps_tagged` Indexes

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Modify: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add failing index assertions**

Append these assertions after the skeleton assertions:

```js
assert.deepStrictEqual(
    report.ps_tagged.objects.map(object => object.name).sort(),
    ['Background', 'Goal', 'Player'],
    'ps_tagged should preserve object names'
);
assert.deepStrictEqual(
    report.ps_tagged.collision_layers.map(layer => layer.objects),
    [['Background'], ['Player', 'Goal']],
    'ps_tagged should preserve collision layer membership'
);
assert.deepStrictEqual(
    report.ps_tagged.properties.find(property => property.name === 'Player').members,
    ['Player'],
    'ps_tagged should expose legend property members with original casing'
);
assert.strictEqual(report.ps_tagged.levels.length, 1, 'ps_tagged should summarize levels');
assert.deepStrictEqual(
    report.ps_tagged.objects.find(object => object.name === 'Player').tags.present_in_all_levels,
    true,
    'object tags should include aggregate level presence'
);
assert.ok(report.ps_tagged.winconditions.length > 0, 'ps_tagged should expose win conditions');
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `objects` is missing.

- [ ] **Step 3: Add bit-mask and index helpers**

Add these helpers to `src/tests/ps_static_analysis.js`:

```js
function uniqueSorted(values) {
    return Array.from(new Set(values)).sort((left, right) =>
        left.localeCompare(right, undefined, { numeric: true })
    );
}

function displayName(state, name) {
    return (state.original_case_names && state.original_case_names[name]) || name;
}

function objectInternalNamesFromMask(state, mask) {
    const bitMask = Array.isArray(mask) ? mask[1] : mask;
    const names = [];
    for (const [name, object] of Object.entries(state.objects)) {
        if (bitMask.get(object.id)) {
            names.push(name);
        }
    }
    return uniqueSorted(names);
}

function objectNamesFromMask(state, mask) {
    return objectInternalNamesFromMask(state, mask).map(name => displayName(state, name));
}

function buildObjects(state) {
    return Object.keys(state.objects)
        .map(name => ({
            id: state.objects[name].id,
            name: displayName(state, name),
            canonical_name: name,
            layer: state.objects[name].layer,
            tags: {},
        }))
        .sort((left, right) => left.id - right.id);
}

function buildProperties(state) {
    const properties = [];
    for (const [name, members] of Object.entries(state.propertiesDict || {})) {
        properties.push({
            name: displayName(state, name),
            canonical_name: name,
            kind: 'property',
            members: uniqueSorted(members.map(member => displayName(state, member))),
            tags: {},
        });
    }
    for (const [name, target] of Object.entries(state.synonymsDict || {})) {
        properties.push({
            name: displayName(state, name),
            canonical_name: name,
            kind: 'synonym',
            members: [displayName(state, target)],
            tags: {},
        });
    }
    return properties.sort((left, right) =>
        left.name.localeCompare(right.name, undefined, { numeric: true })
    );
}

function buildCollisionLayers(state) {
    return state.collisionLayers.map((objects, id) => ({
        id,
        objects: objects.map(name => displayName(state, name)),
        canonical_objects: objects.slice(),
        tags: {},
    }));
}

function buildWinconditions(state) {
    return state.winconditions.map((condition, index) => ({
        id: `win_${index}`,
        quantifier: condition[0],
        subjects: objectNamesFromMask(state, condition[1]),
        targets: condition[2] ? objectNamesFromMask(state, condition[2]) : [],
        tags: {},
    }));
}
```

- [ ] **Step 4: Add level summary and object presence tags**

Add:

```js
function buildLevels(state) {
    return state.levels.map((level, index) => {
        if (level.message !== undefined) {
            return { index, kind: 'message', objects_present: [], layers_present: [], tags: {} };
        }
        const objects = new Set();
        const layers = new Set();
        for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
            const cell = level.getCell(cellIndex);
            for (const name of objectInternalNamesFromMask(state, cell)) {
                objects.add(displayName(state, name));
                layers.add(state.objects[name].layer);
            }
        }
        return {
            index,
            kind: 'level',
            width: level.width,
            height: level.height,
            objects_present: uniqueSorted(Array.from(objects)),
            layers_present: Array.from(layers).sort((left, right) => left - right),
            tags: {},
        };
    });
}

function tagObjectLevelPresence(psTagged) {
    const playableLevels = psTagged.levels.filter(level => level.kind === 'level');
    for (const object of psTagged.objects) {
        const presentCount = playableLevels.filter(level => level.objects_present.includes(object.name)).length;
        object.tags.present_in_all_levels = playableLevels.length > 0 && presentCount === playableLevels.length;
        object.tags.present_in_some_levels = presentCount > 0 && presentCount < playableLevels.length;
        object.tags.present_in_no_levels = presentCount === 0;
    }
}
```

- [ ] **Step 5: Wire `buildPsTagged()`**

Replace the stub `psTagged` in `analyzeSource()`:

```js
const psTagged = buildPsTagged(compiled.state, { sourcePath });
```

Add:

```js
function buildPsTagged(state, options = {}) {
    const psTagged = {
        game: {
            title: state.metadata && state.metadata.title,
            source_path: options.sourcePath || '<memory>',
            tags: {},
        },
        objects: buildObjects(state),
        properties: buildProperties(state),
        collision_layers: buildCollisionLayers(state),
        winconditions: buildWinconditions(state),
        levels: buildLevels(state),
        rule_sections: [],
    };
    tagObjectLevelPresence(psTagged);
    return psTagged;
}
```

- [ ] **Step 6: Run test**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: test prints `ps_static_analysis_node: ok`.

- [ ] **Step 7: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: build static analyzer game indexes"
```

---

### Task 4: Build Rule Sections, Groups, Terms, And Tags

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Modify: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add failing rule representation fixture**

Add this fixture and assertions:

```js
const RULE_SHAPE_GAME = `
title Rule Shape

========
OBJECTS
========

Background
black

A
white

B
red

C
blue

${'======='}
LEGEND
${'======='}

. = Background
a = A
b = B
c = C
Player = A

================
COLLISIONLAYERS
================

Background
A, B
C

=====
RULES
=====

right [ A | right B no C ] -> [ up A | B right C ]
late [ C ] -> [ no C ]

=============
WINCONDITIONS
=============

Some A

======
LEVELS
======

ab.
...
`;

const ruleShape = analyzeSource(RULE_SHAPE_GAME, { sourcePath: 'rule_shape.txt' });
const early = ruleShape.ps_tagged.rule_sections.find(section => section.name === 'early');
const late = ruleShape.ps_tagged.rule_sections.find(section => section.name === 'late');
assert.strictEqual(early.groups.length, 1, 'early section should contain one group');
assert.strictEqual(late.groups.length, 1, 'late section should contain one group');
const shapeRule = early.groups[0].rules[0];
assert.strictEqual(shapeRule.direction, 'right');
assert.deepStrictEqual(shapeRule.lhs[0][0], [
    { kind: 'present', ref: { type: 'object', name: 'A' }, movement: null },
]);
assert.deepStrictEqual(shapeRule.lhs[0][1], [
    { kind: 'present', ref: { type: 'object', name: 'B' }, movement: 'right' },
    { kind: 'absent', ref: { type: 'object', name: 'C' }, movement: null },
]);
assert.deepStrictEqual(shapeRule.rhs[0][0], [
    { kind: 'present', ref: { type: 'object', name: 'A' }, movement: 'up' },
]);
assert.deepStrictEqual(shapeRule.rhs[0][1], [
    { kind: 'present', ref: { type: 'object', name: 'B' }, movement: null },
    { kind: 'present', ref: { type: 'object', name: 'C' }, movement: 'right' },
]);
```

- [ ] **Step 2: Add failing command tag fixture**

Add:

```js
const COMMAND_GAME = `
title Command Tags
========
OBJECTS
========
Background
black
A
white
${'======='}
LEGEND
${'======='}
. = Background
a = A
Player = A
================
COLLISIONLAYERS
================
Background
A
=====
RULES
=====
[ A ] -> sfx0
[ A ] -> checkpoint
=============
WINCONDITIONS
=============
Some A
======
LEVELS
======
a
`;

const commandReport = analyzeSource(COMMAND_GAME, { sourcePath: 'commands.txt' });
const commandRules = commandReport.ps_tagged.rule_sections[0].groups.flatMap(group => group.rules);
assert.strictEqual(commandRules.length, 2, 'command-only rules should remain present');
assert.strictEqual(commandRules[0].tags.inert_command_only, true, 'sfx-only rule is inert for solver state');
assert.strictEqual(commandRules[0].tags.solver_state_active, false, 'sfx-only rule is not solver-state active');
assert.strictEqual(commandRules[1].tags.command_only, true, 'checkpoint-only rule is command-only');
assert.strictEqual(commandRules[1].tags.solver_state_active, true, 'checkpoint is semantic/metagame-active');
```

- [ ] **Step 3: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `rule_sections` is empty or tags are missing.

- [ ] **Step 4: Implement term conversion**

Add:

```js
const INERT_COMMANDS = new Set(['message', 'sfx0', 'sfx1', 'sfx2', 'sfx3', 'sfx4', 'sfx5', 'sfx6', 'sfx7', 'sfx8', 'sfx9', 'sfx10']);
const SEMANTIC_COMMANDS = new Set(['cancel', 'again', 'restart', 'win', 'checkpoint']);
const MOVEMENTS = new Set(['stationary', 'up', 'down', 'left', 'right', 'moving', 'action', 'randomdir']);

function directionName(direction) {
    if (typeof direction === 'string') return direction;
    if (direction === 1) return 'up';
    if (direction === 2) return 'down';
    if (direction === 4) return 'left';
    if (direction === 8) return 'right';
    if (direction === 15) return 'orthogonal';
    if (direction === 16) return 'action';
    return String(direction);
}

function termFromPair(state, direction, name) {
    if (direction === '...' && name === '...') {
        return { kind: 'present', ref: { type: 'ellipsis' }, movement: null };
    }
    if (direction === 'no') {
        return { kind: 'absent', ref: refForName(state, name), movement: null };
    }
    if (direction === 'random') {
        return { kind: 'random_object', ref: refForName(state, name), movement: null };
    }
    if (direction === '') {
        return { kind: 'present', ref: refForName(state, name), movement: null };
    }
    return { kind: 'present', ref: refForName(state, name), movement: directionName(direction) };
}

function refForName(state, name) {
    if (state.objects[name]) {
        return { type: 'object', name: displayName(state, name), canonical_name: name };
    }
    if (state.propertiesDict && state.propertiesDict[name]) {
        return {
            type: 'property',
            name: displayName(state, name),
            canonical_name: name,
            members: uniqueSorted(state.propertiesDict[name].map(member => displayName(state, member))),
        };
    }
    if (state.synonymsDict && state.synonymsDict[name]) {
        return {
            type: 'synonym',
            name: displayName(state, name),
            canonical_name: name,
            members: [displayName(state, state.synonymsDict[name])],
        };
    }
    return { type: 'unknown', name };
}

function termsFromCell(state, cell) {
    const terms = [];
    for (let index = 0; index < cell.length; index += 2) {
        terms.push(termFromPair(state, cell[index], cell[index + 1]));
    }
    return terms;
}

function termsFromSide(state, side) {
    return (side || []).map(row => row.map(cell => termsFromCell(state, cell)));
}
```

- [ ] **Step 5: Implement summaries and tag taxonomy**

Add:

```js
function flattenTerms(side) {
    return side.flat(2);
}

function summarizeRule(rule) {
    const lhsTerms = flattenTerms(rule.lhs);
    const rhsTerms = flattenTerms(rule.rhs);
    return {
        lhs_terms: lhsTerms,
        rhs_terms: rhsTerms,
        lhs_present: lhsTerms.filter(term => term.kind === 'present'),
        lhs_absent: lhsTerms.filter(term => term.kind === 'absent'),
        lhs_movement: lhsTerms.filter(term => term.movement !== null),
        rhs_present: rhsTerms.filter(term => term.kind === 'present'),
        rhs_absent: rhsTerms.filter(term => term.kind === 'absent'),
        rhs_random_objects: rhsTerms.filter(term => term.kind === 'random_object'),
        rhs_movement: rhsTerms.filter(term => term.movement !== null),
        semantic_commands: rule.commands.map(command => command[0]).filter(command => SEMANTIC_COMMANDS.has(command)),
        inert_commands: rule.commands.map(command => command[0]).filter(command => INERT_COMMANDS.has(command)),
    };
}

function objectTermSignature(terms) {
    return JSON.stringify(terms
        .filter(term => term.kind === 'present')
        .map(term => JSON.stringify(term.ref))
        .sort());
}

function movementTermSignature(terms) {
    return JSON.stringify(terms
        .filter(term => term.kind === 'present' && term.movement !== null)
        .map(term => `${JSON.stringify(term.ref)}:${term.movement}`)
        .sort());
}

function tagRule(rule) {
    rule.summary = summarizeRule(rule);
    const commandNames = rule.commands.map(command => command[0]);
    const hasOnlyInertCommands = commandNames.length > 0
        && commandNames.every(command => INERT_COMMANDS.has(command));
    const hasReplacement = rule.rhs.length > 0;
    const objectMutating = hasReplacement
        && (objectTermSignature(rule.summary.lhs_terms) !== objectTermSignature(rule.summary.rhs_terms)
        || rule.summary.rhs_absent.length > 0
        || rule.summary.rhs_random_objects.length > 0);
    const writesMovement = hasReplacement
        && (rule.summary.rhs_movement.length > 0
        || movementTermSignature(rule.summary.lhs_terms) !== movementTermSignature(rule.summary.rhs_terms));
    const hasSemanticCommand = rule.summary.semantic_commands.length > 0;

    rule.tags.command_only = commandNames.length > 0 && !objectMutating && !writesMovement;
    rule.tags.inert_command_only = hasOnlyInertCommands && rule.tags.command_only;
    rule.tags.object_mutating = objectMutating;
    rule.tags.writes_movement = writesMovement;
    rule.tags.movement_only = writesMovement && !objectMutating && !hasSemanticCommand;
    rule.tags.reads_action = rule.summary.lhs_movement.some(term => term.movement === 'action');
    rule.tags.has_again = rule.summary.semantic_commands.includes('again');
    const hasNonInertEffect = objectMutating || writesMovement || hasSemanticCommand;
    rule.tags.solver_state_active = !rule.tags.inert_command_only && hasNonInertEffect;
    if (rule.rigid && hasNonInertEffect) {
        rule.tags.rigid_active = true;
    }
}
```

- [ ] **Step 6: Implement rule sections and groups**

Add:

```js
function buildRuleSections(state) {
    return [
        buildRuleSection(state, 'early', state.rules.filter(rule => !rule.late)),
        buildRuleSection(state, 'late', state.rules.filter(rule => rule.late)),
    ];
}

function buildRuleSection(state, name, rules) {
    const groupNumbers = [];
    const groupMap = new Map();
    for (const rule of rules) {
        if (!groupMap.has(rule.groupNumber)) {
            groupMap.set(rule.groupNumber, []);
            groupNumbers.push(rule.groupNumber);
        }
        groupMap.get(rule.groupNumber).push(rule);
    }
    const groups = groupNumbers.map((groupNumber, index) =>
        buildRuleGroup(state, name, index, groupNumber, groupMap.get(groupNumber))
    );
    return {
        name,
        loops: buildLoopSummaries(state, groups),
        groups,
    };
}

function buildLoopSummaries(state, groups) {
    const loops = [];
    const stack = [];
    for (const loop of state.loops || []) {
        const line = loop[0];
        const bracket = loop[1];
        if (bracket === 1) {
            stack.push({ id: `loop_${loops.length}`, start_line: line, end_line: null });
        } else if (bracket === -1 && stack.length > 0) {
            const active = stack.pop();
            active.end_line = line;
            active.group_ids = groups
                .filter(group => group.source_line_min > active.start_line && group.source_line_max < active.end_line)
                .map(group => group.id);
            loops.push(active);
        }
    }
    return loops;
}

function buildRuleGroup(state, sectionName, groupIndex, groupNumber, sourceRules) {
    const rules = sourceRules.map((rule, ruleIndex) => buildRuleIr(state, sectionName, groupIndex, rule, ruleIndex));
    const group = {
        id: `${sectionName}_group_${groupIndex}`,
        group_index: groupIndex,
        group_number: groupNumber,
        source_line_min: Math.min(...sourceRules.map(rule => rule.lineNumber)),
        source_line_max: Math.max(...sourceRules.map(rule => rule.lineNumber)),
        random: sourceRules.some(rule => rule.randomRule),
        tags: {},
        rules,
    };
    for (const rule of rules) {
        tagRule(rule);
    }
    tagGroup(group);
    return group;
}

function buildRuleIr(state, sectionName, groupIndex, rule, ruleIndex) {
    return {
        id: `${sectionName}_group_${groupIndex}_rule_${ruleIndex}`,
        source_line: rule.lineNumber,
        direction: directionName(rule.direction),
        late: !!rule.late || sectionName === 'late',
        rigid: !!rule.rigid,
        random_rule: !!rule.randomRule,
        tags: {},
        commands: (rule.commands || []).map(command => command.slice()),
        lhs: termsFromSide(state, rule.lhs),
        rhs: termsFromSide(state, rule.rhs),
        summary: {},
    };
}

function tagGroup(group) {
    group.tags.has_again = group.rules.some(rule => rule.tags.has_again);
    group.tags.object_mutating = group.rules.some(rule => rule.tags.object_mutating);
    group.tags.movement_only = group.rules.some(rule => rule.tags.movement_only) && !group.tags.object_mutating;
    group.tags.command_only = group.rules.every(rule => rule.tags.command_only);
    group.tags.solver_state_active = group.rules.some(rule => rule.tags.solver_state_active);
}
```

Update `buildPsTagged()`:

```js
rule_sections: buildRuleSections(state),
```

- [ ] **Step 7: Add game tags**

Add:

```js
function tagGame(psTagged) {
    const rules = allRuleEntries(psTagged).map(entry => entry.rule);
    psTagged.game.tags.has_again = rules.some(rule => rule.tags.has_again);
    psTagged.game.tags.has_random = rules.some(rule => rule.random_rule || rule.summary.rhs_random_objects.length > 0);
    psTagged.game.tags.has_rigid = rules.some(rule => rule.rigid);
    psTagged.game.tags.has_action_rules = rules.some(rule => rule.tags.reads_action);
    psTagged.game.tags.has_autonomous_tick_rules = rules.some(rule => rule.tags.solver_state_active && rule.summary.lhs_movement.length === 0);
}

function allRuleEntries(psTagged) {
    return psTagged.rule_sections.flatMap(section =>
        section.groups.flatMap(group =>
            group.rules.map(rule => ({ section, group, rule }))
        )
    );
}
```

Call `tagGame(psTagged)` before returning from `buildPsTagged()`.

- [ ] **Step 8: Run test**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: test prints `ps_static_analysis_node: ok`.

- [ ] **Step 9: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: build tagged PuzzleScript rule view"
```

---

### Task 5: Implement Mergeability End-To-End

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Modify: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add mergeability fixtures**

Add:

```js
const MERGEABLE_GAME = `
title Mergeable
========
OBJECTS
========
Background
black
BodyH
white
BodyV
white
Goal
yellow
${'======='}
LEGEND
${'======='}
. = Background
h = BodyH
v = BodyV
g = Goal
Player = BodyH or BodyV
Body = BodyH or BodyV
================
COLLISIONLAYERS
================
Background
BodyH, BodyV, Goal
=====
RULES
=====
[ Body ] -> [ Body ]
[ no Body ] -> [ no Body ]
[ > Body ] -> [ > Body ]
=============
WINCONDITIONS
=============
Some Body on Goal
======
LEVELS
======
h.g
`;

const mergeable = analyzeSource(MERGEABLE_GAME, { sourcePath: 'mergeable.txt' });
const mergeFact = mergeable.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.ok(mergeFact, 'BodyH/BodyV should produce a mergeability fact');
assert.strictEqual(mergeFact.status, 'candidate');
assert.ok(mergeFact.proof.includes('same_collision_layer'));
assert.ok(mergeFact.proof.includes('observed_only_through_shared_sets'));

const DIRECT_READ_GAME = MERGEABLE_GAME.replace('[ Body ] -> [ Body ]', '[ BodyH ] -> [ BodyH ]');
const directRead = analyzeSource(DIRECT_READ_GAME, { sourcePath: 'direct_read.txt' });
const directReadFact = directRead.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directReadFact.status, 'rejected');
assert.ok(directReadFact.blockers.includes('individual_lhs_read'));

const DIRECT_NEGATION_GAME = MERGEABLE_GAME.replace('[ no Body ] -> [ no Body ]', '[ no BodyH ] -> [ no BodyH ]');
const directNegation = analyzeSource(DIRECT_NEGATION_GAME, { sourcePath: 'direct_negation.txt' });
const directNegationFact = directNegation.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directNegationFact.status, 'rejected');
assert.ok(directNegationFact.blockers.includes('individual_lhs_read'));

const DIRECT_WIN_GAME = MERGEABLE_GAME.replace('Some Body on Goal', 'Some BodyH on Goal');
const directWin = analyzeSource(DIRECT_WIN_GAME, { sourcePath: 'direct_win.txt' });
const directWinFact = directWin.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directWinFact.status, 'rejected');
assert.ok(directWinFact.blockers.includes('different_win_roles'));
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `mergeability` is empty.

- [ ] **Step 3: Add fact and expansion helpers**

Add:

```js
function fact(family, id, status, fields) {
    return Object.assign({
        family,
        id,
        status,
        subjects: {},
        tags: {},
        proof: [],
        blockers: [],
        evidence: [],
    }, fields);
}

function membersForRef(psTagged, ref) {
    if (ref.type === 'object') return [ref.name];
    if (ref.type === 'object_set') return ref.objects.slice();
    if (ref.type === 'property' || ref.type === 'synonym' || ref.type === 'unknown') {
        const property = psTagged.properties.find(item => item.name === ref.name);
        return property ? property.members.slice() : [ref.name];
    }
    return [];
}

function normalizeTermRefs(psTagged) {
    for (const { rule } of allRuleEntries(psTagged)) {
        for (const term of rule.summary.lhs_terms.concat(rule.summary.rhs_terms)) {
            const members = membersForRef(psTagged, term.ref);
            if (members.length === 1 && psTagged.objects.some(object => object.name === members[0])) {
                term.expanded_objects = members;
                term.ref = { type: 'object', name: members[0] };
            } else if (members.length > 1) {
                term.expanded_objects = uniqueSorted(members);
                term.ref = { type: 'object_set', objects: uniqueSorted(members), source: term.ref.name || term.ref.type };
            } else {
                term.expanded_objects = [];
            }
        }
    }
}
```

Call `normalizeTermRefs(psTagged)` after `tagGame(psTagged)` in `buildPsTagged()`.

- [ ] **Step 4: Implement mergeability derivation**

Add:

```js
function sameArray(left, right) {
    return JSON.stringify(uniqueSorted(left)) === JSON.stringify(uniqueSorted(right));
}

function winRoleForObject(psTagged, objectName) {
    return psTagged.winconditions.map(win => ({
        id: win.id,
        in_subjects: win.subjects.includes(objectName),
        in_targets: win.targets.includes(objectName),
    }));
}

function directObservationsForObject(psTagged, objectName) {
    const observations = [];
    for (const { rule } of allRuleEntries(psTagged)) {
        if (!rule.tags.solver_state_active) continue;
        for (const term of rule.summary.lhs_terms) {
            if (term.ref.type === 'object' && term.ref.name === objectName) {
                observations.push({ rule_id: rule.id, source_line: rule.source_line, kind: term.kind, movement: term.movement });
            }
        }
    }
    return observations;
}

function groupObservationIsShared(psTagged, objects) {
    for (const { rule } of allRuleEntries(psTagged)) {
        if (!rule.tags.solver_state_active) continue;
        for (const term of rule.summary.lhs_terms) {
            if (term.ref.type !== 'object_set') continue;
            const overlap = objects.filter(objectName => term.expanded_objects.includes(objectName));
            if (overlap.length > 0 && overlap.length !== objects.length) {
                return false;
            }
        }
    }
    return true;
}

function deriveMergeabilityFacts(psTagged) {
    const results = [];
    for (const layer of psTagged.collision_layers) {
        if (layer.objects.length < 2) continue;
        for (let left = 0; left < layer.objects.length; left++) {
            for (let right = left + 1; right < layer.objects.length; right++) {
                const objects = [layer.objects[left], layer.objects[right]].sort();
                const blockers = [];
                const directObservations = objects.flatMap(objectName => directObservationsForObject(psTagged, objectName));
                if (directObservations.length > 0) blockers.push('individual_lhs_read');
                if (JSON.stringify(winRoleForObject(psTagged, objects[0])) !== JSON.stringify(winRoleForObject(psTagged, objects[1]))) {
                    blockers.push('different_win_roles');
                }
                if (!groupObservationIsShared(psTagged, objects)) {
                    blockers.push('partial_property_observation');
                }
                results.push(fact('mergeability', `merge_${objects.join('_')}`, blockers.length === 0 ? 'candidate' : 'rejected', {
                    subjects: { objects },
                    proof: blockers.length === 0 ? ['same_collision_layer', 'observed_only_through_shared_sets', 'same_win_roles'] : ['same_collision_layer'],
                    blockers,
                    evidence: directObservations.map(item => item.rule_id).concat(`layer_${layer.id}`),
                }));
            }
        }
    }
    return results;
}
```

- [ ] **Step 5: Wire fact derivation**

Add:

```js
function deriveFacts(psTagged) {
    return {
        mergeability: deriveMergeabilityFacts(psTagged),
        movement_action: [],
        count_layer_invariants: [],
        transient_boundary: [],
    };
}
```

Replace `facts: emptyFacts()` in successful `analyzeSource()` with:

```js
facts: filterFacts(deriveFacts(psTagged), options.familyFilter),
```

Add:

```js
function filterFacts(facts, familyFilter) {
    if (!familyFilter) return facts;
    return { [familyFilter]: facts[familyFilter] || [] };
}
```

- [ ] **Step 6: Run test**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: test prints `ps_static_analysis_node: ok`.

- [ ] **Step 7: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: derive mergeability facts"
```

---

### Task 6: Add Movement And Action Facts

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Modify: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add movement/action fixtures**

Add:

```js
const AUTO_TICK_GAME = SIMPLE_GAME.replace('[ > Player ] -> [ > Player ]', '[ Goal ] -> [ Player ]');
const autoTick = analyzeSource(AUTO_TICK_GAME, { sourcePath: 'auto_tick.txt' });
const autoAction = autoTick.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(autoAction.status, 'rejected');
assert.ok(autoAction.blockers.includes('autonomous_solver_active_rule'));

const ACTION_RULE_GAME = SIMPLE_GAME.replace('[ > Player ] -> [ > Player ]', '[ action Player ] -> [ Player Goal ]');
const actionRule = analyzeSource(ACTION_RULE_GAME, { sourcePath: 'action_rule.txt' });
const actionRuleFact = actionRule.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(actionRuleFact.status, 'rejected');
assert.ok(actionRuleFact.blockers.includes('reads_action'));

const ACTION_MOVEMENT_GAME = SIMPLE_GAME.replace('[ > Player ] -> [ > Player ]', '[ action Player ] -> [ > Player ]');
const actionMovement = analyzeSource(ACTION_MOVEMENT_GAME, { sourcePath: 'action_movement.txt' });
const actionMovementFact = actionMovement.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(actionMovementFact.status, 'rejected');
assert.ok(actionMovementFact.blockers.includes('action_may_create_directional_movement'));
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `movement_action` is empty.

- [ ] **Step 3: Implement action fixpoint helpers**

Add:

```js
const DIRECTIONAL_MOVEMENTS = new Set(['up', 'down', 'left', 'right', 'moving', 'randomdir']);

function layerForObject(psTagged, objectName) {
    const object = psTagged.objects.find(item => item.name === objectName);
    return object ? object.layer : null;
}

function playerLayers(psTagged) {
    const player = psTagged.properties.find(item => item.canonical_name === 'player' || item.name.toLowerCase() === 'player');
    if (!player) return [];
    return uniqueSorted(player.members.map(objectName => String(layerForObject(psTagged, objectName))).filter(layer => layer !== 'null'));
}

function movementPairsFromTerms(psTagged, terms) {
    const pairs = [];
    for (const term of terms) {
        if (term.kind !== 'present' || term.movement === null) continue;
        for (const objectName of term.expanded_objects || []) {
            const layer = layerForObject(psTagged, objectName);
            if (layer !== null) pairs.push(`${layer}:${term.movement}`);
        }
    }
    return pairs;
}

function ruleMovementRequirementsReachable(psTagged, rule, possibleMovements) {
    const requirements = movementPairsFromTerms(psTagged, rule.summary.lhs_terms);
    if (requirements.length === 0) return true;
    return requirements.every(pair => possibleMovements.has(pair));
}
```

- [ ] **Step 4: Implement movement/action facts**

Add:

```js
function deriveMovementActionFacts(psTagged) {
    const activeRules = allRuleEntries(psTagged).map(entry => entry.rule).filter(rule => rule.tags.solver_state_active);
    const possibleMovements = new Set(playerLayers(psTagged).map(layer => `${layer}:action`));
    const blockers = [];
    let changed = true;
    while (changed) {
        changed = false;
        for (const rule of activeRules) {
            if (!ruleMovementRequirementsReachable(psTagged, rule, possibleMovements)) continue;
            if (rule.tags.reads_action) blockers.push('reads_action');
            if (rule.tags.has_again) blockers.push('queues_again');
            if (rule.rigid) blockers.push('rigid_rule');
            if (rule.summary.lhs_movement.length === 0) blockers.push('autonomous_solver_active_rule');
            if (rule.tags.object_mutating) blockers.push('action_may_mutate_objects');
            for (const pair of movementPairsFromTerms(psTagged, rule.summary.rhs_terms)) {
                const movement = pair.split(':')[1];
                if (DIRECTIONAL_MOVEMENTS.has(movement)) blockers.push('action_may_create_directional_movement');
                if (!possibleMovements.has(pair)) {
                    possibleMovements.add(pair);
                    changed = true;
                }
            }
        }
    }
    const uniqueBlockers = uniqueSorted(blockers);
    return [
        fact('movement_action', 'movement_pairs', 'proved', {
            value: Array.from(possibleMovements).sort(),
            proof: ['conservative_movement_reachability_fixpoint'],
        }),
        fact('movement_action', 'action_noop', uniqueBlockers.length === 0 ? 'proved' : 'rejected', {
            value: uniqueBlockers.length === 0,
            blockers: uniqueBlockers,
            proof: uniqueBlockers.length === 0 ? ['no_reachable_action_effects'] : [],
            evidence: activeRules.map(rule => rule.id),
        }),
    ];
}
```

Update `deriveFacts()`:

```js
movement_action: deriveMovementActionFacts(psTagged),
```

- [ ] **Step 5: Run test**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: test prints `ps_static_analysis_node: ok`.

- [ ] **Step 6: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: derive movement and action facts"
```

---

### Task 7: Add Count And Layer Invariant Facts

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Modify: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add invariant fixtures**

Add:

```js
const countFacts = report.facts.count_layer_invariants;
assert.ok(countFacts.some(item => item.id === 'object_Player_count_preserved'), 'Player count fact should exist');
assert.ok(countFacts.some(item => item.id === 'layer_0_static'), 'Background layer static fact should exist');

const SPAWN_GAME = SIMPLE_GAME.replace('[ > Player ] -> [ > Player ]', '[ Player ] -> [ Player Goal ]');
const spawnReport = analyzeSource(SPAWN_GAME, { sourcePath: 'spawn.txt' });
const goalCount = spawnReport.facts.count_layer_invariants.find(item => item.id === 'object_Goal_count_preserved');
assert.strictEqual(goalCount.status, 'rejected');
assert.ok(goalCount.blockers.includes('object_written_by_solver_active_rule'));
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `count_layer_invariants` is empty.

- [ ] **Step 3: Implement object write detection**

Add:

```js
function termMentionsObject(term, objectName) {
    return (term.expanded_objects || []).includes(objectName);
}

function ruleWritesObject(rule, objectName) {
    if (!rule.tags.solver_state_active) return false;
    return rule.summary.rhs_terms.some(term => termMentionsObject(term, objectName));
}

function deriveCountLayerInvariantFacts(psTagged) {
    const activeRules = allRuleEntries(psTagged).map(entry => entry.rule).filter(rule => rule.tags.solver_state_active);
    const results = [];
    for (const object of psTagged.objects) {
        const writers = activeRules.filter(rule => ruleWritesObject(rule, object.name));
        object.tags.may_be_created = writers.length > 0;
        object.tags.may_be_destroyed = writers.length > 0;
        object.tags.count_invariant = writers.length === 0;
        results.push(fact('count_layer_invariants', `object_${object.name}_count_preserved`, writers.length === 0 ? 'proved' : 'rejected', {
            subjects: { objects: [object.name] },
            proof: writers.length === 0 ? ['no_solver_active_rule_writes_object'] : [],
            blockers: writers.length === 0 ? [] : ['object_written_by_solver_active_rule'],
            evidence: writers.map(rule => rule.id),
        }));
    }
    for (const layer of psTagged.collision_layers) {
        const layerWriterIds = uniqueSorted(layer.objects.flatMap(objectName =>
            activeRules.filter(rule => ruleWritesObject(rule, objectName)).map(rule => rule.id)
        ));
        layer.tags.static = layerWriterIds.length === 0;
        results.push(fact('count_layer_invariants', `layer_${layer.id}_static`, layerWriterIds.length === 0 ? 'proved' : 'candidate', {
            subjects: { layers: [layer.id] },
            proof: layerWriterIds.length === 0 ? ['no_solver_active_rule_writes_layer_objects'] : [],
            blockers: layerWriterIds.length === 0 ? [] : ['layer_objects_may_change'],
            evidence: layerWriterIds,
        }));
    }
    return results;
}
```

Update `deriveFacts()`:

```js
count_layer_invariants: deriveCountLayerInvariantFacts(psTagged),
```

- [ ] **Step 4: Run test**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: test prints `ps_static_analysis_node: ok`.

- [ ] **Step 5: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: derive count and layer invariant facts"
```

---

### Task 8: Add End-Of-Turn Transient Facts

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Modify: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add transient fixtures**

Add:

```js
const TRANSIENT_GAME = `
title Transient
========
OBJECTS
========
Background
black
Player
white
Mark
red
${'======='}
LEGEND
${'======='}
. = Background
P = Player
M = Mark
================
COLLISIONLAYERS
================
Background
Player
Mark
=====
RULES
=====
[ Player ] -> [ Player Mark ]
late [ Mark ] -> [ no Mark ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
P
`;

const transient = analyzeSource(TRANSIENT_GAME, { sourcePath: 'transient.txt' });
const markTransient = transient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(markTransient.status, 'proved');
assert.strictEqual(markTransient.tags.single_turn_only, true);

const AGAIN_TAINT_GAME = TRANSIENT_GAME.replace('[ Player ] -> [ Player Mark ]', '[ Player ] -> [ Player Mark ] again');
const againTaint = analyzeSource(AGAIN_TAINT_GAME, { sourcePath: 'again_taint.txt' });
const againMark = againTaint.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(againMark.status, 'rejected');
assert.ok(againMark.blockers.includes('has_again_taint'));
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `transient_boundary` is empty.

- [ ] **Step 3: Implement transient derivation**

Add:

```js
function rulesInSection(psTagged, sectionName) {
    const section = psTagged.rule_sections.find(item => item.name === sectionName);
    return section ? section.groups.flatMap(group => group.rules.map(rule => ({ group, rule }))) : [];
}

function earlySettersForObject(psTagged, objectName) {
    return rulesInSection(psTagged, 'early')
        .filter(entry => entry.rule.tags.solver_state_active)
        .filter(entry => entry.rule.summary.rhs_terms.some(term => term.kind === 'present' && termMentionsObject(term, objectName)));
}

function lateClearersForObject(psTagged, objectName) {
    return rulesInSection(psTagged, 'late')
        .filter(entry => entry.rule.tags.solver_state_active)
        .filter(entry => entry.rule.summary.rhs_terms.some(term => term.kind === 'absent' && termMentionsObject(term, objectName)));
}

function objectInWincondition(psTagged, objectName) {
    return psTagged.winconditions.some(win => win.subjects.includes(objectName) || win.targets.includes(objectName));
}

function deriveTransientBoundaryFacts(psTagged) {
    const results = [];
    for (const object of psTagged.objects) {
        const setters = earlySettersForObject(psTagged, object.name);
        const clearers = lateClearersForObject(psTagged, object.name);
        const blockers = [];
        if (setters.length === 0) blockers.push('not_created_in_early_rules');
        if (clearers.length === 0) blockers.push('no_late_cleanup_clear');
        if (!object.tags.present_in_no_levels) blockers.push('present_in_some_initial_levels');
        if (objectInWincondition(psTagged, object.name)) blockers.push('appears_in_wincondition');
        if (setters.some(entry => entry.group.tags.has_again || entry.rule.tags.has_again)) blockers.push('has_again_taint');
        if (setters.some(entry => entry.rule.rigid) || clearers.some(entry => entry.rule.rigid)) blockers.push('rigid_rule');
        const status = blockers.length === 0 ? 'proved' : 'rejected';
        results.push(fact('transient_boundary', `object_${object.name}_end_turn_transient`, status, {
            subjects: { objects: [object.name] },
            tags: { single_turn_only: true },
            proof: status === 'proved' ? ['created_in_early_rules', 'cleared_in_late_rules', 'absent_from_initial_levels_and_winconditions'] : [],
            blockers,
            evidence: setters.concat(clearers).map(entry => entry.rule.id),
        }));
    }
    return results;
}
```

Update `deriveFacts()`:

```js
transient_boundary: deriveTransientBoundaryFacts(psTagged),
```

- [ ] **Step 4: Run test**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: test prints `ps_static_analysis_node: ok`.

- [ ] **Step 5: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: derive end-turn transient facts"
```

---

### Task 9: Add Batch CLI, Filters, And Corpus Validation

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Modify: `src/tests/run_ps_static_analysis.js`
- Modify: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add filter assertions**

Add:

```js
const noTagged = analyzeSource(SIMPLE_GAME, { sourcePath: 'simple.txt', includePsTagged: false });
assert.strictEqual(noTagged.ps_tagged, undefined, 'includePsTagged=false should remove ps_tagged');

const onlyMerge = analyzeSource(SIMPLE_GAME, { sourcePath: 'simple.txt', familyFilter: 'mergeability' });
assert.deepStrictEqual(Object.keys(onlyMerge.facts), ['mergeability'], 'familyFilter should keep one family');
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `familyFilter` is not applied.

- [ ] **Step 3: Add path discovery**

Add to `src/tests/ps_static_analysis.js`:

```js
function discoverInputFiles(inputs) {
    const files = [];
    for (const input of inputs) {
        const stat = fs.statSync(input);
        if (stat.isDirectory()) {
            for (const entry of fs.readdirSync(input).sort()) {
                const fullPath = path.join(input, entry);
                if (fs.statSync(fullPath).isFile() && fullPath.endsWith('.txt')) {
                    files.push(fullPath);
                }
            }
        } else {
            files.push(input);
        }
    }
    return files;
}

function analyzePaths(inputs, options = {}) {
    return discoverInputFiles(inputs)
        .filter(filePath => !options.gameFilter || filePath.toLowerCase().includes(options.gameFilter.toLowerCase()))
        .map(filePath => analyzeFile(filePath, options));
}
```

Export it:

```js
module.exports = {
    SCHEMA,
    analyzeFile,
    analyzePaths,
    analyzeSource,
};
```

- [ ] **Step 4: Replace CLI with batch output**

Replace `src/tests/run_ps_static_analysis.js` with:

```js
#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { analyzePaths } = require('./ps_static_analysis');

function usage() {
    console.error([
        'Usage: node src/tests/run_ps_static_analysis.js <file-or-dir> [more paths]',
        '  [--out PATH] [--family NAME] [--game SUBSTRING]',
        '  [--include-ps-tagged] [--no-ps-tagged]',
    ].join('\n'));
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    usage();
}

const inputs = [];
const options = { includePsTagged: true, familyFilter: null, gameFilter: null };
let outPath = null;

for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--family' && index + 1 < args.length) {
        options.familyFilter = args[++index];
    } else if (arg === '--game' && index + 1 < args.length) {
        options.gameFilter = args[++index];
    } else if (arg === '--include-ps-tagged') {
        options.includePsTagged = true;
    } else if (arg === '--no-ps-tagged') {
        options.includePsTagged = false;
    } else if (arg.startsWith('--')) {
        throw new Error(`Unsupported argument: ${arg}`);
    } else {
        inputs.push(arg);
    }
}

const reports = analyzePaths(inputs, options);
const output = {
    schema: 'ps-static-analysis-batch-v1',
    generated_at: new Date().toISOString(),
    source_count: reports.length,
    reports,
};

const json = `${JSON.stringify(output, null, 2)}\n`;
if (outPath) {
    fs.writeFileSync(outPath, json);
} else {
    process.stdout.write(json);
}
```

- [ ] **Step 5: Add proved-fact invariant**

Add to the test file:

```js
function assertProvedFactsHaveProof(reportToCheck) {
    for (const familyFacts of Object.values(reportToCheck.facts)) {
        for (const item of familyFacts) {
            if (item.status === 'proved') {
                assert.ok(Array.isArray(item.proof) && item.proof.length > 0, `${item.id} should have proof`);
                assert.ok(Array.isArray(item.evidence), `${item.id} should have evidence array`);
            }
        }
    }
}

assertProvedFactsHaveProof(report);
assertProvedFactsHaveProof(mergeable);
assertProvedFactsHaveProof(transient);
```

- [ ] **Step 6: Run syntax and unit checks**

Run:

```bash
node --check src/tests/ps_static_analysis.js
node --check src/tests/run_ps_static_analysis.js
node --check src/tests/ps_static_analysis_node.js
node src/tests/ps_static_analysis_node.js
```

Expected: all checks pass and test prints `ps_static_analysis_node: ok`.

- [ ] **Step 7: Run corpus analysis**

Run:

```bash
node src/tests/run_ps_static_analysis.js src/demo src/tests/solver_tests --no-ps-tagged --out /private/tmp/ps_static_corpus.json
node -e "const fs=require('fs'); const j=JSON.parse(fs.readFileSync('/private/tmp/ps_static_corpus.json','utf8')); const statuses={}; for (const r of j.reports) statuses[r.status]=(statuses[r.status]||0)+1; console.log(JSON.stringify({source_count:j.source_count,statuses},null,2));"
```

Expected: the first command exits 0 and the second prints JSON with `source_count` greater than 100.

- [ ] **Step 8: Check Limerick mergeability output**

Run:

```bash
node src/tests/run_ps_static_analysis.js src/demo/limerick.txt --family mergeability --out /private/tmp/ps_static_limerick_merge.json
node -e "const fs=require('fs'); const j=JSON.parse(fs.readFileSync('/private/tmp/ps_static_limerick_merge.json','utf8')); const facts=j.reports[0].facts.mergeability || []; console.log(facts.filter(f=>JSON.stringify(f.subjects).toLowerCase().includes('body')).map(f=>({id:f.id,status:f.status,subjects:f.subjects,blockers:f.blockers})));"
```

Expected: prints mergeability facts mentioning body-like objects, or prints an empty list that is called out in the final implementation summary as a conservative-analysis gap.

- [ ] **Step 9: Commit**

```bash
git add src/tests/ps_static_analysis.js src/tests/run_ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: add static analyzer CLI and corpus checks"
```

---

## Completion Criteria

- `src/tests/run_ps_static_analysis.js` can analyze a single file and a directory.
- JSON reports contain `ps_tagged` by default and omit it with `--no-ps-tagged`.
- `ps_tagged` preserves original names and section -> group -> rule hierarchy.
- Rule terms use `{kind, ref, movement}` and keep absence, movement, and random-object choice distinct.
- Rule tags obey the implication table in the design doc.
- The four fact families are emitted with `proved`, `candidate`, or `rejected` statuses.
- Mergeability has hand-checkable fixture coverage for property reads, property negation, direct reads, direct negation, and direct win-condition distinction.
- `again` is tagged with `has_again`; no analysis tries to prove facts across drained again chains.
- Rigid rules block proved simplifications unless the rule is inert command-only.
- Corpus analysis over `src/demo` and `src/tests/solver_tests` completes without uncaught exceptions.

## Final Verification Commands

```bash
node --check src/tests/ps_static_analysis.js
node --check src/tests/run_ps_static_analysis.js
node --check src/tests/ps_static_analysis_node.js
node src/tests/canonicalizer_node.js
node src/tests/ps_static_analysis_node.js
node src/tests/run_ps_static_analysis.js src/demo src/tests/solver_tests --no-ps-tagged --out /private/tmp/ps_static_corpus.json
git diff --check
```
