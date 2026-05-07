# Static Analysis Testdata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-friction static-analysis testdata suite for first-slice object claims.

**Architecture:** Add a claim-description glossary, an area-grouped `static_analysis_testdata/object_tags/` corpus, and a focused Node runner that derives author-facing object claims from existing `analyzeSource()` output. The runner auto-creates missing `.json` files for orphan `.txt` files, never overwrites existing JSON, and checks only listed expectations.

**Tech Stack:** Node.js CommonJS test scripts, existing `src/tests/ps_static_analysis.js`, JSON testdata, `make static_analysis_tests`.

---

## File Structure

- Create `src/tests/static_analysis_claim_descriptions.json`
  - Shared glossary for valid, decided object claims.
  - First entries: `is_player`, `is_background`, `level_presence`, `not_created_or_destroyed_by_rules`.

- Create `src/tests/static_analysis_testdata_runner.js`
  - Discovers `src/tests/static_analysis_testdata/object_tags/*.txt` and `*.json`.
  - Generates missing JSON from orphan `.txt` files.
  - Validates and checks existing JSON expectations.
  - Exports helper functions for future unit tests.

- Create `src/tests/static_analysis_testdata/object_tags/roles-basic.txt`
  - Whole valid PuzzleScript source.
  - Exercises `is_player`, `is_background`, and simple level presence.

- Create `src/tests/static_analysis_testdata/object_tags/level-presence.txt`
  - Whole valid PuzzleScript source.
  - Exercises `level_presence` values `all`, `some`, and `none`; includes a message level so message levels are ignored.

- Create `src/tests/static_analysis_testdata/object_tags/rule-creation-destruction.txt`
  - Whole valid PuzzleScript source.
  - Exercises `not_created_or_destroyed_by_rules` true and false.

- Modify `Makefile`
  - Add `$(NODE) src/tests/static_analysis_testdata_runner.js` to `static_analysis_tests`.
  - Preserve all existing commands in that target.

---

### Task 1: Add Claim Descriptions And Seed Test Sources

**Files:**
- Create: `src/tests/static_analysis_claim_descriptions.json`
- Create: `src/tests/static_analysis_testdata/object_tags/roles-basic.txt`
- Create: `src/tests/static_analysis_testdata/object_tags/level-presence.txt`
- Create: `src/tests/static_analysis_testdata/object_tags/rule-creation-destruction.txt`

- [ ] **Step 1: Add `src/tests/static_analysis_claim_descriptions.json`**

```json
{
  "schema": "ps-static-analysis-claim-descriptions-v1",
  "objectTags": [
    {
      "name": "is_player",
      "description": "Object is part of the resolved Player role.",
      "specification": "An object has is_player when it is the object named by the Player object, Player synonym, or a member of the Player property in the compiled game. If Player resolves to a property with multiple member objects, every member object has is_player true."
    },
    {
      "name": "is_background",
      "description": "Object is part of the resolved Background role.",
      "specification": "An object has is_background when it is the object named by the Background object, Background synonym, or a member of the Background property in the compiled game. If Background resolves to a property with multiple member objects on the same collision layer, every member object has is_background true."
    },
    {
      "name": "level_presence",
      "description": "Whether the object appears in all, some, or no playable levels.",
      "specification": "The level_presence tag is one of all, some, or none. Message levels are ignored. The value all means there is at least one playable level and the object appears in every playable level. The value some means the object appears in at least one but not every playable level. The value none means the object appears in no playable levels; if a valid source has zero playable levels, every object has level_presence none.",
      "values": ["all", "some", "none"]
    },
    {
      "name": "not_created_or_destroyed_by_rules",
      "description": "No solver-active rule creates or destroys this object.",
      "specification": "An object has not_created_or_destroyed_by_rules when the analyzer proves no solver-active rule can create or destroy an instance of that object according to rule object-write analysis. Pure movement or relocation of an existing object does not count as creation or destruction."
    }
  ]
}
```

- [ ] **Step 2: Add `src/tests/static_analysis_testdata/object_tags/roles-basic.txt`**

```text
title Static Analysis Roles Basic

========
OBJECTS
========

Background
black

Avatar
white

Goal
yellow

========
LEGEND
========

. = Background
P = Avatar
G = Goal
Player = Avatar

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Avatar
Goal

=====
RULES
=====

=============
WINCONDITIONS
=============

Some Player on Goal

======
LEVELS
======

P.G
```

- [ ] **Step 3: Add `src/tests/static_analysis_testdata/object_tags/level-presence.txt`**

```text
title Static Analysis Level Presence

========
OBJECTS
========

Background
black

Player
white

Always
green

Sometimes
yellow

Never
red

========
LEGEND
========

. = Background
P = Player
A = Always
S = Sometimes
PlayerAlias = Player

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Always
Sometimes
Never

=====
RULES
=====

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

message message levels do not count

PA.

PAS
```

- [ ] **Step 4: Add `src/tests/static_analysis_testdata/object_tags/rule-creation-destruction.txt`**

```text
title Static Analysis Rule Creation Destruction

========
OBJECTS
========

Background
black

Player
white

Seed
green

Mark
yellow

Dust
gray

Wall
brown

========
LEGEND
========

. = Background
P = Player
S = Seed
M = Mark
D = Dust
# = Wall

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Seed
Mark
Dust
Wall

=====
RULES
=====

[ Seed ] -> [ Seed Mark ]
[ Dust ] -> [ ]
[ > Player | Wall ] -> [ > Player | Wall ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

PSMD#
```

- [ ] **Step 5: Run the missing runner to verify the test setup is red**

Run:

```bash
node src/tests/static_analysis_testdata_runner.js
```

Expected: FAIL because `src/tests/static_analysis_testdata_runner.js` does not exist yet. Do not commit this task yet; Task 2 will add the runner, generate expectations, and commit the first green suite.

---

### Task 2: Add The Object-Tag Testdata Runner

**Files:**
- Create: `src/tests/static_analysis_testdata_runner.js`
- Generated by command: `src/tests/static_analysis_testdata/object_tags/*.json`

- [ ] **Step 1: Create `src/tests/static_analysis_testdata_runner.js`**

```js
#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const { analyzeSource } = require('./ps_static_analysis');

const CLAIM_DESCRIPTIONS_PATH = path.join(__dirname, 'static_analysis_claim_descriptions.json');
const TESTDATA_ROOT = path.join(__dirname, 'static_analysis_testdata');
const FIXTURE_SCHEMA = 'ps-static-analysis-testdata-v1';

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, value) {
    fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function loadClaimDescriptions(filePath = CLAIM_DESCRIPTIONS_PATH) {
    const claims = readJson(filePath);
    assert.strictEqual(claims.schema, 'ps-static-analysis-claim-descriptions-v1', `${filePath}: unsupported claim-description schema`);
    assert.ok(Array.isArray(claims.objectTags), `${filePath}: objectTags must be an array`);
    const seen = new Set();
    for (const tag of claims.objectTags) {
        assert.ok(tag && typeof tag.name === 'string' && tag.name.length > 0, `${filePath}: object tag missing name`);
        assert.ok(!seen.has(tag.name), `${filePath}: duplicate object tag ${tag.name}`);
        seen.add(tag.name);
        assert.ok(typeof tag.description === 'string' && tag.description.length > 0, `${filePath}: ${tag.name} missing description`);
        assert.ok(typeof tag.specification === 'string' && tag.specification.length > 0, `${filePath}: ${tag.name} missing specification`);
        if (tag.values !== undefined) {
            assert.ok(Array.isArray(tag.values) && tag.values.every(value => typeof value === 'string'), `${filePath}: ${tag.name}.values must be string[]`);
        }
    }
    return claims;
}

function propertyMembers(psTagged, canonicalName) {
    const property = (psTagged.properties || []).find(item =>
        item.canonical_name === canonicalName || String(item.name).toLowerCase() === canonicalName
    );
    return property ? new Set(property.members || []) : null;
}

function roleObjectNames(psTagged, canonicalName) {
    const fromProperty = propertyMembers(psTagged, canonicalName);
    if (fromProperty) return fromProperty;
    const object = (psTagged.objects || []).find(item =>
        item.canonical_name === canonicalName || String(item.name).toLowerCase() === canonicalName
    );
    return new Set(object ? [object.name] : []);
}

function deriveLevelPresence(object) {
    const tags = object.tags || {};
    if (tags.present_in_all_levels) return 'all';
    if (tags.present_in_some_levels) return 'some';
    return 'none';
}

function deriveObjectTagValue(report, object, tagName) {
    const psTagged = report.ps_tagged || {};
    if (tagName === 'is_player') {
        return roleObjectNames(psTagged, 'player').has(object.name);
    }
    if (tagName === 'is_background') {
        return roleObjectNames(psTagged, 'background').has(object.name);
    }
    if (tagName === 'level_presence') {
        return deriveLevelPresence(object);
    }
    if (tagName === 'not_created_or_destroyed_by_rules') {
        return !!((object.tags || {}).count_invariant);
    }
    return !!((object.tags || {})[tagName]);
}

function buildObjectTagExpectations(report, claimDescriptions) {
    const objectTags = claimDescriptions.objectTags || [];
    const expect = [];
    for (const object of (report.ps_tagged && report.ps_tagged.objects) || []) {
        for (const tag of objectTags) {
            expect.push({
                type: 'objectTag',
                object: object.name,
                tag: tag.name,
                is: deriveObjectTagValue(report, object, tag.name),
            });
        }
    }
    return {
        schema: FIXTURE_SCHEMA,
        expect,
    };
}

function objectByName(report, objectName) {
    return ((report.ps_tagged && report.ps_tagged.objects) || []).find(object => object.name === objectName) || null;
}

function claimByName(claimDescriptions, tagName) {
    return (claimDescriptions.objectTags || []).find(tag => tag.name === tagName) || null;
}

function validateExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(Array.isArray(payload.expect), `${filePath}: expect must be an array`);
    for (const [index, item] of payload.expect.entries()) {
        assert.ok(item && item.type === 'objectTag', `${filePath}: expect[${index}] unsupported type ${item && item.type}`);
        assert.ok(typeof item.object === 'string' && item.object.length > 0, `${filePath}: expect[${index}] missing object`);
        assert.ok(typeof item.tag === 'string' && item.tag.length > 0, `${filePath}: expect[${index}] missing tag`);
        assert.ok(Object.prototype.hasOwnProperty.call(item, 'is'), `${filePath}: expect[${index}] missing is`);
    }
}

function checkObjectTagExpectation(filePath, report, claimDescriptions, expectation) {
    const claim = claimByName(claimDescriptions, expectation.tag);
    assert.ok(claim, `${filePath}: unknown object tag ${expectation.tag}`);

    const object = objectByName(report, expectation.object);
    if (!object) {
        const available = ((report.ps_tagged && report.ps_tagged.objects) || []).map(item => item.name).join(', ');
        assert.fail(`${filePath}: unknown object ${expectation.object}; available objects: ${available}`);
    }

    if (claim.values) {
        assert.ok(claim.values.includes(expectation.is), `${filePath}: ${expectation.tag} expected value must be one of ${claim.values.join(', ')}`);
    } else {
        assert.strictEqual(typeof expectation.is, 'boolean', `${filePath}: ${expectation.tag} expected value must be boolean`);
    }

    const actual = deriveObjectTagValue(report, object, expectation.tag);
    if (actual !== expectation.is) {
        assert.fail([
            `${filePath}`,
            `objectTag ${expectation.object}.${expectation.tag} expected ${JSON.stringify(expectation.is)}, got ${JSON.stringify(actual)}`,
            `  object: ${object.name} id=${object.id} layer=${object.layer}`,
        ].join('\n'));
    }
}

function checkFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    validateExpectationShape(jsonPath, payload);
    for (const expectation of payload.expect) {
        checkObjectTagExpectation(jsonPath, report, claimDescriptions, expectation);
    }
}

function sortedFiles(dirPath, ext) {
    return fs.readdirSync(dirPath)
        .filter(name => name.endsWith(ext))
        .sort((left, right) => left.localeCompare(right));
}

function runObjectTagsDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
    const txtFiles = sortedFiles(dirPath, '.txt');
    const jsonFiles = sortedFiles(dirPath, '.json');
    const txtStems = new Set(txtFiles.map(name => path.basename(name, '.txt')));
    const jsonStems = new Set(jsonFiles.map(name => path.basename(name, '.json')));

    for (const stem of jsonStems) {
        assert.ok(txtStems.has(stem), `${path.join(dirPath, `${stem}.json`)}: missing matching .txt`);
    }

    for (const txtName of txtFiles) {
        const stem = path.basename(txtName, '.txt');
        const txtPath = path.join(dirPath, txtName);
        const jsonPath = path.join(dirPath, `${stem}.json`);
        if (!fs.existsSync(jsonPath)) {
            const source = fs.readFileSync(txtPath, 'utf8');
            const report = analyzeSource(source, { sourcePath: txtPath });
            assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
            writeJson(jsonPath, buildObjectTagExpectations(report, claimDescriptions));
            log(`generated static analysis testdata: object_tags/${stem}.json\n`);
        }
        checkFixture(txtPath, jsonPath, claimDescriptions);
    }
}

function runStaticAnalysisTestdata(options = {}) {
    const root = options.root || TESTDATA_ROOT;
    const claimDescriptions = loadClaimDescriptions(options.claimDescriptionsPath || CLAIM_DESCRIPTIONS_PATH);
    const objectTagsDir = path.join(root, 'object_tags');
    assert.ok(fs.existsSync(objectTagsDir), `${objectTagsDir}: missing object_tags testdata directory`);
    runObjectTagsDir(objectTagsDir, claimDescriptions, options.log);
    process.stdout.write('static_analysis_testdata_runner: ok\n');
}

if (require.main === module) {
    runStaticAnalysisTestdata();
}

module.exports = {
    buildObjectTagExpectations,
    deriveObjectTagValue,
    loadClaimDescriptions,
    runObjectTagsDir,
    runStaticAnalysisTestdata,
};
```

- [ ] **Step 2: Run the runner to generate JSON and verify the first pass**

Run:

```bash
node src/tests/static_analysis_testdata_runner.js
```

Expected output includes:

```text
generated static analysis testdata: object_tags/level-presence.json
generated static analysis testdata: object_tags/roles-basic.json
generated static analysis testdata: object_tags/rule-creation-destruction.json
static_analysis_testdata_runner: ok
```

Expected filesystem changes:

```text
src/tests/static_analysis_testdata/object_tags/level-presence.json
src/tests/static_analysis_testdata/object_tags/roles-basic.json
src/tests/static_analysis_testdata/object_tags/rule-creation-destruction.json
```

- [ ] **Step 3: Re-run the runner and confirm no files are regenerated**

Run:

```bash
node src/tests/static_analysis_testdata_runner.js
```

Expected output:

```text
static_analysis_testdata_runner: ok
```

There should be no `generated static analysis testdata` lines on the second run.

- [ ] **Step 4: Commit the green testdata suite**

Run:

```bash
git add src/tests/static_analysis_claim_descriptions.json src/tests/static_analysis_testdata_runner.js src/tests/static_analysis_testdata/object_tags
git commit -m "test: add static analysis object tag testdata runner"
```

Expected: commit succeeds with the claim descriptions, seed `.txt` files, generated `.json` files, and runner.

---

### Task 3: Wire The Runner Into `make static_analysis_tests`

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the runner to the Makefile target**

In `Makefile`, update `static_analysis_tests` so it includes the new runner. Preserve existing lines.

Expected target body:

```make
static_analysis_tests:
	$(NODE) src/tests/ps_static_analysis_node.js
	$(NODE) src/tests/static_analysis_testdata_runner.js
	$(NODE) src/tests/static_analysis_explorer_node.js
	$(NODE) src/tests/solver_static_opt_node.js
	$(NODE) src/tests/compare_solver_static_opt_runs_node.js
```

- [ ] **Step 2: Run the full static-analysis test target**

Run:

```bash
make static_analysis_tests
```

Expected output includes:

```text
ps_static_analysis_node: ok
static_analysis_testdata_runner: ok
static_analysis_explorer_node: ok
solver_static_opt_node: ok
compare_solver_static_opt_runs_node: ok
```

- [ ] **Step 3: Commit the Makefile wiring**

Run:

```bash
git add Makefile
git commit -m "test: run static analysis testdata suite"
```

Expected: commit succeeds.

---

### Task 4: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run the focused runner directly**

Run:

```bash
node src/tests/static_analysis_testdata_runner.js
```

Expected:

```text
static_analysis_testdata_runner: ok
```

- [ ] **Step 2: Run the full target**

Run:

```bash
make static_analysis_tests
```

Expected output includes every static-analysis subtest ending in `ok`, with no generated JSON messages.

- [ ] **Step 3: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Inspect status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing worktree changes remain. No untracked or modified files from this plan should remain.
