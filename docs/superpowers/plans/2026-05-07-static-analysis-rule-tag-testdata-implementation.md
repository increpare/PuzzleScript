# Static Analysis Rule Tag Testdata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first carefully scoped `ruleTag` static-analysis testdata slice for per-rule object read/write tags.

**Architecture:** Store rule object tags on each analyzed rule in `ps_tagged.rule_sections[].groups[].rules[].tags`, then have the static-analysis testdata runner validate and generate `rule_tags/*.json` fixtures using `line + text` identity. Keep the runner parallel to the existing `object_tags` path, with array values compared as sets and unlisted tags ignored.

**Tech Stack:** Node.js CommonJS test harnesses, existing PuzzleScript semantic compiler via `src/tests/ps_static_analysis.js`, existing `make static_analysis_tests` target.

---

## File Structure

- Modify `src/tests/static_analysis_claim_descriptions.json`: add the five approved `ruleTags` descriptions.
- Modify `src/tests/ps_static_analysis.js`: derive rule tag arrays after term refs are normalized, and tighten rule write analysis so OR-property LHS terms are matched but not required.
- Modify `src/tests/ps_static_analysis_node.js`: add focused analyzer tests for direct rules, relocation, and OR-property replacement.
- Modify `src/tests/static_analysis_testdata_runner.js`: validate/check/generate `rule_tags` fixtures.
- Modify `src/tests/static_analysis_testdata_runner_node.js`: test rule-tag generation, no-overwrite behavior, and ambiguous `line + text` failure.
- Modify `src/tests/static_analysis_testdata/README.md`: document how to add rule-tag tests and repeat the generated-testdata review policy.
- Create `src/tests/static_analysis_testdata/rule_tags/`: first small fixture directory.
- Create `src/tests/static_analysis_testdata/rule_tags/simple-object-delete.txt`
- Create `src/tests/static_analysis_testdata/rule_tags/simple-object-delete.json`
- Create `src/tests/static_analysis_testdata/rule_tags/absent-object-write.txt`
- Create `src/tests/static_analysis_testdata/rule_tags/absent-object-write.json`
- Create `src/tests/static_analysis_testdata/rule_tags/relocation-cell-local.txt`
- Create `src/tests/static_analysis_testdata/rule_tags/relocation-cell-local.json`

Do not commit any auto-generated fixture JSON from `src/tests/static_analysis_testdata/rule_tags/` until Stephen has reviewed it. The three nucleus fixtures below are hand-authored, not runner-generated.

### Task 1: Add Rule-Tag Claim Descriptions

**Files:**
- Modify: `src/tests/static_analysis_claim_descriptions.json`
- Modify: `src/tests/static_analysis_testdata_runner.js`
- Test: `src/tests/static_analysis_testdata_runner_node.js`

- [ ] **Step 1: Write the failing claim-description validation test**

In `src/tests/static_analysis_testdata_runner_node.js`, change the import to include `loadClaimDescriptions`, then add this assertion after `const claimDescriptions = loadClaimDescriptions();`:

```js
const ruleTagNames = claimDescriptions.ruleTags.map(tag => tag.name);
assert.deepStrictEqual(ruleTagNames, [
    'objects_required',
    'objects_matched',
    'object_absences_matched',
    'objects_written',
    'objects_erased',
]);
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```sh
node src/tests/static_analysis_testdata_runner_node.js
```

Expected: FAIL with a `TypeError` or assertion showing `claimDescriptions.ruleTags` is missing.

- [ ] **Step 3: Add `ruleTags` to the claim descriptions**

In `src/tests/static_analysis_claim_descriptions.json`, add a comma after the existing `objectTags` array and append:

```json
  "ruleTags": [
    {
      "name": "objects_required",
      "description": "Concrete objects definitely required present by the rule LHS.",
      "specification": "A rule has objects_required entries for concrete objects whose presence is definitely required by positive LHS terms. Single-object terms are required. Objects of required aggregates are all tagged as required. Members of an OR property are not individually required because any one member may satisfy the term."
    },
    {
      "name": "objects_matched",
      "description": "Concrete objects whose presence may satisfy a positive LHS term.",
      "specification": "A rule has objects_matched entries for every concrete object that may satisfy a positive LHS term after resolving synonyms, properties, and aggregates to object names. This includes concrete required objects, OR-property alternatives, and objects in aggregates."
    },
    {
      "name": "object_absences_matched",
      "description": "Concrete objects whose absence may satisfy a negative LHS term.",
      "specification": "A rule has object_absences_matched entries for every concrete object that may satisfy a negative LHS term after resolving synonyms, properties, and aggregates to object names. For example, no Obstacle records every concrete object in Obstacle."
    },
    {
      "name": "objects_written",
      "description": "Concrete objects written present by the rule RHS.",
      "specification": "A rule has objects_written entries for concrete objects the RHS may write present in a cell where the LHS does not already require that object present in the same cell. This is cell-local: relocation writes the object at the destination even if total object count is preserved."
    },
    {
      "name": "objects_erased",
      "description": "Concrete objects written absent by the rule RHS.",
      "specification": "A rule has objects_erased entries for concrete objects the RHS may write absent from a cell where the LHS allows that object to be present in the same cell. This includes explicit no terms, removal from the original cell, and possible same-layer erasure caused by writing another object on the layer."
    }
  ]
```

- [ ] **Step 4: Generalize claim-description validation**

In `src/tests/static_analysis_testdata_runner.js`, replace the object-tag-only validation loop inside `loadClaimDescriptions` with a small helper:

```js
function validateClaimDescriptionList(filePath, familyName, tags) {
    assert.ok(Array.isArray(tags), `${filePath}: ${familyName} must be an array`);
    const seen = new Set();
    for (const tag of tags) {
        assert.ok(tag && typeof tag.name === 'string' && tag.name.length > 0, `${filePath}: ${familyName} tag missing name`);
        assert.ok(!seen.has(tag.name), `${filePath}: duplicate ${familyName} tag ${tag.name}`);
        seen.add(tag.name);
        assert.ok(typeof tag.description === 'string' && tag.description.length > 0, `${filePath}: ${tag.name} missing description`);
        assert.ok(typeof tag.specification === 'string' && tag.specification.length > 0, `${filePath}: ${tag.name} missing specification`);
        if (tag.values !== undefined) {
            assert.ok(Array.isArray(tag.values) && tag.values.every(value => typeof value === 'string'), `${filePath}: ${tag.name}.values must be string[]`);
        }
    }
}

function loadClaimDescriptions(filePath = CLAIM_DESCRIPTIONS_PATH) {
    const claims = readJson(filePath);
    assert.strictEqual(claims.schema, 'ps-static-analysis-claim-descriptions-v1', `${filePath}: unsupported claim-description schema`);
    validateClaimDescriptionList(filePath, 'objectTags', claims.objectTags);
    validateClaimDescriptionList(filePath, 'ruleTags', claims.ruleTags);
    return claims;
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run:

```sh
node src/tests/static_analysis_testdata_runner_node.js
```

Expected: PASS and print `static_analysis_testdata_runner_node: ok`.

- [ ] **Step 6: Commit**

```sh
git add src/tests/static_analysis_claim_descriptions.json src/tests/static_analysis_testdata_runner.js src/tests/static_analysis_testdata_runner_node.js
git commit -m "test: register static analysis rule tags"
```

### Task 2: Derive Rule Tags For Simple Object Rules

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Test: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add failing analyzer tests for direct object tags**

Near the existing top-level helpers in `src/tests/ps_static_analysis_node.js`, add:

```js
function allAnalyzedRules(analysisReport) {
    return analysisReport.ps_tagged.rule_sections.flatMap(section =>
        section.groups.flatMap(group => group.rules)
    );
}

function onlyAnalyzedRule(source) {
    const analysisReport = analyzeSource(source);
    assert.strictEqual(analysisReport.status, 'ok');
    const rules = allAnalyzedRules(analysisReport);
    assert.strictEqual(rules.length, 1);
    return rules[0];
}
```

Then add this game and assertion block before the final `console.log`:

```js
const RULE_TAG_DELETE_GAME = `
title Static Analysis Rule Tag Delete

========
OBJECTS
========

Background
black

Player
white

Wall
brown

Mark
yellow

========
LEGEND
========

. = Background
P = Player
# = Wall
M = Mark

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Wall
Mark

=====
RULES
=====

[ Wall ] -> [ ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

P#M
`;

const deleteRule = onlyAnalyzedRule(RULE_TAG_DELETE_GAME);
assert.deepStrictEqual(deleteRule.tags.objects_required, ['Wall']);
assert.deepStrictEqual(deleteRule.tags.objects_matched, ['Wall']);
assert.deepStrictEqual(deleteRule.tags.object_absences_matched, []);
assert.deepStrictEqual(deleteRule.tags.objects_written, []);
assert.deepStrictEqual(deleteRule.tags.objects_erased, ['Wall']);
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```sh
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because `deleteRule.tags.objects_required` is `undefined`.

- [ ] **Step 3: Add rule-tag derivation helpers**

In `src/tests/ps_static_analysis.js`, add these helpers after `ruleFlowWrites`:

```js
function requiredPresentObjectSet(terms) {
    const objects = new Set();
    for (const term of terms) {
        if (term.kind === 'present' && term.ref && term.ref.type === 'object') {
            addValues(objects, termObjects(term));
        }
    }
    return objects;
}

function ruleFlowReadTags(rule) {
    const objectsRequired = new Set();
    const objectsMatched = new Set();
    const objectAbsencesMatched = new Set();
    for (const term of rule.summary.lhs_terms) {
        if (term.kind === 'present') {
            addValues(objectsMatched, termObjects(term));
            if (term.ref && term.ref.type === 'object') {
                addValues(objectsRequired, termObjects(term));
            }
        } else if (term.kind === 'absent') {
            addValues(objectAbsencesMatched, termObjects(term));
        }
    }
    return {
        objects_required: objectsRequired,
        objects_matched: objectsMatched,
        object_absences_matched: objectAbsencesMatched,
    };
}

function tagRuleObjectTags(psTagged) {
    for (const { rule } of allRuleEntries(psTagged)) {
        const reads = ruleFlowReadTags(rule);
        const writes = ruleFlowWrites(psTagged, rule);
        rule.tags.objects_required = uniqueSorted(reads.objects_required);
        rule.tags.objects_matched = uniqueSorted(reads.objects_matched);
        rule.tags.object_absences_matched = uniqueSorted(reads.object_absences_matched);
        rule.tags.objects_written = uniqueSorted(writes.object_present);
        rule.tags.objects_erased = uniqueSorted(writes.object_absent);
    }
}
```

In `buildPsTagged`, call the new pass immediately after `normalizeTermRefs(psTagged);`:

```js
    normalizeTermRefs(psTagged);
    tagRuleObjectTags(psTagged);
    tagInertCollisionLayers(psTagged);
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```sh
node src/tests/ps_static_analysis_node.js
```

Expected: PASS and print `ps_static_analysis_node: ok`.

- [ ] **Step 5: Commit**

```sh
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "feat: derive static analysis rule object tags"
```

### Task 3: Separate Required Objects From Matched OR-Property Objects

**Files:**
- Modify: `src/tests/ps_static_analysis.js`
- Test: `src/tests/ps_static_analysis_node.js`

- [ ] **Step 1: Add failing tests for absent terms, relocation, and OR-property replacement**

Append these assertions before the final `console.log` in `src/tests/ps_static_analysis_node.js`:

```js
const RULE_TAG_ABSENT_WRITE_GAME = `
title Static Analysis Rule Tag Absent Write

========
OBJECTS
========

Background
black

Player
white

Wall
brown

Mark
yellow

========
LEGEND
========

. = Background
P = Player
# = Wall
M = Mark

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Wall
Mark

=====
RULES
=====

[ Player no Wall ] -> [ Player stationary Mark ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

P#M
`;

const absentWriteRule = onlyAnalyzedRule(RULE_TAG_ABSENT_WRITE_GAME);
assert.deepStrictEqual(absentWriteRule.tags.objects_required, ['Player']);
assert.deepStrictEqual(absentWriteRule.tags.objects_matched, ['Player']);
assert.deepStrictEqual(absentWriteRule.tags.object_absences_matched, ['Wall']);
assert.deepStrictEqual(absentWriteRule.tags.objects_written, ['Mark']);
assert.deepStrictEqual(absentWriteRule.tags.objects_erased, []);

const RULE_TAG_RELOCATION_GAME = `
title Static Analysis Rule Tag Relocation

========
OBJECTS
========

Background
black

Player
white

Alpha
red

========
LEGEND
========

. = Background
P = Player
A = Alpha

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Alpha

=====
RULES
=====

right [ Alpha | ] -> [ | Alpha ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

PA
`;

const relocationRule = onlyAnalyzedRule(RULE_TAG_RELOCATION_GAME);
assert.deepStrictEqual(relocationRule.tags.objects_required, ['Alpha']);
assert.deepStrictEqual(relocationRule.tags.objects_matched, ['Alpha']);
assert.deepStrictEqual(relocationRule.tags.object_absences_matched, []);
assert.deepStrictEqual(relocationRule.tags.objects_written, ['Alpha']);
assert.deepStrictEqual(relocationRule.tags.objects_erased, ['Alpha']);

const RULE_TAG_PROPERTY_REPLACEMENT_GAME = `
title Static Analysis Rule Tag Property Replacement

========
OBJECTS
========

Background
black

Player
white

Crate
green

Wall
brown

========
LEGEND
========

. = Background
P = Player
C = Crate
# = Wall
Obstacle = Crate or Wall

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Crate, Wall

=====
RULES
=====

[ Obstacle ] -> [ Crate ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

PC#
`;

const propertyReplacementRule = onlyAnalyzedRule(RULE_TAG_PROPERTY_REPLACEMENT_GAME);
assert.deepStrictEqual(propertyReplacementRule.tags.objects_required, []);
assert.deepStrictEqual(propertyReplacementRule.tags.objects_matched, ['Crate', 'Wall']);
assert.deepStrictEqual(propertyReplacementRule.tags.object_absences_matched, []);
assert.deepStrictEqual(propertyReplacementRule.tags.objects_written, ['Crate']);
assert.deepStrictEqual(propertyReplacementRule.tags.objects_erased, ['Wall']);
```

- [ ] **Step 2: Run the test to verify it fails on the property replacement case**

Run:

```sh
node src/tests/ps_static_analysis_node.js
```

Expected: FAIL because the current write logic treats `Crate` as already present whenever `[ Obstacle ]` matches and misses at least `objects_written: ["Crate"]`.

- [ ] **Step 3: Update cell-local write analysis to use required-vs-matched LHS sets**

In `src/tests/ps_static_analysis.js`, move `requiredPresentObjectSet` above `ruleFlowWrites` if needed, then change the inner cell loop in `ruleFlowWrites` from:

```js
                const lhsPresent = presentObjectSet(lhsCell);
                const lhsAbsent = absentObjectSet(lhsCell);
                const rhsCellPresent = presentObjectSet(rhsCell);
```

to:

```js
                const lhsRequiredPresent = requiredPresentObjectSet(lhsCell);
                const lhsMatchedPresent = presentObjectSet(lhsCell);
                const lhsAbsent = absentObjectSet(lhsCell);
                const rhsCellPresent = presentObjectSet(rhsCell);
```

Change the RHS-present write filter from:

```js
                        const writtenObjects = termObjects(term).filter(objectName =>
                            term.kind === 'random_object' || !lhsPresent.has(objectName)
                        );
```

to:

```js
                        const writtenObjects = termObjects(term).filter(objectName =>
                            term.kind === 'random_object' || !lhsRequiredPresent.has(objectName)
                        );
```

Change the same-layer erasure check from:

```js
                                if (cellCouldContainObjectBefore(psTagged, lhsPresent, lhsAbsent, sibling)) {
                                    objectAbsent.add(sibling);
                                }
```

to:

```js
                                if (cellCouldContainObjectBefore(psTagged, lhsMatchedPresent, lhsAbsent, sibling)) {
                                    objectAbsent.add(sibling);
                                }
```

Change explicit absent writes from:

```js
                            if (cellCouldContainObjectBefore(psTagged, lhsPresent, lhsAbsent, objectName)) {
                                objectAbsent.add(objectName);
                            }
```

to:

```js
                            if (cellCouldContainObjectBefore(psTagged, lhsMatchedPresent, lhsAbsent, objectName)) {
                                objectAbsent.add(objectName);
                            }
```

Change LHS-removal iteration from:

```js
                for (const objectName of lhsPresent) {
                    if (!rhsCellPresent.has(objectName)) {
                        objectAbsent.add(objectName);
                    }
                }
```

to:

```js
                for (const objectName of lhsMatchedPresent) {
                    if (!rhsCellPresent.has(objectName)) {
                        objectAbsent.add(objectName);
                    }
                }
```

- [ ] **Step 4: Run the analyzer tests**

Run:

```sh
node src/tests/ps_static_analysis_node.js
```

Expected: PASS and print `ps_static_analysis_node: ok`.

- [ ] **Step 5: Run object-tag regression tests**

Run:

```sh
node src/tests/static_analysis_testdata_runner.js
```

Expected: PASS and print `static_analysis_testdata_runner: ok`; no committed object-tag JSON should be regenerated.

- [ ] **Step 6: Commit**

```sh
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js
git commit -m "fix: distinguish matched and required rule objects"
```

### Task 4: Add Rule-Tag Testdata Runner Support

**Files:**
- Modify: `src/tests/static_analysis_testdata_runner.js`
- Modify: `src/tests/static_analysis_testdata_runner_node.js`

- [ ] **Step 1: Add failing runner-node coverage for rule-tag generation and no-overwrite**

In `src/tests/static_analysis_testdata_runner_node.js`, update the import:

```js
const {
    findRuleRecord,
    loadClaimDescriptions,
    runObjectTagsDir,
    runRuleTagsDir,
} = require('./static_analysis_testdata_runner');
```

Add this helper:

```js
function findRuleTag(payload, text) {
    return payload.ruleTag.find(item => item.text === text);
}
```

Inside `run()`, after the object-tag no-overwrite assertions, add:

```js
        const ruleTagsDir = path.join(tmpRoot, 'rule_tags');
        fs.mkdirSync(ruleTagsDir, { recursive: true });
        const ruleTagSource = `title Static Analysis Rule Tag Tmp

========
OBJECTS
========

Background
black

Player
white

Wall
brown

========
LEGEND
========

. = Background
P = Player
# = Wall

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Wall

=====
RULES
=====

[ Wall ] -> [ ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

P#
`;
        fs.writeFileSync(path.join(ruleTagsDir, 'tmp-rule.txt'), ruleTagSource, 'utf8');

        const generatedRuleLog = [];
        runRuleTagsDir(ruleTagsDir, claimDescriptions, message => generatedRuleLog.push(message));
        assert.deepStrictEqual(generatedRuleLog, ['generated static analysis testdata: rule_tags/tmp-rule.json (review before committing)\n']);

        const ruleJsonPath = path.join(ruleTagsDir, 'tmp-rule.json');
        const generatedRulePayload = JSON.parse(fs.readFileSync(ruleJsonPath, 'utf8'));
        assert.strictEqual(generatedRulePayload.schema, FIXTURE_SCHEMA);
        assert.strictEqual(generatedRulePayload.ruleTag.length, 1);
        assert.deepStrictEqual(findRuleTag(generatedRulePayload, '[ Wall ] -> [ ]').tags.objects_required, ['Wall']);
        assert.deepStrictEqual(findRuleTag(generatedRulePayload, '[ Wall ] -> [ ]').tags.objects_erased, ['Wall']);

        const curatedRulePayload = {
            schema: FIXTURE_SCHEMA,
            ruleTag: [
                {
                    line: 40,
                    text: '[ Wall ] -> [ ]',
                    tags: {
                        objects_erased: ['Wall'],
                    },
                },
            ],
        };
        writeJson(ruleJsonPath, curatedRulePayload);
        const curatedRuleText = fs.readFileSync(ruleJsonPath, 'utf8');

        const rerunRuleLog = [];
        runRuleTagsDir(ruleTagsDir, claimDescriptions, message => rerunRuleLog.push(message));
        assert.deepStrictEqual(rerunRuleLog, []);
        assert.strictEqual(fs.readFileSync(ruleJsonPath, 'utf8'), curatedRuleText);

        assert.throws(
            () => findRuleRecord('ambiguous-rule.json', [
                { line: 12, text: '[ Wall ] -> [ ]', rule: { tags: {} } },
                { line: 12, text: '[ Wall ] -> [ ]', rule: { tags: {} } },
            ], { line: 12, text: '[ Wall ] -> [ ]' }),
            /matched 2 analyzed rules; expected exactly 1/
        );
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```sh
node src/tests/static_analysis_testdata_runner_node.js
```

Expected: FAIL with `runRuleTagsDir` missing or not a function.

- [ ] **Step 3: Implement rule-tag helpers in the runner**

In `src/tests/static_analysis_testdata_runner.js`, add:

```js
function allRuleRecords(report, source) {
    const sourceLines = source.split(/\r?\n/);
    const records = [];
    for (const section of (report.ps_tagged && report.ps_tagged.rule_sections) || []) {
        for (const group of section.groups || []) {
            for (const rule of group.rules || []) {
                const text = (sourceLines[rule.source_line - 1] || '').trim();
                records.push({ rule, line: rule.source_line, text });
            }
        }
    }
    return records;
}

function ruleClaimByName(claimDescriptions, tagName) {
    return (claimDescriptions.ruleTags || []).find(tag => tag.name === tagName) || null;
}

function deriveRuleTagValue(rule, tagName) {
    const value = rule.tags ? rule.tags[tagName] : undefined;
    return Array.isArray(value) ? value.slice() : [];
}

function assertStringArray(filePath, label, value) {
    assert.ok(Array.isArray(value), `${filePath}: ${label} expected value must be string[]`);
    assert.ok(value.every(item => typeof item === 'string'), `${filePath}: ${label} expected value must be string[]`);
}

function sortedStringSet(value) {
    return Array.from(new Set(value)).sort((left, right) =>
        left.localeCompare(right, undefined, { numeric: true })
    );
}

function assertSameStringSet(filePath, label, expected, actual) {
    const expectedSet = sortedStringSet(expected);
    const actualSet = sortedStringSet(actual);
    if (JSON.stringify(expectedSet) !== JSON.stringify(actualSet)) {
        assert.fail(`${filePath}\n${label} expected ${JSON.stringify(expectedSet)}, got ${JSON.stringify(actualSet)}`);
    }
}

function buildRuleTagExpectations(source, report, claimDescriptions) {
    const ruleTags = claimDescriptions.ruleTags || [];
    return {
        schema: FIXTURE_SCHEMA,
        ruleTag: allRuleRecords(report, source).map(record => ({
            line: record.line,
            text: record.text,
            tags: Object.fromEntries(ruleTags.map(tag => [
                tag.name,
                deriveRuleTagValue(record.rule, tag.name),
            ])),
        })),
    };
}

function validateRuleTagExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(Array.isArray(payload.ruleTag), `${filePath}: ruleTag must be an array`);
    for (const [index, item] of payload.ruleTag.entries()) {
        assert.ok(item && typeof item === 'object' && !Array.isArray(item), `${filePath}: ruleTag[${index}] must be an object`);
        assert.ok(Number.isInteger(item.line) && item.line > 0, `${filePath}: ruleTag[${index}] missing positive integer line`);
        assert.ok(typeof item.text === 'string' && item.text.length > 0, `${filePath}: ruleTag[${index}] missing text`);
        assert.ok(item.tags && typeof item.tags === 'object' && !Array.isArray(item.tags), `${filePath}: ruleTag[${index}] missing tags object`);
    }
}

function findRuleRecord(filePath, records, expected) {
    const matches = records.filter(record => record.line === expected.line && record.text === expected.text);
    if (matches.length !== 1) {
        assert.fail(`${filePath}: ruleTag line ${expected.line} text ${JSON.stringify(expected.text)} matched ${matches.length} analyzed rules; expected exactly 1`);
    }
    return matches[0];
}

function checkRuleTagExpectation(filePath, record, claimDescriptions, tags, tagName) {
    const claim = ruleClaimByName(claimDescriptions, tagName);
    assert.ok(claim, `${filePath}: unknown rule tag ${tagName}`);
    const expected = tags[tagName];
    assertStringArray(filePath, tagName, expected);
    const actual = deriveRuleTagValue(record.rule, tagName);
    assertSameStringSet(filePath, `ruleTag line ${record.line} ${record.text} ${tagName}`, expected, actual);
}

function checkRuleFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    validateRuleTagExpectationShape(jsonPath, payload);
    const records = allRuleRecords(report, source);
    for (const row of payload.ruleTag) {
        const record = findRuleRecord(jsonPath, records, row);
        for (const tagName of Object.keys(row.tags)) {
            checkRuleTagExpectation(jsonPath, record, claimDescriptions, row.tags, tagName);
        }
    }
}

function runRuleTagsDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
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
            writeJson(jsonPath, buildRuleTagExpectations(source, report, claimDescriptions));
            log(`generated static analysis testdata: rule_tags/${stem}.json (review before committing)\n`);
        }
        checkRuleFixture(txtPath, jsonPath, claimDescriptions);
    }
}
```

Update `runStaticAnalysisTestdata`:

```js
    const objectTagsDir = path.join(root, 'object_tags');
    assert.ok(fs.existsSync(objectTagsDir), `${objectTagsDir}: missing object_tags testdata directory`);
    runObjectTagsDir(objectTagsDir, claimDescriptions, options.log);
    const ruleTagsDir = path.join(root, 'rule_tags');
    assert.ok(fs.existsSync(ruleTagsDir), `${ruleTagsDir}: missing rule_tags testdata directory`);
    runRuleTagsDir(ruleTagsDir, claimDescriptions, options.log);
```

Update `module.exports`:

```js
    buildRuleTagExpectations,
    deriveRuleTagValue,
    findRuleRecord,
    runRuleTagsDir,
```

- [ ] **Step 4: Run the runner-node test**

Run:

```sh
node src/tests/static_analysis_testdata_runner_node.js
```

Expected: PASS and print `static_analysis_testdata_runner_node: ok`.

- [ ] **Step 5: Commit**

```sh
git add src/tests/static_analysis_testdata_runner.js src/tests/static_analysis_testdata_runner_node.js
git commit -m "feat: run static analysis rule tag testdata"
```

### Task 5: Add The Small Rule-Tag Fixture Nucleus

**Files:**
- Create: `src/tests/static_analysis_testdata/rule_tags/simple-object-delete.txt`
- Create: `src/tests/static_analysis_testdata/rule_tags/simple-object-delete.json`
- Create: `src/tests/static_analysis_testdata/rule_tags/absent-object-write.txt`
- Create: `src/tests/static_analysis_testdata/rule_tags/absent-object-write.json`
- Create: `src/tests/static_analysis_testdata/rule_tags/relocation-cell-local.txt`
- Create: `src/tests/static_analysis_testdata/rule_tags/relocation-cell-local.json`

- [ ] **Step 1: Create the fixture directory**

Create:

```sh
mkdir -p src/tests/static_analysis_testdata/rule_tags
```

- [ ] **Step 2: Add `simple-object-delete.txt`**

```text
title Static Analysis Rule Tag Delete

========
OBJECTS
========

Background
black

Player
white

Wall
brown

Mark
yellow

========
LEGEND
========

. = Background
P = Player
# = Wall
M = Mark

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Wall
Mark

=====
RULES
=====

[ Wall ] -> [ ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

P#M
```

- [ ] **Step 3: Add `simple-object-delete.json`**

```json
{
  "schema": "ps-static-analysis-testdata-v1",
  "ruleTag": [
    {
      "line": 45,
      "text": "[ Wall ] -> [ ]",
      "tags": {
        "objects_required": ["Wall"],
        "objects_matched": ["Wall"],
        "object_absences_matched": [],
        "objects_written": [],
        "objects_erased": ["Wall"]
      }
    }
  ]
}
```

- [ ] **Step 4: Add `absent-object-write.txt`**

```text
title Static Analysis Rule Tag Absent Write

========
OBJECTS
========

Background
black

Player
white

Wall
brown

Mark
yellow

========
LEGEND
========

. = Background
P = Player
# = Wall
M = Mark

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Wall
Mark

=====
RULES
=====

[ Player no Wall ] -> [ Player stationary Mark ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

P#M
```

- [ ] **Step 5: Add `absent-object-write.json`**

```json
{
  "schema": "ps-static-analysis-testdata-v1",
  "ruleTag": [
    {
      "line": 45,
      "text": "[ Player no Wall ] -> [ Player stationary Mark ]",
      "tags": {
        "objects_required": ["Player"],
        "objects_matched": ["Player"],
        "object_absences_matched": ["Wall"],
        "objects_written": ["Mark"],
        "objects_erased": []
      }
    }
  ]
}
```

- [ ] **Step 6: Add `relocation-cell-local.txt`**

```text
title Static Analysis Rule Tag Relocation

========
OBJECTS
========

Background
black

Player
white

Alpha
red

========
LEGEND
========

. = Background
P = Player
A = Alpha

========
SOUNDS
========

================
COLLISIONLAYERS
================

Background
Player
Alpha

=====
RULES
=====

right [ Alpha | ] -> [ | Alpha ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

PA
```

- [ ] **Step 7: Add `relocation-cell-local.json`**

```json
{
  "schema": "ps-static-analysis-testdata-v1",
  "ruleTag": [
    {
      "line": 40,
      "text": "right [ Alpha | ] -> [ | Alpha ]",
      "tags": {
        "objects_required": ["Alpha"],
        "objects_matched": ["Alpha"],
        "object_absences_matched": [],
        "objects_written": ["Alpha"],
        "objects_erased": ["Alpha"]
      }
    }
  ]
}
```

- [ ] **Step 8: Run the testdata runner**

Run:

```sh
node src/tests/static_analysis_testdata_runner.js
```

Expected: PASS and print `static_analysis_testdata_runner: ok`; it must not print any generated `rule_tags/*.json` message for these hand-authored fixtures.

- [ ] **Step 9: Commit**

```sh
git add src/tests/static_analysis_testdata/rule_tags
git commit -m "test: add static analysis rule tag fixtures"
```

### Task 6: Document The Rule-Tag Test Workflow

**Files:**
- Modify: `src/tests/static_analysis_testdata/README.md`

- [ ] **Step 1: Add README coverage**

Append this section after “Adding An Object-Tag Test”:

```md
## Adding A Rule-Tag Test

1. Add a small whole-source `.txt` file under `rule_tags/`.
2. Run:

   ```sh
   make static_analysis_tests
   ```

3. The runner will create the missing matching `.json` file and print a
   `generated static analysis testdata` message.
4. Read the generated JSON with Stephen before committing it. Keep only the
   rule rows and tag fields that express the intended test.
5. Run `make static_analysis_tests` again. Existing JSON files are never
   overwritten, and only listed rule-tag expectations are checked.

Rule-tag expectations identify rules by `line` plus trimmed source `text`.
If those two fields do not identify exactly one analyzed rule, the runner fails
instead of inventing another locator.

Rule-tag test sources must use compiler-idempotent rule text, so the fixture is
testing the analysis rather than incidental rule expansion. Non-idempotent rule
text belongs in a separate fixture area whose explicit purpose is compiler
normalization or rule decomposition.
```

- [ ] **Step 2: Run no-op README verification**

Run:

```sh
node src/tests/static_analysis_testdata_runner_node.js
```

Expected: PASS and print `static_analysis_testdata_runner_node: ok`.

- [ ] **Step 3: Commit**

```sh
git add src/tests/static_analysis_testdata/README.md
git commit -m "docs: explain static analysis rule tag testdata"
```

### Task 7: Final Verification

**Files:**
- Verify all files touched by Tasks 1-6.

- [ ] **Step 1: Run the focused analyzer suite**

Run:

```sh
node src/tests/ps_static_analysis_node.js
```

Expected: PASS and print `ps_static_analysis_node: ok`.

- [ ] **Step 2: Run the testdata self-test**

Run:

```sh
node src/tests/static_analysis_testdata_runner_node.js
```

Expected: PASS and print `static_analysis_testdata_runner_node: ok`.

- [ ] **Step 3: Run the full static-analysis target**

Run:

```sh
make static_analysis_tests
```

Expected: PASS with these lines present:

```text
ps_static_analysis_node: ok
static_analysis_testdata_runner: ok
static_analysis_testdata_runner_node: ok
static_analysis_explorer_node: ok
solver_static_opt_node: ok
compare_solver_static_opt_runs_node: ok
```

- [ ] **Step 4: Check for unreviewed generated fixtures**

Run:

```sh
git status --short
```

Expected: no unexpected `src/tests/static_analysis_testdata/rule_tags/*.json` files. If any rule-tag JSON was generated rather than hand-authored during implementation, show Stephen the diff and get explicit approval before staging it.

- [ ] **Step 5: Commit any final fixes**

If Task 7 required changes, commit them:

```sh
git add src/tests/ps_static_analysis.js src/tests/ps_static_analysis_node.js src/tests/static_analysis_testdata_runner.js src/tests/static_analysis_testdata_runner_node.js src/tests/static_analysis_testdata/README.md src/tests/static_analysis_testdata/rule_tags src/tests/static_analysis_claim_descriptions.json
git commit -m "test: wire static analysis rule tag testdata"
```

Skip this step if there are no changes after Tasks 1-6.

## Self-Review

- Spec coverage: the plan covers claim descriptions, per-rule tag derivation, `rule_tags/` fixture format, `line + text` identity, set comparison, orphan `.txt` generation, no overwrite behavior, generated-test review policy, and the initial small specimen nucleus.
- Scope check: this deliberately does not add rule-group flow, movement-flow decomposition, dashboard UI, static object tags, or cosmetic object tags.
- Semantics check: the OR-property replacement test locks in Stephen’s chosen semantics for `[ Obstacle ] -> [ Crate ]`: `objects_written: ["Crate"]`, `objects_erased: ["Wall"]`.
- Placeholder scan: no pending/expected-failure area is introduced; every committed fixture is either valid or absent.
