#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {
    findRuleRecord,
    formatFixtureJson,
    loadClaimDescriptions,
    runObjectTagsDir,
    runRuleTagsDir,
} = require('./static_analysis_testdata_runner');

const FIXTURE_SCHEMA = 'ps-static-analysis-testdata-v1';

function findObjectTag(payload, object) {
    return payload.objectTag.find(item => item.object === object);
}

function findRuleTag(payload, text) {
    return payload.ruleTag.find(item => item.text === text);
}

function writeJson(filePath, payload) {
    fs.writeFileSync(filePath, `${formatFixtureJson(payload)}\n`, 'utf8');
}

function run() {
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-static-analysis-testdata-runner-'));
    try {
        const objectTagsDir = path.join(tmpRoot, 'object_tags');
        fs.mkdirSync(objectTagsDir, { recursive: true });
        fs.copyFileSync(
            path.join(__dirname, 'static_analysis_testdata', 'object_tags', 'roles-basic.txt'),
            path.join(objectTagsDir, 'roles-basic.txt'),
        );

        const claimDescriptions = loadClaimDescriptions();
        const ruleTagNames = claimDescriptions.ruleTags.map(tag => tag.name);
        assert.deepStrictEqual(ruleTagNames, [
            'objects_required',
            'objects_matched',
            'object_absences_matched',
            'movements_required',
            'movements_matched',
            'objects_written',
            'objects_erased',
            'movements_written',
            'movements_removed',
        ]);
        const generatedLog = [];
        runObjectTagsDir(objectTagsDir, claimDescriptions, message => generatedLog.push(message));
        assert.deepStrictEqual(generatedLog, ['generated static analysis testdata: object_tags/roles-basic.json (review before committing)\n']);

        const jsonPath = path.join(objectTagsDir, 'roles-basic.json');
        const generated = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        assert.strictEqual(generated.schema, FIXTURE_SCHEMA);
        assert.strictEqual(generated.objectTag.length, 3);
        assert.strictEqual(findObjectTag(generated, 'Avatar').is_player, true);
        assert.strictEqual(findObjectTag(generated, 'Avatar').created_by_rules, false);
        assert.strictEqual(findObjectTag(generated, 'Avatar').destroyed_by_rules, false);
        assert.strictEqual(findObjectTag(generated, 'Background').is_background, true);
        assert.strictEqual(findObjectTag(generated, 'Goal').level_presence, 'all');

        const curated = {
            schema: FIXTURE_SCHEMA,
            note: 'This intentionally keeps only one focused expectation.',
            objectTag: [
                {
                    object: 'Avatar',
                    is_player: true,
                },
            ],
        };
        writeJson(jsonPath, curated);
        const curatedText = fs.readFileSync(jsonPath, 'utf8');

        const rerunLog = [];
        runObjectTagsDir(objectTagsDir, claimDescriptions, message => rerunLog.push(message));
        assert.deepStrictEqual(rerunLog, []);
        assert.strictEqual(fs.readFileSync(jsonPath, 'utf8'), curatedText);

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

[ wall ] -> [ ]

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
        assert.deepStrictEqual(findRuleTag(generatedRulePayload, '[ wall ] -> [ ]').tags.objects_required, ['Wall']);
        assert.deepStrictEqual(findRuleTag(generatedRulePayload, '[ wall ] -> [ ]').tags.objects_erased, ['Wall']);
        const generatedRuleText = fs.readFileSync(ruleJsonPath, 'utf8');
        assert.ok(generatedRuleText.includes('"objects_required": ["Wall"]'));
        assert.ok(generatedRuleText.includes('"object_absences_matched": []'));
        assert.ok(!generatedRuleText.includes('"objects_required": [\n'));
        assert.strictEqual(formatFixtureJson({
            schema: FIXTURE_SCHEMA,
            ruleTag: [
                {
                    line: 40,
                    text: '[ wall ] -> [ ]',
                    tags: {
                        objects_required: ['Wall'],
                        object_absences_matched: [],
                    },
                },
            ],
        }).trim(), `{
  "schema": "ps-static-analysis-testdata-v1",
  "ruleTag": [
    {
      "line": 40,
      "text": "[ wall ] -> [ ]",
      "tags": {
        "objects_required": ["Wall"],
        "object_absences_matched": []
      }
    }
  ]
}`);

        const curatedRulePayload = {
            schema: FIXTURE_SCHEMA,
            ruleTag: [
                {
                    line: 40,
                    text: '[ wall ] -> [ ]',
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
                { line: 12, text: '[ wall ] -> [ ]', rule: { tags: {} } },
                { line: 12, text: '[ wall ] -> [ ]', rule: { tags: {} } },
            ], { line: 12, text: '[ wall ] -> [ ]' }),
            /matched 2 analyzed rules; expected exactly 1/
        );

        const nonIdempotentDir = path.join(tmpRoot, 'rule_tags_non_idempotent');
        fs.mkdirSync(nonIdempotentDir, { recursive: true });
        const nonIdempotentSource = `title Static Analysis Rule Tag Non Idempotent

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

[ Player no Wall ] -> [ Player Mark ]

=============
WINCONDITIONS
=============

Some Player

======
LEVELS
======

P#M
`;
        fs.writeFileSync(path.join(nonIdempotentDir, 'non-idempotent.txt'), nonIdempotentSource, 'utf8');
        writeJson(path.join(nonIdempotentDir, 'non-idempotent.json'), {
            schema: FIXTURE_SCHEMA,
            ruleTag: [
                {
                    line: 45,
                    text: '[ Player no Wall ] -> [ Player Mark ]',
                    tags: {
                        objects_written: ['Mark'],
                    },
                },
            ],
        });
        assert.throws(
            () => runRuleTagsDir(nonIdempotentDir, claimDescriptions, () => {}),
            /non-idempotent rule text/
        );
    } finally {
        fs.rmSync(tmpRoot, { recursive: true, force: true });
    }

    process.stdout.write('static_analysis_testdata_runner_node: ok\n');
}

run();
