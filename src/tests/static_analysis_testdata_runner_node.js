#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {
    loadClaimDescriptions,
    runObjectTagsDir,
} = require('./static_analysis_testdata_runner');

const FIXTURE_SCHEMA = 'ps-static-analysis-testdata-v1';

function findObjectTag(payload, object) {
    return payload.objectTag.find(item => item.object === object);
}

function writeJson(filePath, payload) {
    fs.writeFileSync(filePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
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
    } finally {
        fs.rmSync(tmpRoot, { recursive: true, force: true });
    }

    process.stdout.write('static_analysis_testdata_runner_node: ok\n');
}

run();
