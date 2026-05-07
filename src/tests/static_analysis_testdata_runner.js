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
