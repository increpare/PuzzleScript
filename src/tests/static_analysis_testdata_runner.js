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

function isInlineJsonArray(value) {
    return value.every(item => item === null || ['string', 'number', 'boolean'].includes(typeof item));
}

function formatJsonValue(value, depth) {
    const indent = '  '.repeat(depth);
    const childIndent = '  '.repeat(depth + 1);

    if (Array.isArray(value)) {
        if (value.length === 0) return '[]';
        if (isInlineJsonArray(value)) {
            return `[${value.map(item => JSON.stringify(item)).join(', ')}]`;
        }
        return `[\n${value.map(item => `${childIndent}${formatJsonValue(item, depth + 1)}`).join(',\n')}\n${indent}]`;
    }

    if (value && typeof value === 'object') {
        const entries = Object.keys(value).map(key =>
            `${childIndent}${JSON.stringify(key)}: ${formatJsonValue(value[key], depth + 1)}`
        );
        return entries.length === 0 ? '{}' : `{\n${entries.join(',\n')}\n${indent}}`;
    }

    return JSON.stringify(value);
}

function formatFixtureJson(value) {
    return formatJsonValue(value, 0);
}

function writeJson(filePath, value) {
    fs.writeFileSync(filePath, `${formatFixtureJson(value)}\n`, 'utf8');
}

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
    validateClaimDescriptionList(filePath, 'factFamilies', claims.factFamilies);
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
    if (tagName === 'created_by_rules') {
        return !!((object.tags || {}).may_be_created);
    }
    if (tagName === 'destroyed_by_rules') {
        return !!((object.tags || {}).may_be_destroyed);
    }
    return !!((object.tags || {})[tagName]);
}

function buildObjectTagExpectations(report, claimDescriptions) {
    const objectTags = claimDescriptions.objectTags || [];
    const objectTag = [];
    for (const object of (report.ps_tagged && report.ps_tagged.objects) || []) {
        const row = { object: object.name };
        for (const tag of objectTags) {
            row[tag.name] = deriveObjectTagValue(report, object, tag.name);
        }
        objectTag.push(row);
    }
    return {
        schema: FIXTURE_SCHEMA,
        objectTag,
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
    assert.ok(Array.isArray(payload.objectTag), `${filePath}: objectTag must be an array`);
    for (const [index, item] of payload.objectTag.entries()) {
        assert.ok(item && typeof item === 'object' && !Array.isArray(item), `${filePath}: objectTag[${index}] must be an object`);
        assert.ok(typeof item.object === 'string' && item.object.length > 0, `${filePath}: objectTag[${index}] missing object`);
        for (const tagName of Object.keys(item)) {
            assert.ok(tagName === 'object' || tagName.length > 0, `${filePath}: objectTag[${index}] has an empty tag name`);
        }
    }
}

function checkObjectTagExpectation(filePath, report, claimDescriptions, row, tagName) {
    const claim = claimByName(claimDescriptions, tagName);
    assert.ok(claim, `${filePath}: unknown object tag ${tagName}`);

    const object = objectByName(report, row.object);
    if (!object) {
        const available = ((report.ps_tagged && report.ps_tagged.objects) || []).map(item => item.name).join(', ');
        assert.fail(`${filePath}: unknown object ${row.object}; available objects: ${available}`);
    }

    const expected = row[tagName];
    if (claim.values) {
        assert.ok(claim.values.includes(expected), `${filePath}: ${tagName} expected value must be one of ${claim.values.join(', ')}`);
    } else {
        assert.strictEqual(typeof expected, 'boolean', `${filePath}: ${tagName} expected value must be boolean`);
    }

    const actual = deriveObjectTagValue(report, object, tagName);
    if (actual !== expected) {
        assert.fail([
            `${filePath}`,
            `objectTag ${row.object}.${tagName} expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
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
    for (const row of payload.objectTag) {
        for (const tagName of Object.keys(row)) {
            if (tagName !== 'object') checkObjectTagExpectation(jsonPath, report, claimDescriptions, row, tagName);
        }
    }
}

function termRefName(term) {
    const ref = term.ref || {};
    if (ref.type === 'object_set') return String(ref.source || ref.objects.join(' or ')).toLowerCase();
    if (ref.canonical_name) return String(ref.canonical_name).toLowerCase();
    if (ref.name) return String(ref.name).toLowerCase();
    if (ref.type === 'ellipsis') return '...';
    return '';
}

function renderRuleTerm(term) {
    if (term.kind === 'absent') return `no ${termRefName(term)}`;
    if (term.kind === 'random_object') return `random ${termRefName(term)}`;
    const name = termRefName(term);
    return term.movement === null ? name : `${term.movement} ${name}`;
}

function renderRuleCell(cell) {
    return cell.map(renderRuleTerm).join(' ');
}

function renderRuleSide(side) {
    return (side || []).map(row => {
        let text = '[';
        row.map(renderRuleCell).forEach((cell, index) => {
            if (index === 0) {
                if (cell.length > 0) text += ` ${cell}`;
            } else {
                text += cell.length > 0 ? ` | ${cell}` : ' |';
            }
        });
        return `${text} ]`;
    }).join(' ');
}

function ruleHasMultipleCells(rule) {
    return rule.lhs.concat(rule.rhs).some(row => row.length > 1);
}

function renderRuleCommands(rule) {
    return (rule.commands || [])
        .map(command => command.join(' '))
        .join(' ');
}

function renderRuleText(rule) {
    const lateMark = rule.late ? 'late ' : '';
    const prefix = ruleHasMultipleCells(rule) ? `${rule.direction} ` : '';
    const body = `${lateMark}${prefix}${renderRuleSide(rule.lhs)} -> ${renderRuleSide(rule.rhs)}`;
    const suffix = renderRuleCommands(rule);
    return suffix.length > 0 ? `${body} ${suffix}` : body;
}

function allRuleRecords(report, source) {
    const sourceLines = source.split(/\r?\n/);
    const records = [];
    for (const section of (report.ps_tagged && report.ps_tagged.rule_sections) || []) {
        for (const group of section.groups || []) {
            for (const rule of group.rules || []) {
                const text = (sourceLines[rule.source_line - 1] || '').trim();
                records.push({ rule, line: rule.source_line, text, canonicalText: renderRuleText(rule) });
            }
        }
    }
    return records;
}

function assertRuleRecordsIdempotent(filePath, records) {
    for (const record of records) {
        if (record.text !== record.canonicalText) {
            assert.fail([
                `${filePath}: non-idempotent rule text at line ${record.line}`,
                `  source:    ${record.text}`,
                `  canonical: ${record.canonicalText}`,
            ].join('\n'));
        }
    }
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
    const records = allRuleRecords(report, source);
    assertRuleRecordsIdempotent(report.source.path, records);
    return {
        schema: FIXTURE_SCHEMA,
        ruleTag: records.map(record => ({
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
    assertRuleRecordsIdempotent(txtPath, records);
    for (const row of payload.ruleTag) {
        const record = findRuleRecord(jsonPath, records, row);
        for (const tagName of Object.keys(row.tags)) {
            checkRuleTagExpectation(jsonPath, record, claimDescriptions, row.tags, tagName);
        }
    }
}

const REASON_VALUES = ['object_presence', 'object_absence', 'movement'];

function recordById(records) {
    const byId = new Map();
    for (const record of records) {
        byId.set(record.rule.id, record);
    }
    return byId;
}

function compareEdgeRows(a, b) {
    if (a.from_line !== b.from_line) return a.from_line - b.from_line;
    if (a.from_text !== b.from_text) return a.from_text.localeCompare(b.from_text);
    if (a.to_line !== b.to_line) return a.to_line - b.to_line;
    return a.to_text.localeCompare(b.to_text);
}

function compareAgainRows(a, b) {
    if (a.line !== b.line) return a.line - b.line;
    return a.text.localeCompare(b.text);
}

function programFlowFactValue(report) {
    const facts = (report.facts && report.facts.program_flow) || [];
    if (facts.length === 0) return { rule_ids: [], wake_edges: [], again_rules: [], tick_restart_possible: false };
    return facts[0].value;
}

function buildProgramFlowExpectations(source, report) {
    const records = allRuleRecords(report, source);
    assertRuleRecordsIdempotent(report.source.path, records);
    const byId = recordById(records);
    const value = programFlowFactValue(report);
    const wakeEdges = value.wake_edges.map(edge => {
        const from = byId.get(edge.from);
        const to = byId.get(edge.to);
        assert.ok(from, `program_flow edge from rule id ${edge.from} not found in records`);
        assert.ok(to, `program_flow edge to rule id ${edge.to} not found in records`);
        return {
            from_line: from.line,
            from_text: from.text,
            to_line: to.line,
            to_text: to.text,
            reasons: edge.reasons.slice(),
        };
    });
    wakeEdges.sort(compareEdgeRows);
    const againRules = value.again_rules.map(ruleId => {
        const record = byId.get(ruleId);
        assert.ok(record, `program_flow again rule id ${ruleId} not found in records`);
        return { line: record.line, text: record.text };
    });
    againRules.sort(compareAgainRows);
    return {
        schema: FIXTURE_SCHEMA,
        wakeEdges,
        againRules,
    };
}

function validateProgramFlowExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(Array.isArray(payload.wakeEdges), `${filePath}: wakeEdges must be an array`);
    assert.ok(Array.isArray(payload.againRules), `${filePath}: againRules must be an array`);
    for (const [index, edge] of payload.wakeEdges.entries()) {
        assert.ok(edge && typeof edge === 'object' && !Array.isArray(edge), `${filePath}: wakeEdges[${index}] must be an object`);
        assert.ok(Number.isInteger(edge.from_line) && edge.from_line > 0, `${filePath}: wakeEdges[${index}] missing positive integer from_line`);
        assert.ok(typeof edge.from_text === 'string' && edge.from_text.length > 0, `${filePath}: wakeEdges[${index}] missing from_text`);
        assert.ok(Number.isInteger(edge.to_line) && edge.to_line > 0, `${filePath}: wakeEdges[${index}] missing positive integer to_line`);
        assert.ok(typeof edge.to_text === 'string' && edge.to_text.length > 0, `${filePath}: wakeEdges[${index}] missing to_text`);
        assert.ok(Array.isArray(edge.reasons) && edge.reasons.length > 0, `${filePath}: wakeEdges[${index}].reasons must be a non-empty array`);
        for (const reason of edge.reasons) {
            assert.ok(REASON_VALUES.includes(reason), `${filePath}: wakeEdges[${index}].reasons contains unknown reason ${JSON.stringify(reason)}`);
        }
    }
    for (const [index, row] of payload.againRules.entries()) {
        assert.ok(row && typeof row === 'object' && !Array.isArray(row), `${filePath}: againRules[${index}] must be an object`);
        assert.ok(Number.isInteger(row.line) && row.line > 0, `${filePath}: againRules[${index}] missing positive integer line`);
        assert.ok(typeof row.text === 'string' && row.text.length > 0, `${filePath}: againRules[${index}] missing text`);
    }
}

function checkProgramFlowFixture(txtPath, jsonPath) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    validateProgramFlowExpectationShape(jsonPath, payload);
    const actual = buildProgramFlowExpectations(source, report);
    const expectedEdges = payload.wakeEdges.slice().sort(compareEdgeRows);
    const expectedAgain = payload.againRules.slice().sort(compareAgainRows);
    assert.deepStrictEqual(actual.wakeEdges, expectedEdges, `${jsonPath}: wakeEdges mismatch`);
    assert.deepStrictEqual(actual.againRules, expectedAgain, `${jsonPath}: againRules mismatch`);
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
            log(`generated static analysis testdata: object_tags/${stem}.json (review before committing)\n`);
        }
        checkFixture(txtPath, jsonPath, claimDescriptions);
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

function runProgramFlowDir(dirPath, log = process.stdout.write.bind(process.stdout)) {
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
            writeJson(jsonPath, buildProgramFlowExpectations(source, report));
            log(`generated static analysis testdata: program_flow/${stem}.json (review before committing)\n`);
        }
        checkProgramFlowFixture(txtPath, jsonPath);
    }
}

function allWinConditionRecords(report, source) {
    const sourceLines = source.split(/\r?\n/);
    return (report.ps_tagged && report.ps_tagged.winconditions || []).map(wincondition => {
        const text = (sourceLines[(wincondition.source_line || 1) - 1] || '').trim();
        return { wincondition, line: wincondition.source_line, text };
    });
}

function winConditionClaimByName(claimDescriptions, tagName) {
    return (claimDescriptions.winConditionTags || []).find(tag => tag.name === tagName) || null;
}

function deriveWinConditionTagValue(wincondition, tagName) {
    const value = wincondition.tags ? wincondition.tags[tagName] : undefined;
    return Array.isArray(value) ? value.slice() : [];
}

function buildWinConditionTagExpectations(source, report, claimDescriptions) {
    const winConditionTags = claimDescriptions.winConditionTags || [];
    const records = allWinConditionRecords(report, source);
    return {
        schema: FIXTURE_SCHEMA,
        winConditionTag: records.map(record => ({
            line: record.line,
            text: record.text,
            tags: Object.fromEntries(winConditionTags.map(tag => [
                tag.name,
                deriveWinConditionTagValue(record.wincondition, tag.name),
            ])),
        })),
    };
}

function validateWinConditionTagExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(Array.isArray(payload.winConditionTag), `${filePath}: winConditionTag must be an array`);
    for (const [index, item] of payload.winConditionTag.entries()) {
        assert.ok(item && typeof item === 'object' && !Array.isArray(item), `${filePath}: winConditionTag[${index}] must be an object`);
        assert.ok(Number.isInteger(item.line) && item.line > 0, `${filePath}: winConditionTag[${index}] missing positive integer line`);
        assert.ok(typeof item.text === 'string' && item.text.length > 0, `${filePath}: winConditionTag[${index}] missing text`);
        assert.ok(item.tags && typeof item.tags === 'object' && !Array.isArray(item.tags), `${filePath}: winConditionTag[${index}] missing tags object`);
    }
}

function findWinConditionRecord(filePath, records, expected) {
    const matches = records.filter(record => record.line === expected.line && record.text === expected.text);
    if (matches.length !== 1) {
        assert.fail(`${filePath}: winConditionTag line ${expected.line} text ${JSON.stringify(expected.text)} matched ${matches.length} analyzed win conditions; expected exactly 1`);
    }
    return matches[0];
}

function checkWinConditionFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    validateWinConditionTagExpectationShape(jsonPath, payload);
    const records = allWinConditionRecords(report, source);
    for (const row of payload.winConditionTag) {
        const record = findWinConditionRecord(jsonPath, records, row);
        for (const tagName of Object.keys(row.tags)) {
            const claim = winConditionClaimByName(claimDescriptions, tagName);
            assert.ok(claim, `${jsonPath}: unknown win condition tag ${tagName}`);
            assertStringArray(jsonPath, tagName, row.tags[tagName]);
            const actual = deriveWinConditionTagValue(record.wincondition, tagName);
            assertSameStringSet(jsonPath, `winConditionTag line ${record.line} ${record.text} ${tagName}`, row.tags[tagName], actual);
        }
    }
}

function runWinConditionTagsDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
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
            writeJson(jsonPath, buildWinConditionTagExpectations(source, report, claimDescriptions));
            log(`generated static analysis testdata: wincondition_tags/${stem}.json (review before committing)\n`);
        }
        checkWinConditionFixture(txtPath, jsonPath, claimDescriptions);
    }
}

function runStaticAnalysisTestdata(options = {}) {
    const root = options.root || TESTDATA_ROOT;
    const claimDescriptions = loadClaimDescriptions(options.claimDescriptionsPath || CLAIM_DESCRIPTIONS_PATH);
    const objectTagsDir = path.join(root, 'object_tags');
    assert.ok(fs.existsSync(objectTagsDir), `${objectTagsDir}: missing object_tags testdata directory`);
    runObjectTagsDir(objectTagsDir, claimDescriptions, options.log);
    const ruleTagsDir = path.join(root, 'rule_tags');
    assert.ok(fs.existsSync(ruleTagsDir), `${ruleTagsDir}: missing rule_tags testdata directory`);
    runRuleTagsDir(ruleTagsDir, claimDescriptions, options.log);
    const programFlowDir = path.join(root, 'program_flow');
    assert.ok(fs.existsSync(programFlowDir), `${programFlowDir}: missing program_flow testdata directory`);
    runProgramFlowDir(programFlowDir, options.log);
    const winConditionTagsDir = path.join(root, 'wincondition_tags');
    assert.ok(fs.existsSync(winConditionTagsDir), `${winConditionTagsDir}: missing wincondition_tags testdata directory`);
    runWinConditionTagsDir(winConditionTagsDir, claimDescriptions, options.log);
    process.stdout.write('static_analysis_testdata_runner: ok\n');
}

if (require.main === module) {
    runStaticAnalysisTestdata();
}

module.exports = {
    buildObjectTagExpectations,
    buildProgramFlowExpectations,
    buildRuleTagExpectations,
    buildWinConditionTagExpectations,
    deriveObjectTagValue,
    deriveRuleTagValue,
    deriveWinConditionTagValue,
    findRuleRecord,
    findWinConditionRecord,
    formatFixtureJson,
    loadClaimDescriptions,
    runObjectTagsDir,
    runProgramFlowDir,
    runRuleTagsDir,
    runStaticAnalysisTestdata,
    runWinConditionTagsDir,
};
