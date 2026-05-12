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
        if (tag.fields !== undefined) {
            validateClaimDescriptionList(filePath, `${familyName}.${tag.name}.fields`, tag.fields);
        }
        if (tag.items !== undefined) {
            assert.ok(tag.items && typeof tag.items === 'object' && !Array.isArray(tag.items), `${filePath}: ${tag.name}.items must be an object`);
            if (tag.items.fields !== undefined) {
                validateClaimDescriptionList(filePath, `${familyName}.${tag.name}[].fields`, tag.items.fields);
            }
        }
    }
}

function loadClaimDescriptions(filePath = CLAIM_DESCRIPTIONS_PATH) {
    const claims = readJson(filePath);
    assert.strictEqual(claims.schema, 'ps-static-analysis-claim-descriptions-v1', `${filePath}: unsupported claim-description schema`);
    validateClaimDescriptionList(filePath, 'fixtureSchemas', claims.fixtureSchemas);
    return claims;
}

function fixtureSchemaByName(claimDescriptions, fixtureName) {
    const fixtureSchema = (claimDescriptions.fixtureSchemas || []).find(item => item.name === fixtureName) || null;
    assert.ok(fixtureSchema, `static analysis claim descriptions missing fixture schema ${fixtureName}`);
    return fixtureSchema;
}

function fieldByName(fields, fieldName) {
    return (fields || []).find(field => field.name === fieldName) || null;
}

function childFieldsForField(field) {
    if (!field) return [];
    if (field.fields) return field.fields;
    if (field.items && field.items.fields) return field.items.fields;
    return [];
}

function fixtureFieldsAtPath(fixtureSchema, pathParts) {
    let fields = fixtureSchema.fields || [];
    for (const part of pathParts) {
        const field = fieldByName(fields, part);
        assert.ok(field, `fixture schema ${fixtureSchema.name} missing path ${pathParts.join('.')}`);
        fields = childFieldsForField(field);
    }
    return fields;
}

function assertFixtureFieldsDocumented(filePath, fixtureSchema, value, pathPrefix = '') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return;
    const fields = pathPrefix === ''
        ? (fixtureSchema.fields || [])
        : fixtureFieldsAtPath(fixtureSchema, pathPrefix.split('.').map(part => part.replace(/\[\]$/, '')));
    for (const key of Object.keys(value)) {
        const field = fieldByName(fields, key);
        const pathLabel = pathPrefix ? `${pathPrefix}.${key}` : key;
        assert.ok(field, `${filePath}: undocumented fixture field ${pathLabel}`);
        const childValue = value[key];
        if (Array.isArray(childValue)) {
            for (const item of childValue) {
                assertFixtureFieldsDocumented(filePath, fixtureSchema, item, `${pathLabel}[]`);
            }
        } else if (childValue && typeof childValue === 'object') {
            assertFixtureFieldsDocumented(filePath, fixtureSchema, childValue, pathLabel);
        }
    }
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
    if (tagName === 'quantity_never_increases') {
        return !!(object.tags && object.tags.quantity && object.tags.quantity.never_increases);
    }
    if (tagName === 'quantity_never_decreases') {
        return !!(object.tags && object.tags.quantity && object.tags.quantity.never_decreases);
    }
    return !!((object.tags || {})[tagName]);
}

function buildObjectTagExpectations(report, claimDescriptions) {
    const objectTags = fixtureFieldsAtPath(fixtureSchemaByName(claimDescriptions, 'object_tags'), ['objectTag'])
        .filter(field => field.name !== 'object');
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
    return fieldByName(
        fixtureFieldsAtPath(fixtureSchemaByName(claimDescriptions, 'object_tags'), ['objectTag']),
        tagName
    );
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
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'object_tags'), payload);
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
    return fieldByName(
        fixtureFieldsAtPath(fixtureSchemaByName(claimDescriptions, 'rule_tags'), ['ruleTag', 'tags']),
        tagName
    );
}

function deriveRuleTagValue(rule, tagName) {
    const value = rule.tags ? rule.tags[tagName] : undefined;
    return Array.isArray(value) ? value.slice() : [];
}

function deriveRuleTagBooleanValue(rule, tagName) {
    const value = rule.tags ? rule.tags[tagName] : undefined;
    return typeof value === 'boolean' ? value : false;
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
    const ruleTags = fixtureFieldsAtPath(fixtureSchemaByName(claimDescriptions, 'rule_tags'), ['ruleTag', 'tags']);
    const records = allRuleRecords(report, source);
    assertRuleRecordsIdempotent(report.source.path, records);
    return {
        schema: FIXTURE_SCHEMA,
        ruleTag: records.map(record => ({
            line: record.line,
            text: record.text,
            tags: Object.fromEntries(ruleTags.map(tag => [
                tag.name,
                tag.type === 'boolean'
                    ? deriveRuleTagBooleanValue(record.rule, tag.name)
                    : deriveRuleTagValue(record.rule, tag.name),
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
    if (claim.type === 'boolean') {
        assert.ok(typeof expected === 'boolean', `${filePath}: ${tagName} expected value must be boolean`);
        const actual = deriveRuleTagBooleanValue(record.rule, tagName);
        assert.strictEqual(actual, expected, `${filePath}: ruleTag line ${record.line} ${record.text} ${tagName} expected ${expected}, got ${actual}`);
        return;
    }
    assertStringArray(filePath, tagName, expected);
    const actual = deriveRuleTagValue(record.rule, tagName);
    assertSameStringSet(filePath, `ruleTag line ${record.line} ${record.text} ${tagName}`, expected, actual);
}

function checkRuleFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'rule_tags'), payload);
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

function checkProgramFlowFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'program_flow'), payload);
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

function runProgramFlowDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
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
        checkProgramFlowFixture(txtPath, jsonPath, claimDescriptions);
    }
}

function winflowFactValue(report) {
    const facts = (report.facts && report.facts.winflow) || [];
    if (facts.length === 0) return { rule_ids: [], win_ids: [], wake_edges: [] };
    return facts[0].value;
}

function buildWinflowExpectations(source, report) {
    const ruleRecords = allRuleRecords(report, source);
    assertRuleRecordsIdempotent(report.source.path, ruleRecords);
    const winRecords = allWinConditionRecords(report, source);
    const ruleById = recordById(ruleRecords);
    const winById = new Map(winRecords.map(r => [r.wincondition.id, r]));
    const value = winflowFactValue(report);
    const wakeEdges = value.wake_edges.map(edge => {
        const from = ruleById.get(edge.from);
        const to = winById.get(edge.to);
        assert.ok(from, `winflow edge from rule id ${edge.from} not found in records`);
        assert.ok(to, `winflow edge to win id ${edge.to} not found in records`);
        return {
            from_line: from.line,
            from_text: from.text,
            to_line: to.line,
            to_text: to.text,
            reasons: edge.reasons.slice(),
        };
    });
    wakeEdges.sort(compareEdgeRows);
    return { schema: FIXTURE_SCHEMA, wakeEdges };
}

function validateWinflowExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(Array.isArray(payload.wakeEdges), `${filePath}: wakeEdges must be an array`);
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
}

function checkWinflowFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'winflow'), payload);
    validateWinflowExpectationShape(jsonPath, payload);
    const actual = buildWinflowExpectations(source, report);
    const expectedEdges = payload.wakeEdges.slice().sort(compareEdgeRows);
    assert.deepStrictEqual(actual.wakeEdges, expectedEdges, `${jsonPath}: wakeEdges mismatch`);
}

function runWinflowDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
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
            writeJson(jsonPath, buildWinflowExpectations(source, report));
            log(`generated static analysis testdata: winflow/${stem}.json (review before committing)\n`);
        }
        checkWinflowFixture(txtPath, jsonPath, claimDescriptions);
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
    return fieldByName(
        fixtureFieldsAtPath(fixtureSchemaByName(claimDescriptions, 'wincondition_tags'), ['winConditionTag', 'tags']),
        tagName
    );
}

function deriveWinConditionTagValue(wincondition, tagName) {
    const value = wincondition.tags ? wincondition.tags[tagName] : undefined;
    return Array.isArray(value) ? value.slice() : [];
}

function buildWinConditionTagExpectations(source, report, claimDescriptions) {
    const winConditionTags = fixtureFieldsAtPath(fixtureSchemaByName(claimDescriptions, 'wincondition_tags'), ['winConditionTag', 'tags']);
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
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'wincondition_tags'), payload);
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

// ─── mergeability ────────────────────────────────────────────────────────────

function buildMergeabilityExpectations(report) {
    const facts = (report.facts && report.facts.mergeability) || [];
    const mergePairs = facts.map(fact => ({
        objects: (fact.subjects && fact.subjects.objects ? fact.subjects.objects.slice() : []).sort(),
        status: fact.status,
        blockers: (fact.blockers || []).slice().sort(),
    }));
    mergePairs.sort((a, b) => {
        const cmp = a.objects[0].localeCompare(b.objects[0]);
        return cmp !== 0 ? cmp : (a.objects[1] || '').localeCompare(b.objects[1] || '');
    });
    return { schema: FIXTURE_SCHEMA, mergePairs };
}

function validateMergeabilityExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(Array.isArray(payload.mergePairs), `${filePath}: mergePairs must be an array`);
    for (const [index, item] of payload.mergePairs.entries()) {
        assert.ok(item && typeof item === 'object' && !Array.isArray(item), `${filePath}: mergePairs[${index}] must be an object`);
        assert.ok(Array.isArray(item.objects) && item.objects.length === 2 && item.objects.every(o => typeof o === 'string' && o.length > 0), `${filePath}: mergePairs[${index}].objects must be a 2-element string[]`);
        assert.ok(item.status === 'candidate' || item.status === 'rejected', `${filePath}: mergePairs[${index}].status must be candidate or rejected`);
        assertStringArray(filePath, `mergePairs[${index}].blockers`, item.blockers);
    }
}

function checkMergeabilityFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'mergeability'), payload);
    validateMergeabilityExpectationShape(jsonPath, payload);
    const actual = buildMergeabilityExpectations(report);
    const actualByKey = new Map(actual.mergePairs.map(p => [p.objects.join('\0'), p]));
    for (const expected of payload.mergePairs) {
        const key = expected.objects.slice().sort().join('\0');
        const actualPair = actualByKey.get(key);
        if (!actualPair) {
            const available = Array.from(actualByKey.keys()).map(k => k.replace('\0', '+')).join(', ');
            assert.fail(`${jsonPath}: mergePairs pair ${expected.objects.join('+')} not found; available: ${available}`);
        }
        assert.strictEqual(actualPair.status, expected.status, `${jsonPath}: pair ${expected.objects.join('+')} status expected ${expected.status}, got ${actualPair.status}`);
        assertSameStringSet(jsonPath, `pair ${expected.objects.join('+')} blockers`, expected.blockers, actualPair.blockers);
    }
}

function runMergeabilityDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
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
            writeJson(jsonPath, buildMergeabilityExpectations(report));
            log(`generated static analysis testdata: mergeability/${stem}.json (review before committing)\n`);
        }
        checkMergeabilityFixture(txtPath, jsonPath, claimDescriptions);
    }
}

// ─── rulegroup_flow ───────────────────────────────────────────────────────────

function allGroupRecords(report) {
    const records = [];
    for (const section of (report.ps_tagged && report.ps_tagged.rule_sections) || []) {
        for (const group of section.groups || []) {
            records.push({ group, section });
        }
    }
    return records;
}

function ruleLocator(record) {
    return {
        line: record.line,
        text: record.text,
    };
}

function compareRuleLocators(a, b) {
    if (a.line !== b.line) return a.line - b.line;
    return a.text.localeCompare(b.text);
}

function compareRuleLocatorEdges(a, b) {
    const from = compareRuleLocators(
        { line: a.from_line, text: a.from_text },
        { line: b.from_line, text: b.from_text }
    );
    if (from !== 0) return from;
    return compareRuleLocators(
        { line: a.to_line, text: a.to_text },
        { line: b.to_line, text: b.to_text }
    );
}

function buildRulegroupFlowExpectations(source, report) {
    const facts = (report.facts && report.facts.rulegroup_flow) || [];
    const groupRecords = allGroupRecords(report);
    const groupById = new Map(groupRecords.map(r => [r.group.id, r.group]));
    const ruleRecords = allRuleRecords(report, source);
    const ruleById = recordById(ruleRecords);
    const rulegroupFlow = facts.map(fact => {
        const groupId = (fact.subjects && fact.subjects.groups && fact.subjects.groups[0]) || '';
        const group = groupById.get(groupId);
        assert.ok(group, `rulegroup_flow fact ${fact.id} references unknown group ${groupId}`);
        const value = fact.value || {};
        const interactionEdges = (value.interaction_edges || []).map(edge => {
            const from = ruleById.get(edge.from);
            const to = ruleById.get(edge.to);
            assert.ok(from, `rulegroup_flow edge from rule id ${edge.from} not found in records`);
            assert.ok(to, `rulegroup_flow edge to rule id ${edge.to} not found in records`);
            return {
                from_line: from.line,
                from_text: from.text,
                to_line: to.line,
                to_text: to.text,
                reasons: edge.reasons.slice(),
            };
        }).sort(compareRuleLocatorEdges);
        const rerunMasks = Object.keys(value.rerun_masks || {}).sort().map(ruleId => {
            const from = ruleById.get(ruleId);
            assert.ok(from, `rulegroup_flow rerun mask from rule id ${ruleId} not found in records`);
            return {
                ...ruleLocator(from),
                rerun: (value.rerun_masks[ruleId] || []).map(rerunRuleId => {
                    const to = ruleById.get(rerunRuleId);
                    assert.ok(to, `rulegroup_flow rerun mask to rule id ${rerunRuleId} not found in records`);
                    return ruleLocator(to);
                }).sort(compareRuleLocators),
            };
        });
        return {
            line: group.source_line_min,
            split_candidate: value.split_candidate || false,
            components_count: (value.components || []).length,
            interactionEdges,
            rerunMasks,
            blockers: (fact.blockers || []).slice().sort(),
        };
    });
    rulegroupFlow.sort((a, b) => a.line - b.line);
    return { schema: FIXTURE_SCHEMA, rulegroupFlow };
}

function validateRulegroupFlowExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(Array.isArray(payload.rulegroupFlow), `${filePath}: rulegroupFlow must be an array`);
    for (const [index, item] of payload.rulegroupFlow.entries()) {
        assert.ok(item && typeof item === 'object' && !Array.isArray(item), `${filePath}: rulegroupFlow[${index}] must be an object`);
        assert.ok(Number.isInteger(item.line) && item.line > 0, `${filePath}: rulegroupFlow[${index}] missing positive integer line`);
        assert.ok(typeof item.split_candidate === 'boolean', `${filePath}: rulegroupFlow[${index}].split_candidate must be boolean`);
        assert.ok(Number.isInteger(item.components_count) && item.components_count >= 0, `${filePath}: rulegroupFlow[${index}].components_count must be a non-negative integer`);
        assertStringArray(filePath, `rulegroupFlow[${index}].blockers`, item.blockers);
        if (item.interactionEdges !== undefined) {
            assert.ok(Array.isArray(item.interactionEdges), `${filePath}: rulegroupFlow[${index}].interactionEdges must be an array`);
            for (const [edgeIndex, edge] of item.interactionEdges.entries()) {
                assert.ok(edge && typeof edge === 'object' && !Array.isArray(edge), `${filePath}: rulegroupFlow[${index}].interactionEdges[${edgeIndex}] must be an object`);
                assert.ok(Number.isInteger(edge.from_line) && edge.from_line > 0, `${filePath}: rulegroupFlow[${index}].interactionEdges[${edgeIndex}] missing positive integer from_line`);
                assert.ok(typeof edge.from_text === 'string' && edge.from_text.length > 0, `${filePath}: rulegroupFlow[${index}].interactionEdges[${edgeIndex}] missing from_text`);
                assert.ok(Number.isInteger(edge.to_line) && edge.to_line > 0, `${filePath}: rulegroupFlow[${index}].interactionEdges[${edgeIndex}] missing positive integer to_line`);
                assert.ok(typeof edge.to_text === 'string' && edge.to_text.length > 0, `${filePath}: rulegroupFlow[${index}].interactionEdges[${edgeIndex}] missing to_text`);
                assert.ok(Array.isArray(edge.reasons) && edge.reasons.length > 0, `${filePath}: rulegroupFlow[${index}].interactionEdges[${edgeIndex}].reasons must be a non-empty array`);
                for (const reason of edge.reasons) {
                    assert.ok(REASON_VALUES.includes(reason), `${filePath}: rulegroupFlow[${index}].interactionEdges[${edgeIndex}].reasons contains unknown reason ${JSON.stringify(reason)}`);
                }
            }
        }
        if (item.rerunMasks !== undefined) {
            assert.ok(Array.isArray(item.rerunMasks), `${filePath}: rulegroupFlow[${index}].rerunMasks must be an array`);
            for (const [maskIndex, mask] of item.rerunMasks.entries()) {
                assert.ok(mask && typeof mask === 'object' && !Array.isArray(mask), `${filePath}: rulegroupFlow[${index}].rerunMasks[${maskIndex}] must be an object`);
                assert.ok(Number.isInteger(mask.line) && mask.line > 0, `${filePath}: rulegroupFlow[${index}].rerunMasks[${maskIndex}] missing positive integer line`);
                assert.ok(typeof mask.text === 'string' && mask.text.length > 0, `${filePath}: rulegroupFlow[${index}].rerunMasks[${maskIndex}] missing text`);
                assert.ok(Array.isArray(mask.rerun), `${filePath}: rulegroupFlow[${index}].rerunMasks[${maskIndex}].rerun must be an array`);
                for (const [rerunIndex, rerun] of mask.rerun.entries()) {
                    assert.ok(rerun && typeof rerun === 'object' && !Array.isArray(rerun), `${filePath}: rulegroupFlow[${index}].rerunMasks[${maskIndex}].rerun[${rerunIndex}] must be an object`);
                    assert.ok(Number.isInteger(rerun.line) && rerun.line > 0, `${filePath}: rulegroupFlow[${index}].rerunMasks[${maskIndex}].rerun[${rerunIndex}] missing positive integer line`);
                    assert.ok(typeof rerun.text === 'string' && rerun.text.length > 0, `${filePath}: rulegroupFlow[${index}].rerunMasks[${maskIndex}].rerun[${rerunIndex}] missing text`);
                }
            }
        }
    }
}

function checkRulegroupFlowFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'rulegroup_flow'), payload);
    validateRulegroupFlowExpectationShape(jsonPath, payload);
    const actual = buildRulegroupFlowExpectations(source, report);
    const actualByLine = new Map(actual.rulegroupFlow.map(r => [r.line, r]));
    for (const expected of payload.rulegroupFlow) {
        const actualRow = actualByLine.get(expected.line);
        if (!actualRow) {
            const available = Array.from(actualByLine.keys()).join(', ');
            assert.fail(`${jsonPath}: rulegroupFlow group at line ${expected.line} not found; available lines: ${available}`);
        }
        assert.strictEqual(actualRow.split_candidate, expected.split_candidate, `${jsonPath}: group at line ${expected.line} split_candidate expected ${expected.split_candidate}, got ${actualRow.split_candidate}`);
        assert.strictEqual(actualRow.components_count, expected.components_count, `${jsonPath}: group at line ${expected.line} components_count expected ${expected.components_count}, got ${actualRow.components_count}`);
        if (expected.interactionEdges !== undefined) {
            assert.deepStrictEqual(
                actualRow.interactionEdges,
                expected.interactionEdges.slice().sort(compareRuleLocatorEdges),
                `${jsonPath}: group at line ${expected.line} interactionEdges mismatch`
            );
        }
        if (expected.rerunMasks !== undefined) {
            const actualMasks = actualRow.rerunMasks.map(mask => ({
                ...mask,
                rerun: mask.rerun.slice().sort(compareRuleLocators),
            }));
            const expectedMasks = expected.rerunMasks.map(mask => ({
                ...mask,
                rerun: mask.rerun.slice().sort(compareRuleLocators),
            }));
            assert.deepStrictEqual(
                actualMasks,
                expectedMasks,
                `${jsonPath}: group at line ${expected.line} rerunMasks mismatch`
            );
        }
        assertSameStringSet(jsonPath, `group at line ${expected.line} blockers`, expected.blockers, actualRow.blockers);
    }
}

function runRulegroupFlowDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
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
            writeJson(jsonPath, buildRulegroupFlowExpectations(source, report));
            log(`generated static analysis testdata: rulegroup_flow/${stem}.json (review before committing)\n`);
        }
        checkRulegroupFlowFixture(txtPath, jsonPath, claimDescriptions);
    }
}

// ─── movement_action ──────────────────────────────────────────────────────────

function buildMovementActionExpectations(report) {
    const facts = (report.facts && report.facts.movement_action) || [];
    const noopFact = facts.find(f => f.id === 'action_noop');
    const movementsReachableFromActionInputFact = facts.find(f => f.id === 'movements_reachable_from_action_input');
    return {
        schema: FIXTURE_SCHEMA,
        actionNoop: noopFact ? !!noopFact.value : true,
        actionNoopBlockers: noopFact ? (noopFact.blockers || []).slice().sort() : [],
        movements_reachable_from_action_input: movementsReachableFromActionInputFact
            ? (movementsReachableFromActionInputFact.value || []).slice().sort()
            : [],
    };
}

function validateMovementActionExpectationShape(filePath, payload) {
    assert.strictEqual(payload.schema, FIXTURE_SCHEMA, `${filePath}: unsupported fixture schema`);
    assert.ok(typeof payload.actionNoop === 'boolean', `${filePath}: actionNoop must be boolean`);
    assertStringArray(filePath, 'actionNoopBlockers', payload.actionNoopBlockers);
    if (payload.movements_reachable_from_action_input !== undefined) {
        assertStringArray(
            filePath,
            'movements_reachable_from_action_input',
            payload.movements_reachable_from_action_input
        );
    }
}

function checkMovementActionFixture(txtPath, jsonPath, claimDescriptions) {
    const source = fs.readFileSync(txtPath, 'utf8');
    const report = analyzeSource(source, { sourcePath: txtPath });
    assert.strictEqual(report.status, 'ok', `${txtPath}: static analysis status ${report.status}`);
    const payload = readJson(jsonPath);
    assertFixtureFieldsDocumented(jsonPath, fixtureSchemaByName(claimDescriptions, 'movement_action'), payload);
    validateMovementActionExpectationShape(jsonPath, payload);
    const actual = buildMovementActionExpectations(report);
    assert.strictEqual(actual.actionNoop, payload.actionNoop, `${jsonPath}: actionNoop expected ${payload.actionNoop}, got ${actual.actionNoop}`);
    assertSameStringSet(jsonPath, 'actionNoopBlockers', payload.actionNoopBlockers, actual.actionNoopBlockers);
    if (payload.movements_reachable_from_action_input !== undefined) {
        assertSameStringSet(
            jsonPath,
            'movements_reachable_from_action_input',
            payload.movements_reachable_from_action_input,
            actual.movements_reachable_from_action_input
        );
    }
}

function runMovementActionDir(dirPath, claimDescriptions, log = process.stdout.write.bind(process.stdout)) {
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
            writeJson(jsonPath, buildMovementActionExpectations(report));
            log(`generated static analysis testdata: movement_action/${stem}.json (review before committing)\n`);
        }
        checkMovementActionFixture(txtPath, jsonPath, claimDescriptions);
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
    runProgramFlowDir(programFlowDir, claimDescriptions, options.log);
    const winflowDir = path.join(root, 'winflow');
    assert.ok(fs.existsSync(winflowDir), `${winflowDir}: missing winflow testdata directory`);
    runWinflowDir(winflowDir, claimDescriptions, options.log);
    const winConditionTagsDir = path.join(root, 'wincondition_tags');
    assert.ok(fs.existsSync(winConditionTagsDir), `${winConditionTagsDir}: missing wincondition_tags testdata directory`);
    runWinConditionTagsDir(winConditionTagsDir, claimDescriptions, options.log);
    const mergeabilityDir = path.join(root, 'mergeability');
    assert.ok(fs.existsSync(mergeabilityDir), `${mergeabilityDir}: missing mergeability testdata directory`);
    runMergeabilityDir(mergeabilityDir, claimDescriptions, options.log);
    const rulegroupFlowDir = path.join(root, 'rulegroup_flow');
    assert.ok(fs.existsSync(rulegroupFlowDir), `${rulegroupFlowDir}: missing rulegroup_flow testdata directory`);
    runRulegroupFlowDir(rulegroupFlowDir, claimDescriptions, options.log);
    const movementActionDir = path.join(root, 'movement_action');
    assert.ok(fs.existsSync(movementActionDir), `${movementActionDir}: missing movement_action testdata directory`);
    runMovementActionDir(movementActionDir, claimDescriptions, options.log);
    process.stdout.write('static_analysis_testdata_runner: ok\n');
}

if (require.main === module) {
    runStaticAnalysisTestdata();
}

module.exports = {
    assertFixtureFieldsDocumented,
    buildMergeabilityExpectations,
    buildMovementActionExpectations,
    buildObjectTagExpectations,
    buildProgramFlowExpectations,
    buildRuleTagExpectations,
    buildRulegroupFlowExpectations,
    buildWinConditionTagExpectations,
    buildWinflowExpectations,
    deriveObjectTagValue,
    deriveRuleTagValue,
    deriveWinConditionTagValue,
    fixtureFieldsAtPath,
    fixtureSchemaByName,
    findRuleRecord,
    findWinConditionRecord,
    formatFixtureJson,
    loadClaimDescriptions,
    runMergeabilityDir,
    runMovementActionDir,
    runObjectTagsDir,
    runProgramFlowDir,
    runRuleTagsDir,
    runRulegroupFlowDir,
    runStaticAnalysisTestdata,
    runWinConditionTagsDir,
    runWinflowDir,
};
