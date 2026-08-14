#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const garden = require('./garden');

const tests = [];

function test(name, fn) {
    tests.push({ name, fn });
}

const SAMPLE = `title Garden Sample
flickscreen 5x5

========
OBJECTS
========

Background
black

Player
white

Wall
gray

=======
LEGEND
=======

. = Background
P = Player
Obstacle = Player or Wall
Together = Player and Background

=========
SOUNDS
=========

================
COLLISIONLAYERS
================

Background
Player, Wall

======
RULES
======

[ > Player | Wall ] -> [ > Player | > Wall ] again

==============
WINCONDITIONS
==============

=======
LEVELS
=======

PP
..
`;

test('seeded random streams are repeatable and bounded', function() {
    const first = new garden.Random(123456);
    const second = new garden.Random(123456);
    const values = [];
    for (let i = 0; i < 20; i++) {
        values.push(first.integer(7));
    }
    assert.deepStrictEqual(values, values.map(function() { return second.integer(7); }));
    assert(values.every(function(value) { return value >= 0 && value < 7; }));
    assert.throws(function() { first.integer(0); }, /positive/);
});

test('the existing simulation and compiler-message fixtures form one corpus', function() {
    const resourceDir = path.join(__dirname, '..', 'resources');
    const corpus = garden.loadCorpus(resourceDir);
    const simulation = corpus.filter(function(item) { return item.kind === 'simulation'; });
    const compiler = corpus.filter(function(item) { return item.kind === 'compiler-message'; });
    assert(simulation.length > 0);
    assert(compiler.length > 0);
    assert.strictEqual(simulation.length + compiler.length, corpus.length);
    assert.strictEqual(typeof corpus[0].source, 'string');
    assert(Array.isArray(corpus[0].inputs));
    assert.strictEqual(corpus[0].fixtureIndex, 0);
    assert.strictEqual(corpus[0].kind, 'simulation');
    assert.strictEqual(corpus[simulation.length].kind, 'compiler-message');
    assert.strictEqual(corpus[simulation.length].fixtureIndex, 0);
    assert.deepStrictEqual(corpus[simulation.length].inputs, []);
});

test('corpusIndex is unique even when names and kind-local indexes collide', function() {
    const corpus = garden.loadCorpus(path.join(__dirname, '..', 'resources'));
    const indexes = corpus.map(function(item) { return item.corpusIndex; });
    assert.strictEqual(indexes.length, new Set(indexes).size);
    corpus.forEach(function(item, i) {
        assert.strictEqual(item.corpusIndex, i);
        assert.strictEqual(typeof item.kind, 'string');
        assert.strictEqual(typeof item.fixtureIndex, 'number');
    });
    const icy = corpus.filter(function(item) { return item.name === 'icycrates'; });
    if (icy.length >= 2) {
        assert.notStrictEqual(icy[0].kind, icy[1].kind);
        assert.notStrictEqual(icy[0].corpusIndex, icy[1].corpusIndex);
    }
});

test('every named mutator either changes a suitable source or reports inapplicable', function() {
    const expected = [
        'delete-rule-punctuation',
        'duplicate-rule-punctuation',
        'swap-legend-operator',
        'invalid-viewport',
        'duplicate-rule-command',
        'legend-cycle',
        'swap-sections',
        'odd-whitespace',
        'unterminated-comment',
        'duplicate-rule-line',
        'swap-object-colors',
        'nudge-level-cell',
        'flip-win-quantifier'
    ];
    assert.deepStrictEqual(garden.mutators.map(function(mutator) { return mutator.name; }), expected);

    const winSource = SAMPLE.replace(
        '==============\nWINCONDITIONS\n==============\n\n',
        '==============\nWINCONDITIONS\n==============\n\nall crate on target\n\n'
    );

    for (let i = 0; i < garden.mutators.length; i++) {
        const mutator = garden.mutators[i];
        let source = SAMPLE;
        let result = mutator.apply(source, new garden.Random(100 + i));
        if (!result) {
            source = winSource;
            result = mutator.apply(source, new garden.Random(100 + i));
        }
        assert(result, mutator.name + ' should apply to a suitable source');
        assert.notStrictEqual(result.source, source, mutator.name + ' should change the source');
        assert.strictEqual(typeof result.detail, 'string');
        assert(result.detail.length > 0);
    }
});

test('mutating a fixture records enough information to reproduce it', function() {
    const fixture = {
        name: 'sample',
        fixtureIndex: 7,
        kind: 'simulation',
        source: SAMPLE,
        inputs: [0, 3],
        level: 0,
        randomSeed: null
    };
    const first = garden.mutateFixture(fixture, new garden.Random(44), ['legend-cycle']);
    const second = garden.mutateFixture(fixture, new garden.Random(44), ['legend-cycle']);
    assert.deepStrictEqual(first, second);
    assert.strictEqual(first.mutator, 'legend-cycle');
    assert.strictEqual(first.fixtureName, 'sample');
    assert.strictEqual(first.fixtureIndex, 7);
    assert.notStrictEqual(first.source, fixture.source);
});

test('mutateFixture retries then fails when no mutator applies', function() {
    const fixture = {
        name: 'empty',
        fixtureIndex: 0,
        kind: 'simulation',
        source: 'title X\n',
        inputs: [],
        level: 0,
        randomSeed: null
    };
    assert.throws(function() {
        garden.mutateFixture(fixture, new garden.Random(1), ['delete-rule-punctuation'], { maxAttempts: 2 });
    }, /inapplicable/);
});

test('arguments have reproducible defaults and reject unsafe numeric values', function() {
    const defaults = garden.parseArguments([], { now: function() { return 98765; } });
    assert.strictEqual(defaults.seed, 98765);
    assert.strictEqual(defaults.count, 100);
    assert.strictEqual(defaults.timeoutMs, 2000);
    assert.strictEqual(defaults.shrink, true);
    assert.strictEqual(defaults.replay, true);
    assert.strictEqual(defaults.maxInputs, 8);
    assert.strictEqual(defaults.shrinkBudget, 200);
    assert.strictEqual(defaults.maxAttempts, 8);
    assert.strictEqual(defaults.output, '.build/monster_garden');
    assert.strictEqual(defaults.fixture, null);
    assert.strictEqual(defaults.mutators, null);
    assert.strictEqual(defaults.listMutators, false);

    const parsed = garden.parseArguments([
        '--seed', '42', '--count', '3', '--timeout-ms', '900',
        '--fixture', 'sokoban', '--mutator', 'legend-cycle,odd-whitespace',
        '--output', 'somewhere', '--no-shrink', '--no-replay', '--max-inputs', '4',
        '--shrink-budget', '50', '--max-attempts', '3'
    ]);
    assert.strictEqual(parsed.seed, 42);
    assert.strictEqual(parsed.count, 3);
    assert.strictEqual(parsed.timeoutMs, 900);
    assert.deepStrictEqual(parsed.mutators, ['legend-cycle', 'odd-whitespace']);
    assert.strictEqual(parsed.shrink, false);
    assert.strictEqual(parsed.replay, false);
    assert.strictEqual(parsed.maxInputs, 4);
    assert.strictEqual(parsed.shrinkBudget, 50);
    assert.strictEqual(parsed.maxAttempts, 3);
    assert.strictEqual(parsed.listMutators, false);
    assert.strictEqual(garden.parseArguments(['--list-mutators']).listMutators, true);
    assert.throws(function() { garden.parseArguments(['--count', '0']); }, /count/);
    assert.throws(function() { garden.parseArguments(['--timeout-ms', '0']); }, /timeout-ms/);
    assert.throws(function() { garden.parseArguments(['--timeout-ms', '2147483648']); }, /timeout-ms/);
    assert.strictEqual(garden.parseArguments(['--timeout-ms', '2147483647']).timeoutMs, 2147483647);
    assert.throws(function() { garden.parseArguments(['--wat']); }, /Unknown option/);
    assert.throws(function() { garden.parseArguments(['--mutator', 'imaginary']); }, /Unknown mutator/);
    assert.throws(function() { garden.parseArguments(['--mutator', '']); }, /mutator/);
    assert.throws(function() { garden.parseArguments(['--seed', '']); }, /seed/);
    assert.throws(function() { garden.parseArguments(['--seed', '4294967296']); }, /seed/);
    assert.strictEqual(garden.parseArguments(['--seed', '4294967295']).seed, 4294967295);
});

test('extra inputs are generated deterministically and appended after the truncated prefix', function() {
    assert.strictEqual(garden.parseArguments([]).extraInputs, 0);
    assert.throws(function() { garden.parseArguments(['--extra-inputs', '0']); }, /extra-inputs/);
    const rng = new garden.Random(1);
    const recorded = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    const first = garden.extendInputs(recorded, rng, { maxInputs: 8, extraInputs: 3 });
    const rng2 = new garden.Random(1);
    const second = garden.extendInputs(recorded, rng2, { maxInputs: 8, extraInputs: 3 });
    assert.deepStrictEqual(first, second);
    assert.strictEqual(first.length, 11);
    assert.deepStrictEqual(first.slice(0, 8), recorded.slice(0, 8));
    first.slice(8).forEach(function(value) {
        assert.notStrictEqual([0, 1, 2, 3, 4, 'tick'].indexOf(value), -1);
    });
});

test('default extraInputs keeps the unsliced recorded tape and campaign maxInputs', function() {
    const recorded = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    const rng = new garden.Random(1);
    const unsliced = garden.prepareTrialInputs(recorded, rng, { maxInputs: 8, extraInputs: 0 });
    assert.strictEqual(unsliced.length, recorded.length);
    assert.deepStrictEqual(unsliced, recorded);

    const omitted = garden.prepareTrialInputs(recorded, new garden.Random(1), { maxInputs: 8 });
    assert.strictEqual(omitted.length, recorded.length);
    assert.deepStrictEqual(omitted, recorded);

    const extras = garden.prepareTrialInputs(recorded, new garden.Random(1), { maxInputs: 8, extraInputs: 3 });
    const expectedExtras = garden.extendInputs(recorded, new garden.Random(1), { maxInputs: 8, extraInputs: 3 });
    assert.deepStrictEqual(extras, expectedExtras);
    assert.strictEqual(extras.length, 11);

    const options = { maxInputs: 8, extraInputs: 0 };
    assert.strictEqual(garden.trialMaxInputs(options, unsliced), 8);
    assert.strictEqual(garden.trialMaxInputs({ maxInputs: 8 }, unsliced), 8);
    assert.strictEqual(garden.trialMaxInputs({ maxInputs: 8, extraInputs: 3 }, extras), extras.length);
});

test('only inapplicable mutation errors are skippable', function() {
    assert.strictEqual(garden.isInapplicableMutation(new Error('inapplicable mutation after 2 attempts')), true);
    assert.strictEqual(garden.isInapplicableMutation(new TypeError('mutator exploded')), false);
    assert.strictEqual(garden.isInapplicableMutation(new Error('Cannot read property apply of undefined')), false);
});

test('failure signatures distinguish timeouts from parser crashes and include fingerprints', function() {
    const crash = garden.failureSignature({
        kind: 'crash',
        error: { name: 'TypeError', message: 'bad thing\nwith stack noise' },
        fingerprint: 'ignored',
        detail: ''
    });
    assert.strictEqual(crash, garden.failureSignature({
        kind: 'crash',
        error: { name: 'TypeError', message: 'bad thing\nelsewhere' },
        fingerprint: 'other'
    }));
    assert.notStrictEqual(crash, garden.failureSignature({ kind: 'timeout' }));
    const first = garden.failureSignature({
        kind: 'replay-divergence',
        fingerprint: 'alpha-board',
        detail: 'beta'
    });
    const second = garden.failureSignature({
        kind: 'replay-divergence',
        fingerprint: 'gamma-board',
        detail: 'beta'
    });
    assert.notStrictEqual(first, second);
    assert.strictEqual(garden.failureSignature({ kind: 'timeout', fingerprint: 'x' }), 'timeout');
});

test('timeouts are not shrunk and a signature change after shrink is reverted', async function() {
    const original = 'keep\nlater-crash\n';
    const mutant = { source: original, inputs: [], level: 0, randomSeed: null };
    const ok = { kind: 'ok', error: null, fingerprint: 'ok', detail: '', errorCount: 0 };
    const crash = {
        kind: 'crash',
        error: { name: 'TypeError', message: 'keep' },
        fingerprint: '',
        detail: '',
        errorCount: 0
    };

    let timeoutCalls = 0;
    const timeoutResult = { kind: 'timeout', error: null, fingerprint: '', detail: 'timeout', errorCount: 0 };
    const shrunkTimeout = await garden.shrinkInteresting(mutant, timeoutResult, {
        shrink: true,
        shrinkBudget: 20,
        evaluate: async function() {
            timeoutCalls++;
            return { kind: 'compiler-error', error: null, fingerprint: 'compiler-error:1', detail: '', errorCount: 1 };
        }
    });
    assert.strictEqual(shrunkTimeout.source, original);
    assert.strictEqual(shrunkTimeout.result.kind, 'timeout');
    assert.strictEqual(timeoutCalls, 0);

    const shrinkCandidates = [];
    const shrunkCrash = await garden.shrinkInteresting(mutant, crash, {
        shrink: true,
        shrinkBudget: 20,
        evaluate: async function(partial) {
            shrinkCandidates.push(partial.source);
            if (partial.source.indexOf('keep') >= 0) {
                return crash;
            }
            return ok;
        }
    });
    assert.strictEqual(shrunkCrash.source, 'keep\n');
    assert.strictEqual(shrunkCrash.result.kind, 'crash');
    assert(shrinkCandidates.length > 0);
    shrinkCandidates.forEach(function(source) {
        assert(source.endsWith('\n'), JSON.stringify(source));
    });

    const seen = Object.create(null);
    const reverted = await garden.shrinkInteresting(mutant, crash, {
        shrink: true,
        shrinkBudget: 20,
        evaluate: async function(partial) {
            const source = partial.source;
            seen[source] = (seen[source] || 0) + 1;
            if (source.indexOf('later-crash') < 0 && seen[source] > 1) {
                return ok;
            }
            if (source.indexOf('keep') >= 0) {
                return crash;
            }
            return ok;
        }
    });
    assert.strictEqual(reverted.source, original);
    assert.strictEqual(reverted.result, crash);
});

test('level invariants accept a well-formed level and name the first broken field', function() {
    const good = {
        width: 2,
        height: 3,
        n_tiles: 6,
        objects: new Int32Array(12),
        movements: new Int32Array(6),
        commandQueue: [],
        rowCellContents: [0, 0, 0],
        colCellContents: [0, 0]
    };
    assert.strictEqual(garden.checkLevelInvariants(good, 2, 1), null);
    assert(/stride/.test(garden.checkLevelInvariants(good, 0, 1)));
    assert(/missing/.test(garden.checkLevelInvariants(null, 2, 1)));
    assert(/dimensions/.test(garden.checkLevelInvariants({
        width: 0, height: 3, n_tiles: 0, objects: new Int32Array(0), movements: new Int32Array(0)
    }, 2, 1)));
    assert(/n_tiles/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 5, objects: new Int32Array(10), movements: new Int32Array(5)
    }, 2, 1)));
    assert(/objects/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 6, objects: new Int32Array(5), movements: new Int32Array(6)
    }, 2, 1)));
    assert(/movements/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 6, objects: new Int32Array(12)
    }, 2, 1)));
    assert(/commandQueue/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 6,
        objects: new Int32Array(12), movements: new Int32Array(6),
        commandQueue: ['win']
    }, 2, 1)));
});

test('level occupancy counts a sign-bit object as one occupant', function() {
    const signBit = 1 << 31;
    const oneTile = {
        width: 1,
        height: 1,
        n_tiles: 1,
        objects: new Int32Array([signBit]),
        movements: new Int32Array(1)
    };
    const signMask = { layerMasks: [{ data: new Int32Array([signBit]) }] };
    assert.strictEqual(garden.checkLevelInvariants(oneTile, 1, 1, signMask), null);
    const twoBits = {
        width: 1,
        height: 1,
        n_tiles: 1,
        objects: new Int32Array([signBit | 1]),
        movements: new Int32Array(1)
    };
    const bothBits = { layerMasks: [{ data: new Int32Array([-1]) }] };
    assert(/collision layer/.test(garden.checkLevelInvariants(twoBits, 1, 1, bothBits)));
});

test('only crashes, timeouts, invariants, nondeterminism, and replay divergence are interesting', function() {
    assert.strictEqual(garden.isInteresting({ kind: 'ok' }), false);
    assert.strictEqual(garden.isInteresting({ kind: 'compiler-error' }), false);
    assert.strictEqual(garden.isInteresting({ kind: 'crash' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'timeout' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'invariant' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'nondeterministic' }), true);
    assert.strictEqual(garden.isInteresting({ kind: 'replay-divergence' }), true);
});

test('semantic-mismatch is interesting and baseline oracle fields are gated', function() {
    assert.strictEqual(garden.isInteresting({ kind: 'semantic-mismatch' }), true);

    const simulation = {
        kind: 'simulation',
        expectedOutput: 'board',
        expectedErrors: null,
        expectedErrorCount: null,
        inputs: [0, 1, 2]
    };
    assert.deepStrictEqual(
        garden.baselineOracleFields(simulation, { extraInputs: 0, maxInputs: 8 }),
        { expectedOutput: 'board' }
    );
    assert.deepStrictEqual(
        garden.baselineOracleFields(simulation, { extraInputs: 0, maxInputs: 3 }),
        { expectedOutput: 'board' }
    );
    assert.deepStrictEqual(
        garden.baselineOracleFields(simulation, { extraInputs: 0, maxInputs: 2 }),
        {}
    );
    assert.deepStrictEqual(
        garden.baselineOracleFields(simulation, { extraInputs: 3, maxInputs: 8 }),
        {}
    );
    assert.deepStrictEqual(
        garden.baselineOracleFields({
            kind: 'simulation',
            expectedOutput: null,
            inputs: [0]
        }, { extraInputs: 0, maxInputs: 8 }),
        {}
    );

    const compiler = {
        kind: 'compiler-message',
        expectedErrors: ['msg'],
        expectedErrorCount: 1,
        expectedOutput: null,
        inputs: []
    };
    assert.deepStrictEqual(
        garden.baselineOracleFields(compiler, { extraInputs: 3, maxInputs: 8 }),
        { expectedErrors: ['msg'], expectedErrorCount: 1 }
    );
    assert.deepStrictEqual(
        garden.baselineOracleFields(compiler, { extraInputs: 0, maxInputs: 8 }),
        { expectedErrors: ['msg'], expectedErrorCount: 1 }
    );
});

test('only causal mutants are attributed when the unmutated fixture already fails', function() {
    const baseline = {
        kind: 'replay-divergence',
        error: null,
        fingerprint: 'same',
        detail: 'x',
        errorCount: 0
    };
    const same = garden.attributeMonster(baseline, baseline);
    assert.strictEqual(same.save, false);
    assert.strictEqual(same.tally, 'baseline');
    const different = garden.attributeMonster(baseline, {
        kind: 'crash',
        error: { name: 'TypeError', message: 'boom' },
        fingerprint: '',
        detail: '',
        errorCount: 0
    });
    assert.strictEqual(different.save, true);
    assert.strictEqual(different.tally, 'crash');
    const healthy = garden.attributeMonster({
        kind: 'ok', error: null, fingerprint: 'f', detail: '', errorCount: 0
    }, baseline);
    assert.strictEqual(healthy.save, true);
    assert.strictEqual(healthy.tally, 'replay-divergence');
    assert.strictEqual(garden.attributeMonster({
        kind: 'compiler-error', error: null, fingerprint: 'compiler-error:1', detail: '', errorCount: 1
    }, { kind: 'ok', error: null, fingerprint: 'f', detail: '', errorCount: 0 }).save, false);
    assert.strictEqual(garden.attributeMonster({
        kind: 'compiler-warning', error: null, fingerprint: 'compiler-warning:1', detail: '', errorCount: 1
    }, { kind: 'ok', error: null, fingerprint: 'f', detail: '', errorCount: 0 }).save, false);
    const oracleMiss = garden.attributeMonster({
        kind: 'semantic-mismatch',
        error: null,
        fingerprint: 'diag',
        detail: 'expectedOutput',
        errorCount: 0
    }, {
        kind: 'ok',
        error: null,
        fingerprint: 'play',
        detail: '',
        errorCount: 0
    });
    assert.strictEqual(oracleMiss.save, false);
    assert.strictEqual(oracleMiss.tally, 'baseline');
    assert.strictEqual(garden.isHealthyKind('ok'), true);
    assert.strictEqual(garden.isHealthyKind('compiler-error'), true);
    assert.strictEqual(garden.isHealthyKind('compiler-warning'), true);
    assert.strictEqual(garden.isHealthyKind('crash'), false);
    assert.strictEqual(garden.isHealthyKind('replay-divergence'), false);
});

test('line shrinking keeps a deletion only when the signature stays the same', function() {
    const source = 'keep\nnoise\nkeep\n';
    const result = garden.shrinkSource(source, function(candidate) {
        return candidate.split('\n').filter(function(line) { return line === 'keep'; }).length === 2
            && candidate.indexOf('noise') < 0
            && candidate.endsWith('\n');
    }, 20);
    assert.strictEqual(result.source, 'keep\nkeep\n');
    assert(result.steps > 0);
    assert(result.steps <= 20);
});

test('artifact names and regression fixtures are copy-pasteable and path-safe', function() {
    const name = garden.artifactDirName('crash:TypeError:bad thing / \\ : *', 99, 3);
    assert.strictEqual(name, 'crash-TypeError-bad-thing-s99_0003');
    assert(!/[\/\\:*]/.test(name));
    const snippet = garden.formatRegression('monster garden 99 3', 'title "X"\nline\n', {
        inputs: [2, 3],
        level: 4,
        randomSeed: '1397263843369.0808'
    });
    assert.strictEqual(
        snippet,
        '[\n    "monster garden 99 3",\n    ["title \\"X\\"\\nline\\n", [2,3], "", 4, "1397263843369.0808"]\n],\n'
    );
});

test('writeArtifacts uses a unique temp directory and leaves dest intact until rename', function() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'monster-garden-art-'));
    const destName = 'crash-demo-s1_0001';
    const dest = path.join(root, destName);
    fs.mkdirSync(dest);
    fs.writeFileSync(path.join(dest, 'report.json'), '{"old":true}\n');
    const origRm = fs.rmSync;
    const origMkdtemp = fs.mkdtempSync;
    const removed = [];
    const prefixes = [];
    fs.rmSync = function(target, opts) {
        removed.push(target);
        return origRm.call(fs, target, opts);
    };
    fs.mkdtempSync = function(prefix, opts) {
        prefixes.push(prefix);
        return origMkdtemp.call(fs, prefix, opts);
    };
    let written;
    try {
        written = garden.writeArtifacts(root, destName, {
            'report.json': '{"new":true}\n',
            'original.txt': 'src\n'
        });
    } finally {
        fs.rmSync = origRm;
        fs.mkdtempSync = origMkdtemp;
    }
    assert.strictEqual(written, dest);
    assert.strictEqual(fs.readFileSync(path.join(dest, 'report.json'), 'utf8'), '{"new":true}\n');
    assert.strictEqual(fs.readFileSync(path.join(dest, 'original.txt'), 'utf8'), 'src\n');
    removed.forEach(function(target) {
        assert.notStrictEqual(path.resolve(target), path.resolve(dest));
    });
    assert(prefixes.length > 0);
    prefixes.forEach(function(prefix) {
        assert.strictEqual(prefix.indexOf(path.join(root, '.' + destName + '-')), 0);
    });
    const names = fs.readdirSync(root);
    assert.ok(names.indexOf(destName + '.tmp') < 0);
    names.forEach(function(name) {
        assert.ok(name.indexOf('.' + destName + '-') !== 0, name);
    });
});

test('writeArtifacts cleans unique temp dirs if writing fails', function() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'monster-garden-art-fail-'));
    const destName = 'crash-demo-s1_0002';
    const origWrite = fs.writeFileSync;
    fs.writeFileSync = function() {
        throw new Error('disk full');
    };
    try {
        assert.throws(function() {
            garden.writeArtifacts(root, destName, { 'report.json': '{"new":true}\n' });
        }, /disk full/);
    } finally {
        fs.writeFileSync = origWrite;
    }
    const leftovers = fs.readdirSync(root).filter(function(name) {
        return name.indexOf('.' + destName + '-') === 0;
    });
    assert.deepStrictEqual(leftovers, []);
});

test('writeArtifacts restores dest if the replacement rename fails', function() {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'monster-garden-art-restore-'));
    const destName = 'crash-demo-s1_0003';
    const dest = path.join(root, destName);
    fs.mkdirSync(dest);
    fs.writeFileSync(path.join(dest, 'report.json'), '{"old":true}\n');
    const origRename = fs.renameSync;
    fs.renameSync = function(from, to) {
        const fromNames = fs.existsSync(from) ? fs.readdirSync(from) : [];
        if (path.resolve(to) === path.resolve(dest)
            && path.resolve(from) !== path.resolve(dest)
            && fromNames.indexOf('original.txt') >= 0
            && !fs.existsSync(dest)) {
            const err = new Error('replacement rename failed');
            err.code = 'EXDEV';
            throw err;
        }
        return origRename.call(fs, from, to);
    };
    try {
        assert.throws(function() {
            garden.writeArtifacts(root, destName, {
                'report.json': '{"new":true}\n',
                'original.txt': 'src\n'
            });
        }, /replacement rename failed/);
    } finally {
        fs.renameSync = origRename;
    }
    assert.strictEqual(fs.readFileSync(path.join(dest, 'report.json'), 'utf8'), '{"old":true}\n');
    const leftovers = fs.readdirSync(root).filter(function(name) {
        return name.indexOf('.' + destName + '-') === 0;
    });
    assert.deepStrictEqual(leftovers, []);
});

function runWorkerSync(job, timeoutMs) {
    return spawnSync(process.execPath, [path.join(__dirname, 'worker.js')], {
        input: JSON.stringify(job),
        encoding: 'utf8',
        timeout: timeoutMs || 20000
    });
}

function workerResult(job) {
    const child = runWorkerSync(job);
    assert.strictEqual(child.error, undefined, child.stderr);
    const line = child.stdout.trim().split('\n').pop();
    return JSON.parse(line);
}

test('the worker compiles a valid sample and returns a stable ok fingerprint', function() {
    const job = {
        source: SAMPLE,
        inputs: [0, 3],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8
    };
    const first = workerResult(job);
    const second = workerResult(job);
    assert.strictEqual(first.kind, 'ok', JSON.stringify(first));
    assert.strictEqual(first.error, null);
    assert.strictEqual(typeof first.fingerprint, 'string');
    const parsed = JSON.parse(first.fingerprint);
    assert.strictEqual(parsed.errorCount, 0);
    assert.strictEqual(typeof parsed.board, 'string');
    assert(Array.isArray(parsed.objects));
    assert.strictEqual(first.errorCount, 0);
    assert.deepStrictEqual(first, second);
});

test('structure-aware mutators usually keep a compiling sample compiling', function() {
    const names = ['duplicate-rule-line', 'swap-object-colors', 'nudge-level-cell', 'flip-win-quantifier'];
    names.forEach(function(name) {
        const mutator = garden.mutators.find(function(item) { return item.name === name; });
        assert(mutator, name);
        const rng = new garden.Random(7);
        let source = SAMPLE;
        let applied = mutator.apply(source, rng);
        if (!applied) {
            source = SAMPLE.replace(
                '==============\nWINCONDITIONS\n==============\n\n',
                '==============\nWINCONDITIONS\n==============\n\nall crate on target\n\n'
            );
            applied = mutator.apply(source, new garden.Random(7));
        }
        assert(applied);
        assert.notStrictEqual(applied.source, source);
        const result = workerResult({
            source: applied.source,
            inputs: [0],
            level: 0,
            randomSeed: 'garden-seed',
            replay: false,
            maxInputs: 8
        });
        assert.notStrictEqual(result.kind, 'crash', name + ' ' + JSON.stringify(result));
    });
});

test('a compiled game with fewer levels than job.level is ok, not a crash', function() {
    const result = workerResult({
        source: SAMPLE,
        inputs: [],
        level: 13,
        randomSeed: null,
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
    assert.strictEqual(result.error, null);
});

test('ok fingerprints include rng and mode, not only the board string', function() {
    const result = workerResult({
        source: SAMPLE,
        inputs: [0],
        level: 0,
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
    const parsed = JSON.parse(result.fingerprint);
    assert.strictEqual(parsed.errorCount, 0);
    assert.strictEqual(typeof parsed.board, 'string');
    assert(Array.isArray(parsed.objects));
    assert(parsed.rng && typeof parsed.rng.i === 'number');
    assert.strictEqual(typeof parsed.titleScreen, 'boolean');
});

test('invariants are checked after each input', function() {
    const result = workerResult({
        source: SAMPLE,
        inputs: [0, 3],
        level: 0,
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
});

test('play fingerprints stay JSON after a compile-success runtime error', function() {
    const result = workerResult({
        source: `title Loop Fingerprint
========
OBJECTS
========

Background
black

Player
white

Crate
gray

=======
LEGEND
=======

. = Background
P = Player

=========
SOUNDS
=========

================
COLLISIONLAYERS
================

Background
Player, Crate

======
RULES
======

[ Player ] -> [ Crate ]
+ [ Crate ] -> [ Player ]

==============
WINCONDITIONS
==============

=======
LEVELS
=======

P
`,
        inputs: [0],
        level: 0,
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
    assert.notStrictEqual(result.fingerprint.indexOf('compiler-error:'), 0, result.fingerprint.slice(0, 80));
    const parsed = JSON.parse(result.fingerprint);
    assert.strictEqual(typeof parsed.board, 'string');
    assert(parsed.errorCount > 0, JSON.stringify(parsed));
});

test('replay after winning onto a differently sized level does not crash', function() {
    const result = workerResult({
        source: `title Size Change
========
OBJECTS
========

Background
black

Player
white

=======
LEGEND
=======

. = Background
P = Player

=========
SOUNDS
=========

================
COLLISIONLAYERS
================

Background
Player

======
RULES
======

[ Player ] -> [ Player ] win

==============
WINCONDITIONS
==============

=======
LEVELS
=======

P...

P.
`,
        inputs: [0],
        level: 0,
        randomSeed: 'garden-seed',
        replay: true,
        maxInputs: 8
    });
    assert.notStrictEqual(result.kind, 'crash', JSON.stringify(result));
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
    const parsed = JSON.parse(result.fingerprint);
    assert.strictEqual(typeof parsed.board, 'string');
});

test('winning onto the next level then replaying stays ok with a canonical seed', function() {
    const result = workerResult({
        source: `title Win Then Replay
========
OBJECTS
========

Background
black

Player
white

=======
LEGEND
=======

. = Background
P = Player

=========
SOUNDS
=========

================
COLLISIONLAYERS
================

Background
Player

======
RULES
======

==============
WINCONDITIONS
==============

some Player

=======
LEVELS
=======

P

P.
`,
        inputs: [0],
        level: 0,
        randomSeed: 'garden-seed',
        replay: true,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
    const parsed = JSON.parse(result.fingerprint);
    assert.strictEqual(parsed.curlevel, 1);
    assert.strictEqual(parsed.rng.seed, 'garden-seed');
});

test('the worker treats compile diagnostics as compiler-error, not a crash', function() {
    const result = workerResult({
        source: `title No Background
=======
OBJECTS
=======

Player
white

=======
LEGEND
=======
P = Player
`,
        inputs: [0],
        level: 0,
        randomSeed: null,
        replay: true,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'compiler-error', JSON.stringify(result));
    assert.strictEqual(result.error, null);
    assert(result.errorCount > 0);
    assert(result.fingerprint.indexOf('compiler-error:' + result.errorCount + ':') === 0);
});

test('the worker echoes a canonical engineSeed and rejects a non-integer level', function() {
    const withSeed = workerResult({
        source: SAMPLE,
        inputs: [],
        level: 0,
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(withSeed.kind, 'ok', JSON.stringify(withSeed));
    assert.strictEqual(withSeed.engineSeed, 'garden-seed');
    const invented = workerResult({
        source: SAMPLE,
        inputs: [],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(invented.kind, 'ok', JSON.stringify(invented));
    assert.strictEqual(typeof invented.engineSeed, 'string');
    assert(invented.engineSeed.length > 0);
    const badLevel = workerResult({
        source: SAMPLE,
        inputs: [],
        level: '0',
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(badLevel.kind, 'crash', JSON.stringify(badLevel));
    assert.strictEqual(badLevel.engineSeed, 'garden-seed');
});

test('expectedOutput mismatches are semantic-mismatch', function() {
    const ok = workerResult({
        source: SAMPLE,
        inputs: [0],
        level: 0,
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(ok.kind, 'ok', JSON.stringify(ok));
    const parsed = JSON.parse(ok.fingerprint);
    const mismatch = workerResult({
        source: SAMPLE,
        inputs: [0],
        level: 0,
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8,
        expectedOutput: 'not-the-board'
    });
    assert.strictEqual(mismatch.kind, 'semantic-mismatch', JSON.stringify(mismatch));
    const match = workerResult({
        source: SAMPLE,
        inputs: [0],
        level: 0,
        randomSeed: 'garden-seed',
        replay: false,
        maxInputs: 8,
        expectedOutput: parsed.board
    });
    assert.strictEqual(match.kind, 'ok', JSON.stringify(match));
});

test('compiler-message oracles compare stripped messages', function() {
    const corpus = garden.loadCorpus(path.join(__dirname, '..', 'resources'));
    const item = corpus.find(function(row) { return row.name === 'Background missing'; });
    assert(item);
    const match = workerResult({
        source: item.source,
        inputs: [],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8,
        expectedErrors: item.expectedErrors,
        expectedErrorCount: item.expectedErrorCount
    });
    assert.strictEqual(match.kind, 'compiler-error', JSON.stringify(match));
    const mismatch = workerResult({
        source: item.source,
        inputs: [],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8,
        expectedErrors: ['this message is not produced'],
        expectedErrorCount: 2
    });
    assert.strictEqual(mismatch.kind, 'semantic-mismatch', JSON.stringify(mismatch));
});

test('warning-only compiles are compiler-warning, not ok', function() {
    const corpus = garden.loadCorpus(path.join(__dirname, '..', 'resources'));
    const warning = corpus.find(function(item) {
        return item.kind === 'compiler-message' && item.expectedErrorCount === 0;
    });
    assert(warning, 'need a warning-only compiler-message fixture');
    const result = workerResult({
        source: warning.source,
        inputs: [],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'compiler-warning', JSON.stringify(result));
    assert.strictEqual(result.errorCount, 0);
    assert(Array.isArray(result.errorStrings));
    assert(result.errorStrings.length > 0);
});

test('the worker reports a crash when execution throws', function() {
    const result = workerResult({
        source: SAMPLE,
        inputs: ['not-a-command'],
        level: 0,
        randomSeed: null,
        replay: false,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'crash', JSON.stringify(result));
    assert.strictEqual(typeof result.error.name, 'string');
    assert.strictEqual(typeof result.error.message, 'string');
});

test('unmutated legend of zokoban with replay is ok', function() {
    const corpus = garden.loadCorpus(path.join(__dirname, '..', 'resources'));
    const zokoban = corpus.find(function(item) { return item.name === 'legend of zokoban'; });
    assert(zokoban, 'legend of zokoban should be in the simulation corpus');
    const result = workerResult({
        source: zokoban.source,
        inputs: zokoban.inputs,
        level: zokoban.level,
        randomSeed: zokoban.randomSeed,
        replay: true,
        maxInputs: 8
    });
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
});

test('a compiling message-only game is ok, not an invariant failure', function() {
    const result = workerResult({
        source: `title Message Only

========
OBJECTS
========

Background
black

Player
white

=======
LEGEND
=======

. = Background
P = Player

=========
SOUNDS
=========

================
COLLISIONLAYERS
================

Background
Player

======
RULES
======

==============
WINCONDITIONS
==============

=======
LEVELS
=======

message hello
`,
        inputs: [0, 3],
        level: 0,
        randomSeed: null,
        replay: true,
        maxInputs: 8
    });
    assert.notStrictEqual(result.kind, 'invariant', JSON.stringify(result));
    assert.strictEqual(result.kind, 'ok', JSON.stringify(result));
});

test('the parent classifies a hung child as timeout', function() {
    const hung = path.join(os.tmpdir(), 'monster-garden-hang.js');
    fs.writeFileSync(hung, 'setTimeout(function() {}, 100000);\n');
    const largeStdin = JSON.stringify({ source: 'x'.repeat(200 * 1024), inputs: [], level: 0 });
    return garden.runChild(process.execPath, [hung], largeStdin, 80).then(function(result) {
        assert.strictEqual(result.kind, 'timeout');
    });
});

const KNOWN = [
    'ok', 'compiler-error', 'compiler-warning', 'crash',
    'invariant', 'nondeterministic', 'replay-divergence', 'semantic-mismatch'
];

test('runChild rejects parseable non-results and nonzero exits', function() {
    assert.deepStrictEqual(garden.KNOWN_RESULT_KINDS, KNOWN);
    const empty = path.join(os.tmpdir(), 'monster-garden-empty-json.js');
    fs.writeFileSync(empty, 'process.stdout.write("{}\\n"); process.exit(0);\n');
    return garden.runChild(process.execPath, [empty], '{}', 2000).then(function(result) {
        assert.strictEqual(result.kind, 'crash');
        const liar = path.join(os.tmpdir(), 'monster-garden-ok-nonzero.js');
        fs.writeFileSync(liar, 'process.stdout.write(JSON.stringify({kind:"ok",error:null,fingerprint:"x",detail:"",errorCount:0})+"\\n"); process.exit(73);\n');
        return garden.runChild(process.execPath, [liar], '{}', 2000);
    }).then(function(result) {
        assert.strictEqual(result.kind, 'crash');
        const euro = path.join(os.tmpdir(), 'monster-garden-utf8.js');
        fs.writeFileSync(euro, 'process.stdout.write(JSON.stringify({kind:"crash",error:{name:"Error",message:"euro € here"},fingerprint:"",detail:"",errorCount:0})+"\\n");\n');
        return garden.runChild(process.execPath, [euro], '{}', 2000);
    }).then(function(result) {
        assert.strictEqual(result.kind, 'crash');
        assert.strictEqual(result.error.message, 'euro € here');
        const jsonNull = path.join(os.tmpdir(), 'monster-garden-null-json.js');
        fs.writeFileSync(jsonNull, 'process.stdout.write("null\\n");\n');
        return garden.runChild(process.execPath, [jsonNull], '{}', 2000);
    }).then(function(result) {
        assert.strictEqual(result.kind, 'crash');
        const silent = path.join(os.tmpdir(), 'monster-garden-empty-stdout.js');
        fs.writeFileSync(silent, 'process.exit(0);\n');
        return garden.runChild(process.execPath, [silent], '{}', 2000);
    }).then(function(result) {
        assert.strictEqual(result.kind, 'crash');
        assert.strictEqual(result.error.message, 'empty worker stdout');
        const array = path.join(os.tmpdir(), 'monster-garden-array-json.js');
        fs.writeFileSync(array, 'process.stdout.write("[]\\n");\n');
        return garden.runChild(process.execPath, [array], '{}', 2000);
    }).then(function(result) {
        assert.strictEqual(result.kind, 'crash');
        const workerTimeout = path.join(os.tmpdir(), 'monster-garden-timeout-kind.js');
        fs.writeFileSync(workerTimeout, 'process.stdout.write(JSON.stringify({kind:"timeout",error:null,fingerprint:"",detail:"timeout",errorCount:0})+"\\n");\n');
        return garden.runChild(process.execPath, [workerTimeout], '{}', 2000);
    }).then(function(result) {
        assert.strictEqual(result.kind, 'crash');
    });
});

test('runChild prefers a finished ok result over a late timeout', function() {
    const payload = JSON.stringify({
        kind: 'ok', error: null, fingerprint: 'x', detail: '', errorCount: 0
    });
    const kinds = [];
    let chain = Promise.resolve();
    for (let i = 0; i < 40; i++) {
        chain = chain.then(function() {
            return garden.runChild('/bin/echo', [payload], '{}', 1).then(function(result) {
                kinds.push(result.kind);
                assert.ok(result.kind === 'ok' || result.kind === 'timeout', result.kind);
            });
        });
    }
    return chain.then(function() {
        assert.ok(kinds.indexOf('ok') >= 0, 'expected a finished ok among ' + kinds.join(','));
    });
});

test('run.js --list-mutators prints every mutator and exits 0', function() {
    const child = spawnSync(process.execPath, [path.join(__dirname, 'run.js'), '--list-mutators'], {
        encoding: 'utf8'
    });
    assert.strictEqual(child.status, 0, child.stderr);
    garden.mutators.forEach(function(mutator) {
        assert(child.stdout.indexOf(mutator.name) >= 0, mutator.name);
    });
});

test('run.js rejects malformed options with a nonzero exit', function() {
    const child = spawnSync(process.execPath, [path.join(__dirname, 'run.js'), '--count', '0'], {
        encoding: 'utf8'
    });
    assert.notStrictEqual(child.status, 0);
    assert(/count/.test(child.stderr));
    const oversized = spawnSync(process.execPath, [path.join(__dirname, 'run.js'), '--timeout-ms', '2147483648'], {
        encoding: 'utf8'
    });
    assert.notStrictEqual(oversized.status, 0);
    assert(/timeout-ms/.test(oversized.stderr));
});

test('a one-mutant CLI run is deterministic and writes no artifacts for healthy output', function() {
    const output = fs.mkdtempSync(path.join(os.tmpdir(), 'monster-garden-'));
    const args = [
        path.join(__dirname, 'run.js'),
        '--seed', '12345',
        '--count', '1',
        '--no-shrink',
        '--no-replay',
        '--timeout-ms', '20000',
        '--output', output
    ];
    const first = spawnSync(process.execPath, args, { encoding: 'utf8' });
    const second = spawnSync(process.execPath, args, { encoding: 'utf8' });
    assert.strictEqual(first.status, 0, first.stderr + first.stdout);
    assert.strictEqual(second.status, 0, second.stderr + second.stdout);
    assert.strictEqual(first.stdout, second.stdout);
    assert.strictEqual(fs.readdirSync(output).length, 0);
});

async function main() {
    let passed = 0;
    for (let i = 0; i < tests.length; i++) {
        try {
            await tests[i].fn();
            passed++;
            process.stdout.write('.');
        } catch (error) {
            process.stdout.write('F');
            console.error('\n\n' + tests[i].name + '\n' + error.stack);
        }
    }
    console.log('\n' + passed + '/' + tests.length + ' monster garden tests passed');
    process.exitCode = passed === tests.length ? 0 : 1;
}

main();

