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
        'unterminated-comment'
    ];
    assert.deepStrictEqual(garden.mutators.map(function(mutator) { return mutator.name; }), expected);

    for (let i = 0; i < garden.mutators.length; i++) {
        const result = garden.mutators[i].apply(SAMPLE, new garden.Random(100 + i));
        assert(result, garden.mutators[i].name + ' should apply to the sample');
        assert.notStrictEqual(result.source, SAMPLE, garden.mutators[i].name + ' should change the source');
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

test('only inapplicable mutation errors are skippable', function() {
    assert.strictEqual(garden.isInapplicableMutation(new Error('inapplicable mutation after 2 attempts')), true);
    assert.strictEqual(garden.isInapplicableMutation(new TypeError('mutator exploded')), false);
    assert.strictEqual(garden.isInapplicableMutation(new Error('Cannot read property apply of undefined')), false);
});

test('failure signatures are stable but distinguish different monsters', function() {
    const first = garden.failureSignature({
        kind: 'crash',
        error: { name: 'TypeError', message: 'bad thing\nwith stack noise' }
    });
    const second = garden.failureSignature({
        kind: 'crash',
        error: { name: 'TypeError', message: 'bad thing\nelsewhere' }
    });
    assert.strictEqual(first, second);
    assert.notStrictEqual(first, garden.failureSignature({ kind: 'timeout' }));
    assert.notStrictEqual(first, garden.failureSignature({
        kind: 'crash', error: { name: 'RangeError', message: 'bad thing' }
    }));
});

test('level invariants accept a well-formed level and name the first broken field', function() {
    const good = {
        width: 2,
        height: 3,
        n_tiles: 6,
        objects: { length: 12 },
        movements: { length: 6 }
    };
    assert.strictEqual(garden.checkLevelInvariants(good, 2, 1), null);
    assert(/missing/.test(garden.checkLevelInvariants(null, 2, 1)));
    assert(/dimensions/.test(garden.checkLevelInvariants({ width: 0, height: 3, n_tiles: 0, objects: { length: 0 } }, 2, 1)));
    assert(/n_tiles/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 5, objects: { length: 10 }
    }, 2, 1)));
    assert(/objects/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 6, objects: { length: 5 }
    }, 2, 1)));
    assert(/movements/.test(garden.checkLevelInvariants({
        width: 2, height: 3, n_tiles: 6, objects: { length: 12 }, movements: { length: 2 }
    }, 2, 1)));
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
    const snippet = garden.formatRegression('monster garden 99 3', 'title "X"\nline\n');
    assert.strictEqual(
        snippet,
        '[\n    "monster garden 99 3",\n    ["title \\"X\\"\\nline\\n", [], ""]\n],\n'
    );
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
    assert(first.fingerprint.indexOf('\n') >= 0);
    assert.strictEqual(first.errorCount, 0);
    assert.deepStrictEqual(first, second);
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
    assert.strictEqual(result.fingerprint, 'compiler-error:' + result.errorCount);
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

