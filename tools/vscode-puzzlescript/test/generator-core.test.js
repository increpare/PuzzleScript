#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const {
    candidateToRows,
    findPlayableLevels,
    generatorSidecarPath,
    insertionAfterLevel,
    normalizeRunOptions,
    replacementForLevel,
    resolveGeneratorPath,
    selectedLevelForLine,
} = require('../src/puzzlescriptGeneratorCore');
const {
    parseGeneratorJson,
    parseProgressLine,
} = require('../src/puzzlescriptGeneratorRunner');

const source = [
    'title Generator',
    '',
    'legend',
    '. = Background',
    '# = Wall',
    'P = Player',
    '* = Crate',
    'O = Target',
    '@ = Crate and Target',
    '',
    'levels',
    '#####',
    '#P*.#',
    '#..O#',
    '#####',
    '',
    'message break',
    '',
    '#####',
    '#..P#',
    '#*.O#',
    '#####',
].join('\n');

const levels = findPlayableLevels(source);
assert.strictEqual(levels.length, 2);
assert.deepStrictEqual(levels.map(level => [level.level, level.startLine, level.endLine]), [
    [0, 11, 15],
    [2, 18, 22],
]);
assert.strictEqual(selectedLevelForLine(source, 19).level, 2);
assert.strictEqual(selectedLevelForLine(source, 0).level, 0);

const candidate = {
    cells: [
        ['background wall', 'background wall', 'background wall'],
        ['background player', 'background crate target', 'background target'],
    ],
};
assert.deepStrictEqual(candidateToRows(candidate, source), [
    '###',
    'P@O',
]);

assert.deepStrictEqual(replacementForLevel(source, levels[0], candidate), {
    startLine: 11,
    endLine: 15,
    text: '###\nP@O',
});
assert.deepStrictEqual(insertionAfterLevel(source, levels[0], candidate), {
    line: 15,
    text: '\n\n###\nP@O',
});

assert.strictEqual(generatorSidecarPath('/tmp/game.ps'), '/tmp/game.ps.gen');

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-generator-test-'));
const fakeGenerator = path.join(tmp, process.platform === 'win32' ? 'puzzlescript_generator.exe' : 'puzzlescript_generator');
fs.writeFileSync(fakeGenerator, '');
assert.deepStrictEqual(resolveGeneratorPath(fakeGenerator, '/missing'), {
    path: fakeGenerator,
    exists: true,
    source: 'setting',
});
assert.strictEqual(resolveGeneratorPath('', tmp).exists, false);

assert.deepStrictEqual(
    parseProgressLine('generator_progress elapsed_s=5.0 jobs=2 top=1 samples=44 valid=40 solved=3 timeout=2'),
    { elapsed_s: 5, jobs: 2, top: 1, samples: 44, valid: 40, solved: 3, timeout: 2 }
);
assert.strictEqual(parseProgressLine('hello'), null);
assert.deepStrictEqual(parseGeneratorJson('noise\n{"top":[]}\n', ''), { top: [] });
assert.throws(() => parseGeneratorJson('not json', ''), /did not contain JSON/);

assert.deepStrictEqual(normalizeRunOptions({
    seed: '12',
    timeMs: '1000',
    samples: '20',
    jobs: 'auto',
    solverTimeoutMs: '75',
    solverStrategy: 'greedy',
    topK: '3',
}), {
    seed: 12,
    timeMs: 1000,
    samples: '20',
    jobs: 'auto',
    solverTimeoutMs: 75,
    solverStrategy: 'greedy',
    topK: 3,
});

console.log('generator core tests passed');
