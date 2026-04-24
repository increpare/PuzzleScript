#!/usr/bin/env node
'use strict';

const assert = require('assert');
const path = require('path');
const { PuzzleScriptDebugRuntime } = require('../src/puzzlescriptDebugRuntime');

const repoRoot = path.resolve(__dirname, '..', '..', '..');

function lineNumber(source, text) {
    const index = source.split('\n').findIndex(line => line === text);
    assert.notStrictEqual(index, -1, `Expected to find line ${JSON.stringify(text)}`);
    return index + 1;
}

const source = [
    'title Debug Test',
    '',
    'objects',
    'Background',
    'black',
    'Player',
    'red',
    'Crate',
    'blue',
    '',
    'legend',
    '. = Background',
    'P = Player',
    '* = Crate',
    '',
    'sounds',
    '',
    'collisionlayers',
    'Background',
    'Player, Crate',
    '',
    'rules',
    '[ > Player | Crate ] -> [ > Player | > Crate ]',
    '',
    'winconditions',
    '',
    'levels',
    'P*.',
].join('\n');

{
    const runtime = new PuzzleScriptDebugRuntime({ repoRoot });
    const result = runtime.compile(source);
    const ruleLine = lineNumber(source, '[ > Player | Crate ] -> [ > Player | > Crate ]');
    assert(result.ruleLines.includes(ruleLine), 'compiled rule line should be a breakpoint target');

    const validations = runtime.validateBreakpoints([
        ruleLine,
        lineNumber(source, 'Player'),
        lineNumber(source, 'levels'),
    ]);
    assert.strictEqual(validations[0].verified, true);
    assert.strictEqual(validations[1].verified, false);
    assert.strictEqual(validations[2].verified, false);
}

{
    const runtime = new PuzzleScriptDebugRuntime({ repoRoot });
    const ruleLine = lineNumber(source, '[ > Player | Crate ] -> [ > Player | > Crate ]');
    runtime.compile(source);
    const snapshots = runtime.input('right');
    assert(snapshots.length > 0, 'debug input should capture runtime snapshots');
    assert(
        snapshots.some(snapshot => snapshot.sourceLine === ruleLine),
        'right input should pause-capable snapshot on the pushing rule line'
    );
    const ruleSnapshot = snapshots.find(snapshot => snapshot.sourceLine === ruleLine);
    assert(/Rule line/.test(ruleSnapshot.label));
    assert.strictEqual(ruleSnapshot.input, 'right');
}

console.log('debugger tests passed');
