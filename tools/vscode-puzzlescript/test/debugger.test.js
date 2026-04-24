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
    assert.strictEqual(result.current.width, 3);
    assert.strictEqual(result.current.height, 1);
    assert(result.current.objectInfos.some(object => object.name === 'player'));
    assert(result.current.objectInfos.some(object => object.name === 'crate'));
    assert(result.current.objectInfos.every(object => Array.isArray(object.spriteMatrix)));
    assert(result.current.objectInfos.some(object => object.name === 'player' && object.spriteMatrix.length === 5));
    assert(result.current.objects.length > 0);

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
    assert(/Rule group|Rule line/.test(ruleSnapshot.label));
    assert.strictEqual(ruleSnapshot.input, 'right');
}

{
    const runtime = new PuzzleScriptDebugRuntime({ repoRoot });
    const ruleLine = lineNumber(source, '[ > Player | Crate ] -> [ > Player | > Crate ]');
    runtime.compile(source);
    const paused = runtime.runInput('right', { breakpoints: [ruleLine] });
    assert.strictEqual(paused.paused, true, 'debug input should stop as soon as the rule breakpoint is hit');
    assert.strictEqual(paused.snapshot.sourceLine, ruleLine);
    assert.strictEqual(paused.snapshot.kind, 'rule-group');
    assert(/^Rule group/.test(paused.snapshot.label));
    assert.strictEqual(runtime.current.sourceLine, ruleLine);

    const completed = runtime.resume({ breakpoints: [ruleLine] });
    assert.strictEqual(completed.paused, false, 'resume should finish the current input after the already-hit breakpoint');
    assert(/player/i.test(completed.snapshot.serializedLevel));
}

{
    const runtime = new PuzzleScriptDebugRuntime({ repoRoot });
    const ruleLine = lineNumber(source, '[ > Player | Crate ] -> [ > Player | > Crate ]');
    runtime.compile(source);
    const paused = runtime.runInput('right', { breakpoints: [ruleLine] });
    const stepped = runtime.resume({ step: true });
    assert.strictEqual(stepped.paused, false, 'step from a breakpoint should finish if there is no later rule snapshot');
    assert(stepped.snapshot.index > paused.snapshot.index);
}

{
    const runtime = new PuzzleScriptDebugRuntime({ repoRoot });
    runtime.compile(source);
    const paused = runtime.runInput('right', { step: true });
    assert.strictEqual(paused.paused, true, 'step mode should stop at the first rule-group boundary');
    assert.strictEqual(paused.snapshot.sourceLine, lineNumber(source, '[ > Player | Crate ] -> [ > Player | > Crate ]'));
    assert.strictEqual(paused.snapshot.kind, 'rule-group');
    const stepped = runtime.resume({ step: true });
    assert.strictEqual(stepped.paused, false, 'step resume should finish if there is no later rule-group boundary');
    assert.notStrictEqual(stepped.snapshot.index, paused.snapshot.index);
}

console.log('debugger tests passed');
