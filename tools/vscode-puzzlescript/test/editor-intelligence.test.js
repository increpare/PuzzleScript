#!/usr/bin/env node
'use strict';

const assert = require('assert');
const path = require('path');
const {
    PuzzleScriptEditorIntelligence,
} = require('../src/puzzlescriptEditorIntelligence');

const repoRoot = path.resolve(__dirname, '..', '..', '..');
const ps = new PuzzleScriptEditorIntelligence({ repoRoot });

function lineIndex(source, text) {
    const index = source.split('\n').findIndex(line => line === text);
    assert.notStrictEqual(index, -1, `Expected to find line ${JSON.stringify(text)}`);
    return index;
}

function completionTexts(source, line, character) {
    return ps.complete(source, { line, character }).list.map(item => item.text);
}

const baseSource = [
    'title Test',
    '',
    'objects',
    'Background',
    'black',
    '00000',
    '00000',
    '00000',
    '00000',
    '00000',
    'Player',
    'red blue',
    '.1...',
    '.....',
    '.....',
    '.....',
    '.....',
    '',
    'legend',
    '. = Background',
    'P = Player',
    '',
    'sounds',
    '',
    'collisionlayers',
    'Background',
    'Player',
    '',
    'rules',
    '[ > Player ] -> [ > Player ]',
    '',
    'winconditions',
    '',
    'levels',
    'P',
    '',
].join('\n');

{
    const tokens = ps.tokenize(baseSource);
    assert(tokens.some(token => token.tokenType === 'psHeader' && token.text === 'objects'));
    assert(tokens.some(token => token.tokenType === 'psName' && token.text === 'Player'));
    assert(tokens.some(token => token.tokenType === 'psArrow' && token.text === '->'));
}

{
    const decorations = ps.colorDecorations(baseSource);
    const spriteLine = lineIndex(baseSource, '.1...');
    assert(decorations.some(span => span.line === spriteLine && span.start === 1 && span.color.toLowerCase() === '#1d57f7'));
}

{
    const source = ['objects', 'Player', 'r'].join('\n');
    assert(completionTexts(source, 2, 1).includes('red'));
}

{
    const source = [
        'objects',
        'Player_up',
        'red',
        '.....',
        '..0..',
        '.....',
        '.....',
        '.....',
        'P',
    ].join('\n');
    const texts = completionTexts(source, 8, 1);
    assert(texts.some(text => text.startsWith('Player_down\nred\n')));
    assert(texts.some(text => text.startsWith('Player_left\nred\n')));
    assert(texts.some(text => text.startsWith('Player_right\nred\n')));
}

{
    const source = [
        'objects',
        'Player',
        'red',
        '.....',
        '.....',
        '.....',
        '.....',
        '.....',
        'Crate_up',
        'blue',
        '.....',
        '.....',
        '.....',
        '.....',
        '.....',
        'Crate_down',
        'blue',
        '.....',
        '.....',
        '.....',
        '.....',
        '.....',
        '',
        'legend',
        '. = Player',
        '',
        'sounds',
        '',
        'collisionlayers',
        'Player',
        'Crate_up, Crate_down',
        '',
        'rules',
        'up [ > Player | Crate_up ] -> [ > Player | > Crate_up ]',
        'd',
    ].join('\n');
    const texts = completionTexts(source, 34, 1);
    assert(texts.includes('down [ > Player | Crate_down ] -> [ > Player | > Crate_down ]'));
}

{
    const diagnostics = ps.diagnose([
        'objects',
        'Player',
        'red',
        '.....',
        '.....',
        '.....',
        '.....',
        '.....',
        '',
        'legend',
        '. = Background',
        '',
        'sounds',
        '',
        'collisionlayers',
        'Player',
        '',
        'rules',
        '[ Ghost ] -> [ Ghost ]',
    ].join('\n'));
    assert(diagnostics.some(diagnostic => /Ghost/i.test(diagnostic.message)));
}

console.log('editor-intelligence tests passed');
