#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {
    PuzzleScriptEditorIntelligence,
} = require('../src/puzzlescriptEditorIntelligence');

const repoRoot = path.resolve(__dirname, '..', '..', '..');
const ps = new PuzzleScriptEditorIntelligence({ repoRoot });

function readSample(name) {
    return fs.readFileSync(path.join(repoRoot, 'tools', 'linguist', 'samples', name), 'utf8');
}

function assertToken(tokens, predicate, label) {
    assert(tokens.some(predicate), `Expected syntax token: ${label}`);
}

function assertDecoration(decorations, predicate, label) {
    assert(decorations.some(predicate), `Expected color decoration: ${label}`);
}

{
    const source = readSample('simple-block-pushing.puzzlescript');
    const tokens = ps.tokenize(source);
    const decorations = ps.colorDecorations(source);

    assertToken(tokens, token => token.tokenType === 'psMetadata' && token.text === 'title', 'metadata key');
    assertToken(tokens, token => token.tokenType === 'psHeader' && token.text.toLowerCase() === 'objects', 'OBJECTS header');
    assertToken(tokens, token => token.tokenType === 'psName' && token.text === 'Player', 'object name');
    assertToken(tokens, token => token.tokenType === 'psAssignment' && token.text === '=', 'legend assignment');
    assertToken(tokens, token => token.tokenType === 'psLogic' && token.text === 'and', 'legend logic word');
    assertToken(tokens, token => token.tokenType === 'psSound' && token.text === '36772507', 'sound seed');
    assertToken(tokens, token => token.tokenType === 'psArrow' && token.text === '->', 'rule arrow');
    assertToken(tokens, token => token.tokenType === 'psDirection' && token.text === '>', 'movement direction');
    assertToken(tokens, token => token.tokenType === 'psLevel' && token.text === '#', 'level glyph');
    assertDecoration(decorations, span => span.color.toLowerCase() === '#a3ce27', 'lightgreen palette color');
}

{
    const source = readSample('push-directional-excerpt.puzzlescript');
    const tokens = ps.tokenize(source);
    const decorations = ps.colorDecorations(source);

    assertToken(tokens, token => token.tokenType === 'psMetadata' && token.text === 'noaction', 'no-parameter metadata key');
    assertToken(tokens, token => token.tokenType === 'psName' && token.text === 'GunUSingle', 'directional object name');
    assertToken(tokens, token => token.tokenType === 'psLogic' && token.text === 'or', 'property logic word');
    assertToken(tokens, token => token.tokenType === 'psSound' && token.text === 'sfx0', 'sound event');
    assertToken(tokens, token => token.tokenType === 'psDirection' && token.text === 'up', 'rule direction');
    assertToken(tokens, token => token.tokenType === 'psLogic' && token.text === 'no', 'win condition quantifier');
    assertDecoration(decorations, span => span.color.toLowerCase() === '#f7e26b', 'yellow palette color');
}

console.log('syntax-highlighting tests passed');
