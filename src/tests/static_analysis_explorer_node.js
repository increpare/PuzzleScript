#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const { analyzeSource } = require('./ps_static_analysis');
const {
    buildExplorerModel,
    editorHrefForSource,
    renderExplorerHtml,
} = require('./build_static_analysis_explorer');

const EXPLORER_FIXTURE = `
title Explorer Fixture
========
OBJECTS
========
Background
black
Player
white
Wall
gray
Rock
brown
BodyH
red
BodyV
blue
BodyD
purple
Alpha
green
Beta
yellow
MarkerX
pink
MarkerY
orange
${'======='}
LEGEND
${'======='}
. = Background
P = Player
# = Wall
R = Rock
h = BodyH
v = BodyV
d = BodyD
a = Alpha
b = Beta
Body = BodyH or BodyV or BodyD
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
Wall
Rock
BodyH, BodyV, BodyD
Alpha
Beta
MarkerX
MarkerY
=====
RULES
=====
[ Body ] -> [ Body ]
[ Wall ] -> sfx0
[ Alpha ] -> [ Alpha MarkerX ]
+ [ Beta ] -> [ Beta MarkerY ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
P#Rhvdab
`;

const repoRoot = path.resolve(__dirname, '..', '..');
const sourcePath = path.join(repoRoot, 'src/tests/solver_tests/explorer_fixture.txt');
const report = analyzeSource(EXPLORER_FIXTURE, { sourcePath });
assert.strictEqual(report.status, 'ok');

const model = buildExplorerModel([report], { repoRoot });
assert.strictEqual(model.games.length, 1);
const game = model.games[0];
assert.strictEqual(game.display_name, 'explorer_fixture.txt');
assert.ok(game.editor_href.endsWith('/src/editor.html?file=tests%2Fsolver_tests%2Fexplorer_fixture.txt'));
assert.ok(game.mergeable.some(pair => pair.objects.join(',') === 'BodyH,BodyV'));
assert.ok(game.mergeable_groups.some(group => group.objects.join(',') === 'BodyD,BodyH,BodyV'));
assert.ok(game.static_objects.includes('Wall'));
assert.ok(game.static_objects.includes('Rock'));
assert.strictEqual(game.static_objects_label, 'Background, Wall, Rock, BodyH, BodyV, BodyD, Alpha, Beta');
assert.ok(game.static_layers.some(layer => layer.objects.includes('Wall')));
assert.ok(game.inert_rules.some(rule => rule.text.includes('sfx0')));
assert.ok(game.rulegroup_flow.some(group => group.status === 'candidate' && group.components.length === 2));

assert.strictEqual(
    editorHrefForSource(sourcePath, { repoRoot }),
    '/src/editor.html?file=tests%2Fsolver_tests%2Fexplorer_fixture.txt'
);

const html = renderExplorerHtml(model);
assert.ok(html.includes('PuzzleScript Static Analysis Explorer'));
assert.ok(html.includes('explorer_fixture.txt'));
assert.ok(html.includes('BodyH'));
assert.ok(html.includes('BodyD, BodyH, BodyV'));
assert.ok(!html.includes('BodyD = BodyH = BodyV'));
assert.ok(html.includes('Wall, Rock'));
assert.ok(html.includes('Open in editor'));
assert.ok(html.includes('<details class="section"'));
assert.ok(html.includes('rulegroup_flow'));
assert.ok(html.includes('Inert Collision Layers'));
assert.ok(html.includes('Likely cosmetic objects'));

const editorSource = fs.readFileSync(path.join(repoRoot, 'src/js/editor.js'), 'utf8');
assert.ok(editorSource.includes('getParameterByName("file")'), 'editor should accept explorer file links');
assert.ok(editorSource.includes('tryLoadSourceFile'), 'editor should load same-origin source files');

console.log('static_analysis_explorer_node: ok');
