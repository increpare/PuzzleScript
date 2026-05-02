#!/usr/bin/env node
'use strict';

const assert = require('assert');

const { analyzeSource } = require('./ps_static_analysis');

const SIMPLE_GAME = `
title Static Analysis Simple

========
OBJECTS
========

Background
black

Hero
white

Goal
yellow

${'======='}
LEGEND
${'======='}

. = Background
P = Hero
G = Goal
Player = Hero
Avatar = Hero

${'======='}
SOUNDS
${'======='}

================
COLLISIONLAYERS
================

Background
Hero, Goal

=====
RULES
=====

[ > Hero ] -> [ > Hero ]

=============
WINCONDITIONS
=============

Some Player on Goal

======
LEVELS
======

P.G
...
`;

const report = analyzeSource(SIMPLE_GAME, { sourcePath: 'simple.txt' });
assert.strictEqual(report.schema, 'ps-static-analysis-v1');
assert.strictEqual(report.status, 'ok');
assert.strictEqual(report.source.path, 'simple.txt');
assert.ok(report.ps_tagged, 'report should include ps_tagged by default');
assert.ok(report.facts.mergeability, 'report should include mergeability facts');
assert.ok(report.facts.movement_action, 'report should include movement_action facts');
assert.ok(report.facts.count_layer_invariants, 'report should include count_layer_invariants facts');
assert.ok(report.facts.transient_boundary, 'report should include transient_boundary facts');
assert.deepStrictEqual(
    report.ps_tagged.objects.map(object => object.name).sort(),
    ['Background', 'Goal', 'Hero'],
    'ps_tagged should preserve object names'
);
assert.deepStrictEqual(
    report.ps_tagged.collision_layers.map(layer => layer.objects),
    [['Background'], ['Hero', 'Goal']],
    'ps_tagged should preserve collision layer membership'
);
assert.deepStrictEqual(
    report.ps_tagged.properties.find(property => property.name === 'avatar').members,
    ['Hero'],
    'ps_tagged should expose legend synonym members'
);
assert.strictEqual(report.ps_tagged.levels.length, 1, 'ps_tagged should summarize levels');
assert.deepStrictEqual(
    report.ps_tagged.objects.find(object => object.name === 'Hero').tags.present_in_all_levels,
    true,
    'object tags should include aggregate level presence'
);
assert.ok(report.ps_tagged.winconditions.length > 0, 'ps_tagged should expose win conditions');

const RULE_SHAPE_GAME = `
title Rule Shape

========
OBJECTS
========

Background
black

Alpha
white

Beta
red

Gamma
blue

${'======='}
LEGEND
${'======='}

. = Background
a = Alpha
b = Beta
c = Gamma
Player = Alpha

${'======='}
SOUNDS
${'======='}

================
COLLISIONLAYERS
================

Background
Alpha, Beta
Gamma

=====
RULES
=====

right [ Alpha | right Beta no Gamma ] -> [ up Alpha | Beta right Gamma ]
late [ Gamma ] -> [ no Gamma ]

=============
WINCONDITIONS
=============

Some Alpha

======
LEVELS
======

ab.
...
`;

const ruleShape = analyzeSource(RULE_SHAPE_GAME, { sourcePath: 'rule_shape.txt' });
const early = ruleShape.ps_tagged.rule_sections.find(section => section.name === 'early');
const late = ruleShape.ps_tagged.rule_sections.find(section => section.name === 'late');
assert.strictEqual(early.groups.length, 1, 'early section should contain one group');
assert.strictEqual(late.groups.length, 1, 'late section should contain one group');
const shapeRule = early.groups[0].rules[0];
assert.strictEqual(shapeRule.direction, 'right');
assert.deepStrictEqual(shapeRule.lhs[0][0], [
    { kind: 'present', ref: { type: 'object', name: 'Alpha', canonical_name: 'alpha' }, movement: null },
]);
assert.deepStrictEqual(shapeRule.lhs[0][1], [
    { kind: 'present', ref: { type: 'object', name: 'Beta', canonical_name: 'beta' }, movement: 'right' },
    { kind: 'absent', ref: { type: 'object', name: 'Gamma', canonical_name: 'gamma' }, movement: null },
]);
assert.deepStrictEqual(shapeRule.rhs[0][0], [
    { kind: 'present', ref: { type: 'object', name: 'Alpha', canonical_name: 'alpha' }, movement: 'up' },
]);
assert.deepStrictEqual(shapeRule.rhs[0][1], [
    { kind: 'present', ref: { type: 'object', name: 'Beta', canonical_name: 'beta' }, movement: null },
    { kind: 'present', ref: { type: 'object', name: 'Gamma', canonical_name: 'gamma' }, movement: 'right' },
]);

const COMMAND_GAME = `
title Command Tags
========
OBJECTS
========
Background
black
Alpha
white
${'======='}
LEGEND
${'======='}
. = Background
a = Alpha
Player = Alpha
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Alpha
=====
RULES
=====
[ Alpha ] -> sfx0
[ Alpha ] -> checkpoint
=============
WINCONDITIONS
=============
Some Alpha
======
LEVELS
======
a
`;

const commandReport = analyzeSource(COMMAND_GAME, { sourcePath: 'commands.txt' });
const commandRules = commandReport.ps_tagged.rule_sections[0].groups.flatMap(group => group.rules);
assert.strictEqual(commandRules.length, 2, 'command-only rules should remain present');
assert.strictEqual(commandRules[0].tags.inert_command_only, true, 'sfx-only rule is inert for solver state');
assert.strictEqual(commandRules[0].tags.solver_state_active, false, 'sfx-only rule is not solver-state active');
assert.strictEqual(commandRules[1].tags.command_only, true, 'checkpoint-only rule is command-only');
assert.strictEqual(commandRules[1].tags.solver_state_active, true, 'checkpoint is semantic/metagame-active');

console.log('ps_static_analysis_node: ok');
