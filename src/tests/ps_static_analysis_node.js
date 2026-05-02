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

console.log('ps_static_analysis_node: ok');
