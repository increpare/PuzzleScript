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

Player
white

Goal
yellow

${'======='}
LEGEND
${'======='}

. = Background
P = Player
G = Goal

${'======='}
SOUNDS
${'======='}

================
COLLISIONLAYERS
================

Background
Player, Goal

=====
RULES
=====

[ > Player ] -> [ > Player ]

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

console.log('ps_static_analysis_node: ok');
