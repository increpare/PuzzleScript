#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const { analyzeSource } = require('./ps_static_analysis');
const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

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
assert.ok(report.summary.proved > 0, 'summary should count proved facts');
assert.ok(report.ps_tagged, 'report should include ps_tagged by default');
assert.ok(report.facts.mergeability, 'report should include mergeability facts');
assert.ok(report.facts.movement_action, 'report should include movement_action facts');
assert.ok(report.facts.count_layer_invariants, 'report should include count_layer_invariants facts');
assert.ok(report.facts.transient_boundary, 'report should include transient_boundary facts');
assert.ok(report.facts.rulegroup_flow, 'report should include rulegroup_flow facts');
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
assert.deepStrictEqual(
    report.ps_tagged.properties.find(property => property.name === 'avatar'),
    {
        name: 'avatar',
        canonical_name: 'avatar',
        kind: 'synonym',
        members: ['Hero'],
        tags: {},
    },
    'ps_tagged should distinguish one-object synonyms from properties'
);
assert.strictEqual(report.ps_tagged.levels.length, 1, 'ps_tagged should summarize levels');
assert.deepStrictEqual(
    report.ps_tagged.objects.find(object => object.name === 'Hero').tags.present_in_all_levels,
    true,
    'object tags should include aggregate level presence'
);
assert.ok(report.ps_tagged.winconditions.length > 0, 'ps_tagged should expose win conditions');

const LEVEL_PRESENCE_GAME = `
title Level Presence
========
OBJECTS
========
Background
black
Always
white
Sometimes
yellow
Never
red
${'======='}
LEGEND
${'======='}
. = Background
A = Always
S = Sometimes
Player = Always
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Always
Sometimes
Never
=====
RULES
=====
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
message ignored for aggregate presence
A..

A.S
`;

const levelPresence = analyzeSource(LEVEL_PRESENCE_GAME, { sourcePath: 'level_presence.txt' });
const levelPresenceObjects = new Map(levelPresence.ps_tagged.objects.map(object => [object.name, object.tags]));
assert.deepStrictEqual(levelPresence.ps_tagged.levels[0], {
    index: 0,
    kind: 'message',
    objects_present: [],
    layers_present: [],
    tags: {},
});
assert.deepStrictEqual(levelPresence.ps_tagged.levels[1].objects_present, ['Always', 'Background']);
assert.deepStrictEqual(levelPresence.ps_tagged.levels[1].layers_present, [0, 1]);
assert.strictEqual(levelPresence.ps_tagged.levels[1].width, 3);
assert.strictEqual(levelPresence.ps_tagged.levels[1].height, 1);
assert.deepStrictEqual(levelPresence.ps_tagged.levels[2].objects_present, ['Always', 'Background', 'Sometimes']);
assert.deepStrictEqual(levelPresence.ps_tagged.levels[2].layers_present, [0, 1, 2]);
assert.strictEqual(levelPresenceObjects.get('Always').present_in_all_levels, true);
assert.strictEqual(levelPresenceObjects.get('Always').present_in_some_levels, false);
assert.strictEqual(levelPresenceObjects.get('Always').present_in_no_levels, false);
assert.strictEqual(levelPresenceObjects.get('Sometimes').present_in_all_levels, false);
assert.strictEqual(levelPresenceObjects.get('Sometimes').present_in_some_levels, true);
assert.strictEqual(levelPresenceObjects.get('Sometimes').present_in_no_levels, false);
assert.strictEqual(levelPresenceObjects.get('Never').present_in_all_levels, false);
assert.strictEqual(levelPresenceObjects.get('Never').present_in_some_levels, false);
assert.strictEqual(levelPresenceObjects.get('Never').present_in_no_levels, true);

const WINCONDITION_SHAPE_GAME = `
title Wincondition Shape
========
OBJECTS
========
Background
black
Alpha
white
Goal
yellow
${'======='}
LEGEND
${'======='}
. = Background
a = Alpha
g = Goal
Player = Alpha
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Alpha
Goal
=====
RULES
=====
=============
WINCONDITIONS
=============
Some Alpha
All Alpha on Goal
======
LEVELS
======
ag
`;

const winconditionShape = analyzeSource(WINCONDITION_SHAPE_GAME, { sourcePath: 'wincondition_shape.txt' });
assert.strictEqual(winconditionShape.ps_tagged.winconditions[0].tags.plain, true);
assert.deepStrictEqual(winconditionShape.ps_tagged.winconditions[0].targets, []);
assert.strictEqual(winconditionShape.ps_tagged.winconditions[1].tags.plain, false);
assert.deepStrictEqual(winconditionShape.ps_tagged.winconditions[1].targets, ['Goal']);

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
    { kind: 'present', ref: { type: 'object', name: 'Alpha', canonical_name: 'alpha' }, movement: null, expanded_objects: ['Alpha'] },
]);
assert.deepStrictEqual(shapeRule.lhs[0][1], [
    { kind: 'present', ref: { type: 'object', name: 'Beta', canonical_name: 'beta' }, movement: 'right', expanded_objects: ['Beta'] },
    { kind: 'absent', ref: { type: 'object', name: 'Gamma', canonical_name: 'gamma' }, movement: null, expanded_objects: ['Gamma'] },
]);
assert.deepStrictEqual(shapeRule.rhs[0][0], [
    { kind: 'present', ref: { type: 'object', name: 'Alpha', canonical_name: 'alpha' }, movement: 'up', expanded_objects: ['Alpha'] },
]);
assert.deepStrictEqual(shapeRule.rhs[0][1], [
    { kind: 'present', ref: { type: 'object', name: 'Beta', canonical_name: 'beta' }, movement: null, expanded_objects: ['Beta'] },
    { kind: 'present', ref: { type: 'object', name: 'Gamma', canonical_name: 'gamma' }, movement: 'right', expanded_objects: ['Gamma'] },
]);
const lateClearShapeRule = late.groups[0].rules[0];
assert.deepStrictEqual(lateClearShapeRule.rhs[0][0], [
    { kind: 'absent', ref: { type: 'object', name: 'Gamma', canonical_name: 'gamma' }, movement: null, expanded_objects: ['Gamma'] },
]);
assert.strictEqual(lateClearShapeRule.tags.object_mutating, true, 'RHS exclusion terms should be object-mutating');

const LOOP_HIERARCHY_GAME = `
title Loop Hierarchy
========
OBJECTS
========
Background
black
Player
white
Marker
red
Goal
yellow
${'======='}
LEGEND
${'======='}
. = Background
P = Player
M = Marker
G = Goal
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
Marker
Goal
=====
RULES
=====
[ Player ] -> [ Player ]
startLoop
[ Player ] -> [ Player Marker ]
+ [ Goal ] -> [ Goal Marker ]
[ Marker ] -> [ Marker ]
endLoop
late [ Marker ] -> [ Marker ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
PG
`;

const loopHierarchy = analyzeSource(LOOP_HIERARCHY_GAME, { sourcePath: 'loop_hierarchy.txt' });
const loopEarly = loopHierarchy.ps_tagged.rule_sections.find(section => section.name === 'early');
const loopLate = loopHierarchy.ps_tagged.rule_sections.find(section => section.name === 'late');
assert.deepStrictEqual(loopEarly.loops.map(loop => loop.group_ids), [['early_group_1', 'early_group_2']]);
assert.deepStrictEqual(loopLate.loops, [], 'late section should not inherit empty early-loop summaries');
assert.deepStrictEqual(loopEarly.groups.map(group => group.rules.map(rule => rule.id)), [
    ['early_group_0_rule_0'],
    ['early_group_1_rule_0', 'early_group_1_rule_1'],
    ['early_group_2_rule_0'],
]);
assert.deepStrictEqual(loopLate.groups.map(group => group.rules.map(rule => rule.id)), [
    ['late_group_0_rule_0'],
]);

const GROUP_TAGS_GAME = `
title Group Tags
========
OBJECTS
========
Background
black
Alpha
white
Beta
blue
Gamma
green
${'======='}
LEGEND
${'======='}
. = Background
a = Alpha
b = Beta
g = Gamma
Player = Alpha
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Alpha
Beta
Gamma
=====
RULES
=====
[ Alpha ] -> sfx0
+ [ Beta ] -> [ Beta ] sfx1
[ Alpha ] -> [ right Alpha ]
+ [ Beta ] -> [ left Beta ]
[ Alpha ] -> [ Alpha Gamma ]
+ [ Beta ] -> [ Beta ] again
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
ab
`;

const groupTags = analyzeSource(GROUP_TAGS_GAME, { sourcePath: 'group_tags.txt' });
const groupTagGroups = groupTags.ps_tagged.rule_sections.find(section => section.name === 'early').groups;
assert.deepStrictEqual(groupTagGroups.map(group => group.tags), [
    {
        has_again: false,
        object_mutating: false,
        movement_only: false,
        command_only: true,
        solver_state_active: false,
    },
    {
        has_again: false,
        object_mutating: false,
        movement_only: true,
        command_only: false,
        solver_state_active: true,
    },
    {
        has_again: true,
        object_mutating: true,
        movement_only: false,
        command_only: false,
        solver_state_active: true,
    },
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
[ Alpha ] -> cancel
[ Alpha ] -> restart
[ Alpha ] -> win
[ Alpha ] -> again
[ Alpha ] -> message hello there
[ Alpha ] -> [ Alpha ] sfx0
[ right Alpha ] -> [ right Alpha ] sfx0
[ no Alpha ] -> [ no Alpha ] sfx0
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
assert.strictEqual(commandRules.length, 10, 'command-only rules should remain present');
assert.strictEqual(commandRules[0].tags.inert_command_only, true, 'sfx-only rule is inert for solver state');
assert.strictEqual(commandRules[0].tags.solver_state_active, false, 'sfx-only rule is not solver-state active');
assert.strictEqual(commandRules[1].tags.command_only, true, 'checkpoint-only rule is command-only');
assert.strictEqual(commandRules[1].tags.solver_state_active, true, 'checkpoint is semantic/metagame-active');
assert.deepStrictEqual(
    commandRules.slice(1, 6).map(rule => rule.summary.semantic_commands),
    [['checkpoint'], ['cancel'], ['restart'], ['win'], ['again']],
    'semantic commands should be separated from inert command noise'
);
for (const rule of commandRules.slice(1, 6)) {
    assert.strictEqual(rule.tags.command_only, true);
    assert.strictEqual(rule.tags.inert_command_only, false);
    assert.strictEqual(rule.tags.solver_state_active, true);
}
assert.strictEqual(commandRules[5].tags.has_again, true, 'again should be tracked separately on command-only rules');
assert.deepStrictEqual(commandRules[6].summary.inert_commands, ['message']);
assert.strictEqual(commandRules[6].tags.inert_command_only, true, 'message-only rule is inert for solver state');
assert.strictEqual(commandRules[6].tags.solver_state_active, false);
assert.strictEqual(commandRules[7].tags.inert_command_only, true, 'unchanged RHS plus sfx is inert-command-only');
assert.strictEqual(commandRules[7].tags.solver_state_active, false);
assert.strictEqual(commandRules[8].tags.inert_command_only, true, 'unchanged movement RHS plus sfx is inert-command-only');
assert.strictEqual(commandRules[8].tags.solver_state_active, false);
assert.strictEqual(commandRules[9].tags.object_mutating, false, 'unchanged negated RHS does not mutate objects');
assert.strictEqual(commandRules[9].tags.command_only, true);
assert.strictEqual(commandRules[9].tags.inert_command_only, true, 'unchanged negated RHS plus sfx is inert-command-only');
assert.strictEqual(commandRules[9].tags.solver_state_active, false);

const MERGEABLE_GAME = `
title Mergeable
========
OBJECTS
========
Background
black
BodyH
white
BodyV
white
Goal
yellow
${'======='}
LEGEND
${'======='}
. = Background
h = BodyH
v = BodyV
g = Goal
Player = BodyH or BodyV
Body = BodyH or BodyV
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
BodyH, BodyV, Goal
=====
RULES
=====
[ Body ] -> [ Body ]
[ no Body ] -> [ no Body ]
[ > Body ] -> [ > Body ]
=============
WINCONDITIONS
=============
Some Body on Goal
======
LEVELS
======
h.g
`;

const mergeable = analyzeSource(MERGEABLE_GAME, { sourcePath: 'mergeable.txt' });
assert.deepStrictEqual(
    mergeable.ps_tagged.properties.find(property => property.name === 'body'),
    {
        name: 'body',
        canonical_name: 'body',
        kind: 'property',
        members: ['BodyH', 'BodyV'],
        tags: {},
    },
    'ps_tagged should distinguish multi-object properties from synonyms'
);
const mergeFact = mergeable.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.ok(mergeFact, 'BodyH/BodyV should produce a mergeability fact');
assert.strictEqual(mergeFact.status, 'candidate');
assert.ok(mergeFact.proof.includes('same_collision_layer'));
assert.ok(mergeFact.proof.includes('observed_only_through_shared_sets'));

const RHS_SPAWN_MERGEABLE_GAME = `
title RHS Spawn Mergeable
========
OBJECTS
========
Background
black
SpawnerH
red
SpawnerV
blue
BodyH
white
BodyV
gray
Goal
yellow
${'======='}
LEGEND
${'======='}
. = Background
h = SpawnerH
v = SpawnerV
g = Goal
Body = BodyH or BodyV
Player = SpawnerH
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
SpawnerH, SpawnerV
BodyH, BodyV, Goal
=====
RULES
=====
[ SpawnerH ] -> [ SpawnerH BodyH ]
[ SpawnerV ] -> [ SpawnerV BodyV ]
=============
WINCONDITIONS
=============
Some Body on Goal
======
LEVELS
======
hvg
`;
const rhsSpawnMergeable = analyzeSource(RHS_SPAWN_MERGEABLE_GAME, { sourcePath: 'rhs_spawn_mergeable.txt' });
const rhsSpawnMergeFact = rhsSpawnMergeable.facts.mergeability.find(item =>
    item.subjects.objects.join(',') === 'BodyH,BodyV'
);
assert.strictEqual(rhsSpawnMergeFact.status, 'candidate', 'RHS-only direct spawning should not distinguish merge candidates');
assert.ok(rhsSpawnMergeFact.proof.includes('observed_only_through_shared_sets'));

const DIRECT_READ_GAME = MERGEABLE_GAME.replace('[ Body ] -> [ Body ]', '[ BodyH ] -> [ BodyH ]');
const directRead = analyzeSource(DIRECT_READ_GAME, { sourcePath: 'direct_read.txt' });
const directReadFact = directRead.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directReadFact.status, 'rejected');
assert.ok(directReadFact.blockers.includes('individual_lhs_read'));

const DIRECT_NEGATION_GAME = MERGEABLE_GAME.replace('[ no Body ] -> [ no Body ]', '[ no BodyH ] -> [ no BodyH ]');
const directNegation = analyzeSource(DIRECT_NEGATION_GAME, { sourcePath: 'direct_negation.txt' });
const directNegationFact = directNegation.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directNegationFact.status, 'rejected');
assert.ok(directNegationFact.blockers.includes('individual_lhs_read'));

const DIRECT_MOVEMENT_READ_GAME = MERGEABLE_GAME.replace('[ > Body ] -> [ > Body ]', '[ right BodyH ] -> [ right BodyH ]');
const directMovementRead = analyzeSource(DIRECT_MOVEMENT_READ_GAME, { sourcePath: 'direct_movement_read.txt' });
const directMovementReadFact = directMovementRead.facts.mergeability.find(item =>
    item.subjects.objects.join(',') === 'BodyH,BodyV'
);
assert.strictEqual(directMovementReadFact.status, 'rejected');
assert.ok(directMovementReadFact.blockers.includes('individual_lhs_read'));

const PARTIAL_PROPERTY_GAME = MERGEABLE_GAME
    .replace('Body = BodyH or BodyV', 'Body = BodyH or BodyV\nPartialBody = BodyH or Goal')
    .replace('[ Body ] -> [ Body ]', '[ PartialBody ] -> [ PartialBody ]');
const partialProperty = analyzeSource(PARTIAL_PROPERTY_GAME, { sourcePath: 'partial_property.txt' });
const partialPropertyFact = partialProperty.facts.mergeability.find(item =>
    item.subjects.objects.join(',') === 'BodyH,BodyV'
);
assert.strictEqual(partialPropertyFact.status, 'rejected');
assert.ok(partialPropertyFact.blockers.includes('partial_property_observation'));

const DIRECT_WIN_GAME = MERGEABLE_GAME.replace('Some Body on Goal', 'Some BodyH on Goal');
const directWin = analyzeSource(DIRECT_WIN_GAME, { sourcePath: 'direct_win.txt' });
const directWinFact = directWin.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directWinFact.status, 'rejected');
assert.ok(directWinFact.blockers.includes('different_win_roles'));

const DIRECT_TARGET_WIN_GAME = MERGEABLE_GAME.replace('Some Body on Goal', 'Some Goal on BodyH');
const directTargetWin = analyzeSource(DIRECT_TARGET_WIN_GAME, { sourcePath: 'direct_target_win.txt' });
const directTargetWinFact = directTargetWin.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directTargetWinFact.status, 'rejected');
assert.ok(directTargetWinFact.blockers.includes('different_win_roles'));

const DUPLICATE_LAYER_GAME = MERGEABLE_GAME.replace('BodyH, BodyV, Goal', 'BodyH, BodyH, BodyV, Goal');
const duplicateLayer = analyzeSource(DUPLICATE_LAYER_GAME, { sourcePath: 'duplicate_layer.txt' });
assert.deepStrictEqual(
    duplicateLayer.ps_tagged.collision_layers[1].objects,
    ['BodyH', 'BodyV', 'Goal'],
    'collision layer summaries should deduplicate object names'
);
assert.ok(
    duplicateLayer.facts.mergeability.every(item => item.subjects.objects[0] !== item.subjects.objects[1]),
    'mergeability should not emit self-merge facts'
);

const AUTO_TICK_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ Goal ] -> [ Hero ]');
const autoTick = analyzeSource(AUTO_TICK_GAME, { sourcePath: 'auto_tick.txt' });
const autoAction = autoTick.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(autoAction.status, 'rejected');
assert.ok(autoAction.blockers.includes('autonomous_solver_active_rule'));

const ACTION_RULE_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ action Player ] -> [ Player Goal ]');
const actionRule = analyzeSource(ACTION_RULE_GAME, { sourcePath: 'action_rule.txt' });
const actionRuleFact = actionRule.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(actionRuleFact.status, 'rejected');
assert.ok(actionRuleFact.blockers.includes('reads_action'));

const ACTION_AGAIN_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ action Player ] -> [ Player ] again');
const actionAgain = analyzeSource(ACTION_AGAIN_GAME, { sourcePath: 'action_again.txt' });
const actionAgainFact = actionAgain.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(actionAgainFact.status, 'rejected');
assert.ok(actionAgainFact.blockers.includes('queues_again'), 'reachable again commands should reject action_noop');
assert.ok(actionAgainFact.blockers.includes('reads_action'));

const ACTION_MOVEMENT_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ action Player ] -> [ > Player ]');
const actionMovement = analyzeSource(ACTION_MOVEMENT_GAME, { sourcePath: 'action_movement.txt' });
const actionMovementFact = actionMovement.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(actionMovementFact.status, 'rejected');
assert.ok(actionMovementFact.blockers.includes('action_may_create_directional_movement'));

const MOVEMENT_CLEAR_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ right Hero ] -> [ Hero ]');
const movementClear = analyzeSource(MOVEMENT_CLEAR_GAME, { sourcePath: 'movement_clear.txt' });
const movementClearRule = movementClear.ps_tagged.rule_sections[0].groups[0].rules[0];
assert.strictEqual(movementClearRule.tags.writes_movement, true, 'clearing movement should count as writing movement');
assert.strictEqual(movementClearRule.tags.movement_only, true);
assert.strictEqual(movementClearRule.tags.object_mutating, false);

const RANDOMDIR_MOVEMENT_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ Hero ] -> [ randomDir Hero ]');
const randomdirMovement = analyzeSource(RANDOMDIR_MOVEMENT_GAME, { sourcePath: 'randomdir_movement.txt' });
const randomdirMovementRule = randomdirMovement.ps_tagged.rule_sections[0].groups[0].rules[0];
assert.strictEqual(randomdirMovement.ps_tagged.game.tags.has_random, false, 'randomDir movement is tracked separately from has_random');
assert.strictEqual(randomdirMovementRule.tags.writes_movement, true, 'randomDir should count as writing movement');
assert.strictEqual(randomdirMovementRule.tags.movement_only, true);
assert.strictEqual(randomdirMovementRule.summary.rhs_movement[0].movement, 'randomdir');

const MOVEMENT_PAIRS_GAME = `
title Movement Pairs
========
OBJECTS
========
Background
black
Player
white
Crate
orange
${'======='}
LEGEND
${'======='}
. = Background
P = Player
C = Crate
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
Crate
=====
RULES
=====
[ action Player ] -> [ right Player ]
[ right Player | Crate ] -> [ right Player | right Crate ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
PC
`;
const movementPairs = analyzeSource(MOVEMENT_PAIRS_GAME, { sourcePath: 'movement_pairs.txt' });
const movementPairsFact = movementPairs.facts.movement_action.find(item => item.id === 'movement_pairs');
assert.deepStrictEqual(movementPairsFact.value, [
    '1:action',
    '1:moving',
    '1:right',
    '2:moving',
    '2:right',
]);

const RANDOMDIR_MOVEMENT_PAIRS_GAME = MOVEMENT_PAIRS_GAME.replace(
    '[ right Player | Crate ] -> [ right Player | right Crate ]',
    '[ right Player | Crate ] -> [ right Player | randomDir Crate ]'
);
const randomdirMovementPairs = analyzeSource(RANDOMDIR_MOVEMENT_PAIRS_GAME, { sourcePath: 'randomdir_movement_pairs.txt' });
const randomdirMovementPairsFact = randomdirMovementPairs.facts.movement_action.find(item => item.id === 'movement_pairs');
assert.deepStrictEqual(randomdirMovementPairsFact.value, [
    '1:action',
    '1:moving',
    '1:right',
    '2:down',
    '2:left',
    '2:moving',
    '2:right',
    '2:up',
]);

const MOVING_REQUIREMENT_PAIRS_GAME = `
title Moving Requirement Pairs
========
OBJECTS
========
Background
black
Player
white
Crate
orange
Goal
green
${'======='}
LEGEND
${'======='}
. = Background
P = Player
C = Crate
G = Goal
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
Crate
Goal
=====
RULES
=====
[ action Player ] -> [ right Crate ]
[ moving Crate ] -> [ right Goal ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
PCG
`;
const movingRequirementPairs = analyzeSource(MOVING_REQUIREMENT_PAIRS_GAME, { sourcePath: 'moving_requirement_pairs.txt' });
const movingRequirementPairsFact = movingRequirementPairs.facts.movement_action.find(item => item.id === 'movement_pairs');
assert.deepStrictEqual(movingRequirementPairsFact.value, [
    '1:action',
    '2:moving',
    '2:right',
    '3:moving',
    '3:right',
]);

const DIRECT_PLAYER_ACTION_GAME = `
title Direct Player Action
========
OBJECTS
========
Background
black
Player
white
${'======='}
LEGEND
${'======='}
. = Background
P = Player
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
=====
RULES
=====
[ action Player ] -> [ right Player ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
P
`;
const directPlayerAction = analyzeSource(DIRECT_PLAYER_ACTION_GAME, { sourcePath: 'direct_player_action.txt' });
const directPlayerActionFact = directPlayerAction.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(directPlayerActionFact.status, 'rejected', 'direct Player objects should seed action reachability');
assert.ok(directPlayerActionFact.blockers.includes('reads_action'));

const STATIONARY_TICK_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ stationary Goal ] -> [ randomDir Goal ]');
const stationaryTick = analyzeSource(STATIONARY_TICK_GAME, { sourcePath: 'stationary_tick.txt' });
const stationaryTickFact = stationaryTick.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(stationaryTick.ps_tagged.game.tags.has_autonomous_tick_rules, true);
assert.strictEqual(stationaryTickFact.status, 'rejected');
assert.ok(stationaryTickFact.blockers.includes('autonomous_solver_active_rule'));
assert.ok(stationaryTickFact.blockers.includes('action_may_create_directional_movement'));

const RANDOM_RULE_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', 'random [ Hero ] -> [ Hero ]');
const randomRule = analyzeSource(RANDOM_RULE_GAME, { sourcePath: 'random_rule.txt' });
const randomRuleIr = randomRule.ps_tagged.rule_sections[0].groups[0].rules[0];
assert.strictEqual(randomRule.ps_tagged.game.tags.has_random, true, 'random rules should tag the game as random');
assert.strictEqual(randomRuleIr.random_rule, true, 'random source rule should be preserved in rule IR');
assert.strictEqual(randomRuleIr.tags.solver_state_active, false, 'unchanged random rule has no solver-state effect');

const RANDOM_OBJECT_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ Hero ] -> [ random Goal ]');
const randomObject = analyzeSource(RANDOM_OBJECT_GAME, { sourcePath: 'random_object.txt' });
const randomObjectRule = randomObject.ps_tagged.rule_sections[0].groups[0].rules[0];
assert.strictEqual(randomObject.ps_tagged.game.tags.has_random, true, 'random RHS objects should tag the game as random');
assert.strictEqual(randomObjectRule.summary.rhs_random_objects.length, 1);
assert.strictEqual(randomObjectRule.tags.object_mutating, true, 'random RHS object writes are object-mutating');

const RIGID_RULE_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', 'rigid [ > Hero | Goal ] -> [ > Hero | > Goal ]');
const rigidRule = analyzeSource(RIGID_RULE_GAME, { sourcePath: 'rigid_rule.txt' });
const rigidRuleIr = rigidRule.ps_tagged.rule_sections[0].groups[0].rules[0];
assert.strictEqual(rigidRule.ps_tagged.game.tags.has_rigid, true, 'rigid rules should tag the game as rigid');
assert.strictEqual(rigidRuleIr.rigid, true);
assert.strictEqual(rigidRuleIr.tags.rigid_active, true, 'solver-active rigid rules should receive rigid_active');

const RIGID_ACTION_GAME = DIRECT_PLAYER_ACTION_GAME.replace(
    '[ action Player ] -> [ right Player ]',
    'rigid [ action Player ] -> [ right Player ]'
);
const rigidAction = analyzeSource(RIGID_ACTION_GAME, { sourcePath: 'rigid_action.txt' });
const rigidActionFact = rigidAction.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(rigidActionFact.status, 'rejected');
assert.ok(rigidActionFact.blockers.includes('rigid_rule'), 'reachable rigid rules should reject action_noop');
assert.ok(rigidActionFact.blockers.includes('reads_action'));

function firstFlowFact(reportToCheck) {
    return reportToCheck.facts.rulegroup_flow.find(item => item.subjects.groups[0] === 'early_group_0');
}

const SPLITTABLE_GROUP_GAME = `
title Splittable Rule Group
========
OBJECTS
========
Background
black
Player
white
Alpha
red
Beta
blue
MarkerX
green
MarkerY
yellow
${'======='}
LEGEND
${'======='}
. = Background
P = Player
a = Alpha
b = Beta
x = MarkerX
y = MarkerY
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
Alpha
Beta
MarkerX
MarkerY
=====
RULES
=====
[ Alpha ] -> [ Alpha MarkerX ]
+ [ Beta ] -> [ Beta MarkerY ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
Pab
`;

const splittableGroup = analyzeSource(SPLITTABLE_GROUP_GAME, { sourcePath: 'splittable_group.txt' });
const splittableFlow = firstFlowFact(splittableGroup);
assert.strictEqual(splittableFlow.status, 'candidate', 'independent plus-group rules should be split candidates');
assert.deepStrictEqual(splittableFlow.value.components.map(component => component.length), [1, 1]);
assert.deepStrictEqual(splittableFlow.value.interaction_edges, []);
assert.deepStrictEqual(Object.values(splittableFlow.value.rerun_masks), [[], []]);

const MOVEMENT_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ right Alpha ] -> [ Alpha ]\n+ [ Alpha ] -> [ right Alpha ]\n+ [ Beta ] -> [ Beta MarkerY ]'
);
const movementEnableGroup = analyzeSource(MOVEMENT_ENABLE_GROUP_GAME, { sourcePath: 'movement_enable_group.txt' });
const movementFlow = firstFlowFact(movementEnableGroup);
assert.deepStrictEqual(movementFlow.value.interaction_edges.map(edge => [edge.from, edge.to, edge.reasons]), [
    ['early_group_0_rule_1', 'early_group_0_rule_0', ['movement']],
]);
assert.deepStrictEqual(movementFlow.value.rerun_masks.early_group_0_rule_1, ['early_group_0_rule_0']);
assert.deepStrictEqual(movementFlow.value.components.map(component => component.length), [2, 1]);

function assertBackwardMovementInteraction(flow) {
    const ruleIndex = ruleId => Number(ruleId.match(/_rule_(\d+)$/)[1]);
    const edge = flow.value.interaction_edges.find(candidate =>
        candidate.reasons.includes('movement') && ruleIndex(candidate.to) < ruleIndex(candidate.from)
    );
    assert.ok(edge, 'flow should contain a backward movement edge');
    assert.ok(ruleIndex(edge.to) < ruleIndex(edge.from), 'movement edge should target an earlier rule');
    assert.ok(flow.value.rerun_masks[edge.from].includes(edge.to), 'backward movement edge should add a rerun mask entry');
    assert.ok(flow.value.components.some(component =>
        component.includes(edge.from) && component.includes(edge.to)
    ), 'interacting rules should share a connected component');
}

const MOVING_READ_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ moving Alpha ] -> [ Alpha MarkerX ]\n+ [ Alpha ] -> [ right Alpha ]\n+ [ Beta ] -> [ Beta MarkerY ]'
);
const movingReadGroup = analyzeSource(MOVING_READ_GROUP_GAME, { sourcePath: 'moving_read_group.txt' });
const movingReadFlow = firstFlowFact(movingReadGroup);
assertBackwardMovementInteraction(movingReadFlow);

const RANDOMDIR_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ left Alpha ] -> [ Alpha MarkerX ]\n+ [ Alpha ] -> [ randomDir Alpha ]\n+ [ Beta ] -> [ Beta MarkerY ]'
);
const randomdirEnableGroup = analyzeSource(RANDOMDIR_ENABLE_GROUP_GAME, { sourcePath: 'randomdir_enable_group.txt' });
const randomdirFlow = firstFlowFact(randomdirEnableGroup);
assertBackwardMovementInteraction(randomdirFlow);

const STATIONARY_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ stationary Alpha ] -> [ Alpha MarkerX ]\n+ [ right Alpha ] -> [ Alpha ]\n+ [ Beta ] -> [ Beta MarkerY ]'
);
const stationaryEnableGroup = analyzeSource(STATIONARY_ENABLE_GROUP_GAME, { sourcePath: 'stationary_enable_group.txt' });
const stationaryFlow = firstFlowFact(stationaryEnableGroup);
assertBackwardMovementInteraction(stationaryFlow);

const ABSENCE_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ no MarkerX Alpha ] -> [ no MarkerX Alpha MarkerY ]\n+ [ MarkerX ] -> []'
);
const absenceEnableGroup = analyzeSource(ABSENCE_ENABLE_GROUP_GAME, { sourcePath: 'absence_enable_group.txt' });
const absenceFlow = firstFlowFact(absenceEnableGroup);
assert.strictEqual(absenceFlow.status, 'rejected', 'absence-enabling interactions should keep the group connected');
assert.deepStrictEqual(absenceFlow.value.interaction_edges.map(edge => [edge.from, edge.to, edge.reasons]), [
    ['early_group_0_rule_1', 'early_group_0_rule_0', ['object_absence']],
]);
assert.deepStrictEqual(absenceFlow.value.rerun_masks.early_group_0_rule_1, ['early_group_0_rule_0']);

const BACKWARD_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ MarkerX ] -> [ MarkerX MarkerY ]\n+ [ Alpha ] -> [ Alpha MarkerX ]'
);
const backwardEnableGroup = analyzeSource(BACKWARD_ENABLE_GROUP_GAME, { sourcePath: 'backward_enable_group.txt' });
const backwardFlow = firstFlowFact(backwardEnableGroup);
assert.strictEqual(backwardFlow.status, 'rejected', 'backward enabling interaction should keep the group connected');
assert.deepStrictEqual(backwardFlow.value.interaction_edges.map(edge => [edge.from, edge.to, edge.reasons]), [
    ['early_group_0_rule_1', 'early_group_0_rule_0', ['object_presence']],
]);
assert.deepStrictEqual(backwardFlow.value.rerun_masks.early_group_0_rule_1, ['early_group_0_rule_0']);

const FORWARD_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ MarkerX ] -> [ MarkerX MarkerY ]'
);
const forwardEnableGroup = analyzeSource(FORWARD_ENABLE_GROUP_GAME, { sourcePath: 'forward_enable_group.txt' });
const forwardFlow = firstFlowFact(forwardEnableGroup);
assert.deepStrictEqual(forwardFlow.value.interaction_edges.map(edge => [edge.from, edge.to, edge.reasons]), [
    ['early_group_0_rule_0', 'early_group_0_rule_1', ['object_presence']],
]);
assert.deepStrictEqual(forwardFlow.value.rerun_masks.early_group_0_rule_0, [], 'forward interactions do not require next-iteration reruns');

const AGGREGATE_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    'y = MarkerY',
    'y = MarkerY\nThing = MarkerX or MarkerY'
).replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ Thing ] -> [ Thing Player ]\n+ [ Alpha ] -> [ Alpha MarkerX ]'
);
const aggregateEnableGroup = analyzeSource(AGGREGATE_ENABLE_GROUP_GAME, { sourcePath: 'aggregate_enable_group.txt' });
const aggregateFlow = firstFlowFact(aggregateEnableGroup);
assert.ok(
    Object.values(aggregateFlow.value.rerun_masks).some(mask => mask.includes('early_group_0_rule_0')),
    'object-set aggregate reads should be expanded for rerun masks'
);

const AND_AGGREGATE_ENABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    'y = MarkerY',
    'y = MarkerY\nPair = MarkerX and MarkerY'
).replace(
    '[ Alpha ] -> [ Alpha MarkerX ]\n+ [ Beta ] -> [ Beta MarkerY ]',
    '[ Pair ] -> [ Pair Player ]\n+ [ Alpha ] -> [ Alpha MarkerX ]'
);
const andAggregateEnableGroup = analyzeSource(AND_AGGREGATE_ENABLE_GROUP_GAME, { sourcePath: 'and_aggregate_enable_group.txt' });
const andAggregateFlow = firstFlowFact(andAggregateEnableGroup);
assert.ok(
    Object.values(andAggregateFlow.value.rerun_masks).some(mask => mask.includes('early_group_0_rule_0')),
    'and-aggregate reads should be expanded for rerun masks'
);

const RIGID_SPLITTABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]',
    'rigid [ Alpha ] -> [ Alpha MarkerX ]'
);
const rigidSplittableGroup = analyzeSource(RIGID_SPLITTABLE_GROUP_GAME, { sourcePath: 'rigid_splittable_group.txt' });
const rigidSplittableFlow = firstFlowFact(rigidSplittableGroup);
assert.strictEqual(rigidSplittableFlow.status, 'rejected', 'rigid groups should not be split candidates');
assert.ok(rigidSplittableFlow.blockers.includes('rigid_rule'));

const RANDOM_SPLITTABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]',
    'random [ Alpha ] -> [ Alpha MarkerX ]'
);
const randomSplittableGroup = analyzeSource(RANDOM_SPLITTABLE_GROUP_GAME, { sourcePath: 'random_splittable_group.txt' });
const randomSplittableFlow = firstFlowFact(randomSplittableGroup);
assert.strictEqual(randomSplittableFlow.status, 'rejected', 'random groups should not be split candidates');
assert.ok(randomSplittableFlow.blockers.includes('random_rule_group'));

const SEMANTIC_SPLITTABLE_GROUP_GAME = SPLITTABLE_GROUP_GAME.replace(
    '[ Alpha ] -> [ Alpha MarkerX ]',
    '[ Alpha ] -> [ Alpha MarkerX ] checkpoint'
);
const semanticSplittableGroup = analyzeSource(SEMANTIC_SPLITTABLE_GROUP_GAME, { sourcePath: 'semantic_splittable_group.txt' });
const semanticSplittableFlow = firstFlowFact(semanticSplittableGroup);
assert.strictEqual(semanticSplittableFlow.status, 'rejected', 'semantic commands should block split candidates');
assert.ok(semanticSplittableFlow.blockers.includes('semantic_command'));

const PUSHER_STYLE_GROUP_GAME = `
title Pusher Style Rule Group
========
OBJECTS
========
Background
black
Pusher
white
Pushable
orange
${'======='}
LEGEND
${'======='}
. = Background
P = Pusher
C = Pushable
Player = Pusher
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Pusher
Pushable
=====
RULES
=====
down [ Pushable | up Pusher ] -> [ up Pushable | up Pusher ]
+ down [ down Pusher | Pushable ] -> [ down Pusher | down Pushable ]
+ right [ Pushable | left Pusher ] -> [ left Pushable | left Pusher ]
+ right [ right Pusher | Pushable ] -> [ right Pusher | right Pushable ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
PC
`;
const pusherStyleGroup = analyzeSource(PUSHER_STYLE_GROUP_GAME, { sourcePath: 'pusher_style_group.txt' });
const pusherFlow = firstFlowFact(pusherStyleGroup);
assert.strictEqual(pusherFlow.status, 'candidate', 'pusher-style movement rules should not be blocked by shared layer writes');
assert.deepStrictEqual(pusherFlow.value.components.map(component => component.length), [1, 1, 1, 1]);
assert.deepStrictEqual(Object.values(pusherFlow.value.rerun_masks), [[], [], [], []]);

const countFacts = report.facts.count_layer_invariants;
assert.ok(countFacts.some(item => item.id === 'object_Hero_count_preserved'), 'Hero count fact should exist');
assert.ok(countFacts.some(item => item.id === 'layer_0_static'), 'Background layer static fact should exist');

const randomGoalCount = randomObject.facts.count_layer_invariants.find(item => item.id === 'object_Goal_count_preserved');
const randomHeroCount = randomObject.facts.count_layer_invariants.find(item => item.id === 'object_Hero_count_preserved');
const randomGoalTags = randomObject.ps_tagged.objects.find(object => object.name === 'Goal').tags;
assert.strictEqual(randomGoalCount.status, 'rejected', 'random RHS object writes should reject target count preservation');
assert.strictEqual(randomHeroCount.status, 'rejected', 'random same-layer writes should reject sibling count preservation');
assert.ok(randomGoalCount.blockers.includes('object_written_by_solver_active_rule'));
assert.strictEqual(randomGoalTags.count_invariant, false);

const SPAWN_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ Hero ] -> [ Hero Goal ]');
const spawnReport = analyzeSource(SPAWN_GAME, { sourcePath: 'spawn.txt' });
const goalCount = spawnReport.facts.count_layer_invariants.find(item => item.id === 'object_Goal_count_preserved');
const spawnedGoalTags = spawnReport.ps_tagged.objects.find(object => object.name === 'Goal').tags;
assert.strictEqual(goalCount.status, 'rejected');
assert.ok(goalCount.blockers.includes('object_written_by_solver_active_rule'));
assert.strictEqual(spawnedGoalTags.may_be_created, true);
assert.strictEqual(spawnedGoalTags.may_be_destroyed, true, 'object write tags are currently conservative both-way churn tags');
assert.strictEqual(spawnedGoalTags.count_invariant, false);

const DESTROY_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ Hero ] -> []');
const destroyReport = analyzeSource(DESTROY_GAME, { sourcePath: 'destroy.txt' });
const heroDestroyCount = destroyReport.facts.count_layer_invariants.find(item => item.id === 'object_Hero_count_preserved');
const destroyedHeroTags = destroyReport.ps_tagged.objects.find(object => object.name === 'Hero').tags;
assert.strictEqual(heroDestroyCount.status, 'rejected', 'deleting an object to an empty RHS should reject count preservation');
assert.strictEqual(destroyedHeroTags.may_be_created, true, 'object write tags are currently conservative both-way churn tags');
assert.strictEqual(destroyedHeroTags.may_be_destroyed, true);
assert.strictEqual(destroyedHeroTags.count_invariant, false);

const LAYER_OVERWRITE_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ no Goal ] -> [ Goal ]');
const layerOverwriteReport = analyzeSource(LAYER_OVERWRITE_GAME, { sourcePath: 'layer_overwrite.txt' });
const heroOverwriteCount = layerOverwriteReport.facts.count_layer_invariants.find(item => item.id === 'object_Hero_count_preserved');
const overwrittenHeroTags = layerOverwriteReport.ps_tagged.objects.find(object => object.name === 'Hero').tags;
assert.strictEqual(heroOverwriteCount.status, 'rejected', 'writing one object in a collision layer can remove siblings');
assert.strictEqual(overwrittenHeroTags.may_be_created, true, 'same-layer writes can overwrite siblings');
assert.strictEqual(overwrittenHeroTags.may_be_destroyed, true);

const STATIC_OBJECT_GAME = `
title Static Object
========
OBJECTS
========
Background
black
PlayerObject
white
Wall
gray
Crate
orange
Mark
red
${'======='}
LEGEND
${'======='}
. = Background
P = PlayerObject
# = Wall
C = Crate
Player = PlayerObject
Solid = Wall or Crate
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
PlayerObject
Wall, Crate
Mark
=====
RULES
=====
[ > PlayerObject ] -> [ > PlayerObject ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
P#C
`;

const staticObject = analyzeSource(STATIC_OBJECT_GAME, { sourcePath: 'static_object.txt' });
const staticWall = staticObject.ps_tagged.objects.find(object => object.name === 'Wall');
const staticPlayer = staticObject.ps_tagged.objects.find(object => object.name === 'PlayerObject');
const staticWallFact = staticObject.facts.count_layer_invariants.find(item => item.id === 'object_Wall_static');
const staticPlayerLayerFact = staticObject.facts.count_layer_invariants.find(item => item.id === 'layer_1_static');
assert.strictEqual(staticWall.tags.static, true, 'unwritten, unmoved wall should be tagged static');
assert.strictEqual(staticWallFact.status, 'proved', 'unwritten, unmoved wall should have a proved static fact');
assert.strictEqual(staticPlayer.tags.count_invariant, true, 'player object count can be invariant');
assert.strictEqual(staticPlayer.tags.static, false, 'player object is not static because input applies movement');
assert.strictEqual(staticPlayerLayerFact.status, 'candidate', 'a layer containing a moving player is not proved static');
assert.ok(staticPlayerLayerFact.blockers.includes('layer_contains_nonstatic_object'));

const STATIC_LAYER_CREATE_GAME = STATIC_OBJECT_GAME.replace(
    '[ > PlayerObject ] -> [ > PlayerObject ]',
    '[ PlayerObject ] -> [ PlayerObject Crate ]'
);
const staticLayerCreate = analyzeSource(STATIC_LAYER_CREATE_GAME, { sourcePath: 'static_layer_create.txt' });
const layerCreateWall = staticLayerCreate.ps_tagged.objects.find(object => object.name === 'Wall');
const layerCreateWallFact = staticLayerCreate.facts.count_layer_invariants.find(item => item.id === 'object_Wall_static');
const layerCreateWallLayerFact = staticLayerCreate.facts.count_layer_invariants.find(item => item.id === 'layer_2_static');
assert.strictEqual(layerCreateWall.tags.static, false, 'creating a sibling on the wall layer should reject wall static');
assert.ok(layerCreateWallFact.blockers.includes('collision_layer_object_may_be_created'));
assert.strictEqual(layerCreateWallLayerFact.status, 'candidate', 'layers with possible writes should not be proved static');
assert.ok(layerCreateWallLayerFact.blockers.includes('layer_objects_may_change'));
assert.ok(layerCreateWallLayerFact.blockers.includes('layer_contains_nonstatic_object'));

const STATIC_AGGREGATE_WRITE_GAME = STATIC_OBJECT_GAME.replace(
    '[ > PlayerObject ] -> [ > PlayerObject ]',
    '[ Solid ] -> []'
);
const staticAggregateWrite = analyzeSource(STATIC_AGGREGATE_WRITE_GAME, { sourcePath: 'static_aggregate_write.txt' });
const aggregateWriteWall = staticAggregateWrite.ps_tagged.objects.find(object => object.name === 'Wall');
const aggregateWriteWallFact = staticAggregateWrite.facts.count_layer_invariants.find(item => item.id === 'object_Wall_static');
const aggregateWriteWallCount = staticAggregateWrite.facts.count_layer_invariants.find(item => item.id === 'object_Wall_count_preserved');
assert.strictEqual(aggregateWriteWall.tags.static, false, 'aggregate deletion mentioning wall should reject wall static');
assert.strictEqual(aggregateWriteWall.tags.count_invariant, false, 'aggregate deletion mentioning wall should reject wall count preservation');
assert.ok(aggregateWriteWallFact.blockers.includes('object_written_by_solver_active_rule'));
assert.ok(aggregateWriteWallCount.blockers.includes('object_written_by_solver_active_rule'));

const STATIC_AND_AGGREGATE_WRITE_GAME = STATIC_OBJECT_GAME
    .replace('Solid = Wall or Crate', 'Solid = Wall or Crate\nPair = Wall and Mark')
    .replace('[ > PlayerObject ] -> [ > PlayerObject ]', '[ Pair ] -> []');
const staticAndAggregateWrite = analyzeSource(STATIC_AND_AGGREGATE_WRITE_GAME, { sourcePath: 'static_and_aggregate_write.txt' });
const andAggregateWriteWall = staticAndAggregateWrite.ps_tagged.objects.find(object => object.name === 'Wall');
const andAggregateWriteWallFact = staticAndAggregateWrite.facts.count_layer_invariants.find(item => item.id === 'object_Wall_static');
const andAggregateWriteWallCount = staticAndAggregateWrite.facts.count_layer_invariants.find(item => item.id === 'object_Wall_count_preserved');
assert.strictEqual(andAggregateWriteWall.tags.static, false, 'and-aggregate deletion mentioning wall should reject wall static');
assert.strictEqual(andAggregateWriteWall.tags.count_invariant, false, 'and-aggregate deletion mentioning wall should reject wall count preservation');
assert.ok(andAggregateWriteWallFact.blockers.includes('object_written_by_solver_active_rule'));
assert.ok(andAggregateWriteWallCount.blockers.includes('object_written_by_solver_active_rule'));

const STATIC_AGGREGATE_MOVEMENT_GAME = STATIC_OBJECT_GAME.replace(
    '[ > PlayerObject ] -> [ > PlayerObject ]',
    '[ Solid ] -> [ right Solid ]'
);
const staticAggregateMovement = analyzeSource(STATIC_AGGREGATE_MOVEMENT_GAME, { sourcePath: 'static_aggregate_movement.txt' });
const aggregateMovementWall = staticAggregateMovement.ps_tagged.objects.find(object => object.name === 'Wall');
const aggregateMovementWallFact = staticAggregateMovement.facts.count_layer_invariants.find(item => item.id === 'object_Wall_static');
assert.strictEqual(aggregateMovementWall.tags.count_invariant, true, 'aggregate movement preserves wall count');
assert.strictEqual(aggregateMovementWall.tags.static, false, 'aggregate movement mentioning wall should reject wall static');
assert.ok(aggregateMovementWallFact.blockers.includes('object_may_receive_movement'));

const STATIC_AND_AGGREGATE_MOVEMENT_GAME = STATIC_OBJECT_GAME
    .replace('Solid = Wall or Crate', 'Solid = Wall or Crate\nPair = Wall and Mark')
    .replace('[ > PlayerObject ] -> [ > PlayerObject ]', '[ Pair ] -> [ right Pair ]');
const staticAndAggregateMovement = analyzeSource(STATIC_AND_AGGREGATE_MOVEMENT_GAME, { sourcePath: 'static_and_aggregate_movement.txt' });
const andAggregateMovementWall = staticAndAggregateMovement.ps_tagged.objects.find(object => object.name === 'Wall');
const andAggregateMovementWallFact = staticAndAggregateMovement.facts.count_layer_invariants.find(item => item.id === 'object_Wall_static');
assert.strictEqual(andAggregateMovementWall.tags.count_invariant, true, 'and-aggregate movement preserves wall count');
assert.strictEqual(andAggregateMovementWall.tags.static, false, 'and-aggregate movement mentioning wall should reject wall static');
assert.ok(andAggregateMovementWallFact.blockers.includes('object_may_receive_movement'));

const TRANSIENT_GAME = `
title Transient
========
OBJECTS
========
Background
black
Player
white
Mark
red
${'======='}
LEGEND
${'======='}
. = Background
P = Player
M = Mark
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
Mark
=====
RULES
=====
[ Player ] -> [ Player Mark ]
late [ Mark ] -> [ no Mark ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
P
`;

const transient = analyzeSource(TRANSIENT_GAME, { sourcePath: 'transient.txt' });
const markTransient = transient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(markTransient.status, 'proved');
assert.strictEqual(markTransient.tags.single_turn_only, true);

const EMPTY_CLEAR_TRANSIENT_GAME = TRANSIENT_GAME.replace('late [ Mark ] -> [ no Mark ]', 'late [ Mark ] -> []');
const emptyClearTransient = analyzeSource(EMPTY_CLEAR_TRANSIENT_GAME, { sourcePath: 'empty_clear_transient.txt' });
const emptyClearMark = emptyClearTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(emptyClearMark.status, 'proved', 'empty RHS cleanup should count as an end-turn clear');

const PRESENT_TRANSIENT_GAME = TRANSIENT_GAME.replace('P\n', 'PM\n');
const presentTransient = analyzeSource(PRESENT_TRANSIENT_GAME, { sourcePath: 'present_transient.txt' });
const presentMark = presentTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(presentMark.status, 'rejected', 'initially present objects should not be proved end-turn transient');
assert.ok(presentMark.blockers.includes('present_in_some_initial_levels'));

const WINCONDITION_TRANSIENT_GAME = TRANSIENT_GAME.replace('Some Player', 'Some Mark');
const winconditionTransient = analyzeSource(WINCONDITION_TRANSIENT_GAME, { sourcePath: 'wincondition_transient.txt' });
const winconditionMark = winconditionTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(winconditionMark.status, 'rejected', 'win-condition objects should not be proved end-turn transient');
assert.ok(winconditionMark.blockers.includes('appears_in_wincondition'));

const NO_CLEANUP_TRANSIENT_GAME = TRANSIENT_GAME.replace('late [ Mark ] -> [ no Mark ]\n', '');
const noCleanupTransient = analyzeSource(NO_CLEANUP_TRANSIENT_GAME, { sourcePath: 'no_cleanup_transient.txt' });
const noCleanupMark = noCleanupTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(noCleanupMark.status, 'rejected');
assert.ok(noCleanupMark.blockers.includes('no_late_cleanup_clear'));

const NO_CREATOR_TRANSIENT_GAME = TRANSIENT_GAME.replace('[ Player ] -> [ Player Mark ]\n', '');
const noCreatorTransient = analyzeSource(NO_CREATOR_TRANSIENT_GAME, { sourcePath: 'no_creator_transient.txt' });
const noCreatorMark = noCreatorTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(noCreatorMark.status, 'rejected');
assert.ok(noCreatorMark.blockers.includes('not_created_before_end_cleanup'));

const CREATOR_AFTER_CLEANUP_TRANSIENT_GAME = TRANSIENT_GAME
    .replace('[ Player ] -> [ Player Mark ]\n', '')
    .replace('late [ Mark ] -> [ no Mark ]', 'late [ Mark ] -> [ no Mark ]\nlate [ Player ] -> [ Player Mark ]');
const creatorAfterCleanupTransient = analyzeSource(CREATOR_AFTER_CLEANUP_TRANSIENT_GAME, { sourcePath: 'creator_after_cleanup_transient.txt' });
const creatorAfterCleanupMark = creatorAfterCleanupTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(creatorAfterCleanupMark.status, 'rejected', 'late creators after cleanup should not prove end-turn transience');
assert.ok(creatorAfterCleanupMark.blockers.includes('creator_not_followed_by_late_cleanup'));

const AGAIN_PRESERVE_TRANSIENT_GAME = TRANSIENT_GAME.replace(
    '[ Player ] -> [ Player Mark ]',
    '[ Player no Mark ] -> [ Player Mark ]\n[ Player Mark ] -> [ Player Mark ] again'
);
const againPreserveTransient = analyzeSource(AGAIN_PRESERVE_TRANSIENT_GAME, { sourcePath: 'again_preserve_transient.txt' });
const againPreserveMark = againPreserveTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(againPreserveMark.status, 'proved', 'rules that only preserve a transient object should not count as creators');

const AGAIN_TAINT_GAME = TRANSIENT_GAME.replace('[ Player ] -> [ Player Mark ]', '[ Player ] -> [ Player Mark ] again');
const againTaint = analyzeSource(AGAIN_TAINT_GAME, { sourcePath: 'again_taint.txt' });
const againMark = againTaint.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(againMark.status, 'rejected');
assert.ok(againMark.blockers.includes('has_again_taint'));

const SIBLING_AGAIN_TAINT_GAME = TRANSIENT_GAME.replace(
    '[ Player ] -> [ Player Mark ]',
    '[ Player no Mark ] -> [ Player Mark ]\n+ [ Player Mark ] -> [ Player Mark ] again'
);
const siblingAgainTaint = analyzeSource(SIBLING_AGAIN_TAINT_GAME, { sourcePath: 'sibling_again_taint.txt' });
const siblingAgainMark = siblingAgainTaint.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(siblingAgainMark.status, 'rejected', 'same-group again should taint transient creators');
assert.ok(siblingAgainMark.blockers.includes('has_again_taint'));

const RIGID_TRANSIENT_GAME = TRANSIENT_GAME.replace('[ Player ] -> [ Player Mark ]', 'rigid [ Player ] -> [ Player Mark ]');
const rigidTransient = analyzeSource(RIGID_TRANSIENT_GAME, { sourcePath: 'rigid_transient.txt' });
const rigidMark = rigidTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(rigidMark.status, 'rejected', 'rigid transient creators should not produce proved transient facts');
assert.ok(rigidMark.blockers.includes('rigid_rule'));

const LATE_CHAIN_TRANSIENT_GAME = `
title Late Chain Transient
========
OBJECTS
========
Background
black
Player
white
Door
red
Mark
pink
${'======='}
LEGEND
${'======='}
. = Background
P = Player
D = Door
M = Mark
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Player
Door, Mark
=====
RULES
=====
late [ Door ] -> [ Mark ]
late [ Mark ] -> [ Door ]
=============
WINCONDITIONS
=============
Some Player
======
LEVELS
======
PD
`;
const lateChainTransient = analyzeSource(LATE_CHAIN_TRANSIENT_GAME, { sourcePath: 'late_chain_transient.txt' });
const lateChainMark = lateChainTransient.facts.transient_boundary.find(item => item.id === 'object_Mark_end_turn_transient');
assert.strictEqual(lateChainMark.status, 'proved', 'late-created objects cleared by a later late rule should be end-turn transient');

const noTagged = analyzeSource(SIMPLE_GAME, { sourcePath: 'simple.txt', includePsTagged: false });
assert.strictEqual(noTagged.ps_tagged, undefined, 'includePsTagged=false should remove ps_tagged');

const onlyMerge = analyzeSource(SIMPLE_GAME, { sourcePath: 'simple.txt', familyFilter: 'mergeability' });
assert.deepStrictEqual(Object.keys(onlyMerge.facts), ['mergeability'], 'familyFilter should keep one family');

const TOO_MANY_ERRORS_GAME = SIMPLE_GAME.replace('P.G\n...', Array.from({ length: 110 }, () => 'Z').join('\n'));
const tooManyErrors = analyzeSource(TOO_MANY_ERRORS_GAME, { sourcePath: 'too_many_errors.txt' });
assert.strictEqual(tooManyErrors.status, 'compile_error');
assert.ok(tooManyErrors.errors.length > 1, 'too many errors should preserve partial diagnostics');

function assertProvedFactsHaveProof(reportToCheck) {
    for (const familyFacts of Object.values(reportToCheck.facts)) {
        for (const item of familyFacts) {
            if (item.status === 'proved') {
                assert.ok(Array.isArray(item.proof) && item.proof.length > 0, `${item.id} should have proof`);
                assert.ok(Array.isArray(item.evidence), `${item.id} should have evidence array`);
            }
        }
    }
}

assertProvedFactsHaveProof(report);
assertProvedFactsHaveProof(mergeable);
assertProvedFactsHaveProof(transient);

let puzzleScriptRuntimeLoaded = false;

function ensurePuzzleScriptRuntime() {
    if (!puzzleScriptRuntimeLoaded) {
        loadPuzzleScript({ messageSink: [] });
        puzzleScriptRuntimeLoaded = true;
    }
}

function engineObjectName(displayName) {
    const target = displayName.toLowerCase();
    const match = Object.keys(state.objects).find(name =>
        name === target || (state.original_case_names && state.original_case_names[name] === displayName)
    );
    assert.ok(match, `runtime object should exist: ${displayName}`);
    return match;
}

function objectOccupancySnapshot(displayName) {
    const object = state.objects[engineObjectName(displayName)];
    const cells = [];
    for (let index = 0; index < level.n_tiles; index++) {
        cells.push(level.getCell(index).get(object.id) ? 1 : 0);
    }
    return cells;
}

function drainAgainRuntime() {
    while (againing) {
        againing = false;
        processInput(-1);
    }
}

function compileRuntimeSource(source, options = {}) {
    const levelIndex = options.levelIndex || 0;
    const seed = options.seed || 'static-analysis-runtime';
    ensurePuzzleScriptRuntime();
    if (typeof resetParserErrorState === 'function') {
        resetParserErrorState();
    }
    compile(['loadLevel', levelIndex], `${source}\n`, seed);
    assert.strictEqual(errorCount, 0, errorStrings.map(stripHTMLTags).join('\n'));
    drainAgainRuntime();
}

function withRuntimeSource(source, callback, options = {}) {
    ensurePuzzleScriptRuntime();
    const previousUnitTesting = unitTesting;
    const previousLazyFunctionGeneration = lazyFunctionGeneration;
    unitTesting = false;
    lazyFunctionGeneration = false;
    try {
        compileRuntimeSource(source, options);
        callback();
    } finally {
        unitTesting = previousUnitTesting;
        lazyFunctionGeneration = previousLazyFunctionGeneration;
    }
}

function inputForToken(token) {
    const inputByToken = { up: 0, left: 1, down: 2, right: 3, action: 4 };
    assert.ok(Object.prototype.hasOwnProperty.call(inputByToken, token), `known input token: ${token}`);
    return inputByToken[token];
}

function inputsForTokens(tokens) {
    return tokens.map(inputForToken);
}

function processRuntimeInputs(inputs, afterTurn) {
    inputs.forEach((input, index) => {
        if (afterTurn) afterTurn(index, 'before');
        processInput(input);
        drainAgainRuntime();
        if (afterTurn) afterTurn(index, 'after');
    });
}

function runtimeObjectId(displayName) {
    return state.objects[engineObjectName(displayName)].id;
}

function runtimeObjectCount(displayName) {
    return objectOccupancySnapshot(displayName).reduce((sum, present) => sum + present, 0);
}

function levelSolverStateSnapshot() {
    return {
        objects: Array.from(level.objects),
        movements: Array.from(level.movements),
        rigidGroupIndexMask: Array.from(level.rigidGroupIndexMask || []),
        rigidMovementAppliedMask: Array.from(level.rigidMovementAppliedMask || []),
    };
}

function splitPlusContinuationRules(source) {
    return source.replace(/^(\s*)\+\s+/gm, '$1');
}

function rewriteRuntimeObjects(objects, turnIndex) {
    const ids = objects.map(runtimeObjectId);
    for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
        const cell = level.getCell(cellIndex);
        const hasGroupObject = ids.some(id => cell.get(id));
        if (!hasGroupObject) continue;
        for (const id of ids) {
            cell.ibitclear(id);
        }
        cell.ibitset(ids[(cellIndex + turnIndex) % ids.length]);
        level.setCell(cellIndex, cell);
    }
    state.calculateRowColMasks(level);
}

function layerOccupancySnapshot(layerId) {
    const objectNames = Array.from(state.collisionLayers[layerId] || []);
    const snapshots = [];
    for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
        const cell = level.getCell(cellIndex);
        snapshots.push(objectNames
            .filter(objectName => cell.get(state.objects[objectName].id))
            .sort()
            .join('|'));
    }
    return snapshots;
}

function assertCountInvariantsPreservedAfterReplay(source, inputs, options = {}) {
    const analysis = analyzeSource(source, { sourcePath: options.sourcePath || 'count_runtime.txt' });
    const transientObjects = new Set(analysis.facts.transient_boundary
        .filter(item => item.status === 'proved')
        .map(item => item.subjects.objects[0]));
    const countInvariantObjects = analysis.ps_tagged.objects
        .filter(object => object.tags.count_invariant && !transientObjects.has(object.name))
        .map(object => object.name);
    assert.ok(countInvariantObjects.length > 0, 'runtime fixture should prove count-invariant objects');
    for (const objectName of options.expectedObjects || []) {
        assert.ok(
            countInvariantObjects.includes(objectName),
            `${objectName} should be included in the count-invariant replay set`
        );
    }

    withRuntimeSource(source, () => {
        const before = new Map(countInvariantObjects.map(objectName => [
            objectName,
            runtimeObjectCount(objectName),
        ]));
        const assertCountsUnchanged = label => {
            for (const objectName of countInvariantObjects) {
                assert.strictEqual(
                    runtimeObjectCount(objectName),
                    before.get(objectName),
                    `${objectName} count should not change at ${label}`
                );
            }
        };
        assertCountsUnchanged('initial boundary');
        processRuntimeInputs(inputs, (turnIndex, boundary) => {
            assertCountsUnchanged(`turn ${turnIndex} ${boundary}`);
        });
        if (options.expectWinning) {
            assert.strictEqual(winning, true, 'known replay should solve while count invariants hold');
        }
    }, { levelIndex: options.levelIndex || 0 });
}

function assertActionNoopAfterReplay(source, inputs, options = {}) {
    const analysis = analyzeSource(source, { sourcePath: options.sourcePath || 'action_noop_runtime.txt' });
    const actionNoop = analysis.facts.movement_action.find(item => item.id === 'action_noop');
    assert.ok(actionNoop, 'runtime fixture should emit action_noop fact');
    assert.strictEqual(actionNoop.status, 'proved', 'runtime fixture should prove action_noop before replay checks');

    withRuntimeSource(source, () => {
        const assertActionDoesNothing = label => {
            const before = levelSolverStateSnapshot();
            const modified = processInput(inputForToken('action'));
            drainAgainRuntime();
            assert.strictEqual(modified, false, `action should report no modification at ${label}`);
            assert.deepStrictEqual(
                levelSolverStateSnapshot(),
                before,
                `action should leave solver-visible state unchanged at ${label}`
            );
        };
        assertActionDoesNothing('initial boundary');
        inputs.forEach((input, turnIndex) => {
            processInput(input);
            drainAgainRuntime();
            assertActionDoesNothing(`turn ${turnIndex} after`);
        });
        if (options.expectWinning) {
            assert.strictEqual(winning, true, 'known replay should solve while action is noop');
        }
    }, { levelIndex: options.levelIndex || 0 });
}

function replayRuntimeSnapshot(source, inputs, options = {}) {
    let snapshot;
    withRuntimeSource(source, () => {
        processRuntimeInputs(inputs);
        snapshot = {
            winning,
            solver_state: levelSolverStateSnapshot(),
        };
    }, { levelIndex: options.levelIndex || 0 });
    return snapshot;
}

function assertStaticObjectsUnchangedAfterReplay(source, inputs) {
    const analysis = analyzeSource(source, { sourcePath: 'static_runtime.txt' });
    const staticObjects = analysis.ps_tagged.objects
        .filter(object => object.tags.static)
        .map(object => object.name);
    assert.ok(staticObjects.includes('Wall'), 'runtime fixture should prove Wall static');

    ensurePuzzleScriptRuntime();
    const previousUnitTesting = unitTesting;
    const previousLazyFunctionGeneration = lazyFunctionGeneration;
    unitTesting = false;
    lazyFunctionGeneration = false;
    try {
        compile(['loadLevel', 0], `${source}\n`, 'static-object-runtime');
        drainAgainRuntime();
        const before = new Map(staticObjects.map(objectName => [objectName, objectOccupancySnapshot(objectName)]));
        for (const input of inputs) {
            processInput(input);
            drainAgainRuntime();
        }
        assert.strictEqual(winning, true, 'known replay should solve the fixture');
        for (const objectName of staticObjects) {
            assert.deepStrictEqual(
                objectOccupancySnapshot(objectName),
                before.get(objectName),
                `${objectName} occupancy should not change during known replay`
            );
        }
    } finally {
        unitTesting = previousUnitTesting;
        lazyFunctionGeneration = previousLazyFunctionGeneration;
    }
}

function assertMergeabilitySwapsPreserveReplay() {
    const source = `
title Mergeability Runtime
========
OBJECTS
========
background
black
player
white
bodyh
red
bodyv
blue
goal
green
${'======='}
LEGEND
${'======='}
. = background
P = player
h = bodyh
v = bodyv
G = goal
body = bodyh or bodyv
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
background
goal
player
bodyh, bodyv
=====
RULES
=====
[ > player | body ] -> [ > player | > body ]
=============
WINCONDITIONS
=============
all goal on body
======
LEVELS
======
P.hG
`;
    const analysis = analyzeSource(source, { sourcePath: 'mergeability_runtime.txt' });
    const mergeFact = analysis.facts.mergeability.find(item => item.subjects.objects.join(',') === 'bodyh,bodyv');
    assert.ok(mergeFact, 'runtime merge fixture should emit bodyh/bodyv merge fact');
    assert.strictEqual(mergeFact.status, 'candidate');

    withRuntimeSource(source, () => {
        processRuntimeInputs([3, 3], turnIndex => {
            rewriteRuntimeObjects(['bodyh', 'bodyv'], turnIndex);
        });
        assert.strictEqual(winning, true, 'known solution should survive deterministic merge-family rewrites');
        assert.strictEqual(runtimeObjectCount('bodyh') + runtimeObjectCount('bodyv'), 1);
    });
}

function assertTransientObjectsAbsentAfterReplay() {
    const source = `
title Transient Runtime
========
OBJECTS
========
background
black
player
white
spark
yellow
${'======='}
LEGEND
${'======='}
. = background
P = player
S = spark
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
background
spark
player
=====
RULES
=====
[ > player ] -> [ > player spark ]
late [ spark ] -> []
=============
WINCONDITIONS
=============
======
LEVELS
======
P..
`;
    const analysis = analyzeSource(source, { sourcePath: 'transient_runtime.txt' });
    const sparkTransient = analysis.facts.transient_boundary.find(item => item.id === 'object_spark_end_turn_transient');
    assert.ok(sparkTransient, 'runtime transient fixture should emit spark transient fact');
    assert.strictEqual(sparkTransient.status, 'proved');

    withRuntimeSource(source, () => {
        assert.strictEqual(runtimeObjectCount('spark'), 0);
        processRuntimeInputs([3, 3], () => {
            assert.strictEqual(runtimeObjectCount('spark'), 0, 'transient object should be absent at turn boundaries');
        });
        assert.strictEqual(runtimeObjectCount('spark'), 0);
    });
}

function assertStaticLayersUnchangedAfterReplay(source, inputs) {
    const analysis = analyzeSource(source, { sourcePath: 'static_layer_runtime.txt' });
    const staticLayerIds = analysis.ps_tagged.collision_layers
        .filter(layer => layer.tags.static)
        .map(layer => layer.id);
    assert.ok(staticLayerIds.length > 0, 'runtime fixture should prove at least one layer static');
    assert.ok(!staticLayerIds.includes(1), 'moving player layer should not be proved static');

    withRuntimeSource(source, () => {
        const before = new Map(staticLayerIds.map(layerId => [layerId, layerOccupancySnapshot(layerId)]));
        processRuntimeInputs(inputs);
        for (const layerId of staticLayerIds) {
            assert.deepStrictEqual(
                layerOccupancySnapshot(layerId),
                before.get(layerId),
                `static layer ${layerId} occupancy should not change during known replay`
            );
        }
    });
}

function solverTestSource(fileName) {
    return fs.readFileSync(path.join(__dirname, 'solver_tests', fileName), 'utf8');
}

function assertLimerickMergeabilityReplay() {
    const source = solverTestSource('limerick.txt');
    const analysis = analyzeSource(source, { sourcePath: 'limerick.txt' });
    const bodyMerge = analysis.facts.mergeability.find(item =>
        item.subjects.objects.join(',') === 'PlayerBodyH,PlayerBodyV'
    );
    assert.ok(bodyMerge, 'limerick should expose the PlayerBodyH/PlayerBodyV mergeability fact');
    assert.strictEqual(bodyMerge.status, 'candidate');

    const solution = inputsForTokens([
        'right', 'right', 'right', 'right',
        'up', 'up',
        'right', 'right', 'right', 'right',
        'up', 'up',
        'right',
        'up', 'up',
        'right', 'right', 'right', 'right',
    ]);

    withRuntimeSource(source, () => {
        processRuntimeInputs(solution, turnIndex => {
            rewriteRuntimeObjects(['PlayerBodyH', 'PlayerBodyV'], turnIndex);
        });
        assert.strictEqual(winning, true, 'limerick level 1 solution should survive body variant rewrites');
    }, { levelIndex: 1 });
}

function assertAtlasTransientReplay() {
    const source = solverTestSource('atlas shrank.txt');
    const analysis = analyzeSource(source, { sourcePath: 'atlas shrank.txt' });
    const transientObjects = analysis.facts.transient_boundary
        .filter(item => item.status === 'proved')
        .map(item => item.subjects.objects[0]);
    assert.ok(transientObjects.includes('H_pickup'), 'atlas should prove H_pickup transient');
    assert.ok(transientObjects.includes('H_grav'), 'atlas should prove H_grav transient');
    assert.ok(transientObjects.includes('H_step'), 'atlas should prove H_step transient');
    assert.ok(transientObjects.includes('ShadowDoor'), 'atlas should prove ShadowDoor transient');
    assert.ok(transientObjects.includes('ShadowDoorO'), 'atlas should prove ShadowDoorO transient');

    const solution = inputsForTokens([
        'right', 'right', 'action', 'right', 'right', 'action',
        'right', 'right', 'action', 'right', 'action',
        'left', 'action', 'right', 'right', 'left',
        'action', 'right', 'action', 'right', 'right', 'right',
    ]);

    withRuntimeSource(source, () => {
        const assertTransientsAbsent = () => {
            for (const objectName of transientObjects) {
                assert.strictEqual(
                    runtimeObjectCount(objectName),
                    0,
                    `${objectName} should be absent at atlas turn boundaries`
                );
            }
        };
        assertTransientsAbsent();
        processRuntimeInputs(solution, assertTransientsAbsent);
        assertTransientsAbsent();
        assert.strictEqual(winning, true, 'atlas shrank level 4 solution should still solve');
    }, { levelIndex: 4 });
}

function assertOneMoveStaticReplay() {
    const source = solverTestSource('one_move.txt');
    const analysis = analyzeSource(source, { sourcePath: 'one_move.txt' });
    const backgroundLayer = analysis.ps_tagged.collision_layers.find(layer => layer.objects.includes('Background'));
    const targetLayer = analysis.ps_tagged.collision_layers.find(layer => layer.objects.includes('Target'));
    const playerLayer = analysis.ps_tagged.collision_layers.find(layer => layer.objects.includes('Player'));
    assert.strictEqual(backgroundLayer.tags.static, true, 'one_move background layer should be static');
    assert.strictEqual(targetLayer.tags.static, true, 'one_move target layer should be static');
    assert.strictEqual(playerLayer.tags.static, false, 'one_move player layer should not be static');

    withRuntimeSource(source, () => {
        const staticLayerIds = [backgroundLayer.id, targetLayer.id];
        const beforeLayers = new Map(staticLayerIds.map(layerId => [layerId, layerOccupancySnapshot(layerId)]));
        const beforeObjects = new Map(['Background', 'Target'].map(objectName => [
            objectName,
            objectOccupancySnapshot(objectName),
        ]));
        processRuntimeInputs(inputsForTokens(['right']));
        assert.strictEqual(winning, true, 'one_move known solution should solve');
        for (const layerId of staticLayerIds) {
            assert.deepStrictEqual(layerOccupancySnapshot(layerId), beforeLayers.get(layerId));
        }
        for (const objectName of ['Background', 'Target']) {
            assert.deepStrictEqual(objectOccupancySnapshot(objectName), beforeObjects.get(objectName));
        }
    });
}

function assertOneMoveCountInvariantReplay() {
    assertCountInvariantsPreservedAfterReplay(
        solverTestSource('one_move.txt'),
        inputsForTokens(['right']),
        {
            sourcePath: 'one_move.txt',
            expectedObjects: ['Background', 'Target', 'Player'],
            expectWinning: true,
        }
    );
}

function assertCratesMoveCountInvariantReplay() {
    assertCountInvariantsPreservedAfterReplay(
        solverTestSource('Crates move when you move.txt'),
        inputsForTokens(['up', 'up', 'up', 'left', 'down', 'right', 'right', 'down', 'down']),
        {
            sourcePath: 'Crates move when you move.txt',
            levelIndex: 1,
            expectedObjects: ['Background', 'Target', 'Player', 'Wall', 'Crate'],
            expectWinning: true,
        }
    );
}

function assertOneMoveActionNoopReplay() {
    assertActionNoopAfterReplay(
        solverTestSource('one_move.txt'),
        inputsForTokens(['right']),
        {
            sourcePath: 'one_move.txt',
            expectWinning: true,
        }
    );
}

function assertCastleClosetActionNoopRejected() {
    const analysis = analyzeSource(solverTestSource('castlecloset.txt'), { sourcePath: 'castlecloset.txt' });
    const actionNoop = analysis.facts.movement_action.find(item => item.id === 'action_noop');
    assert.strictEqual(analysis.ps_tagged.game.tags.has_action_rules, true, 'castlecloset should contain action rules');
    assert.strictEqual(actionNoop.status, 'rejected', 'castlecloset direct-Player action rules should not be action-noop');
    assert.ok(actionNoop.blockers.includes('reads_action'));
}

function assertPushRulegroupFlowReplay() {
    const source = solverTestSource('push.txt');
    const analysis = analyzeSource(source, { sourcePath: 'push.txt' });
    const splitFlow = analysis.facts.rulegroup_flow.find(item =>
        item.status === 'candidate'
        && item.subjects.groups[0] === 'early_group_1'
    );
    assert.ok(splitFlow, 'push should expose the pusher-style split candidate group');
    assert.deepStrictEqual(splitFlow.value.components.map(component => component.length), [1, 1, 1, 1]);
    assert.deepStrictEqual(splitFlow.value.interaction_edges, []);
    assert.deepStrictEqual(Object.values(splitFlow.value.rerun_masks), [[], [], [], []]);

    withRuntimeSource(source, () => {
        processRuntimeInputs(inputsForTokens(['up', 'up', 'right', 'up']));
        assert.strictEqual(winning, true, 'push level 1 solution should solve while rulegroup-flow facts are present');
    }, { levelIndex: 1 });
}

function assertSplittableRulegroupTransformReplay() {
    const source = `
title Splittable Rulegroup Runtime
========
OBJECTS
========
Background
black
Pusher
white
Pushable
orange
Goal
green
${'======='}
LEGEND
${'======='}
. = Background
P = Pusher
C = Pushable
G = Goal
Player = Pusher
${'======='}
SOUNDS
${'======='}
================
COLLISIONLAYERS
================
Background
Goal
Pusher
Pushable
=====
RULES
=====
down [ Pushable | up Pusher ] -> [ up Pushable | up Pusher ]
+ down [ down Pusher | Pushable ] -> [ down Pusher | down Pushable ]
+ right [ Pushable | left Pusher ] -> [ left Pushable | left Pusher ]
+ right [ right Pusher | Pushable ] -> [ right Pusher | right Pushable ]
=============
WINCONDITIONS
=============
All Pushable on Goal
======
LEVELS
======
PCG
`;
    const analysis = analyzeSource(source, { sourcePath: 'splittable_rulegroup_runtime.txt' });
    const splitFlow = analysis.facts.rulegroup_flow.find(item => item.status === 'candidate');
    assert.ok(splitFlow, 'runtime pusher fixture should expose a split candidate');
    assert.deepStrictEqual(splitFlow.value.components.map(component => component.length), [1, 1, 1, 1]);
    assert.deepStrictEqual(splitFlow.value.interaction_edges, []);

    const solution = inputsForTokens(['right']);
    const originalSnapshot = replayRuntimeSnapshot(source, solution);
    const splitSnapshot = replayRuntimeSnapshot(splitPlusContinuationRules(source), solution);
    assert.strictEqual(originalSnapshot.winning, true, 'original plus-group fixture should solve');
    assert.deepStrictEqual(splitSnapshot, originalSnapshot, 'split plus-group fixture should replay identically');
}

assertStaticObjectsUnchangedAfterReplay(
    STATIC_OBJECT_GAME
        .replace('[ > PlayerObject ] -> [ > PlayerObject ]', '[ action PlayerObject ] -> win')
        .replace('Some Player', 'No PlayerObject'),
    [4]
);

assertMergeabilitySwapsPreserveReplay();
assertTransientObjectsAbsentAfterReplay();
assertStaticLayersUnchangedAfterReplay(STATIC_OBJECT_GAME.replace('Some Player', ''), [3, 1, 3]);
assertLimerickMergeabilityReplay();
assertAtlasTransientReplay();
assertOneMoveStaticReplay();
assertOneMoveCountInvariantReplay();
assertCratesMoveCountInvariantReplay();
assertOneMoveActionNoopReplay();
assertCastleClosetActionNoopRejected();
assertPushRulegroupFlowReplay();
assertSplittableRulegroupTransformReplay();

console.log('ps_static_analysis_node: ok');
