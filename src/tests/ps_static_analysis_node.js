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
assert.strictEqual(levelPresenceObjects.get('Always').present_in_all_levels, true);
assert.strictEqual(levelPresenceObjects.get('Always').present_in_some_levels, false);
assert.strictEqual(levelPresenceObjects.get('Always').present_in_no_levels, false);
assert.strictEqual(levelPresenceObjects.get('Sometimes').present_in_all_levels, false);
assert.strictEqual(levelPresenceObjects.get('Sometimes').present_in_some_levels, true);
assert.strictEqual(levelPresenceObjects.get('Sometimes').present_in_no_levels, false);
assert.strictEqual(levelPresenceObjects.get('Never').present_in_all_levels, false);
assert.strictEqual(levelPresenceObjects.get('Never').present_in_some_levels, false);
assert.strictEqual(levelPresenceObjects.get('Never').present_in_no_levels, true);

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
[ Alpha ] -> [ Alpha ] sfx0
[ right Alpha ] -> [ right Alpha ] sfx0
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
assert.strictEqual(commandRules.length, 4, 'command-only rules should remain present');
assert.strictEqual(commandRules[0].tags.inert_command_only, true, 'sfx-only rule is inert for solver state');
assert.strictEqual(commandRules[0].tags.solver_state_active, false, 'sfx-only rule is not solver-state active');
assert.strictEqual(commandRules[1].tags.command_only, true, 'checkpoint-only rule is command-only');
assert.strictEqual(commandRules[1].tags.solver_state_active, true, 'checkpoint is semantic/metagame-active');
assert.strictEqual(commandRules[2].tags.inert_command_only, true, 'unchanged RHS plus sfx is inert-command-only');
assert.strictEqual(commandRules[2].tags.solver_state_active, false);
assert.strictEqual(commandRules[3].tags.inert_command_only, true, 'unchanged movement RHS plus sfx is inert-command-only');
assert.strictEqual(commandRules[3].tags.solver_state_active, false);

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
const mergeFact = mergeable.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.ok(mergeFact, 'BodyH/BodyV should produce a mergeability fact');
assert.strictEqual(mergeFact.status, 'candidate');
assert.ok(mergeFact.proof.includes('same_collision_layer'));
assert.ok(mergeFact.proof.includes('observed_only_through_shared_sets'));

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

const DIRECT_WIN_GAME = MERGEABLE_GAME.replace('Some Body on Goal', 'Some BodyH on Goal');
const directWin = analyzeSource(DIRECT_WIN_GAME, { sourcePath: 'direct_win.txt' });
const directWinFact = directWin.facts.mergeability.find(item => item.subjects.objects.join(',') === 'BodyH,BodyV');
assert.strictEqual(directWinFact.status, 'rejected');
assert.ok(directWinFact.blockers.includes('different_win_roles'));

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

const ACTION_MOVEMENT_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ action Player ] -> [ > Player ]');
const actionMovement = analyzeSource(ACTION_MOVEMENT_GAME, { sourcePath: 'action_movement.txt' });
const actionMovementFact = actionMovement.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(actionMovementFact.status, 'rejected');
assert.ok(actionMovementFact.blockers.includes('action_may_create_directional_movement'));

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

const SPAWN_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ Hero ] -> [ Hero Goal ]');
const spawnReport = analyzeSource(SPAWN_GAME, { sourcePath: 'spawn.txt' });
const goalCount = spawnReport.facts.count_layer_invariants.find(item => item.id === 'object_Goal_count_preserved');
assert.strictEqual(goalCount.status, 'rejected');
assert.ok(goalCount.blockers.includes('object_written_by_solver_active_rule'));

const DESTROY_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ Hero ] -> []');
const destroyReport = analyzeSource(DESTROY_GAME, { sourcePath: 'destroy.txt' });
const heroDestroyCount = destroyReport.facts.count_layer_invariants.find(item => item.id === 'object_Hero_count_preserved');
assert.strictEqual(heroDestroyCount.status, 'rejected', 'deleting an object to an empty RHS should reject count preservation');

const LAYER_OVERWRITE_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ no Goal ] -> [ Goal ]');
const layerOverwriteReport = analyzeSource(LAYER_OVERWRITE_GAME, { sourcePath: 'layer_overwrite.txt' });
const heroOverwriteCount = layerOverwriteReport.facts.count_layer_invariants.find(item => item.id === 'object_Hero_count_preserved');
assert.strictEqual(heroOverwriteCount.status, 'rejected', 'writing one object in a collision layer can remove siblings');

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
assert.strictEqual(layerCreateWall.tags.static, false, 'creating a sibling on the wall layer should reject wall static');
assert.ok(layerCreateWallFact.blockers.includes('collision_layer_object_may_be_created'));

const STATIC_AGGREGATE_WRITE_GAME = STATIC_OBJECT_GAME.replace(
    '[ > PlayerObject ] -> [ > PlayerObject ]',
    '[ Solid ] -> []'
);
const staticAggregateWrite = analyzeSource(STATIC_AGGREGATE_WRITE_GAME, { sourcePath: 'static_aggregate_write.txt' });
const aggregateWriteWall = staticAggregateWrite.ps_tagged.objects.find(object => object.name === 'Wall');
const aggregateWriteWallFact = staticAggregateWrite.facts.count_layer_invariants.find(item => item.id === 'object_Wall_static');
assert.strictEqual(aggregateWriteWall.tags.static, false, 'aggregate deletion mentioning wall should reject wall static');
assert.ok(aggregateWriteWallFact.blockers.includes('object_written_by_solver_active_rule'));

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
