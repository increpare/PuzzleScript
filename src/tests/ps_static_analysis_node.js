#!/usr/bin/env node
'use strict';

const assert = require('assert');

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

const STATIONARY_TICK_GAME = SIMPLE_GAME.replace('[ > Hero ] -> [ > Hero ]', '[ stationary Goal ] -> [ randomDir Goal ]');
const stationaryTick = analyzeSource(STATIONARY_TICK_GAME, { sourcePath: 'stationary_tick.txt' });
const stationaryTickFact = stationaryTick.facts.movement_action.find(item => item.id === 'action_noop');
assert.strictEqual(stationaryTick.ps_tagged.game.tags.has_autonomous_tick_rules, true);
assert.strictEqual(stationaryTickFact.status, 'rejected');
assert.ok(stationaryTickFact.blockers.includes('autonomous_solver_active_rule'));
assert.ok(stationaryTickFact.blockers.includes('action_may_create_directional_movement'));

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
assert.strictEqual(staticWall.tags.static, true, 'unwritten, unmoved wall should be tagged static');
assert.strictEqual(staticWallFact.status, 'proved', 'unwritten, unmoved wall should have a proved static fact');
assert.strictEqual(staticPlayer.tags.count_invariant, true, 'player object count can be invariant');
assert.strictEqual(staticPlayer.tags.static, false, 'player object is not static because input applies movement');

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

assertStaticObjectsUnchangedAfterReplay(
    STATIC_OBJECT_GAME
        .replace('[ > PlayerObject ] -> [ > PlayerObject ]', '[ action PlayerObject ] -> win')
        .replace('Some Player', 'No PlayerObject'),
    [4]
);

console.log('ps_static_analysis_node: ok');
