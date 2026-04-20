#!/usr/bin/env node
'use strict';

const assert = require('assert');

const {
    buildComparisonHashes,
    canonicalizeSource,
    hashCanonical,
} = require('../canonicalize');

const baseGame = `
title Base
author Test

========
OBJECTS
========

Background
black
00000
00000
00000
00000
00000

Hero
blue
00000
00000
00000
00000
00000

Goal
yellow
00000
00000
00000
00000
00000

=======
LEGEND
=======

. = Background
P = Hero
G = Goal
Player = Hero

=======
SOUNDS
=======

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

All Hero on Goal

======
LEVELS
======

P.G
...
...
`;

const renamedAndReskinned = `
(comment)
title Something Else
author Someone Else

========
OBJECTS
========

Bg
white
00000
00000
00000
00000
00000

Avatar
red
00000
00000
00000
00000
00000

Exit
green
00000
00000
00000
00000
00000

=======
LEGEND
=======

. = Bg
P = Avatar
G = Exit
Player = Avatar

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

Bg
Avatar, Exit

=====
RULES
=====

[ > Avatar ] -> [ > Avatar ]

=============
WINCONDITIONS
=============

All Avatar on Exit

======
LEVELS
======

P.G
...
...
`;

const differentLevels = `
title Base

========
OBJECTS
========

Background
black
00000
00000
00000
00000
00000

Hero
blue
00000
00000
00000
00000
00000

Goal
yellow
00000
00000
00000
00000
00000

=======
LEGEND
=======

. = Background
P = Hero
G = Goal
Player = Hero

=======
SOUNDS
=======

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

All Hero on Goal

======
LEVELS
======

PG.
...
...
`;

const reorderedObjects = `
title Base

========
OBJECTS
========

Goal
yellow
00000
00000
00000
00000
00000

Background
black
00000
00000
00000
00000
00000

Hero
blue
00000
00000
00000
00000
00000

=======
LEGEND
=======

. = Background
P = Hero
G = Goal
Player = Hero

=======
SOUNDS
=======

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

All Hero on Goal

======
LEVELS
======

P.G
...
...
`;

const structuralBase = canonicalizeSource(baseGame, 'structural');
const structuralVariant = canonicalizeSource(renamedAndReskinned, 'structural');
assert.deepStrictEqual(structuralVariant, structuralBase, 'structural mode should ignore object naming and visuals');

const structuralBaseHash = hashCanonical(structuralBase);
const structuralDifferentLevelsHash = hashCanonical(canonicalizeSource(differentLevels, 'structural'));
assert.notStrictEqual(structuralBaseHash, structuralDifferentLevelsHash, 'structural mode should keep level differences');

const noLevelsBaseHash = hashCanonical(canonicalizeSource(baseGame, 'no-levels'));
const noLevelsDifferentHash = hashCanonical(canonicalizeSource(differentLevels, 'no-levels'));
assert.strictEqual(noLevelsBaseHash, noLevelsDifferentHash, 'no-levels mode should ignore level changes');

const hashes = buildComparisonHashes(baseGame);
assert.ok(hashes.full && hashes.structural && hashes['no-levels'] && hashes.mechanics && hashes.ruleset && hashes.semantic && hashes.family, 'comparison hashes should expose all modes');

const rulesetBaseHash = hashCanonical(canonicalizeSource(baseGame, 'ruleset'));
const rulesetDifferentHash = hashCanonical(canonicalizeSource(differentLevels, 'ruleset'));
assert.strictEqual(rulesetBaseHash, rulesetDifferentHash, 'ruleset mode should ignore map differences');

const semanticBaseHash = hashCanonical(canonicalizeSource(baseGame, 'semantic'));
const semanticDifferentHash = hashCanonical(canonicalizeSource(differentLevels, 'semantic'));
assert.notStrictEqual(semanticBaseHash, semanticDifferentHash, 'semantic mode should include per-cell map contents');

const semanticBase = canonicalizeSource(baseGame, 'semantic');
assert.deepStrictEqual(semanticBase.playerObjects, ['obj_0'], 'semantic mode should preserve canonical player objects after pruning inert layers');
assert.deepStrictEqual(semanticBase.backgroundObjects, [], 'semantic mode should drop inert background-only layers');
assert.deepStrictEqual(semanticBase.collisionLayers, [['obj_0', 'obj_1']], 'semantic mode should merge/prune inert objects at the collision-layer level by default');
assert.ok(
    semanticBase.rules.every(rule => rule.commands.every(command => !/^sfx(?:10|[0-9])$/.test(command[0]))),
    'semantic mode should discard sfxN commands'
);

const semanticReordered = canonicalizeSource(reorderedObjects, 'semantic');
assert.deepStrictEqual(semanticReordered, semanticBase, 'semantic mode should ignore object declaration order');

const familySource = `
========
OBJECTS
========

Background
black
00000
00000
00000
00000
00000

Player
white
00000
00000
00000
00000
00000

ObjA
red
00000
00000
00000
00000
00000

ObjB
blue
00000
00000
00000
00000
00000

=======
LEGEND
=======

. = Background
P = Player
a = ObjA
b = ObjB

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

Background
Player, ObjA, ObjB

=====
RULES
=====

[ > Player | ObjA ] -> [ > Player | ObjA ]
[ > Player | ObjB ] -> [ > Player | ObjB ]

=============
WINCONDITIONS
=============

Some ObjA

======
LEVELS
======

Pa
Pb
`;

const familyCanonical = canonicalizeSource(familySource, 'family');
assert.strictEqual(familyCanonical.format, 'puzzlescript-family-canonical-v1', 'family mode should expose the family format');
assert.deepStrictEqual(familyCanonical.collisionLayers, [['fam_0', 'fam_1', 'fam_2']], 'family mode should keep rule-mentioned objects distinct while dropping inert background-only layers');
assert.strictEqual(familyCanonical.rules.length, 8, 'family mode should preserve distinct compiled rules for distinct rule-mentioned objects');
assert.deepStrictEqual(familyCanonical.playerObjects, ['fam_0'], 'family mode should preserve player role after family relabeling');

const inertLayerSource = `
========
OBJECTS
========

Background
black
00000
00000
00000
00000
00000

Hero
white
00000
00000
00000
00000
00000

Wall
gray
00000
00000
00000
00000
00000

Gem
green
00000
00000
00000
00000
00000

Flower
pink
00000
00000
00000
00000
00000

=======
LEGEND
=======

. = Background
P = Hero
W = Wall
G = Gem
F = Flower
Player = Hero

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

Background
Hero, Wall
Gem, Flower

=====
RULES
=====

[ > Hero | Wall ] -> [ > Hero | Wall ]

=============
WINCONDITIONS
=============

Some Gem

======
LEVELS
======

PW
GF
`;

const inertLayerCanonical = canonicalizeSource(inertLayerSource, 'family');
assert.deepStrictEqual(inertLayerCanonical.collisionLayers, [['fam_0', 'fam_1']], 'family mode should remove layers with no player or rule-mentioned objects');
assert.deepStrictEqual(
    Array.from(new Set(inertLayerCanonical.rules.flatMap(rule =>
        rule.lhs.flatMap(row => row.flatMap(cell => cell.flatMap(entry => entry.obj ? [entry.obj] : (entry.objs || []))))
    ))).sort(),
    ['fam_0', 'fam_1'],
    'family mode should keep surviving rule-mentioned objects intact when pruning inert layers'
);

const inertMergeSource = `
========
OBJECTS
========

Background
black
00000
00000
00000
00000
00000

Hero
white
00000
00000
00000
00000
00000

Wall
gray
00000
00000
00000
00000
00000

Tree
green
00000
00000
00000
00000
00000

Rock
brown
00000
00000
00000
00000
00000

=======
LEGEND
=======

. = Background
P = Hero
W = Wall
T = Tree
R = Rock
Player = Hero

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

Background
Hero, Wall, Tree, Rock

=====
RULES
=====

[ > Hero | Wall ] -> [ > Hero | Wall ]

=============
WINCONDITIONS
=============

Some Tree

======
LEVELS
======

PW
TR
`;

const inertMergeCanonical = canonicalizeSource(inertMergeSource, 'family');
assert.deepStrictEqual(inertMergeCanonical.collisionLayers, [['fam_0', 'fam_1', 'fam_2']], 'family mode should merge multiple rule-inert non-player objects within a retained layer');

const semanticInertMerge = canonicalizeSource(inertMergeSource, 'semantic');
assert.deepStrictEqual(semanticInertMerge.collisionLayers, [['obj_0', 'obj_1', 'obj_2']], 'semantic mode should merge multiple rule-inert non-player objects within a retained layer by default');

console.log('canonicalizer_node: ok');
