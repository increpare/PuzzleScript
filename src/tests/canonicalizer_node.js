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
assert.ok(hashes.full && hashes.structural && hashes['no-levels'] && hashes.mechanics && hashes.ruleset && hashes.semantic, 'comparison hashes should expose all modes');

const rulesetBaseHash = hashCanonical(canonicalizeSource(baseGame, 'ruleset'));
const rulesetDifferentHash = hashCanonical(canonicalizeSource(differentLevels, 'ruleset'));
assert.strictEqual(rulesetBaseHash, rulesetDifferentHash, 'ruleset mode should ignore map differences');

const semanticBaseHash = hashCanonical(canonicalizeSource(baseGame, 'semantic'));
const semanticDifferentHash = hashCanonical(canonicalizeSource(differentLevels, 'semantic'));
assert.notStrictEqual(semanticBaseHash, semanticDifferentHash, 'semantic mode should include per-cell map contents');

const semanticBase = canonicalizeSource(baseGame, 'semantic');
assert.deepStrictEqual(semanticBase.playerObjects, ['obj_1'], 'semantic mode should preserve canonical player objects');
assert.deepStrictEqual(semanticBase.backgroundObjects, ['obj_0'], 'semantic mode should preserve canonical background objects');
assert.ok(
    semanticBase.rules.every(rule => rule.commands.every(command => !/^sfx(?:10|[0-9])$/.test(command[0]))),
    'semantic mode should discard sfxN commands'
);

const semanticReordered = canonicalizeSource(reorderedObjects, 'semantic');
assert.deepStrictEqual(semanticReordered, semanticBase, 'semantic mode should ignore object declaration order');

console.log('canonicalizer_node: ok');
