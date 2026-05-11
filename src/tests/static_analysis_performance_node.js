#!/usr/bin/env node
'use strict';

const assert = require('assert');
const { performance } = require('perf_hooks');

const { analyzeSource } = require('./ps_static_analysis');

function generatedWriterSource(objectCount) {
    const objectNames = Array.from({ length: objectCount }, (_, index) => `Obj${index}`);
    const objectLines = [
        'Background\nblack',
        'Player\nwhite',
        ...objectNames.map(name => `${name}\nred`),
    ];
    const layers = [
        'Background',
        'Player',
        ...objectNames,
    ];
    const rules = objectNames.map(name => `[ Player ] -> [ Player ${name} ]`);

    return [
        'title Static Analysis Performance Fixture',
        '========',
        'OBJECTS',
        '========',
        objectLines.join('\n'),
        '========',
        'LEGEND',
        '========',
        '. = Background',
        'P = Player',
        '======',
        'SOUNDS',
        '======',
        '================',
        'COLLISIONLAYERS',
        '================',
        layers.join('\n'),
        '=====',
        'RULES',
        '=====',
        rules.join('\n'),
        '=============',
        'WINCONDITIONS',
        '=============',
        'Some Player',
        '======',
        'LEVELS',
        '======',
        'P',
        '',
    ].join('\n');
}

function manyInteractingRulesSource(ruleCount) {
    const rules = Array.from({ length: ruleCount }, (_, index) =>
        index % 2 === 0 ? '[ Player ] -> [ Box ]' : '[ Box ] -> [ Player ]'
    );

    return [
        'title Static Analysis Family Filter Fixture',
        '========',
        'OBJECTS',
        '========',
        'Background',
        'black',
        'Player',
        'white',
        'Box',
        'red',
        '========',
        'LEGEND',
        '========',
        '. = Background',
        'P = Player',
        'B = Box',
        '======',
        'SOUNDS',
        '======',
        '================',
        'COLLISIONLAYERS',
        '================',
        'Background',
        'Player, Box',
        '=====',
        'RULES',
        '=====',
        rules.join('\n'),
        '=============',
        'WINCONDITIONS',
        '=============',
        'Some Player',
        '======',
        'LEVELS',
        '======',
        'P',
        '',
    ].join('\n');
}

let source = generatedWriterSource(650);
let start = performance.now();
let report = analyzeSource(source, { sourcePath: 'generated:static-analysis-performance' });
let elapsedMs = performance.now() - start;

assert.strictEqual(report.status, 'ok');
assert.ok(
    elapsedMs < 5000,
    `static analysis should reuse per-rule flow facts for large writer sets; took ${elapsedMs.toFixed(0)}ms`
);

source = manyInteractingRulesSource(5000);
start = performance.now();
report = analyzeSource(source, {
    sourcePath: 'generated:static-analysis-family-filter-performance',
    familyFilter: 'count_layer_invariants',
});
elapsedMs = performance.now() - start;

assert.strictEqual(report.status, 'ok');
assert.deepStrictEqual(Object.keys(report.facts), ['count_layer_invariants']);
assert.ok(
    elapsedMs < 3500,
    `familyFilter=count_layer_invariants should not derive unrelated rule-flow families; took ${elapsedMs.toFixed(0)}ms`
);

console.log(`static_analysis_performance_node: ok`);
