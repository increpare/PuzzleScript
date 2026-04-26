#!/usr/bin/env node
'use strict';

const fs = require('fs');

function usage() {
    console.error('Usage: node src/tests/assert_compiled_rules_coverage_shape.js path/to/coverage.json');
    process.exit(1);
}

if (process.argv.length !== 3) {
    usage();
}

const coveragePath = process.argv[2];
const coverage = JSON.parse(fs.readFileSync(coveragePath, 'utf8'));

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function assertMatchingFullTurnObjects(owner, label) {
    const specialized = owner.specialized_full_turn;
    const alias = owner.compiled_tick;
    assert(specialized && typeof specialized === 'object', `${label} missing specialized_full_turn`);
    assert(alias && typeof alias === 'object', `${label} missing compiled_tick compatibility alias`);
    for (const key of [
        'backend_codegen_available',
        'fully_generated',
        'whole_turn_supported',
    ]) {
        assert(
            specialized[key] === alias[key],
            `${label} specialized_full_turn.${key}=${specialized[key]} differs from compiled_tick.${key}=${alias[key]}`
        );
    }
}

assert(coverage.aggregate && typeof coverage.aggregate === 'object', 'missing aggregate coverage');
assertMatchingFullTurnObjects(coverage.aggregate, 'aggregate');

assert(Array.isArray(coverage.sources), 'missing sources coverage array');
assert(coverage.sources.length > 0, 'coverage sources array is empty');
for (const source of coverage.sources) {
    assertMatchingFullTurnObjects(source, `source ${source.path ?? source.index ?? '<unknown>'}`);
}

console.log(`coverage_shape_ok sources=${coverage.sources.length}`);
