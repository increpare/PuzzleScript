#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./lib/puzzlescript_node_env');
const { buildCompiledIr } = require('./lib/puzzlescript_ir');
const { runInputTrace } = require('./lib/puzzlescript_trace');

function parseArgs(argv) {
    const args = argv.slice(2);
    return {
        outputDir: args[0] ? path.resolve(args[0]) : path.resolve('build/js-parity-data'),
    };
}

function hashFixtureId(prefix, name, source, randomSeed, levelIndex) {
    const digest = crypto
        .createHash('sha256')
        .update(prefix)
        .update('\0')
        .update(name)
        .update('\0')
        .update(source)
        .update('\0')
        .update(String(randomSeed ?? ''))
        .update('\0')
        .update(String(levelIndex ?? ''))
        .digest('hex');
    return `${prefix}-${digest.slice(0, 16)}`;
}

function ensureDir(dir) {
    fs.mkdirSync(dir, { recursive: true });
}

function compileSimulationFixture(testEntry, outputDir) {
    const [name, payload] = testEntry;
    const [source, inputs, expectedSerializedLevel, targetLevelRaw, randomSeedRaw, expectedSoundsRaw] = payload;
    const targetLevel = targetLevelRaw === undefined ? 0 : targetLevelRaw;
    const randomSeed = randomSeedRaw === undefined ? null : randomSeedRaw;
    const expectedSounds = expectedSoundsRaw === undefined ? null : expectedSoundsRaw;
    const fixtureId = hashFixtureId('sim', name, source, randomSeed, targetLevel);

    unitTesting = true;
    lazyFunctionGeneration = false;
    let ir;
    let trace;
    try {
        compile(['loadLevel', targetLevel], `${source}\n`, randomSeed);
        while (againing) {
            againing = false;
            processInput(-1);
        }
        ir = buildCompiledIr({
            fixture_id: fixtureId,
            fixture_name: name,
            command: ['loadLevel', targetLevel],
            random_seed: randomSeed,
            source_kind: 'simulation_fixture',
        });
        trace = runInputTrace(inputs);
    } finally {
        unitTesting = false;
        lazyFunctionGeneration = true;
    }

    const irRelativePath = path.join('ir', `${fixtureId}.json`);
    const irAbsolutePath = path.join(outputDir, irRelativePath);
    const traceRelativePath = path.join('traces', `${fixtureId}.json`);
    const traceAbsolutePath = path.join(outputDir, traceRelativePath);
    ensureDir(path.dirname(irAbsolutePath));
    ensureDir(path.dirname(traceAbsolutePath));

    fs.writeFileSync(irAbsolutePath, `${JSON.stringify(ir, null, 2)}\n`, 'utf8');

    const finalSnapshot = trace.snapshots.length > 0 ? trace.snapshots[trace.snapshots.length - 1] : null;
    if (!finalSnapshot || finalSnapshot.serialized_level !== expectedSerializedLevel) {
        throw new Error(`Trace export mismatch for fixture "${name}"`);
    }
    if (expectedSounds !== null) {
        const actualSounds = trace.snapshots.flatMap(snapshot => snapshot.new_sounds);
        if (actualSounds.join(';') !== expectedSounds.join(';')) {
            throw new Error(`Trace sound mismatch for fixture "${name}"`);
        }
    }
    fs.writeFileSync(
        traceAbsolutePath,
        `${JSON.stringify({
            schema_version: 1,
            fixture_id: fixtureId,
            fixture_name: name,
            target_level: targetLevel,
            random_seed: randomSeed,
            inputs,
            expected_serialized_level: expectedSerializedLevel,
            expected_sounds: expectedSounds,
            trace,
        }, null, 2)}\n`,
        'utf8'
    );

    return {
        id: fixtureId,
        name,
        ir_file: irRelativePath.replace(/\\/g, '/'),
        trace_file: traceRelativePath.replace(/\\/g, '/'),
        target_level: targetLevel,
        random_seed: randomSeed,
        inputs,
        initial_serialized_level: ir.prepared_session.serialized_level,
        expected_serialized_level: expectedSerializedLevel,
        expected_sounds: expectedSounds,
    };
}

function compileErrorFixture(testEntry) {
    const [name, payload] = testEntry;
    const [source, expectedErrors, expectedErrorCount] = payload;

    if (typeof resetParserErrorState === 'function') {
        resetParserErrorState();
    } else {
        errorStrings = [];
        errorCount = 0;
    }

    unitTesting = true;
    lazyFunctionGeneration = false;
    try {
        compile(['restart'], `${source}\n`);
    } finally {
        unitTesting = false;
        lazyFunctionGeneration = true;
    }

    return {
        name,
        source,
        expected_errors: expectedErrors,
        expected_error_count: expectedErrorCount,
    };
}

function main() {
    const { outputDir } = parseArgs(process.argv);
    const messageSink = [];
    loadPuzzleScript({ includeTests: true, messageSink });

    ensureDir(outputDir);
    ensureDir(path.join(outputDir, 'ir'));

    const simulationFixtures = global.testdata.map(entry => compileSimulationFixture(entry, outputDir));
    const compilationFixtures = global.errormessage_testdata.map(compileErrorFixture);

    const manifest = {
        schema_version: 1,
        simulation_fixture_count: simulationFixtures.length,
        compilation_fixture_count: compilationFixtures.length,
        simulation_fixtures: simulationFixtures,
        compilation_fixtures: compilationFixtures,
    };

    fs.writeFileSync(
        path.join(outputDir, 'fixtures.json'),
        `${JSON.stringify(manifest, null, 2)}\n`,
        'utf8'
    );
}

main();
