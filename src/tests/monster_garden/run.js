#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const garden = require('./garden');

const workerPath = path.join(__dirname, 'worker.js');
const resourceDir = path.join(__dirname, '..', 'resources');

function filterCorpus(corpus, fixture) {
    if (!fixture) {
        return corpus;
    }
    const needle = fixture.toLowerCase();
    return corpus.filter(function(item) {
        return item.name.toLowerCase().indexOf(needle) >= 0;
    });
}

function allowedMutators(option) {
    return option || garden.mutators.map(function(mutator) { return mutator.name; });
}

function readGitRev() {
    try {
        return require('child_process').execSync('git rev-parse HEAD', {
            cwd: __dirname,
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'ignore']
        }).trim();
    } catch (error) {
        return '';
    }
}

async function evaluateMutant(mutant, options) {
    const job = {
        source: mutant.source,
        inputs: mutant.inputs,
        level: mutant.level,
        randomSeed: mutant.randomSeed,
        replay: options.replay,
        maxInputs: options.maxInputs
    };
    const result = await garden.runChild(
        process.execPath,
        [workerPath],
        JSON.stringify(job),
        options.timeoutMs
    );
    return result;
}

async function shrinkMutant(mutant, result, options) {
    return garden.shrinkInteresting(mutant, result, Object.assign({}, options, {
        evaluate: function(partial) {
            return evaluateMutant(Object.assign({}, mutant, partial), options);
        }
    }));
}

async function main() {
    let options;
    try {
        options = garden.parseArguments(process.argv.slice(2));
    } catch (error) {
        process.stderr.write(error.message + '\n');
        process.exitCode = 1;
        return;
    }
    if (options.listMutators) {
        garden.mutators.forEach(function(mutator) {
            process.stdout.write(mutator.name + '\n');
        });
        return;
    }
    const corpus = filterCorpus(garden.loadCorpus(resourceDir), options.fixture);
    if (corpus.length === 0) {
        process.stderr.write('No fixtures matched --fixture\n');
        process.exitCode = 1;
        return;
    }
    const rng = new garden.Random(options.seed);
    const counts = {
        ok: 0,
        'compiler-error': 0,
        crash: 0,
        timeout: 0,
        invariant: 0,
        nondeterministic: 0,
        'replay-divergence': 0,
        skipped: 0
    };
    let artifactIndex = 0;
    for (let i = 0; i < options.count; i++) {
        const fixture = corpus[rng.integer(corpus.length)];
        let mutant;
        try {
            mutant = garden.mutateFixture(fixture, rng, allowedMutators(options.mutators), {
                maxAttempts: options.maxAttempts
            });
        } catch (error) {
            if (!garden.isInapplicableMutation(error)) {
                throw error;
            }
            counts.skipped++;
            continue;
        }
        const result = await evaluateMutant(mutant, options);
        counts[result.kind] = (counts[result.kind] || 0) + 1;
        process.stdout.write(
            '#' + (i + 1) + ' ' + result.kind + ' ' + mutant.mutator + ' ' + mutant.fixtureName + '\n'
        );
        if (!garden.isInteresting(result)) {
            continue;
        }
        const minimized = await shrinkMutant(mutant, result, options);
        artifactIndex++;
        const dirName = garden.artifactDirName(minimized.signature, options.seed, artifactIndex);
        fs.mkdirSync(options.output, { recursive: true });
        garden.writeArtifacts(options.output, dirName, {
            'original.txt': mutant.source,
            'minimized.txt': minimized.source,
            'report.json': JSON.stringify({
                seed: options.seed,
                campaignIndex: i,
                gitRev: readGitRev(),
                fixtureName: mutant.fixtureName,
                corpusIndex: mutant.corpusIndex,
                fixtureIndex: mutant.fixtureIndex,
                fixtureKind: mutant.kind,
                mutator: mutant.mutator,
                detail: mutant.detail,
                attempt: mutant.attempt,
                inputs: mutant.inputs,
                level: mutant.level,
                randomSeed: mutant.randomSeed,
                replay: options.replay,
                maxInputs: options.maxInputs,
                timeoutMs: options.timeoutMs,
                originalResult: result,
                minimizedResult: minimized.result,
                signature: minimized.signature,
                shrinkSteps: minimized.steps
            }, null, 2) + '\n',
            'regression.js': garden.formatRegression(
                'monster garden ' + options.seed + ' ' + artifactIndex,
                minimized.source,
                { inputs: mutant.inputs, level: mutant.level, randomSeed: mutant.randomSeed }
            )
        });
    }
    process.stdout.write(JSON.stringify(counts) + '\n');
}

main().catch(function(error) {
    process.stderr.write(error.stack + '\n');
    process.exitCode = 1;
});
