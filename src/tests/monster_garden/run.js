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
    const signature = garden.failureSignature(result);
    if (!options.shrink) {
        return { source: mutant.source, steps: 0, signature: signature };
    }
    let current = mutant.source.split('\n');
    let steps = 0;
    let remaining = options.shrinkBudget;
    let changed = true;
    while (changed && remaining > 0) {
        changed = false;
        let i = 0;
        while (i < current.length && remaining > 0) {
            const candidateSource = current.slice(0, i).concat(current.slice(i + 1)).join('\n');
            remaining--;
            steps++;
            const next = await evaluateMutant({
                source: candidateSource,
                inputs: mutant.inputs,
                level: mutant.level,
                randomSeed: mutant.randomSeed
            }, options);
            if (garden.failureSignature(next) === signature) {
                current = candidateSource.split('\n');
                changed = true;
            } else {
                i++;
            }
        }
    }
    return { source: current.join('\n'), steps: steps, signature: signature };
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
                fixtureName: mutant.fixtureName,
                fixtureIndex: mutant.fixtureIndex,
                mutator: mutant.mutator,
                detail: mutant.detail,
                result: result,
                signature: minimized.signature,
                shrinkSteps: minimized.steps
            }, null, 2) + '\n',
            'regression.js': garden.formatRegression(
                'monster garden ' + options.seed + ' ' + artifactIndex,
                minimized.source
            )
        });
    }
    process.stdout.write(JSON.stringify(counts) + '\n');
}

main().catch(function(error) {
    process.stderr.write(error.stack + '\n');
    process.exitCode = 1;
});
