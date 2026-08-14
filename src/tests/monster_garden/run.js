#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const garden = require('./garden');

const workerPath = path.join(__dirname, 'worker.js');
const resourceDir = path.join(__dirname, '..', 'resources');

let currentChild = null;

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

async function evaluateMutant(mutant, options, oracle) {
    const job = {
        source: mutant.source,
        inputs: mutant.inputs,
        level: mutant.level,
        randomSeed: mutant.randomSeed,
        replay: options.replay,
        maxInputs: garden.trialMaxInputs(options, mutant.inputs)
    };
    if (oracle) {
        Object.assign(job, oracle);
    }
    try {
        return await garden.runChild(
            process.execPath,
            [workerPath],
            JSON.stringify(job),
            options.timeoutMs,
            function(child) { currentChild = child; }
        );
    } finally {
        currentChild = null;
    }
}

function publishTally(options, counts, trialIndex, saved, lastTrial, lastSaved) {
    const payload = garden.tallyPayload({
        seed: options.seed,
        forever: options.forever,
        trials: trialIndex + 1,
        saved: saved,
        counts: counts,
        lastTrial: lastTrial,
        lastSaved: lastSaved
    });
    garden.writeTally(options.output, payload);
    if (process.stderr.isTTY) {
        process.stderr.write('\r' + garden.formatForeverStatus(counts, trialIndex + 1, saved));
    }
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
        'compiler-warning': 0,
        crash: 0,
        timeout: 0,
        invariant: 0,
        nondeterministic: 0,
        'replay-divergence': 0,
        'semantic-mismatch': 0,
        baseline: 0,
        skipped: 0
    };
    let artifactIndex = 0;
    let lastSaved = null;
    let stopRequested = false;
    let finishing = false;

    function finishCampaign() {
        if (finishing) {
            return;
        }
        finishing = true;
        if (process.stderr.isTTY) {
            process.stderr.write('\n');
        }
        process.stdout.write(JSON.stringify(counts) + '\n');
    }

    function onSignal() {
        if (stopRequested) {
            if (currentChild) {
                try {
                    currentChild.kill('SIGKILL');
                } catch (error) {}
            }
            finishCampaign();
            process.exit(0);
            return;
        }
        stopRequested = true;
    }

    if (options.forever) {
        process.on('SIGINT', onSignal);
        process.on('SIGTERM', onSignal);
    }

    for (let i = 0; options.forever ? !stopRequested : i < options.count; i++) {
        const fixture = corpus[rng.integer(corpus.length)];
        const executedInputs = garden.prepareTrialInputs(fixture.inputs, rng, {
            maxInputs: options.maxInputs,
            extraInputs: options.extraInputs
        });
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
            publishTally(options, counts, i, artifactIndex, {
                index: i + 1,
                tally: 'skipped',
                mutator: null,
                fixtureName: fixture.name
            }, lastSaved);
            await new Promise(function(resolve) { setImmediate(resolve); });
            continue;
        }
        if (JSON.stringify(mutant.inputs || []) === JSON.stringify(fixture.inputs || [])) {
            mutant.inputs = executedInputs;
        }
        const baseline = await evaluateMutant({
            source: fixture.source,
            inputs: executedInputs,
            level: fixture.level,
            randomSeed: fixture.randomSeed
        }, options, garden.baselineOracleFields(fixture, options));
        const result = await evaluateMutant(mutant, options);
        const attributed = garden.attributeMonster(baseline, result);
        counts[attributed.tally] = (counts[attributed.tally] || 0) + 1;
        process.stdout.write(
            '#' + (i + 1) + ' ' + attributed.tally + ' ' + mutant.mutator + ' ' + mutant.fixtureName + '\n'
        );
        const lastTrial = {
            index: i + 1,
            tally: attributed.tally,
            mutator: mutant.mutator,
            fixtureName: mutant.fixtureName
        };
        if (!attributed.save) {
            publishTally(options, counts, i, artifactIndex, lastTrial, lastSaved);
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
                maxInputs: garden.trialMaxInputs(options, mutant.inputs),
                timeoutMs: options.timeoutMs,
                originalResult: result,
                minimizedResult: minimized.result,
                baselineKind: baseline.kind,
                baselineSignature: garden.failureSignature(baseline),
                signature: minimized.signature,
                shrinkSteps: minimized.steps
            }, null, 2) + '\n',
            'regression.js': garden.formatRegression(
                'monster garden ' + options.seed + ' ' + artifactIndex,
                minimized.source,
                { inputs: mutant.inputs, level: mutant.level, randomSeed: mutant.randomSeed }
            )
        });
        lastSaved = { dir: dirName, signature: minimized.signature, kind: result.kind };
        publishTally(options, counts, i, artifactIndex, lastTrial, lastSaved);
    }
    finishCampaign();
}

main().catch(function(error) {
    process.stderr.write(error.stack + '\n');
    process.exitCode = 1;
});
