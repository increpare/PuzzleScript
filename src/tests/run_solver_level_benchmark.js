#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/run_solver_level_benchmark.js <puzzlescript_solver> <solver_tests_dir> <manifest> [--runs N] [--out PATH] [--timeout-ms N] [--strategy NAME] [--game NAME] [--level N] [--profile-runtime-counters]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 3) {
    usage();
}

const solverPath = path.resolve(args[0]);
const corpusPath = path.resolve(args[1]);
const manifestPath = path.resolve(args[2]);
let runs = 5;
let outPath = path.resolve('build/native/solver_target_benchmark.json');
let timeoutOverrideMs = null;
let strategyOverride = null;
let gameFilter = null;
let levelFilter = null;
let profileRuntimeCounters = false;

function parsePositiveInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`${label} must be a positive integer: ${value}`);
    }
    return parsed;
}

function parseNonNegativeInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`${label} must be a non-negative integer: ${value}`);
    }
    return parsed;
}

for (let index = 3; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--runs' && index + 1 < args.length) {
        runs = parsePositiveInt(args[++index], '--runs');
    } else if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--timeout-ms' && index + 1 < args.length) {
        timeoutOverrideMs = parsePositiveInt(args[++index], '--timeout-ms');
    } else if (arg === '--strategy' && index + 1 < args.length) {
        strategyOverride = args[++index];
    } else if (arg === '--game' && index + 1 < args.length) {
        gameFilter = args[++index];
    } else if (arg === '--level' && index + 1 < args.length) {
        levelFilter = parseNonNegativeInt(args[++index], '--level');
    } else if (arg === '--profile-runtime-counters') {
        profileRuntimeCounters = true;
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

function median(values) {
    if (values.length === 0) {
        return null;
    }
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

function statusCounts(samples) {
    const counts = {};
    for (const sample of samples) {
        counts[sample.status] = (counts[sample.status] || 0) + 1;
    }
    return counts;
}

function resultKey(result) {
    return `${result.game}#${result.level}`;
}

function targetKey(target) {
    return `${target.game}#${target.level}`;
}

function parseRuntimeCounters(stderr) {
    const match = stderr.match(/solver_runtime_counters ([^\n]+)/);
    if (!match) {
        return null;
    }
    const counters = {};
    for (const part of match[1].trim().split(/\s+/)) {
        const [key, value] = part.split('=');
        if (!key || value === undefined) {
            continue;
        }
        const parsed = Number.parseInt(value, 10);
        counters[key] = Number.isFinite(parsed) ? parsed : value;
    }
    return counters;
}

function runTarget(target, runIndex, strategy, timeoutMs) {
    const solverArgs = [
        corpusPath,
        '--timeout-ms', String(timeoutMs),
        '--jobs', '1',
        '--strategy', strategy,
        '--game', target.game,
        '--level', String(target.level),
        '--no-solutions',
        '--quiet',
        '--json',
    ];
    if (profileRuntimeCounters) {
        solverArgs.push('--profile-runtime-counters');
    }
    const started = process.hrtime.bigint();
    const result = spawnSync(solverPath, solverArgs, {
        encoding: 'utf8',
        maxBuffer: 512 * 1024 * 1024,
    });
    const wallMs = Number(process.hrtime.bigint() - started) / 1e6;
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        throw new Error(`solver exited ${result.status} for ${targetKey(target)} run=${runIndex + 1}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    const json = JSON.parse(result.stdout);
    if (json.totals.errors !== 0) {
        throw new Error(`solver reported errors=${json.totals.errors} for ${targetKey(target)} run=${runIndex + 1}`);
    }
    if (json.results.length !== 1) {
        throw new Error(`expected one result for ${targetKey(target)}, got ${json.results.length}`);
    }
    const solverResult = json.results[0];
    if (solverResult.game !== target.game || solverResult.level !== target.level) {
        throw new Error(`solver returned ${resultKey(solverResult)} for requested ${targetKey(target)}`);
    }
    return {
        run: runIndex + 1,
        status: solverResult.status,
        wall_ms: wallMs,
        elapsed_ms: solverResult.elapsed_ms,
        expanded: solverResult.expanded,
        generated: solverResult.generated,
        unique_states: solverResult.unique_states,
        duplicates: solverResult.duplicates,
        max_frontier: solverResult.max_frontier,
        solution_length: solverResult.solution_length,
        timeout_ms: solverResult.timeout_ms,
        compiled_rules_attached: Boolean(solverResult.compiled_rules_attached),
        compiled_tick_attached: Boolean(solverResult.compiled_tick_attached),
        compile_ms: solverResult.compile_ms,
        load_ms: solverResult.load_ms,
        clone_ms: solverResult.clone_ms,
        step_ms: solverResult.step_ms,
        hash_ms: solverResult.hash_ms,
        queue_ms: solverResult.queue_ms,
        reconstruct_ms: solverResult.reconstruct_ms,
        runtime_counters: profileRuntimeCounters ? parseRuntimeCounters(result.stderr) : null,
    };
}

if (!fs.existsSync(manifestPath)) {
    throw new Error(`manifest does not exist: ${manifestPath}`);
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const manifestTargets = manifest.targets || [];
const strategy = strategyOverride || manifest.strategy || 'weighted-astar';
let targets = manifestTargets;
if (gameFilter !== null) {
    targets = targets.filter((target) => target.game === gameFilter);
}
if (levelFilter !== null) {
    targets = targets.filter((target) => target.level === levelFilter);
}

const summaries = [];
for (const target of targets) {
    const timeoutMs = timeoutOverrideMs || target.first_solved_timeout_ms;
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
        throw new Error(`target ${targetKey(target)} does not have a usable timeout`);
    }

    const samples = [];
    for (let index = 0; index < runs; index++) {
        const sample = runTarget(target, index, strategy, timeoutMs);
        samples.push(sample);
        process.stderr.write(`solver_target_benchmark target=${targetKey(target)} run=${index + 1}/${runs} status=${sample.status} wall_ms=${sample.wall_ms.toFixed(1)} elapsed_ms=${sample.elapsed_ms}\n`);
    }

    summaries.push({
        game: target.game,
        level: target.level,
        timeout_ms: timeoutMs,
        strategy,
        runs,
        status_counts: statusCounts(samples),
        median: {
            wall_ms: median(samples.map((sample) => sample.wall_ms)),
            elapsed_ms: median(samples.map((sample) => sample.elapsed_ms)),
            expanded: median(samples.map((sample) => sample.expanded)),
            generated: median(samples.map((sample) => sample.generated)),
            unique_states: median(samples.map((sample) => sample.unique_states)),
            duplicates: median(samples.map((sample) => sample.duplicates)),
            max_frontier: median(samples.map((sample) => sample.max_frontier)),
            solution_length: median(samples.map((sample) => sample.solution_length)),
            compile_ms: median(samples.map((sample) => sample.compile_ms).filter(Number.isFinite)),
            load_ms: median(samples.map((sample) => sample.load_ms).filter(Number.isFinite)),
            clone_ms: median(samples.map((sample) => sample.clone_ms).filter(Number.isFinite)),
            step_ms: median(samples.map((sample) => sample.step_ms).filter(Number.isFinite)),
            hash_ms: median(samples.map((sample) => sample.hash_ms).filter(Number.isFinite)),
            queue_ms: median(samples.map((sample) => sample.queue_ms).filter(Number.isFinite)),
            reconstruct_ms: median(samples.map((sample) => sample.reconstruct_ms).filter(Number.isFinite)),
        },
        mined_target: target,
        samples,
    });
}

const allSamples = summaries.flatMap((summary) => summary.samples);
const output = {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    solver: solverPath,
    corpus: corpusPath,
    manifest: manifestPath,
    strategy,
    jobs: '1',
    runs_per_target: runs,
    profile_runtime_counters: profileRuntimeCounters,
    target_count: summaries.length,
    filters: {
        game: gameFilter,
        level: levelFilter,
    },
    timeout_override_ms: timeoutOverrideMs,
    median: {
        wall_ms: median(allSamples.map((sample) => sample.wall_ms)),
        elapsed_ms: median(allSamples.map((sample) => sample.elapsed_ms)),
        expanded: median(allSamples.map((sample) => sample.expanded)),
        generated: median(allSamples.map((sample) => sample.generated)),
        compile_ms: median(allSamples.map((sample) => sample.compile_ms).filter(Number.isFinite)),
        load_ms: median(allSamples.map((sample) => sample.load_ms).filter(Number.isFinite)),
        clone_ms: median(allSamples.map((sample) => sample.clone_ms).filter(Number.isFinite)),
        step_ms: median(allSamples.map((sample) => sample.step_ms).filter(Number.isFinite)),
        hash_ms: median(allSamples.map((sample) => sample.hash_ms).filter(Number.isFinite)),
        queue_ms: median(allSamples.map((sample) => sample.queue_ms).filter(Number.isFinite)),
        reconstruct_ms: median(allSamples.map((sample) => sample.reconstruct_ms).filter(Number.isFinite)),
    },
    targets: summaries,
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, `${JSON.stringify(output, null, 2)}\n`);
process.stdout.write(
    `solver_target_benchmark summary targets=${summaries.length}` +
    ` runs=${runs}` +
    ` median_wall_ms=${output.median.wall_ms}` +
    ` median_elapsed_ms=${output.median.elapsed_ms}` +
    ` median_expanded=${output.median.expanded}` +
    ` median_generated=${output.median.generated}\n`
);
process.stdout.write(`solver_target_benchmark wrote ${outPath} targets=${summaries.length} samples=${allSamples.length}\n`);
