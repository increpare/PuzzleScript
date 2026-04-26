#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/run_solver_level_benchmark.js <puzzlescript_solver> <solver_tests_dir> <manifest> [--runs N] [--out PATH] [--timeout-ms N] [--strategy NAME] [--game NAME] [--level N] [--profile-runtime-counters] [--solver-arg ARG ...]');
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
const solverExtraArgs = [];

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
    } else if (arg === '--solver-arg' && index + 1 < args.length) {
        solverExtraArgs.push(args[++index]);
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

const timingFields = [
    'compile_ms',
    'load_ms',
    'clone_ms',
    'step_ms',
    'hash_ms',
    'queue_ms',
    'frontier_pop_ms',
    'frontier_push_ms',
    'visited_lookup_ms',
    'visited_insert_ms',
    'visited_lookup_probes',
    'visited_insert_probes',
    'visited_grows',
    'visited_capacity',
    'visited_max_probe',
    'visited_key_collisions',
    'compact_state_bytes',
    'compact_max_state_bytes',
    'compact_turn_attempts',
    'compact_turn_hits',
    'compact_turn_fallbacks',
    'compact_turn_unsupported',
    'node_store_ms',
    'heuristic_ms',
    'solved_check_ms',
    'timeout_check_ms',
    'reconstruct_ms',
    'unattributed_ms',
];

function readField(source, name, oldName = null) {
    if (source[name] !== undefined) return source[name];
    if (oldName !== null && source[oldName] !== undefined) return source[oldName];
    return undefined;
}

function copyTimingFields(target, source) {
    for (const fieldName of timingFields) {
        const oldField = fieldName.startsWith('compact_turn_')
            ? fieldName.replace('compact_turn_', 'compact_tick_')
            : null;
        target[fieldName] = oldField === null ? source[fieldName] : readField(source, fieldName, oldField);
    }
}

function medianTiming(samples, field) {
    return median(samples.map((sample) => sample[field]).filter(Number.isFinite));
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
    solverArgs.push(...solverExtraArgs);
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
    const sample = {
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
        specialized_rulegroups_attached: Boolean(readField(solverResult, 'specialized_rulegroups_attached', 'compiled_rules_attached')),
        compiled_rules_attached: Boolean(solverResult.compiled_rules_attached),
        compiled_tick_attached: Boolean(solverResult.compiled_tick_attached),
        specialized_compact_turn_attached: Boolean(readField(solverResult, 'specialized_compact_turn_attached', 'compiled_compact_tick_attached')),
        compile_ms: solverResult.compile_ms,
        load_ms: solverResult.load_ms,
        clone_ms: solverResult.clone_ms,
        step_ms: solverResult.step_ms,
        hash_ms: solverResult.hash_ms,
        queue_ms: solverResult.queue_ms,
        reconstruct_ms: solverResult.reconstruct_ms,
        runtime_counters: profileRuntimeCounters ? parseRuntimeCounters(result.stderr) : null,
    };
    copyTimingFields(sample, solverResult);
    return sample;
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

    const targetMedian = {
        wall_ms: median(samples.map((sample) => sample.wall_ms)),
        elapsed_ms: median(samples.map((sample) => sample.elapsed_ms)),
        expanded: median(samples.map((sample) => sample.expanded)),
        generated: median(samples.map((sample) => sample.generated)),
        unique_states: median(samples.map((sample) => sample.unique_states)),
        duplicates: median(samples.map((sample) => sample.duplicates)),
        max_frontier: median(samples.map((sample) => sample.max_frontier)),
        solution_length: median(samples.map((sample) => sample.solution_length)),
    };
    for (const field of timingFields) {
        targetMedian[field] = medianTiming(samples, field);
    }

    summaries.push({
        game: target.game,
        level: target.level,
        timeout_ms: timeoutMs,
        strategy,
        runs,
        status_counts: statusCounts(samples),
        median: targetMedian,
        mined_target: target,
        samples,
    });
}

const allSamples = summaries.flatMap((summary) => summary.samples);
const outputMedian = {
    wall_ms: median(allSamples.map((sample) => sample.wall_ms)),
    elapsed_ms: median(allSamples.map((sample) => sample.elapsed_ms)),
    expanded: median(allSamples.map((sample) => sample.expanded)),
    generated: median(allSamples.map((sample) => sample.generated)),
};
for (const field of timingFields) {
    outputMedian[field] = medianTiming(allSamples, field);
}

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
    solver_extra_args: solverExtraArgs,
    target_count: summaries.length,
    filters: {
        game: gameFilter,
        level: levelFilter,
    },
    timeout_override_ms: timeoutOverrideMs,
    median: outputMedian,
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
