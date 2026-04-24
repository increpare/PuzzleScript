#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/mine_solver_near_threshold.js <puzzlescript_solver> <solver_tests_dir> [--timeouts-ms A,B,C] [--out PATH] [--strategy NAME] [--near-ratio N] [--game NAME] [--level N] [--max-targets N]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const solverPath = path.resolve(args[0]);
const corpusPath = path.resolve(args[1]);
let timeoutsMs = [50, 100, 250, 500];
let outPath = path.resolve('build/native/solver_pippable_targets.json');
let strategy = 'weighted-astar';
let nearRatio = 0.5;
let gameFilter = null;
let levelFilter = null;
let maxTargets = null;

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

function parseTimeoutList(value) {
    const parsed = value.split(',').map((part) => parsePositiveInt(part.trim(), '--timeouts-ms'));
    if (parsed.length === 0) {
        throw new Error('--timeouts-ms must include at least one timeout');
    }
    return parsed;
}

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--timeouts-ms' && index + 1 < args.length) {
        timeoutsMs = parseTimeoutList(args[++index]);
    } else if (arg === '--timeout-ms' && index + 1 < args.length) {
        timeoutsMs = [parsePositiveInt(args[++index], '--timeout-ms')];
    } else if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--strategy' && index + 1 < args.length) {
        strategy = args[++index];
    } else if (arg === '--near-ratio' && index + 1 < args.length) {
        nearRatio = Number.parseFloat(args[++index]);
        if (!Number.isFinite(nearRatio) || nearRatio < 0) {
            throw new Error(`--near-ratio must be a non-negative number: ${nearRatio}`);
        }
    } else if (arg === '--game' && index + 1 < args.length) {
        gameFilter = args[++index];
    } else if (arg === '--level' && index + 1 < args.length) {
        levelFilter = parseNonNegativeInt(args[++index], '--level');
    } else if (arg === '--max-targets' && index + 1 < args.length) {
        maxTargets = parsePositiveInt(args[++index], '--max-targets');
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

timeoutsMs = Array.from(new Set(timeoutsMs)).sort((a, b) => a - b);

function resultKey(result) {
    return `${result.game}#${result.level}`;
}

function runSolver(timeoutMs) {
    const commandArgs = [
        corpusPath,
        '--timeout-ms', String(timeoutMs),
        '--jobs', '1',
        '--strategy', strategy,
        '--no-solutions',
        '--quiet',
        '--json',
    ];
    if (gameFilter !== null) {
        commandArgs.push('--game', gameFilter);
    }
    if (levelFilter !== null) {
        commandArgs.push('--level', String(levelFilter));
    }

    const started = process.hrtime.bigint();
    const result = spawnSync(solverPath, commandArgs, {
        encoding: 'utf8',
        maxBuffer: 512 * 1024 * 1024,
    });
    const wallMs = Number(process.hrtime.bigint() - started) / 1e6;
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        throw new Error(`solver exited ${result.status} at timeout=${timeoutMs}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    const json = JSON.parse(result.stdout);
    if (json.totals.errors !== 0) {
        throw new Error(`solver reported errors=${json.totals.errors} at timeout=${timeoutMs}`);
    }
    process.stderr.write(`solver_mine timeout_ms=${timeoutMs} levels=${json.totals.levels} solved=${json.totals.solved} timeout=${json.totals.timeout} wall_ms=${wallMs.toFixed(1)}\n`);
    return { timeout_ms: timeoutMs, wall_ms: wallMs, totals: json.totals, results: json.results };
}

const runs = timeoutsMs.map(runSolver);
const byKey = new Map();
for (const run of runs) {
    for (const result of run.results) {
        const key = resultKey(result);
        if (!byKey.has(key)) {
            byKey.set(key, { game: result.game, level: result.level, observations: [] });
        }
        byKey.get(key).observations.push({
            timeout_ms: run.timeout_ms,
            status: result.status,
            elapsed_ms: result.elapsed_ms,
            expanded: result.expanded,
            generated: result.generated,
            unique_states: result.unique_states,
            duplicates: result.duplicates,
            max_frontier: result.max_frontier,
            solution_length: result.solution_length,
        });
    }
}

const targets = [];
for (const entry of byKey.values()) {
    entry.observations.sort((a, b) => a.timeout_ms - b.timeout_ms);
    const firstSolvedIndex = entry.observations.findIndex((observation) => observation.status === 'solved');
    if (firstSolvedIndex < 0) {
        continue;
    }

    const solved = entry.observations[firstSolvedIndex];
    const previous = firstSolvedIndex > 0 ? entry.observations[firstSolvedIndex - 1] : null;
    const timedOutBeforeSolve = entry.observations.slice(0, firstSolvedIndex).some((observation) => observation.status === 'timeout');
    const elapsedRatio = solved.timeout_ms > 0 ? solved.elapsed_ms / solved.timeout_ms : 0;
    if (!timedOutBeforeSolve && elapsedRatio < nearRatio) {
        continue;
    }

    targets.push({
        game: entry.game,
        level: entry.level,
        first_solved_timeout_ms: solved.timeout_ms,
        previous_timeout_ms: previous ? previous.timeout_ms : null,
        previous_status: previous ? previous.status : null,
        solved_elapsed_ms: solved.elapsed_ms,
        solved_elapsed_ratio: elapsedRatio,
        solved_expanded: solved.expanded,
        solved_generated: solved.generated,
        observations: entry.observations,
    });
}

targets.sort((a, b) => {
    if (a.first_solved_timeout_ms !== b.first_solved_timeout_ms) {
        return a.first_solved_timeout_ms - b.first_solved_timeout_ms;
    }
    if (b.solved_elapsed_ratio !== a.solved_elapsed_ratio) {
        return b.solved_elapsed_ratio - a.solved_elapsed_ratio;
    }
    return `${a.game}#${a.level}`.localeCompare(`${b.game}#${b.level}`);
});

const selectedTargets = maxTargets === null ? targets : targets.slice(0, maxTargets);
const manifest = {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    solver: solverPath,
    corpus: corpusPath,
    strategy,
    jobs: '1',
    timeouts_ms: timeoutsMs,
    near_ratio: nearRatio,
    filters: {
        game: gameFilter,
        level: levelFilter,
    },
    target_count: selectedTargets.length,
    candidate_count: targets.length,
    totals_by_timeout: runs.map((run) => ({
        timeout_ms: run.timeout_ms,
        wall_ms: run.wall_ms,
        totals: run.totals,
    })),
    targets: selectedTargets,
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`solver_mine wrote ${outPath} targets=${selectedTargets.length} candidates=${targets.length}\n`);
