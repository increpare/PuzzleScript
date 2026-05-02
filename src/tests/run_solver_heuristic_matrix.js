#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const GROUP_HEURISTICS = {
    all_on: [
        'winconditions',
        'all-on-count',
        'all-on-rowcol-tiebreak',
        'all-on-line-distance',
        'all-on-clear-path',
        'all-on-goal-coverage',
        'all-on-rowcol-matching',
        'all-on-player-nearest-tiebreak',
        'all-on-push-access',
        'all-on-dead-position',
    ],
    some_on: [
        'winconditions',
        'some-on-rowcol-tiebreak',
        'some-on-line-distance',
        'some-on-clear-path',
        'some-on-min',
        'some-on-player',
        'some-on-obstacle',
        'some-on-static-blockers',
        'some-on-static-blockers-tiebreak',
        'some-on-role-static',
    ],
    no_on: ['zero', 'winconditions', 'no-on-count', 'no-on-escape', 'no-on-player'],
    all_plain: ['zero', 'winconditions'],
    some_plain: ['zero', 'winconditions', 'some-plain-exists', 'some-plain-player'],
    no_plain: ['zero', 'winconditions', 'no-plain-count', 'no-plain-cluster', 'no-plain-player'],
};

const NUMERIC_FIELDS = [
    'elapsed_ms',
    'wall_ms',
    'generated',
    'unique_states',
    'expanded',
    'duplicates',
    'max_frontier',
    'solution_length',
    'heuristic_ms',
    'step_ms',
    'hash_ms',
    'queue_ms',
];

function usage() {
    console.error([
        'Usage: node src/tests/run_solver_heuristic_matrix.js <solver_tests_dir> <focus_index_or_group_json>',
        '  [--out PATH] [--solver PATH] [--timeout-ms N] [--runs N]',
        '  [--group NAME] [--heuristic NAME] [--max-targets N] [--quiet]',
    ].join('\n'));
    process.exit(1);
}

function parsePositiveInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`${label} must be a positive integer: ${value}`);
    }
    return parsed;
}

const args = process.argv.slice(2);
if (args.length < 2 || args.includes('--help') || args.includes('-h')) {
    usage();
}

const options = {
    corpusPath: path.resolve(args[0]),
    focusInput: path.resolve(args[1]),
    outPath: '/private/tmp/js_solver_heuristic_matrix.json',
    solverPath: path.resolve(__dirname, 'run_solver_tests_js.js'),
    timeoutMs: 30000,
    runs: 1,
    groupFilter: null,
    heuristicFilter: null,
    maxTargets: null,
    quiet: false,
};

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out' && index + 1 < args.length) {
        options.outPath = path.resolve(args[++index]);
    } else if (arg === '--solver' && index + 1 < args.length) {
        options.solverPath = path.resolve(args[++index]);
    } else if (arg === '--timeout-ms' && index + 1 < args.length) {
        options.timeoutMs = parsePositiveInt(args[++index], '--timeout-ms');
    } else if (arg === '--runs' && index + 1 < args.length) {
        options.runs = parsePositiveInt(args[++index], '--runs');
    } else if (arg === '--group' && index + 1 < args.length) {
        options.groupFilter = args[++index];
    } else if (arg === '--heuristic' && index + 1 < args.length) {
        options.heuristicFilter = args[++index];
    } else if (arg === '--max-targets' && index + 1 < args.length) {
        options.maxTargets = parsePositiveInt(args[++index], '--max-targets');
    } else if (arg === '--quiet') {
        options.quiet = true;
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

function log(message) {
    if (!options.quiet) {
        process.stderr.write(`${message}\n`);
    }
}

function median(values) {
    const finite = values.filter(Number.isFinite).sort((left, right) => left - right);
    if (finite.length === 0) {
        return null;
    }
    return finite[Math.floor(finite.length / 2)];
}

function targetKey(target) {
    return `${target.game}#${target.level}`;
}

function statusCounts(samples) {
    const counts = {};
    for (const sample of samples) {
        counts[sample.status] = (counts[sample.status] || 0) + 1;
    }
    return counts;
}

function representativeStatus(samples) {
    const counts = statusCounts(samples);
    return Object.keys(counts).sort((left, right) => {
        if (counts[right] !== counts[left]) return counts[right] - counts[left];
        return left.localeCompare(right);
    })[0] || 'unknown';
}

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function resolveManifestPath(baseDir, entryPath) {
    return path.isAbsolute(entryPath) ? entryPath : path.resolve(baseDir, entryPath);
}

function loadFocusManifests(inputPath) {
    const stat = fs.statSync(inputPath);
    const indexPath = stat.isDirectory() ? path.join(inputPath, 'index.json') : inputPath;
    const root = readJson(indexPath);
    const baseDir = path.dirname(indexPath);
    if (root.kind === 'solver_wincondition_focus_group') {
        return [root];
    }
    if (root.kind !== 'solver_wincondition_focus_index') {
        throw new Error(`Unsupported focus manifest kind: ${root.kind || 'unknown'}`);
    }
    const manifests = [];
    for (const [group, entry] of Object.entries(root.groups || {})) {
        if (options.groupFilter !== null && group !== options.groupFilter) {
            continue;
        }
        manifests.push(readJson(resolveManifestPath(baseDir, entry.path)));
    }
    return manifests;
}

function runTarget(target, heuristic, runIndex) {
    const solverArgs = [
        options.corpusPath,
        '--timeout-ms', String(options.timeoutMs),
        '--strategy', 'weighted-astar',
        '--astar-weight', '2',
        '--solver-heuristic', heuristic,
        '--game', target.game,
        '--level', String(target.level),
        '--no-solutions',
        '--quiet',
        '--json',
    ];
    const started = process.hrtime.bigint();
    const result = spawnSync(process.execPath, [options.solverPath, ...solverArgs], {
        encoding: 'utf8',
        maxBuffer: 512 * 1024 * 1024,
    });
    const wallMs = Number(process.hrtime.bigint() - started) / 1e6;
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        throw new Error(`solver failed for ${targetKey(target)} heuristic=${heuristic} run=${runIndex + 1}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    const json = JSON.parse(result.stdout);
    if (json.results.length !== 1) {
        throw new Error(`expected one result for ${targetKey(target)}, got ${json.results.length}`);
    }
    const sample = json.results[0];
    return {
        run: runIndex + 1,
        status: sample.status,
        wall_ms: wallMs,
        elapsed_ms: sample.elapsed_ms,
        generated: sample.generated,
        unique_states: sample.unique_states,
        expanded: sample.expanded,
        duplicates: sample.duplicates,
        max_frontier: sample.max_frontier,
        solution_length: sample.solution_length,
        heuristic_ms: sample.heuristic_ms,
        step_ms: sample.step_ms,
        hash_ms: sample.hash_ms,
        queue_ms: sample.queue_ms,
        strategy: sample.strategy,
        heuristic: sample.heuristic,
    };
}

function summarizeTarget(target, samples) {
    const result = {
        game: target.game,
        level: target.level,
        status: representativeStatus(samples),
        status_counts: statusCounts(samples),
        median: {},
        samples,
    };
    for (const field of NUMERIC_FIELDS) {
        result.median[field] = median(samples.map((sample) => sample[field]));
    }
    return result;
}

function summarizeHeuristic(targetSummaries) {
    const allSamples = targetSummaries.flatMap((target) => target.samples);
    const totals = {
        generated: 0,
        unique_states: 0,
        expanded: 0,
        duplicates: 0,
        elapsed_ms: 0,
        heuristic_ms: 0,
    };
    let solvedTargets = 0;
    for (const target of targetSummaries) {
        if (target.status === 'solved') {
            solvedTargets++;
        }
        for (const field of Object.keys(totals)) {
            totals[field] += target.median[field] || 0;
        }
    }
    return {
        target_count: targetSummaries.length,
        solved_targets: solvedTargets,
        status_counts: statusCounts(allSamples),
        totals: {
            ...totals,
            generated_transitions: totals.generated,
            unique_states_reached: totals.unique_states,
            states_expanded: totals.expanded,
            duplicate_transitions: totals.duplicates,
        },
        medians: Object.fromEntries(NUMERIC_FIELDS.map((field) => [
            field,
            median(targetSummaries.map((target) => target.median[field])),
        ])),
    };
}

function geomeanRatio(currentTargets, baselineTargets, field) {
    const baselineByKey = new Map(baselineTargets.map((target) => [targetKey(target), target]));
    const ratios = [];
    for (const current of currentTargets) {
        const baseline = baselineByKey.get(targetKey(current));
        if (!baseline || current.status !== 'solved' || baseline.status !== 'solved') {
            continue;
        }
        const currentValue = current.median[field];
        const baselineValue = baseline.median[field];
        if (!Number.isFinite(currentValue) || !Number.isFinite(baselineValue)) {
            continue;
        }
        ratios.push((currentValue + 1) / (baselineValue + 1));
    }
    if (ratios.length === 0) {
        return null;
    }
    return Math.exp(ratios.reduce((sum, ratio) => sum + Math.log(ratio), 0) / ratios.length);
}

function heuristicsForGroup(group) {
    const heuristics = GROUP_HEURISTICS[group] || ['zero', 'winconditions'];
    return options.heuristicFilter === null
        ? heuristics
        : heuristics.filter((heuristic) => heuristic === options.heuristicFilter);
}

function main() {
    const manifests = loadFocusManifests(options.focusInput);
    const output = {
        schema_version: 1,
        kind: 'solver_heuristic_matrix',
        generated_at: new Date().toISOString(),
        corpus: options.corpusPath,
        focus_input: options.focusInput,
        solver: options.solverPath,
        strategy: 'weighted-astar',
        astar_weight: 2,
        timeout_ms: options.timeoutMs,
        runs: options.runs,
        groups: {},
    };

    for (const manifest of manifests) {
        const group = manifest.group;
        if (options.groupFilter !== null && group !== options.groupFilter) {
            continue;
        }
        let targets = manifest.targets || [];
        if (options.maxTargets !== null) {
            targets = targets.slice(0, options.maxTargets);
        }
        output.groups[group] = {
            target_count: targets.length,
            heuristics: {},
        };
        const targetSummariesByHeuristic = new Map();
        for (const heuristic of heuristicsForGroup(group)) {
            log(`heuristic_matrix group=${group} heuristic=${heuristic} targets=${targets.length}`);
            const targetSummaries = [];
            for (const target of targets) {
                const samples = [];
                for (let runIndex = 0; runIndex < options.runs; runIndex++) {
                    const sample = runTarget(target, heuristic, runIndex);
                    samples.push(sample);
                    log(`heuristic_matrix target=${targetKey(target)} heuristic=${heuristic} run=${runIndex + 1}/${options.runs} status=${sample.status} generated=${sample.generated} unique=${sample.unique_states}`);
                }
                targetSummaries.push(summarizeTarget(target, samples));
            }
            targetSummariesByHeuristic.set(heuristic, targetSummaries);
            output.groups[group].heuristics[heuristic] = {
                summary: summarizeHeuristic(targetSummaries),
                targets: targetSummaries,
            };
        }

        const baselineTargets = targetSummariesByHeuristic.get('winconditions') || null;
        if (baselineTargets) {
            const baselineSolved = output.groups[group].heuristics.winconditions.summary.solved_targets;
            for (const [heuristic, targetSummaries] of targetSummariesByHeuristic) {
                const entry = output.groups[group].heuristics[heuristic];
                entry.comparison_to_winconditions = {
                    generated_geomean_ratio: geomeanRatio(targetSummaries, baselineTargets, 'generated'),
                    unique_states_geomean_ratio: geomeanRatio(targetSummaries, baselineTargets, 'unique_states'),
                    expanded_geomean_ratio: geomeanRatio(targetSummaries, baselineTargets, 'expanded'),
                    duplicates_geomean_ratio: geomeanRatio(targetSummaries, baselineTargets, 'duplicates'),
                    risky_solved_regression: entry.summary.solved_targets < baselineSolved,
                };
                const ratio = entry.comparison_to_winconditions.generated_geomean_ratio;
                process.stdout.write(
                    `heuristic_matrix group=${group}` +
                    ` heuristic=${heuristic}` +
                    ` solved=${entry.summary.solved_targets}/${targets.length}` +
                    ` generated=${entry.summary.totals.generated}` +
                    ` unique=${entry.summary.totals.unique_states}` +
                    ` expanded=${entry.summary.totals.expanded}` +
                    ` duplicates=${entry.summary.totals.duplicates}` +
                    ` generated_ratio=${ratio === null ? 'n/a' : ratio.toFixed(3)}` +
                    ` risky=${entry.comparison_to_winconditions.risky_solved_regression ? 'yes' : 'no'}\n`
                );
            }
        }
    }

    fs.mkdirSync(path.dirname(options.outPath), { recursive: true });
    fs.writeFileSync(options.outPath, `${JSON.stringify(output, null, 2)}\n`);
    process.stdout.write(`heuristic_matrix wrote ${options.outPath}\n`);
}

main();
