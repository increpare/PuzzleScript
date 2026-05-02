#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

const GROUPS = ['all_on', 'some_on', 'no_on', 'all_plain', 'some_plain', 'no_plain'];
const QUANTIFIER_NAMES = {
    '-1': 'no',
    0: 'some',
    1: 'all',
};

function usage() {
    console.error([
        'Usage: node src/tests/mine_solver_wincondition_focus.js <solver_tests_dir>',
        '  [--out-dir DIR] [--weighted-json PATH] [--all-strategies-json PATH] [--features-json PATH]',
        '  [--max-targets N] [--min-elapsed-ms N] [--preferred-max-elapsed-ms N] [--hard-max-elapsed-ms N]',
        '  [--benchmark-timeout-ms N] [--solver PATH] [--supplement-timeout-ms N] [--supplement-probe-limit N]',
        '  [--no-supplement] [--quiet]',
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

function parseNonNegativeInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`${label} must be a non-negative integer: ${value}`);
    }
    return parsed;
}

const args = process.argv.slice(2);
if (args.length < 1 || args.includes('--help') || args.includes('-h')) {
    usage();
}

const options = {
    corpusPath: path.resolve(args[0]),
    outDir: '/private/tmp/js_solver_wincondition_focus',
    weightedJson: '/private/tmp/js_solver_all_500_weighted.json',
    allStrategiesJson: '/private/tmp/js_solver_focus_all_strategies_2s.json',
    featuresJson: '/private/tmp/js_solver_focus_features.json',
    maxTargets: 50,
    minElapsedMs: 250,
    preferredMaxElapsedMs: 1000,
    hardMaxElapsedMs: 5000,
    benchmarkTimeoutMs: 30000,
    solverPath: path.resolve(__dirname, 'run_solver_tests_js.js'),
    supplement: true,
    supplementTimeoutMs: 5000,
    supplementProbeLimit: 200,
    quiet: false,
};

for (let index = 1; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out-dir' && index + 1 < args.length) {
        options.outDir = path.resolve(args[++index]);
    } else if (arg === '--weighted-json' && index + 1 < args.length) {
        options.weightedJson = path.resolve(args[++index]);
    } else if (arg === '--all-strategies-json' && index + 1 < args.length) {
        options.allStrategiesJson = path.resolve(args[++index]);
    } else if (arg === '--features-json' && index + 1 < args.length) {
        options.featuresJson = path.resolve(args[++index]);
    } else if (arg === '--max-targets' && index + 1 < args.length) {
        options.maxTargets = parsePositiveInt(args[++index], '--max-targets');
    } else if (arg === '--min-elapsed-ms' && index + 1 < args.length) {
        options.minElapsedMs = parseNonNegativeInt(args[++index], '--min-elapsed-ms');
    } else if (arg === '--preferred-max-elapsed-ms' && index + 1 < args.length) {
        options.preferredMaxElapsedMs = parsePositiveInt(args[++index], '--preferred-max-elapsed-ms');
    } else if (arg === '--hard-max-elapsed-ms' && index + 1 < args.length) {
        options.hardMaxElapsedMs = parsePositiveInt(args[++index], '--hard-max-elapsed-ms');
    } else if (arg === '--benchmark-timeout-ms' && index + 1 < args.length) {
        options.benchmarkTimeoutMs = parsePositiveInt(args[++index], '--benchmark-timeout-ms');
    } else if (arg === '--solver' && index + 1 < args.length) {
        options.solverPath = path.resolve(args[++index]);
    } else if (arg === '--supplement-timeout-ms' && index + 1 < args.length) {
        options.supplementTimeoutMs = parsePositiveInt(args[++index], '--supplement-timeout-ms');
    } else if (arg === '--supplement-probe-limit' && index + 1 < args.length) {
        options.supplementProbeLimit = parseNonNegativeInt(args[++index], '--supplement-probe-limit');
    } else if (arg === '--no-supplement') {
        options.supplement = false;
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

function normalizeGamePath(filePath) {
    return path.relative(options.corpusPath, filePath).split(path.sep).join('/');
}

function isHiddenRelativePath(relativePath) {
    return relativePath.split(path.sep).some((part) => part.startsWith('.'));
}

function listGameFiles(root) {
    const files = [];
    function visit(dir) {
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            const relative = path.relative(root, full);
            if (isHiddenRelativePath(relative)) {
                continue;
            }
            if (entry.isDirectory()) {
                visit(full);
            } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.txt')) {
                files.push(full);
            }
        }
    }
    visit(root);
    return files.sort((left, right) => normalizeGamePath(left).localeCompare(normalizeGamePath(right)));
}

function targetKey(value) {
    return `${value.game}#${value.level}`;
}

function masksEqual(left, right) {
    if (left === right) {
        return true;
    }
    if (!left || !right || !left.data || !right.data || left.data.length !== right.data.length) {
        return false;
    }
    for (let word = 0; word < left.data.length; word++) {
        if ((left.data[word] | 0) !== (right.data[word] | 0)) {
            return false;
        }
    }
    return true;
}

function ruleGroupsUseCommand(groups, commandName) {
    for (const group of groups || []) {
        for (const rule of group || []) {
            for (const command of (rule && rule.commands) || []) {
                const name = Array.isArray(command) ? command[0] : command;
                if (name === commandName) {
                    return true;
                }
            }
        }
    }
    return false;
}

function ruleGroupsUseRandom(groups) {
    for (const group of groups || []) {
        for (const rule of group || []) {
            if (!rule) {
                continue;
            }
            if (rule.isRandom) {
                return true;
            }
            for (const row of rule.cells || []) {
                const replacement = row && row.replacement;
                if (replacement && (
                    (replacement.randomEntityMask && !replacement.randomEntityMask.iszero()) ||
                    (replacement.randomDirMask && !replacement.randomDirMask.iszero())
                )) {
                    return true;
                }
            }
        }
    }
    return false;
}

function countRules(groups) {
    let total = 0;
    for (const group of groups || []) {
        total += group.length;
    }
    return total;
}

function levelObjectBits(levelEntry) {
    const data = levelEntry && (levelEntry.objects || levelEntry.dat);
    if (!data) {
        return 0;
    }
    let bits = 0;
    for (const word of data) {
        let value = word >>> 0;
        while (value !== 0) {
            value &= value - 1;
            bits++;
        }
    }
    return bits;
}

function compileGameFeatures(gameFile) {
    const game = normalizeGamePath(gameFile);
    let source = fs.readFileSync(gameFile, 'utf8');
    if (!source.endsWith('\n')) {
        source += '\n';
    }
    if (typeof resetParserErrorState === 'function') {
        resetParserErrorState();
    }
    unitTesting = true;
    lazyFunctionGeneration = false;
    try {
        compile(['loadLevel', 0], source, `wincondition-focus:${game}:0`);
    } finally {
        unitTesting = false;
        lazyFunctionGeneration = true;
    }

    if (errorCount > 0) {
        return {
            game,
            status: 'compile_error',
            error: errorStrings.map(stripHTMLTags).join('\n'),
            levels: new Map(),
        };
    }

    const ruleGroups = [...(state.rules || []), ...(state.lateRules || [])];
    const hasWinCommand = ruleGroupsUseCommand(ruleGroups, 'win');
    const winconditions = state.winconditions || [];
    const allObjectsMask = state.objectMasks && state.objectMasks["\nall\n"];
    let group = null;
    let excludeReason = null;

    if (hasWinCommand) {
        excludeReason = 'win_command';
    } else if (winconditions.length !== 1) {
        excludeReason = winconditions.length === 0 ? 'no_winconditions' : 'multiple_winconditions';
    } else {
        const condition = winconditions[0];
        const quantifier = QUANTIFIER_NAMES[String(condition[0])];
        const shape = allObjectsMask && masksEqual(condition[2], allObjectsMask) ? 'plain' : 'on';
        group = quantifier ? `${quantifier}_${shape}` : null;
        if (!GROUPS.includes(group)) {
            excludeReason = 'unsupported_wincondition';
        }
    }

    const baseFeatures = {
        game,
        group,
        exclude_reason: excludeReason,
        object_count: state.objectCount || 0,
        layer_count: state.collisionLayers ? state.collisionLayers.length : 0,
        rule_count: countRules(state.rules) + countRules(state.lateRules),
        rule_group_count: state.rules ? state.rules.length : 0,
        late_rule_count: countRules(state.lateRules),
        late_rule_group_count: state.lateRules ? state.lateRules.length : 0,
        wincondition_count: winconditions.length,
        has_win_command: hasWinCommand,
        has_again: ruleGroupsUseCommand(ruleGroups, 'again'),
        has_rigid: Boolean(state.rigid),
        uses_random: ruleGroupsUseRandom(ruleGroups),
    };

    const levels = new Map();
    for (let levelIndex = 0; levelIndex < (state.levels || []).length; levelIndex++) {
        const levelEntry = state.levels[levelIndex];
        levels.set(levelIndex, {
            ...baseFeatures,
            level: levelIndex,
            is_message: Boolean(levelEntry && levelEntry.message !== undefined),
            width: levelEntry && Number.isFinite(levelEntry.width) ? levelEntry.width : 0,
            height: levelEntry && Number.isFinite(levelEntry.height) ? levelEntry.height : 0,
            area: levelEntry && Number.isFinite(levelEntry.width) && Number.isFinite(levelEntry.height)
                ? levelEntry.width * levelEntry.height
                : 0,
            level_object_bits: levelObjectBits(levelEntry),
        });
    }

    return {
        game,
        status: 'ok',
        group,
        excludeReason,
        baseFeatures,
        levels,
    };
}

function readJsonIfExists(filePath, required) {
    if (!fs.existsSync(filePath)) {
        if (required) {
            throw new Error(`Missing JSON artifact: ${filePath}`);
        }
        return null;
    }
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function addResult(resultsByKey, result, source) {
    if (!result || typeof result.game !== 'string' || !Number.isFinite(result.level) || result.level < 0) {
        return;
    }
    const key = targetKey(result);
    const list = resultsByKey.get(key) || [];
    list.push({ ...result, source });
    resultsByKey.set(key, list);
}

function loadArtifactResults() {
    const resultsByKey = new Map();
    const weighted = readJsonIfExists(options.weightedJson, true);
    for (const result of weighted.results || []) {
        addResult(resultsByKey, result, path.basename(options.weightedJson));
    }

    const allStrategies = readJsonIfExists(options.allStrategiesJson, false);
    const astarW2 = allStrategies && allStrategies.configs && allStrategies.configs.astar_w2;
    for (const result of (astarW2 && astarW2.targets) || []) {
        addResult(resultsByKey, result, 'focus_all_strategies:astar_w2');
    }
    return resultsByKey;
}

function loadFocusFeatures() {
    const features = readJsonIfExists(options.featuresJson, false);
    const byKey = new Map();
    for (const feature of Array.isArray(features) ? features : []) {
        byKey.set(targetKey(feature), feature);
    }
    return byKey;
}

function bestSolvedObservation(observations) {
    return observations
        .filter((observation) => observation.status === 'solved')
        .filter((observation) => observation.elapsed_ms <= options.hardMaxElapsedMs)
        .sort((left, right) => {
            if (left.elapsed_ms !== right.elapsed_ms) return left.elapsed_ms - right.elapsed_ms;
            return String(left.source).localeCompare(String(right.source));
        })[0] || null;
}

function roundRobinAppend(output, rows, limit, seenKeys) {
    const byGame = new Map();
    for (const row of rows) {
        const key = targetKey(row);
        if (seenKeys.has(key)) {
            continue;
        }
        const queue = byGame.get(row.game) || [];
        queue.push(row);
        byGame.set(row.game, queue);
    }
    for (const queue of byGame.values()) {
        queue.sort((left, right) => {
            if (left.solved_elapsed_ms !== right.solved_elapsed_ms) {
                return left.solved_elapsed_ms - right.solved_elapsed_ms;
            }
            return targetKey(left).localeCompare(targetKey(right));
        });
    }
    const games = Array.from(byGame.keys()).sort((left, right) => {
        const leftFirst = byGame.get(left)[0];
        const rightFirst = byGame.get(right)[0];
        if (leftFirst.solved_elapsed_ms !== rightFirst.solved_elapsed_ms) {
            return leftFirst.solved_elapsed_ms - rightFirst.solved_elapsed_ms;
        }
        return left.localeCompare(right);
    });

    let changed = true;
    while (output.length < limit && changed) {
        changed = false;
        for (const game of games) {
            if (output.length >= limit) {
                break;
            }
            const queue = byGame.get(game);
            if (!queue || queue.length === 0) {
                continue;
            }
            const row = queue.shift();
            const key = targetKey(row);
            if (seenKeys.has(key)) {
                continue;
            }
            seenKeys.add(key);
            output.push(row);
            changed = true;
        }
    }
}

function selectTargets(rows, limit) {
    const bestByKey = new Map();
    for (const row of rows) {
        const key = targetKey(row);
        const existing = bestByKey.get(key);
        if (!existing || row.solved_elapsed_ms < existing.solved_elapsed_ms) {
            bestByKey.set(key, row);
        }
    }
    const uniqueRows = Array.from(bestByKey.values());
    const preferred = uniqueRows.filter((row) =>
        row.solved_elapsed_ms >= options.minElapsedMs && row.solved_elapsed_ms <= options.preferredMaxElapsedMs
    );
    const fill = uniqueRows.filter((row) =>
        row.solved_elapsed_ms > options.preferredMaxElapsedMs && row.solved_elapsed_ms <= options.hardMaxElapsedMs
    );
    const fast = uniqueRows.filter((row) => row.solved_elapsed_ms < options.minElapsedMs);
    const selected = [];
    const seen = new Set();
    roundRobinAppend(selected, preferred, limit, seen);
    roundRobinAppend(selected, fill, limit, seen);
    roundRobinAppend(selected, fast, limit, seen);
    return selected;
}

function runSupplementProbe(target) {
    const result = spawnSync(process.execPath, [
        options.solverPath,
        options.corpusPath,
        '--timeout-ms', String(options.supplementTimeoutMs),
        '--strategy', 'weighted-astar',
        '--astar-weight', '2',
        '--solver-heuristic', 'winconditions',
        '--game', target.game,
        '--level', String(target.level),
        '--no-solutions',
        '--quiet',
        '--json',
    ], {
        encoding: 'utf8',
        maxBuffer: 512 * 1024 * 1024,
    });
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        throw new Error(`supplement probe failed for ${targetKey(target)}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    const json = JSON.parse(result.stdout);
    return json.results && json.results[0] ? {
        ...json.results[0],
        source: `supplement_${options.supplementTimeoutMs}ms`,
    } : null;
}

function makeTargetRow(result, features, focusFeature) {
    return {
        game: result.game,
        level: result.level,
        wincondition_group: features.group,
        first_solved_timeout_ms: options.benchmarkTimeoutMs,
        mined_timeout_ms: result.timeout_ms,
        solved_elapsed_ms: result.elapsed_ms,
        solved_generated: result.generated,
        solved_unique_states: result.unique_states,
        solved_expanded: result.expanded,
        solved_duplicates: result.duplicates,
        source: result.source,
        features: {
            ...features,
            focus_feature: focusFeature || null,
        },
        observations: [{
            source: result.source,
            timeout_ms: result.timeout_ms,
            status: result.status,
            elapsed_ms: result.elapsed_ms,
            generated: result.generated,
            unique_states: result.unique_states,
            expanded: result.expanded,
            duplicates: result.duplicates,
            max_frontier: result.max_frontier,
            solution_length: result.solution_length,
        }],
    };
}

function main() {
    loadPuzzleScript();
    const gameFeatures = new Map();
    const levelFeatures = new Map();
    const exclusions = {
        compile_error: 0,
        win_command: 0,
        no_winconditions: 0,
        multiple_winconditions: 0,
        unsupported_wincondition: 0,
    };

    const gameFiles = listGameFiles(options.corpusPath);
    log(`wincondition_focus compile games=${gameFiles.length}`);
    for (const gameFile of gameFiles) {
        const compiled = compileGameFeatures(gameFile);
        gameFeatures.set(compiled.game, compiled);
        if (compiled.status === 'compile_error') {
            exclusions.compile_error++;
            continue;
        }
        if (compiled.excludeReason) {
            exclusions[compiled.excludeReason] = (exclusions[compiled.excludeReason] || 0) + 1;
        }
        for (const [levelIndex, features] of compiled.levels) {
            levelFeatures.set(`${compiled.game}#${levelIndex}`, features);
        }
    }

    const artifactResults = loadArtifactResults();
    const focusFeatures = loadFocusFeatures();
    const rowsByGroup = Object.fromEntries(GROUPS.map((group) => [group, []]));
    const timeoutRowsByGroup = Object.fromEntries(GROUPS.map((group) => [group, []]));

    for (const [key, observations] of artifactResults) {
        const features = levelFeatures.get(key);
        if (!features || features.exclude_reason || features.is_message || !GROUPS.includes(features.group)) {
            continue;
        }
        const solved = bestSolvedObservation(observations);
        if (solved) {
            rowsByGroup[features.group].push(makeTargetRow(solved, features, focusFeatures.get(key)));
        }
        const timeout = observations.find((observation) => observation.status === 'timeout');
        if (timeout) {
            timeoutRowsByGroup[features.group].push({
                game: timeout.game,
                level: timeout.level,
                wincondition_group: features.group,
                features,
            });
        }
    }

    if (options.supplement) {
        for (const group of GROUPS) {
            let selected = selectTargets(rowsByGroup[group], options.maxTargets);
            if (selected.length >= options.maxTargets) {
                continue;
            }
            const seen = new Set(rowsByGroup[group].map(targetKey));
            const probeCandidates = selectTargets(timeoutRowsByGroup[group].map((row) => ({
                ...row,
                solved_elapsed_ms: options.supplementTimeoutMs,
            })), options.supplementProbeLimit).filter((row) => !seen.has(targetKey(row)));
            let probes = 0;
            for (const candidate of probeCandidates) {
                if (selected.length >= options.maxTargets || probes >= options.supplementProbeLimit) {
                    break;
                }
                probes++;
                log(`wincondition_focus supplement group=${group} probe=${probes}/${probeCandidates.length} target=${targetKey(candidate)}`);
                const result = runSupplementProbe(candidate);
                if (!result || result.status !== 'solved' || result.elapsed_ms > options.hardMaxElapsedMs) {
                    continue;
                }
                const key = targetKey(result);
                const features = levelFeatures.get(key);
                rowsByGroup[group].push(makeTargetRow(result, features, focusFeatures.get(key)));
                selected = selectTargets(rowsByGroup[group], options.maxTargets);
            }
        }
    }

    fs.mkdirSync(options.outDir, { recursive: true });
    const index = {
        schema_version: 1,
        kind: 'solver_wincondition_focus_index',
        generated_at: new Date().toISOString(),
        corpus: options.corpusPath,
        artifacts: {
            weighted_json: options.weightedJson,
            all_strategies_json: fs.existsSync(options.allStrategiesJson) ? options.allStrategiesJson : null,
            features_json: fs.existsSync(options.featuresJson) ? options.featuresJson : null,
        },
        max_targets: options.maxTargets,
        min_elapsed_ms: options.minElapsedMs,
        preferred_max_elapsed_ms: options.preferredMaxElapsedMs,
        hard_max_elapsed_ms: options.hardMaxElapsedMs,
        benchmark_timeout_ms: options.benchmarkTimeoutMs,
        supplement: {
            enabled: options.supplement,
            timeout_ms: options.supplementTimeoutMs,
            probe_limit: options.supplementProbeLimit,
        },
        exclusions,
        groups: {},
    };

    for (const group of GROUPS) {
        const candidates = rowsByGroup[group];
        const targets = selectTargets(candidates, options.maxTargets);
        const manifest = {
            schema_version: 1,
            kind: 'solver_wincondition_focus_group',
            generated_at: index.generated_at,
            group,
            corpus: options.corpusPath,
            strategy: 'weighted-astar',
            astar_weight: 2,
            baseline_heuristic: 'winconditions',
            timeout_ms: options.benchmarkTimeoutMs,
            target_count: targets.length,
            candidate_count: candidates.length,
            targets,
        };
        const fileName = `${group}.json`;
        const filePath = path.join(options.outDir, fileName);
        fs.writeFileSync(filePath, `${JSON.stringify(manifest, null, 2)}\n`);
        index.groups[group] = {
            path: filePath,
            target_count: targets.length,
            candidate_count: candidates.length,
            timeout_candidate_count: timeoutRowsByGroup[group].length,
        };
        log(`wincondition_focus group=${group} targets=${targets.length} candidates=${candidates.length}`);
    }

    const indexPath = path.join(options.outDir, 'index.json');
    fs.writeFileSync(indexPath, `${JSON.stringify(index, null, 2)}\n`);
    process.stdout.write(`wincondition_focus wrote ${indexPath}\n`);
}

main();
