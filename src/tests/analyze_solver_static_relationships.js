#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');
const staticAnalysis = require('./lib/solver_static_analysis');

const {
    maskHasBits,
    masksIntersect,
    masksEqual,
    cloneMask,
    iorIntersection,
    objectPresenceMask,
    rowObjectsSetMask,
    foreignSetMask,
    cellHasMovement,
    cellChangesObjects,
    isCancelRule,
} = staticAnalysis;

function usage() {
    console.error([
        'Usage: node src/tests/analyze_solver_static_relationships.js <solver_tests_dir> <focus_group_json>',
        '  [--out PATH] [--matrix PATH]... [--quiet]',
    ].join('\n'));
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2 || args.includes('--help') || args.includes('-h')) {
    usage();
}

const options = {
    corpusPath: path.resolve(args[0]),
    focusPath: path.resolve(args[1]),
    outPath: '/private/tmp/js_solver_static_relationships.json',
    matrixPaths: [],
    quiet: false,
};

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out' && index + 1 < args.length) {
        options.outPath = path.resolve(args[++index]);
    } else if (arg === '--matrix' && index + 1 < args.length) {
        options.matrixPaths.push(path.resolve(args[++index]));
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

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function loadFocus(inputPath) {
    const root = readJson(inputPath);
    if (root.kind !== 'solver_wincondition_focus_group') {
        throw new Error(`Expected focus group manifest, got ${root.kind || 'unknown'}`);
    }
    return root;
}

function targetKey(target) {
    return `${target.game}#${target.level}`;
}

function compileGame(game) {
    const filePath = path.join(options.corpusPath, game);
    let source = fs.readFileSync(filePath, 'utf8');
    if (!source.endsWith('\n')) {
        source += '\n';
    }
    if (typeof resetParserErrorState === 'function') {
        resetParserErrorState();
    }
    unitTesting = true;
    lazyFunctionGeneration = false;
    try {
        compile(['loadLevel', 0], source, `static-relationships:${game}:0`);
    } finally {
        unitTesting = false;
        lazyFunctionGeneration = true;
    }
    if (typeof resetParserErrorState === 'function') {
        resetParserErrorState();
    }
    if (errorCount > 0) {
        throw new Error(errorStrings.map(stripHTMLTags).join('\n'));
    }
}

function bit(mask, objectId) {
    return Boolean(mask && mask.data && ((mask.data[objectId >> 5] >>> (objectId & 31)) & 1));
}

function maskNames(mask) {
    if (!mask || !mask.data) {
        return [];
    }
    const names = [];
    state.idDict.forEach((name, objectId) => {
        if (bit(mask, objectId)) {
            names.push(state.original_case_names[name] || name);
        }
    });
    return names;
}

function maskCount(mask) {
    if (!mask || !mask.data) {
        return 0;
    }
    let count = 0;
    for (const word of mask.data) {
        count += popcount32(word >>> 0);
    }
    return count;
}

function popcount32(value) {
    value = value - ((value >>> 1) & 0x55555555);
    value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
    return (((value + (value >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

const movementMaskTouchesObjectMask = (movementMask, objectMask) =>
    staticAnalysis.movementMaskTouchesObjectMask(state, movementMask, objectMask);

const cellChangesObjectMask = (cell, objectMask) =>
    staticAnalysis.cellChangesObjectMask(state, cell, objectMask);

function layerNamesForMask(mask) {
    const layerNames = new Set();
    for (const objectName of state.idDict) {
        const object = state.objects[objectName];
        if (object && bit(mask, object.id)) {
            layerNames.add(String(object.layer));
        }
    }
    return [...layerNames].sort((left, right) => Number(left) - Number(right));
}

function playerMaskForState() {
    return Array.isArray(state.playerMask) ? state.playerMask[1] : state.playerMask;
}

const inferStaticBlockerMask = (condition, playerMask) =>
    staticAnalysis.inferStaticBlockerMask(state, condition, { playerMask });

function relationSummary(stats, condition, playerMask) {
    const sourceIsPlayer = playerMask && masksIntersect(condition[1], playerMask);
    const targetIsPlayer = playerMask && masksIntersect(condition[2], playerMask);
    const parts = [];
    if (sourceIsPlayer) {
        parts.push('player_is_A');
    }
    if (targetIsPlayer) {
        parts.push('player_is_B');
    }
    if (!sourceIsPlayer && stats.player_adjacent_source_changed > 0) {
        parts.push('player_changes_A');
    }
    if (!targetIsPlayer && stats.player_adjacent_target_changed > 0) {
        parts.push('player_changes_B');
    }
    if (!sourceIsPlayer && stats.player_source_same_cell_changed > 0) {
        parts.push('player_overlaps_A');
    }
    if (!targetIsPlayer && stats.player_target_same_cell_changed > 0) {
        parts.push('player_overlaps_B');
    }
    if (stats.source_trail_rules > 0) {
        parts.push('A_leaves_trail');
    }
    if (stats.target_changed_rules === 0 && stats.target_moved_rules === 0) {
        parts.push('B_passive');
    }
    if (stats.source_moved_without_player_rules > 0) {
        parts.push('A_moves_without_player');
    }
    return parts.length > 0 ? parts.join('+') : 'indirect_or_unclear';
}

function analyzeGame(game) {
    compileGame(game);
    const condition = state.winconditions && state.winconditions.length === 1 ? state.winconditions[0] : null;
    if (!condition || condition[0] !== 0) {
        throw new Error(`${game} is not a single some wincondition game`);
    }

    const playerMask = playerMaskForState();
    const backgroundMask = state.layerMasks && Number.isInteger(state.backgroundlayer)
        ? state.layerMasks[state.backgroundlayer]
        : null;
    const excludedTrailMask = cloneMask(condition[1]);
    excludedTrailMask.ior(condition[2]);
    if (playerMask && playerMask.data) {
        excludedTrailMask.ior(playerMask);
    }
    if (backgroundMask && backgroundMask.data) {
        excludedTrailMask.ior(backgroundMask);
    }

    const stats = {
        rule_count: 0,
        late_rule_count: 0,
        cancel_rule_count: 0,
        player_source_same_cell_rules: 0,
        player_target_same_cell_rules: 0,
        source_target_same_cell_rules: 0,
        player_adjacent_source_rules: 0,
        player_adjacent_target_rules: 0,
        source_adjacent_target_rules: 0,
        player_adjacent_source_changed: 0,
        player_adjacent_target_changed: 0,
        player_source_same_cell_changed: 0,
        player_target_same_cell_changed: 0,
        source_changed_rules: 0,
        target_changed_rules: 0,
        source_moved_rules: 0,
        target_moved_rules: 0,
        source_moved_without_player_rules: 0,
        target_moved_without_player_rules: 0,
        source_trail_rules: 0,
    };
    const trailObjects = new BitVec(STRIDE_OBJ);

    const groups = [
        { late: false, groups: state.rules || [] },
        { late: true, groups: state.lateRules || [] },
    ];
    for (const { late, groups: ruleGroups } of groups) {
        for (const group of ruleGroups) {
            for (const rule of group || []) {
                stats.rule_count++;
                if (late) {
                    stats.late_rule_count++;
                }
                if (isCancelRule(rule)) {
                    stats.cancel_rule_count++;
                }
                for (const row of rule.patterns || []) {
                    const rowHasPlayer = row.some((cell) => playerMask && masksIntersect(objectPresenceMask(cell), playerMask));
                    for (let cellIndex = 0; cellIndex < row.length; cellIndex++) {
                        const cell = row[cellIndex];
                        if (!cell || !cell.objectsPresent) {
                            continue;
                        }
                        const present = objectPresenceMask(cell);
                        const hasPlayer = playerMask && masksIntersect(present, playerMask);
                        const hasSource = masksIntersect(present, condition[1]);
                        const hasTarget = masksIntersect(present, condition[2]);
                        const sourceChanged = cellChangesObjectMask(cell, condition[1]);
                        const targetChanged = cellChangesObjectMask(cell, condition[2]);
                        const sourceMoved = hasSource && cell.replacement && movementMaskTouchesObjectMask(cell.replacement.movementsSet, condition[1]);
                        const targetMoved = hasTarget && cell.replacement && movementMaskTouchesObjectMask(cell.replacement.movementsSet, condition[2]);

                        if (hasPlayer && hasSource) {
                            stats.player_source_same_cell_rules++;
                            if (sourceChanged) {
                                stats.player_source_same_cell_changed++;
                            }
                        }
                        if (hasPlayer && hasTarget) {
                            stats.player_target_same_cell_rules++;
                            if (targetChanged) {
                                stats.player_target_same_cell_changed++;
                            }
                        }
                        if (hasSource && hasTarget) {
                            stats.source_target_same_cell_rules++;
                        }
                        if (sourceChanged) {
                            stats.source_changed_rules++;
                        }
                        if (targetChanged) {
                            stats.target_changed_rules++;
                        }
                        if (sourceMoved) {
                            stats.source_moved_rules++;
                            if (!rowHasPlayer) {
                                stats.source_moved_without_player_rules++;
                            }
                        }
                        if (targetMoved) {
                            stats.target_moved_rules++;
                            if (!rowHasPlayer) {
                                stats.target_moved_without_player_rules++;
                            }
                        }
                        if (hasSource && cellHasMovement(cell)) {
                            const foreign = foreignSetMask(cell, excludedTrailMask);
                            if (!foreign.iszero()) {
                                stats.source_trail_rules++;
                                trailObjects.ior(foreign);
                            }
                        }
                    }

                    for (let cellIndex = 0; cellIndex + 1 < row.length; cellIndex++) {
                        const left = row[cellIndex];
                        const right = row[cellIndex + 1];
                        if (!left || !right || !left.objectsPresent || !right.objectsPresent) {
                            continue;
                        }
                        const leftPresent = objectPresenceMask(left);
                        const rightPresent = objectPresenceMask(right);
                        const leftPlayer = playerMask && masksIntersect(leftPresent, playerMask);
                        const rightPlayer = playerMask && masksIntersect(rightPresent, playerMask);
                        const leftSource = masksIntersect(leftPresent, condition[1]);
                        const rightSource = masksIntersect(rightPresent, condition[1]);
                        const leftTarget = masksIntersect(leftPresent, condition[2]);
                        const rightTarget = masksIntersect(rightPresent, condition[2]);
                        if ((leftPlayer && rightSource) || (rightPlayer && leftSource)) {
                            stats.player_adjacent_source_rules++;
                            const sourceCell = leftSource ? left : right;
                            if (cellChangesObjectMask(sourceCell, condition[1])) {
                                stats.player_adjacent_source_changed++;
                            }
                        }
                        if ((leftPlayer && rightTarget) || (rightPlayer && leftTarget)) {
                            stats.player_adjacent_target_rules++;
                            const targetCell = leftTarget ? left : right;
                            if (cellChangesObjectMask(targetCell, condition[2])) {
                                stats.player_adjacent_target_changed++;
                            }
                        }
                        if ((leftSource && rightTarget) || (rightSource && leftTarget)) {
                            stats.source_adjacent_target_rules++;
                        }
                    }
                }
            }
        }
    }

    const staticBlockers = inferStaticBlockerMask(condition, playerMask);
    return {
        game,
        wincondition: {
            type: 'some_on',
            source_names: maskNames(condition[1]),
            target_names: maskNames(condition[2]),
            source_aggregate: Boolean(condition[4]),
            target_aggregate: Boolean(condition[5]),
        },
        player: {
            names: maskNames(playerMask),
            source_intersects_player: Boolean(playerMask && masksIntersect(condition[1], playerMask)),
            target_intersects_player: Boolean(playerMask && masksIntersect(condition[2], playerMask)),
            source_equals_player: Boolean(playerMask && masksEqual(condition[1], playerMask)),
            target_equals_player: Boolean(playerMask && masksEqual(condition[2], playerMask)),
            object_count: maskCount(playerMask),
        },
        layers: {
            source: layerNamesForMask(condition[1]),
            target: layerNamesForMask(condition[2]),
            player: layerNamesForMask(playerMask),
        },
        relationship: null,
        stats,
        inferred: {
            static_blocker_names: maskNames(staticBlockers.blockers),
            consumed_neighbor_names: maskNames(staticBlockers.consumed),
            trail_object_names: maskNames(trailObjects),
        },
    };
}

function loadMatrixMetrics(paths) {
    const metrics = new Map();
    for (const matrixPath of paths) {
        const matrix = readJson(matrixPath);
        const groups = matrix.groups || {};
        for (const group of Object.values(groups)) {
            for (const [heuristic, entry] of Object.entries(group.heuristics || {})) {
                for (const target of entry.targets || []) {
                    const key = targetKey(target);
                    if (!metrics.has(key)) {
                        metrics.set(key, {});
                    }
                    metrics.get(key)[heuristic] = {
                        status: target.status,
                        generated: target.median.generated,
                        unique_states: target.median.unique_states,
                        expanded: target.median.expanded,
                        elapsed_ms: target.median.elapsed_ms,
                        solution_length: target.median.solution_length,
                    };
                }
            }
        }
    }
    return metrics;
}

function summarizeOutcomes(targets, metrics) {
    const byGame = new Map();
    for (const target of targets) {
        if (!byGame.has(target.game)) {
            byGame.set(target.game, {
                target_count: 0,
                best_heuristic_counts: {},
                solved_by_heuristic: {},
                levels: [],
            });
        }
        const entry = byGame.get(target.game);
        entry.target_count++;
        const targetMetrics = metrics.get(targetKey(target)) || {};
        let best = null;
        for (const [heuristic, metric] of Object.entries(targetMetrics)) {
            if (metric.status === 'solved') {
                entry.solved_by_heuristic[heuristic] = (entry.solved_by_heuristic[heuristic] || 0) + 1;
                if (!best || metric.generated < best.generated) {
                    best = { heuristic, generated: metric.generated };
                }
            }
        }
        if (best) {
            entry.best_heuristic_counts[best.heuristic] = (entry.best_heuristic_counts[best.heuristic] || 0) + 1;
        }
        entry.levels.push({
            level: target.level,
            best_heuristic: best ? best.heuristic : null,
            metrics: targetMetrics,
        });
    }
    return byGame;
}

function printHuman(report) {
    for (const game of report.games) {
        const outcome = game.outcome || {};
        const best = Object.entries(outcome.best_heuristic_counts || {})
            .sort((left, right) => right[1] - left[1])
            .map(([heuristic, count]) => `${heuristic}:${count}`)
            .join(', ');
        process.stdout.write(
            `${game.game}: ${game.relationship}` +
            ` | A=${game.wincondition.source_names.join('|')}` +
            ` B=${game.wincondition.target_names.join('|')}` +
            ` player=${game.player.names.join('|')}` +
            ` | blockers=${game.inferred.static_blocker_names.join('|') || '-'}` +
            ` | trail=${game.inferred.trail_object_names.join('|') || '-'}` +
            ` | best=${best || '-'}\n`
        );
    }
}

function main() {
    loadPuzzleScript();
    const focus = loadFocus(options.focusPath);
    const games = [...new Set((focus.targets || []).map((target) => target.game))].sort();
    const metrics = loadMatrixMetrics(options.matrixPaths);
    const outcomes = summarizeOutcomes(focus.targets || [], metrics);
    const report = {
        schema_version: 1,
        kind: 'solver_static_relationships',
        generated_at: new Date().toISOString(),
        corpus: options.corpusPath,
        focus_input: options.focusPath,
        matrix_inputs: options.matrixPaths,
        games: [],
    };

    for (const game of games) {
        log(`static_relationships game=${game}`);
        const analysis = analyzeGame(game);
        analysis.relationship = relationSummary(analysis.stats, state.winconditions[0], playerMaskForState());
        analysis.outcome = outcomes.get(game) || null;
        report.games.push(analysis);
    }

    fs.mkdirSync(path.dirname(options.outPath), { recursive: true });
    fs.writeFileSync(options.outPath, `${JSON.stringify(report, null, 2)}\n`);
    printHuman(report);
    process.stdout.write(`static_relationships wrote ${options.outPath}\n`);
}

main();
