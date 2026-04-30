#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/compare_solver_focus_benchmarks.js <interpreted.json> <compiled.json> [--detail] [--goal-ratio N]');
    process.exit(1);
}

if (process.argv.length < 4) {
    usage();
}

const options = {
    detail: false,
    goalRatio: null,
};
for (let index = 4; index < process.argv.length; index++) {
    const arg = process.argv[index];
    if (arg === '--detail') {
        options.detail = true;
    } else if (arg === '--goal-ratio' && index + 1 < process.argv.length) {
        options.goalRatio = Number.parseFloat(process.argv[++index]);
        if (!Number.isFinite(options.goalRatio) || options.goalRatio <= 0) {
            throw new Error(`--goal-ratio must be positive: ${process.argv[index]}`);
        }
    } else {
        usage();
    }
}

function readJson(filePath) {
    return JSON.parse(fs.readFileSync(path.resolve(filePath), 'utf8'));
}

function targetKey(target) {
    return `${target.game}#${target.level}`;
}

function formatNumber(value, digits = 1) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
        return 'n/a';
    }
    return Number(value).toFixed(digits);
}

function ratio(newValue, oldValue) {
    if (!Number.isFinite(newValue) || !Number.isFinite(oldValue) || oldValue === 0) {
        return null;
    }
    return newValue / oldValue;
}

function formatDelta(newValue, oldValue, digits = 1) {
    const valueRatio = ratio(newValue, oldValue);
    if (valueRatio === null) {
        return 'n/a';
    }
    const percent = (valueRatio - 1) * 100;
    const sign = percent > 0 ? '+' : '';
    return `${formatNumber(valueRatio, 3)}x (${sign}${formatNumber(percent, digits)}%)`;
}

function formatArgs(value) {
    return JSON.stringify(Array.isArray(value) ? value : []);
}

function metricMedian(summary, key) {
    let value = summary && summary.median && summary.median[key];
    if (!Number.isFinite(value) && key.startsWith('compact_turn_')) {
        value = summary && summary.median && summary.median[key.replace('compact_turn_', 'compact_tick_')];
    }
    return Number.isFinite(value) ? value : null;
}

function metricSum(summary, keys) {
    let total = 0;
    let seen = false;
    for (const key of keys) {
        const value = metricMedian(summary, key);
        if (value !== null) {
            total += value;
            seen = true;
        }
    }
    return seen ? total : null;
}

function sampleMetric(sample, key) {
    let value = sample && sample[key];
    if (!Number.isFinite(value) && key.startsWith('compact_turn_')) {
        value = sample && sample[key.replace('compact_turn_', 'compact_tick_')];
    }
    return Number.isFinite(value) ? value : 0;
}

function aggregateSampleMetric(benchmark, key) {
    let total = 0;
    for (const target of benchmark.targets || []) {
        for (const sample of target.samples || []) {
            total += sampleMetric(sample, key);
        }
    }
    return total;
}

const interpreted = readJson(process.argv[2]);
const compiled = readJson(process.argv[3]);
const interpretedKeys = (interpreted.targets || []).map(targetKey);
const compiledKeys = (compiled.targets || []).map(targetKey);
const sameTargets = interpretedKeys.length === compiledKeys.length
    && interpretedKeys.every((key, index) => key === compiledKeys[index]);

const interpretedStatus = {};
const compiledStatus = {};
for (const target of interpreted.targets || []) {
    for (const [status, count] of Object.entries(target.status_counts || {})) {
        interpretedStatus[status] = (interpretedStatus[status] || 0) + count;
    }
}
for (const target of compiled.targets || []) {
    for (const [status, count] of Object.entries(target.status_counts || {})) {
        compiledStatus[status] = (compiledStatus[status] || 0) + count;
    }
}

function countersMedian(summary, key) {
    const values = (summary.samples || [])
        .map((sample) => sample.runtime_counters && sample.runtime_counters[key])
        .filter((value) => Number.isFinite(value));
    if (values.length === 0) {
        return null;
    }
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
}

function benchmarkCountersMedian(benchmark, key) {
    const values = [];
    for (const target of benchmark.targets || []) {
        for (const sample of target.samples || []) {
            const value = sample.runtime_counters && sample.runtime_counters[key];
            if (Number.isFinite(value)) {
                values.push(value);
            }
        }
    }
    if (values.length === 0) {
        return null;
    }
    values.sort((a, b) => a - b);
    return values[Math.floor(values.length / 2)];
}

function countersMedianAny(summary, keys) {
    for (const key of keys) {
        const value = countersMedian(summary, key);
        if (value !== null) {
            return value;
        }
    }
    return null;
}

function sampleBoolean(summary, key) {
    const values = (summary.samples || [])
        .map((sample) => sample[key])
        .filter((value) => value !== null && value !== undefined);
    if (values.length === 0) {
        return null;
    }
    return values.some(Boolean);
}

function sampleBooleanAny(summary, keys) {
    for (const key of keys) {
        const value = sampleBoolean(summary, key);
        if (value !== null) {
            return value;
        }
    }
    return null;
}

function targetRows() {
    const compiledByKey = new Map((compiled.targets || []).map((target) => [targetKey(target), target]));
    return (interpreted.targets || []).map((target) => {
        const key = targetKey(target);
        const compiledTarget = compiledByKey.get(key);
        if (!compiledTarget) {
            return null;
        }
        const compiledRuleHits = countersMedian(compiledTarget, 'specialized_rulegroup_hits')
            ?? countersMedian(compiledTarget, 'compiled_rule_group_hits');
        const fullTurnHits = countersMedianAny(compiledTarget, ['specialized_full_turn_hits', 'compiled_tick_hits']);
        const compactTurnNativeHits = metricMedian(compiledTarget, 'compact_turn_native_hits');
        const compactTurnBridgeHits = metricMedian(compiledTarget, 'compact_turn_bridge_hits');
        const specializedRulegroupsAttached = sampleBoolean(compiledTarget, 'specialized_rulegroups_attached')
            ?? sampleBoolean(compiledTarget, 'compiled_rules_attached');
        const fullTurnAttached = sampleBooleanAny(compiledTarget, ['specialized_full_turn_attached', 'compiled_tick_attached']);
        const compactTurnAttached = sampleBoolean(compiledTarget, 'specialized_compact_turn_attached');
        const interpretedFrontier = metricSum(target, ['frontier_pop_ms', 'frontier_push_ms']);
        const compiledFrontier = metricSum(compiledTarget, ['frontier_pop_ms', 'frontier_push_ms']);
        const interpretedVisited = metricSum(target, ['visited_lookup_ms', 'visited_insert_ms']);
        const compiledVisited = metricSum(compiledTarget, ['visited_lookup_ms', 'visited_insert_ms']);
        const graphKeys = [
            'frontier_pop_ms',
            'frontier_push_ms',
            'visited_lookup_ms',
            'visited_insert_ms',
            'node_store_ms',
            'heuristic_ms',
            'solved_check_ms',
            'timeout_check_ms',
            'unattributed_ms',
        ];
        const interpretedGraph = metricSum(target, graphKeys);
        const compiledGraph = metricSum(compiledTarget, graphKeys);
        return {
            key,
            interpreted: target,
            compiled: compiledTarget,
            wallRatio: ratio(compiledTarget.median.wall_ms, target.median.wall_ms),
            elapsedRatio: ratio(compiledTarget.median.elapsed_ms, target.median.elapsed_ms),
            generatedRatio: ratio(compiledTarget.median.generated, target.median.generated),
            expandedRatio: ratio(compiledTarget.median.expanded, target.median.expanded),
            stepRatio: ratio(metricMedian(compiledTarget, 'step_ms'), metricMedian(target, 'step_ms')),
            cloneRatio: ratio(metricMedian(compiledTarget, 'clone_ms'), metricMedian(target, 'clone_ms')),
            hashRatio: ratio(metricMedian(compiledTarget, 'hash_ms'), metricMedian(target, 'hash_ms')),
            frontierRatio: ratio(compiledFrontier, interpretedFrontier),
            visitedRatio: ratio(compiledVisited, interpretedVisited),
            nodeStoreRatio: ratio(metricMedian(compiledTarget, 'node_store_ms'), metricMedian(target, 'node_store_ms')),
            heuristicRatio: ratio(metricMedian(compiledTarget, 'heuristic_ms'), metricMedian(target, 'heuristic_ms')),
            unattributedRatio: ratio(metricMedian(compiledTarget, 'unattributed_ms'), metricMedian(target, 'unattributed_ms')),
            graphOverheadRatio: ratio(compiledGraph, interpretedGraph),
            interpretedFrontier,
            compiledFrontier,
            interpretedVisited,
            compiledVisited,
            interpretedGraph,
            compiledGraph,
            compiledRuleHits,
            fullTurnHits,
            compactTurnNativeHits,
            compactTurnBridgeHits,
            bucket: compiledUsageBucket(fullTurnHits, compiledRuleHits, compactTurnNativeHits, compactTurnBridgeHits, compactTurnAttached),
            reason: compiledUsageReason(
                fullTurnHits,
                compiledRuleHits,
                compactTurnNativeHits,
                compactTurnBridgeHits,
                compactTurnAttached,
                specializedRulegroupsAttached,
                fullTurnAttached
            ),
            specializedRulegroupsAttached,
            fullTurnAttached,
            compactTurnAttached,
            candidateCells: countersMedian(compiledTarget, 'candidate_cells_tested'),
            rowScans: countersMedian(compiledTarget, 'row_scans'),
            patternTests: countersMedian(compiledTarget, 'pattern_tests'),
            maskRebuilds: countersMedian(compiledTarget, 'mask_rebuild_calls'),
            maskRebuildDirtyCalls: countersMedian(compiledTarget, 'mask_rebuild_dirty_calls'),
            maskRebuildRows: countersMedian(compiledTarget, 'mask_rebuild_rows'),
            maskRebuildColumns: countersMedian(compiledTarget, 'mask_rebuild_columns'),
        };
    }).filter(Boolean);
}

function compiledUsageBucket(fullTurnHits, compiledRuleHits, compactTurnNativeHits, compactTurnBridgeHits, compactTurnAttached) {
    if (compactTurnNativeHits !== null && compactTurnNativeHits > 0) {
        return 'compact_turn_native';
    }
    if (compactTurnBridgeHits !== null && compactTurnBridgeHits > 0) {
        return 'compact_turn_bridge';
    }
    if (compactTurnAttached === false) {
        return 'compact_turn_not_attached';
    }
    if (fullTurnHits === null || compiledRuleHits === null) {
        return 'no_counters';
    }
    if (fullTurnHits === 0 && compiledRuleHits === 0) {
        return 'no_full_turn_no_rulegroups';
    }
    if (fullTurnHits > 0 && compiledRuleHits === 0) {
        return 'full_turn_no_rulegroups';
    }
    if (compiledRuleHits > 0) {
        return 'specialized_rulegroups';
    }
    return 'unknown';
}

function compiledUsageReason(
    fullTurnHits,
    compiledRuleHits,
    compactTurnNativeHits,
    compactTurnBridgeHits,
    compactTurnAttached,
    specializedRulegroupsAttached,
    fullTurnAttached
) {
    if (compactTurnNativeHits !== null && compactTurnNativeHits > 0) {
        return 'compact_turn_native_hit';
    }
    if (compactTurnBridgeHits !== null && compactTurnBridgeHits > 0) {
        return 'compact_turn_bridge_hit';
    }
    if (compactTurnAttached === false) {
        return 'specialized_compact_turn_backend_not_attached';
    }
    if (fullTurnHits === null || compiledRuleHits === null) {
        return 'runtime_counters_missing';
    }
    if (!fullTurnAttached) {
        return 'specialized_full_turn_backend_not_attached';
    }
    if (!specializedRulegroupsAttached) {
        return 'specialized_rulegroups_backend_not_attached';
    }
    if (fullTurnHits === 0) {
        return 'specialized_full_turn_not_called';
    }
    if (compiledRuleHits === 0) {
        return 'specialized_full_turn_bypassed_generic_rulegroup_counter';
    }
    return 'specialized_rulegroups_hit';
}

function printCompiledUsageSummary(rows) {
    const counts = new Map();
    const reasons = new Map();
    for (const row of rows) {
        counts.set(row.bucket, (counts.get(row.bucket) || 0) + 1);
        reasons.set(row.reason, (reasons.get(row.reason) || 0) + 1);
    }
    const labels = [
        'compact_turn_native',
        'compact_turn_bridge',
        'compact_turn_not_attached',
        'specialized_rulegroups',
        'full_turn_no_rulegroups',
        'no_full_turn_no_rulegroups',
        'no_counters',
        'unknown',
    ];
    const parts = labels
        .filter((label) => counts.has(label))
        .map((label) => `${label}=${counts.get(label)}`);
    process.stdout.write(`  compiled_usage: ${parts.length === 0 ? 'n/a' : parts.join(' ')}\n`);
    const reasonParts = Array.from(reasons.entries())
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
        .map(([reason, count]) => `${reason}=${count}`);
    process.stdout.write(`  compiled_usage_reasons: ${reasonParts.length === 0 ? 'n/a' : reasonParts.join(' ')}\n`);
}

function printTargetTable(label, rows) {
    process.stdout.write(`  ${label}\n`);
    for (const row of rows) {
        process.stdout.write(
            `    ${formatNumber(row.elapsedRatio, 3)}x elapsed` +
            ` wall=${formatNumber(row.wallRatio, 3)}x` +
            ` generated=${formatNumber(row.generatedRatio, 3)}x` +
            ` expanded=${formatNumber(row.expandedRatio, 3)}x` +
            ` step=${formatNumber(row.stepRatio, 3)}x` +
            ` clone=${formatNumber(row.cloneRatio, 3)}x` +
            ` hash=${formatNumber(row.hashRatio, 3)}x` +
            ` graph=${formatNumber(row.graphOverheadRatio, 3)}x` +
            ` visited=${formatNumber(row.visitedRatio, 3)}x` +
            ` frontier=${formatNumber(row.frontierRatio, 3)}x` +
            ` bucket=${row.bucket}` +
            ` reason=${row.reason}` +
            ` ${row.key}` +
            ` hits=${row.compiledRuleHits === null ? 'n/a' : row.compiledRuleHits}` +
            ` full_turn=${row.fullTurnHits === null ? 'n/a' : row.fullTurnHits}` +
            ` compact_native=${row.compactTurnNativeHits === null ? 'n/a' : row.compactTurnNativeHits}` +
            ` compact_bridge=${row.compactTurnBridgeHits === null ? 'n/a' : row.compactTurnBridgeHits}` +
            ` rows=${row.rowScans === null ? 'n/a' : row.rowScans}` +
            ` cells=${row.candidateCells === null ? 'n/a' : row.candidateCells}` +
            ` pattern_tests=${row.patternTests === null ? 'n/a' : row.patternTests}` +
            ` mask_rebuilds=${row.maskRebuilds === null ? 'n/a' : row.maskRebuilds}` +
            ` dirty=${row.maskRebuildDirtyCalls === null ? 'n/a' : row.maskRebuildDirtyCalls}` +
            `\n`
        );
    }
}

function printMedianMetric(label, key, digits = 1) {
    const interpretedValue = metricMedian(interpreted, key);
    const compiledValue = metricMedian(compiled, key);
    if (interpretedValue === null && compiledValue === null) {
        return;
    }
    process.stdout.write(
        `  median_${label}: interpreted=${formatNumber(interpretedValue, digits)}` +
        ` compiled=${formatNumber(compiledValue, digits)}` +
        ` compiled/interpreted=${formatDelta(compiledValue, interpretedValue)}\n`
    );
}

function printMedianMetricSum(label, keys, digits = 1) {
    const interpretedValue = metricSum(interpreted, keys);
    const compiledValue = metricSum(compiled, keys);
    if (interpretedValue === null && compiledValue === null) {
        return;
    }
    process.stdout.write(
        `  median_${label}: interpreted=${formatNumber(interpretedValue, digits)}` +
        ` compiled=${formatNumber(compiledValue, digits)}` +
        ` compiled/interpreted=${formatDelta(compiledValue, interpretedValue)}\n`
    );
}

function printCompactTurnSummary() {
    const interpretedAttempts = metricMedian(interpreted, 'compact_turn_attempts');
    const compiledAttempts = metricMedian(compiled, 'compact_turn_attempts');
    if (interpretedAttempts === null && compiledAttempts === null) {
        return;
    }
    const interpretedHits = metricMedian(interpreted, 'compact_turn_hits');
    const compiledHits = metricMedian(compiled, 'compact_turn_hits');
    const interpretedHitRate = ratio(interpretedHits, interpretedAttempts);
    const compiledHitRate = ratio(compiledHits, compiledAttempts);
    process.stdout.write(
        `  median_compact_turn_hit_rate:` +
        ` interpreted=${interpretedHitRate === null ? 'n/a' : formatNumber(interpretedHitRate * 100, 1) + '%'}` +
        ` compiled=${compiledHitRate === null ? 'n/a' : formatNumber(compiledHitRate * 100, 1) + '%'}\n`
    );
    const nativeAttempts = metricMedian(compiled, 'compact_turn_native_attempts');
    const nativeHits = metricMedian(compiled, 'compact_turn_native_hits');
    const bridgeAttempts = metricMedian(compiled, 'compact_turn_bridge_attempts');
    const bridgeHits = metricMedian(compiled, 'compact_turn_bridge_hits');
    if (nativeAttempts !== null || nativeHits !== null || bridgeAttempts !== null || bridgeHits !== null) {
        const nativeHitRate = ratio(nativeHits, nativeAttempts);
        const bridgeHitRate = ratio(bridgeHits, bridgeAttempts);
        process.stdout.write(
            `  median_compact_turn_modes:` +
            ` native=${formatNumber(nativeHits, 0)}/${formatNumber(nativeAttempts, 0)}` +
            `(${nativeHitRate === null ? 'n/a' : formatNumber(nativeHitRate * 100, 1) + '%'})` +
            ` bridge=${formatNumber(bridgeHits, 0)}/${formatNumber(bridgeAttempts, 0)}` +
            `(${bridgeHitRate === null ? 'n/a' : formatNumber(bridgeHitRate * 100, 1) + '%'})\n`
        );
    }
    const totalNativeAttempts = aggregateSampleMetric(compiled, 'compact_turn_native_attempts');
    const totalNativeHits = aggregateSampleMetric(compiled, 'compact_turn_native_hits');
    const totalBridgeAttempts = aggregateSampleMetric(compiled, 'compact_turn_bridge_attempts');
    const totalBridgeHits = aggregateSampleMetric(compiled, 'compact_turn_bridge_hits');
    const totalUnsupported = aggregateSampleMetric(compiled, 'compact_turn_unsupported');
    if (totalNativeAttempts > 0 || totalBridgeAttempts > 0 || totalUnsupported > 0) {
        process.stdout.write(
            `  total_compact_turn_modes:` +
            ` native=${formatNumber(totalNativeHits, 0)}/${formatNumber(totalNativeAttempts, 0)}` +
            ` bridge=${formatNumber(totalBridgeHits, 0)}/${formatNumber(totalBridgeAttempts, 0)}` +
            ` unsupported=${formatNumber(totalUnsupported, 0)}\n`
        );
    }
}

function printRuntimeCounterMedian(label, key, digits = 1, scale = 1) {
    const interpretedValue = benchmarkCountersMedian(interpreted, key);
    const compiledValue = benchmarkCountersMedian(compiled, key);
    if (interpretedValue === null && compiledValue === null) {
        return;
    }
    const interpretedScaled = interpretedValue === null ? null : interpretedValue / scale;
    const compiledScaled = compiledValue === null ? null : compiledValue / scale;
    process.stdout.write(
        `  median_runtime_${label}: interpreted=${formatNumber(interpretedScaled, digits)}` +
        ` compiled=${formatNumber(compiledScaled, digits)}` +
        ` compiled/interpreted=${formatDelta(compiledScaled, interpretedScaled)}\n`
    );
}

function runtimeCounterMedianSum(benchmark, keys, scale = 1) {
    let total = 0;
    let seen = false;
    for (const key of keys) {
        const value = benchmarkCountersMedian(benchmark, key);
        if (value !== null) {
            total += value / scale;
            seen = true;
        }
    }
    return seen ? total : null;
}

function printRuntimeCounterMedianSum(label, keys, digits = 1, scale = 1) {
    const interpretedValue = runtimeCounterMedianSum(interpreted, keys, scale);
    const compiledValue = runtimeCounterMedianSum(compiled, keys, scale);
    if (interpretedValue === null && compiledValue === null) {
        return;
    }
    process.stdout.write(
        `  median_runtime_${label}: interpreted=${formatNumber(interpretedValue, digits)}` +
        ` compiled=${formatNumber(compiledValue, digits)}` +
        ` compiled/interpreted=${formatDelta(compiledValue, interpretedValue)}\n`
    );
}

function printCompactTurnRuntimeCounters() {
    printRuntimeCounterMedian('compact_turn_native_calls', 'compact_turn_native_calls', 0);
    printRuntimeCounterMedian('compact_turn_bridge_calls', 'compact_turn_bridge_calls', 0);
    printRuntimeCounterMedian('compact_turn_setup_ms', 'compact_turn_setup_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_early_rules_ms', 'compact_turn_early_rules_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_movement_ms', 'compact_turn_movement_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_late_rules_ms', 'compact_turn_late_rules_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_win_ms', 'compact_turn_win_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_canonicalize_ms', 'compact_turn_canonicalize_ns', 3, 1e6);
    printRuntimeCounterMedianSum('compact_turn_native_phase_ms', [
        'compact_turn_setup_ns',
        'compact_turn_early_rules_ns',
        'compact_turn_movement_ns',
        'compact_turn_late_rules_ns',
        'compact_turn_win_ns',
        'compact_turn_canonicalize_ns',
    ], 3, 1e6);
    printRuntimeCounterMedian('compact_turn_bridge_create_ms', 'compact_turn_bridge_create_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_bridge_materialize_ms', 'compact_turn_bridge_materialize_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_bridge_turn_ms', 'compact_turn_bridge_turn_ns', 3, 1e6);
    printRuntimeCounterMedian('compact_turn_bridge_copyback_ms', 'compact_turn_bridge_copyback_ns', 3, 1e6);
    printRuntimeCounterMedianSum('compact_turn_bridge_phase_ms', [
        'compact_turn_bridge_create_ns',
        'compact_turn_bridge_materialize_ns',
        'compact_turn_bridge_turn_ns',
        'compact_turn_bridge_copyback_ns',
    ], 3, 1e6);
}

function printGraphSplit() {
    const split = [
        ['step', ['step_ms']],
        ['clone', ['clone_ms']],
        ['hash', ['hash_ms']],
        ['visited', ['visited_lookup_ms', 'visited_insert_ms']],
        ['frontier', ['frontier_pop_ms', 'frontier_push_ms']],
        ['node_store', ['node_store_ms']],
        ['heuristic', ['heuristic_ms']],
        ['solved_check', ['solved_check_ms']],
        ['timeout_check', ['timeout_check_ms']],
        ['unattributed', ['unattributed_ms']],
    ];
    const parts = [];
    for (const [label, keys] of split) {
        const interpretedValue = metricSum(interpreted, keys);
        const compiledValue = metricSum(compiled, keys);
        if (interpretedValue === null && compiledValue === null) {
            continue;
        }
        parts.push(`${label}=${formatNumber(interpretedValue, 1)}->${formatNumber(compiledValue, 1)}ms(${formatDelta(compiledValue, interpretedValue, 0)})`);
    }
    if (parts.length > 0) {
        process.stdout.write(`  median_graph_split: ${parts.join(' ')}\n`);
    }
}

function printWorkMismatchSummary(rows) {
    const generatedMismatches = rows.filter((row) => row.generatedRatio !== null && row.generatedRatio !== 1);
    const expandedMismatches = rows.filter((row) => row.expandedRatio !== null && row.expandedRatio !== 1);
    process.stdout.write(
        `  work_mismatches: generated=${generatedMismatches.length}` +
        ` expanded=${expandedMismatches.length}\n`
    );
    if (generatedMismatches.length === 0 && expandedMismatches.length === 0) {
        return;
    }
    const examples = generatedMismatches
        .slice()
        .sort((a, b) => Math.abs((a.generatedRatio || 1) - 1) > Math.abs((b.generatedRatio || 1) - 1) ? -1 : 1)
        .slice(0, 5);
    if (examples.length > 0) {
        process.stdout.write('  generated_mismatch_examples:\n');
        for (const row of examples) {
            process.stdout.write(
                `    ${formatNumber(row.generatedRatio, 3)}x generated` +
                ` interpreted=${formatNumber(row.interpreted.median.generated, 0)}` +
                ` compiled=${formatNumber(row.compiled.median.generated, 0)}` +
                ` ${row.key}\n`
            );
        }
    }
}

function printMaskRebuildTable(rows) {
    const hotRows = rows
        .filter((row) => Number.isFinite(row.maskRebuilds) && row.maskRebuilds > 0)
        .sort((a, b) => b.maskRebuilds - a.maskRebuilds)
        .slice(0, 10);
    if (hotRows.length === 0) {
        process.stdout.write('  top_mask_rebuilds: n/a\n');
        return;
    }
    process.stdout.write('  top_mask_rebuilds:\n');
    for (const row of hotRows) {
        process.stdout.write(
            `    mask_rebuilds=${row.maskRebuilds}` +
            ` elapsed=${formatNumber(row.compiled.median.elapsed_ms, 1)}ms` +
            ` interpreted=${formatNumber(row.interpreted.median.elapsed_ms, 1)}ms` +
            ` ratio=${formatNumber(row.elapsedRatio, 3)}x` +
            ` bucket=${row.bucket}` +
            ` reason=${row.reason}` +
            ` full_turn=${row.fullTurnHits === null ? 'n/a' : row.fullTurnHits}` +
            ` hits=${row.compiledRuleHits === null ? 'n/a' : row.compiledRuleHits}` +
            ` compact_native=${row.compactTurnNativeHits === null ? 'n/a' : row.compactTurnNativeHits}` +
            ` compact_bridge=${row.compactTurnBridgeHits === null ? 'n/a' : row.compactTurnBridgeHits}` +
            ` dirty=${row.maskRebuildDirtyCalls === null ? 'n/a' : row.maskRebuildDirtyCalls}` +
            ` rebuild_rows=${row.maskRebuildRows === null ? 'n/a' : row.maskRebuildRows}` +
            ` rebuild_columns=${row.maskRebuildColumns === null ? 'n/a' : row.maskRebuildColumns}` +
            ` rows=${row.rowScans === null ? 'n/a' : row.rowScans}` +
            ` cells=${row.candidateCells === null ? 'n/a' : row.candidateCells}` +
            ` pattern_tests=${row.patternTests === null ? 'n/a' : row.patternTests}` +
            ` ${row.key}` +
            `\n`
        );
    }
}

function printStepTimeTable(label, rows) {
    const visibleRows = rows
        .filter((row) => Number.isFinite(row.stepRatio))
        .slice(0, 10);
    if (visibleRows.length === 0) {
        process.stdout.write(`  ${label} n/a\n`);
        return;
    }
    process.stdout.write(`  ${label}\n`);
    for (const row of visibleRows) {
        process.stdout.write(
            `    ${formatNumber(row.stepRatio, 3)}x step` +
            ` elapsed=${formatNumber(row.elapsedRatio, 3)}x` +
            ` wall=${formatNumber(row.wallRatio, 3)}x` +
            ` generated=${formatNumber(row.generatedRatio, 3)}x` +
            ` expanded=${formatNumber(row.expandedRatio, 3)}x` +
            ` interpreted_step=${formatNumber(metricMedian(row.interpreted, 'step_ms'), 1)}ms` +
            ` compiled_step=${formatNumber(metricMedian(row.compiled, 'step_ms'), 1)}ms` +
            ` bucket=${row.bucket}` +
            ` reason=${row.reason}` +
            ` ${row.key}` +
            `\n`
        );
    }
}

function printGraphOverheadTable(label, rows) {
    const visibleRows = rows
        .filter((row) => Number.isFinite(row.graphOverheadRatio))
        .slice(0, 10);
    if (visibleRows.length === 0) {
        process.stdout.write(`  ${label} n/a\n`);
        return;
    }
    process.stdout.write(`  ${label}\n`);
    for (const row of visibleRows) {
        process.stdout.write(
            `    ${formatNumber(row.graphOverheadRatio, 3)}x graph` +
            ` elapsed=${formatNumber(row.elapsedRatio, 3)}x` +
            ` step=${formatNumber(row.stepRatio, 3)}x` +
            ` graph_ms=${formatNumber(row.interpretedGraph, 1)}->${formatNumber(row.compiledGraph, 1)}` +
            ` visited=${formatNumber(row.visitedRatio, 3)}x` +
            ` frontier=${formatNumber(row.frontierRatio, 3)}x` +
            ` node_store=${formatNumber(row.nodeStoreRatio, 3)}x` +
            ` heuristic=${formatNumber(row.heuristicRatio, 3)}x` +
            ` unattributed=${formatNumber(row.unattributedRatio, 3)}x` +
            ` ${row.key}` +
            `\n`
        );
    }
}

process.stdout.write('solver_focus_compare\n');
process.stdout.write(`  targets: interpreted=${interpreted.target_count} compiled=${compiled.target_count} same=${sameTargets ? 'yes' : 'no'}\n`);
process.stdout.write(`  runs_per_target: interpreted=${interpreted.runs_per_target} compiled=${compiled.runs_per_target}\n`);
process.stdout.write(`  solver_extra_args: interpreted=${formatArgs(interpreted.solver_extra_args)} compiled=${formatArgs(compiled.solver_extra_args)}\n`);
process.stdout.write(`  status: interpreted=${JSON.stringify(interpretedStatus)} compiled=${JSON.stringify(compiledStatus)}\n`);
process.stdout.write(
    `  median_wall_ms: interpreted=${formatNumber(interpreted.median.wall_ms)}` +
    ` compiled=${formatNumber(compiled.median.wall_ms)}` +
    ` compiled/interpreted=${formatDelta(compiled.median.wall_ms, interpreted.median.wall_ms)}\n`
);
process.stdout.write(
    `  median_elapsed_ms: interpreted=${formatNumber(interpreted.median.elapsed_ms)}` +
    ` compiled=${formatNumber(compiled.median.elapsed_ms)}` +
    ` compiled/interpreted=${formatDelta(compiled.median.elapsed_ms, interpreted.median.elapsed_ms)}\n`
);
process.stdout.write(
    `  median_generated: interpreted=${formatNumber(interpreted.median.generated, 0)}` +
    ` compiled=${formatNumber(compiled.median.generated, 0)}` +
    ` compiled/interpreted=${formatDelta(compiled.median.generated, interpreted.median.generated)}\n`
);
printMedianMetric('step_ms', 'step_ms');
printMedianMetric('clone_ms', 'clone_ms');
printMedianMetric('hash_ms', 'hash_ms');
printMedianMetricSum('visited_ms', ['visited_lookup_ms', 'visited_insert_ms']);
printMedianMetric('visited_lookup_probes', 'visited_lookup_probes');
printMedianMetric('visited_insert_probes', 'visited_insert_probes');
printMedianMetric('visited_max_probe', 'visited_max_probe');
printMedianMetric('visited_key_collisions', 'visited_key_collisions');
printMedianMetric('compact_state_bytes', 'compact_state_bytes');
printMedianMetric('compact_max_state_bytes', 'compact_max_state_bytes');
printMedianMetric('compact_turn_attempts', 'compact_turn_attempts');
printMedianMetric('compact_turn_hits', 'compact_turn_hits');
printMedianMetric('compact_turn_native_attempts', 'compact_turn_native_attempts');
printMedianMetric('compact_turn_native_hits', 'compact_turn_native_hits');
printMedianMetric('compact_turn_bridge_attempts', 'compact_turn_bridge_attempts');
printMedianMetric('compact_turn_bridge_hits', 'compact_turn_bridge_hits');
printMedianMetric('compact_turn_fallbacks', 'compact_turn_fallbacks');
printMedianMetric('compact_turn_unsupported', 'compact_turn_unsupported');
printCompactTurnSummary();
printCompactTurnRuntimeCounters();
printMedianMetricSum('frontier_ms', ['frontier_pop_ms', 'frontier_push_ms']);
printMedianMetric('node_store_ms', 'node_store_ms');
printMedianMetric('heuristic_ms', 'heuristic_ms');
printMedianMetric('solved_check_ms', 'solved_check_ms');
printMedianMetric('timeout_check_ms', 'timeout_check_ms');
printMedianMetric('unattributed_ms', 'unattributed_ms');
printMedianMetricSum('graph_overhead_ms', [
    'frontier_pop_ms',
    'frontier_push_ms',
    'visited_lookup_ms',
    'visited_insert_ms',
    'node_store_ms',
    'heuristic_ms',
    'solved_check_ms',
    'timeout_check_ms',
    'unattributed_ms',
]);
printGraphSplit();
if (options.goalRatio !== null) {
    const elapsedRatio = ratio(compiled.median.elapsed_ms, interpreted.median.elapsed_ms);
    process.stdout.write(
        `  goal_elapsed_ratio: target<=${formatNumber(options.goalRatio, 3)}` +
        ` actual=${elapsedRatio === null ? 'n/a' : formatNumber(elapsedRatio, 3)}` +
        ` pass=${elapsedRatio !== null && elapsedRatio <= options.goalRatio ? 'yes' : 'no'}\n`
    );
}

if (options.detail) {
    const rows = targetRows();
    printWorkMismatchSummary(rows);
    printCompiledUsageSummary(rows);
    const bySlowest = rows.slice().sort((a, b) => (b.elapsedRatio || 0) - (a.elapsedRatio || 0));
    const byFastest = rows.slice().sort((a, b) => (a.elapsedRatio || Infinity) - (b.elapsedRatio || Infinity));
    const byStepSlowest = rows.slice().sort((a, b) => (b.stepRatio || 0) - (a.stepRatio || 0));
    const byStepFastest = rows.slice().sort((a, b) => (a.stepRatio || Infinity) - (b.stepRatio || Infinity));
    const byGraphSlowest = rows.slice().sort((a, b) => (b.graphOverheadRatio || 0) - (a.graphOverheadRatio || 0));
    const byGraphLargest = rows.slice().sort((a, b) => (b.compiledGraph || 0) - (a.compiledGraph || 0));
    printTargetTable('slowest_targets:', bySlowest.slice(0, 10));
    printTargetTable('fastest_targets:', byFastest.slice(0, 10));
    printStepTimeTable('slowest_step_targets:', byStepSlowest);
    printStepTimeTable('fastest_step_targets:', byStepFastest);
    printGraphOverheadTable('slowest_graph_overhead_targets:', byGraphSlowest);
    printGraphOverheadTable('largest_compiled_graph_overhead_targets:', byGraphLargest);
    printMaskRebuildTable(rows);
}

if (!sameTargets) {
    process.exitCode = 1;
}
if (options.goalRatio !== null) {
    const elapsedRatio = ratio(compiled.median.elapsed_ms, interpreted.median.elapsed_ms);
    if (elapsedRatio === null || elapsedRatio > options.goalRatio) {
        process.exitCode = 2;
    }
}
