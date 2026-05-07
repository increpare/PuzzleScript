#!/usr/bin/env node
'use strict';

const assert = require('assert');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');
const { analyzeSource } = require('./ps_static_analysis');
const {
    parseSolverOptPassList,
    resolveSolverPasses,
    createSolverOptimizationHook,
    collectWinconditionLegendRefs,
    collectObjectNamesFromCompiledLevels,
    expandLegendRefsToConcreteObjectNames,
    effectiveSolverPassesForHook,
    buildSolverOptimizationJsonTotals,
    formatSolverOptimizationHumanSuffixFromTotals,
    isInertCommandOnlyCompiledRule,
    applyNameSubstitutionToWinconditions,
} = require('./solver_static_opt');

function assertThrows(fn, msg) {
    let threw = false;
    try {
        fn();
    } catch (e) {
        threw = true;
        if (msg && !String(e.message || e).includes(msg)) {
            throw e;
        }
    }
    assert.ok(threw, 'expected throw');
}

function run() {
    const mockState = {
        objects: { rock: {}, crate: {}, gem: {} },
        aggregatesDict: { pile: ['rock', 'crate'] },
        propertiesDict: { shiny: ['gem', 'crate'] },
        winconditions: [['some', 'pile', 'on', 'shiny', 99]],
    };
    const winRefs = collectWinconditionLegendRefs(mockState);
    assert.ok(winRefs.has('pile'));
    assert.ok(winRefs.has('shiny'));

    const mergeWinState = {
        objects: { a: {}, b: {} },
        aggregatesDict: {},
        propertiesDict: {},
        winconditions: [['all', 'oldname', 'on', 'b', 777]],
    };
    applyNameSubstitutionToWinconditions(mergeWinState, nm => (nm === 'oldname' ? 'a' : undefined));
    assert.deepStrictEqual(mergeWinState.winconditions[0], ['all', 'a', 'on', 'b', 777]);

    const lineOnly = { objects: { y: {} }, aggregatesDict: {}, propertiesDict: {}, winconditions: [['no', 'x', 99]] };
    applyNameSubstitutionToWinconditions(lineOnly, nm => (nm === 'x' ? 'y' : undefined));
    assert.deepStrictEqual(lineOnly.winconditions[0], ['no', 'y', 99], 'last wincondition token is line number, not rewritten');
    const expanded = expandLegendRefsToConcreteObjectNames(mockState, new Set(['pile', 'shiny']));
    assert.deepStrictEqual(new Set([...expanded].sort()), new Set(['crate', 'gem', 'rock']));

    const levelScanState = {
        objectCount: 3,
        STRIDE_OBJ: 1,
        idDict: ['bg', 'player', 'star'],
        levels: [{ n_tiles: 1, objects: new Int32Array([1 << 2]) }],
    };
    const onMap = collectObjectNamesFromCompiledLevels(levelScanState);
    assert.ok(onMap.has('star'));
    assert.ok(!onMap.has('player'));

    assert.deepStrictEqual(
        effectiveSolverPassesForHook(null, { inert: true, cosmetic: true, merge: true }),
        { inert: true, cosmetic: false, merge: false },
    );
    assert.deepStrictEqual(
        effectiveSolverPassesForHook({ status: 'compile_error' }, { inert: true, cosmetic: true, merge: true }),
        { inert: true, cosmetic: false, merge: false },
    );
    assert.deepStrictEqual(
        effectiveSolverPassesForHook({ status: 'ok' }, { inert: false, cosmetic: true, merge: true }),
        { inert: false, cosmetic: true, merge: true },
    );
    const nest = buildSolverOptimizationJsonTotals({
        static_optimization_removed_rules: 1,
        removed_cosmetic_objects: 0,
        removed_collision_layers: 0,
        merged_object_aliases: 0,
        merged_object_groups: 0,
        solver_opt_ms_inert: 0.01,
        solver_opt_ms_cosmetic: 0,
        solver_opt_ms_merge: 0,
    });
    assert.strictEqual(nest.removed_inert_rules, 1);
    assert.ok(nest.ms_hook > 0);

    const nestGated = buildSolverOptimizationJsonTotals({
        static_optimization_removed_rules: 0,
        removed_cosmetic_objects: 0,
        removed_collision_layers: 0,
        merged_object_aliases: 0,
        merged_object_groups: 0,
        solver_opt_ms_inert: 0,
        solver_opt_ms_cosmetic: 0,
        solver_opt_ms_merge: 0,
        solver_optimization_gated: true,
    });
    assert.strictEqual(nestGated.gated, true);
    assert.ok(formatSolverOptimizationHumanSuffixFromTotals({ solver_optimization_gated: true }).includes('opt_gated=1'));

    assert.deepStrictEqual(parseSolverOptPassList('all'), { inert: true, cosmetic: true, merge: true });
    assert.deepStrictEqual(parseSolverOptPassList('inert,cosmetic'), { inert: true, cosmetic: true, merge: false });
    assertThrows(() => parseSolverOptPassList('nope'), 'Unknown');

    const inertLine = new Set([42]);
    assert.strictEqual(
        isInertCommandOnlyCompiledRule(
            { lineNumber: 42, randomRule: true, hasReplacements: false, commands: [['message', 'x']] },
            inertLine,
        ),
        false,
        'never drop random rules tagged inert-command-only (can affect nondeterminism)',
    );
    assert.strictEqual(
        isInertCommandOnlyCompiledRule(
            { lineNumber: 42, hasReplacements: false, commands: [['message', 'x']] },
            inertLine,
        ),
        true,
    );

    const opt = { solverOptimizeStatic: true, solverOptPasses: { cosmetic: true, merge: false } };
    const merged = resolveSolverPasses(opt);
    assert.strictEqual(merged.inert, true);
    assert.strictEqual(merged.cosmetic, true);

    const baseline = resolveSolverPasses(Object.assign({}, opt, { solverOptParityBaseline: true }));
    assert.deepStrictEqual(baseline, { inert: false, cosmetic: false, merge: false });

    loadPuzzleScript();
    const smokePath = path.join(__dirname, 'solver_smoke_tests', 'one_move.txt');
    const source = fs.readFileSync(smokePath, 'utf8');
    const report = analyzeSource(source, { sourcePath: smokePath });
    assert.strictEqual(report.status, 'ok');

    const hook = createSolverOptimizationHook(report, { inert: true, cosmetic: true, merge: true });
    setPluginOptimizationHook(hook);
    try {
        compile(['loadLevel', 0], source, 'solver_static_opt_node');
    } finally {
        setPluginOptimizationHook(null);
    }
    assert.strictEqual(errorCount, 0, 'compile should succeed with full optimization hook');
    assert.ok(state && state.solverOptimizationTelemetry, 'telemetry attached');
    const tel = state.solverOptimizationTelemetry;
    assert.ok(typeof tel.removed_inert_rules === 'number');
    assert.ok(typeof tel.ms_inert === 'number' && tel.ms_inert >= 0);
    assert.ok(typeof tel.ms_cosmetic === 'number' && tel.ms_cosmetic >= 0);
    assert.ok(typeof tel.ms_merge === 'number' && tel.ms_merge >= 0);

    const backgroundOnlySource = `
title Solver Static Cosmetic Background

========
OBJECTS
========

background
black

Player
blue

Target
green

========
LEGEND
========

. = background
P = Player and background
T = Target and background

========
SOUNDS
========

================
COLLISIONLAYERS
================

background
Target
Player

======
RULES
======

=============
WINCONDITIONS
=============

all Player on Target

======
LEVELS
======

PT
`;
    const backgroundReport = analyzeSource(backgroundOnlySource, { sourcePath: 'solver_static_cosmetic_background.txt' });
    assert.strictEqual(backgroundReport.status, 'ok');
    assert.strictEqual(
        backgroundReport.ps_tagged.objects.find(object => object.name === 'background').tags.cosmetic,
        true,
        'fixture should exercise a cosmetic-tagged background object',
    );
    const backgroundHook = createSolverOptimizationHook(backgroundReport, { inert: false, cosmetic: true, merge: false });
    setPluginOptimizationHook(backgroundHook);
    try {
        compile(['loadLevel', 0], backgroundOnlySource, 'solver_static_cosmetic_background');
    } finally {
        setPluginOptimizationHook(null);
    }
    assert.strictEqual(errorCount, 0, 'cosmetic pruning must preserve the compiler background object');
    assert.ok(state && state.objects && state.objects.background, 'background object should remain after cosmetic pruning');

    const corpusDir = path.join(__dirname, 'solver_smoke_tests');
    const runner = path.join(__dirname, 'run_solver_tests_js.js');
    const mlJson = execFileSync(
        process.execPath,
        [
            runner,
            corpusDir,
            '--game',
            'multi_level.txt',
            '--quiet',
            '--no-solutions',
            '--solver-optimize-static',
            '--json',
        ],
        { encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 },
    );
    const mlPayload = JSON.parse(mlJson);
    const mlLevels = mlPayload.results.filter(r => r.game === 'multi_level.txt' && r.level >= 0);
    assert.ok(mlLevels.length >= 2, 'multi_level fixture should yield 2+ level rows');
    const rowsWithCompileMs = mlLevels.filter(r => (r.compile_ms || 0) > 0);
    assert.strictEqual(
        rowsWithCompileMs.length,
        1,
        'per-game compile_ms should appear on first level row only (totals aggregation)',
    );
    const tMl = mlPayload.totals;
    assert.strictEqual(tMl.compile_ms, rowsWithCompileMs[0].compile_ms);
    assert.strictEqual(
        tMl.solver_opt_ms_inert + tMl.solver_opt_ms_cosmetic + tMl.solver_opt_ms_merge,
        (rowsWithCompileMs[0].solver_opt_ms_inert || 0)
            + (rowsWithCompileMs[0].solver_opt_ms_cosmetic || 0)
            + (rowsWithCompileMs[0].solver_opt_ms_merge || 0),
    );

    process.stdout.write('solver_static_opt_node: ok\n');
}

run();
