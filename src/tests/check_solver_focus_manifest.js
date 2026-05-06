#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '../..');
const makefilePath = path.join(repoRoot, 'Makefile');
const minerPath = path.join(repoRoot, 'src/tests/mine_solver_focus_group.js');
const manifestPath = path.join(repoRoot, 'src/tests/solver_focus_group.json');
const corpusPath = path.join(repoRoot, 'src/tests/solver_tests');

function fail(message) {
    console.error(`solver_focus_manifest check failed: ${message}`);
    process.exit(1);
}

function targetKey(target) {
    return `${target.game}#${target.level}`;
}

const makefile = fs.readFileSync(makefilePath, 'utf8');
if (!/^SOLVER_FOCUS_MANIFEST \?= src\/tests\/solver_focus_group\.json$/m.test(makefile)) {
    fail('Makefile default SOLVER_FOCUS_MANIFEST must point at src/tests/solver_focus_group.json');
}

const miner = fs.readFileSync(minerPath, 'utf8');
if (!/let outPath = path\.resolve\('src\/tests\/solver_focus_group\.json'\);/.test(miner)) {
    fail('mine_solver_focus_group.js default --out must point at src/tests/solver_focus_group.json');
}

if (!fs.existsSync(manifestPath)) {
    fail(`missing tracked focus manifest: ${manifestPath}`);
}

const ignoreResult = spawnSync('git', ['check-ignore', '-q', 'src/tests/solver_focus_group.json'], {
    cwd: repoRoot,
    encoding: 'utf8',
});
if (ignoreResult.status === 0) {
    fail('src/tests/solver_focus_group.json is ignored by git');
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const manifestText = JSON.stringify(manifest);
if (manifestText.includes(repoRoot) || manifestText.includes('/Users/')) {
    fail('checked-in focus manifest must not contain machine-local absolute paths');
}
if (manifest.schema_version !== 1) {
    fail(`schema_version must be 1, got ${manifest.schema_version}`);
}
if (manifest.kind !== 'solver_focus_group') {
    fail(`kind must be solver_focus_group, got ${manifest.kind}`);
}
if (!Array.isArray(manifest.targets) || manifest.targets.length === 0) {
    fail('targets must be a non-empty array');
}
if (manifest.target_count !== manifest.targets.length) {
    fail(`target_count=${manifest.target_count}, expected ${manifest.targets.length}`);
}

const seen = new Set();
for (const target of manifest.targets) {
    if (!target || typeof target.game !== 'string' || !Number.isInteger(target.level)) {
        fail(`invalid target shape: ${JSON.stringify(target)}`);
    }
    if (!Number.isFinite(target.first_solved_timeout_ms) || target.first_solved_timeout_ms <= 0) {
        fail(`target ${targetKey(target)} has no positive first_solved_timeout_ms`);
    }
    const key = targetKey(target);
    if (seen.has(key)) {
        fail(`duplicate target ${key}`);
    }
    seen.add(key);
    const gamePath = path.resolve(corpusPath, target.game);
    const relative = path.relative(corpusPath, gamePath);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        fail(`target escapes solver corpus: ${target.game}`);
    }
    if (!fs.existsSync(gamePath) || !fs.statSync(gamePath).isFile()) {
        fail(`target game does not exist: ${target.game}`);
    }
}

console.log(`solver_focus_manifest ok targets=${manifest.targets.length}`);
