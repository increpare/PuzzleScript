#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/extract_solver_focus_corpus.js <focus_manifest.json> <solver_tests_dir> <out_dir>');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length !== 3) {
    usage();
}

const manifestPath = path.resolve(args[0]);
const corpusPath = path.resolve(args[1]);
const outPath = path.resolve(args[2]);

function assertInside(parent, child, label) {
    const relative = path.relative(parent, child);
    if (relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))) {
        return relative;
    }
    throw new Error(`${label} escapes corpus: ${child}`);
}

function targetKey(target) {
    return `${target.game}#${target.level}`;
}

if (!fs.existsSync(manifestPath)) {
    throw new Error(`focus manifest does not exist: ${manifestPath}`);
}
if (!fs.existsSync(corpusPath) || !fs.statSync(corpusPath).isDirectory()) {
    throw new Error(`solver corpus is not a directory: ${corpusPath}`);
}
if (outPath === path.parse(outPath).root || outPath.length < 8) {
    throw new Error(`refusing suspicious output directory: ${outPath}`);
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const targets = manifest.targets || [];
if (!Array.isArray(targets)) {
    throw new Error(`focus manifest targets must be an array: ${manifestPath}`);
}

const selected = new Map();
for (const target of targets) {
    if (!target || typeof target.game !== 'string' || target.game.length === 0) {
        throw new Error(`focus target has no game: ${JSON.stringify(target)}`);
    }
    const sourcePath = path.resolve(corpusPath, target.game);
    const relative = assertInside(corpusPath, sourcePath, `focus target ${targetKey(target)}`);
    if (!fs.existsSync(sourcePath) || !fs.statSync(sourcePath).isFile()) {
        throw new Error(`focus target source does not exist: ${sourcePath}`);
    }
    selected.set(relative, sourcePath);
}

fs.rmSync(outPath, { recursive: true, force: true });
for (const [relative, sourcePath] of selected) {
    const destination = path.join(outPath, relative);
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(sourcePath, destination);
}

const index = {
    schema_version: 1,
    kind: 'solver_focus_corpus',
    manifest: manifestPath,
    source_corpus: corpusPath,
    target_count: targets.length,
    source_count: selected.size,
    sources: Array.from(selected.keys()).sort(),
};
fs.writeFileSync(path.join(outPath, 'focus-corpus.json'), `${JSON.stringify(index, null, 2)}\n`);
process.stdout.write(`solver_focus_corpus wrote ${outPath} sources=${selected.size} targets=${targets.length}\n`);
