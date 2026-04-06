#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const { canonicalizeSource } = require('../canonicalize');
const { decanonicalizeSemantic } = require('../decanonicalize');

function getDemoFiles() {
    return fs.readdirSync(path.join('src', 'demo'))
        .filter(name => name.endsWith('.txt'))
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))
        .map(name => path.join('src', 'demo', name));
}

function compilePuzzleScript(inputPath, outputPath) {
    execFileSync(process.execPath, ['src/compile_cli.js', inputPath, outputPath], {
        cwd: process.cwd(),
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
    });
}

function ensureCleanDir(dirPath) {
    fs.rmSync(dirPath, { recursive: true, force: true });
    fs.mkdirSync(dirPath, { recursive: true });
}

function main() {
    const demoFiles = getDemoFiles();
    const canonicalDir = path.join('src', 'demo_canonical');
    const rehydratedDir = path.join('src', 'demo_rehydrated');
    ensureCleanDir(canonicalDir);
    ensureCleanDir(rehydratedDir);
    const failures = [];
    let passed = 0;

    for (const demoFile of demoFiles) {
        const baseName = path.basename(demoFile, '.txt');
        const canonicalPath = path.join(canonicalDir, `${baseName}.json`);
        const regeneratedPath = path.join(rehydratedDir, `${baseName}.txt`);
        const compiledHtmlPath = path.join(rehydratedDir, `${baseName}.html`);

        try {
            const source = fs.readFileSync(demoFile, 'utf8');
            const canonical = canonicalizeSource(source, 'semantic');
            fs.writeFileSync(canonicalPath, `${JSON.stringify(canonical, null, 2)}\n`, 'utf8');
            const rehydrated = decanonicalizeSemantic(canonical);
            fs.writeFileSync(regeneratedPath, rehydrated, 'utf8');
            compilePuzzleScript(regeneratedPath, compiledHtmlPath);
            passed++;
            console.log(`PASS ${demoFile}`);
        } catch (error) {
            failures.push({
                file: demoFile,
                message: error.stderr || error.stdout || error.message || String(error),
            });
            console.log(`FAIL ${demoFile}`);
        }
    }

    console.log(`\nRound-trip compile check: ${passed}/${demoFiles.length} passed`);

    if (failures.length > 0) {
        console.log('');
        for (const failure of failures) {
            console.log(`--- ${failure.file} ---`);
            console.log(String(failure.message).trim());
            console.log('');
        }
        process.exit(1);
    }
}

main();
