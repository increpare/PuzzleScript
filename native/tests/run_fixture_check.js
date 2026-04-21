#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

function runChecked(command, args, label) {
    const result = spawnSync(command, args, {
        stdio: 'inherit',
        cwd: process.cwd(),
    });
    if (result.status !== 0) {
        throw new Error(`${label} failed with exit code ${result.status}`);
    }
}

function main(argv) {
    const [exporterScript, psCliPath, outputDir, ...psCliArgs] = argv.slice(2);
    if (!exporterScript || !psCliPath || !outputDir) {
        throw new Error('Usage: run_fixture_check.js <exporter-script> <ps-cli> <output-dir> [ps-cli args...]');
    }

    fs.rmSync(outputDir, { recursive: true, force: true });
    fs.mkdirSync(outputDir, { recursive: true });

    runChecked(process.execPath, [path.resolve(exporterScript), path.resolve(outputDir)], 'fixture export');
    runChecked(
        path.resolve(psCliPath),
        ['test-fixtures', path.join(path.resolve(outputDir), 'fixtures.json'), ...psCliArgs],
        'fixture validation'
    );
}

main(process.argv);
