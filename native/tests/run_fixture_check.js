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
    const [exporterScript, cliPath, outputDir, ...cliArgs] = argv.slice(2);
    if (!exporterScript || !cliPath || !outputDir) {
        throw new Error('Usage: run_fixture_check.js <exporter-script> <puzzlescript_cpp> <output-dir> [puzzlescript_cpp args...]');
    }

    fs.rmSync(outputDir, { recursive: true, force: true });
    fs.mkdirSync(outputDir, { recursive: true });

    runChecked(process.execPath, [path.resolve(exporterScript), path.resolve(outputDir)], 'JS parity data export');
    runChecked(
        path.resolve(cliPath),
        ['test-js-parity-data', path.join(path.resolve(outputDir), 'fixtures.json'), ...cliArgs],
        'JS parity data validation'
    );
}

main(process.argv);
