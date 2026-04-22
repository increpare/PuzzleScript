#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

function firstNonEmptyLine(text) {
    if (!text) {
        return '';
    }
    return text.split(/\r?\n/).find(line => line.trim().length > 0) || '';
}

function main() {
    const manifestPath = process.argv[2] || 'build/native/coverage-fixtures/fixtures.json';
    const cliPath = process.argv[3] || 'build/native/native/ps_cli';
    const manifestDir = path.dirname(manifestPath);
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    const fixtures = manifest.simulation_fixtures || [];

    let failed = 0;
    for (const fixture of fixtures) {
        if (!fixture.trace_file) {
            continue;
        }

        const irPath = path.join(manifestDir, fixture.ir_file);
        const tracePath = path.join(manifestDir, fixture.trace_file);
        const run = spawnSync(cliPath, ['diff-trace', irPath, tracePath], {
            encoding: 'utf8',
            timeout: 30000,
            maxBuffer: 16 * 1024 * 1024,
        });

        if (!run.error && run.status === 0) {
            continue;
        }

        failed += 1;
        const detail = run.error
            ? String(run.error.message || run.error)
            : firstNonEmptyLine(run.stderr) || firstNonEmptyLine(run.stdout);
        process.stdout.write(`${failed}. ${fixture.name}\n`);
        if (detail) {
            process.stdout.write(`   ${detail}\n`);
        }
    }

    process.stdout.write(`failed_total=${failed}\n`);
}

main();
