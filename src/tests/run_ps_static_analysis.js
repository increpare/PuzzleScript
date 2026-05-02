#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { analyzePaths } = require('./ps_static_analysis');

function usage() {
    console.error([
        'Usage: node src/tests/run_ps_static_analysis.js <file-or-dir> [more paths]',
        '  [--out PATH] [--family NAME] [--game SUBSTRING]',
        '  [--include-ps-tagged] [--no-ps-tagged]',
    ].join('\n'));
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    usage();
}

const inputs = [];
const options = { includePsTagged: true, familyFilter: null, gameFilter: null };
let outPath = null;

for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--family' && index + 1 < args.length) {
        options.familyFilter = args[++index];
    } else if (arg === '--game' && index + 1 < args.length) {
        options.gameFilter = args[++index];
    } else if (arg === '--include-ps-tagged') {
        options.includePsTagged = true;
    } else if (arg === '--no-ps-tagged') {
        options.includePsTagged = false;
    } else if (arg.startsWith('--')) {
        throw new Error(`Unsupported argument: ${arg}`);
    } else {
        inputs.push(arg);
    }
}

const reports = analyzePaths(inputs, options);
const output = {
    schema: 'ps-static-analysis-batch-v1',
    generated_at: new Date().toISOString(),
    source_count: reports.length,
    reports,
};

const json = `${JSON.stringify(output, null, 2)}\n`;
if (outPath) {
    fs.writeFileSync(outPath, json);
} else {
    process.stdout.write(json);
}
