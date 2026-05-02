#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { analyzeFile } = require('./ps_static_analysis');

function usage() {
    console.error('Usage: node src/tests/run_ps_static_analysis.js <file.txt> [--out PATH] [--no-ps-tagged]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    usage();
}

const inputPath = args[0];
const options = { includePsTagged: true };
let outPath = null;

for (let index = 1; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--no-ps-tagged') {
        options.includePsTagged = false;
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

const report = analyzeFile(inputPath, options);
const json = `${JSON.stringify(report, null, 2)}\n`;
if (outPath) {
    fs.writeFileSync(outPath, json);
} else {
    process.stdout.write(json);
}
