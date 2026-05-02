#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { analyzeFile, discoverInputFiles } = require('./ps_static_analysis');

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

const files = discoverInputFiles(inputs)
    .filter(filePath => !options.gameFilter || filePath.toLowerCase().includes(options.gameFilter.toLowerCase()));

function writeIndentedJson(write, value, indent) {
    const prefix = ' '.repeat(indent);
    const json = JSON.stringify(value, null, 2).split('\n').map(line => `${prefix}${line}`).join('\n');
    write(json);
}

function writeBatch(write) {
    write('{\n');
    write('  "schema": "ps-static-analysis-batch-v1",\n');
    write(`  "generated_at": ${JSON.stringify(new Date().toISOString())},\n`);
    write(`  "source_count": ${files.length},\n`);
    write('  "reports": [\n');
    files.forEach((filePath, index) => {
        if (index > 0) write(',\n');
        writeIndentedJson(write, analyzeFile(filePath, options), 4);
    });
    write('\n  ]\n');
    write('}\n');
}

if (outPath) {
    const fd = fs.openSync(outPath, 'w');
    try {
        writeBatch(chunk => fs.writeSync(fd, chunk));
    } finally {
        fs.closeSync(fd);
    }
} else {
    writeBatch(chunk => process.stdout.write(chunk));
}
