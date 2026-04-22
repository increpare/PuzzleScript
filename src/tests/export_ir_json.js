#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./lib/puzzlescript_node_env');
const { buildCompiledIr } = require('./lib/puzzlescript_ir');
const { buildParserStateSnapshot } = require('./lib/puzzlescript_parser_snapshot');

function parseArgs(argv) {
    const result = {
        command: 'restart',
        level: 0,
        randomseed: null,
        settleAgain: false,
        snapshotPhase: null,
        inputFile: null,
        outputFile: null,
    };

    const args = argv.slice(2);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === '--level') {
            result.level = Number.parseInt(args[++index], 10);
            result.command = 'loadLevel';
        } else if (arg === '--seed') {
            result.randomseed = args[++index];
        } else if (arg === '--settle-again') {
            result.settleAgain = true;
        } else if (arg === '--snapshot-phase') {
            result.snapshotPhase = args[++index];
        } else if (arg === '--restart') {
            result.command = 'restart';
        } else if (arg === '--help' || arg === '-h') {
            result.help = true;
        } else if (result.inputFile === null) {
            result.inputFile = arg;
        } else if (result.outputFile === null) {
            result.outputFile = arg;
        } else {
            throw new Error(`Unexpected argument: ${arg}`);
        }
    }

    return result;
}

function usage() {
    console.error('Usage: node src/tests/export_ir_json.js <input.ps> [output.json] [--level N] [--seed seed] [--settle-again] [--snapshot-phase parser]');
}

function main() {
    const options = parseArgs(process.argv);
    if (options.help || !options.inputFile) {
        usage();
        process.exit(options.help ? 0 : 1);
    }

    const inputFile = path.resolve(options.inputFile);
    const outputFile = options.outputFile ? path.resolve(options.outputFile) : null;

    const source = fs.readFileSync(inputFile, 'utf8');
    loadPuzzleScript();

    if (options.snapshotPhase === 'parser') {
        const snapshot = buildParserStateSnapshot(`${source}\n`);
        const payload = JSON.stringify(snapshot, null, 2);
        if (outputFile) {
            fs.mkdirSync(path.dirname(outputFile), { recursive: true });
            fs.writeFileSync(outputFile, `${payload}\n`, 'utf8');
        } else {
            process.stdout.write(`${payload}\n`);
        }
        return;
    }

    const command = options.command === 'loadLevel'
        ? ['loadLevel', options.level]
        : ['restart'];

    unitTesting = true;
    lazyFunctionGeneration = false;
    try {
        compile(command, `${source}\n`, options.randomseed);
        if (options.settleAgain) {
            while (againing) {
                againing = false;
                processInput(-1);
            }
        }
    } finally {
        unitTesting = false;
        lazyFunctionGeneration = true;
    }

    const document = {
        input_file: inputFile,
        command,
        random_seed: options.randomseed,
        error_count: errorCount,
        errors: errorStrings.map(stripHTMLTags),
    };
    const ir = buildCompiledIr(document);
    const payload = JSON.stringify(ir, null, 2);

    if (outputFile) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, `${payload}\n`, 'utf8');
    } else {
        process.stdout.write(`${payload}\n`);
    }
}

main();
