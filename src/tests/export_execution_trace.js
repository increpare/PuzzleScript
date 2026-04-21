#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./lib/puzzlescript_node_env');
const { runInputTrace } = require('./lib/puzzlescript_trace');

function parseArgs(argv) {
    const result = {
        command: 'restart',
        level: 0,
        randomseed: null,
        inputFile: null,
        outputFile: null,
        inputs: [],
    };

    const args = argv.slice(2);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === '--level') {
            result.level = Number.parseInt(args[++index], 10);
            result.command = 'loadLevel';
        } else if (arg === '--seed') {
            result.randomseed = args[++index];
        } else if (arg === '--inputs-json') {
            result.inputs = JSON.parse(args[++index]);
        } else if (arg === '--inputs-file') {
            const inputPath = path.resolve(args[++index]);
            result.inputs = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
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
    console.error('Usage: node src/tests/export_execution_trace.js <input.ps> [output.json] [--level N] [--seed seed] [--inputs-json json] [--inputs-file path]');
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

    const command = options.command === 'loadLevel'
        ? ['loadLevel', options.level]
        : ['restart'];

    unitTesting = true;
    lazyFunctionGeneration = false;
    try {
        compile(command, `${source}\n`, options.randomseed);
        while (againing) {
            againing = false;
            processInput(-1);
        }
        const trace = runInputTrace(options.inputs);
        const payload = JSON.stringify({
            schema_version: 1,
            input_file: inputFile,
            command,
            random_seed: options.randomseed,
            inputs: options.inputs,
            trace,
        }, null, 2);

        if (outputFile) {
            fs.mkdirSync(path.dirname(outputFile), { recursive: true });
            fs.writeFileSync(outputFile, `${payload}\n`, 'utf8');
        } else {
            process.stdout.write(`${payload}\n`);
        }
    } finally {
        unitTesting = false;
        lazyFunctionGeneration = true;
    }
}

main();
