#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const { spawnSync } = require('child_process');

function loadJsArray(filePath, symbol) {
    const source = fs.readFileSync(filePath, 'utf8');
    const sandbox = {};
    vm.createContext(sandbox);
    vm.runInContext(source, sandbox, { filename: filePath });
    return sandbox[symbol];
}

function parseArgs(argv) {
    const result = {
        limit: 0,
        start: 0,
        cliPath: path.resolve('build/native/native/ps_cli'),
        keepTemps: false,
    };
    const args = argv.slice(2);
    for (let index = 0; index < args.length; ++index) {
        const arg = args[index];
        if (arg === '--limit' && index + 1 < args.length) {
            result.limit = Number.parseInt(args[++index], 10);
        } else if (arg === '--start' && index + 1 < args.length) {
            result.start = Number.parseInt(args[++index], 10);
        } else if (arg === '--cli' && index + 1 < args.length) {
            result.cliPath = path.resolve(args[++index]);
        } else if (arg === '--keep-temps') {
            result.keepTemps = true;
        } else {
            throw new Error(`Unexpected argument: ${arg}`);
        }
    }
    return result;
}

function run(command, args, options = {}) {
    return spawnSync(command, args, {
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024,
        ...options,
    });
}

function main() {
    const options = parseArgs(process.argv);
    const corpus = loadJsArray(path.join('src', 'tests', 'resources', 'testdata.js'), 'testdata');
    const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-parser-corpus-'));
    const start = Math.max(0, options.start);
    const end = options.limit > 0 ? Math.min(corpus.length, start + options.limit) : corpus.length;

    let checked = 0;
    let passed = 0;
    let failed = 0;

    try {
        for (let index = start; index < end; ++index) {
            const fixture = corpus[index];
            const name = fixture[0];
            const source = fixture[1][0];
            const fixtureBase = `fixture-${index}`;
            const sourcePath = path.join(tempRoot, `${fixtureBase}.txt`);
            const jsPath = path.join(tempRoot, `${fixtureBase}.js.json`);
            const cppPath = path.join(tempRoot, `${fixtureBase}.cpp.json`);

            fs.writeFileSync(sourcePath, source, 'utf8');

            const jsRun = run('node', ['src/tests/export_ir_json.js', sourcePath, jsPath, '--snapshot-phase', 'parser']);
            if (jsRun.status !== 0) {
                ++failed;
                ++checked;
                console.log(`parser_corpus index=${index} outcome=js_error name=${name}`);
                console.log(jsRun.stderr || jsRun.stdout);
                continue;
            }

            const cppRun = run(options.cliPath, ['compile-source', sourcePath, '--emit-parser-state']);
            if (cppRun.status !== 0) {
                ++failed;
                ++checked;
                console.log(`parser_corpus index=${index} outcome=cpp_error name=${name}`);
                console.log(cppRun.stderr || cppRun.stdout);
                continue;
            }
            fs.writeFileSync(cppPath, cppRun.stdout, 'utf8');

            const diffRun = run('diff', ['-u', jsPath, cppPath]);
            ++checked;
            if (diffRun.status === 0) {
                ++passed;
            } else {
                ++failed;
                console.log(`parser_corpus index=${index} outcome=diff_failed name=${name}`);
                console.log(diffRun.stdout);
                break;
            }
        }
    } finally {
        if (!options.keepTemps) {
            fs.rmSync(tempRoot, { recursive: true, force: true });
        } else {
            console.error(`kept_temps=${tempRoot}`);
        }
    }

    console.log(`parser_corpus_checked=${checked} parser_corpus_passed=${passed} parser_corpus_failed=${failed}`);
    process.exit(failed === 0 ? 0 : 1);
}

main();
