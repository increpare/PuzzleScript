#!/usr/bin/env node
'use strict';

/**
 * Emit an NDJSON corpus for ps_cli diagnostics-parity: one JSON object per line with
 * { "index", "name", "source", "reference": [ canonical diagnostic strings ] }.
 *
 * Reference strings match scripts/compare_parser_phase_diagnostics.js after
 * canonicalizeDiagnosticText(HTML) — same contract as export_ir_json parser-diagnostics.
 *
 * Usage:
 *   node scripts/build_parser_corpus_bundle.js errormessage > build/parser_corpus_errormessage.bundle.ndjson
 *   node scripts/build_parser_corpus_bundle.js testdata > build/parser_corpus_testdata.bundle.ndjson
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const repoRoot = path.join(__dirname, '..');
const { loadPuzzleScript } = require(path.join(repoRoot, 'src', 'tests', 'lib', 'puzzlescript_node_env'));
const { collectParserPhaseDiagnostics } = require(path.join(repoRoot, 'src', 'tests', 'lib', 'puzzlescript_parser_snapshot'));

function canonicalizeDiagnosticText(raw) {
    let t = String(raw);
    t = t.replace(/<br\s*\/?>/gi, '\n');
    t = t.replace(/<\/?[a-zA-Z][^>]*>/g, '');
    const lines = t.split('\n').map((line) => line.trim());
    const out = [];
    for (const line of lines) {
        if (line.length === 0 && out.length > 0 && out[out.length - 1] === '') {
            continue;
        }
        out.push(line);
    }
    while (out.length > 0 && out[out.length - 1] === '') {
        out.pop();
    }
    while (out.length > 0 && out[0] === '') {
        out.shift();
    }
    return out.join('\n');
}

function loadJsArray(filePath, symbol) {
    const source = fs.readFileSync(filePath, 'utf8');
    const sandbox = {};
    vm.createContext(sandbox);
    vm.runInContext(source, sandbox, { filename: filePath });
    return sandbox[symbol];
}

function corpusPath(name) {
    if (name === 'testdata') {
        return path.join('src', 'tests', 'resources', 'testdata.js');
    }
    if (name === 'errormessage') {
        return path.join('src', 'tests', 'resources', 'errormessage_testdata.js');
    }
    throw new Error(`Unknown corpus: ${name} (expected testdata or errormessage)`);
}

function corpusSymbol(name) {
    return name === 'errormessage' ? 'errormessage_testdata' : 'testdata';
}

function main() {
    const corpusName = process.argv[2];
    if (!corpusName || corpusName === '--help' || corpusName === '-h') {
        console.error('Usage: node scripts/build_parser_corpus_bundle.js <testdata|errormessage>');
        process.exit(2);
    }

    loadPuzzleScript({ includeTests: true, messageSink: [] });

    const filePath = path.join(repoRoot, corpusPath(corpusName));
    const symbol = corpusSymbol(corpusName);
    const rows = loadJsArray(filePath, symbol);

    for (let index = 0; index < rows.length; ++index) {
        const row = rows[index];
        const name = row[0];
        const source = row[1][0];
        const diagnostics = collectParserPhaseDiagnostics(`${source}\n`);
        const reference = diagnostics.map((html) => canonicalizeDiagnosticText(html));
        process.stdout.write(
            `${JSON.stringify({
                index,
                name,
                source,
                reference,
            })}\n`,
        );
    }
}

main();
