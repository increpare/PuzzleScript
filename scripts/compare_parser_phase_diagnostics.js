#!/usr/bin/env node
'use strict';

/**
 * Compare two diagnostic stream files (one JSON-string per line, UTF-8)
 * after canonicalization per docs/superpowers/specs/2026-04-22-cpp-frontend-design.md §5.3.
 */

const fs = require('fs');

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

function loadStream(path) {
    const text = fs.readFileSync(path, 'utf8');
    const rows = [];
    for (const line of text.split('\n')) {
        const trimmed = line.trim();
        if (trimmed.length === 0) {
            continue;
        }
        rows.push(canonicalizeDiagnosticText(JSON.parse(trimmed)));
    }
    return rows;
}

function main() {
    const referencePath = process.argv[2];
    const nativePath = process.argv[3];
    if (!referencePath || !nativePath) {
        console.error(
            'Usage: node scripts/compare_parser_phase_diagnostics.js <reference.ndjson> <native.ndjson>',
        );
        console.error(
            '(reference = parser-diagnostics from export_ir_json.js; native = ps_cli compile-source --emit-diagnostics)',
        );
        process.exit(2);
    }
    const referenceRows = loadStream(referencePath);
    const nativeRows = loadStream(nativePath);
    if (referenceRows.length !== nativeRows.length) {
        console.error(
            `diagnostic_count_mismatch reference=${referenceRows.length} native=${nativeRows.length}`,
        );
        process.exit(1);
    }
    for (let i = 0; i < referenceRows.length; ++i) {
        if (referenceRows[i] !== nativeRows[i]) {
            console.error(`diagnostic_mismatch index=${i}`);
            console.error('--- reference (JS parser export)');
            console.error(referenceRows[i]);
            console.error('--- native (C++ frontend)');
            console.error(nativeRows[i]);
            process.exit(1);
        }
    }
}

main();
