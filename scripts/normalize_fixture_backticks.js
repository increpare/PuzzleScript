#!/usr/bin/env node
'use strict';

/**
 * Replace JS template literals (`...`) with equivalent double-quoted string literals
 * in fixture files (testdata.js, errormessage_testdata.js).
 *
 * Only touches backticks outside of "..." strings, so embedded game sources are safe.
 *
 * Usage:
 *   node scripts/normalize_fixture_backticks.js              # dry-run, both defaults
 *   node scripts/normalize_fixture_backticks.js --write      # write files in place
 *   node scripts/normalize_fixture_backticks.js path/to/a.js path/to/b.js [--write]
 */

const fs = require('fs');
const path = require('path');

const DEFAULT_FILES = [
    path.join('src', 'tests', 'resources', 'testdata.js'),
    path.join('src', 'tests', 'resources', 'errormessage_testdata.js'),
];

function readTemplateLiteral(src, openIdx) {
    if (src[openIdx] !== '`') {
        throw new Error('internal: expected ` at ' + openIdx);
    }
    let j = openIdx + 1;
    const out = [];
    while (j < src.length) {
        const ch = src[j];
        if (ch === '\\') {
            if (j + 1 >= src.length) {
                throw new Error('Unterminated escape before end of file');
            }
            const n = src[j + 1];
            if (n === '`' || n === '\\' || n === '$') {
                out.push(n);
                j += 2;
                continue;
            }
            if (n === 'n') {
                out.push('\n');
                j += 2;
                continue;
            }
            if (n === 'r') {
                out.push('\r');
                j += 2;
                continue;
            }
            if (n === 't') {
                out.push('\t');
                j += 2;
                continue;
            }
            if (n === '\n' || n === '\r') {
                j += 2;
                if (n === '\r' && src[j] === '\n') {
                    j++;
                }
                continue;
            }
            out.push(n);
            j += 2;
            continue;
        }
        if (ch === '`') {
            return { endExclusive: j + 1, value: out.join('') };
        }
        if (ch === '$' && src[j + 1] === '{') {
            throw new Error(
                'Template literal contains ${...} at offset ' + j + '; normalize that entry by hand.',
            );
        }
        out.push(ch);
        j++;
    }
    throw new Error('Unclosed template literal starting at offset ' + openIdx);
}

function replaceBackticksOutsideDoubleQuotes(src) {
    let i = 0;
    let out = '';
    let inDouble = false;
    let escape = false;
    let replaced = 0;

    while (i < src.length) {
        const c = src[i];
        if (inDouble) {
            out += c;
            if (escape) {
                escape = false;
            } else if (c === '\\') {
                escape = true;
            } else if (c === '"') {
                inDouble = false;
            }
            i++;
            continue;
        }

        if (c === '"') {
            inDouble = true;
            out += c;
            i++;
            continue;
        }

        if (c === '`') {
            const { endExclusive, value } = readTemplateLiteral(src, i);
            out += JSON.stringify(value);
            replaced += 1;
            i = endExclusive;
            continue;
        }

        out += c;
        i++;
    }

    return { text: out, replaced };
}

function parseArgs(argv) {
    const write = argv.includes('--write');
    const files = argv.slice(2).filter((a) => a !== '--write');
    return { write, files: files.length > 0 ? files : DEFAULT_FILES };
}

function main() {
    const { write, files } = parseArgs(process.argv);
    let total = 0;
    for (const rel of files) {
        const abs = path.isAbsolute(rel) ? rel : path.join(process.cwd(), rel);
        if (!fs.existsSync(abs)) {
            console.error('missing file: ' + abs);
            process.exit(1);
        }
        const before = fs.readFileSync(abs, 'utf8');
        const { text: after, replaced } = replaceBackticksOutsideDoubleQuotes(before);
        total += replaced;
        console.log(path.relative(process.cwd(), abs) + ': ' + replaced + ' template literal(s)');
        if (replaced === 0) {
            continue;
        }
        if (write) {
            fs.writeFileSync(abs, after, 'utf8');
        }
    }
    if (!write && total > 0) {
        console.error('\nDry-run only. Re-run with --write to apply changes.');
    }
    process.exit(0);
}

main();
