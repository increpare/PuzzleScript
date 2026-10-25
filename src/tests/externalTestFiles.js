'use strict';

// Keeps testdata.js and errormessage_testdata.js small enough to browse.
//
// Big game sources and end states live in test_scripts/ and test_endstates/
// as .txt files. A test refers to one with {file:"test_scripts/name.txt"} in
// place of the string. Pasted-in tests still contain the full strings; the
// audit here moves any that are too long out to files and rewrites the data
// file to point at them.
//
// It also writes resources/external_testdata.js, a generated copy of every
// referenced file, so tests.html still works when opened from disk (browsers
// won't fetch local .txt files from a file:// page).

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const crypto = require('crypto');

const testsDir = __dirname;

const SCRIPT_LIMIT = 800;
const ENDSTATE_LIMIT = 400;

const DATA_FILES = [
    { file: 'resources/testdata.js', variable: 'testdata', endstateSlot: 2 },
    { file: 'resources/errormessage_testdata.js', variable: 'errormessage_testdata', endstateSlot: null },
];

const BUNDLE_FILE = 'resources/external_testdata.js';

function isFileRef(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value) &&
        typeof value.file === 'string';
}

function readTestFile(relPath) {
    return fs.readFileSync(path.join(testsDir, relPath), 'utf8');
}

// Replaces every {file:...} in each test's data array with the file's text.
// Returns the list of referenced paths that couldn't be read.
function resolveRefs(data, read) {
    const missing = [];
    for (const test of data) {
        const fields = test[1];
        for (let i = 0; i < fields.length; i++) {
            if (!isFileRef(fields[i])) continue;
            const text = read(fields[i].file);
            if (text === undefined) {
                missing.push(fields[i].file);
                fields[i] = '';
            } else {
                fields[i] = text;
            }
        }
    }
    return missing;
}

function readOrUndefined(relPath) {
    try {
        return readTestFile(relPath);
    } catch (e) {
        return undefined;
    }
}

function evalDataFile(source, variable) {
    const context = {};
    vm.runInNewContext(source, context);
    return context[variable];
}

// Finds every quoted string literal in a JS data file, skipping comments.
// Returns [{start, end, value}] with end exclusive.
function findStringLiterals(source) {
    const literals = [];
    let i = 0;
    const n = source.length;
    while (i < n) {
        const c = source[i];
        if (c === '/' && source[i + 1] === '/') {
            const nl = source.indexOf('\n', i);
            i = nl === -1 ? n : nl;
        } else if (c === '/' && source[i + 1] === '*') {
            const close = source.indexOf('*/', i + 2);
            i = close === -1 ? n : close + 2;
        } else if (c === '"' || c === "'" || c === '`') {
            const start = i;
            i++;
            while (i < n && source[i] !== c) {
                if (source[i] === '\\') i++;
                i++;
            }
            i++;
            const literal = source.slice(start, i);
            let value = null;
            if (c !== '`' || !literal.includes('${')) {
                try {
                    value = c === '"' ? JSON.parse(literal) : vm.runInNewContext(literal);
                } catch (e) {
                    value = vm.runInNewContext(literal);
                }
            }
            literals.push({ start, end: i, value });
        } else {
            i++;
        }
    }
    return literals;
}

function slugify(name) {
    const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 60).replace(/_+$/, '');
    return slug || 'test';
}

function hashText(text) {
    return crypto.createHash('sha1').update(text, 'utf8').digest('hex');
}

// Files already in a folder, indexed by content so identical text is shared.
function indexFolder(folder) {
    const dir = path.join(testsDir, folder);
    const byHash = new Map();
    const names = new Set();
    fs.mkdirSync(dir, { recursive: true });
    for (const name of fs.readdirSync(dir)) {
        if (!name.endsWith('.txt')) continue;
        names.add(name);
        byHash.set(hashText(fs.readFileSync(path.join(dir, name), 'utf8')), name);
    }
    return { folder, byHash, names };
}

// Returns the path for this text, writing a new file if no existing one matches.
function fileFor(index, text, testName) {
    const hash = hashText(text);
    if (index.byHash.has(hash)) {
        return index.folder + '/' + index.byHash.get(hash);
    }
    const slug = slugify(testName);
    let name = slug + '.txt';
    for (let k = 2; index.names.has(name); k++) {
        name = slug + '_' + k + '.txt';
    }
    fs.writeFileSync(path.join(testsDir, index.folder, name), text, 'utf8');
    index.names.add(name);
    index.byHash.set(hash, name);
    return index.folder + '/' + name;
}

function writeIfChanged(absPath, text) {
    if (fs.existsSync(absPath) && fs.readFileSync(absPath, 'utf8') === text) return false;
    const tmp = absPath + '.tmp' + process.pid;
    fs.writeFileSync(tmp, text, 'utf8');
    fs.renameSync(tmp, absPath);
    return true;
}

// Moves over-long strings in one data file out to .txt files.
// Returns a list of messages describing what it did.
function externalizeDataFile(spec, scripts, endstates) {
    const absPath = path.join(testsDir, spec.file);
    const source = fs.readFileSync(absPath, 'utf8');
    const original = evalDataFile(source, spec.variable);

    // Decide which strings move, keyed by their text.
    const replacements = new Map();
    let moved = 0;
    for (const test of original) {
        const fields = test[1];
        const name = test[0];
        if (typeof fields[0] === 'string' && fields[0].length > SCRIPT_LIMIT && !replacements.has(fields[0])) {
            replacements.set(fields[0], fileFor(scripts, fields[0], name));
        }
        const slot = spec.endstateSlot;
        if (slot !== null && typeof fields[slot] === 'string' && fields[slot].length > ENDSTATE_LIMIT &&
            !replacements.has(fields[slot])) {
            replacements.set(fields[slot], fileFor(endstates, fields[slot], name));
        }
    }
    if (replacements.size === 0) return [];

    let rewritten = '';
    let last = 0;
    for (const lit of findStringLiterals(source)) {
        if (lit.value === null || !replacements.has(lit.value)) continue;
        rewritten += source.slice(last, lit.start) + '{file:' + JSON.stringify(replacements.get(lit.value)) + '}';
        last = lit.end;
        moved++;
    }
    rewritten += source.slice(last);

    // Only save if the rewritten file loads back to exactly the same tests.
    const check = evalDataFile(rewritten, spec.variable);
    const missing = resolveRefs(check, readOrUndefined);
    if (missing.length > 0 || JSON.stringify(check) !== JSON.stringify(original)) {
        throw new Error(`Externalizing ${spec.file} would change its tests; left it unchanged.`);
    }

    // Don't clobber edits made to the data file while this ran.
    if (fs.readFileSync(absPath, 'utf8') !== source) {
        return [`${spec.file} changed during the audit; skipped externalizing it.`];
    }
    writeIfChanged(absPath, rewritten);
    return [`Moved ${moved} long string(s) out of ${spec.file} into test_scripts/ and test_endstates/.`];
}

function collectRefs(data, into) {
    for (const test of data) {
        for (const field of test[1]) {
            if (isFileRef(field)) into.add(field.file);
        }
    }
}

// Runs the audit, rebuilds the browser bundle, and returns the loaded tests.
// Throws if a referenced file is missing.
function prepareTestData(options) {
    const messages = [];
    const scripts = indexFolder('test_scripts');
    const endstates = indexFolder('test_endstates');
    if (!options || options.externalize !== false) {
        for (const spec of DATA_FILES) {
            messages.push(...externalizeDataFile(spec, scripts, endstates));
        }
    }

    const refs = new Set();
    const sources = {};
    for (const spec of DATA_FILES) {
        const source = fs.readFileSync(path.join(testsDir, spec.file), 'utf8');
        sources[spec.variable] = source;
        collectRefs(evalDataFile(source, spec.variable), refs);
    }

    const bundle = {};
    const missing = [];
    for (const ref of [...refs].sort()) {
        const text = readOrUndefined(ref);
        if (text === undefined) missing.push(ref);
        else bundle[ref] = text;
    }
    if (missing.length > 0) {
        throw new Error('Missing external test files:\n  ' + missing.join('\n  '));
    }

    const bundleText = '// Generated by run_tests_node.js from test_scripts/ and test_endstates/. Do not edit.\n' +
        'var external_testdata = ' + JSON.stringify(bundle, null, 0).replace(/,"/g, ',\n"') + ';\n';
    writeIfChanged(path.join(testsDir, BUNDLE_FILE), bundleText);

    for (const index of [scripts, endstates]) {
        const unused = [...index.names].filter(name => !refs.has(index.folder + '/' + name));
        if (unused.length > 0) {
            messages.push(`${unused.length} file(s) in ${index.folder}/ aren't used by any test: ${unused.join(', ')}`);
        }
    }

    return { sources, bundle, messages };
}

module.exports = { prepareTestData, resolveRefs, SCRIPT_LIMIT, ENDSTATE_LIMIT };
