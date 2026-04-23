#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

function usage() {
  return [
    'Usage: diff_ir_json.js <game.txt> [--cli path/to/puzzlescript_cpp] [--seed SEED] [--level N]',
    '',
    'Compares JS-exported IR JSON against native-lowered IR JSON.',
    'This is intended for debugging native lowering mismatches without running the full simulation suite.',
  ].join('\n');
}

function parseArgs(argv) {
  const out = {
    sourcePath: null,
    cliPath: path.resolve('build/native/puzzlescript_cpp'),
    seed: null,
    level: 0,
  };
  const args = argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--cli' && i + 1 < args.length) out.cliPath = path.resolve(args[++i]);
    else if (a === '--seed' && i + 1 < args.length) out.seed = String(args[++i]);
    else if (a === '--level' && i + 1 < args.length) out.level = Number.parseInt(args[++i], 10);
    else if (!out.sourcePath) out.sourcePath = path.resolve(a);
    else throw new Error(`Unexpected arg: ${a}\n\n${usage()}`);
  }
  if (!out.sourcePath) throw new Error(usage());
  return out;
}

function run(command, args, opts = {}) {
  const res = spawnSync(command, args, { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024, ...opts });
  if (res.error) throw new Error(res.error.message);
  if (res.status !== 0) {
    const msg = `Command failed (${res.status}): ${command} ${args.join(' ')}`;
    throw new Error(`${msg}\n\nstderr:\n${res.stderr}\n\nstdout:\n${res.stdout}`);
  }
  return res.stdout;
}

function stableStringify(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(',')}}`;
}

function diff(a, b, pathParts, out) {
  if (a === b) return true;
  const aObj = a && typeof a === 'object';
  const bObj = b && typeof b === 'object';
  if (!aObj || !bObj) {
    out.push({ path: pathParts.join('.'), a, b });
    return false;
  }
  if (Array.isArray(a) !== Array.isArray(b)) {
    out.push({ path: pathParts.join('.'), aType: Array.isArray(a) ? 'array' : 'object', bType: Array.isArray(b) ? 'array' : 'object' });
    return false;
  }
  if (Array.isArray(a)) {
    if (a.length !== b.length) {
      // Allow native to include extra trailing levels (some parsers retain an
      // empty sentinel); compare shared prefix for debugging.
      const min = Math.min(a.length, b.length);
      for (let i = 0; i < min; i++) {
        if (!diff(a[i], b[i], pathParts.concat([`[${i}]`]), out)) return false;
      }
      out.push({ path: pathParts.join('.'), aLen: a.length, bLen: b.length });
      return false;
    }
    for (let i = 0; i < a.length; i++) {
      if (!diff(a[i], b[i], pathParts.concat([`[${i}]`]), out)) return false;
    }
    return true;
  }
  const aKeys = Object.keys(a).sort();
  const bKeys = Object.keys(b).sort();
  if (aKeys.join('\n') !== bKeys.join('\n')) {
    // Allow missing optional keys from native emitter; we care about mismatched
    // values of keys that exist in both.
    const aOnly = aKeys.filter((k) => !bKeys.includes(k));
    const bOnly = bKeys.filter((k) => !aKeys.includes(k));
    if (bOnly.length > 0) {
      out.push({ path: pathParts.join('.'), aKeys, bKeys, aOnly, bOnly });
      return false;
    }
    // Continue comparing shared keys only.
    for (const k of bKeys) {
      if (!diff(a[k], b[k], pathParts.concat([k]), out)) return false;
    }
    return true;
  }
  for (const k of aKeys) {
    if (!diff(a[k], b[k], pathParts.concat([k]), out)) return false;
  }
  return true;
}

function main() {
  const opts = parseArgs(process.argv);
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'puzzlescript-ir-diff-'));
  const jsIrPath = path.join(tmpDir, 'js_ir.json');

  const exportArgs = [opts.sourcePath, jsIrPath, '--level', String(opts.level)];
  if (opts.seed !== null) exportArgs.push('--seed', opts.seed);

  // Use the JS oracle exporter (node) for JS IR.
  run(process.execPath, ['src/tests/js_oracle/export_ir_json.js', ...exportArgs], { cwd: path.resolve(__dirname, '..', '..') });

  // Native lowered IR.
  const nativeArgs = ['compile', opts.sourcePath, '--emit-runtime-ir'];
  const nativeText = run(opts.cliPath, nativeArgs);

  const jsObj = JSON.parse(fs.readFileSync(jsIrPath, 'utf8'));
  const nativeObj = JSON.parse(nativeText);

  // The native IR emitter currently does not populate some document metadata
  // fields. Normalize those so we can focus on game-structure diffs.
  if (nativeObj && nativeObj.document && jsObj && jsObj.document) {
    nativeObj.document.input_file = jsObj.document.input_file;
    nativeObj.document.random_seed = jsObj.document.random_seed;
    nativeObj.document.command = jsObj.document.command;
  }

  const first = [];
  diff(jsObj, nativeObj, ['<root>'], first);
  if (first.length === 0) {
    process.stdout.write('IR identical (stable JSON compare)\n');
    return;
  }
  process.stdout.write(`First diff at ${first[0].path}\n`);
  process.stdout.write(`JS: ${stableStringify(first[0].a ?? first[0])}\n`);
  process.stdout.write(`Native: ${stableStringify(first[0].b ?? first[0])}\n`);
  process.exitCode = 1;
}

main();

