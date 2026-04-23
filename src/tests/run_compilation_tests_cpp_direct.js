#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const vm = require('vm');

function parseArgs(argv) {
  const args = argv.slice(2);
  const out = {
    errdataPath: null,
    cliPath: path.resolve('build/native/puzzlescript_cpp'),
    progressEvery: 50,
    keepTemps: false,
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--cli' && i + 1 < args.length) {
      out.cliPath = path.resolve(args[++i]);
    } else if (a === '--progress-every' && i + 1 < args.length) {
      out.progressEvery = Number.parseInt(args[++i], 10);
    } else if (a === '--keep-temps') {
      out.keepTemps = true;
    } else if (out.errdataPath === null) {
      out.errdataPath = path.resolve(a);
    } else {
      throw new Error(`Unexpected argument: ${a}`);
    }
  }
  if (!out.errdataPath) {
    throw new Error('Usage: run_compilation_tests_cpp_direct.js <path/to/errormessage_testdata.js> [--cli path]');
  }
  return out;
}

function loadVarFromJsFile(filePath, varName) {
  const text = fs.readFileSync(filePath, 'utf8');
  const context = {};
  vm.createContext(context);
  vm.runInContext(text, context, { filename: filePath, displayErrors: true });
  const value = context[varName];
  if (!Array.isArray(value)) {
    throw new Error(`Expected ${varName} to be an array in ${filePath}`);
  }
  return value;
}

function runCommand(command, args, options = {}) {
  return spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 64 * 1024 * 1024,
    ...options,
  });
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function parseDiagnosticsLines(stdout) {
  const lines = stdout.split('\n').filter(Boolean);
  const out = [];
  for (const line of lines) {
    try {
      out.push(JSON.parse(line));
    } catch {
      // If a line isn't JSON, keep it raw so mismatches are visible.
      out.push(line);
    }
  }
  return out;
}

function main() {
  const startedAt = Date.now();
  const opts = parseArgs(process.argv);
  const errdata = loadVarFromJsFile(opts.errdataPath, 'errormessage_testdata');

  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'puzzlescript-cpp-errdata-'));
  const srcDir = path.join(tmpRoot, 'src');
  ensureDir(srcDir);

  let passed = 0;
  let failed = 0;
  for (let i = 0; i < errdata.length; i++) {
    const entry = errdata[i];
    const name = entry[0];
    const payload = entry[1];
    const source = payload[0];
    const expectedErrors = payload[1] || [];
    const expectedCount = payload[2];

    const sourcePath = path.join(srcDir, `case-${String(i).padStart(4, '0')}.txt`);
    fs.writeFileSync(sourcePath, `${source}\n`, 'utf8');

    const args = ['compile', sourcePath, '--diagnostics'];
    const res = runCommand(opts.cliPath, args, { timeout: 120000 });
    if (res.error) {
      failed++;
      process.stderr.write(`${name}: failed to run: ${res.error.message}\n`);
      continue;
    }
    if (res.status !== 0) {
      // compile returns 0 even on errors (it prints diagnostics). Non-zero here is a harness error.
      failed++;
      process.stderr.write(`${name}: puzzlescript_cpp exited ${res.status}\n`);
      if (res.stderr) process.stderr.write(res.stderr);
      continue;
    }

    const actualErrors = parseDiagnosticsLines(res.stdout);
    const actualCount = actualErrors.length;

    const expectedCountNum = Number.isFinite(expectedCount) ? expectedCount : expectedErrors.length;
    const countOk = actualCount === expectedCountNum;
    const messagesOk = JSON.stringify(actualErrors) === JSON.stringify(expectedErrors);
    if (!countOk || !messagesOk) {
      failed++;
      process.stderr.write(`${name}: diagnostics mismatch\n`);
      process.stderr.write(`expected_count=${expectedCountNum} actual_count=${actualCount}\n`);
      if (!messagesOk) {
        process.stderr.write(`expected_messages=${JSON.stringify(expectedErrors)}\n`);
        process.stderr.write(`actual_messages=${JSON.stringify(actualErrors)}\n`);
      }
      continue;
    }

    passed++;
    if (opts.progressEvery > 0 && ((i + 1) % opts.progressEvery) === 0) {
      process.stderr.write(`progress cases=${i + 1}/${errdata.length} passed=${passed} failed=${failed}\n`);
    }
  }

  const elapsedMs = Date.now() - startedAt;
  process.stdout.write(`cpp_compilation_tests_direct passed=${passed} failed=${failed} total=${errdata.length} elapsed_ms=${elapsedMs}\n`);

  if (!opts.keepTemps) {
    try {
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    } catch {
      // ignore
    }
  } else {
    process.stderr.write(`kept_temp_dir=${tmpRoot}\n`);
  }

  process.exit(failed === 0 ? 0 : 1);
}

main();

