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
    testdataPath: null,
    cliPath: path.resolve('build/native/puzzlescript_cpp'),
    progressEvery: 25,
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
    } else if (out.testdataPath === null) {
      out.testdataPath = path.resolve(a);
    } else {
      throw new Error(`Unexpected argument: ${a}`);
    }
  }
  if (!out.testdataPath) {
    throw new Error('Usage: run_simulation_tests_cpp_direct.js <path/to/testdata.js> [--cli path]');
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
    maxBuffer: 128 * 1024 * 1024,
    ...options,
  });
}

function ensureDir(p) {
  fs.mkdirSync(p, { recursive: true });
}

function main() {
  const startedAt = Date.now();
  const opts = parseArgs(process.argv);
  const testdata = loadVarFromJsFile(opts.testdataPath, 'testdata');

  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'puzzlescript-cpp-testdata-'));
  const srcDir = path.join(tmpRoot, 'src');
  ensureDir(srcDir);

  let passed = 0;
  let failed = 0;
  for (let i = 0; i < testdata.length; i++) {
    const entry = testdata[i];
    const name = entry[0];
    const payload = entry[1];
    const source = payload[0];
    const inputs = payload[1];
    const expectedSerialized = payload[2];
    const targetLevel = payload.length >= 4 && payload[3] !== undefined ? payload[3] : 0;
    const seed = payload.length >= 5 && payload[4] !== undefined ? payload[4] : null;

    const sourcePath = path.join(srcDir, `case-${String(i).padStart(4, '0')}.txt`);
    fs.writeFileSync(sourcePath, `${source}\n`, 'utf8');

    const args = ['run', sourcePath, '--headless', '--native-compile', '--final-only', '--json', '--level', String(targetLevel)];
    if (seed !== null) {
      args.push('--seed', String(seed));
    }
    args.push('--inputs-json', JSON.stringify(inputs));

    const res = runCommand(opts.cliPath, args, { timeout: 120000 });
    if (res.error) {
      failed++;
      process.stderr.write(`${name}: failed to run: ${res.error.message}\n`);
      continue;
    }
    if (res.status !== 0) {
      failed++;
      process.stderr.write(`${name}: puzzlescript_cpp exited ${res.status}\n`);
      if (res.stderr) process.stderr.write(res.stderr);
      continue;
    }

    let parsed;
    try {
      parsed = JSON.parse(res.stdout.trim());
    } catch (e) {
      failed++;
      process.stderr.write(`${name}: could not parse runner JSON output\n`);
      if (res.stdout) process.stderr.write(`stdout:\n${res.stdout}\n`);
      if (res.stderr) process.stderr.write(`stderr:\n${res.stderr}\n`);
      continue;
    }

    const actualSerialized = parsed.serialized_level ?? '';
    if (actualSerialized !== expectedSerialized) {
      failed++;
      process.stderr.write(`${name}: final serialized level mismatch\n`);
      process.stderr.write(`expected:\n${expectedSerialized}\n`);
      process.stderr.write(`actual:\n${actualSerialized}\n`);
      continue;
    }

    passed++;
    if (opts.progressEvery > 0 && ((i + 1) % opts.progressEvery) === 0) {
      process.stderr.write(`progress cases=${i + 1}/${testdata.length} passed=${passed} failed=${failed}\n`);
    }
  }

  const elapsedMs = Date.now() - startedAt;
  process.stdout.write(`cpp_simulation_tests_direct passed=${passed} failed=${failed} total=${testdata.length} elapsed_ms=${elapsedMs}\n`);

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

