#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const { spawnSync } = require('child_process');

function parseArgs(argv) {
  const args = argv.slice(2);
  const out = {
    testdataPath: path.resolve('src/tests/resources/testdata.js'),
    cliPath: path.resolve('build/native/puzzlescript_cpp'),
    startIndex: 0,
    seedOverride: null,
    onlyName: null,
    keepTemps: false,
  };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--testdata' && i + 1 < args.length) out.testdataPath = path.resolve(args[++i]);
    else if (a === '--cli' && i + 1 < args.length) out.cliPath = path.resolve(args[++i]);
    else if (a === '--start' && i + 1 < args.length) out.startIndex = Number.parseInt(args[++i], 10);
    else if (a === '--seed' && i + 1 < args.length) out.seedOverride = String(args[++i]);
    else if (a === '--name' && i + 1 < args.length) out.onlyName = String(args[++i]);
    else if (a === '--keep-temps') out.keepTemps = true;
    else throw new Error(`Unexpected argument: ${a}`);
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

function run(command, args, options = {}) {
  const res = spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    ...options,
  });
  if (res.error) throw res.error;
  return res;
}

function main() {
  const opts = parseArgs(process.argv);
  const testdata = loadVarFromJsFile(opts.testdataPath, 'testdata');

  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'puzzlescript-firstfail-'));
  const srcDir = path.join(tmpRoot, 'src');
  fs.mkdirSync(srcDir, { recursive: true });

  try {
    for (let i = Math.max(0, opts.startIndex); i < testdata.length; i++) {
      const entry = testdata[i];
      const name = entry[0];
      if (opts.onlyName && name !== opts.onlyName) continue;

      const payload = entry[1];
      const source = payload[0];
      const inputs = payload[1];
      const expectedSerialized = payload[2];
      const targetLevel = payload.length >= 4 && payload[3] !== undefined ? payload[3] : 0;
      const seedFromCase = payload.length >= 5 && payload[4] !== undefined ? String(payload[4]) : null;
      const seed = opts.seedOverride !== null ? opts.seedOverride : seedFromCase;

      const sourcePath = path.join(srcDir, `case-${String(i).padStart(4, '0')}.txt`);
      fs.writeFileSync(sourcePath, `${source}\n`, 'utf8');

      const args = ['run', sourcePath, '--headless', '--native-compile', '--final-only', '--json', '--level', String(targetLevel)];
      if (seed !== null) args.push('--seed', seed);
      args.push('--inputs-json', JSON.stringify(inputs));

      const res = run(opts.cliPath, args, { timeout: 120000 });
      if (res.status !== 0) {
        process.stderr.write(`${name}: puzzlescript_cpp exited ${res.status}\n`);
        if (res.stderr) process.stderr.write(res.stderr);
        continue;
      }
      let parsed;
      try {
        parsed = JSON.parse(res.stdout.trim());
      } catch {
        process.stderr.write(`${name}: could not parse runner JSON output\n`);
        continue;
      }
      const actualSerialized = parsed.serialized_level ?? '';
      if (actualSerialized === expectedSerialized) continue;

      process.stderr.write(`${name}: first failing case at index=${i}\n`);
      process.stderr.write(`case_file=${sourcePath}\n`);
      process.stderr.write(`level=${targetLevel} seed=${seed ?? '<none>'}\n`);

      const diffScript = path.resolve('src/tests/diff_ir_json.js');
      const diffArgs = [diffScript, sourcePath, '--cli', opts.cliPath, '--level', String(targetLevel)];
      if (seed !== null) diffArgs.push('--seed', seed);
      const diffRes = run(process.execPath, diffArgs, { timeout: 120000 });
      process.stdout.write(diffRes.stdout || '');
      process.stderr.write(diffRes.stderr || '');
      process.exitCode = 1;
      return;
    }

    process.stdout.write('No failing cases found.\n');
  } finally {
    if (!opts.keepTemps) {
      try { fs.rmSync(tmpRoot, { recursive: true, force: true }); } catch {}
    } else {
      process.stderr.write(`kept_temp_dir=${tmpRoot}\n`);
    }
  }
}

main();

