#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const vm = require('vm');

function usage() {
  return [
    'Usage: run_rule_plan_parity.js <src/tests/resources/testdata.js> [--cli path] [--artifacts-dir path]',
    '',
    'Compares JS and native game.rule_plan_v1 from emitted IR JSON for unique sources.',
    'Cases are skipped only if the JS exporter cannot emit IR or native IR compilation fails.',
  ].join('\n');
}

function parseArgs(argv) {
  const out = {
    testdataPath: null,
    cliPath: path.resolve('build/native/puzzlescript_cpp'),
    artifactsDir: path.resolve('build/native/rule_plan_parity'),
    progressEvery: 25,
    keepTemps: false,
  };
  const args = argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--cli' && i + 1 < args.length) out.cliPath = path.resolve(args[++i]);
    else if (a === '--artifacts-dir' && i + 1 < args.length) out.artifactsDir = path.resolve(args[++i]);
    else if (a === '--progress-every' && i + 1 < args.length) out.progressEvery = Number.parseInt(args[++i], 10);
    else if (a === '--keep-temps') out.keepTemps = true;
    else if (a === '--help' || a === '-h') out.help = true;
    else if (!out.testdataPath) out.testdataPath = path.resolve(a);
    else throw new Error(`Unexpected argument: ${a}\n\n${usage()}`);
  }
  if (out.help) return out;
  if (!out.testdataPath) throw new Error(usage());
  return out;
}

function loadTestdataByArrayLiteral(filePath) {
  const text = fs.readFileSync(filePath, 'utf8');
  const start = text.indexOf('[');
  const end = text.lastIndexOf('];');
  if (start < 0 || end < start) {
    throw new Error(`Could not locate testdata array literal in ${filePath}`);
  }
  const arrayLiteral = text.slice(start, end + 1);
  const value = vm.runInNewContext(arrayLiteral, {}, { filename: filePath, displayErrors: true });
  if (!Array.isArray(value)) {
    throw new Error(`Expected array literal in ${filePath}`);
  }
  return value;
}

function uniqueSourceCases(testdata) {
  const seen = new Map();
  const cases = [];
  for (let i = 0; i < testdata.length; i++) {
    const entry = testdata[i];
    const name = Array.isArray(entry) ? String(entry[0]) : `<case ${i}>`;
    const payload = Array.isArray(entry) ? entry[1] : null;
    const source = Array.isArray(payload) ? payload[0] : null;
    if (typeof source !== 'string') {
      throw new Error(`Unexpected testdata shape at index ${i} (${name})`);
    }
    if (seen.has(source)) continue;
    seen.set(source, i);
    cases.push({ index: i, name, source });
  }
  return cases;
}

function runCommand(command, args, options = {}) {
  return spawnSync(command, args, {
    encoding: 'utf8',
    maxBuffer: 256 * 1024 * 1024,
    ...options,
  });
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function stableStringify(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(',')}}`;
}

function cloneWithoutRuleIndex(rule) {
  if (!rule || typeof rule !== 'object' || Array.isArray(rule)) return rule;
  const out = {};
  for (const key of Object.keys(rule)) {
    if (key === 'rule_index') continue;
    out[key] = rule[key];
  }
  return out;
}

function canonicalizeRulePlan(plan) {
  const cloned = JSON.parse(JSON.stringify(plan));
  for (const bucketName of ['rules', 'late_rules']) {
    if (!Array.isArray(cloned[bucketName])) continue;
    for (const group of cloned[bucketName]) {
      if (!Array.isArray(group)) continue;
      group.sort((a, b) => {
        const ka = stableStringify(cloneWithoutRuleIndex(a));
        const kb = stableStringify(cloneWithoutRuleIndex(b));
        return ka < kb ? -1 : (ka > kb ? 1 : 0);
      });
      group.forEach((rule, index) => {
        rule.rule_index = index;
      });
    }
  }
  return cloned;
}

function hashText(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

function firstDiff(a, b, pathParts = ['game', 'rule_plan_v1']) {
  if (Object.is(a, b)) return null;
  const aObj = a !== null && typeof a === 'object';
  const bObj = b !== null && typeof b === 'object';
  if (!aObj || !bObj) return { path: pathParts.join('.'), js: a, native: b };
  if (Array.isArray(a) !== Array.isArray(b)) {
    return { path: pathParts.join('.'), jsType: Array.isArray(a) ? 'array' : 'object', nativeType: Array.isArray(b) ? 'array' : 'object' };
  }
  if (Array.isArray(a)) {
    const min = Math.min(a.length, b.length);
    for (let i = 0; i < min; i++) {
      const diff = firstDiff(a[i], b[i], pathParts.concat(`[${i}]`));
      if (diff) return diff;
    }
    if (a.length !== b.length) return { path: pathParts.join('.'), jsLength: a.length, nativeLength: b.length };
    return null;
  }

  const aKeys = Object.keys(a).sort();
  const bKeys = Object.keys(b).sort();
  const keyCount = Math.max(aKeys.length, bKeys.length);
  for (let i = 0; i < keyCount; i++) {
    if (aKeys[i] !== bKeys[i]) {
      return {
        path: pathParts.join('.'),
        jsOnly: aKeys.filter((k) => !Object.prototype.hasOwnProperty.call(b, k)).slice(0, 10),
        nativeOnly: bKeys.filter((k) => !Object.prototype.hasOwnProperty.call(a, k)).slice(0, 10),
      };
    }
  }
  for (const key of aKeys) {
    const diff = firstDiff(a[key], b[key], pathParts.concat(key));
    if (diff) return diff;
  }
  return null;
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function rulePlanFrom(ir, side, caseInfo, artifactDir) {
  if (!ir || typeof ir !== 'object' || !ir.game || typeof ir.game !== 'object') {
    throw new Error(`${side} IR is missing game object for case ${caseInfo.index} (${caseInfo.name}); artifacts: ${artifactDir}`);
  }
  if (!Object.prototype.hasOwnProperty.call(ir.game, 'rule_plan_v1')) {
    throw new Error(`${side} IR is missing game.rule_plan_v1 for case ${caseInfo.index} (${caseInfo.name}); artifacts: ${artifactDir}`);
  }
  return ir.game.rule_plan_v1;
}

function commandText(command, args) {
  return [command].concat(args).join(' ');
}

function compileJsIr(caseInfo, sourcePath, jsIrPath, repoRoot) {
  const args = ['src/tests/js_oracle/export_ir_json.js', sourcePath, jsIrPath];
  return runCommand(process.execPath, args, { cwd: repoRoot, timeout: 120000 });
}

function compileNativeIr(cliPath, sourcePath) {
  const args = ['compile', sourcePath, '--emit-ir-json'];
  const res = runCommand(cliPath, args, { timeout: 120000 });
  res.args = args;
  return res;
}

function writeFailureArtifacts(artifactDir, source, jsIr, nativeIr, jsPlan, nativePlan, extra = {}) {
  ensureDir(artifactDir);
  fs.writeFileSync(path.join(artifactDir, 'source.ps'), `${source}\n`, 'utf8');
  if (jsIr !== undefined) writeJson(path.join(artifactDir, 'js_ir.json'), jsIr);
  if (nativeIr !== undefined) writeJson(path.join(artifactDir, 'native_ir.json'), nativeIr);
  if (jsPlan !== undefined) writeJson(path.join(artifactDir, 'js_rule_plan_v1.json'), jsPlan);
  if (nativePlan !== undefined) writeJson(path.join(artifactDir, 'native_rule_plan_v1.json'), nativePlan);
  writeJson(path.join(artifactDir, 'summary.json'), extra);
}

function main() {
  const startedAt = Date.now();
  const opts = parseArgs(process.argv);
  if (opts.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }

  const repoRoot = path.resolve(__dirname, '..', '..');
  const testdata = loadTestdataByArrayLiteral(opts.testdataPath);
  const cases = uniqueSourceCases(testdata);
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'puzzlescript-rule-plan-parity-'));
  const srcDir = path.join(tmpRoot, 'src');
  const jsDir = path.join(tmpRoot, 'js-ir');
  ensureDir(srcDir);
  ensureDir(jsDir);
  ensureDir(opts.artifactsDir);

  let compared = 0;
  let skippedIrFailures = 0;
  let preserveTmp = false;

  try {
    for (let i = 0; i < cases.length; i++) {
      const caseInfo = cases[i];
      const caseSlug = `case-${String(caseInfo.index).padStart(4, '0')}`;
      const sourcePath = path.join(srcDir, `${caseSlug}.ps`);
      const jsIrPath = path.join(jsDir, `${caseSlug}.json`);
      fs.writeFileSync(sourcePath, `${caseInfo.source}\n`, 'utf8');

      const jsRes = compileJsIr(caseInfo, sourcePath, jsIrPath, repoRoot);
      if (jsRes.error) throw new Error(`JS exporter failed to run for case ${caseInfo.index} (${caseInfo.name}): ${jsRes.error.message}`);
      if (jsRes.status !== 0) {
        skippedIrFailures++;
        process.stderr.write(`skip_js_compile_failure index=${caseInfo.index} name=${JSON.stringify(caseInfo.name)} status=${jsRes.status}\n`);
        continue;
      }

      const jsIr = readJson(jsIrPath);
      const nativeRes = compileNativeIr(opts.cliPath, sourcePath);
      if (nativeRes.error) throw new Error(`Native CLI failed to run for case ${caseInfo.index} (${caseInfo.name}): ${nativeRes.error.message}`);
      if (nativeRes.status !== 0) {
        const stderr = nativeRes.stderr || '';
        const stderrLower = stderr.toLowerCase();
        if (stderr.includes('--emit-ir-json') || stderrLower.includes('unknown') || stderrLower.includes('requires')) {
          preserveTmp = true;
          throw new Error(
            `Native IR compile command failed for case ${caseInfo.index} (${caseInfo.name}). ` +
            `Expected command surface: ${commandText(opts.cliPath, nativeRes.args)}\n\nstderr:\n${stderr}\nstdout:\n${nativeRes.stdout}`
          );
        }
        skippedIrFailures++;
        process.stderr.write(`skip_native_compile_failure index=${caseInfo.index} name=${JSON.stringify(caseInfo.name)} status=${nativeRes.status}\n`);
        continue;
      }

      let nativeIr;
      try {
        nativeIr = JSON.parse(nativeRes.stdout);
      } catch (e) {
        throw new Error(`Native IR JSON parse failed for case ${caseInfo.index} (${caseInfo.name}): ${e.message}\nstdout:\n${nativeRes.stdout.slice(0, 4000)}`);
      }

      const artifactDir = path.join(opts.artifactsDir, caseSlug);
      let jsPlan;
      let nativePlan;
      try {
        jsPlan = rulePlanFrom(jsIr, 'JS', caseInfo, artifactDir);
        nativePlan = rulePlanFrom(nativeIr, 'Native', caseInfo, artifactDir);
      } catch (e) {
        preserveTmp = true;
        writeFailureArtifacts(artifactDir, caseInfo.source, jsIr, nativeIr, jsPlan, nativePlan, {
          index: caseInfo.index,
          name: caseInfo.name,
          error: e.message,
          source_path: sourcePath,
        });
        throw e;
      }

      const jsCanonicalPlan = canonicalizeRulePlan(jsPlan);
      const nativeCanonicalPlan = canonicalizeRulePlan(nativePlan);
      const jsCanonicalStable = stableStringify(jsCanonicalPlan);
      const nativeCanonicalStable = stableStringify(nativeCanonicalPlan);
      if (jsCanonicalStable !== nativeCanonicalStable) {
        const diff = firstDiff(jsCanonicalPlan, nativeCanonicalPlan);
        const jsHash = hashText(jsCanonicalStable);
        const nativeHash = hashText(nativeCanonicalStable);
        preserveTmp = true;
        writeFailureArtifacts(artifactDir, caseInfo.source, jsIr, nativeIr, jsCanonicalPlan, nativeCanonicalPlan, {
          index: caseInfo.index,
          name: caseInfo.name,
          source_path: sourcePath,
          js_hash: jsHash,
          native_hash: nativeHash,
          first_diff: diff,
        });
        process.stderr.write(`rule_plan_v1 mismatch index=${caseInfo.index} name=${JSON.stringify(caseInfo.name)}\n`);
        process.stderr.write(`source_path=${sourcePath}\n`);
        process.stderr.write(`artifacts=${artifactDir}\n`);
        process.stderr.write(`js_sha256=${jsHash}\n`);
        process.stderr.write(`native_sha256=${nativeHash}\n`);
        if (diff) process.stderr.write(`first_diff=${stableStringify(diff)}\n`);
        process.exitCode = 1;
        return;
      }

      compared++;
      if (opts.progressEvery > 0 && compared > 0 && (compared % opts.progressEvery) === 0) {
        process.stderr.write(`progress compared=${compared}/${cases.length} skipped_ir_failures=${skippedIrFailures}\n`);
      }
    }

    const elapsedMs = Date.now() - startedAt;
    process.stdout.write(
      `rule_plan_parity passed compared=${compared} unique_sources=${cases.length} ` +
      `skipped_ir_failures=${skippedIrFailures} elapsed_ms=${elapsedMs}\n`
    );
  } finally {
    if (!opts.keepTemps && !preserveTmp) {
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    } else {
      process.stderr.write(`kept_temp_dir=${tmpRoot}\n`);
    }
  }
}

try {
  main();
} catch (e) {
  process.stderr.write(`${e.stack || e.message}\n`);
  process.exit(1);
}
