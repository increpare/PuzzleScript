#!/usr/bin/env node
'use strict';

const childProcess = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const RUNTIMES = ['js', 'native_interpreter', 'hybrid', 'compiled'];
const RUNTIME_LABELS = {
  js: 'JS',
  native_interpreter: 'C++ Interpreter',
  hybrid: 'C++ Hybrid',
  compiled: 'C++ Compiled',
};

function parseArgs(argv) {
  const options = {
    outRoot: path.resolve(ROOT, process.env.PERFORMANCE_TESTPAGE_OUT || 'build/performance-testpage'),
    quick: /^true|1|yes$/i.test(process.env.PERFORMANCE_TESTPAGE_QUICK || ''),
    profile: /^true|1|yes$/i.test(process.env.PERFORMANCE_TESTPAGE_PROFILE || ''),
    runs: null,
    solverTimeoutMs: Number.parseInt(process.env.PERFORMANCE_TESTPAGE_SOLVER_TIMEOUT_MS || process.env.SOLVER_FOCUS_TIMEOUT_MS || '500', 10),
    generatorSamples: null,
    generatorRuns: null,
    generatorTopK: Number.parseInt(process.env.PERFORMANCE_TESTPAGE_GENERATOR_TOP_K || '10', 10),
    jobs: process.env.PERFORMANCE_TESTPAGE_JOBS || '1',
  };

  const args = argv.slice(2);
  for (let index = 0; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--out' && index + 1 < args.length) {
      options.outRoot = path.resolve(ROOT, args[++index]);
    } else if (arg === '--quick') {
      options.quick = true;
    } else if (arg === '--profile') {
      options.profile = true;
    } else if (arg === '--runs' && index + 1 < args.length) {
      options.runs = positiveInt(args[++index], '--runs');
    } else if (arg === '--solver-timeout-ms' && index + 1 < args.length) {
      options.solverTimeoutMs = positiveInt(args[++index], '--solver-timeout-ms');
    } else if (arg === '--generator-samples' && index + 1 < args.length) {
      options.generatorSamples = positiveInt(args[++index], '--generator-samples');
    } else if (arg === '--generator-runs' && index + 1 < args.length) {
      options.generatorRuns = positiveInt(args[++index], '--generator-runs');
    } else if (arg === '--jobs' && index + 1 < args.length) {
      options.jobs = args[++index];
    } else if (arg === '--help' || arg === '-h') {
      usage(0);
    } else {
      throw new Error(`Unsupported argument: ${arg}`);
    }
  }

  if (options.runs === null) {
    options.runs = options.quick ? 1 : 3;
  }
  if (options.generatorSamples === null) {
    options.generatorSamples = Number.parseInt(
      process.env.PERFORMANCE_TESTPAGE_GENERATOR_SAMPLES || (options.quick ? '20' : '200'),
      10
    );
  }
  if (options.generatorRuns === null) {
    options.generatorRuns = Number.parseInt(
      process.env.PERFORMANCE_TESTPAGE_GENERATOR_RUNS || String(options.runs),
      10
    );
  }
  if (!Number.isFinite(options.solverTimeoutMs) || options.solverTimeoutMs <= 0) {
    options.solverTimeoutMs = 500;
  }
  return options;
}

function usage(exitCode) {
  const text = [
    'Usage: node scripts/build_performance_testpage.js [options]',
    '',
    'Options:',
    '  --out PATH                 Output root (default: build/performance-testpage)',
    '  --quick                    Use one repeat and smaller generator sample count',
    '  --profile                  Run optional native profiler artifact collection',
    '  --runs N                   Timing repeats for comparable suites',
    '  --solver-timeout-ms N      Native/C++ solver focus timeout',
    '  --generator-samples N      Generator samples per preset',
    '  --generator-runs N         Generator runs per preset',
    '  --jobs N|auto              Comparable benchmark jobs (default: 1)',
  ].join('\n');
  (exitCode === 0 ? process.stdout : process.stderr).write(`${text}\n`);
  process.exit(exitCode);
}

function positiveInt(value, label) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive integer: ${value}`);
  }
  return parsed;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function rmDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
}

function readTextIfExists(filePath) {
  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch {
    return null;
  }
}

function readJsonIfExists(filePath) {
  const text = readTextIfExists(filePath);
  if (text === null) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function writeJson(filePath, value) {
  ensureDir(path.dirname(filePath));
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function sha256File(relativePath) {
  const fullPath = path.resolve(ROOT, relativePath);
  try {
    const hash = crypto.createHash('sha256');
    hash.update(fs.readFileSync(fullPath));
    return hash.digest('hex');
  } catch {
    return null;
  }
}

function shortSha(value) {
  return value ? value.slice(0, 12) : 'unknown';
}

function timestampForPath(date) {
  return date.toISOString().replace(/[-:]/g, '').replace('T', '_').replace(/\..+$/, '');
}

function shellQuote(value) {
  return String(value).replace(/'/g, "'\"'\"'");
}

function commandString(command, args, env = {}) {
  const envPairs = Object.entries(env)
    .filter(([, value]) => value !== undefined && value !== null)
    .map(([key, value]) => `${key}='${shellQuote(value)}'`);
  return [...envPairs, command, ...args.map((arg) => `'${shellQuote(arg)}'`)].join(' ');
}

function runCommand(context, name, command, args, extra = {}) {
  const started = process.hrtime.bigint();
  const env = { ...process.env, ...(extra.env || {}) };
  const display = commandString(command, args, extra.env || {});
  process.stderr.write(`performance_testpage: ${name}\n`);
  const result = childProcess.spawnSync(command, args, {
    cwd: ROOT,
    env,
    encoding: 'utf8',
    maxBuffer: extra.maxBuffer || 1024 * 1024 * 1024,
    timeout: extra.timeoutMs,
  });
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
  const stdout = result.stdout || '';
  const stderr = result.stderr || '';
  const logBase = safeFileName(name);
  fs.writeFileSync(path.join(context.logsDir, `${logBase}.stdout.log`), stdout, 'utf8');
  fs.writeFileSync(path.join(context.logsDir, `${logBase}.stderr.log`), stderr, 'utf8');
  fs.writeFileSync(
    path.join(context.logsDir, `${logBase}.command.txt`),
    `${display}\nexit_status=${result.status === null ? 'null' : result.status}\nelapsed_ms=${elapsedMs.toFixed(1)}\n`,
    'utf8'
  );
  const record = {
    name,
    command: display,
    status: result.status,
    signal: result.signal,
    elapsed_ms: elapsedMs,
    stdout_log: `logs/${logBase}.stdout.log`,
    stderr_log: `logs/${logBase}.stderr.log`,
    command_log: `logs/${logBase}.command.txt`,
  };
  context.commands.push(record);
  if (result.error || result.status !== 0) {
    const message = result.error ? result.error.message : `exit status ${result.status}`;
    context.warnings.push({
      suite: extra.suite || 'general',
      message: `${name} failed: ${message}`,
      log: record.stderr_log,
    });
  }
  return { ...record, stdout, stderr, error: result.error };
}

function safeFileName(name) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'command';
}

function parseKeyValueLine(line) {
  const result = {};
  const re = /([A-Za-z_][A-Za-z0-9_]*)=("[^"\\]*(?:\\.[^"\\]*)*"|[^\s]+)/g;
  let match;
  while ((match = re.exec(line)) !== null) {
    const key = match[1];
    const rawValue = match[2];
    let value = rawValue;
    if (rawValue.startsWith('"')) {
      try {
        value = JSON.parse(rawValue);
      } catch {
        value = rawValue.slice(1, -1);
      }
    } else if (/^-?\d+(?:\.\d+)?$/.test(rawValue)) {
      value = Number(rawValue);
    }
    result[key] = value;
  }
  return result;
}

function linesWithPrefix(text, prefix) {
  return text.split(/\r?\n/).filter((line) => line.startsWith(prefix));
}

function lastKeyValueLine(text, prefix) {
  const lines = linesWithPrefix(text, prefix);
  if (lines.length === 0) {
    return null;
  }
  return parseKeyValueLine(lines[lines.length - 1]);
}

function allKeyValueLines(text, prefix) {
  return linesWithPrefix(text, prefix).map(parseKeyValueLine);
}

function median(values) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (finite.length === 0) {
    return null;
  }
  return finite[Math.floor(finite.length / 2)];
}

function sum(values) {
  return values.filter(Number.isFinite).reduce((acc, value) => acc + value, 0);
}

function ratioVsJs(jsMs, runtimeMs) {
  if (!Number.isFinite(jsMs) || !Number.isFinite(runtimeMs) || runtimeMs <= 0) {
    return null;
  }
  return jsMs / runtimeMs;
}

function targetHasStatus(target, status) {
  return Boolean(target && target.status_counts && target.status_counts[status] > 0);
}

function isComparableSolverTarget(target) {
  return Boolean(
    target &&
    target.median &&
    Number.isFinite(target.median.elapsed_ms) &&
    !targetHasStatus(target, 'timeout')
  );
}

function msFromNs(value) {
  return Number.isFinite(value) ? value / 1e6 : null;
}

function compactNumber(value, digits = 1) {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  if (Math.abs(value) >= 1000000) {
    return `${(value / 1000000).toFixed(digits)}m`;
  }
  if (Math.abs(value) >= 1000) {
    return `${(value / 1000).toFixed(digits)}k`;
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

function formatMs(value) {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)}s`;
  }
  return `${value.toFixed(1)}ms`;
}

function formatTimeout(value) {
  return Number.isFinite(value) ? `${value}ms` : 'no limit';
}

function formatSpeedup(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}x` : 'n/a';
}

function collectMetadata(context) {
  const gitSha = captureTrim('git', ['rev-parse', 'HEAD']);
  const branch = captureTrim('git', ['branch', '--show-current']);
  const dirty = captureTrim('git', ['status', '--porcelain']).length > 0;
  const compiler = captureTrim('cc', ['--version']).split(/\r?\n/)[0] || captureTrim('clang', ['--version']).split(/\r?\n/)[0] || null;
  const cmake = captureTrim('cmake', ['--version']).split(/\r?\n/)[0] || null;
  return {
    generated_at: context.startedAt.toISOString(),
    git_sha: gitSha || null,
    git_short_sha: shortSha(gitSha),
    branch: branch || null,
    dirty,
    platform: `${os.type()} ${os.release()} ${os.arch()}`,
    cpu: (os.cpus()[0] && os.cpus()[0].model) || null,
    cpu_count: os.cpus().length,
    node: process.version,
    compiler,
    cmake,
  };
}

function captureTrim(command, args) {
  try {
    const result = childProcess.spawnSync(command, args, {
      cwd: ROOT,
      encoding: 'utf8',
      maxBuffer: 8 * 1024 * 1024,
    });
    if (result.status !== 0 || result.error) {
      return '';
    }
    return (result.stdout || '').trim();
  } catch {
    return '';
  }
}

function collectInputs(options) {
  return {
    quick: options.quick,
    optional_profile: options.profile,
    runs: options.runs,
    solver_timeout_ms: options.solverTimeoutMs,
    solver_js_timeout_ms: null,
    jobs: options.jobs,
    generator_samples: options.generatorSamples,
    generator_runs: options.generatorRuns,
    corpora: {
      simulation: {
        path: 'src/tests/resources/testdata.js',
        sha256: sha256File('src/tests/resources/testdata.js'),
      },
      diagnostics: {
        path: 'src/tests/resources/errormessage_testdata.js',
        sha256: sha256File('src/tests/resources/errormessage_testdata.js'),
      },
      solver: {
        path: 'src/tests/solver_tests',
        file_count: countFiles('src/tests/solver_tests', '.txt'),
      },
      generator_presets: {
        path: 'src/tests/generator_presets',
        file_count: countFiles('src/tests/generator_presets', '.gen'),
      },
    },
  };
}

function countFiles(relativeRoot, extension) {
  const root = path.resolve(ROOT, relativeRoot);
  let count = 0;
  const walk = (dir) => {
    let entries = [];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full);
      } else if (entry.isFile() && entry.name.toLowerCase().endsWith(extension)) {
        count++;
      }
    }
  };
  walk(root);
  return count;
}

function buildRuntimeSummary(runtime, medianMs, extra = {}) {
  return {
    runtime,
    label: RUNTIME_LABELS[runtime],
    median_ms: Number.isFinite(medianMs) ? medianMs : null,
    speedup_vs_js: null,
    ...extra,
  };
}

function finalizeSpeedups(suite) {
  const jsMs = suite.runtimes && suite.runtimes.js && suite.runtimes.js.median_ms;
  if (!Number.isFinite(jsMs)) {
    return;
  }
  for (const runtime of Object.values(suite.runtimes || {})) {
    runtime.speedup_vs_js = runtime.runtime === 'js' ? 1 : ratioVsJs(jsMs, runtime.median_ms);
  }
}

function parseJsSimulation(json) {
  const breakdown = json.breakdown_avg_ms ? {
    source_compile_ms: json.breakdown_avg_ms.compile_ms,
    process_input_ms: json.breakdown_avg_ms.process_input_ms,
    undo_ms: json.breakdown_avg_ms.undo_ms,
    restart_ms: json.breakdown_avg_ms.restart_ms,
    compile_count: json.breakdown_avg_ms.compile_count,
    process_input_count: json.breakdown_avg_ms.process_input_count,
    undo_count: json.breakdown_avg_ms.undo_count,
    restart_count: json.breakdown_avg_ms.restart_count,
  } : {};
  return buildRuntimeSummary('js', json.median_ms, {
    average_ms: json.average_ms,
    min_ms: json.min_ms,
    max_ms: json.max_ms,
    samples_ms: json.samples_ms || [],
    timing_breakdown_ms: breakdown,
    raw_profile: json,
  });
}

function mergeSimulationInstrumentation(summary, instrumentation) {
  if (!summary || !instrumentation) {
    return summary;
  }
  summary.instrumentation = {
    median_ms: instrumentation.median_ms,
    average_ms: instrumentation.average_ms,
    aggregate_wall_ms: instrumentation.aggregate_wall_ms,
    repeat_count: instrumentation.repeat_count,
    timing_breakdown_ms: instrumentation.timing_breakdown_ms || {},
    status_summary: instrumentation.status_summary,
  };
  if (instrumentation.runtime_counters) {
    summary.runtime_counters = instrumentation.runtime_counters;
    summary.instrumentation.runtime_counters = instrumentation.runtime_counters;
  }
  if (instrumentation.compact) {
    summary.compact = instrumentation.compact;
    summary.instrumentation.compact = instrumentation.compact;
  }
  if (Object.keys(summary.timing_breakdown_ms || {}).length === 0
      && Object.keys(instrumentation.timing_breakdown_ms || {}).length > 0) {
    summary.timing_breakdown_ms = instrumentation.timing_breakdown_ms;
  }
  return summary;
}

function perRepeat(value, repeats) {
  return Number.isFinite(value) ? value / Math.max(1, repeats || 1) : value;
}

function normalizeNativeSlowCases(slowCases, repeats) {
  const scale = Math.max(1, repeats || 1);
  return (slowCases || []).map(row => ({
    ...row,
    total_ms: perRepeat(row.total_ms, scale),
    source_compile_ms: perRepeat(row.source_compile_ms, scale),
    replay_ms: perRepeat(row.replay_ms, scale),
    session_create_ms: perRepeat(row.session_create_ms, scale),
    level_load_ms: perRepeat(row.level_load_ms, scale),
    serialize_ms: perRepeat(row.serialize_ms, scale),
  }));
}

function parseNativeSimulation(runtime, json) {
  const summary = json.status_summary || {};
  const profile = json.profile || {};
  const repeats = Math.max(1, summary.repeats || 1);
  const slowCases = normalizeNativeSlowCases(json.slow_cases || [], repeats);
  const timingBreakdown = profile ? {
    testdata_parse_ms: perRepeat(profile.testdata_parse_ms, repeats),
    source_compile_ms: Number.isFinite(profile.source_compile_median_ms) ? profile.source_compile_median_ms : perRepeat(profile.source_compile_ms, repeats),
    session_create_ms: perRepeat(profile.session_create_ms, repeats),
    level_load_ms: perRepeat(profile.level_load_ms, repeats),
    replay_ms: Number.isFinite(profile.replay_median_ms) ? profile.replay_median_ms : perRepeat(profile.replay_ms, repeats),
    replay_avg_ms: profile.replay_avg_ms,
    replay_median_ms: profile.replay_median_ms,
    serialize_ms: perRepeat(profile.serialize_ms, repeats),
  } : {};
  const aggregateWallMs = profile && Number.isFinite(profile.wall_ms) ? profile.wall_ms : summary.elapsed_ms;
  return buildRuntimeSummary(runtime, perRepeat(aggregateWallMs, repeats), {
    aggregate_wall_ms: aggregateWallMs,
    repeat_count: repeats,
    status_summary: summary,
    timing_breakdown_ms: timingBreakdown,
    runtime_counters: { ...(json.runtime_counters || {}), ...(json.compact || {}) },
    compact: json.compact || {},
    slow_cases: slowCases,
    raw_summary: json,
  });
}

function runSimulationSuite(context, options) {
  const suite = {
    runtimes: {},
    slow_cases: [],
    notes: [
      'Headline simulation timings use clean runs without hot-loop runtime counters.',
      'Runtime counters come from separate one-repeat instrumentation runs.',
    ],
  };
  const jsTimingArgs = ['--profile', '--profile-runs', String(options.runs), '--profile-json', '--sim-only'];
  const js = runCommand(context, 'simulation-js', 'node', ['src/tests/run_tests_node.js', ...jsTimingArgs], { suite: 'simulation' });
  if (js.status === 0) {
    try {
      suite.runtimes.js = parseJsSimulation(JSON.parse(js.stdout));
    } catch (error) {
      context.warnings.push({ suite: 'simulation', message: `JS simulation profile JSON parse failed: ${error.message}`, log: js.stdout_log });
    }
  }
  const jsInstrumented = runCommand(
    context,
    'simulation-js-instrumented',
    'node',
    ['src/tests/run_tests_node.js', '--profile', '--profile-runs', '1', '--profile-json', '--sim-only', '--breakdown'],
    { suite: 'simulation' }
  );
  if (jsInstrumented.status === 0 && suite.runtimes.js) {
    try {
      mergeSimulationInstrumentation(suite.runtimes.js, parseJsSimulation(JSON.parse(jsInstrumented.stdout)));
    } catch (error) {
      context.warnings.push({ suite: 'simulation', message: `JS simulation instrumentation JSON parse failed: ${error.message}`, log: jsInstrumented.stdout_log });
    }
  }

  const benchArgs = ['--jobs', String(options.jobs), '--progress-every', '0', '--repeat', String(options.runs), '--quiet', '--top-slow-cases', '10'];
  const instrumentedBenchArgs = ['--jobs', String(options.jobs), '--progress-every', '0', '--profile-timers', '--repeat', '1', '--quiet', '--top-slow-cases', '10'];
  const nativeSummary = path.join(context.artifactsDir, 'simulation_native_interpreter.json');
  const native = runCommand(
    context,
    'simulation-native-interpreter',
    'build/native/puzzlescript_cpp',
    ['test', 'simulation-corpus', 'src/tests/resources/testdata.js', ...benchArgs, '--json-summary-out', nativeSummary],
    { suite: 'simulation' }
  );
  if (native.status === 0 && readJsonIfExists(nativeSummary)) {
    suite.runtimes.native_interpreter = parseNativeSimulation('native_interpreter', readJsonIfExists(nativeSummary));
  }
  const nativeInstrumentedSummary = path.join(context.artifactsDir, 'simulation_native_interpreter_instrumented.json');
  const nativeInstrumented = runCommand(
    context,
    'simulation-native-interpreter-instrumented',
    'build/native/puzzlescript_cpp',
    ['test', 'simulation-corpus', 'src/tests/resources/testdata.js', ...instrumentedBenchArgs, '--json-summary-out', nativeInstrumentedSummary],
    { suite: 'simulation' }
  );
  if (nativeInstrumented.status === 0 && suite.runtimes.native_interpreter && readJsonIfExists(nativeInstrumentedSummary)) {
    mergeSimulationInstrumentation(suite.runtimes.native_interpreter, parseNativeSimulation('native_interpreter', readJsonIfExists(nativeInstrumentedSummary)));
  }

  const hybridSummary = path.join(context.artifactsDir, 'simulation_hybrid.json');
  const hybrid = runCommand(
    context,
    'simulation-hybrid',
    'make',
    [
      '--no-print-directory',
      'simulation_corpus_compiled_rulegroups_benchmark',
      'COMPILED_RULES_OPT_LEVEL=3',
      `SIMULATION_CORPUS_BENCH_ARGS=--jobs ${options.jobs} --progress-every 0 --repeat ${options.runs} --quiet --top-slow-cases 10 --json-summary-out ${hybridSummary}`,
    ],
    { suite: 'simulation' }
  );
  if (hybrid.status === 0 && readJsonIfExists(hybridSummary)) {
    suite.runtimes.hybrid = parseNativeSimulation('hybrid', readJsonIfExists(hybridSummary));
  }
  const hybridInstrumentedSummary = path.join(context.artifactsDir, 'simulation_hybrid_instrumented.json');
  const hybridInstrumented = runCommand(
    context,
    'simulation-hybrid-instrumented',
    'make',
    [
      '--no-print-directory',
      'simulation_corpus_compiled_rulegroups_benchmark',
      'COMPILED_RULES_OPT_LEVEL=3',
      `SIMULATION_CORPUS_BENCH_ARGS=--jobs ${options.jobs} --progress-every 0 --profile-timers --repeat 1 --quiet --top-slow-cases 10 --json-summary-out ${hybridInstrumentedSummary}`,
    ],
    { suite: 'simulation' }
  );
  if (hybridInstrumented.status === 0 && suite.runtimes.hybrid && readJsonIfExists(hybridInstrumentedSummary)) {
    mergeSimulationInstrumentation(suite.runtimes.hybrid, parseNativeSimulation('hybrid', readJsonIfExists(hybridInstrumentedSummary)));
  }

  const compiledSummary = path.join(context.artifactsDir, 'simulation_compiled.json');
  const compiled = runCommand(
    context,
    'simulation-compiled',
    'make',
    [
      '--no-print-directory',
      'simulation_corpus_compiled_compact_benchmark',
      'COMPILED_RULES_OPT_LEVEL=3',
      `SIMULATION_CORPUS_BENCH_ARGS=--jobs ${options.jobs} --progress-every 0 --repeat ${options.runs} --quiet --top-slow-cases 10 --json-summary-out ${compiledSummary}`,
    ],
    { suite: 'simulation' }
  );
  if (compiled.status === 0 && readJsonIfExists(compiledSummary)) {
    suite.runtimes.compiled = parseNativeSimulation('compiled', readJsonIfExists(compiledSummary));
  }
  const compiledInstrumentedSummary = path.join(context.artifactsDir, 'simulation_compiled_instrumented.json');
  const compiledInstrumented = runCommand(
    context,
    'simulation-compiled-instrumented',
    'make',
    [
      '--no-print-directory',
      'simulation_corpus_compiled_compact_benchmark',
      'COMPILED_RULES_OPT_LEVEL=3',
      `SIMULATION_CORPUS_BENCH_ARGS=--jobs ${options.jobs} --progress-every 0 --profile-timers --repeat 1 --quiet --top-slow-cases 10 --json-summary-out ${compiledInstrumentedSummary}`,
    ],
    { suite: 'simulation' }
  );
  if (compiledInstrumented.status === 0 && suite.runtimes.compiled && readJsonIfExists(compiledInstrumentedSummary)) {
    mergeSimulationInstrumentation(suite.runtimes.compiled, parseNativeSimulation('compiled', readJsonIfExists(compiledInstrumentedSummary)));
  }

  finalizeSpeedups(suite);
  suite.slowest_cases = collectSlowSimulationCases(suite);
  suite.timer_profiles = Object.fromEntries(Object.entries(suite.runtimes).map(([key, value]) => [key, value.instrumentation && value.instrumentation.timing_breakdown_ms || value.timing_breakdown_ms || {}]));
  return suite;
}

function collectSlowSimulationCases(suite) {
  const rows = [];
  for (const [runtime, summary] of Object.entries(suite.runtimes || {})) {
    for (const row of summary.slow_cases || []) {
      rows.push({
        runtime,
        index: row.index,
        name: row.name,
        total_ms: row.total_ms,
        source_compile_ms: row.source_compile_ms,
        replay_ms: row.replay_ms,
        session_create_ms: row.session_create_ms,
        level_load_ms: row.level_load_ms,
        serialize_ms: row.serialize_ms,
      });
    }
  }
  return rows.sort((a, b) => (b.total_ms || 0) - (a.total_ms || 0)).slice(0, 20);
}

function ensureFocusManifest(context) {
  const manifestPath = path.resolve(ROOT, 'build/native/solver_focus_group.json');
  if (fs.existsSync(manifestPath)) {
    return manifestPath;
  }
  runCommand(context, 'solver-focus-mine', 'make', ['--no-print-directory', 'solver_focus_mine'], {
    suite: 'solver',
    timeoutMs: 30 * 60 * 1000,
  });
  return fs.existsSync(manifestPath) ? manifestPath : null;
}

function resultKey(result) {
  return `${result.game}#${result.level}`;
}

function targetKey(target) {
  return `${target.game}#${target.level}`;
}

function summarizeJsSolverTargets(targetSamples, runs) {
  const targetRows = Array.from(targetSamples.values());
  const elapsedSums = [];
  const generatedSums = [];
  const expandedSums = [];
  for (let runIndex = 0; runIndex < runs; runIndex++) {
    const runSamples = targetRows.map((row) => row.samples[runIndex]).filter(Boolean);
    elapsedSums.push(sum(runSamples.map((sample) => sample.elapsed_ms)));
    generatedSums.push(sum(runSamples.map((sample) => sample.generated)));
    expandedSums.push(sum(runSamples.map((sample) => sample.expanded)));
  }
  const rows = targetRows.map((row) => ({
    game: row.game,
    level: row.level,
    status_counts: statusCounts(row.samples),
    median: {
      elapsed_ms: median(row.samples.map((sample) => sample.elapsed_ms)),
      expanded: median(row.samples.map((sample) => sample.expanded)),
      generated: median(row.samples.map((sample) => sample.generated)),
      unique_states: median(row.samples.map((sample) => sample.unique_states)),
      duplicates: median(row.samples.map((sample) => sample.duplicates)),
      max_frontier: median(row.samples.map((sample) => sample.max_frontier)),
      solution_length: median(row.samples.map((sample) => sample.solution_length)),
      compile_ms: median(row.samples.map((sample) => sample.compile_ms)),
      load_ms: median(row.samples.map((sample) => sample.load_ms)),
      clone_ms: median(row.samples.map((sample) => sample.clone_ms)),
      snapshot_ms: median(row.samples.map((sample) => sample.snapshot_ms)),
      step_ms: median(row.samples.map((sample) => sample.step_ms)),
      heuristic_ms: median(row.samples.map((sample) => sample.heuristic_ms)),
      hash_ms: median(row.samples.map((sample) => sample.hash_ms)),
      queue_ms: median(row.samples.map((sample) => sample.queue_ms)),
      reconstruct_ms: median(row.samples.map((sample) => sample.reconstruct_ms)),
      strategy: row.samples[0] && row.samples[0].strategy,
      heuristic: row.samples[0] && row.samples[0].heuristic,
      hash_mode: row.samples[0] && row.samples[0].hash_mode,
      snapshot_mode: row.samples[0] && row.samples[0].snapshot_mode,
    },
  }));

  return buildRuntimeSummary('js', median(elapsedSums), {
    focus_target_count: rows.length,
    runs,
    focus_elapsed_samples_ms: elapsedSums,
    focus_generated_median: median(generatedSums),
    focus_expanded_median: median(expandedSums),
    targets: rows,
  });
}

function statusCounts(samples) {
  const counts = {};
  for (const sample of samples) {
    counts[sample.status] = (counts[sample.status] || 0) + 1;
  }
  return counts;
}

function runJsSolverBaseline(context, options, manifestTargets) {
  if (!Array.isArray(manifestTargets) || manifestTargets.length === 0) {
    context.warnings.push({ suite: 'solver', message: 'JS solver focus baseline skipped because the focus manifest has no targets.' });
    return null;
  }
  const targetSamples = new Map();
  for (const target of manifestTargets) {
    targetSamples.set(targetKey(target), { game: target.game, level: target.level, samples: [] });
  }
  for (const target of manifestTargets) {
    const key = targetKey(target);
    for (let runIndex = 0; runIndex < options.runs; runIndex++) {
      const run = runCommand(
        context,
        `solver-js-${safeFileName(key)}-${runIndex + 1}`,
        'node',
        [
          'src/tests/run_solver_tests_js.js',
          'src/tests/solver_tests',
          '--no-timeout',
          '--strategy',
          'portfolio',
          '--portfolio-bfs-ms',
          String(Math.max(1, Math.floor(options.solverTimeoutMs / 6))),
          '--no-solutions',
          '--quiet',
          '--json',
          '--game',
          target.game,
          '--level',
          String(target.level),
        ],
        { suite: 'solver' }
      );
      if (run.status === 0) {
        try {
          const json = JSON.parse(run.stdout);
          const result = Array.isArray(json.results) ? json.results[0] : null;
          if (result) {
            targetSamples.get(key).samples.push(result);
          } else {
            context.warnings.push({ suite: 'solver', message: `JS solver returned no result for ${key}`, log: run.stdout_log });
          }
        } catch (error) {
          context.warnings.push({ suite: 'solver', message: `JS solver JSON parse failed for ${key}: ${error.message}`, log: run.stdout_log });
        }
      }
    }
  }
  const completed = Array.from(targetSamples.values()).filter((row) => row.samples.length > 0);
  if (completed.length === 0) {
    return null;
  }
  return summarizeJsSolverTargets(new Map(completed.map((row) => [`${row.game}#${row.level}`, row])), options.runs);
}

function runSolverBenchmark(context, options, runtime, outFile, extraMakeArgs = []) {
  const args = [
    '--no-print-directory',
    'solver_focus_benchmark',
    `SOLVER_FOCUS_RUNS=${options.runs}`,
    'SOLVER_FOCUS_PROFILE_COUNTERS=true',
    `SOLVER_FOCUS_TIMEOUT_MS=${options.solverTimeoutMs}`,
    `SOLVER_FOCUS_JOBS=${options.jobs}`,
    `SOLVER_FOCUS_OUT=${outFile}`,
    ...extraMakeArgs,
  ];
  const run = runCommand(context, `solver-${runtime}`, 'make', args, {
    suite: 'solver',
    timeoutMs: 45 * 60 * 1000,
  });
  const json = readJsonIfExists(outFile);
  if (!json) {
    if (run.status === 0) {
      context.warnings.push({ suite: 'solver', message: `${runtime} solver benchmark did not write readable JSON`, log: run.stdout_log });
    }
    return null;
  }
  return summarizeNativeSolverBenchmark(runtime, json);
}

function summarizeNativeSolverBenchmark(runtime, json) {
  const targets = Array.isArray(json.targets) ? json.targets : [];
  const elapsedSum = sum(targets.map((target) => target.median && target.median.elapsed_ms));
  const generatedSum = sum(targets.map((target) => target.median && target.median.generated));
  const expandedSum = sum(targets.map((target) => target.median && target.median.expanded));
  const timingTotals = {};
  for (const key of [
    'compile_ms',
    'load_ms',
    'clone_ms',
    'snapshot_ms',
    'step_ms',
    'heuristic_ms',
    'hash_ms',
    'queue_ms',
    'frontier_pop_ms',
    'frontier_push_ms',
    'visited_lookup_ms',
    'visited_insert_ms',
    'node_store_ms',
    'heuristic_ms',
    'solved_check_ms',
    'timeout_check_ms',
    'reconstruct_ms',
    'unattributed_ms',
  ]) {
    timingTotals[key] = sum(targets.map((target) => target.median && target.median[key]));
  }
  const compactTotals = {};
  for (const key of [
    'compact_turn_attempts',
    'compact_turn_hits',
    'compact_turn_native_attempts',
    'compact_turn_native_hits',
    'compact_turn_bridge_attempts',
    'compact_turn_bridge_hits',
    'compact_turn_fallbacks',
    'compact_turn_unsupported',
    'compact_state_bytes',
    'compact_max_state_bytes',
  ]) {
    compactTotals[key] = sum(targets.map((target) => target.median && target.median[key]));
  }
  return buildRuntimeSummary(runtime, elapsedSum, {
    generated_sum: generatedSum,
    expanded_sum: expandedSum,
    target_count: targets.length,
    runs_per_target: json.runs_per_target,
    median_target: json.median || {},
    timing_totals_ms: timingTotals,
    compact_totals: compactTotals,
    targets,
  });
}

function buildSolverFocusRows(suite) {
  const keys = new Set();
  for (const runtime of Object.values(suite.runtimes || {})) {
    for (const target of runtime.targets || []) {
      keys.add(`${target.game}#${target.level}`);
    }
  }
  const rows = [];
  for (const key of Array.from(keys).sort()) {
    const [game, levelText] = key.split('#');
    const row = { key, game, level: Number(levelText), runtimes: {}, speedups_vs_js: {} };
    const jsTarget = findRuntimeTarget(suite.runtimes.js, key);
    const jsComparable = isComparableSolverTarget(jsTarget);
    const jsElapsed = jsComparable && jsTarget.median && jsTarget.median.elapsed_ms;
    for (const runtime of RUNTIMES) {
      const target = findRuntimeTarget(suite.runtimes[runtime], key);
      if (!target) {
        continue;
      }
      row.runtimes[runtime] = target;
      const elapsed = target.median && target.median.elapsed_ms;
      row.speedups_vs_js[runtime] = runtime === 'js'
        ? (jsComparable ? 1 : null)
        : (isComparableSolverTarget(target) ? ratioVsJs(jsElapsed, elapsed) : null);
    }
    rows.push(row);
  }
  return rows;
}

function findRuntimeTarget(runtimeSummary, key) {
  if (!runtimeSummary) {
    return null;
  }
  return (runtimeSummary.targets || []).find((target) => `${target.game}#${target.level}` === key) || null;
}

function runSolverSuite(context, options) {
  const suite = {
    runtimes: {},
    focus_rows: [],
    worst_underperformers: [],
    biggest_wins: [],
  };
  const manifestPath = ensureFocusManifest(context);
  const manifest = manifestPath ? readJsonIfExists(manifestPath) : null;
  const allManifestTargets = manifest && Array.isArray(manifest.targets) ? manifest.targets : [];
  const manifestTargets = options.quick ? allManifestTargets.slice(0, Math.min(5, allManifestTargets.length)) : allManifestTargets;
  suite.scope = {
    js_baseline: 'selected solver focus levels only',
    native_baseline: 'selected solver focus levels only',
    js_strategy: 'portfolio: native-timeout/6 BFS slice, then weighted A*',
    js_heuristic: 'winconditions',
    target_count: manifestTargets.length,
    original_target_count: allManifestTargets.length,
    quick_subset: options.quick,
    js_solver_timeout_ms: null,
  };
  const reportManifestPath = path.join(context.artifactsDir, 'solver_focus_manifest_used.json');
  if (manifest) {
    writeJson(reportManifestPath, {
      ...manifest,
      targets: manifestTargets,
      original_target_count: allManifestTargets.length,
      target_count: manifestTargets.length,
      performance_testpage_subset: options.quick,
    });
  }
  if (!manifestPath) {
    context.warnings.push({ suite: 'solver', message: 'Solver focus manifest is unavailable; solver focus rows will be incomplete.' });
  }

  const js = runJsSolverBaseline(context, options, manifestTargets);
  if (js) {
    suite.runtimes.js = js;
  }

  const nativeOut = path.join(context.artifactsDir, 'solver_native_interpreter.json');
  const manifestArg = manifest ? [`SOLVER_FOCUS_MANIFEST=${reportManifestPath}`] : [];
  const native = runSolverBenchmark(context, options, 'native_interpreter', nativeOut, manifestArg);
  if (native) {
    suite.runtimes.native_interpreter = native;
  }

  const hybridOut = path.join(context.artifactsDir, 'solver_hybrid.json');
  const hybrid = runSolverBenchmark(context, options, 'hybrid', hybridOut, ['SPECIALIZE=true', ...manifestArg]);
  if (hybrid) {
    suite.runtimes.hybrid = hybrid;
  }

  const compiledOut = path.join(context.artifactsDir, 'solver_compiled.json');
  const compiled = runSolverBenchmark(context, options, 'compiled', compiledOut, [
    'SPECIALIZE=true',
    ...manifestArg,
    'SOLVER_FOCUS_SOLVER_ARGS=--compact-node-storage',
    'SOLVER_FOCUS_COMPILED_RULES_ARGS=--compact-turn-only --compact-turn-mode=compiler',
  ]);
  if (compiled) {
    suite.runtimes.compiled = compiled;
  }

  finalizeSpeedups(suite);
  suite.focus_rows = buildSolverFocusRows(suite);
  suite.biggest_wins = rankedSolverRows(suite.focus_rows, 'compiled', true, { minSpeedup: 1 }).slice(0, 15);
  suite.worst_underperformers = rankedSolverRows(suite.focus_rows, 'compiled', false, { maxSpeedup: 1 }).slice(0, 15);
  const jsTimeoutCount = suite.focus_rows.filter((row) => targetHasStatus(row.runtimes.js, 'timeout')).length;
  if (jsTimeoutCount > 0) {
    context.warnings.push({
      suite: 'solver',
      message: `Omitted ${jsTimeoutCount} compiled focus row(s) from exact win/underperformer rankings because the JS baseline timed out unexpectedly.`,
    });
  }
  return suite;
}

function rankedSolverRows(rows, runtime, descending, options = {}) {
  return rows
    .filter((row) => Number.isFinite(row.speedups_vs_js[runtime]))
    .filter((row) => !Number.isFinite(options.minSpeedup) || row.speedups_vs_js[runtime] > options.minSpeedup)
    .filter((row) => !Number.isFinite(options.maxSpeedup) || row.speedups_vs_js[runtime] < options.maxSpeedup)
    .map((row) => {
      const target = row.runtimes[runtime] || {};
      return {
        key: row.key,
        game: row.game,
        level: row.level,
        speedup_vs_js: row.speedups_vs_js[runtime],
        js_elapsed_ms: row.runtimes.js && row.runtimes.js.median && row.runtimes.js.median.elapsed_ms,
        runtime_elapsed_ms: target.median && target.median.elapsed_ms,
        generated: target.median && target.median.generated,
        expanded: target.median && target.median.expanded,
      };
    })
    .sort((a, b) => descending ? b.speedup_vs_js - a.speedup_vs_js : a.speedup_vs_js - b.speedup_vs_js);
}

function runGenerationSuite(context, options) {
  const outFile = path.join(context.artifactsDir, 'generator_benchmark.json');
  runCommand(
    context,
    'generation-native',
    'node',
    [
      'src/tests/run_generator_benchmark.js',
      'build/native/puzzlescript_generator',
      'src/demo/sokoban_basic.txt',
      '--presets-dir',
      'src/tests/generator_presets',
      '--samples',
      String(options.generatorSamples),
      '--runs',
      String(options.generatorRuns),
      '--jobs',
      String(options.jobs),
      '--seed',
      process.env.GENERATOR_BENCH_SEED || '11',
      '--solver-timeout-ms',
      process.env.GENERATOR_BENCH_SOLVER_TIMEOUT_MS || '50',
      '--solver-strategy',
      process.env.GENERATOR_BENCH_SOLVER_STRATEGY || 'portfolio',
      '--top-k',
      String(options.generatorTopK),
      '--out',
      outFile,
    ],
    { suite: 'generation', timeoutMs: 30 * 60 * 1000 }
  );
  const json = readJsonIfExists(outFile);
  if (!json) {
    return {
      runtimes: {
        native_interpreter: buildRuntimeSummary('native_interpreter', null, { note: 'Generation benchmark unavailable.' }),
      },
      presets: [],
    };
  }
  const elapsed = sum((json.presets || []).map((preset) => preset.elapsed_ms && preset.elapsed_ms.p50));
  return {
    runtimes: {
      native_interpreter: buildRuntimeSummary('native_interpreter', elapsed, {
        speedup_vs_js: null,
        note: 'Native-only suite; no JS baseline.',
      }),
    },
    config: json.config || {},
    presets: json.presets || [],
  };
}

function summarizeCoverageObject(json) {
  if (!json || !json.aggregate) {
    return null;
  }
  const aggregate = json.aggregate;
  const compact = aggregate.compact_turn || aggregate.compact_tick || null;
  const rulegroups = aggregate.specialized_rulegroups || aggregate.compiled_rulegroups || aggregate.rulegroups || null;
  const fullTurn = aggregate.specialized_full_turn || aggregate.compiled_tick || null;
  return {
    source_count: Array.isArray(json.sources) ? json.sources.length : null,
    max_rows: json.max_rows,
    rulegroups,
    full_turn: fullTurn,
    compact_turn: compact,
    sources: Array.isArray(json.sources) ? json.sources.map(source => ({
      index: source.index,
      path: source.path,
      early_groups: source.early_groups,
      late_groups: source.late_groups,
      compiled_groups: source.compiled_groups,
      early_rules: source.early_rules,
      late_rules: source.late_rules,
      compiled_rules: source.compiled_rules,
      misses: source.misses || {},
      full_turn_reason: source.specialized_full_turn && source.specialized_full_turn.whole_turn_fallback_reason,
      compact_mode: source.compact_turn && source.compact_turn.mode,
      compact_native_reason: source.compact_turn && source.compact_turn.native_kernel_status_reason,
    })) : [],
    aggregate,
  };
}

function runCoverageSuite(context) {
  const suite = {};
  const compiledRulesOut = path.join(context.artifactsDir, 'compiled_rules_simulation_suite_coverage.json');
  runCommand(
    context,
    'coverage-compiled-rulegroups',
    'make',
    [
      '--no-print-directory',
      'compiled_rules_simulation_suite_coverage',
      `COMPILED_RULES_SIMULATION_SUITE_COVERAGE_JSON=${compiledRulesOut}`,
    ],
    { suite: 'coverage', timeoutMs: 20 * 60 * 1000 }
  );
  suite.compiled_rulegroups = summarizeCoverageObject(readJsonIfExists(compiledRulesOut));

  const compactOut = path.join(context.artifactsDir, 'compact_turn_coverage.json');
  runCommand(
    context,
    'coverage-compact-turn',
    'make',
    [
      '--no-print-directory',
      'compact_turn_coverage',
      `COMPACT_TURN_COVERAGE_JSON=${compactOut}`,
    ],
    { suite: 'coverage', timeoutMs: 20 * 60 * 1000 }
  );
  suite.compact_turn = summarizeCoverageObject(readJsonIfExists(compactOut));

  const compactCodegenOut = path.join(context.artifactsDir, 'compact_turn_codegen_coverage.json');
  runCommand(
    context,
    'coverage-compact-codegen',
    'make',
    [
      '--no-print-directory',
      'compact_turn_codegen_coverage',
      `COMPACT_TURN_CODEGEN_COVERAGE_JSON=${compactCodegenOut}`,
    ],
    { suite: 'coverage', timeoutMs: 20 * 60 * 1000 }
  );
  suite.compact_codegen = summarizeCoverageObject(readJsonIfExists(compactCodegenOut));
  return suite;
}

function runProfilingSuite(context, options, simulationSuite) {
  const suite = {
    timer_profiles: simulationSuite.timer_profiles || {},
    optional_profile_enabled: options.profile,
    artifacts: [],
  };
  if (!options.profile) {
    return suite;
  }
  const profileOut = path.join(context.artifactsDir, 'profile_stats.txt');
  const run = runCommand(
    context,
    'profile-simulation-tests',
    'src/tests/profile_native_trace_suite.sh',
    [],
    {
      suite: 'profiling',
      env: {
        PROFILE_STATS_OUT: profileOut,
        PROFILE_REPLAY_REPEATS: String(options.runs),
        PROFILE_JOBS: String(options.jobs),
      },
      timeoutMs: 30 * 60 * 1000,
    }
  );
  if (fs.existsSync(profileOut)) {
    suite.artifacts.push({
      label: 'Native simulation profile stats',
      path: `artifacts/${path.basename(profileOut)}`,
    });
    suite.summary = parseProfileStats(readTextIfExists(profileOut) || '');
  } else if (run.status === 0) {
    context.warnings.push({ suite: 'profiling', message: 'Profile command completed without writing profile_stats.txt.' });
  }
  return suite;
}

function parseProfileStats(text) {
  return {
    native_simulation_profile: lastKeyValueLine(text, 'native_simulation_profile'),
    cpp_simulation_profile: lastKeyValueLine(text, 'cpp_simulation_profile'),
  };
}

function buildOverview(report) {
  const overview = {
    speedups: [],
    slowest_workloads: [],
    biggest_wins: [],
    worst_underperformers: [],
  };
  for (const [suiteName, suite] of Object.entries(report.suites || {})) {
    if (!suite || !suite.runtimes) {
      continue;
    }
    for (const runtime of RUNTIMES) {
      const summary = suite.runtimes[runtime];
      if (!summary) {
        continue;
      }
      overview.speedups.push({
        suite: suiteName,
        runtime,
        label: RUNTIME_LABELS[runtime],
        median_ms: summary.median_ms,
        speedup_vs_js: summary.speedup_vs_js,
      });
      if (Number.isFinite(summary.median_ms)) {
        overview.slowest_workloads.push({
          suite: suiteName,
          runtime,
          label: RUNTIME_LABELS[runtime],
          median_ms: summary.median_ms,
        });
      }
    }
  }
  overview.slowest_workloads.sort((a, b) => b.median_ms - a.median_ms);
  const solver = report.suites && report.suites.solver;
  if (solver) {
    overview.biggest_wins = (solver.biggest_wins || []).slice(0, 10);
    overview.worst_underperformers = (solver.worst_underperformers || []).slice(0, 10);
  }
  report.overview = overview;
}

function renderMarkdown(report, latestDir, runDir) {
  const lines = [];
  lines.push('# PuzzleScript Performance Testpage');
  lines.push('');
  lines.push(`Generated: ${report.metadata.generated_at}`);
  lines.push(`Git: ${report.metadata.git_short_sha}${report.metadata.dirty ? ' (dirty)' : ''}`);
  lines.push(`Mode: ${report.inputs.quick ? 'quick' : 'normal'}, runs=${report.inputs.runs}, jobs=${report.inputs.jobs}, solver_timeout=${formatTimeout(report.inputs.solver_timeout_ms)}, js_solver_timeout=${formatTimeout(report.inputs.solver_js_timeout_ms)}`);
  lines.push('');
  lines.push('## Speedups vs JS');
  lines.push('');
  lines.push('| Suite | Runtime | Median | Speedup |');
  lines.push('| --- | --- | ---: | ---: |');
  for (const row of report.overview.speedups || []) {
    lines.push(`| ${row.suite} | ${row.label} | ${formatMs(row.median_ms)} | ${formatSpeedup(row.speedup_vs_js)} |`);
  }
  lines.push('');
  lines.push('## Simulation Highlights');
  lines.push('');
  appendRuntimeList(lines, report.suites.simulation);
  const slowSim = report.suites.simulation && report.suites.simulation.slowest_cases || [];
  if (slowSim.length > 0) {
    lines.push('');
    lines.push('Slowest simulation cases:');
    for (const row of slowSim.slice(0, 8)) {
      lines.push(`- ${row.runtime} case ${row.index}: ${row.name || 'unknown'} (${formatMs(row.total_ms)})`);
    }
  }
  lines.push('');
  lines.push('## Solver Focus Highlights');
  lines.push('');
  if (report.suites.solver && report.suites.solver.scope) {
    const scope = report.suites.solver.scope;
    const subset = scope.quick_subset ? ` quick subset of ${scope.original_target_count}` : '';
    lines.push(`Scope: JS/native solver benchmarks run selected focus levels only (${scope.target_count}${subset}); JS solver timeout=${formatTimeout(scope.js_solver_timeout_ms)}; JS strategy=${scope.js_strategy || 'n/a'}.`);
    lines.push('');
  }
  appendRuntimeList(lines, report.suites.solver);
  if ((report.suites.solver.biggest_wins || []).length > 0) {
    lines.push('');
    lines.push('Biggest compiled wins:');
    for (const row of report.suites.solver.biggest_wins.slice(0, 8)) {
      lines.push(`- ${row.key}: ${formatSpeedup(row.speedup_vs_js)} (${formatMs(row.js_elapsed_ms)} JS -> ${formatMs(row.runtime_elapsed_ms)} compiled)`);
    }
  }
  if ((report.suites.solver.worst_underperformers || []).length > 0) {
    lines.push('');
    lines.push('Worst compiled underperformers:');
    for (const row of report.suites.solver.worst_underperformers.slice(0, 8)) {
      lines.push(`- ${row.key}: ${formatSpeedup(row.speedup_vs_js)} (${formatMs(row.js_elapsed_ms)} JS -> ${formatMs(row.runtime_elapsed_ms)} compiled)`);
    }
  }
  lines.push('');
  lines.push('## Generation Highlights');
  lines.push('');
  const generation = report.suites.generation || {};
  for (const preset of generation.presets || []) {
    lines.push(`- ${preset.preset}: ${compactNumber(preset.samples_per_second && preset.samples_per_second.p50)} samples/s, solve_rate=${formatPercent(preset.solve_rate)}, dedupe=${formatPercent(preset.dedupe_rate)}`);
  }
  if (!generation.presets || generation.presets.length === 0) {
    lines.push('- n/a');
  }
  lines.push('');
  lines.push('## Coverage Highlights');
  lines.push('');
  for (const [name, coverage] of Object.entries(report.suites.coverage || {})) {
    lines.push(`- ${name}: ${coverageSummaryText(coverage)}`);
  }
  lines.push('');
  lines.push('## Profiling Highlights');
  lines.push('');
  lines.push(report.suites.profiling.optional_profile_enabled ? '- Optional profiler was enabled.' : '- Optional profiler was not enabled; timer profiles and counters are included from separate instrumentation runs where available.');
  for (const artifact of report.suites.profiling.artifacts || []) {
    lines.push(`- ${artifact.label}: ${artifact.path}`);
  }
  lines.push('');
  lines.push('## Warnings');
  lines.push('');
  if (report.warnings.length === 0) {
    lines.push('- none');
  } else {
    for (const warning of report.warnings) {
      lines.push(`- [${warning.suite || 'general'}] ${warning.message}${warning.log ? ` (${warning.log})` : ''}`);
    }
  }
  lines.push('');
  lines.push('## Artifacts');
  lines.push('');
  lines.push(`- Latest: ${latestDir}`);
  lines.push(`- Timestamped: ${runDir}`);
  lines.push(`- Commands/logs: ${path.join(latestDir, 'logs')}`);
  return `${lines.join('\n')}\n`;
}

function appendRuntimeList(lines, suite) {
  if (!suite || !suite.runtimes) {
    lines.push('- n/a');
    return;
  }
  for (const runtime of RUNTIMES) {
    const summary = suite.runtimes[runtime];
    if (!summary) {
      continue;
    }
    lines.push(`- ${RUNTIME_LABELS[runtime]}: ${formatMs(summary.median_ms)}, ${formatSpeedup(summary.speedup_vs_js)}`);
  }
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'n/a';
}

function coverageSummaryText(coverage) {
  if (!coverage) {
    return 'n/a';
  }
  const compact = coverage.compact_turn;
  const fullTurn = coverage.full_turn;
  const rulegroups = coverage.rulegroups;
  const parts = [`sources=${coverage.source_count ?? 'n/a'}`];
  if (rulegroups && Number.isFinite(rulegroups.fully_compiled)) {
    parts.push(`rulegroups=${rulegroups.fully_compiled}/${rulegroups.sources ?? coverage.source_count}`);
  }
  if (fullTurn && Number.isFinite(fullTurn.whole_turn_supported)) {
    parts.push(`full_turn=${fullTurn.whole_turn_supported}/${fullTurn.sources ?? coverage.source_count}`);
  }
  if (compact && Number.isFinite(compact.whole_turn_supported)) {
    parts.push(`compact=${compact.whole_turn_supported}/${compact.sources ?? coverage.source_count}`);
  }
  return parts.join(', ');
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderHtml(report) {
  const embedded = JSON.stringify(report).replace(/</g, '\\u003c');
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PuzzleScript Performance Testpage</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #5e6875;
      --line: #d9dee8;
      --blue: #2563eb;
      --green: #0f8a5f;
      --amber: #b7791f;
      --red: #b42318;
      --violet: #6d5bd0;
      --shadow: 0 1px 2px rgba(18, 25, 38, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 18px 22px; }
    h1 { margin: 0 0 4px; font-size: 22px; font-weight: 700; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }
    .subtle { color: var(--muted); }
    .tabs {
      display: flex;
      gap: 6px;
      padding-top: 14px;
      overflow-x: auto;
    }
    .tab {
      border: 1px solid var(--line);
      background: #eef2f8;
      color: var(--ink);
      padding: 8px 12px;
      border-radius: 6px 6px 0 0;
      cursor: pointer;
      white-space: nowrap;
      font: inherit;
    }
    .tab[aria-selected="true"] {
      background: var(--bg);
      border-bottom-color: var(--bg);
      color: var(--blue);
      font-weight: 650;
    }
    main .view { display: none; }
    main .view.active { display: block; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }
    .chart-grid {
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 430px), 1fr));
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
      min-width: 0;
    }
    .metric {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 8px 0;
      border-top: 1px solid #edf0f5;
    }
    .metric:first-child { border-top: 0; }
    .metric strong { font-size: 18px; }
    .speed-good { color: var(--green); }
    .speed-bad { color: var(--red); }
    .speed-neutral { color: var(--muted); }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
    .table-wrap.compact { overflow-x: auto; }
    .table-wrap.compact table { min-width: 0; table-layout: auto; }
    .table-wrap.compact th, .table-wrap.compact td { overflow-wrap: anywhere; }
    .table-wrap.compact .num { white-space: nowrap; text-align: right; }
    .table-wrap.compact .target { min-width: 14rem; }
    .wide-grid {
      grid-template-columns: minmax(min(100%, 340px), 0.85fr) minmax(min(100%, 520px), 1.35fr);
    }
    table { width: 100%; border-collapse: collapse; min-width: 760px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #edf0f5; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; z-index: 1; background: #f1f4f9; color: #303a48; cursor: pointer; font-weight: 650; }
    tr:hover td { background: #fafcff; }
    .toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
    input[type="search"] {
      min-width: min(420px, 100%);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
      background: var(--panel);
    }
    .bars { display: grid; gap: 8px; }
    .legend {
      display: flex;
      gap: 10px 14px;
      flex-wrap: wrap;
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 3px;
      border: 1px solid rgba(0, 0, 0, 0.12);
      flex: 0 0 auto;
    }
    .swatch.compile { background: #2563eb; }
    .swatch.replay { background: #0f8a5f; }
    .swatch.session { background: #b7791f; }
    .swatch.load { background: #6d5bd0; }
    .swatch.serialize { background: #b42318; }
    .swatch.other { background: #718096; }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(110px, max-content) minmax(120px, 1fr) max-content;
      gap: 10px;
      align-items: center;
      min-width: 0;
    }
    .bar-row.timing-row {
      grid-template-columns: minmax(110px, max-content) minmax(120px, 1fr) max-content;
    }
    .bar-track { height: 18px; background: #edf1f6; border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--blue); min-width: 1px; }
    .bar-fill.hybrid { background: var(--violet); }
    .bar-fill.compiled { background: var(--green); }
    .bar-fill.native_interpreter { background: var(--amber); }
    .bar-row strong { text-align: right; white-space: nowrap; }
    .bar-row > span { text-align: right; white-space: nowrap; }
    .stack { display: flex; height: 20px; overflow: hidden; border-radius: 4px; background: #edf1f6; }
    .seg { height: 100%; min-width: 1px; }
    .seg.compile { background: #2563eb; }
    .seg.replay { background: #0f8a5f; }
    .seg.session { background: #b7791f; }
    .seg.load { background: #6d5bd0; }
    .seg.serialize { background: #b42318; }
    .seg.other { background: #718096; }
    .scatter svg { width: 100%; height: 360px; display: block; background: #fbfcff; border: 1px solid var(--line); border-radius: 8px; }
    .warn { border-left: 4px solid var(--amber); padding: 8px 10px; background: #fff8e6; margin-bottom: 8px; border-radius: 4px; }
    .empty { color: var(--muted); padding: 18px; }
    code { background: #eef2f8; padding: 1px 4px; border-radius: 4px; }
    @media (max-width: 760px) {
      .wrap { padding: 14px; }
      .bar-row, .bar-row.timing-row { grid-template-columns: 1fr; gap: 4px; min-width: 0; }
      table { min-width: 680px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>PuzzleScript Performance Testpage</h1>
      <div class="subtle" id="meta"></div>
      <nav class="tabs" id="tabs"></nav>
    </div>
  </header>
  <main class="wrap" id="app"></main>
  <script>window.REPORT = ${embedded};</script>
  <script>
    const report = window.REPORT;
    const runtimes = ['js', 'native_interpreter', 'hybrid', 'compiled'];
    const labels = { js: 'JS', native_interpreter: 'C++ Interpreter', hybrid: 'C++ Hybrid', compiled: 'C++ Compiled' };
    const views = [
      ['overview', 'Overview'],
      ['simulation', 'Simulation'],
      ['solver', 'Solver'],
      ['generation', 'Generation'],
      ['coverage', 'Coverage'],
      ['profiling', 'Profiling'],
    ];
    const fmtMs = value => Number.isFinite(value) ? (value >= 1000 ? (value / 1000).toFixed(2) + 's' : value.toFixed(1) + 'ms') : 'n/a';
    const fmtNum = value => Number.isFinite(value) ? (Math.abs(value) >= 1000000 ? (value / 1000000).toFixed(1) + 'm' : Math.abs(value) >= 1000 ? (value / 1000).toFixed(1) + 'k' : value.toFixed(value % 1 ? 1 : 0)) : 'n/a';
    const fmtSpeed = value => Number.isFinite(value) ? value.toFixed(2) + 'x' : 'n/a';
    const fmtPct = value => Number.isFinite(value) ? (value * 100).toFixed(1) + '%' : 'n/a';
    const fmtTimeout = value => Number.isFinite(value) ? value + 'ms' : 'no limit';
    const speedClass = value => !Number.isFinite(value) ? 'speed-neutral' : value >= 1 ? 'speed-good' : 'speed-bad';
    const fmtCountPct = value => value && Number.isFinite(value.count) && Number.isFinite(value.total) && value.total > 0 ? \`\${fmtNum(value.count)} / \${fmtNum(value.total)} (\${fmtPct(value.count / value.total)})\` : 'n/a';
    document.getElementById('meta').textContent = \`\${report.metadata.generated_at} | \${report.metadata.git_short_sha || 'unknown'}\${report.metadata.dirty ? ' dirty' : ''} | runs=\${report.inputs.runs} | jobs=\${report.inputs.jobs} | solver=\${fmtTimeout(report.inputs.solver_timeout_ms)} | js solver=\${fmtTimeout(report.inputs.solver_js_timeout_ms)}\`;

    const tabs = document.getElementById('tabs');
    const app = document.getElementById('app');
    for (const [id, label] of views) {
      const button = document.createElement('button');
      button.className = 'tab';
      button.type = 'button';
      button.textContent = label;
      button.dataset.view = id;
      button.addEventListener('click', () => showView(id));
      tabs.appendChild(button);
      const section = document.createElement('section');
      section.className = 'view';
      section.id = 'view-' + id;
      app.appendChild(section);
    }

    function showView(id) {
      document.querySelectorAll('.tab').forEach(tab => tab.setAttribute('aria-selected', String(tab.dataset.view === id)));
      document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === 'view-' + id));
      location.hash = id;
    }

    function panel(title, body) {
      return \`<section class="panel"><h2>\${title}</h2>\${body}</section>\`;
    }

    function speedupRows(suiteName, suite) {
      return runtimes.map(runtime => {
        const summary = suite && suite.runtimes && suite.runtimes[runtime];
        return { suite: suiteName, runtime, label: labels[runtime], median_ms: summary && summary.median_ms, speedup_vs_js: summary && summary.speedup_vs_js };
      }).filter(row => row.median_ms !== undefined || row.runtime === 'js');
    }

    function renderSpeedTable(rows, compact = false) {
      return renderTable(rows, [
        ['suite', 'Suite'],
        ['label', 'Runtime'],
        ['median_ms', 'Median', fmtMs],
        ['speedup_vs_js', 'Speedup vs JS', value => \`<span class="\${speedClass(value)}">\${fmtSpeed(value)}</span>\`],
      ], false, compact);
    }

    function renderBars(rows) {
      const max = Math.max(1, ...rows.map(row => Number.isFinite(row.median_ms) ? row.median_ms : 0));
      return \`<div class="bars">\${rows.map(row => \`
        <div class="bar-row" title="\${row.label}: \${fmtMs(row.median_ms)} (\${fmtSpeed(row.speedup_vs_js)} vs JS)">
          <div>\${row.label}</div>
          <div class="bar-track"><div class="bar-fill \${row.runtime}" style="width:\${Math.max(1, ((row.median_ms || 0) / max) * 100)}%"></div></div>
          <strong class="\${speedClass(row.speedup_vs_js)}">\${fmtMs(row.median_ms)} (\${fmtSpeed(row.speedup_vs_js)})</strong>
        </div>\`).join('')}</div>\`;
    }

    function timingLegend() {
      return \`<div class="legend" aria-label="Timing breakdown legend">
        <span class="legend-item"><span class="swatch compile"></span>Compile</span>
        <span class="legend-item"><span class="swatch session"></span>Session</span>
        <span class="legend-item"><span class="swatch load"></span>Load / parse</span>
        <span class="legend-item"><span class="swatch replay"></span>Replay / input</span>
        <span class="legend-item"><span class="swatch serialize"></span>Serialize</span>
      </div>\`;
    }

    function renderStackedTiming(suite) {
      if (!suite || !suite.runtimes) return '<div class="empty">n/a</div>';
      const rows = runtimes.map(runtime => {
        const timing = suite.runtimes[runtime] && suite.runtimes[runtime].timing_breakdown_ms;
        return { runtime, label: labels[runtime], timing: timing || {} };
      }).filter(row => Object.keys(row.timing).length > 0);
      if (rows.length === 0) return '<div class="empty">n/a</div>';
      return timingLegend() + \`<div class="bars">\${rows.map(row => {
        const t = row.timing;
        const parts = [
          ['compile', 'Compile', t.source_compile_ms || 0],
          ['session', 'Session', t.session_create_ms || 0],
          ['load', 'Load / parse', (t.level_load_ms || 0) + (t.testdata_parse_ms || 0)],
          ['replay', 'Replay / input', t.replay_ms || t.process_input_ms || 0],
          ['serialize', 'Serialize', t.serialize_ms || 0],
        ];
        const total = parts.reduce((acc, item) => acc + item[2], 0) || row.timing.wall_ms || 1;
        const title = parts.map(([, label, value]) => \`\${label}: \${fmtMs(value)}\`).join(' | ');
        return \`<div class="bar-row timing-row" title="\${row.label} timing: \${title}"><div>\${row.label}</div><div class="stack" title="\${title}">\${parts.map(([name, label, value]) => \`<div class="seg \${name}" title="\${label}: \${fmtMs(value)}" aria-label="\${label}: \${fmtMs(value)}" style="width:\${Math.max(0, value / total * 100)}%"></div>\`).join('')}</div><span>\${fmtMs(total)}</span></div>\`;
      }).join('')}</div>\`;
    }

    function renderWarnings() {
      if (!report.warnings || report.warnings.length === 0) return '<div class="empty">No warnings.</div>';
      return report.warnings.map(w => \`<div class="warn"><strong>\${w.suite || 'general'}</strong>: \${escapeHtml(w.message)}\${w.log ? ' <code>' + escapeHtml(w.log) + '</code>' : ''}</div>\`).join('');
    }

    function renderRunScope() {
      const solver = report.suites.solver && report.suites.solver.scope;
      const solverScope = solver
        ? \`Selected focus levels only (\${solver.target_count}\${solver.quick_subset ? ' quick subset of ' + solver.original_target_count : ''}); JS solver timeout: \${fmtTimeout(solver.js_solver_timeout_ms)}\`
        : 'n/a';
      return renderTable([
        { area: 'Simulation', scope: 'Simulation corpus' },
        { area: 'Solver', scope: solverScope },
        { area: 'Generation', scope: 'Native generator presets' },
      ], [['area', 'Area'], ['scope', 'Scope']], false, true);
    }

    function renderOverview() {
      const rows = report.overview.speedups || [];
      const biggestWins = report.overview.biggest_wins || [];
      const underperformers = report.overview.worst_underperformers || [];
      const winColumns = [
        ['key', 'Target', null, 'target'],
        ['speedup_vs_js', 'Speedup', fmtSpeed, 'num'],
        ['js_elapsed_ms', 'JS', fmtMs, 'num'],
        ['runtime_elapsed_ms', 'C++ Compiled', fmtMs, 'num'],
      ];
      document.getElementById('view-overview').innerHTML = \`
        <div class="grid">
          \${panel('Runtime Speedups', renderSpeedTable(rows, true))}
          \${panel('Run Scope', renderRunScope())}
          \${panel('Warnings', renderWarnings())}
        </div>
        <div class="grid wide-grid">
          \${panel('Slowest Workloads', renderTable((report.overview.slowest_workloads || []).slice(0, 12), [['suite', 'Suite'], ['label', 'Runtime'], ['median_ms', 'Median', fmtMs]], false, true))}
          \${panel('Biggest C++ Compiled Wins', renderTable(biggestWins, winColumns, false, true))}
        </div>\`;
      if (underperformers.length > 0) {
        document.getElementById('view-overview').innerHTML += panel('Worst C++ Compiled Underperformers', renderTable(underperformers, winColumns, false, true));
      }
    }

    function renderSimulation() {
      const suite = report.suites.simulation || {};
      const rows = speedupRows('simulation', suite);
      document.getElementById('view-simulation').innerHTML = \`
        <div class="grid chart-grid">
          \${panel('Speedups vs JS', renderBars(rows))}
          \${panel('Timing Breakdown', renderStackedTiming(suite))}
        </div>
        \${panel('Measurement Notes', renderSimulationNotes(suite))}
        \${panel('Runtime Summary', renderSpeedTable(rows))}
        \${panel('Slowest Cases', renderTable(suite.slowest_cases || [], [['runtime', 'Runtime'], ['index', 'Case'], ['name', 'Name'], ['total_ms', 'Total', fmtMs], ['source_compile_ms', 'Compile', fmtMs], ['replay_ms', 'Replay', fmtMs]]))}
        \${renderCounterPanels(suite)}\`;
    }

    function renderSimulationNotes(suite) {
      const rows = (suite.notes || []).map(note => ({ note }));
      return rows.length ? renderTable(rows, [['note', 'Note']], false, true) : '<div class="empty">n/a</div>';
    }

    function renderCounterPanels(suite) {
      if (!suite || !suite.runtimes) return '<div class="empty">n/a</div>';
      const panels = [];
      for (const runtime of runtimes) {
        const counters = suite.runtimes[runtime] && suite.runtimes[runtime].runtime_counters;
        if (!counters) continue;
        const rows = [];
        for (const [key, value] of Object.entries(counters)) {
          if (Number.isFinite(value) && value !== 0) rows.push({ counter: key, value });
        }
        if (rows.length > 0) {
          panels.push(panel(\`\${labels[runtime]} Counters\`, renderTable(rows, [['counter', 'Counter'], ['value', 'Value', fmtNum]], false, true)));
        }
      }
      if (panels.length === 0) return panel('Runtime Counters', '<div class="empty">n/a</div>');
      return \`<div class="grid">\${panels.join('')}</div>\`;
    }

    function renderSolver() {
      const suite = report.suites.solver || {};
      const rows = speedupRows('solver', suite);
      document.getElementById('view-solver').innerHTML = \`
        <div class="grid chart-grid">
          \${panel('Focus Speedups vs JS', renderBars(rows))}
          \${panel('Timing Totals', renderSolverTiming(suite))}
        </div>
        \${panel('Solver Scope', renderSolverScope(suite))}
        \${panel('Focus Scatter', renderScatter(suite.focus_rows || []))}
        \${panel('Focus Targets', renderFocusTable(suite.focus_rows || []))}
        <div class="grid">
          \${panel('Largest C++ Compiled Wins', renderTable(suite.biggest_wins || [], [['key', 'Target'], ['speedup_vs_js', 'Speedup', fmtSpeed], ['js_elapsed_ms', 'JS', fmtMs], ['runtime_elapsed_ms', 'C++ Compiled', fmtMs], ['generated', 'Generated', fmtNum]], false, true))}
          \${panel('C++ Compiled Underperformers', renderTable(suite.worst_underperformers || [], [['key', 'Target'], ['speedup_vs_js', 'Speedup', fmtSpeed], ['js_elapsed_ms', 'JS', fmtMs], ['runtime_elapsed_ms', 'C++ Compiled', fmtMs], ['generated', 'Generated', fmtNum]], false, true))}
        </div>\`;
    }

    function renderSolverScope(suite) {
      const scope = suite.scope;
      if (!scope) return '<div class="empty">n/a</div>';
      const rows = [
        { key: 'JS baseline', value: \`Selected focus levels only; timeout \${fmtTimeout(scope.js_solver_timeout_ms)}\` },
        { key: 'JS strategy', value: scope.js_strategy || 'n/a' },
        { key: 'JS heuristic', value: scope.js_heuristic || 'n/a' },
        { key: 'Native/hybrid/compiled', value: 'Selected focus levels only' },
        { key: 'Targets', value: \`\${scope.target_count}\${scope.quick_subset ? ' quick subset of ' + scope.original_target_count : ''}\` },
      ];
      return renderTable(rows, [['key', 'Item'], ['value', 'Value']], false, true);
    }

    function renderSolverTiming(suite) {
      const rows = runtimes.map(runtime => {
        const t = suite.runtimes && suite.runtimes[runtime] && suite.runtimes[runtime].timing_totals_ms;
        return { runtime, label: labels[runtime], timing: t || {} };
      }).filter(row => Object.keys(row.timing).length > 0);
      if (rows.length === 0) return '<div class="empty">n/a</div>';
      return renderTable(rows.map(row => ({
        runtime: row.label,
        step: row.timing.step_ms,
        clone: row.timing.clone_ms,
        snapshot: row.timing.snapshot_ms,
        hash: row.timing.hash_ms,
        heuristic: row.timing.heuristic_ms,
        frontier: (row.timing.frontier_pop_ms || 0) + (row.timing.frontier_push_ms || 0),
        visited: (row.timing.visited_lookup_ms || 0) + (row.timing.visited_insert_ms || 0),
        heuristic: row.timing.heuristic_ms,
        unattributed: row.timing.unattributed_ms,
      })), [['runtime', 'Runtime'], ['step', 'Step', fmtMs], ['clone', 'Restore', fmtMs], ['snapshot', 'Snapshot', fmtMs], ['hash', 'Hash', fmtMs], ['heuristic', 'Heuristic', fmtMs], ['frontier', 'Frontier', fmtMs], ['visited', 'Visited', fmtMs], ['unattributed', 'Unattributed', fmtMs]]);
    }

    function renderScatter(rows) {
      const points = rows.map(row => {
        const js = row.runtimes.js && row.runtimes.js.median && row.runtimes.js.median.elapsed_ms;
        const compiled = row.runtimes.compiled && row.runtimes.compiled.median && row.runtimes.compiled.median.elapsed_ms;
        return { key: row.key, js, compiled, speedup: row.speedups_vs_js.compiled };
      }).filter(point => Number.isFinite(point.js) && Number.isFinite(point.compiled));
      if (points.length === 0) return '<div class="empty">n/a</div>';
      const max = Math.max(...points.map(p => Math.max(p.js, p.compiled)), 1);
      const pad = 36, w = 760, h = 320;
      const x = value => pad + (value / max) * (w - pad * 2);
      const y = value => h - pad - (value / max) * (h - pad * 2);
      return \`<div class="scatter"><svg viewBox="0 0 \${w} \${h}" role="img" aria-label="JS vs compiled elapsed scatterplot">
        <line x1="\${pad}" y1="\${h-pad}" x2="\${w-pad}" y2="\${pad}" stroke="#9aa5b5" stroke-dasharray="4 4"/>
        <line x1="\${pad}" y1="\${h-pad}" x2="\${w-pad}" y2="\${h-pad}" stroke="#303a48"/>
        <line x1="\${pad}" y1="\${h-pad}" x2="\${pad}" y2="\${pad}" stroke="#303a48"/>
        \${points.map(p => \`<circle cx="\${x(p.js).toFixed(1)}" cy="\${y(p.compiled).toFixed(1)}" r="4" fill="\${p.speedup >= 1 ? '#0f8a5f' : '#b42318'}"><title>\${escapeHtml(p.key)} JS=\${fmtMs(p.js)} compiled=\${fmtMs(p.compiled)} speedup=\${fmtSpeed(p.speedup)}</title></circle>\`).join('')}
        <text x="\${w/2}" y="\${h-6}" text-anchor="middle" fill="#5e6875">JS elapsed</text>
        <text x="12" y="\${h/2}" text-anchor="middle" transform="rotate(-90 12 \${h/2})" fill="#5e6875">C++ compiled elapsed</text>
      </svg></div>\`;
    }

    function renderFocusTable(rows) {
      const mapped = rows.map(row => ({
        key: row.key,
        js_ms: row.runtimes.js && row.runtimes.js.median && row.runtimes.js.median.elapsed_ms,
        native_ms: row.runtimes.native_interpreter && row.runtimes.native_interpreter.median && row.runtimes.native_interpreter.median.elapsed_ms,
        hybrid_ms: row.runtimes.hybrid && row.runtimes.hybrid.median && row.runtimes.hybrid.median.elapsed_ms,
        compiled_ms: row.runtimes.compiled && row.runtimes.compiled.median && row.runtimes.compiled.median.elapsed_ms,
        compiled_speedup: row.speedups_vs_js.compiled,
        generated: row.runtimes.compiled && row.runtimes.compiled.median && row.runtimes.compiled.median.generated,
        expanded: row.runtimes.compiled && row.runtimes.compiled.median && row.runtimes.compiled.median.expanded,
        unique_states: row.runtimes.compiled && row.runtimes.compiled.median && row.runtimes.compiled.median.unique_states,
        duplicates: row.runtimes.compiled && row.runtimes.compiled.median && row.runtimes.compiled.median.duplicates,
        max_frontier: row.runtimes.compiled && row.runtimes.compiled.median && row.runtimes.compiled.median.max_frontier,
      }));
      return renderTable(mapped, [['key', 'Target'], ['js_ms', 'JS', fmtMs], ['native_ms', 'C++ Interpreter', fmtMs], ['hybrid_ms', 'C++ Hybrid', fmtMs], ['compiled_ms', 'C++ Compiled', fmtMs], ['compiled_speedup', 'C++ Compiled Speedup', fmtSpeed], ['generated', 'Generated', fmtNum], ['expanded', 'Expanded', fmtNum], ['unique_states', 'Unique', fmtNum], ['duplicates', 'Duplicates', fmtNum], ['max_frontier', 'Max Frontier', fmtNum]], true);
    }

    function renderGeneration() {
      const suite = report.suites.generation || {};
      document.getElementById('view-generation').innerHTML = \`
        \${panel('Native-Only Summary', renderTable(speedupRows('generation', suite), [['label', 'Runtime'], ['median_ms', 'Median', fmtMs], ['speedup_vs_js', 'Speedup vs JS', fmtSpeed]]))}
        \${panel('Presets', renderTable((suite.presets || []).map(p => ({
          preset: p.preset,
          samples_per_second: p.samples_per_second && p.samples_per_second.p50,
          valid_per_second: p.valid_per_second && p.valid_per_second.p50,
          solve_rate: p.solve_rate,
          dedupe_rate: p.dedupe_rate,
          top_score: p.top_score && p.top_score.p50,
          top_solution_length: p.top_solution_length && p.top_solution_length.p50,
          solver_expanded: p.solver_expanded && p.solver_expanded.p50,
          solver_generated: p.solver_generated && p.solver_generated.p50,
        })), [['preset', 'Preset'], ['samples_per_second', 'Samples/s', fmtNum], ['valid_per_second', 'Valid/s', fmtNum], ['solve_rate', 'Solve Rate', fmtPct], ['dedupe_rate', 'Dedupe Rate', fmtPct], ['top_score', 'Top Score', fmtNum], ['top_solution_length', 'Top Solution', fmtNum], ['solver_expanded', 'Solver Expanded', fmtNum], ['solver_generated', 'Solver Generated', fmtNum]]))}\`;
    }

    function compactReason(reason) {
      return reason ? String(reason).replaceAll('_', ' ') : 'n/a';
    }

    function countObjectRows(coverageName, area, counts, total) {
      return Object.entries(counts || {})
        .filter(([, count]) => Number.isFinite(count) && count > 0)
        .map(([reason, count]) => ({
          coverage: coverageName,
          area,
          reason: compactReason(reason),
          count,
          share: Number.isFinite(total) && total > 0 ? count / total : null,
        }));
    }

    function coverageMetricRows(suite) {
      return Object.entries(suite || {}).map(([name, cov]) => {
        const rg = cov && cov.rulegroups;
        const full = cov && cov.full_turn;
        const compact = cov && cov.compact_turn;
        const groupTotal = rg && (rg.early_groups || 0) + (rg.late_groups || 0);
        const ruleTotal = rg && (rg.early_rules || 0) + (rg.late_rules || 0);
        return {
          name,
          sources: cov && cov.source_count,
          max_rows: cov && cov.max_rows,
          groups: rg && { count: rg.compiled_groups, total: groupTotal },
          rules: rg && { count: rg.compiled_rules, total: ruleTotal },
          missed_groups: rg && Number.isFinite(groupTotal) ? groupTotal - (rg.compiled_groups || 0) : null,
          missed_rules: rg && Number.isFinite(ruleTotal) ? ruleTotal - (rg.compiled_rules || 0) : null,
          early_loops: full && { count: full.early_rule_loops_generated, total: full.sources },
          late_loops: full && { count: full.late_rule_loops_generated, total: full.sources },
          compact_native: compact && { count: compact.native_kernel_supported, total: compact.sources },
          compact_bridge: compact && { count: compact.interpreter_bridge_supported, total: compact.sources },
        };
      });
    }

    function coverageReasonRows(suite) {
      const rows = [];
      for (const [name, cov] of Object.entries(suite || {})) {
        if (!cov) continue;
        const rg = cov.rulegroups || {};
        const full = cov.full_turn || {};
        const compact = cov.compact_turn || {};
        rows.push(...countObjectRows(name, 'Rulegroup misses', rg.misses, Object.values(rg.misses || {}).reduce((sum, count) => sum + count, 0)));
        rows.push(...countObjectRows(name, 'Full-turn fallback', full.whole_turn_fallback_reason_counts, full.sources));
        rows.push(...countObjectRows(name, 'Full-turn commands', full.command_status_counts, full.sources));
        rows.push(...countObjectRows(name, 'Compact status', compact.whole_turn_status_reason_counts, compact.sources));
        rows.push(...countObjectRows(name, 'Compact native fallback', compact.native_kernel_status_reason_counts, compact.sources));
      }
      return rows.sort((a, b) => b.count - a.count);
    }

    function coverageSourceRows(suite) {
      const rows = [];
      for (const [coverage, cov] of Object.entries(suite || {})) {
        for (const source of (cov && cov.sources) || []) {
          const groups = (source.early_groups || 0) + (source.late_groups || 0);
          const rules = (source.early_rules || 0) + (source.late_rules || 0);
          const missedGroups = groups - (source.compiled_groups || 0);
          const missedRules = rules - (source.compiled_rules || 0);
          if (missedGroups <= 0 && missedRules <= 0) continue;
          rows.push({
            coverage,
            source: source.path,
            groups: { count: source.compiled_groups, total: groups },
            rules: { count: source.compiled_rules, total: rules },
            missed_groups: missedGroups,
            missed_rules: missedRules,
            misses: Object.entries(source.misses || {}).map(([reason, count]) => \`\${compactReason(reason)}: \${fmtNum(count)}\`).join(', ') || 'n/a',
            full_turn_reason: compactReason(source.full_turn_reason),
            compact_mode: compactReason(source.compact_mode),
          });
        }
      }
      return rows.sort((a, b) => (b.missed_groups - a.missed_groups) || (b.missed_rules - a.missed_rules)).slice(0, 25);
    }

    function renderCoverage() {
      const suite = report.suites.coverage || {};
      document.getElementById('view-coverage').innerHTML = \`
        \${panel('Coverage Rates', renderTable(coverageMetricRows(suite), [['name', 'Coverage'], ['sources', 'Sources', fmtNum], ['max_rows', 'Max Rows', fmtNum], ['groups', 'Compiled Groups', fmtCountPct], ['rules', 'Compiled Rules', fmtCountPct], ['missed_groups', 'Missed Groups', fmtNum], ['missed_rules', 'Missed Rules', fmtNum], ['early_loops', 'Early Loops', fmtCountPct], ['late_loops', 'Late Loops', fmtCountPct], ['compact_native', 'Compact Native', fmtCountPct], ['compact_bridge', 'Compact Bridge', fmtCountPct]], false, true))}
        \${panel('Coverage Reasons', renderTable(coverageReasonRows(suite), [['coverage', 'Coverage'], ['area', 'Area'], ['reason', 'Reason'], ['count', 'Count', fmtNum], ['share', 'Share', fmtPct]], false, true))}
        \${panel('Rulegroup Miss Sources', renderTable(coverageSourceRows(suite), [['coverage', 'Coverage'], ['source', 'Source'], ['groups', 'Groups', fmtCountPct], ['rules', 'Rules', fmtCountPct], ['missed_groups', 'Missed Groups', fmtNum], ['missed_rules', 'Missed Rules', fmtNum], ['misses', 'Misses'], ['full_turn_reason', 'Full Turn'], ['compact_mode', 'Compact']], true))}\`;
    }

    function renderProfiling() {
      const suite = report.suites.profiling || {};
      const rows = [];
      for (const [runtime, profile] of Object.entries(suite.timer_profiles || {})) {
        rows.push({ runtime: labels[runtime] || runtime, ...profile });
      }
      document.getElementById('view-profiling').innerHTML = \`
        \${panel('Timer Profiles', renderTable(rows, [['runtime', 'Runtime'], ['source_compile_ms', 'Compile', fmtMs], ['session_create_ms', 'Session', fmtMs], ['level_load_ms', 'Load', fmtMs], ['replay_ms', 'Replay', fmtMs], ['serialize_ms', 'Serialize', fmtMs], ['replay_median_ms', 'Replay Median', fmtMs]]))}
        \${panel('Profiler Artifacts', suite.artifacts && suite.artifacts.length ? renderTable(suite.artifacts, [['label', 'Artifact'], ['path', 'Path']]) : '<div class="empty">Optional profiler was not run.</div>')}\`;
    }

    function renderTable(rows, columns, filterable = false, compact = false) {
      rows = rows || [];
      if (rows.length === 0) return '<div class="empty">n/a</div>';
      const id = 'tbl-' + Math.random().toString(36).slice(2);
      const toolbar = filterable ? \`<div class="toolbar"><input type="search" placeholder="Filter rows" oninput="filterTable('\${id}', this.value)"></div>\` : '';
      const head = columns.map(([key, title, , className]) => \`<th class="\${className || ''}" onclick="sortTable('\${id}', '\${key}')">\${title}</th>\`).join('');
      const body = rows.map(row => \`<tr>\${columns.map(([key, , formatter, className]) => {
        const raw = row[key];
        const html = formatter ? formatter(raw) : escapeHtml(raw === undefined || raw === null ? 'n/a' : raw);
        return \`<td class="\${className || ''}" data-key="\${key}" data-sort="\${escapeHtml(raw === undefined || raw === null ? '' : raw)}">\${html}</td>\`;
      }).join('')}</tr>\`).join('');
      return \`\${toolbar}<div class="table-wrap\${compact ? ' compact' : ''}"><table id="\${id}"><thead><tr>\${head}</tr></thead><tbody>\${body}</tbody></table></div>\`;
    }

    window.sortTable = function(id, key) {
      const table = document.getElementById(id);
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.rows);
      const asc = table.dataset.sortKey === key ? table.dataset.sortDir !== 'asc' : true;
      rows.sort((a, b) => {
        const av = a.querySelector(\`td[data-key="\${key}"]\`).dataset.sort;
        const bv = b.querySelector(\`td[data-key="\${key}"]\`).dataset.sort;
        const an = Number(av), bn = Number(bv);
        const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
        return asc ? cmp : -cmp;
      });
      table.dataset.sortKey = key;
      table.dataset.sortDir = asc ? 'asc' : 'desc';
      rows.forEach(row => tbody.appendChild(row));
    };

    window.filterTable = function(id, value) {
      const needle = value.toLowerCase();
      document.querySelectorAll('#' + id + ' tbody tr').forEach(row => {
        row.style.display = row.textContent.toLowerCase().includes(needle) ? '' : 'none';
      });
    };

    function escapeHtml(value) {
      return String(value).replace(/[&<>"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
    }

    renderOverview();
    renderSimulation();
    renderSolver();
    renderGeneration();
    renderCoverage();
    renderProfiling();
    showView((location.hash || '#overview').slice(1));
  </script>
</body>
</html>
`;
}

function copyDir(source, destination) {
  rmDir(destination);
  ensureDir(path.dirname(destination));
  fs.cpSync(source, destination, { recursive: true });
}

function main() {
  const options = parseArgs(process.argv);
  const startedAt = new Date();
  const latestDir = path.join(options.outRoot, 'latest');
  rmDir(latestDir);
  ensureDir(latestDir);
  const context = {
    startedAt,
    outRoot: options.outRoot,
    latestDir,
    logsDir: path.join(latestDir, 'logs'),
    artifactsDir: path.join(latestDir, 'artifacts'),
    warnings: [],
    commands: [],
  };
  ensureDir(context.logsDir);
  ensureDir(context.artifactsDir);

  const report = {
    schema_version: 1,
    metadata: collectMetadata(context),
    inputs: collectInputs(options),
    suites: {},
    warnings: context.warnings,
    commands: context.commands,
  };

  report.suites.simulation = runSimulationSuite(context, options);
  report.suites.solver = runSolverSuite(context, options);
  report.suites.generation = runGenerationSuite(context, options);
  report.suites.coverage = runCoverageSuite(context);
  report.suites.profiling = runProfilingSuite(context, options, report.suites.simulation);
  buildOverview(report);

  const runName = `${timestampForPath(startedAt)}-${report.metadata.git_short_sha || 'unknown'}`;
  const runDir = path.join(options.outRoot, 'runs', runName);
  const summary = renderMarkdown(report, latestDir, runDir);
  writeJson(path.join(latestDir, 'report.json'), report);
  fs.writeFileSync(path.join(latestDir, 'summary.md'), summary, 'utf8');
  fs.writeFileSync(path.join(latestDir, 'index.html'), renderHtml(report), 'utf8');
  copyDir(latestDir, runDir);

  process.stdout.write(`performance_testpage wrote ${path.join(latestDir, 'index.html')}\n`);
  process.stdout.write(`performance_testpage wrote ${path.join(runDir, 'index.html')}\n`);
  if (context.warnings.length > 0) {
    process.stdout.write(`performance_testpage warnings=${context.warnings.length}\n`);
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
}
