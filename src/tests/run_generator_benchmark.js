#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

function usage() {
  return [
    'Usage: run_generator_benchmark.js <puzzlescript_generator> <game.txt> [options]',
    '',
    'Options:',
    '  --presets-dir PATH        Directory of .gen presets',
    '  --samples N               Samples per run (default: 200)',
    '  --runs N                  Runs per preset (default: 3)',
    '  --jobs N|auto             Generator jobs (default: 1)',
    '  --seed N                  Fixed seed for every run (default: 11)',
    '  --solver-timeout-ms N     Solver timeout per sample (default: 50)',
    '  --solver-strategy NAME    portfolio|bfs|weighted-astar|greedy (default: portfolio)',
    '  --top-k N                 Retained top candidates (default: 10)',
    '  --out PATH                Write benchmark JSON to PATH',
  ].join('\n');
}

function parseArgs(argv) {
  if (argv.length < 4) {
    throw new Error(usage());
  }
  const options = {
    generatorPath: path.resolve(argv[2]),
    gamePath: path.resolve(argv[3]),
    presetsDir: path.resolve(__dirname, 'generator_presets'),
    samples: 200,
    runs: 3,
    jobs: '1',
    seed: '11',
    solverTimeoutMs: 50,
    solverStrategy: 'portfolio',
    topK: 10,
    outPath: null,
  };
  for (let i = 4; i < argv.length; ++i) {
    const arg = argv[i];
    if (arg === '--presets-dir' && i + 1 < argv.length) options.presetsDir = path.resolve(argv[++i]);
    else if (arg === '--samples' && i + 1 < argv.length) options.samples = Number.parseInt(argv[++i], 10);
    else if (arg === '--runs' && i + 1 < argv.length) options.runs = Number.parseInt(argv[++i], 10);
    else if (arg === '--jobs' && i + 1 < argv.length) options.jobs = argv[++i];
    else if (arg === '--seed' && i + 1 < argv.length) options.seed = argv[++i];
    else if (arg === '--solver-timeout-ms' && i + 1 < argv.length) options.solverTimeoutMs = Number.parseInt(argv[++i], 10);
    else if (arg === '--solver-strategy' && i + 1 < argv.length) options.solverStrategy = argv[++i];
    else if (arg === '--top-k' && i + 1 < argv.length) options.topK = Number.parseInt(argv[++i], 10);
    else if (arg === '--out' && i + 1 < argv.length) options.outPath = path.resolve(argv[++i]);
    else if (arg === '--help' || arg === '-h') options.help = true;
    else throw new Error(`Unexpected argument: ${arg}\n\n${usage()}`);
  }
  if (options.help) return options;
  for (const [name, value] of [['samples', options.samples], ['runs', options.runs], ['solver-timeout-ms', options.solverTimeoutMs], ['top-k', options.topK]]) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`${name} must be a positive integer`);
    }
  }
  return options;
}

function percentile(sortedValues, fraction) {
  if (sortedValues.length === 0) return 0;
  const index = Math.min(sortedValues.length - 1, Math.floor((sortedValues.length - 1) * fraction));
  return sortedValues[index];
}

function summary(values) {
  if (values.length === 0) {
    return { min: 0, p50: 0, p90: 0, max: 0, mean: 0 };
  }
  const sorted = values.slice().sort((a, b) => a - b);
  const sum = values.reduce((acc, value) => acc + value, 0);
  return {
    min: sorted[0],
    p50: percentile(sorted, 0.5),
    p90: percentile(sorted, 0.9),
    max: sorted[sorted.length - 1],
    mean: sum / values.length,
  };
}

function rate(count, elapsedMs) {
  return elapsedMs > 0 ? count / (elapsedMs / 1000) : 0;
}

function runGenerator(options, presetPath, outPath) {
  const args = [
    options.gamePath,
    presetPath,
    '--samples', String(options.samples),
    '--jobs', String(options.jobs),
    '--seed', String(options.seed),
    '--solver-timeout-ms', String(options.solverTimeoutMs),
    '--solver-strategy', options.solverStrategy,
    '--top-k', String(options.topK),
    '--quiet',
    '--json-out', outPath,
  ];
  const started = process.hrtime.bigint();
  const result = spawnSync(options.generatorPath, args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
  if (result.status !== 0) {
    throw new Error(`generator failed for ${path.basename(presetPath)} status=${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
  }
  const json = JSON.parse(fs.readFileSync(outPath, 'utf8'));
  return { elapsedMs, json };
}

function summarizePreset(presetName, runs) {
  const elapsedMs = runs.map(run => run.elapsed_ms);
  const samplesPerSecond = runs.map(run => run.samples_per_second);
  const validPerSecond = runs.map(run => run.valid_per_second);
  const solverExpanded = runs.map(run => run.solver_totals.expanded);
  const solverGenerated = runs.map(run => run.solver_totals.generated);
  const solverUniqueStates = runs.map(run => run.solver_totals.unique_states);
  const solverDuplicates = runs.map(run => run.solver_totals.duplicates);
  const topScores = runs.flatMap(run => run.top.map(candidate => candidate.difficulty_score));
  const topSolutionLengths = runs.flatMap(run => run.top.map(candidate => candidate.solution_length));
  const totals = runs.reduce((acc, run) => {
    for (const [key, value] of Object.entries(run.totals)) {
      acc[key] = (acc[key] || 0) + value;
    }
    return acc;
  }, {});

  return {
    preset: presetName,
    run_count: runs.length,
    elapsed_ms: summary(elapsedMs),
    samples_per_second: summary(samplesPerSecond),
    valid_per_second: summary(validPerSecond),
    totals,
    dedupe_rate: totals.valid_generated > 0 ? totals.deduped / totals.valid_generated : 0,
    solve_rate: totals.solver_searches > 0 ? totals.solved / totals.solver_searches : 0,
    solver_expanded: summary(solverExpanded),
    solver_generated: summary(solverGenerated),
    solver_unique_states: summary(solverUniqueStates),
    solver_duplicates: summary(solverDuplicates),
    top_score: summary(topScores),
    top_solution_length: summary(topSolutionLengths),
    runs,
  };
}

function main() {
  const options = parseArgs(process.argv);
  if (options.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  const presetFiles = fs.readdirSync(options.presetsDir)
    .filter(name => name.endsWith('.gen'))
    .sort();
  if (presetFiles.length === 0) {
    throw new Error(`No .gen presets found in ${options.presetsDir}`);
  }

  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'psgen-benchmark-'));
  const startedAt = new Date().toISOString();
  const presets = [];
  try {
    for (const presetFile of presetFiles) {
      const presetPath = path.join(options.presetsDir, presetFile);
      const runs = [];
      for (let runIndex = 0; runIndex < options.runs; ++runIndex) {
        const outPath = path.join(tempDir, `${presetFile}.${runIndex}.json`);
        const result = runGenerator(options, presetPath, outPath);
        const json = result.json;
        const run = {
          run_index: runIndex,
          elapsed_ms: result.elapsedMs,
          samples_per_second: rate(json.totals.samples_attempted, result.elapsedMs),
          valid_per_second: rate(json.totals.valid_generated, result.elapsedMs),
          totals: {
            ...json.totals,
            solver_searches: json.solver_totals ? json.solver_totals.searches : 0,
          },
          solver_totals: json.solver_totals || {},
          top: json.top || [],
        };
        runs.push(run);
        process.stderr.write(
          `generator_benchmark preset=${presetFile} run=${runIndex + 1}/${options.runs} ` +
          `samples_per_s=${run.samples_per_second.toFixed(2)} solved=${run.totals.solved}\n`
        );
      }
      presets.push(summarizePreset(presetFile, runs));
    }

    const output = {
      started_at: startedAt,
      finished_at: new Date().toISOString(),
      config: {
        generator: options.generatorPath,
        game: options.gamePath,
        presets_dir: options.presetsDir,
        samples: options.samples,
        runs: options.runs,
        jobs: options.jobs,
        seed: options.seed,
        solver_timeout_ms: options.solverTimeoutMs,
        solver_strategy: options.solverStrategy,
        top_k: options.topK,
      },
      presets,
    };
    const text = `${JSON.stringify(output, null, 2)}\n`;
    if (options.outPath) {
      fs.mkdirSync(path.dirname(options.outPath), { recursive: true });
      fs.writeFileSync(options.outPath, text, 'utf8');
      process.stdout.write(`generator_benchmark wrote ${options.outPath}\n`);
    } else {
      process.stdout.write(text);
    }
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exit(1);
}
