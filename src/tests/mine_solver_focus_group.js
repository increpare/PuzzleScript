#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

function usage() {
    console.error([
        'Usage: node src/tests/mine_solver_focus_group.js <puzzlescript_solver> <solver_tests_dir>',
        '  [--timeout-ms N] [--min-elapsed-ms N] [--max-targets N] [--out PATH]',
        '  [--strategy NAME] [--jobs N] [--exclude-game NAME] [--exclude-games CSV]',
        '  [--repo-root PATH] [--puzzlescript-cpp PATH] [--compile-probe-root PATH]',
        '  [--compile-timeout-seconds N] [--compile-max-rows N]',
        '  [--compile-max-compiled-rules-per-source N]',
        '  [--compile-max-generated-lines-per-source N] [--cmake PATH]',
        '  [--cmake-generator NAME] [--compile-opt-level N] [--compile-build-jobs N|auto]',
    ].join('\n'));
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const solverPath = path.resolve(args[0]);
const corpusPath = path.resolve(args[1]);
let timeoutMs = 500;
let minElapsedMs = 250;
let maxTargets = 50;
let outPath = path.resolve('build/native/solver_focus_group.json');
let strategy = 'portfolio';
let jobs = '1';
let repoRoot = process.cwd();
let puzzlescriptCpp = null;
let compileProbeRoot = path.resolve('build/native/solver_focus_compile_probes');
let compileTimeoutSeconds = 60;
let compileMaxRows = 99;
let compileMaxCompiledRulesPerSource = null;
let compileMaxGeneratedLinesPerSource = null;
let cmakePath = 'cmake';
let cmakeGenerator = '';
let compileOptLevel = '1';
let compileBuildJobs = 'auto';
const excludedGames = new Set();

function parsePositiveInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(`${label} must be a positive integer: ${value}`);
    }
    return parsed;
}

function parseNonNegativeInt(value, label) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`${label} must be a non-negative integer: ${value}`);
    }
    return parsed;
}

function parseNonNegativeNumber(value, label) {
    const parsed = Number.parseFloat(value);
    if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`${label} must be a non-negative number: ${value}`);
    }
    return parsed;
}

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--timeout-ms' && index + 1 < args.length) {
        timeoutMs = parsePositiveInt(args[++index], '--timeout-ms');
    } else if (arg === '--min-elapsed-ms' && index + 1 < args.length) {
        minElapsedMs = parseNonNegativeInt(args[++index], '--min-elapsed-ms');
    } else if (arg === '--max-targets' && index + 1 < args.length) {
        maxTargets = parsePositiveInt(args[++index], '--max-targets');
    } else if (arg === '--out' && index + 1 < args.length) {
        outPath = path.resolve(args[++index]);
    } else if (arg === '--strategy' && index + 1 < args.length) {
        strategy = args[++index];
    } else if (arg === '--jobs' && index + 1 < args.length) {
        jobs = args[++index];
    } else if (arg === '--exclude-game' && index + 1 < args.length) {
        const game = args[++index].trim();
        if (game.length > 0) {
            excludedGames.add(game);
        }
    } else if (arg === '--exclude-games' && index + 1 < args.length) {
        for (const game of args[++index].split(',')) {
            const trimmed = game.trim();
            if (trimmed.length > 0) {
                excludedGames.add(trimmed);
            }
        }
    } else if (arg === '--repo-root' && index + 1 < args.length) {
        repoRoot = path.resolve(args[++index]);
    } else if (arg === '--puzzlescript-cpp' && index + 1 < args.length) {
        puzzlescriptCpp = path.resolve(args[++index]);
    } else if (arg === '--compile-probe-root' && index + 1 < args.length) {
        compileProbeRoot = path.resolve(args[++index]);
    } else if (arg === '--compile-timeout-seconds' && index + 1 < args.length) {
        compileTimeoutSeconds = parseNonNegativeNumber(args[++index], '--compile-timeout-seconds');
    } else if (arg === '--compile-max-rows' && index + 1 < args.length) {
        compileMaxRows = parsePositiveInt(args[++index], '--compile-max-rows');
    } else if (arg === '--compile-max-compiled-rules-per-source' && index + 1 < args.length) {
        compileMaxCompiledRulesPerSource = parseNonNegativeInt(args[++index], '--compile-max-compiled-rules-per-source');
    } else if (arg === '--compile-max-generated-lines-per-source' && index + 1 < args.length) {
        compileMaxGeneratedLinesPerSource = parseNonNegativeInt(args[++index], '--compile-max-generated-lines-per-source');
    } else if (arg === '--cmake' && index + 1 < args.length) {
        cmakePath = args[++index];
    } else if (arg === '--cmake-generator' && index + 1 < args.length) {
        cmakeGenerator = args[++index];
    } else if (arg === '--compile-opt-level' && index + 1 < args.length) {
        compileOptLevel = args[++index];
    } else if (arg === '--compile-build-jobs' && index + 1 < args.length) {
        compileBuildJobs = args[++index];
    } else {
        throw new Error(`Unsupported argument: ${arg}`);
    }
}

function resultKey(result) {
    return `${result.game}#${result.level}`;
}

function normalizeGamePath(filePath) {
    return path.relative(corpusPath, filePath).split(path.sep).join('/');
}

function listGameFiles(dir) {
    const files = [];
    function visit(current) {
        for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
            const fullPath = path.join(current, entry.name);
            if (entry.isDirectory()) {
                visit(fullPath);
            } else if (entry.isFile() && entry.name.endsWith('.txt')) {
                files.push(fullPath);
            }
        }
    }
    visit(dir);
    return files.sort((a, b) => normalizeGamePath(a).localeCompare(normalizeGamePath(b)));
}

function sha256Parts(parts) {
    const hash = crypto.createHash('sha256');
    for (const part of parts) {
        hash.update(part);
    }
    return hash.digest('hex');
}

function safeSnippet(text) {
    return text.trim().split('\n').slice(-8).join('\n').slice(0, 4000);
}

function runCommand(command, commandArgs, options = {}) {
    const started = process.hrtime.bigint();
    const result = spawnSync(command, commandArgs, {
        cwd: options.cwd || repoRoot,
        encoding: 'utf8',
        maxBuffer: 64 * 1024 * 1024,
        timeout: options.timeoutMs,
    });
    const wallMs = Number(process.hrtime.bigint() - started) / 1e6;
    return {
        command,
        args: commandArgs,
        status: result.status,
        signal: result.signal,
        error: result.error ? result.error.message : null,
        error_code: result.error ? result.error.code : null,
        stdout: result.stdout || '',
        stderr: result.stderr || '',
        wall_ms: wallMs,
    };
}

function copyGameToEligibleCorpus(gameFile, eligibleCorpus) {
    const relative = normalizeGamePath(gameFile);
    const destination = path.join(eligibleCorpus, ...relative.split('/'));
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(gameFile, destination);
}

function normalizedSectionName(line) {
    return line.replace(/=/g, '').trim().toUpperCase();
}

function isRulesSectionName(name) {
    return name === 'RULES';
}

function isKnownSectionName(name) {
    return [
        'OBJECTS',
        'LEGEND',
        'SOUNDS',
        'COLLISIONLAYERS',
        'RULES',
        'WINCONDITIONS',
        'LEVELS',
    ].includes(name);
}

function randomRuleHits(gameFile) {
    const lines = fs.readFileSync(gameFile, 'utf8').split(/\r?\n/);
    const hits = [];
    let inRules = false;
    for (let index = 0; index < lines.length; index++) {
        const line = lines[index];
        const trimmed = line.trim();
        const section = normalizedSectionName(line);
        if (isKnownSectionName(section)) {
            inRules = isRulesSectionName(section);
            continue;
        }
        if (!inRules || trimmed.length === 0 || trimmed.startsWith('(')) {
            continue;
        }
        if (/\brandom(?:dir)?\b/i.test(line)) {
            hits.push({
                line: index + 1,
                text: trimmed,
            });
        }
    }
    return hits;
}

function probeGameCompilation(gameFile, gameName, probeRoot) {
    const content = fs.readFileSync(gameFile);
    const hash = sha256Parts([
        `${gameName}\nrows=${compileMaxRows}\ncap=${compileMaxCompiledRulesPerSource ?? ''}\nopt=${compileOptLevel}\ngenerator=${cmakeGenerator}\n`,
        `line_cap=${compileMaxGeneratedLinesPerSource ?? ''}\n`,
        content,
    ]);
    const shortHash = hash.slice(0, 16);
    const gameProbeRoot = path.join(probeRoot, shortHash);
    const outCpp = path.join(gameProbeRoot, 'compiled_rules.cpp');
    const buildDir = path.join(gameProbeRoot, 'build');
    fs.mkdirSync(gameProbeRoot, { recursive: true });

    const compileRulesArgs = [
        'compile-rules',
        gameFile,
        '--emit-cpp',
        outCpp,
        '--symbol',
        `solver_focus_probe_${shortHash}`,
        '--max-rows',
        String(compileMaxRows),
    ];
    if (compileMaxCompiledRulesPerSource !== null) {
        compileRulesArgs.push('--max-compiled-rules-per-source', String(compileMaxCompiledRulesPerSource));
    }
    if (compileMaxGeneratedLinesPerSource !== null) {
        compileRulesArgs.push('--max-generated-lines-per-source', String(compileMaxGeneratedLinesPerSource));
    }

    const started = process.hrtime.bigint();
    const remainingTimeoutMs = () => {
        if (compileTimeoutSeconds === 0) {
            return undefined;
        }
        const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
        return Math.max(1, Math.ceil(compileTimeoutSeconds * 1000 - elapsedMs));
    };

    const emitResult = runCommand(puzzlescriptCpp, compileRulesArgs, { timeoutMs: remainingTimeoutMs() });
    if (emitResult.error_code === 'ETIMEDOUT') {
        return {
            game: gameName,
            status: 'timeout',
            compile_seconds: Number(process.hrtime.bigint() - started) / 1e9,
            threshold_seconds: compileTimeoutSeconds,
            max_rows: compileMaxRows,
            max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
            max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
            reason: 'compile-rules generation exceeded focus compile budget',
            detail: safeSnippet(`${emitResult.stderr}\n${emitResult.stdout}`),
        };
    }
    if (emitResult.error || emitResult.status !== 0) {
        return {
            game: gameName,
            status: 'compile_rules_failed',
            compile_seconds: Number(process.hrtime.bigint() - started) / 1e9,
            threshold_seconds: compileTimeoutSeconds,
            max_rows: compileMaxRows,
            max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
            max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
            reason: 'compile-rules failed',
            detail: safeSnippet(`${emitResult.stderr}\n${emitResult.stdout}`),
        };
    }
    if (/compiled-rules(?:-line)?-skips:/.test(emitResult.stderr)) {
        return {
            game: gameName,
            status: 'compile_budget_excluded',
            compile_seconds: Number(process.hrtime.bigint() - started) / 1e9,
            threshold_seconds: compileTimeoutSeconds,
            max_rows: compileMaxRows,
            max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
            max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
            reason: 'compiled-rules generation exceeded focus source budget',
            detail: safeSnippet(emitResult.stderr),
        };
    }

    const configureArgs = [
        '-S', repoRoot,
        '-B', buildDir,
    ];
    if (cmakeGenerator) {
        configureArgs.push('-G', cmakeGenerator);
    }
    configureArgs.push(
        '-DPS_MASK_WORD_BITS=64',
        '-DPS_ENABLE_LTO=false',
        '-DPS_ENABLE_LINK_DEDUP=false',
        '-DPS_ENABLE_EXPORTED_SYMBOLS=false',
        `-DPS_COMPILED_RULES_OPT_LEVEL=${compileOptLevel}`,
        `-DPS_COMPILED_RULES_SOURCE=${outCpp}`,
        '-DPS_COMPILED_RULES_SOURCES_FILE='
    );

    const configureResult = runCommand(cmakePath, configureArgs, { timeoutMs: remainingTimeoutMs() });
    if (configureResult.error_code === 'ETIMEDOUT') {
        return {
            game: gameName,
            status: 'timeout',
            compile_seconds: Number(process.hrtime.bigint() - started) / 1e9,
            threshold_seconds: compileTimeoutSeconds,
            max_rows: compileMaxRows,
            max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
            max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
            reason: 'cmake configure exceeded focus compile budget',
            detail: safeSnippet(`${configureResult.stderr}\n${configureResult.stdout}`),
        };
    }
    if (configureResult.error || configureResult.status !== 0) {
        return {
            game: gameName,
            status: 'configure_failed',
            compile_seconds: Number(process.hrtime.bigint() - started) / 1e9,
            threshold_seconds: compileTimeoutSeconds,
            max_rows: compileMaxRows,
            max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
            max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
            reason: 'cmake configure failed',
            detail: safeSnippet(`${configureResult.stderr}\n${configureResult.stdout}`),
        };
    }

    const buildArgs = [
        path.join(__dirname, 'run_with_timeout.js'),
        String(Math.max(0.001, remainingTimeoutMs() / 1000)),
        '--',
        cmakePath,
        '--build',
        buildDir,
    ];
    if (compileBuildJobs === 'auto') {
        buildArgs.push('--parallel');
    } else if (compileBuildJobs && compileBuildJobs !== '1') {
        buildArgs.push('--parallel', String(compileBuildJobs));
    }
    buildArgs.push('--target', 'puzzlescript_solver');

    const buildResult = runCommand(process.execPath, buildArgs);
    const compileSeconds = Number(process.hrtime.bigint() - started) / 1e9;
    if (buildResult.status === 124) {
        return {
            game: gameName,
            status: 'timeout',
            compile_seconds: compileSeconds,
            threshold_seconds: compileTimeoutSeconds,
            max_rows: compileMaxRows,
            max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
            max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
            reason: 'compiled-rules build exceeded focus compile budget',
            detail: safeSnippet(`${buildResult.stderr}\n${buildResult.stdout}`),
        };
    }
    if (buildResult.error || buildResult.status !== 0) {
        return {
            game: gameName,
            status: 'build_failed',
            compile_seconds: compileSeconds,
            threshold_seconds: compileTimeoutSeconds,
            max_rows: compileMaxRows,
            max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
            max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
            reason: 'compiled-rules build failed',
            detail: safeSnippet(`${buildResult.stderr}\n${buildResult.stdout}`),
        };
    }

    return {
        game: gameName,
        status: 'compiled',
        compile_seconds: compileSeconds,
        threshold_seconds: compileTimeoutSeconds,
        max_rows: compileMaxRows,
        max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
        max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
    };
}

function prepareEligibleCorpus() {
    const gameFiles = listGameFiles(corpusPath);
    const probeRoot = compileProbeRoot;
    const eligibleCorpus = path.join(probeRoot, 'eligible-corpus');
    fs.rmSync(probeRoot, { recursive: true, force: true });
    fs.mkdirSync(eligibleCorpus, { recursive: true });

    const compileProbeResults = [];
    const compileExcludedGames = [];
    const randomExcludedGames = [];
    let eligibleCount = 0;

    if (compileTimeoutSeconds === 0) {
        for (const gameFile of gameFiles) {
            const gameName = normalizeGamePath(gameFile);
            if (excludedGames.has(gameName)) {
                continue;
            }
            const randomHits = randomRuleHits(gameFile);
            if (randomHits.length > 0) {
                randomExcludedGames.push({
                    game: gameName,
                    status: 'random_rules_excluded',
                    reason: 'focus mining excludes random/randomDir rules',
                    hits: randomHits,
                });
                continue;
            }
            copyGameToEligibleCorpus(gameFile, eligibleCorpus);
            eligibleCount++;
        }
        return {
            enabled: false,
            corpus: eligibleCorpus,
            game_count: gameFiles.length,
            eligible_game_count: eligibleCount,
            compile_probe_results: [],
            compile_excluded_games: [],
            random_excluded_games: randomExcludedGames,
        };
    }

    if (!puzzlescriptCpp) {
        throw new Error('--puzzlescript-cpp is required when --compile-timeout-seconds is non-zero');
    }

    process.stdout.write(`solver_focus_mine compile probe games=${gameFiles.length} timeout_seconds=${compileTimeoutSeconds} max_rows=${compileMaxRows}\n`);
    for (const gameFile of gameFiles) {
        const gameName = normalizeGamePath(gameFile);
        if (excludedGames.has(gameName)) {
            continue;
        }

        const randomHits = randomRuleHits(gameFile);
        if (randomHits.length > 0) {
            const result = {
                game: gameName,
                status: 'random_rules_excluded',
                reason: 'focus mining excludes random/randomDir rules',
                hits: randomHits,
            };
            randomExcludedGames.push(result);
            process.stdout.write(`  exclude ${gameName} status=${result.status} hits=${randomHits.length}\n`);
            continue;
        }

        const result = probeGameCompilation(gameFile, gameName, probeRoot);
        compileProbeResults.push(result);
        if (result.status === 'compiled') {
            copyGameToEligibleCorpus(gameFile, eligibleCorpus);
            eligibleCount++;
            process.stdout.write(`  ok ${gameName} compile_seconds=${result.compile_seconds.toFixed(2)}\n`);
        } else {
            compileExcludedGames.push(result);
            process.stdout.write(`  exclude ${gameName} status=${result.status} compile_seconds=${result.compile_seconds.toFixed(2)}\n`);
        }
    }

    return {
        enabled: true,
        corpus: eligibleCorpus,
        game_count: gameFiles.length,
        eligible_game_count: eligibleCount,
        compile_probe_results: compileProbeResults,
        compile_excluded_games: compileExcludedGames,
        random_excluded_games: randomExcludedGames,
    };
}

const preparedCorpus = prepareEligibleCorpus();
const solverCorpusPath = preparedCorpus.corpus;
const commandArgs = [
    solverCorpusPath,
    '--timeout-ms', String(timeoutMs),
    '--jobs', jobs,
    '--strategy', strategy,
    '--no-solutions',
    '--quiet',
    '--json',
];

const started = process.hrtime.bigint();
const result = spawnSync(solverPath, commandArgs, {
    encoding: 'utf8',
    maxBuffer: 512 * 1024 * 1024,
});
const wallMs = Number(process.hrtime.bigint() - started) / 1e6;
if (result.error) {
    throw result.error;
}
if (result.status !== 0) {
    throw new Error(`solver exited ${result.status}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
}

const json = JSON.parse(result.stdout);
if (json.totals.errors !== 0) {
    throw new Error(`solver reported errors=${json.totals.errors}`);
}

const candidates = json.results
    .filter((entry) => entry.status === 'solved')
    .filter((entry) => !excludedGames.has(entry.game))
    .filter((entry) => entry.elapsed_ms >= minElapsedMs && entry.elapsed_ms <= timeoutMs)
    .sort((a, b) => {
        if (a.elapsed_ms !== b.elapsed_ms) return a.elapsed_ms - b.elapsed_ms;
        return resultKey(a).localeCompare(resultKey(b));
    });

const selectedTargets = candidates.slice(0, maxTargets).map((entry) => ({
    game: entry.game,
    level: entry.level,
    first_solved_timeout_ms: timeoutMs,
    previous_timeout_ms: minElapsedMs,
    previous_status: 'above_focus_min_elapsed',
    solved_elapsed_ms: entry.elapsed_ms,
    solved_elapsed_ratio: timeoutMs > 0 ? entry.elapsed_ms / timeoutMs : 0,
    solved_expanded: entry.expanded,
    solved_generated: entry.generated,
    observations: [{
        timeout_ms: timeoutMs,
        status: entry.status,
        elapsed_ms: entry.elapsed_ms,
        expanded: entry.expanded,
        generated: entry.generated,
        unique_states: entry.unique_states,
        duplicates: entry.duplicates,
        max_frontier: entry.max_frontier,
        solution_length: entry.solution_length,
    }],
}));

const manifest = {
    schema_version: 1,
    kind: 'solver_focus_group',
    generated_at: new Date().toISOString(),
    solver: solverPath,
    corpus: corpusPath,
    mined_corpus: solverCorpusPath,
    strategy,
    jobs,
    timeout_ms: timeoutMs,
    min_elapsed_ms: minElapsedMs,
    max_targets: maxTargets,
    excluded_games: Array.from(excludedGames).sort(),
    compile_probe: {
        enabled: preparedCorpus.enabled,
        timeout_seconds: compileTimeoutSeconds,
        max_rows: compileMaxRows,
        max_compiled_rules_per_source: compileMaxCompiledRulesPerSource,
        max_generated_lines_per_source: compileMaxGeneratedLinesPerSource,
        opt_level: compileOptLevel,
        build_jobs: compileBuildJobs,
        cmake_generator: cmakeGenerator,
        root: compileProbeRoot,
        game_count: preparedCorpus.game_count,
        eligible_game_count: preparedCorpus.eligible_game_count,
        excluded_game_count: preparedCorpus.compile_excluded_games.length,
        random_excluded_game_count: preparedCorpus.random_excluded_games.length,
    },
    compile_excluded_games: preparedCorpus.compile_excluded_games,
    random_excluded_games: preparedCorpus.random_excluded_games,
    compile_probe_results: preparedCorpus.compile_probe_results,
    target_count: selectedTargets.length,
    candidate_count: candidates.length,
    totals: json.totals,
    wall_ms: wallMs,
    targets: selectedTargets,
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, `${JSON.stringify(manifest, null, 2)}\n`);
process.stdout.write(`solver_focus_mine wrote ${outPath} targets=${selectedTargets.length} candidates=${candidates.length} eligible_games=${preparedCorpus.eligible_game_count}/${preparedCorpus.game_count} random_excluded=${preparedCorpus.random_excluded_games.length} compile_excluded=${preparedCorpus.compile_excluded_games.length} wall_ms=${wallMs.toFixed(1)}\n`);
