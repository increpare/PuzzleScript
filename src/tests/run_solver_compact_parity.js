#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

function usage() {
    console.error('Usage: node src/tests/run_solver_compact_parity.js <puzzlescript_solver> <solver_tests_dir> [--timeout-ms N] [--strategy NAME] [--game NAME] [--level N] [--max-games N] [--compact-turn-oracle] [--require-compact-oracle-checks] [--require-compact-handled]');
    process.exit(1);
}

const args = process.argv.slice(2);
if (args.length < 2) {
    usage();
}

const solverPath = path.resolve(args[0]);
const corpusPath = path.resolve(args[1]);
let timeoutMs = 1000;
let strategy = 'bfs';
let targetGame = null;
let targetLevel = null;
let maxGames = null;
let compactTurnOracle = false;
let requireCompactOracleChecks = false;
let requireCompactHandled = false;

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

for (let index = 2; index < args.length; index++) {
    const arg = args[index];
    if (arg === '--timeout-ms' && index + 1 < args.length) {
        timeoutMs = parsePositiveInt(args[++index], '--timeout-ms');
    } else if (arg === '--strategy' && index + 1 < args.length) {
        strategy = args[++index];
    } else if (arg === '--game' && index + 1 < args.length) {
        targetGame = args[++index];
    } else if (arg === '--level' && index + 1 < args.length) {
        targetLevel = parseNonNegativeInt(args[++index], '--level');
    } else if (arg === '--max-games' && index + 1 < args.length) {
        maxGames = parsePositiveInt(args[++index], '--max-games');
    } else if (arg === '--compact-turn-oracle' || arg === '--compact-tick-oracle') {
        compactTurnOracle = true;
    } else if (arg === '--require-compact-oracle-checks') {
        requireCompactOracleChecks = true;
    } else if (arg === '--require-compact-handled') {
        requireCompactHandled = true;
    } else {
        usage();
    }
}

function walkTxtFiles(root) {
    const out = [];
    const stack = [root];
    while (stack.length > 0) {
        const current = stack.pop();
        for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
            const full = path.join(current, entry.name);
            if (entry.isDirectory()) {
                stack.push(full);
            } else if (entry.isFile() && entry.name.endsWith('.txt')) {
                out.push(full);
            }
        }
    }
    return out.sort((a, b) => a.localeCompare(b));
}

function normalizedSectionName(line) {
    return line.replace(/=/g, '').trim().toUpperCase();
}

function randomRuleHits(gameFile) {
    const knownSections = new Set([
        'OBJECTS',
        'LEGEND',
        'SOUNDS',
        'COLLISIONLAYERS',
        'RULES',
        'WINCONDITIONS',
        'LEVELS',
    ]);
    const lines = fs.readFileSync(gameFile, 'utf8').split(/\r?\n/);
    const hits = [];
    let inRules = false;
    for (let index = 0; index < lines.length; index++) {
        const line = lines[index];
        const trimmed = line.trim();
        const section = normalizedSectionName(line);
        if (knownSections.has(section)) {
            inRules = section === 'RULES';
            continue;
        }
        if (!inRules || trimmed.length === 0 || trimmed.startsWith('(')) {
            continue;
        }
        if (/\brandom(?:dir)?\b/i.test(line)) {
            hits.push(index + 1);
        }
    }
    return hits;
}

function runSolver(gameName, compact) {
    const solverArgs = [
        corpusPath,
        '--timeout-ms', String(timeoutMs),
        '--jobs', '1',
        '--strategy', strategy,
        '--game', gameName,
        '--no-solutions',
        '--quiet',
        '--json',
    ];
    if (targetLevel !== null) {
        solverArgs.push('--level', String(targetLevel));
    }
    if (compact) {
        solverArgs.push('--compact-node-storage');
        if (compactTurnOracle) {
            solverArgs.push('--compact-turn-oracle');
        }
    }
    const result = spawnSync(solverPath, solverArgs, {
        encoding: 'utf8',
        maxBuffer: 512 * 1024 * 1024,
    });
    if (result.error) {
        throw result.error;
    }
    if (result.status !== 0) {
        throw new Error(`solver exited ${result.status} game=${gameName} compact=${compact}\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
    }
    return JSON.parse(result.stdout);
}

function resultKey(result) {
    return `${result.game}#${result.level}`;
}

function compareResults(normalResult, compactResult) {
    if (normalResult.status === 'solved') {
        if (compactResult.status === 'timeout') {
            return {
                fatal: false,
                message: 'normal solved but compact timed out',
            };
        }
        if (compactResult.status !== 'solved') {
            return {
                fatal: true,
                message: `normal solved but compact status=${compactResult.status}`,
            };
        }
        const normalSolution = normalResult.solution || [];
        const compactSolution = compactResult.solution || [];
        if (JSON.stringify(normalSolution) !== JSON.stringify(compactSolution)) {
            return {
                fatal: true,
                message: `solution normal=${JSON.stringify(normalSolution)} compact=${JSON.stringify(compactSolution)}`,
            };
        }
        return null;
    }
    if (normalResult.status === 'exhausted' || normalResult.status === 'skipped_message') {
        if (compactResult.status !== normalResult.status) {
            return {
                fatal: true,
                message: `normal status=${normalResult.status} compact status=${compactResult.status}`,
            };
        }
        return null;
    }
    if (normalResult.status === 'timeout') {
        if (compactResult.status === 'solved' || compactResult.status === 'timeout') {
            return null;
        }
        return {
            fatal: true,
            message: `normal timeout but compact status=${compactResult.status}`,
        };
    }
    if (compactResult.status !== normalResult.status) {
        return {
            fatal: true,
            message: `normal status=${normalResult.status} compact status=${compactResult.status}`,
        };
    }
    return null;
}

function total(json, field, oldField = null) {
    if (json.totals[field] !== undefined) return json.totals[field];
    if (oldField !== null && json.totals[oldField] !== undefined) return json.totals[oldField];
    return 0;
}

const allGames = walkTxtFiles(corpusPath);
const eligibleGames = [];
const randomExcluded = [];
for (const gameFile of allGames) {
    const hits = randomRuleHits(gameFile);
    if (hits.length > 0) {
        randomExcluded.push({
            game: path.relative(corpusPath, gameFile),
            hits,
        });
        continue;
    }
    eligibleGames.push(gameFile);
}
let candidateGames = eligibleGames;
if (targetGame !== null) {
    candidateGames = eligibleGames.filter((gameFile) => {
        const relative = path.relative(corpusPath, gameFile);
        return relative === targetGame || path.basename(relative) === targetGame;
    });
    if (candidateGames.length === 0) {
        throw new Error(`target game not found among non-random compact parity games: ${targetGame}`);
    }
}
const selectedGames = maxGames === null ? candidateGames : candidateGames.slice(0, maxGames);
if (selectedGames.length === 0) {
    throw new Error('no non-random games available for compact parity');
}

let totalLevels = 0;
let compactTimeoutRegressions = 0;
let compactOracleChecks = 0;
let compactOracleFailures = 0;
let compactAttempts = 0;
let compactHits = 0;
let compactUnhandled = 0;
for (let index = 0; index < selectedGames.length; index++) {
    const gameFile = selectedGames[index];
    const gameName = path.relative(corpusPath, gameFile);
    process.stderr.write(`solver_compact_parity ${index + 1}/${selectedGames.length} ${gameName}\n`);
    const normal = runSolver(gameName, false);
    const compact = runSolver(gameName, true);
    compactOracleChecks += total(compact, 'compact_turn_oracle_checks', 'compact_tick_oracle_checks');
    compactOracleFailures += total(compact, 'compact_turn_oracle_failures', 'compact_tick_oracle_failures');
    compactAttempts += total(compact, 'compact_turn_attempts');
    compactHits += total(compact, 'compact_turn_hits');
    compactUnhandled += total(compact, 'compact_turn_unhandled', 'compact_turn_fallbacks');
    if (total(compact, 'compact_turn_oracle_failures', 'compact_tick_oracle_failures') !== 0) {
        throw new Error(`compact turn oracle failures=${total(compact, 'compact_turn_oracle_failures', 'compact_tick_oracle_failures')} game=${gameName}`);
    }
    const normalByKey = new Map(normal.results.map((result) => [resultKey(result), result]));
    const compactByKey = new Map(compact.results.map((result) => [resultKey(result), result]));
    const mismatches = [];
    const warnings = [];
    for (const [key, normalResult] of normalByKey) {
        const compactResult = compactByKey.get(key);
        if (!compactResult) {
            mismatches.push(`${key}: missing compact result`);
            continue;
        }
        const mismatch = compareResults(normalResult, compactResult);
        if (mismatch !== null) {
            if (mismatch.fatal) {
                mismatches.push(`${key}: ${mismatch.message}`);
            } else {
                warnings.push(`${key}: ${mismatch.message}`);
            }
        }
    }
    for (const key of compactByKey.keys()) {
        if (!normalByKey.has(key)) {
            mismatches.push(`${key}: extra compact result`);
        }
    }
    if (mismatches.length > 0) {
        throw new Error(`compact parity mismatches=${mismatches.length} game=${gameName}\n${mismatches.slice(0, 20).join('\n')}`);
    }
    compactTimeoutRegressions += warnings.length;
    for (const warning of warnings.slice(0, 5)) {
        process.stderr.write(`solver_compact_parity warning ${warning}\n`);
    }
    totalLevels += normal.results.length;
}
if (requireCompactOracleChecks && compactOracleChecks === 0) {
    throw new Error('compact turn oracle checks were required but no generated compact turn was checked');
}
if (requireCompactHandled && compactAttempts === 0) {
    throw new Error('compact handled checks were required but no compact turn was attempted');
}
if (requireCompactHandled && compactUnhandled !== 0) {
    throw new Error(`compact handled checks were required but compact_turn_unhandled=${compactUnhandled}`);
}
process.stdout.write(
    `solver_compact_parity passed games=${selectedGames.length}/${eligibleGames.length}`
    + ` levels=${totalLevels} random_excluded=${randomExcluded.length}`
    + ` compact_timeout_regressions=${compactTimeoutRegressions}`
    + ` compact_turn_attempts=${compactAttempts} compact_turn_hits=${compactHits} compact_turn_unhandled=${compactUnhandled}`
    + (compactTurnOracle ? ` compact_turn_oracle_checks=${compactOracleChecks} compact_turn_oracle_failures=${compactOracleFailures}` : '')
    + ` strategy=${strategy} timeout_ms=${timeoutMs}\n`
);
