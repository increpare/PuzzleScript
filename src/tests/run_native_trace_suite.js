#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

function parseArgs(argv) {
    const result = {
        manifestPath: null,
        cliPath: null,
        preparedTimeoutMs: 120000,
        timeoutMs: 45000,
        progressEvery: 1,
        allowFailures: false,
        quiet: false,
    };

    const args = argv.slice(2);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === '--cli' && index + 1 < args.length) {
            result.cliPath = path.resolve(args[++index]);
        } else if (arg === '--prepared-timeout-ms' && index + 1 < args.length) {
            result.preparedTimeoutMs = Number.parseInt(args[++index], 10);
        } else if (arg === '--timeout-ms' && index + 1 < args.length) {
            result.timeoutMs = Number.parseInt(args[++index], 10);
        } else if (arg === '--progress-every' && index + 1 < args.length) {
            result.progressEvery = Number.parseInt(args[++index], 10);
        } else if (arg === '--allow-failures') {
            result.allowFailures = true;
        } else if (arg === '--quiet') {
            result.quiet = true;
        } else if (result.manifestPath === null) {
            result.manifestPath = path.resolve(arg);
        } else {
            throw new Error(`Unexpected argument: ${arg}`);
        }
    }

    if (result.manifestPath === null) {
        throw new Error('Missing JS parity data manifest path');
    }
    if (result.cliPath === null) {
        result.cliPath = path.resolve('build/native/puzzlescript_cpp');
    }
    return result;
}

function runCommand(command, args, options = {}) {
    return spawnSync(command, args, {
        encoding: 'utf8',
        maxBuffer: 64 * 1024 * 1024,
        ...options,
    });
}

function printStream(stream) {
    if (stream && stream.length > 0) {
        process.stdout.write(stream);
        if (!stream.endsWith('\n')) {
            process.stdout.write('\n');
        }
    }
}

function main() {
    const suiteStartedAt = Date.now();
    const options = parseArgs(process.argv);
    const manifest = JSON.parse(fs.readFileSync(options.manifestPath, 'utf8'));
    const fixtures = manifest.simulation_fixtures || [];
    const traceFixtureCount = fixtures.filter(f => f.trace_file).length;
    const sweepBudgetMs = Math.max(options.timeoutMs, options.timeoutMs * traceFixtureCount);
    const cliTimeoutMs = Math.max(options.preparedTimeoutMs, sweepBudgetMs);

    process.stderr.write(
        `native_trace_suite starting (single puzzlescript_cpp) timeout_ms=${cliTimeoutMs} ` +
            `(prepared_budget_ms=${options.preparedTimeoutMs} per_fixture_budget_ms=${options.timeoutMs} ` +
            `saved_replay_cases=${traceFixtureCount})\n`,
    );

    const cliStartedAt = Date.now();
    const sweepArgs = ['check-js-parity-data', options.manifestPath];
    if (options.progressEvery > 0) {
        sweepArgs.push('--progress-every', String(options.progressEvery));
    }
    if (options.allowFailures) {
        sweepArgs.push('--allow-failures');
    }
    if (options.quiet) {
        sweepArgs.push('--quiet');
    }
    sweepArgs.push('--profile-timers');

    const sweepRun = runCommand(options.cliPath, sweepArgs, {
        timeout: cliTimeoutMs,
    });

    if (sweepRun.error) {
        throw sweepRun.error;
    }
    printStream(sweepRun.stdout);
    if (sweepRun.stderr) {
        process.stderr.write(sweepRun.stderr);
    }

    const cliElapsedMs = Date.now() - cliStartedAt;
    const suiteElapsedMs = Date.now() - suiteStartedAt;
    process.stdout.write(
        [
            'native_trace_suite_timing',
            `cli_elapsed_ms=${cliElapsedMs}`,
            `total_elapsed_ms=${suiteElapsedMs}`,
        ].join(' ') + '\n',
    );

    if (!options.allowFailures && sweepRun.status !== 0) {
        process.exit(sweepRun.status ?? 1);
    }
}

main();
