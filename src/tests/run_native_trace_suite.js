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
        throw new Error('Missing fixtures manifest path');
    }
    if (result.cliPath === null) {
        result.cliPath = path.resolve('build/native/native/ps_cli');
    }
    return result;
}

function runCommand(command, args, options = {}) {
    return spawnSync(command, args, {
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024,
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

function firstNonEmptyLine(text) {
    if (!text) {
        return '';
    }
    return text.split(/\r?\n/).find(line => line.trim().length > 0) || '';
}

function main() {
    const options = parseArgs(process.argv);
    const manifestDir = path.dirname(options.manifestPath);
    const manifest = JSON.parse(fs.readFileSync(options.manifestPath, 'utf8'));
    const fixtures = manifest.simulation_fixtures || [];

    process.stderr.write(`prepared_checks starting timeout_ms=${options.preparedTimeoutMs}\n`);
    const preparedRun = runCommand(options.cliPath, ['test-fixtures', options.manifestPath], {
        timeout: options.preparedTimeoutMs,
    });

    if (preparedRun.error) {
        throw preparedRun.error;
    }
    printStream(preparedRun.stdout);
    if (preparedRun.stderr && !options.quiet) {
        process.stderr.write(preparedRun.stderr);
    }
    if (preparedRun.status !== 0) {
        process.exit(preparedRun.status ?? 1);
    }
    process.stderr.write(`prepared_checks finished fixture_count=${fixtures.length}\n`);
    process.stderr.write(`trace_sweep starting timeout_ms=${options.timeoutMs}\n`);

    let traceChecked = 0;
    let tracePassed = 0;
    let traceFailed = 0;
    let traceTimedOut = 0;

    for (let index = 0; index < fixtures.length; index++) {
        const fixture = fixtures[index];
        if (!fixture.trace_file) {
            continue;
        }

        traceChecked += 1;
        const irPath = path.join(manifestDir, fixture.ir_file);
        const tracePath = path.join(manifestDir, fixture.trace_file);
        const startedAt = Date.now();

        // Phase 1: fast native-only check (no per-snapshot diagnostic export).
        const fastRun = runCommand(options.cliPath, ['check-trace', irPath, tracePath], {
            timeout: options.timeoutMs,
        });
        const fastElapsedMs = Date.now() - startedAt;

        let outcome = 'passed';
        let run = fastRun;
        let elapsedMs = fastElapsedMs;

        const fastPassed = !fastRun.error && fastRun.status === 0;
        if (fastPassed) {
            tracePassed += 1;
        } else {
            // Phase 2: detailed diff only on failure.
            const detailedStartedAt = Date.now();
            const detailedRun = runCommand(options.cliPath, ['diff-trace', irPath, tracePath], {
                timeout: options.timeoutMs,
            });
            run = detailedRun;
            elapsedMs = Date.now() - detailedStartedAt;

            if (detailedRun.error && detailedRun.error.code === 'ETIMEDOUT') {
                traceTimedOut += 1;
                outcome = 'timed_out';
            } else if (detailedRun.error) {
                traceFailed += 1;
                outcome = 'error';
            } else if (detailedRun.status === 0) {
                tracePassed += 1;
            } else {
                traceFailed += 1;
                outcome = 'failed';
            }
        }

        if (options.progressEvery > 0 && (traceChecked % options.progressEvery) === 0) {
            const progressLine = [
                `trace_progress checked=${traceChecked}`,
                `passed=${tracePassed}`,
                `failed=${traceFailed}`,
                `timed_out=${traceTimedOut}`,
                `current_fixture=${fixture.name}`,
                `outcome=${outcome}`,
                `elapsed_ms=${elapsedMs}`,
            ].join(' ');
            process.stderr.write(`${progressLine}\n`);
        }

        if (outcome !== 'passed' && !options.quiet) {
            const detail = run.error
                ? String(run.error.message || run.error)
                : firstNonEmptyLine(run.stderr) || firstNonEmptyLine(run.stdout);
            process.stderr.write(`${fixture.name}: ${outcome}${detail ? `: ${detail}` : ''}\n`);
        }
    }

    const summary = [
        `trace_replay_checked=${traceChecked}`,
        `trace_replay_passed=${tracePassed}`,
        `trace_replay_failed=${traceFailed}`,
        `trace_replay_timed_out=${traceTimedOut}`,
    ].join(' ');
    process.stdout.write(`${summary}\n`);

    if (!options.allowFailures && (traceFailed > 0 || traceTimedOut > 0)) {
        process.exit(1);
    }
}

main();
