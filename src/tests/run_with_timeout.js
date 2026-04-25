#!/usr/bin/env node
'use strict';

const { spawn } = require('child_process');

function usage() {
    console.error('Usage: node src/tests/run_with_timeout.js <seconds> -- <command> [args...]');
    process.exit(2);
}

const args = process.argv.slice(2);
const separatorIndex = args.indexOf('--');
if (args.length < 3 || separatorIndex !== 1 || separatorIndex === args.length - 1) {
    usage();
}

const timeoutSeconds = Number.parseFloat(args[0]);
if (!Number.isFinite(timeoutSeconds) || timeoutSeconds < 0) {
    console.error(`Timeout must be a non-negative number of seconds: ${args[0]}`);
    process.exit(2);
}

const command = args[separatorIndex + 1];
const commandArgs = args.slice(separatorIndex + 2);
let timedOut = false;
let finished = false;
let killTimer = null;

const child = spawn(command, commandArgs, {
    detached: true,
    stdio: 'inherit',
});

child.on('error', (error) => {
    if (finished) {
        return;
    }
    finished = true;
    if (killTimer) {
        clearTimeout(killTimer);
    }
    console.error(error.message);
    process.exit(127);
});

const timeoutTimer = timeoutSeconds > 0
    ? setTimeout(() => {
        if (finished) {
            return;
        }
        timedOut = true;
        console.error(`Command exceeded timeout (${timeoutSeconds}s), terminating: ${[command, ...commandArgs].join(' ')}`);
        try {
            process.kill(-child.pid, 'SIGTERM');
        } catch (_) {
            try {
                child.kill('SIGTERM');
            } catch (_) {
                // The process may have exited between the timeout and kill.
            }
        }
        killTimer = setTimeout(() => {
            try {
                process.kill(-child.pid, 'SIGKILL');
            } catch (_) {
                try {
                    child.kill('SIGKILL');
                } catch (_) {
                    // Already gone.
                }
            }
        }, 5000);
    }, timeoutSeconds * 1000)
    : null;

child.on('exit', (code, signal) => {
    if (finished) {
        return;
    }
    finished = true;
    if (timeoutTimer) {
        clearTimeout(timeoutTimer);
    }
    if (killTimer) {
        clearTimeout(killTimer);
    }
    if (timedOut) {
        process.exit(124);
    }
    if (signal) {
        console.error(`Command terminated by signal ${signal}: ${[command, ...commandArgs].join(' ')}`);
        process.exit(128);
    }
    process.exit(code === null ? 1 : code);
});
