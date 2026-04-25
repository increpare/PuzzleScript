#!/usr/bin/env node
'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { PuzzleScriptGeneratorRun } = require('../src/puzzlescriptGeneratorRunner');

function writeExecutable(dir, name, body) {
    const script = path.join(dir, name);
    fs.writeFileSync(script, `#!/usr/bin/env node\n${body}`, 'utf8');
    fs.chmodSync(script, 0o755);
    return script;
}

async function runTests() {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'ps-generator-runner-test-'));
    const success = writeExecutable(tmp, 'success.js', `
const fs = require('fs');
const jsonOut = process.argv[process.argv.indexOf('--json-out') + 1];
console.error('generator_progress elapsed_s=0.1 jobs=1 top=1 samples=2 valid=2 solved=1');
fs.writeFileSync(jsonOut, JSON.stringify({ totals: { samples_attempted: 2 }, top: [] }));
`);
    const seenProgress = [];
    const goodRun = new PuzzleScriptGeneratorRun({
        binaryPath: success,
        sourceText: 'title T\\nlevels\\nP',
        specText: '(INIT LEVEL)\\nP\\n\\n(GENERATION RULES)\\nchoose 1 [ player ] -> [ player ]',
        runOptions: {
            timeMs: 10,
            jobs: 1,
            seed: 1,
            solverTimeoutMs: 10,
            solverStrategy: 'portfolio',
            topK: 1,
            samples: '',
        },
        onProgress: progress => seenProgress.push(progress),
    });
    const good = await goodRun.start();
    assert.strictEqual(good.cancelled, false);
    assert.deepStrictEqual(good.result, { totals: { samples_attempted: 2 }, top: [] });
    assert.strictEqual(fs.existsSync(good.tempDir), false);
    assert.strictEqual(seenProgress[0].samples, 2);

    const failure = writeExecutable(tmp, 'failure.js', `
console.error('bad spec');
process.exit(2);
`);
    const badRun = new PuzzleScriptGeneratorRun({
        binaryPath: failure,
        sourceText: '',
        specText: '',
        runOptions: {
            timeMs: 10,
            jobs: 1,
            seed: 1,
            solverTimeoutMs: 10,
            solverStrategy: 'portfolio',
            topK: 1,
            samples: '',
        },
    });
    await assert.rejects(() => badRun.start(), /bad spec/);

    const slow = writeExecutable(tmp, 'slow.js', `
setTimeout(() => {}, 10000);
`);
    const slowRun = new PuzzleScriptGeneratorRun({
        binaryPath: slow,
        sourceText: '',
        specText: '',
        runOptions: {
            timeMs: 10000,
            jobs: 1,
            seed: 1,
            solverTimeoutMs: 10,
            solverStrategy: 'portfolio',
            topK: 1,
            samples: '',
        },
    });
    const pending = slowRun.start();
    setTimeout(() => slowRun.cancel(), 50);
    const cancelled = await pending;
    assert.strictEqual(cancelled.cancelled, true);
    assert.strictEqual(fs.existsSync(cancelled.tempDir), false);

    fs.rmSync(tmp, { recursive: true, force: true });
}

runTests().then(() => {
    console.log('generator runner tests passed');
}).catch(error => {
    console.error(error);
    process.exit(1);
});
