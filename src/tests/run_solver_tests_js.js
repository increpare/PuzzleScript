#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

const ACTIONS = [
    { token: 'up', input: 0 },
    { token: 'left', input: 1 },
    { token: 'down', input: 2 },
    { token: 'right', input: 3 },
    { token: 'action', input: 4 },
];

function parseArgs(argv) {
    const options = {
        corpusPath: null,
        timeoutMs: 5000,
        json: false,
    };
    const args = argv.slice(2);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === '--timeout-ms') {
            options.timeoutMs = Math.max(1, Number.parseInt(args[++index], 10));
        } else if (arg === '--json') {
            options.json = true;
        } else if (arg === '--help' || arg === '-h') {
            usage(0);
        } else if (options.corpusPath === null) {
            options.corpusPath = path.resolve(arg);
        } else {
            throw new Error(`Unsupported argument: ${arg}`);
        }
    }
    if (!options.corpusPath) {
        usage(1);
    }
    return options;
}

function usage(exitCode) {
    const message = 'Usage: node src/tests/run_solver_tests_js.js <solver_tests_dir> [--timeout-ms N] [--json]\n';
    (exitCode === 0 ? process.stdout : process.stderr).write(message);
    process.exit(exitCode);
}

function isHiddenRelativePath(relativePath) {
    return relativePath.split(path.sep).some((part) => part.startsWith('.'));
}

function discoverGames(root) {
    const stat = fs.statSync(root);
    if (stat.isFile()) {
        return [root];
    }
    const result = [];
    const walk = (dir) => {
        for (const name of fs.readdirSync(dir)) {
            const full = path.join(dir, name);
            const rel = path.relative(root, full);
            if (isHiddenRelativePath(rel)) {
                continue;
            }
            const itemStat = fs.statSync(full);
            if (itemStat.isDirectory()) {
                walk(full);
            } else if (itemStat.isFile() && path.extname(name).toLowerCase() === '.txt') {
                result.push(full);
            }
        }
    };
    walk(root);
    return result.sort();
}

function gameName(root, file) {
    const stat = fs.statSync(root);
    return stat.isDirectory() ? path.relative(root, file).split(path.sep).join('/') : path.basename(file);
}

function cloneLevelState(value) {
    if (value == null) {
        return value;
    }
    const clone = {
        width: value.width,
        height: value.height,
        oldflickscreendat: Array.isArray(value.oldflickscreendat) ? value.oldflickscreendat.slice() : [],
    };
    if (value.diff) {
        clone.diff = true;
        clone.dat = new Int32Array(value.dat);
    } else {
        clone.dat = new Int32Array(value.dat);
    }
    return clone;
}

function cloneBackups(values) {
    return Array.isArray(values) ? values.map(cloneLevelState) : [];
}

function cloneRandomState(rng) {
    return {
        seed: rng ? rng.seed : null,
        normal: rng ? rng._normal : null,
        state: rng && rng._state
            ? { i: rng._state.i, j: rng._state.j, s: rng._state.s.slice() }
            : null,
    };
}

function restoreRandomState(snapshot) {
    const rng = new RNG(snapshot.seed);
    rng._normal = snapshot.normal;
    if (snapshot.state) {
        rng._state = {
            i: snapshot.state.i,
            j: snapshot.state.j,
            s: snapshot.state.s.slice(),
        };
    } else {
        rng._state = null;
    }
    RandomGen = rng;
}

function captureSnapshot() {
    return {
        levelState: cloneLevelState(backupLevel()),
        backups: cloneBackups(backups),
        restartTarget: cloneLevelState(restartTarget),
        random: cloneRandomState(RandomGen),
        curlevel,
        curlevelTarget: cloneLevelState(curlevelTarget),
        titleScreen,
        textMode,
        titleMode,
        titleSelection,
        titleSelected,
        messageselected,
        messagetext,
        winning,
        againing,
        loadedLevelSeed,
        hasUsedCheckpoint,
    };
}

function restoreSnapshot(snapshot) {
    restoreLevel(snapshot.levelState);
    backups = cloneBackups(snapshot.backups);
    restartTarget = cloneLevelState(snapshot.restartTarget);
    restoreRandomState(snapshot.random);
    curlevel = snapshot.curlevel;
    curlevelTarget = cloneLevelState(snapshot.curlevelTarget);
    titleScreen = snapshot.titleScreen;
    textMode = snapshot.textMode;
    titleMode = snapshot.titleMode;
    titleSelection = snapshot.titleSelection;
    titleSelected = snapshot.titleSelected;
    messageselected = snapshot.messageselected;
    messagetext = snapshot.messagetext;
    winning = snapshot.winning;
    againing = snapshot.againing;
    loadedLevelSeed = snapshot.loadedLevelSeed;
    hasUsedCheckpoint = snapshot.hasUsedCheckpoint;
    if (level) {
        level.commandQueue = [];
        level.commandQueueSourceRules = [];
    }
}

function randomHashPayload() {
    if (!RandomGen || !RandomGen._state) {
        return '';
    }
    const state = RandomGen._state;
    return `${state.i},${state.j},${state.s.join(',')}`;
}

function hashCurrentState() {
    const payload = JSON.stringify({
        curlevel,
        titleScreen,
        textMode,
        titleMode,
        titleSelection,
        winning,
        againing,
        serialized: typeof convertLevelToString === 'function' ? convertLevelToString() : '',
        message: textMode ? messagetext : '',
        random: randomHashPayload(),
    });
    return crypto.createHash('sha1').update(payload).digest('hex');
}

function settleAgain() {
    for (let pass = 0; pass < 500 && againing; pass++) {
        againing = false;
        processInput(-1);
    }
}

function stepSolverAction(action) {
    const beforeLevel = curlevel;
    const beforeTitle = titleScreen;
    let changed = false;
    if (action.input === 4 && textMode && !titleScreen) {
        if (state.levels[curlevel] && state.levels[curlevel].message !== undefined) {
            nextLevel();
        } else {
            textMode = false;
            messagetext = '';
            messageselected = false;
        }
        changed = true;
    } else {
        changed = Boolean(processInput(action.input));
    }
    settleAgain();
    const solved = changed && (curlevel !== beforeLevel || (!beforeTitle && titleScreen));
    return { changed, solved };
}

class MinHeap {
    constructor() {
        this.items = [];
    }

    get length() {
        return this.items.length;
    }

    less(a, b) {
        return a.priority < b.priority || (a.priority === b.priority && a.tie < b.tie);
    }

    push(item) {
        this.items.push(item);
        let index = this.items.length - 1;
        while (index > 0) {
            const parent = Math.floor((index - 1) / 2);
            if (!this.less(this.items[index], this.items[parent])) {
                break;
            }
            [this.items[index], this.items[parent]] = [this.items[parent], this.items[index]];
            index = parent;
        }
    }

    pop() {
        if (this.items.length === 0) {
            return null;
        }
        const first = this.items[0];
        const last = this.items.pop();
        if (this.items.length > 0) {
            this.items[0] = last;
            let index = 0;
            while (true) {
                const left = index * 2 + 1;
                const right = left + 1;
                let best = index;
                if (left < this.items.length && this.less(this.items[left], this.items[best])) {
                    best = left;
                }
                if (right < this.items.length && this.less(this.items[right], this.items[best])) {
                    best = right;
                }
                if (best === index) {
                    break;
                }
                [this.items[index], this.items[best]] = [this.items[best], this.items[index]];
                index = best;
            }
        }
        return first;
    }
}

function reconstruct(nodes, index, finalToken) {
    const reversed = [finalToken];
    let cursor = index;
    while (cursor >= 0) {
        const node = nodes[cursor];
        if (node.parent >= 0) {
            reversed.push(node.input);
        }
        cursor = node.parent;
    }
    return reversed.reverse();
}

function solveLevel(game, levelIndex, timeoutMs, compileMs) {
    const result = {
        game,
        level: levelIndex,
        status: 'exhausted',
        solution: [],
        solution_length: 0,
        elapsed_ms: 0,
        expanded: 0,
        generated: 0,
        unique_states: 0,
        duplicates: 0,
        max_frontier: 0,
        timeout_ms: timeoutMs,
        compile_ms: compileMs,
        load_ms: 0,
        clone_ms: 0,
        step_ms: 0,
        hash_ms: 0,
        queue_ms: 0,
        reconstruct_ms: 0,
    };
    const seed = `solver:${game}:${levelIndex}`;
    const loadStart = performance.now();
    loadLevelFromState(state, levelIndex, seed);
    result.load_ms = performance.now() - loadStart;
    if (textMode || titleScreen || (state.levels[levelIndex] && state.levels[levelIndex].message !== undefined)) {
        result.status = 'skipped_message';
        return result;
    }

    const started = Date.now();
    const deadline = started + timeoutMs;
    const nodes = [{ snapshot: captureSnapshot(), parent: -1, input: null, depth: 0 }];
    const bestDepth = new Map();
    const hashStart = performance.now();
    bestDepth.set(hashCurrentState(), 0);
    result.hash_ms += performance.now() - hashStart;
    result.unique_states = 1;
    const frontier = new MinHeap();
    frontier.push({ priority: 0, tie: 0, index: 0 });
    result.max_frontier = 1;
    let tie = 1;

    while (frontier.length > 0) {
        if (Date.now() >= deadline) {
            result.status = 'timeout';
            break;
        }
        const queueStart = performance.now();
        const entry = frontier.pop();
        result.queue_ms += performance.now() - queueStart;
        const node = nodes[entry.index];
        result.expanded++;

        for (const action of ACTIONS) {
            if (Date.now() >= deadline) {
                result.status = 'timeout';
                break;
            }
            const cloneStart = performance.now();
            restoreSnapshot(node.snapshot);
            result.clone_ms += performance.now() - cloneStart;

            const stepStart = performance.now();
            const stepResult = stepSolverAction(action);
            result.step_ms += performance.now() - stepStart;
            result.generated++;

            if (stepResult.solved) {
                const reconstructStart = performance.now();
                result.solution = reconstruct(nodes, entry.index, action.token);
                result.solution_length = result.solution.length;
                result.reconstruct_ms += performance.now() - reconstructStart;
                result.elapsed_ms = Date.now() - started;
                result.status = 'solved';
                return result;
            }

            const hashStart2 = performance.now();
            const key = hashCurrentState();
            result.hash_ms += performance.now() - hashStart2;
            const childDepth = node.depth + 1;
            if (bestDepth.has(key) && bestDepth.get(key) <= childDepth) {
                result.duplicates++;
                continue;
            }
            bestDepth.set(key, childDepth);
            result.unique_states = bestDepth.size;
            nodes.push({
                snapshot: captureSnapshot(),
                parent: entry.index,
                input: action.token,
                depth: childDepth,
            });
            const childIndex = nodes.length - 1;
            const queueStart2 = performance.now();
            frontier.push({ priority: childDepth, tie: tie++, index: childIndex });
            result.queue_ms += performance.now() - queueStart2;
            result.max_frontier = Math.max(result.max_frontier, frontier.length);
        }
    }

    result.solution_length = result.solution.length;
    result.elapsed_ms = Date.now() - started;
    return result;
}

function runGame(root, file) {
    const game = gameName(root, file);
    let source = fs.readFileSync(file, 'utf8');
    if (!source.endsWith('\n')) {
        source += '\n';
    }

    const compileStart = performance.now();
    unitTesting = true;
    lazyFunctionGeneration = false;
    compile(['loadLevel', 0], source, `solver:${game}:0`);
    const compileMs = performance.now() - compileStart;
    if (errorCount > 0) {
        return [{
            game,
            level: -1,
            status: 'compile_error',
            error: errorStrings.map(stripHTMLTags).join('\n'),
            solution: [],
            solution_length: 0,
            elapsed_ms: 0,
            expanded: 0,
            generated: 0,
            unique_states: 0,
            duplicates: 0,
            max_frontier: 0,
            timeout_ms: 0,
            compile_ms: compileMs,
            load_ms: 0,
            clone_ms: 0,
            step_ms: 0,
            hash_ms: 0,
            queue_ms: 0,
            reconstruct_ms: 0,
        }];
    }
    return { game, compileMs };
}

function runCorpus(options) {
    loadPuzzleScript();
    const results = [];
    for (const file of discoverGames(options.corpusPath)) {
        if (!fs.existsSync(file)) {
            continue;
        }
        const compiled = runGame(options.corpusPath, file);
        if (Array.isArray(compiled)) {
            results.push(...compiled);
            continue;
        }
        for (let levelIndex = 0; levelIndex < state.levels.length; levelIndex++) {
            results.push(solveLevel(compiled.game, levelIndex, options.timeoutMs, compiled.compileMs));
        }
    }
    return results;
}

function totals(results) {
    const out = {
        levels: results.length,
        solved: 0,
        timeout: 0,
        exhausted: 0,
        skipped_message: 0,
        errors: 0,
        expanded: 0,
        generated: 0,
        compile_ms: 0,
        load_ms: 0,
        clone_ms: 0,
        step_ms: 0,
        hash_ms: 0,
        queue_ms: 0,
        reconstruct_ms: 0,
    };
    for (const result of results) {
        out.solved += result.status === 'solved' ? 1 : 0;
        out.timeout += result.status === 'timeout' ? 1 : 0;
        out.exhausted += result.status === 'exhausted' ? 1 : 0;
        out.skipped_message += result.status === 'skipped_message' ? 1 : 0;
        out.errors += ['compile_error', 'level_error'].includes(result.status) ? 1 : 0;
        out.expanded += result.expanded;
        out.generated += result.generated;
        out.compile_ms += result.compile_ms || 0;
        out.load_ms += result.load_ms || 0;
        out.clone_ms += result.clone_ms || 0;
        out.step_ms += result.step_ms || 0;
        out.hash_ms += result.hash_ms || 0;
        out.queue_ms += result.queue_ms || 0;
        out.reconstruct_ms += result.reconstruct_ms || 0;
    }
    return out;
}

function printHuman(results) {
    for (const result of results) {
        const solution = result.solution.length > 0 ? ` solution=${result.solution.join(',')}` : '';
        const error = result.error ? ` error=${result.error}` : '';
        process.stdout.write(`${result.game} level=${result.level} status=${result.status} solution_length=${result.solution.length} elapsed_ms=${result.elapsed_ms} expanded=${result.expanded} generated=${result.generated} unique_states=${result.unique_states}${solution}${error}\n`);
    }
    const t = totals(results);
    process.stdout.write(`solver_totals levels=${t.levels} solved=${t.solved} timeout=${t.timeout} exhausted=${t.exhausted} skipped_message=${t.skipped_message} errors=${t.errors}\n`);
}

function main() {
    const options = parseArgs(process.argv);
    const results = runCorpus(options);
    if (options.json) {
        process.stdout.write(`${JSON.stringify({ results, totals: totals(results) }, null, 2)}\n`);
    } else {
        printHuman(results);
    }
}

main();
