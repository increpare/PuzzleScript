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
        solutionsDir: path.resolve('build/solver-solutions/js'),
        timeoutMs: 5000,
        progressEvery: 25,
        writeSolutions: true,
        progressPerGame: false,
        json: false,
        quiet: false,
        summaryOnly: false,
    };
    const args = argv.slice(2);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === '--timeout-ms') {
            options.timeoutMs = Math.max(1, Number.parseInt(args[++index], 10));
        } else if (arg === '--solutions-dir') {
            options.solutionsDir = path.resolve(args[++index]);
            options.writeSolutions = true;
        } else if (arg === '--no-solutions') {
            options.writeSolutions = false;
        } else if (arg === '--json') {
            options.json = true;
        } else if (arg === '--summary-only') {
            options.summaryOnly = true;
        } else if (arg === '--quiet') {
            options.quiet = true;
            options.progressEvery = 0;
        } else if (arg === '--progress-every') {
            options.progressEvery = Math.max(0, Number.parseInt(args[++index], 10));
        } else if (arg === '--progress-per-game') {
            options.progressPerGame = true;
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
    const message = 'Usage: node src/tests/run_solver_tests_js.js <solver_tests_dir> [--timeout-ms N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json]\n';
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
        const state = new RC4();
        state.i = snapshot.state.i;
        state.j = snapshot.state.j;
        state.s = snapshot.state.s.slice();
        rng._state = state;
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

function levelErrorResult(game, levelIndex, timeoutMs, compileMs, error) {
    return {
        game,
        level: levelIndex,
        status: 'level_error',
        error: error && error.stack ? error.stack : String(error),
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
    return { game, compileMs, source };
}

function trimLine(line) {
    return line.trim();
}

function isDividerLine(line) {
    const stripped = trimLine(line);
    return stripped.length > 0 && /^=+$/.test(stripped);
}

function isCommentLine(line) {
    const stripped = trimLine(line);
    return stripped.length > 0 && stripped.startsWith('(');
}

function splitLines(source) {
    const normalized = source.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    if (normalized.endsWith('\n')) {
        return normalized.slice(0, -1).split('\n');
    }
    return normalized.length > 0 ? normalized.split('\n') : [''];
}

function findSourceLevels(lines) {
    const levels = [];
    let index = lines.findIndex((line) => trimLine(line).toLowerCase() === 'levels');
    if (index < 0) {
        return levels;
    }
    index++;
    let levelIndex = 0;
    while (index < lines.length) {
        const stripped = trimLine(lines[index]);
        const lower = stripped.toLowerCase();
        if (stripped.length === 0 || isDividerLine(lines[index]) || isCommentLine(lines[index])) {
            index++;
            continue;
        }
        if (lower === 'message' || lower.startsWith('message ')) {
            levels.push({ level: levelIndex++, insertBeforeLine: index, message: true });
            index++;
            continue;
        }

        levels.push({ level: levelIndex++, insertBeforeLine: index, message: false });
        index++;
        while (index < lines.length && trimLine(lines[index]).length > 0) {
            index++;
        }
    }
    return levels;
}

function solutionLetter(input) {
    if (input === 'up') return 'U';
    if (input === 'down') return 'D';
    if (input === 'left') return 'L';
    if (input === 'right') return 'R';
    if (input === 'action') return 'A';
    return '?';
}

function compactSolution(solution) {
    let out = '';
    for (let index = 0; index < solution.length; index++) {
        if (index > 0 && index % 4 === 0) {
            out += ' ';
        }
        out += solutionLetter(solution[index]);
    }
    return out;
}

function writeAnnotatedSolutions(options, game, source, results) {
    if (!options.writeSolutions) {
        return false;
    }
    const solved = new Map();
    for (const result of results) {
        if (result.status === 'solved' && result.solution.length > 0) {
            solved.set(result.level, compactSolution(result.solution));
        }
    }
    if (solved.size === 0) {
        return false;
    }

    const lines = splitLines(source);
    const commentsByLine = new Map();
    for (const level of findSourceLevels(lines)) {
        if (level.message || !solved.has(level.level)) {
            continue;
        }
        const comments = commentsByLine.get(level.insertBeforeLine) || [];
        comments.push(`(${solved.get(level.level)})`);
        commentsByLine.set(level.insertBeforeLine, comments);
    }
    if (commentsByLine.size === 0) {
        return false;
    }

    const annotated = [];
    for (let index = 0; index < lines.length; index++) {
        for (const comment of commentsByLine.get(index) || []) {
            annotated.push(comment);
        }
        annotated.push(lines[index]);
    }

    const outputPath = path.join(options.solutionsDir, game);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, `${annotated.join('\n')}\n`);
    return true;
}

function summarizeHuman(results) {
    const summary = {
        solved: 0,
        timeout: 0,
        exhausted: 0,
        skipped_message: 0,
        errors: 0,
        expanded: 0,
        generated: 0,
    };
    for (const result of results) {
        summary.solved += result.status === 'solved' ? 1 : 0;
        summary.timeout += result.status === 'timeout' ? 1 : 0;
        summary.exhausted += result.status === 'exhausted' ? 1 : 0;
        summary.skipped_message += result.status === 'skipped_message' ? 1 : 0;
        summary.errors += ['compile_error', 'level_error'].includes(result.status) ? 1 : 0;
        summary.expanded += result.expanded;
        summary.generated += result.generated;
    }
    summary.playable_levels = summary.solved + summary.timeout + summary.exhausted + summary.errors;
    return summary;
}

function secondsString(elapsedMs) {
    return (elapsedMs / 1000).toFixed(2);
}

function printHumanBlock(stream, label, summary, elapsedMs) {
    stream.write('===\n');
    stream.write(`${label} (${secondsString(elapsedMs)} sec)\n`);
    stream.write(`Levels Solved: ${summary.solved}/${summary.playable_levels}\n`);
    stream.write(`Timeout: ${summary.timeout}\n`);
    if (summary.exhausted > 0) {
        stream.write(`Unsolvable: ${summary.exhausted}\n`);
    }
    if (summary.errors > 0) {
        stream.write(`Errors: ${summary.errors}\n`);
    }
}

function runCorpus(options) {
    loadPuzzleScript();
    const results = [];
    let attemptedLevels = 0;
    for (const file of discoverGames(options.corpusPath)) {
        if (!fs.existsSync(file)) {
            continue;
        }
        const name = gameName(options.corpusPath, file);
        const gameResultBegin = results.length;
        const gameStarted = Date.now();
        if (!options.quiet && !options.progressPerGame) {
            process.stderr.write(`solver_progress game=${name} phase=compile\n`);
        }
        const compiled = runGame(options.corpusPath, file);
        if (Array.isArray(compiled)) {
            results.push(...compiled);
            if (!options.quiet && options.progressPerGame) {
                printHumanBlock(process.stderr, `Game: ${name}`, summarizeHuman(results.slice(gameResultBegin)), Date.now() - gameStarted);
            } else if (!options.quiet) {
                process.stderr.write(`solver_progress game=${name} level=-1 status=compile_error completed=${results.length}\n`);
            }
            continue;
        }
        if (!options.quiet && !options.progressPerGame) {
            process.stderr.write(`solver_progress game=${compiled.game} phase=levels count=${state.levels.length}\n`);
        }
        for (let levelIndex = 0; levelIndex < state.levels.length; levelIndex++) {
            if (!options.quiet && !options.progressPerGame) {
                process.stderr.write(`solver_progress game=${compiled.game} level=${levelIndex} phase=start\n`);
            }
            let result;
            try {
                if (typeof resetParserErrorState === 'function') {
                    resetParserErrorState();
                }
                result = solveLevel(compiled.game, levelIndex, options.timeoutMs, compiled.compileMs);
            } catch (error) {
                if (typeof resetParserErrorState === 'function') {
                    resetParserErrorState();
                }
                result = levelErrorResult(compiled.game, levelIndex, options.timeoutMs, compiled.compileMs, error);
            }
            results.push(result);
            attemptedLevels++;
            if (!options.quiet && !options.progressPerGame && options.progressEvery > 0 && attemptedLevels % options.progressEvery === 0) {
                process.stderr.write(`solver_progress game=${compiled.game} level=${levelIndex} status=${result.status} solution_length=${result.solution.length} elapsed_ms=${result.elapsed_ms} expanded=${result.expanded} generated=${result.generated} completed=${attemptedLevels}\n`);
            }
        }
        const slice = results.slice(gameResultBegin);
        writeAnnotatedSolutions(options, compiled.game, compiled.source, slice);
        if (!options.quiet && options.progressPerGame) {
            printHumanBlock(process.stderr, `Game: ${compiled.game}`, summarizeHuman(slice), Date.now() - gameStarted);
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

function printSolutionsLocation(options) {
    process.stdout.write(`Solutions: ${options.writeSolutions ? options.solutionsDir : 'disabled'}\n`);
}

function main() {
    const options = parseArgs(process.argv);
    const results = runCorpus(options);
    if (options.json) {
        process.stdout.write(`${JSON.stringify({ results, totals: totals(results) }, null, 2)}\n`);
    } else if (options.summaryOnly) {
        const elapsedMs = results.reduce((sum, result) => sum + result.elapsed_ms, 0);
        printHumanBlock(process.stdout, 'Totals', summarizeHuman(results), elapsedMs);
        printSolutionsLocation(options);
    } else {
        printHuman(results);
        printSolutionsLocation(options);
    }
}

main();
