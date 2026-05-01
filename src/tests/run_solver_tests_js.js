#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

const DEFAULT_STRATEGY = 'weighted-astar';
const DIRECTION_ACTIONS = [
    { token: 'right', input: 3 },
    { token: 'up', input: 0 },
    { token: 'down', input: 2 },
    { token: 'left', input: 1 },
];
const ACTIONS_WITH_ACTION = DIRECTION_ACTIONS.concat([{ token: 'action', input: 4 }]);

function solverActionsForGame() {
    if (state && state.metadata && Object.prototype.hasOwnProperty.call(state.metadata, 'noaction')) {
        return DIRECTION_ACTIONS;
    }
    return ACTIONS_WITH_ACTION;
}

function parseArgs(argv) {
    const options = {
        corpusPath: null,
        solutionsDir: path.resolve('build/solver-solutions/js'),
        timeoutMs: 5000,
        strategy: DEFAULT_STRATEGY,
        astarWeight: 2,
        portfolioBfsMs: null,
        progressEvery: 25,
        writeSolutions: true,
        progressPerGame: false,
        json: false,
        quiet: false,
        summaryOnly: false,
        gameFilter: null,
        levelFilter: null,
    };
    const args = argv.slice(2);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === '--timeout-ms') {
            options.timeoutMs = Math.max(1, Number.parseInt(args[++index], 10));
        } else if (arg === '--no-timeout') {
            options.timeoutMs = null;
        } else if (arg === '--strategy') {
            options.strategy = args[++index];
            if (!['portfolio', 'bfs', 'weighted-astar', 'greedy'].includes(options.strategy)) {
                throw new Error(`Unsupported strategy: ${options.strategy}`);
            }
        } else if (arg === '--astar-weight') {
            options.astarWeight = Math.max(1, Number.parseInt(args[++index], 10));
        } else if (arg === '--portfolio-bfs-ms') {
            options.portfolioBfsMs = Math.max(1, Number.parseInt(args[++index], 10));
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
        } else if (arg === '--game') {
            options.gameFilter = args[++index];
        } else if (arg === '--level') {
            options.levelFilter = Math.max(0, Number.parseInt(args[++index], 10));
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
    const message = 'Usage: node src/tests/run_solver_tests_js.js <solver_tests_dir> [--timeout-ms N|--no-timeout] [--strategy portfolio|bfs|weighted-astar|greedy] [--astar-weight N] [--portfolio-bfs-ms N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--game NAME] [--level N] [--summary-only] [--quiet] [--json]\n';
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

function cloneLevelData(value) {
    if (value == null) {
        return value;
    }
    const source = value.dat || value.objects;
    return {
        dat: source ? new Int32Array(source) : new Int32Array(0),
        width: value.width,
        height: value.height,
        oldflickscreendat: Array.isArray(value.oldflickscreendat) ? value.oldflickscreendat.slice() : [],
    };
}

function zeroBitVecArray(values) {
    if (!Array.isArray(values)) {
        return;
    }
    for (const value of values) {
        if (value && typeof value.setZero === 'function') {
            value.setZero();
        }
    }
}

function createObjectHasher(wordCount) {
    const lines = [
        '"use strict";',
        'let h1 = 0x811c9dc5 | 0;',
        'let h2 = 0x9e3779b9 | 0;',
        'let v = flags | 0;',
        'h1 = Math.imul(h1 ^ v, 16777619);',
        'h2 = Math.imul(h2 + v, 1597334677);',
    ];
    for (let index = 0; index < wordCount; index++) {
        lines.push(`v = objects[${index}] | 0;`);
        lines.push('h1 = Math.imul(h1 ^ v, 16777619);');
        lines.push('h2 = Math.imul(h2 ^ (v + 0x9e3779b9), 1597334677);');
    }
    lines.push('if (randomState) {');
    lines.push('  h1 = Math.imul(h1 ^ (randomState.i | 0), 16777619);');
    lines.push('  h2 = Math.imul(h2 ^ (randomState.j | 0), 1597334677);');
    lines.push('  const s = randomState.s;');
    lines.push('  for (let i = 0; i < s.length; i++) {');
    lines.push('    v = s[i] | 0;');
    lines.push('    h1 = Math.imul(h1 ^ v, 16777619);');
    lines.push('    h2 = Math.imul(h2 ^ (v + i), 1597334677);');
    lines.push('  }');
    lines.push('}');
    lines.push('return (h1 >>> 0).toString(36) + ":" + (h2 >>> 0).toString(36);');
    lines.push('//# sourceURL=puzzlescript/generated/solverObjectHasher.js');
    return new Function('objects', 'flags', 'randomState', lines.join('\n'));
}

function ruleGroupsUseRandom(groups) {
    for (const group of groups || []) {
        for (const rule of group || []) {
            if (!rule) {
                continue;
            }
            if (rule.isRandom) {
                return true;
            }
            for (const row of rule.cells || []) {
                const replacement = row && row.replacement;
                if (replacement && (
                    (replacement.randomEntityMask && !replacement.randomEntityMask.iszero()) ||
                    (replacement.randomDirMask && !replacement.randomDirMask.iszero())
                )) {
                    return true;
                }
            }
        }
    }
    return false;
}

function ruleGroupsUseCommand(groups, commandName) {
    for (const group of groups || []) {
        for (const rule of group || []) {
            for (const command of (rule && rule.commands) || []) {
                const name = Array.isArray(command) ? command[0] : command;
                if (name === commandName) {
                    return true;
                }
            }
        }
    }
    return false;
}

function distanceOrFallback(distance) {
    return distance === Infinity ? 64 : distance;
}

function priorityForMode(mode, depth, heuristic, astarWeight) {
    if (mode === 'weighted-astar') {
        return depth + heuristic * astarWeight;
    }
    if (mode === 'greedy') {
        return heuristic;
    }
    return depth;
}

function createSolverLevelSpecialization() {
    const objectWordCount = level && level.objects ? level.objects.length : 0;
    const movementWordCount = level && level.movements ? level.movements.length : 0;
    const width = level && level.width;
    const height = level && level.height;
    const hasher = createObjectHasher(objectWordCount);
    const ruleGroups = [...(state.rules || []), ...(state.lateRules || [])];
    const usesRandom = ruleGroupsUseRandom(ruleGroups);
    const usesCheckpoint = ruleGroupsUseCommand(ruleGroups, 'checkpoint');
    const heuristicDistances = new Array(level.n_tiles);
    const conditionDistances = [];

    function matchesMask(mask, aggregate, tileIndex) {
        if (!mask || !mask.data) {
            return false;
        }
        const offset = tileIndex * STRIDE_OBJ;
        if (aggregate) {
            for (let word = 0; word < STRIDE_OBJ; word++) {
                const required = mask.data[word] | 0;
                if ((level.objects[offset + word] & required) !== required) {
                    return false;
                }
            }
            return true;
        }
        for (let word = 0; word < STRIDE_OBJ; word++) {
            if ((level.objects[offset + word] & mask.data[word]) !== 0) {
                return true;
            }
        }
        return false;
    }

    function matchingDistanceField(mask, aggregate, distances) {
        for (let tile = 0; tile < level.n_tiles; tile++) {
            distances[tile] = matchesMask(mask, aggregate, tile) ? 0 : Infinity;
        }
        for (let x = 0; x < level.width; x++) {
            for (let y = 0; y < level.height; y++) {
                const tile = x * level.height + y;
                let best = distances[tile];
                if (x > 0) best = Math.min(best, distances[(x - 1) * level.height + y] + 1);
                if (y > 0) best = Math.min(best, distances[x * level.height + y - 1] + 1);
                distances[tile] = best;
            }
        }
        for (let x = level.width - 1; x >= 0; x--) {
            for (let y = level.height - 1; y >= 0; y--) {
                const tile = x * level.height + y;
                let best = distances[tile];
                if (x + 1 < level.width) best = Math.min(best, distances[(x + 1) * level.height + y] + 1);
                if (y + 1 < level.height) best = Math.min(best, distances[x * level.height + y + 1] + 1);
                distances[tile] = best;
            }
        }
    }

    function heuristic() {
        if (!state.winconditions || state.winconditions.length === 0) {
            return 0;
        }
        let score = 0;
        for (let conditionIndex = 0; conditionIndex < state.winconditions.length; conditionIndex++) {
            const condition = state.winconditions[conditionIndex];
            const quantifier = condition[0];
            const filter1 = condition[1];
            const filter2 = condition[2];
            const aggregate1 = condition[4];
            const aggregate2 = condition[5];
            matchingDistanceField(filter2, aggregate2, heuristicDistances);
            if (quantifier === 1) {
                for (let tile = 0; tile < level.n_tiles; tile++) {
                    if (!matchesMask(filter1, aggregate1, tile)) {
                        continue;
                    }
                    if (matchesMask(filter2, aggregate2, tile)) {
                        continue;
                    }
                    score += 10 + distanceOrFallback(heuristicDistances[tile]);
                }
            } else if (quantifier === 0) {
                let passed = false;
                let best = 64;
                for (let tile = 0; tile < level.n_tiles; tile++) {
                    if (!matchesMask(filter1, aggregate1, tile)) {
                        continue;
                    }
                    if (matchesMask(filter2, aggregate2, tile)) {
                        passed = true;
                        break;
                    }
                    best = Math.min(best, distanceOrFallback(heuristicDistances[tile]));
                }
                score += passed ? 0 : best;
            } else if (quantifier === -1) {
                for (let tile = 0; tile < level.n_tiles; tile++) {
                    if (matchesMask(filter1, aggregate1, tile) && matchesMask(filter2, aggregate2, tile)) {
                        score += 10;
                    }
                }
            }
        }

        if (score > 0 && state.playerMask && state.playerMask[1]) {
            const playerAggregate = state.playerMask[0];
            const playerMask = state.playerMask[1];
            let hasPlayer = false;
            let best = 64;
            for (let conditionIndex = 0; conditionIndex < state.winconditions.length; conditionIndex++) {
                const condition = state.winconditions[conditionIndex];
                const distances = conditionDistances[conditionIndex] || new Array(level.n_tiles);
                conditionDistances[conditionIndex] = distances;
                matchingDistanceField(condition[1], condition[4], distances);
            }
            for (let tile = 0; tile < level.n_tiles; tile++) {
                if (!matchesMask(playerMask, playerAggregate, tile)) {
                    continue;
                }
                hasPlayer = true;
                for (const distances of conditionDistances) {
                    best = Math.min(best, distanceOrFallback(distances[tile]));
                }
            }
            if (hasPlayer) {
                score += Math.min(best, 16);
            }
        }
        return score;
    }

    function flagsForHash() {
        let flags = 0;
        flags = ((flags * 31 + (curlevel | 0)) | 0);
        flags = ((flags * 31 + (titleScreen ? 1 : 0)) | 0);
        flags = ((flags * 31 + (textMode ? 1 : 0)) | 0);
        flags = ((flags * 31 + (titleMode | 0)) | 0);
        flags = ((flags * 31 + (titleSelection | 0)) | 0);
        flags = ((flags * 31 + (winning ? 1 : 0)) | 0);
        flags = ((flags * 31 + (againing ? 1 : 0)) | 0);
        return flags;
    }

    function hash() {
        if (!level || !level.objects || level.objects.length !== objectWordCount || level.width !== width || level.height !== height || textMode) {
            return hashCurrentState();
        }
        return hasher(level.objects, flagsForHash(), usesRandom && RandomGen && RandomGen._state);
    }

    function capture() {
        return {
            objects: new Int32Array(level.objects),
            oldflickscreendat: Array.isArray(oldflickscreendat) ? oldflickscreendat.slice() : [],
            restartTarget: usesCheckpoint ? cloneLevelData(restartTarget) : null,
            random: usesRandom ? cloneRandomState(RandomGen) : null,
            curlevel,
            curlevelTarget: cloneLevelData(curlevelTarget),
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

    function restore(snapshot) {
        if (!level || !level.objects || level.objects.length !== objectWordCount || level.width !== width || level.height !== height) {
            restoreSnapshot({
                levelState: {
                    dat: snapshot.objects,
                    width,
                    height,
                    oldflickscreendat: snapshot.oldflickscreendat,
                },
                backups: [],
                restartTarget: usesCheckpoint ? snapshot.restartTarget : restartTarget,
                random: snapshot.random,
                curlevel: snapshot.curlevel,
                curlevelTarget: snapshot.curlevelTarget,
                titleScreen: snapshot.titleScreen,
                textMode: snapshot.textMode,
                titleMode: snapshot.titleMode,
                titleSelection: snapshot.titleSelection,
                titleSelected: snapshot.titleSelected,
                messageselected: snapshot.messageselected,
                messagetext: snapshot.messagetext,
                winning: snapshot.winning,
                againing: snapshot.againing,
                loadedLevelSeed: snapshot.loadedLevelSeed,
                hasUsedCheckpoint: snapshot.hasUsedCheckpoint,
            });
            return;
        }
        level.objects.set(snapshot.objects);
        if (level.movements && level.movements.length === movementWordCount) {
            level.movements.fill(0);
        }
        if (state.rigid) {
            zeroBitVecArray(level.rigidMovementAppliedMask);
            zeroBitVecArray(level.rigidGroupIndexMask);
        }
        zeroBitVecArray(level.rowCellContents);
        zeroBitVecArray(level.rowCellContents_Movements);
        zeroBitVecArray(level.colCellContents);
        zeroBitVecArray(level.colCellContents_Movements);
        oldflickscreendat = Array.isArray(snapshot.oldflickscreendat) ? snapshot.oldflickscreendat.slice() : [];
        backups = [];
        if (usesCheckpoint) {
            restartTarget = cloneLevelData(snapshot.restartTarget);
        }
        if (usesRandom) {
            restoreRandomState(snapshot.random);
        }
        curlevel = snapshot.curlevel;
        curlevelTarget = cloneLevelData(snapshot.curlevelTarget);
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
        level.commandQueue = [];
        level.commandQueueSourceRules = [];
    }

    return {
        capture,
        restore,
        hash,
        heuristic,
        hashMode: usesRandom ? 'specialized_numeric_with_rng' : 'specialized_numeric',
        snapshotMode: usesCheckpoint ? 'specialized_typed_array_checkpoint' : 'specialized_typed_array_no_undo',
    };
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
        processInput(-1, undefined, undefined, true);
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
        changed = Boolean(processInput(action.input, undefined, undefined, true));
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

function createSolverResult(game, levelIndex, timeoutMs, compileMs) {
    return {
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
        snapshot_ms: 0,
        step_ms: 0,
        heuristic_ms: 0,
        hash_ms: 0,
        queue_ms: 0,
        reconstruct_ms: 0,
        hash_mode: null,
        snapshot_mode: null,
        strategy: null,
        heuristic: 'zero',
    };
}

function mergeSearchResultTotals(target, source) {
    for (const key of [
        'expanded',
        'generated',
        'duplicates',
        'clone_ms',
        'snapshot_ms',
        'step_ms',
        'heuristic_ms',
        'hash_ms',
        'queue_ms',
        'reconstruct_ms',
    ]) {
        target[key] = (target[key] || 0) + (source[key] || 0);
    }
    target.unique_states = Math.max(target.unique_states || 0, source.unique_states || 0);
    target.max_frontier = Math.max(target.max_frontier || 0, source.max_frontier || 0);
    if (source.hash_mode) {
        target.hash_mode = source.hash_mode;
    }
    if (source.snapshot_mode) {
        target.snapshot_mode = source.snapshot_mode;
    }
}

function solveLevel(game, levelIndex, timeoutMs, compileMs, options = {}) {
    const result = createSolverResult(game, levelIndex, timeoutMs, compileMs);
    const seed = `solver:${game}:${levelIndex}`;
    const loadStart = performance.now();
    loadLevelFromState(state, levelIndex, seed);
    result.load_ms = performance.now() - loadStart;
    if (textMode || titleScreen || (state.levels[levelIndex] && state.levels[levelIndex].message !== undefined)) {
        result.status = 'skipped_message';
        return result;
    }

    const searchStarted = Date.now();
    const deadline = Number.isFinite(timeoutMs) ? searchStarted + timeoutMs : Infinity;
    const initialSnapshot = createSolverLevelSpecialization().capture();
    const strategy = options.strategy || DEFAULT_STRATEGY;

    const runMode = (mode, modeDeadline) => {
        const modeResult = createSolverResult(game, levelIndex, timeoutMs, compileMs);
        modeResult.load_ms = result.load_ms;
        modeResult.strategy = mode;
        modeResult.heuristic = mode === 'bfs' ? 'zero' : 'winconditions';
        const solverOps = createSolverLevelSpecialization();
        modeResult.hash_mode = solverOps.hashMode;
        modeResult.snapshot_mode = solverOps.snapshotMode;
        const initialRestoreStart = performance.now();
        solverOps.restore(initialSnapshot);
        modeResult.clone_ms += performance.now() - initialRestoreStart;
        const initialSnapshotStart = performance.now();
        const nodes = [{ snapshot: solverOps.capture(), parent: -1, input: null, depth: 0 }];
        modeResult.snapshot_ms += performance.now() - initialSnapshotStart;
        const bestDepth = new Map();
        const hashStart = performance.now();
        bestDepth.set(solverOps.hash(), 0);
        modeResult.hash_ms += performance.now() - hashStart;
        modeResult.unique_states = 1;
        const frontier = new MinHeap();
        let initialHeuristic = 0;
        if (mode !== 'bfs') {
            const heuristicStart = performance.now();
            initialHeuristic = solverOps.heuristic();
            modeResult.heuristic_ms += performance.now() - heuristicStart;
        }
        frontier.push({ priority: priorityForMode(mode, 0, initialHeuristic, options.astarWeight || 2), tie: 0, index: 0 });
        modeResult.max_frontier = 1;
        let tie = 1;
        const actions = solverActionsForGame();

        while (frontier.length > 0) {
            if (Date.now() >= modeDeadline) {
                modeResult.status = 'timeout';
                break;
            }
            const queueStart = performance.now();
            const entry = frontier.pop();
            modeResult.queue_ms += performance.now() - queueStart;
            const node = nodes[entry.index];
            modeResult.expanded++;

            for (const action of actions) {
                if (Date.now() >= modeDeadline) {
                    modeResult.status = 'timeout';
                    break;
                }
                const cloneStart = performance.now();
                solverOps.restore(node.snapshot);
                modeResult.clone_ms += performance.now() - cloneStart;

                const stepStart = performance.now();
                const stepResult = stepSolverAction(action);
                modeResult.step_ms += performance.now() - stepStart;
                modeResult.generated++;

                if (stepResult.solved) {
                    const reconstructStart = performance.now();
                    modeResult.solution = reconstruct(nodes, entry.index, action.token);
                    modeResult.solution_length = modeResult.solution.length;
                    modeResult.reconstruct_ms += performance.now() - reconstructStart;
                    modeResult.elapsed_ms = Date.now() - searchStarted;
                    modeResult.status = 'solved';
                    return modeResult;
                }
                if (!stepResult.changed) {
                    continue;
                }

                const hashStart2 = performance.now();
                const key = solverOps.hash();
                modeResult.hash_ms += performance.now() - hashStart2;
                const childDepth = node.depth + 1;
                if (bestDepth.has(key) && bestDepth.get(key) <= childDepth) {
                    modeResult.duplicates++;
                    continue;
                }
                bestDepth.set(key, childDepth);
                modeResult.unique_states = bestDepth.size;
                const snapshotStart = performance.now();
                const snapshot = solverOps.capture();
                modeResult.snapshot_ms += performance.now() - snapshotStart;
                nodes.push({
                    snapshot,
                    parent: entry.index,
                    input: action.token,
                    depth: childDepth,
                });
                const childIndex = nodes.length - 1;
                let childHeuristic = 0;
                if (mode !== 'bfs') {
                    const heuristicStart = performance.now();
                    childHeuristic = solverOps.heuristic();
                    modeResult.heuristic_ms += performance.now() - heuristicStart;
                }
                const queueStart2 = performance.now();
                frontier.push({
                    priority: priorityForMode(mode, childDepth, childHeuristic, options.astarWeight || 2),
                    tie: tie++,
                    index: childIndex,
                });
                modeResult.queue_ms += performance.now() - queueStart2;
                modeResult.max_frontier = Math.max(modeResult.max_frontier, frontier.length);
            }
        }

        modeResult.solution_length = modeResult.solution.length;
        modeResult.elapsed_ms = Date.now() - searchStarted;
        return modeResult;
    };

    if (strategy === 'portfolio') {
        const configuredBfsMs = Number.isFinite(options.portfolioBfsMs) ? options.portfolioBfsMs : null;
        const bfsDuration = configuredBfsMs !== null
            ? Math.max(0, Number.isFinite(timeoutMs) ? Math.min(configuredBfsMs, timeoutMs) : configuredBfsMs)
            : (Number.isFinite(timeoutMs) ? Math.max(1, Math.floor(timeoutMs / 6)) : 0);
        const bfsDeadline = Math.min(searchStarted + bfsDuration, deadline);
        const bfs = bfsDuration > 0 ? runMode('bfs', bfsDeadline) : createSolverResult(game, levelIndex, timeoutMs, compileMs);
        mergeSearchResultTotals(result, bfs);
        if (bfs.status === 'solved') {
            Object.assign(result, bfs, { strategy: 'bfs', elapsed_ms: Date.now() - searchStarted });
            return result;
        }
        if (Date.now() < deadline) {
            const weighted = runMode('weighted-astar', deadline);
            mergeSearchResultTotals(result, weighted);
            Object.assign(result, weighted, {
                expanded: result.expanded,
                generated: result.generated,
                duplicates: result.duplicates,
                unique_states: Math.max(result.unique_states, weighted.unique_states || 0),
                max_frontier: Math.max(result.max_frontier, weighted.max_frontier || 0),
                clone_ms: result.clone_ms,
                snapshot_ms: result.snapshot_ms,
                step_ms: result.step_ms,
                heuristic_ms: result.heuristic_ms,
                hash_ms: result.hash_ms,
                queue_ms: result.queue_ms,
                reconstruct_ms: result.reconstruct_ms,
                strategy: weighted.status === 'solved' ? 'weighted-astar' : 'portfolio',
                elapsed_ms: Date.now() - searchStarted,
            });
            return result;
        }
        result.status = 'timeout';
        result.strategy = 'portfolio';
        result.elapsed_ms = Date.now() - searchStarted;
        return result;
    }

    return runMode(strategy, deadline);
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
        snapshot_ms: 0,
        step_ms: 0,
        heuristic_ms: 0,
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
        if (options.gameFilter !== null && name !== options.gameFilter) {
            continue;
        }
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
            if (options.levelFilter !== null && levelIndex !== options.levelFilter) {
                continue;
            }
            if (!options.quiet && !options.progressPerGame) {
                process.stderr.write(`solver_progress game=${compiled.game} level=${levelIndex} phase=start\n`);
            }
            let result;
            try {
                if (typeof resetParserErrorState === 'function') {
                    resetParserErrorState();
                }
                result = solveLevel(compiled.game, levelIndex, options.timeoutMs, compiled.compileMs, options);
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
        snapshot_ms: 0,
        step_ms: 0,
        heuristic_ms: 0,
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
        out.snapshot_ms += result.snapshot_ms || 0;
        out.step_ms += result.step_ms || 0;
        out.heuristic_ms += result.heuristic_ms || 0;
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
