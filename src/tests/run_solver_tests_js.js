#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const { loadPuzzleScript } = require('./js_oracle/lib/puzzlescript_node_env');

const DEFAULT_STRATEGY = 'weighted-astar';
const DEFAULT_SOLVER_HEURISTIC = 'winconditions';
const SOLVER_HEURISTICS = new Set([
    'zero',
    'winconditions',
    'all-on-count',
    'all-on-rowcol-tiebreak',
    'all-on-line-distance',
    'all-on-clear-path',
    'all-on-goal-coverage',
    'all-on-rowcol-matching',
    'all-on-player-nearest-tiebreak',
    'all-on-push-access',
    'all-on-dead-position',
    'all-on-player-tiebreak',
    'all-on-min-matching',
    'all-on-matching',
    'all-on-obstacle',
    'all-on-player',
    'all-on-deadlock',
    'some-on-min',
    'some-on-player',
    'some-on-obstacle',
    'no-on-count',
    'no-on-escape',
    'no-on-player',
    'some-plain-exists',
    'some-plain-player',
    'no-plain-count',
    'no-plain-cluster',
    'no-plain-player',
]);
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
        solverHeuristic: DEFAULT_SOLVER_HEURISTIC,
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
        } else if (arg === '--solver-heuristic') {
            options.solverHeuristic = args[++index];
            if (!SOLVER_HEURISTICS.has(options.solverHeuristic)) {
                throw new Error(`Unsupported solver heuristic: ${options.solverHeuristic}`);
            }
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
    const message = 'Usage: node src/tests/run_solver_tests_js.js <solver_tests_dir> [--timeout-ms N|--no-timeout] [--strategy portfolio|bfs|weighted-astar|greedy] [--astar-weight N] [--solver-heuristic NAME] [--portfolio-bfs-ms N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--game NAME] [--level N] [--summary-only] [--quiet] [--json]\n';
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

function int32ArraysEqual(left, right) {
    if (left === right) {
        return true;
    }
    if (!left || !right || left.length !== right.length) {
        return false;
    }
    for (let index = 0; index < left.length; index++) {
        if ((left[index] | 0) !== (right[index] | 0)) {
            return false;
        }
    }
    return true;
}

function arraysEqual(left, right) {
    if (left === right) {
        return true;
    }
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length) {
        return false;
    }
    for (let index = 0; index < left.length; index++) {
        if (left[index] !== right[index]) {
            return false;
        }
    }
    return true;
}

function randomStateEqual(left, right) {
    if (left === right) {
        return true;
    }
    if (!left || !right || left.i !== right.i || left.j !== right.j) {
        return false;
    }
    return arraysEqual(left.s, right.s);
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

function createZobristStateHasher(usesRandom) {
    const lines = [
        '"use strict";',
        'let h1 = Math.imul((lo ^ flags) | 0, 16777619);',
        'let h2 = Math.imul((hi + flags) | 0, 1597334677);',
    ];
    if (usesRandom) {
        lines.push('if (randomState) {');
        lines.push('  h1 = Math.imul(h1 ^ (randomState.i | 0), 16777619);');
        lines.push('  h2 = Math.imul(h2 ^ (randomState.j | 0), 1597334677);');
        lines.push('  const s = randomState.s;');
        lines.push('  for (let i = 0; i < s.length; i++) {');
        lines.push('    const v = s[i] | 0;');
        lines.push('    h1 = Math.imul(h1 ^ v, 16777619);');
        lines.push('    h2 = Math.imul(h2 ^ (v + i), 1597334677);');
        lines.push('  }');
        lines.push('}');
    }
    lines.push('return (h1 >>> 0).toString(36) + ":" + (h2 >>> 0).toString(36);');
    lines.push('//# sourceURL=puzzlescript/generated/solverZobristHasher.js');
    return new Function('lo', 'hi', 'flags', 'randomState', lines.join('\n'));
}

function nextZobristSeed(seed) {
    seed = (seed + 0x9e3779b9) | 0;
    let value = seed;
    value = Math.imul(value ^ (value >>> 16), 0x85ebca6b);
    value = Math.imul(value ^ (value >>> 13), 0xc2b2ae35);
    return {
        seed,
        value: (value ^ (value >>> 16)) | 0,
    };
}

const ZOBRIST_TABLE_CACHE = new Map();

function createZobristTables(wordCount) {
    const cached = ZOBRIST_TABLE_CACHE.get(wordCount);
    if (cached) {
        return cached;
    }
    const tableLength = wordCount * 32;
    const lo = new Int32Array(tableLength);
    const hi = new Int32Array(tableLength);
    let seed = Math.imul(wordCount + 1, 0x45d9f3b) | 0;
    for (let index = 0; index < tableLength; index++) {
        let next = nextZobristSeed(seed);
        seed = next.seed;
        lo[index] = next.value;
        next = nextZobristSeed(seed);
        seed = next.seed;
        hi[index] = next.value;
    }
    const tables = { lo, hi };
    ZOBRIST_TABLE_CACHE.set(wordCount, tables);
    return tables;
}

function computeZobristBoardHash(objects, tableLo, tableHi) {
    let lo = 0;
    let hi = 0;
    for (let wordIndex = 0; wordIndex < objects.length; wordIndex++) {
        let bits = objects[wordIndex] >>> 0;
        const tableOffset = wordIndex * 32;
        while (bits !== 0) {
            const lowBit = bits & -bits;
            const bit = 31 - Math.clz32(lowBit);
            const tableIndex = tableOffset + bit;
            lo ^= tableLo[tableIndex];
            hi ^= tableHi[tableIndex];
            bits = (bits ^ lowBit) >>> 0;
        }
    }
    return { lo: lo | 0, hi: hi | 0 };
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

function priorityForPortfolioMode(mode, depth, heuristic) {
    if (mode === 'wa2') {
        return depth + heuristic * 2;
    }
    if (mode === 'wa8') {
        return depth + heuristic * 8;
    }
    if (mode === 'greedy') {
        return heuristic;
    }
    return depth;
}

function createSolverLevelSpecialization(options = {}) {
    const objectWordCount = level && level.objects ? level.objects.length : 0;
    const movementWordCount = level && level.movements ? level.movements.length : 0;
    const width = level && level.width;
    const height = level && level.height;
    const ruleGroups = [...(state.rules || []), ...(state.lateRules || [])];
    const usesRandom = ruleGroupsUseRandom(ruleGroups);
    const usesCheckpoint = ruleGroupsUseCommand(ruleGroups, 'checkpoint');
    const heuristicName = options.solverHeuristic || DEFAULT_SOLVER_HEURISTIC;
    const zobristHasher = createZobristStateHasher(usesRandom);
    const zobristTables = createZobristTables(objectWordCount);
    const zobristTableId = zobristTables;
    const heuristicDistances = new Array(level.n_tiles);
    const obstacleDistances = new Array(level.n_tiles);
    const obstacleQueue = new Array(level.n_tiles);
    const targetRowPresence = new Uint8Array(height || 0);
    const targetColPresence = new Uint8Array(width || 0);
    const conditionDistances = [];
    const allObjectsMask = state.objectMasks && state.objectMasks["\nall\n"];
    const playerAggregate = Array.isArray(state.playerMask) ? state.playerMask[0] : false;
    const playerMask = Array.isArray(state.playerMask) ? state.playerMask[1] : state.playerMask;
    const backgroundMask = state.layerMasks && Number.isInteger(state.backgroundlayer)
        ? state.layerMasks[state.backgroundlayer]
        : null;
    const nonBackgroundWords = new Int32Array(STRIDE_OBJ);
    for (let word = 0; word < STRIDE_OBJ; word++) {
        nonBackgroundWords[word] = backgroundMask && backgroundMask.data ? ~backgroundMask.data[word] : -1;
    }
    const verifyZobrist = process.env.PUZZLESCRIPT_VERIFY_ZOBRIST === '1';

    function installZobristHash() {
        if (!level || !level.objects || level.objects.length !== objectWordCount) {
            return;
        }
        level.solverZobristTableLo = zobristTables.lo;
        level.solverZobristTableHi = zobristTables.hi;
        level.solverZobristTableId = zobristTableId;
        level.solverZobristUpdateCell = updateZobristCell;
        level.solverZobristRecompute = recomputeZobrist;
        recomputeZobrist();
    }

    function recomputeZobrist() {
        const hash = computeZobristBoardHash(level.objects, zobristTables.lo, zobristTables.hi);
        level.solverZobristLo = hash.lo;
        level.solverZobristHi = hash.hi;
    }

    function updateZobristCell(tileIndex, vec) {
        let lo = level.solverZobristLo | 0;
        let hi = level.solverZobristHi | 0;
        const offset = tileIndex * STRIDE_OBJ;
        if (verifyZobrist) {
            const expectedStart = computeZobristBoardHash(level.objects, zobristTables.lo, zobristTables.hi);
            if (lo !== expectedStart.lo || hi !== expectedStart.hi) {
                throw new Error(`Incremental Zobrist hash already drifted before tile ${tileIndex}: ${lo}:${hi} !== ${expectedStart.lo}:${expectedStart.hi}`);
            }
        }
        for (let word = 0; word < STRIDE_OBJ; word++) {
            const oldWord = level.objects[offset + word] | 0;
            const newWord = vec.data[word] | 0;
            let changed = (oldWord ^ newWord) >>> 0;
            const tableOffset = (offset + word) * 32;
            while (changed !== 0) {
                const lowBit = changed & -changed;
                const bit = 31 - Math.clz32(lowBit);
                const tableIndex = tableOffset + bit;
                lo ^= zobristTables.lo[tableIndex];
                hi ^= zobristTables.hi[tableIndex];
                changed = (changed ^ lowBit) >>> 0;
            }
        }
        level.solverZobristLo = lo | 0;
        level.solverZobristHi = hi | 0;
        if (verifyZobrist) {
            const expected = computeZobristBoardHash(level.objects, zobristTables.lo, zobristTables.hi);
            // updateZobristCell runs before the caller writes vec into level.objects, so
            // compare against the board with this one cell patched in.
            const oldWords = [];
            for (let word = 0; word < STRIDE_OBJ; word++) {
                oldWords[word] = level.objects[offset + word];
                level.objects[offset + word] = vec.data[word];
            }
            const patched = computeZobristBoardHash(level.objects, zobristTables.lo, zobristTables.hi);
            for (let word = 0; word < STRIDE_OBJ; word++) {
                level.objects[offset + word] = oldWords[word];
            }
            if ((level.solverZobristLo | 0) !== patched.lo || (level.solverZobristHi | 0) !== patched.hi) {
                throw new Error(`Incremental Zobrist update drifted at tile ${tileIndex}: ${level.solverZobristLo}:${level.solverZobristHi} !== ${patched.lo}:${patched.hi}; old board ${expected.lo}:${expected.hi}`);
            }
        }
    }

    installZobristHash();

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

    function masksEqual(left, right) {
        if (left === right) {
            return true;
        }
        if (!left || !right || !left.data || !right.data || left.data.length !== right.data.length) {
            return false;
        }
        for (let word = 0; word < left.data.length; word++) {
            if ((left.data[word] | 0) !== (right.data[word] | 0)) {
                return false;
            }
        }
        return true;
    }

    function isPlainCondition(condition) {
        return Boolean(condition && allObjectsMask && masksEqual(condition[2], allObjectsMask));
    }

    function singleCondition() {
        return state.winconditions && state.winconditions.length === 1 ? state.winconditions[0] : null;
    }

    function singleAllOnCondition() {
        const condition = singleCondition();
        return condition && condition[0] === 1 && !isPlainCondition(condition) ? condition : null;
    }

    function collectMatchingTiles(mask, aggregate) {
        const tiles = [];
        for (let tile = 0; tile < level.n_tiles; tile++) {
            if (matchesMask(mask, aggregate, tile)) {
                tiles.push(tile);
            }
        }
        return tiles;
    }

    function collectUnsatisfiedAllOnTiles(condition) {
        const tiles = [];
        for (let tile = 0; tile < level.n_tiles; tile++) {
            if (matchesMask(condition[1], condition[4], tile) && !matchesMask(condition[2], condition[5], tile)) {
                tiles.push(tile);
            }
        }
        return tiles;
    }

    function collectOverlapTiles(condition) {
        const tiles = [];
        for (let tile = 0; tile < level.n_tiles; tile++) {
            if (matchesMask(condition[1], condition[4], tile) && matchesMask(condition[2], condition[5], tile)) {
                tiles.push(tile);
            }
        }
        return tiles;
    }

    function manhattan(left, right) {
        const leftX = (left / level.height) | 0;
        const leftY = left % level.height;
        const rightX = (right / level.height) | 0;
        const rightY = right % level.height;
        return Math.abs(leftX - rightX) + Math.abs(leftY - rightY);
    }

    function tileX(tile) {
        return (tile / level.height) | 0;
    }

    function tileY(tile) {
        return tile % level.height;
    }

    function tileIndexAt(x, y) {
        return x * level.height + y;
    }

    function inBounds(x, y) {
        return x >= 0 && y >= 0 && x < level.width && y < level.height;
    }

    function bestManhattan(tile, targets) {
        let best = Infinity;
        for (const target of targets) {
            best = Math.min(best, manhattan(tile, target));
        }
        return distanceOrFallback(best);
    }

    function minPlayerDistanceToTiles(tiles) {
        if (!playerMask || !playerMask.data || tiles.length === 0) {
            return 0;
        }
        const players = collectMatchingTiles(playerMask, playerAggregate);
        if (players.length === 0) {
            return 0;
        }
        let best = Infinity;
        for (const playerTile of players) {
            for (const tile of tiles) {
                best = Math.min(best, manhattan(playerTile, tile));
            }
        }
        return distanceOrFallback(best);
    }

    function minAssignmentDistance(sources, targets) {
        if (sources.length === 0) {
            return 0;
        }
        if (targets.length === 0) {
            return sources.length * 64;
        }
        if (targets.length >= sources.length && sources.length <= 10 && targets.length <= 20) {
            const memo = new Map();
            const dfs = (sourceIndex, usedMask) => {
                if (sourceIndex >= sources.length) {
                    return 0;
                }
                const key = `${sourceIndex}:${usedMask}`;
                const cached = memo.get(key);
                if (cached !== undefined) {
                    return cached;
                }
                let best = Infinity;
                for (let targetIndex = 0; targetIndex < targets.length; targetIndex++) {
                    const targetBit = 1 << targetIndex;
                    if ((usedMask & targetBit) !== 0) {
                        continue;
                    }
                    best = Math.min(
                        best,
                        manhattan(sources[sourceIndex], targets[targetIndex]) + dfs(sourceIndex + 1, usedMask | targetBit)
                    );
                }
                const result = distanceOrFallback(best);
                memo.set(key, result);
                return result;
            };
            return dfs(0, 0);
        }

        const pairs = [];
        for (let sourceIndex = 0; sourceIndex < sources.length; sourceIndex++) {
            for (let targetIndex = 0; targetIndex < targets.length; targetIndex++) {
                pairs.push({
                    sourceIndex,
                    targetIndex,
                    distance: manhattan(sources[sourceIndex], targets[targetIndex]),
                });
            }
        }
        pairs.sort((left, right) => left.distance - right.distance);
        const usedSources = new Set();
        const usedTargets = new Set();
        let total = 0;
        for (const pair of pairs) {
            if (usedSources.has(pair.sourceIndex) || usedTargets.has(pair.targetIndex)) {
                continue;
            }
            usedSources.add(pair.sourceIndex);
            usedTargets.add(pair.targetIndex);
            total += pair.distance;
            if (usedSources.size === sources.length) {
                break;
            }
        }
        return total + (sources.length - usedSources.size) * 64;
    }

    function cellHasBlockingObject(tile, condition) {
        const offset = tile * STRIDE_OBJ;
        for (let word = 0; word < STRIDE_OBJ; word++) {
            let allowed = 0;
            if (condition && condition[1] && condition[1].data) {
                allowed |= condition[1].data[word] | 0;
            }
            if (condition && condition[2] && condition[2].data) {
                allowed |= condition[2].data[word] | 0;
            }
            if (playerMask && playerMask.data) {
                allowed |= playerMask.data[word] | 0;
            }
            if ((level.objects[offset + word] & nonBackgroundWords[word] & ~allowed) !== 0) {
                return true;
            }
        }
        return false;
    }

    function buildTargetLinePresence(condition) {
        targetRowPresence.fill(0);
        targetColPresence.fill(0);
        let targetCount = 0;
        for (let tile = 0; tile < level.n_tiles; tile++) {
            if (!matchesMask(condition[2], condition[5], tile)) {
                continue;
            }
            targetRowPresence[tileY(tile)] = 1;
            targetColPresence[tileX(tile)] = 1;
            targetCount++;
        }
        return targetCount;
    }

    function rowColAlignmentPenalty(unsatisfied, condition) {
        buildTargetLinePresence(condition);
        let penalty = 0;
        for (const tile of unsatisfied) {
            const rowHasTarget = targetRowPresence[tileY(tile)] !== 0;
            const colHasTarget = targetColPresence[tileX(tile)] !== 0;
            if (!rowHasTarget && !colHasTarget) {
                penalty += 2;
            } else if (!rowHasTarget || !colHasTarget) {
                penalty += 1;
            }
        }
        return penalty;
    }

    function nearestTargetLineDistance(tile, targetCount) {
        if (targetCount === 0) {
            return 64;
        }
        const x = tileX(tile);
        const y = tileY(tile);
        if (targetRowPresence[y] !== 0 || targetColPresence[x] !== 0) {
            return 0;
        }
        let best = 64;
        for (let row = 0; row < level.height; row++) {
            if (targetRowPresence[row] !== 0) {
                best = Math.min(best, Math.abs(y - row));
            }
        }
        for (let col = 0; col < level.width; col++) {
            if (targetColPresence[col] !== 0) {
                best = Math.min(best, Math.abs(x - col));
            }
        }
        return best;
    }

    function targetLineDistancePenalty(unsatisfied, condition) {
        const targetCount = buildTargetLinePresence(condition);
        let penalty = 0;
        for (const tile of unsatisfied) {
            penalty += nearestTargetLineDistance(tile, targetCount);
        }
        return penalty;
    }

    function clearPathBetween(left, right, condition) {
        const leftX = tileX(left);
        const leftY = tileY(left);
        const rightX = tileX(right);
        const rightY = tileY(right);
        const dx = Math.sign(rightX - leftX);
        const dy = Math.sign(rightY - leftY);
        let x = leftX + dx;
        let y = leftY + dy;
        while (x !== rightX || y !== rightY) {
            if (cellHasBlockingObject(tileIndexAt(x, y), condition)) {
                return false;
            }
            x += dx;
            y += dy;
        }
        return true;
    }

    function hasClearAlignedTarget(tile, targets, condition) {
        const x = tileX(tile);
        const y = tileY(tile);
        for (const target of targets) {
            if ((tileX(target) === x || tileY(target) === y) && clearPathBetween(tile, target, condition)) {
                return true;
            }
        }
        return false;
    }

    function clearPathPenalty(unsatisfied, targets, condition) {
        if (targets.length === 0) {
            return unsatisfied.length * 16;
        }
        let penalty = 0;
        buildTargetLinePresence(condition);
        for (const tile of unsatisfied) {
            const aligned = targetRowPresence[tileY(tile)] !== 0 || targetColPresence[tileX(tile)] !== 0;
            if (!aligned) {
                penalty += 4;
            } else if (!hasClearAlignedTarget(tile, targets, condition)) {
                penalty += 2;
            }
        }
        return penalty;
    }

    function plausibleTargetCount(targets, condition) {
        let count = 0;
        for (const target of targets) {
            if (!cellHasBlockingObject(target, condition)) {
                count++;
            }
        }
        return count;
    }

    function rowColGreedyAssignmentDistance(sources, targets) {
        if (sources.length === 0) {
            return 0;
        }
        if (targets.length === 0) {
            return sources.length * 64;
        }
        const pairs = [];
        for (let sourceIndex = 0; sourceIndex < sources.length; sourceIndex++) {
            const source = sources[sourceIndex];
            const sourceX = tileX(source);
            const sourceY = tileY(source);
            for (let targetIndex = 0; targetIndex < targets.length; targetIndex++) {
                const target = targets[targetIndex];
                if (tileX(target) !== sourceX && tileY(target) !== sourceY) {
                    continue;
                }
                pairs.push({
                    sourceIndex,
                    targetIndex,
                    distance: manhattan(source, target),
                });
            }
        }
        pairs.sort((left, right) => left.distance - right.distance);
        const usedSources = new Set();
        const usedTargets = new Set();
        let total = 0;
        for (const pair of pairs) {
            if (usedSources.has(pair.sourceIndex) || usedTargets.has(pair.targetIndex)) {
                continue;
            }
            usedSources.add(pair.sourceIndex);
            usedTargets.add(pair.targetIndex);
            total += pair.distance;
        }
        for (let sourceIndex = 0; sourceIndex < sources.length; sourceIndex++) {
            if (usedSources.has(sourceIndex)) {
                continue;
            }
            total += bestManhattan(sources[sourceIndex], targets);
        }
        return total;
    }

    function pushAccessPenalty(unsatisfied, targets, condition) {
        if (targets.length === 0) {
            return unsatisfied.length * 16;
        }
        let penalty = 0;
        const directions = [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ];
        for (const tile of unsatisfied) {
            const x = tileX(tile);
            const y = tileY(tile);
            const currentDistance = bestManhattan(tile, targets);
            let anyPush = false;
            let usefulPush = false;
            for (const [dx, dy] of directions) {
                const fromX = x - dx;
                const fromY = y - dy;
                const toX = x + dx;
                const toY = y + dy;
                if (!inBounds(fromX, fromY) || !inBounds(toX, toY)) {
                    continue;
                }
                const fromTile = tileIndexAt(fromX, fromY);
                const toTile = tileIndexAt(toX, toY);
                if (cellHasBlockingObject(fromTile, condition) || cellHasBlockingObject(toTile, condition)) {
                    continue;
                }
                anyPush = true;
                if (bestManhattan(toTile, targets) < currentDistance) {
                    usefulPush = true;
                    break;
                }
            }
            if (!anyPush) {
                penalty += 12;
            } else if (!usefulPush) {
                penalty += 4;
            }
        }
        return penalty;
    }

    function deadPositionPenalty(unsatisfied, condition) {
        let penalty = 0;
        buildTargetLinePresence(condition);
        for (const tile of unsatisfied) {
            const horizontalBlocked = adjacentCellBlocked(tile, -1, 0, condition) || adjacentCellBlocked(tile, 1, 0, condition);
            const verticalBlocked = adjacentCellBlocked(tile, 0, -1, condition) || adjacentCellBlocked(tile, 0, 1, condition);
            if (horizontalBlocked && verticalBlocked) {
                penalty += 32;
            } else if ((horizontalBlocked || verticalBlocked) &&
                targetRowPresence[tileY(tile)] === 0 &&
                targetColPresence[tileX(tile)] === 0) {
                penalty += 8;
            }
        }
        return penalty;
    }

    function obstacleDistanceField(mask, aggregate, condition, distances) {
        let head = 0;
        let tail = 0;
        for (let tile = 0; tile < level.n_tiles; tile++) {
            if (matchesMask(mask, aggregate, tile)) {
                distances[tile] = 0;
                obstacleQueue[tail++] = tile;
            } else {
                distances[tile] = Infinity;
            }
        }
        while (head < tail) {
            const tile = obstacleQueue[head++];
            const nextDistance = distances[tile] + 1;
            const x = (tile / level.height) | 0;
            const y = tile % level.height;
            if (x > 0) {
                const next = (x - 1) * level.height + y;
                if (distances[next] === Infinity && !cellHasBlockingObject(next, condition)) {
                    distances[next] = nextDistance;
                    obstacleQueue[tail++] = next;
                }
            }
            if (x + 1 < level.width) {
                const next = (x + 1) * level.height + y;
                if (distances[next] === Infinity && !cellHasBlockingObject(next, condition)) {
                    distances[next] = nextDistance;
                    obstacleQueue[tail++] = next;
                }
            }
            if (y > 0) {
                const next = x * level.height + y - 1;
                if (distances[next] === Infinity && !cellHasBlockingObject(next, condition)) {
                    distances[next] = nextDistance;
                    obstacleQueue[tail++] = next;
                }
            }
            if (y + 1 < level.height) {
                const next = x * level.height + y + 1;
                if (distances[next] === Infinity && !cellHasBlockingObject(next, condition)) {
                    distances[next] = nextDistance;
                    obstacleQueue[tail++] = next;
                }
            }
        }
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

    function winconditionDistanceHeuristic() {
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

        if (score > 0 && playerMask && playerMask.data) {
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

    function allOnCountHeuristic() {
        const condition = singleAllOnCondition();
        if (!condition) {
            return winconditionDistanceHeuristic();
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        return unsatisfied.length * 10;
    }

    function allOnRowColTiebreakHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        return base + rowColAlignmentPenalty(unsatisfied, condition) / 1024;
    }

    function allOnLineDistanceHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        return base + targetLineDistancePenalty(unsatisfied, condition);
    }

    function allOnClearPathHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        return base + clearPathPenalty(unsatisfied, targets, condition);
    }

    function allOnGoalCoverageHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        const usableTargets = plausibleTargetCount(targets, condition);
        const shortage = Math.max(0, unsatisfied.length - usableTargets);
        return base + shortage * 32;
    }

    function allOnRowColMatchingHeuristic() {
        const condition = singleAllOnCondition();
        if (!condition) {
            return winconditionDistanceHeuristic();
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        return unsatisfied.length * 10 + rowColGreedyAssignmentDistance(unsatisfied, targets);
    }

    function allOnPlayerNearestTiebreakHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        return base + Math.min(minPlayerDistanceToTiles(unsatisfied), 16) / 1024;
    }

    function allOnPushAccessHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        return base + pushAccessPenalty(unsatisfied, targets, condition);
    }

    function allOnDeadPositionHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        return base + deadPositionPenalty(unsatisfied, condition);
    }

    function allOnMatchingHeuristic() {
        const condition = singleAllOnCondition();
        if (!condition) {
            return winconditionDistanceHeuristic();
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        return unsatisfied.length * 10 + minAssignmentDistance(unsatisfied, targets);
    }

    function allOnPlayerTiebreakHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (base <= 0 || !condition) {
            return base;
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        const assignment = minAssignmentDistance(unsatisfied, targets);
        const player = minPlayerDistanceToTiles(unsatisfied);
        return base + Math.min(assignment, 255) / 1024 + Math.min(player, 16) / 65536;
    }

    function allOnMinMatchingHeuristic() {
        const base = winconditionDistanceHeuristic();
        const condition = singleAllOnCondition();
        if (!condition) {
            return base;
        }
        return Math.min(base, allOnMatchingHeuristic());
    }

    function allOnObstacleHeuristic() {
        const condition = singleAllOnCondition();
        if (!condition) {
            return winconditionDistanceHeuristic();
        }
        obstacleDistanceField(condition[2], condition[5], condition, obstacleDistances);
        let score = 0;
        for (const tile of collectUnsatisfiedAllOnTiles(condition)) {
            score += 10 + distanceOrFallback(obstacleDistances[tile]);
        }
        return score;
    }

    function allOnPlayerHeuristic() {
        const condition = singleAllOnCondition();
        if (!condition) {
            return winconditionDistanceHeuristic();
        }
        const unsatisfied = collectUnsatisfiedAllOnTiles(condition);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        return unsatisfied.length * 10 +
            minAssignmentDistance(unsatisfied, targets) +
            Math.min(minPlayerDistanceToTiles(unsatisfied), 16);
    }

    function adjacentCellBlocked(tile, dx, dy, condition) {
        const x = (tile / level.height) | 0;
        const y = tile % level.height;
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= level.width || ny >= level.height) {
            return true;
        }
        return cellHasBlockingObject(nx * level.height + ny, condition);
    }

    function allOnDeadlockHeuristic() {
        const condition = singleAllOnCondition();
        if (!condition) {
            return winconditionDistanceHeuristic();
        }
        let deadlocks = 0;
        for (const tile of collectUnsatisfiedAllOnTiles(condition)) {
            const horizontal = adjacentCellBlocked(tile, -1, 0, condition) || adjacentCellBlocked(tile, 1, 0, condition);
            const vertical = adjacentCellBlocked(tile, 0, -1, condition) || adjacentCellBlocked(tile, 0, 1, condition);
            if (horizontal && vertical) {
                deadlocks++;
            }
        }
        return allOnMatchingHeuristic() + deadlocks * 32;
    }

    function someOnMinHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== 0 || isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        const sources = collectMatchingTiles(condition[1], condition[4]);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        let best = Infinity;
        for (const source of sources) {
            if (matchesMask(condition[2], condition[5], source)) {
                return 0;
            }
            best = Math.min(best, bestManhattan(source, targets));
        }
        return distanceOrFallback(best);
    }

    function someOnPlayerHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== 0 || isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        const sources = collectMatchingTiles(condition[1], condition[4]);
        const targets = collectMatchingTiles(condition[2], condition[5]);
        const players = playerMask && playerMask.data ? collectMatchingTiles(playerMask, playerAggregate) : [];
        let best = Infinity;
        for (const source of sources) {
            if (matchesMask(condition[2], condition[5], source)) {
                return 0;
            }
            const sourceToTarget = bestManhattan(source, targets);
            if (players.length === 0) {
                best = Math.min(best, sourceToTarget);
                continue;
            }
            for (const playerTile of players) {
                best = Math.min(best, manhattan(playerTile, source) + sourceToTarget);
            }
        }
        return distanceOrFallback(best);
    }

    function someOnObstacleHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== 0 || isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        obstacleDistanceField(condition[2], condition[5], condition, obstacleDistances);
        let best = Infinity;
        for (const source of collectMatchingTiles(condition[1], condition[4])) {
            if (matchesMask(condition[2], condition[5], source)) {
                return 0;
            }
            best = Math.min(best, obstacleDistances[source]);
        }
        return distanceOrFallback(best);
    }

    function noOnCountHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== -1 || isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        return collectOverlapTiles(condition).length * 10;
    }

    function noOnEscapeHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== -1 || isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        const escapeTiles = [];
        for (let tile = 0; tile < level.n_tiles; tile++) {
            if (!matchesMask(condition[2], condition[5], tile)) {
                escapeTiles.push(tile);
            }
        }
        let score = 0;
        for (const offender of collectOverlapTiles(condition)) {
            score += 10 + bestManhattan(offender, escapeTiles);
        }
        return score;
    }

    function noOnPlayerHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== -1 || isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        const offenders = collectOverlapTiles(condition);
        return noOnEscapeHeuristic() + Math.min(minPlayerDistanceToTiles(offenders), 16);
    }

    function somePlainExistsHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== 0 || !isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        return collectMatchingTiles(condition[1], condition[4]).length > 0 ? 0 : 64;
    }

    function somePlainPlayerHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== 0 || !isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        const sources = collectMatchingTiles(condition[1], condition[4]);
        if (sources.length === 0) {
            return 64;
        }
        return Math.min(minPlayerDistanceToTiles(sources), 16);
    }

    function noPlainCountHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== -1 || !isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        return collectMatchingTiles(condition[1], condition[4]).length * 10;
    }

    function noPlainClusterHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== -1 || !isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        let count = 0;
        let adjacentPairs = 0;
        for (let tile = 0; tile < level.n_tiles; tile++) {
            if (!matchesMask(condition[1], condition[4], tile)) {
                continue;
            }
            count++;
            const x = (tile / level.height) | 0;
            const y = tile % level.height;
            if (x + 1 < level.width && matchesMask(condition[1], condition[4], (x + 1) * level.height + y)) {
                adjacentPairs++;
            }
            if (y + 1 < level.height && matchesMask(condition[1], condition[4], x * level.height + y + 1)) {
                adjacentPairs++;
            }
        }
        return Math.max(0, count * 12 - Math.min(count * 4, adjacentPairs * 3));
    }

    function noPlainPlayerHeuristic() {
        const condition = singleCondition();
        if (!condition || condition[0] !== -1 || !isPlainCondition(condition)) {
            return winconditionDistanceHeuristic();
        }
        const remaining = collectMatchingTiles(condition[1], condition[4]);
        return remaining.length * 10 + Math.min(minPlayerDistanceToTiles(remaining), 16);
    }

    function heuristic() {
        switch (heuristicName) {
            case 'zero':
                return 0;
            case 'all-on-count':
                return allOnCountHeuristic();
            case 'all-on-rowcol-tiebreak':
                return allOnRowColTiebreakHeuristic();
            case 'all-on-line-distance':
                return allOnLineDistanceHeuristic();
            case 'all-on-clear-path':
                return allOnClearPathHeuristic();
            case 'all-on-goal-coverage':
                return allOnGoalCoverageHeuristic();
            case 'all-on-rowcol-matching':
                return allOnRowColMatchingHeuristic();
            case 'all-on-player-nearest-tiebreak':
                return allOnPlayerNearestTiebreakHeuristic();
            case 'all-on-push-access':
                return allOnPushAccessHeuristic();
            case 'all-on-dead-position':
                return allOnDeadPositionHeuristic();
            case 'all-on-player-tiebreak':
                return allOnPlayerTiebreakHeuristic();
            case 'all-on-min-matching':
                return allOnMinMatchingHeuristic();
            case 'all-on-matching':
                return allOnMatchingHeuristic();
            case 'all-on-obstacle':
                return allOnObstacleHeuristic();
            case 'all-on-player':
                return allOnPlayerHeuristic();
            case 'all-on-deadlock':
                return allOnDeadlockHeuristic();
            case 'some-on-min':
                return someOnMinHeuristic();
            case 'some-on-player':
                return someOnPlayerHeuristic();
            case 'some-on-obstacle':
                return someOnObstacleHeuristic();
            case 'no-on-count':
                return noOnCountHeuristic();
            case 'no-on-escape':
                return noOnEscapeHeuristic();
            case 'no-on-player':
                return noOnPlayerHeuristic();
            case 'some-plain-exists':
                return somePlainExistsHeuristic();
            case 'some-plain-player':
                return somePlainPlayerHeuristic();
            case 'no-plain-count':
                return noPlainCountHeuristic();
            case 'no-plain-cluster':
                return noPlainClusterHeuristic();
            case 'no-plain-player':
                return noPlainPlayerHeuristic();
            case 'winconditions':
            default:
                return allOnClearPathHeuristic();
        }
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
        if (verifyZobrist) {
            const expected = computeZobristBoardHash(level.objects, zobristTables.lo, zobristTables.hi);
            if ((level.solverZobristLo | 0) !== expected.lo || (level.solverZobristHi | 0) !== expected.hi) {
                throw new Error(`Incremental Zobrist hash drifted: ${level.solverZobristLo}:${level.solverZobristHi} !== ${expected.lo}:${expected.hi}`);
            }
        }
        return zobristHasher(
            level.solverZobristLo | 0,
            level.solverZobristHi | 0,
            flagsForHash(),
            usesRandom && RandomGen && RandomGen._state
        );
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
            zobristTableId,
            zobristLo: level.solverZobristLo | 0,
            zobristHi: level.solverZobristHi | 0,
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
        if (snapshot.zobristTableId === zobristTableId) {
            level.solverZobristLo = snapshot.zobristLo | 0;
            level.solverZobristHi = snapshot.zobristHi | 0;
        } else {
            recomputeZobrist();
        }
        if (level.movements && level.movements.length === movementWordCount) {
            level.movements.fill(0);
        }
        if (state.rigid) {
            zeroBitVecArray(level.rigidMovementAppliedMask);
            zeroBitVecArray(level.rigidGroupIndexMask);
        }
        // Solver turns call state.calculateRowColMasks(level) at processInput start,
        // so keep post-turn row/column scratch available for heuristics until then.
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

    function matchesSnapshot(snapshot) {
        return level &&
            level.objects &&
            level.objects.length === objectWordCount &&
            level.width === width &&
            level.height === height &&
            int32ArraysEqual(level.objects, snapshot.objects) &&
            (!usesRandom || randomStateEqual(RandomGen && RandomGen._state, snapshot.random && snapshot.random.state)) &&
            curlevel === snapshot.curlevel &&
            titleScreen === snapshot.titleScreen &&
            textMode === snapshot.textMode &&
            titleMode === snapshot.titleMode &&
            titleSelection === snapshot.titleSelection &&
            (!textMode || messagetext === snapshot.messagetext) &&
            winning === snapshot.winning &&
            againing === snapshot.againing;
    }

    return {
        capture,
        restore,
        hash,
        matchesSnapshot,
        heuristic,
        heuristicName,
        hashMode: usesRandom ? 'incremental_zobrist_with_rng' : 'incremental_zobrist',
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

class VisitedStateBuckets {
    constructor(nodes, solverOps) {
        this.nodes = nodes;
        this.solverOps = solverOps;
        this.buckets = new Map();
        this.size = 0;
        this.collisions = 0;
    }

    addInitial(key, nodeIndex) {
        this.buckets.set(key, { depth: 0, index: nodeIndex });
        this.size = 1;
    }

    find(key, depth) {
        const bucket = this.buckets.get(key);
        if (bucket === undefined) {
            return { duplicate: false, entry: null, collided: false };
        }
        const entries = Array.isArray(bucket) ? bucket : [bucket];
        for (const entry of entries) {
            if (!this.solverOps.matchesSnapshot(this.nodes[entry.index].snapshot)) {
                continue;
            }
            if (entry.depth <= depth) {
                return { duplicate: true, entry, collided: false };
            }
            return { duplicate: false, entry, collided: false };
        }
        return { duplicate: false, entry: null, collided: true };
    }

    record(key, depth, nodeIndex, existingEntry, collided) {
        if (existingEntry) {
            existingEntry.depth = depth;
            existingEntry.index = nodeIndex;
            return;
        }
        const entry = { depth, index: nodeIndex };
        const bucket = this.buckets.get(key);
        if (bucket === undefined) {
            this.buckets.set(key, entry);
        } else if (Array.isArray(bucket)) {
            bucket.push(entry);
        } else {
            this.buckets.set(key, [bucket, entry]);
        }
        this.size++;
        if (collided) {
            this.collisions++;
        }
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
        hash_collisions: 0,
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
        'hash_collisions',
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
    const useHashBuckets = process.env.PUZZLESCRIPT_DISABLE_HASH_BUCKETS !== '1';
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
    const initialSnapshot = createSolverLevelSpecialization(options).capture();
    const strategy = options.strategy || DEFAULT_STRATEGY;

    const runMode = (mode, modeDeadline) => {
        const modeResult = createSolverResult(game, levelIndex, timeoutMs, compileMs);
        modeResult.load_ms = result.load_ms;
        modeResult.strategy = mode;
        modeResult.heuristic = mode === 'bfs' ? 'zero' : (options.solverHeuristic || DEFAULT_SOLVER_HEURISTIC);
        const solverOps = createSolverLevelSpecialization(options);
        modeResult.hash_mode = solverOps.hashMode;
        modeResult.snapshot_mode = solverOps.snapshotMode;
        const initialRestoreStart = performance.now();
        solverOps.restore(initialSnapshot);
        modeResult.clone_ms += performance.now() - initialRestoreStart;
        const initialSnapshotStart = performance.now();
        const nodes = [{ snapshot: solverOps.capture(), parent: -1, input: null, depth: 0 }];
        modeResult.snapshot_ms += performance.now() - initialSnapshotStart;
        const visited = useHashBuckets ? new VisitedStateBuckets(nodes, solverOps) : null;
        const bestDepth = useHashBuckets ? null : new Map();
        const hashStart = performance.now();
        const initialHash = solverOps.hash();
        if (useHashBuckets) {
            visited.addInitial(initialHash, 0);
        } else {
            bestDepth.set(initialHash, 0);
        }
        modeResult.hash_ms += performance.now() - hashStart;
        modeResult.unique_states = useHashBuckets ? visited.size : bestDepth.size;
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
                let visitedMatch = null;
                if (useHashBuckets) {
                    visitedMatch = visited.find(key, childDepth);
                    if (visitedMatch.duplicate) {
                        modeResult.duplicates++;
                        continue;
                    }
                } else {
                    if (bestDepth.has(key) && bestDepth.get(key) <= childDepth) {
                        modeResult.duplicates++;
                        continue;
                    }
                    bestDepth.set(key, childDepth);
                }
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
                if (useHashBuckets) {
                    visited.record(key, childDepth, childIndex, visitedMatch.entry, visitedMatch.collided);
                    modeResult.unique_states = visited.size;
                    modeResult.hash_collisions = visited.collisions;
                } else {
                    modeResult.unique_states = bestDepth.size;
                }
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

    const runAdaptivePortfolio = (modeDeadline) => {
        const modeResult = createSolverResult(game, levelIndex, timeoutMs, compileMs);
        modeResult.load_ms = result.load_ms;
        modeResult.strategy = 'portfolio';
        modeResult.heuristic = 'mixed';
        const solverOps = createSolverLevelSpecialization(options);
        modeResult.hash_mode = solverOps.hashMode;
        modeResult.snapshot_mode = solverOps.snapshotMode;
        modeResult.heuristic = `mixed:${solverOps.heuristicName}`;
        const initialRestoreStart = performance.now();
        solverOps.restore(initialSnapshot);
        modeResult.clone_ms += performance.now() - initialRestoreStart;
        const initialHeuristicStart = performance.now();
        const initialHeuristic = solverOps.heuristic();
        modeResult.heuristic_ms += performance.now() - initialHeuristicStart;
        const initialSnapshotStart = performance.now();
        const nodes = [{
            snapshot: solverOps.capture(),
            parent: -1,
            input: null,
            depth: 0,
            expanded: false,
        }];
        modeResult.snapshot_ms += performance.now() - initialSnapshotStart;
        const visited = useHashBuckets ? new VisitedStateBuckets(nodes, solverOps) : null;
        const bestDepth = useHashBuckets ? null : new Map();
        const hashStart = performance.now();
        const initialHash = solverOps.hash();
        if (useHashBuckets) {
            visited.addInitial(initialHash, 0);
        } else {
            bestDepth.set(initialHash, 0);
        }
        modeResult.hash_ms += performance.now() - hashStart;
        modeResult.unique_states = useHashBuckets ? visited.size : bestDepth.size;

        const portfolioModes = [
            { name: 'wa2', heap: new MinHeap(), expansionSlice: 128 },
            { name: 'bfs', heap: new MinHeap(), expansionSlice: 128 },
            { name: 'wa8', heap: new MinHeap(), expansionSlice: 128 },
            { name: 'greedy', heap: new MinHeap(), expansionSlice: 64 },
        ];
        if (Number.isFinite(options.portfolioBfsMs)) {
            const bfsIndex = portfolioModes.findIndex((mode) => mode.name === 'bfs');
            const bfsMode = portfolioModes.splice(bfsIndex, 1)[0];
            bfsMode.expansionSlice = Math.max(1, options.portfolioBfsMs);
            portfolioModes.unshift(bfsMode);
        }
        let tie = 0;
        for (const mode of portfolioModes) {
            mode.heap.push({ priority: priorityForPortfolioMode(mode.name, 0, initialHeuristic), tie: tie++, index: 0 });
        }
        let totalFrontier = portfolioModes.length;
        modeResult.max_frontier = totalFrontier;
        const actions = solverActionsForGame();
        let modeIndex = 0;
        let activeMode = portfolioModes[0];
        let sliceExpansionsLeft = activeMode.expansionSlice;
        let lockedToWeightedAstar = false;
        const weightedMode = portfolioModes.find((mode) => mode.name === 'wa2');

        const hasFrontier = () => totalFrontier > 0;
        const advanceMode = () => {
            if (lockedToWeightedAstar && weightedMode && weightedMode.heap.length > 0) {
                activeMode = weightedMode;
                sliceExpansionsLeft = weightedMode.expansionSlice;
                return true;
            }
            for (let attempt = 0; attempt < portfolioModes.length; attempt++) {
                modeIndex = (modeIndex + 1) % portfolioModes.length;
                const candidate = portfolioModes[modeIndex];
                if (candidate.heap.length > 0) {
                    activeMode = candidate;
                    sliceExpansionsLeft = activeMode.expansionSlice;
                    return true;
                }
            }
            return false;
        };

        while (hasFrontier()) {
            if (Date.now() >= modeDeadline) {
                modeResult.status = 'timeout';
                break;
            }
            if (activeMode.heap.length === 0 || sliceExpansionsLeft <= 0) {
                if (!advanceMode()) {
                    break;
                }
            }

            const queueStart = performance.now();
            const entry = activeMode.heap.pop();
            modeResult.queue_ms += performance.now() - queueStart;
            if (entry === null) {
                continue;
            }
            totalFrontier--;
            const node = nodes[entry.index];
            if (node.expanded) {
                continue;
            }
            node.expanded = true;
            sliceExpansionsLeft--;
            modeResult.expanded++;
            if (!lockedToWeightedAstar && modeResult.expanded >= 128 && modeResult.step_ms / modeResult.generated > 0.05) {
                lockedToWeightedAstar = true;
                if (activeMode !== weightedMode && weightedMode && weightedMode.heap.length > 0) {
                    activeMode = weightedMode;
                    sliceExpansionsLeft = weightedMode.expansionSlice;
                }
            }

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
                    modeResult.strategy = `portfolio:${activeMode.name}`;
                    return modeResult;
                }
                if (!stepResult.changed) {
                    continue;
                }

                const hashStart2 = performance.now();
                const key = solverOps.hash();
                modeResult.hash_ms += performance.now() - hashStart2;
                const childDepth = node.depth + 1;
                let visitedMatch = null;
                if (useHashBuckets) {
                    visitedMatch = visited.find(key, childDepth);
                    if (visitedMatch.duplicate) {
                        modeResult.duplicates++;
                        continue;
                    }
                } else {
                    if (bestDepth.has(key) && bestDepth.get(key) <= childDepth) {
                        modeResult.duplicates++;
                        continue;
                    }
                    bestDepth.set(key, childDepth);
                }
                const heuristicStart = performance.now();
                const childHeuristic = solverOps.heuristic();
                modeResult.heuristic_ms += performance.now() - heuristicStart;
                const snapshotStart = performance.now();
                const snapshot = solverOps.capture();
                modeResult.snapshot_ms += performance.now() - snapshotStart;
                nodes.push({
                    snapshot,
                    parent: entry.index,
                    input: action.token,
                    depth: childDepth,
                    expanded: false,
                });
                const childIndex = nodes.length - 1;
                if (useHashBuckets) {
                    visited.record(key, childDepth, childIndex, visitedMatch.entry, visitedMatch.collided);
                    modeResult.unique_states = visited.size;
                    modeResult.hash_collisions = visited.collisions;
                } else {
                    modeResult.unique_states = bestDepth.size;
                }
                const queueStart2 = performance.now();
                const queueModes = lockedToWeightedAstar && weightedMode ? [weightedMode] : portfolioModes;
                for (const mode of queueModes) {
                    mode.heap.push({
                        priority: priorityForPortfolioMode(mode.name, childDepth, childHeuristic),
                        tie: tie++,
                        index: childIndex,
                    });
                }
                totalFrontier += queueModes.length;
                modeResult.queue_ms += performance.now() - queueStart2;
                modeResult.max_frontier = Math.max(modeResult.max_frontier, totalFrontier);
            }
        }

        modeResult.solution_length = modeResult.solution.length;
        modeResult.elapsed_ms = Date.now() - searchStarted;
        return modeResult;
    };

    if (strategy === 'portfolio') {
        return runAdaptivePortfolio(deadline);
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
        hash_collisions: 0,
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
            hash_collisions: 0,
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
        hash_collisions: 0,
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
        out.hash_collisions += result.hash_collisions || 0;
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
