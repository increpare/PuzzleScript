'use strict';

// Pure static-analysis helpers over a compiled PuzzleScript game state.
//
// All functions are side-effect free and read only the `state`, `cell`, `row`,
// `rule`, and `mask` arguments passed in. They depend on the PuzzleScript
// globals `BitVec` and `STRIDE_OBJ`, which are populated by the engine when
// `loadPuzzleScript()` + `compile()` runs (see
// `src/tests/js_oracle/lib/puzzlescript_node_env.js`).
//
// Originally inlined into both `run_solver_tests_js.js` and
// `analyze_solver_static_relationships.js`; consolidated here so the two files
// can't drift on the algorithm.

function maskHasBits(mask) {
    return Boolean(mask && mask.data && !mask.iszero());
}

function masksIntersect(left, right) {
    if (!left || !right || !left.data || !right.data) {
        return false;
    }
    const length = Math.min(left.data.length, right.data.length);
    for (let word = 0; word < length; word++) {
        if ((left.data[word] & right.data[word]) !== 0) {
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

function cloneMask(mask) {
    const result = new BitVec(STRIDE_OBJ);
    if (mask && mask.data) {
        result.ior(mask);
    }
    return result;
}

function iorIntersection(target, left, right) {
    if (!left || !right || !left.data || !right.data) {
        return;
    }
    const length = Math.min(target.data.length, left.data.length, right.data.length);
    for (let word = 0; word < length; word++) {
        target.data[word] |= left.data[word] & right.data[word];
    }
}

function objectPresenceMask(cell) {
    const mask = new BitVec(STRIDE_OBJ);
    if (!cell || !cell.objectsPresent || !cell.objectsPresent.data) {
        return mask;
    }
    mask.ior(cell.objectsPresent);
    for (const anyMask of cell.anyObjectsPresent || []) {
        mask.ior(anyMask);
    }
    return mask;
}

function rowObjectsSetMask(row) {
    const mask = new BitVec(STRIDE_OBJ);
    for (const cell of row) {
        if (cell && cell.replacement && cell.replacement.objectsSet) {
            mask.ior(cell.replacement.objectsSet);
        }
    }
    return mask;
}

function foreignSetMask(cell, excludedMask) {
    const mask = new BitVec(STRIDE_OBJ);
    if (!cell || !cell.replacement || !cell.replacement.objectsSet) {
        return mask;
    }
    mask.ior(cell.replacement.objectsSet);
    mask.iclear(objectPresenceMask(cell));
    if (excludedMask) {
        mask.iclear(excludedMask);
    }
    return mask;
}

function cellHasMovement(cell) {
    return Boolean(cell && cell.movementsPresent && !cell.movementsPresent.iszero());
}

function cellChangesObjects(cell) {
    return Boolean(cell && cell.replacement &&
        ((!cell.replacement.objectsSet.iszero()) || (!cell.replacement.objectsClear.iszero())));
}

function isCancelRule(rule) {
    return (rule.commands || []).some((command) => command && command[0] === 'cancel');
}

function movementMaskTouchesObjectMask(state, movementMask, objectMask) {
    if (!movementMask || !movementMask.data || !objectMask || !objectMask.data) {
        return false;
    }
    for (const objectName of state.idDict || []) {
        const object = state.objects && state.objects[objectName];
        if (!object || !objectMask.get(object.id)) {
            continue;
        }
        if (movementMask.getshiftor(0x1f, 5 * object.layer) !== 0) {
            return true;
        }
    }
    return false;
}

function cellChangesObjectMask(state, cell, objectMask) {
    if (!cell || !cell.replacement || !objectMask || !objectMask.data) {
        return false;
    }
    const present = objectPresenceMask(cell);
    if (masksIntersect(cell.replacement.objectsSet, objectMask)) {
        return true;
    }
    if (masksIntersect(present, objectMask) && masksIntersect(cell.replacement.objectsClear, objectMask)) {
        return true;
    }
    return masksIntersect(present, objectMask) &&
        (movementMaskTouchesObjectMask(state, cell.replacement.movementsSet, objectMask) ||
            movementMaskTouchesObjectMask(state, cell.replacement.movementsClear, objectMask));
}

// Returns the set of object types that, if present in a cell, would prevent
// the wincondition's actor objects from being moved/changed by any rule.
//
// `condition` is the wincondition tuple `[quantifier, sourceMask, targetMask, …]`.
// `options.playerMask` adds the player to the actor set when the condition's
// source intersects the player.
//
// Returned `consumed` is the set of objects that some rule would clear; the
// caller may want it for diagnostics. `blockers` already has it subtracted.
//
// Walks `state.rules + state.lateRules`. For each pattern row, finds cells
// whose `objectsPresent` intersects the actor mask and that either move,
// change objects, or live in a cancel rule. The two horizontal neighbours
// contribute blocker candidates via their `objectsMissing` constraints, via
// `objectsPresent` for cancel rules, or asymmetrically when the actor cell
// changes objects but the neighbour does not. Vertical adjacency is implicitly
// covered by PuzzleScript's auto-generated rule rotations.
function inferStaticBlockerMask(state, condition, options = {}) {
    const playerMask = options.playerMask || null;
    const blockers = new BitVec(STRIDE_OBJ);
    const consumed = new BitVec(STRIDE_OBJ);
    const actorMask = cloneMask(condition[1]);
    if (playerMask && playerMask.data && masksIntersect(condition[1], playerMask)) {
        actorMask.ior(playerMask);
    }

    for (const group of [...(state.rules || []), ...(state.lateRules || [])]) {
        for (const rule of group || []) {
            const cancelRule = isCancelRule(rule);
            for (const row of rule.patterns || []) {
                const rowSet = rowObjectsSetMask(row);
                for (let cellIndex = 0; cellIndex < row.length; cellIndex++) {
                    const cell = row[cellIndex];
                    if (!cell || !cell.objectsPresent) {
                        continue;
                    }
                    const cellPresent = objectPresenceMask(cell);
                    if (!masksIntersect(cellPresent, actorMask)) {
                        continue;
                    }
                    const actorMoves = cellHasMovement(cell);
                    const actorChanges = cellChangesObjects(cell);
                    if (!actorMoves && !actorChanges && !cancelRule) {
                        continue;
                    }
                    for (const neighborIndex of [cellIndex - 1, cellIndex + 1]) {
                        if (neighborIndex < 0 || neighborIndex >= row.length) {
                            continue;
                        }
                        const neighbor = row[neighborIndex];
                        if (!neighbor || !neighbor.objectsPresent) {
                            continue;
                        }
                        if ((actorMoves || actorChanges) && maskHasBits(neighbor.objectsMissing)) {
                            blockers.ior(neighbor.objectsMissing);
                        }
                        const neighborPresent = objectPresenceMask(neighbor);
                        if (neighbor.replacement && maskHasBits(neighborPresent)) {
                            const cleared = new BitVec(STRIDE_OBJ);
                            iorIntersection(cleared, neighborPresent, neighbor.replacement.objectsClear);
                            cleared.iclear(rowSet);
                            if (!cleared.iszero()) {
                                consumed.ior(cleared);
                            }
                        }
                        if (!maskHasBits(neighborPresent)) {
                            continue;
                        }
                        if (cancelRule) {
                            blockers.ior(neighborPresent);
                        } else if (actorChanges && !cellChangesObjects(neighbor)) {
                            blockers.ior(neighborPresent);
                        }
                    }
                }
            }
        }
    }

    blockers.iclear(consumed);
    blockers.iclear(condition[1]);
    blockers.iclear(condition[2]);
    if (playerMask && playerMask.data) {
        blockers.iclear(playerMask);
    }
    return { blockers, consumed };
}

module.exports = {
    maskHasBits,
    masksIntersect,
    masksEqual,
    cloneMask,
    iorIntersection,
    objectPresenceMask,
    rowObjectsSetMask,
    foreignSetMask,
    cellHasMovement,
    cellChangesObjects,
    isCancelRule,
    movementMaskTouchesObjectMask,
    cellChangesObjectMask,
    inferStaticBlockerMask,
};
