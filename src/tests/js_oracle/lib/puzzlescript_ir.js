'use strict';

function bitVecToArray(vec) {
    if (!vec || !vec.data) {
        return [];
    }
    return Array.from(vec.data);
}

function bitVecHasBits(vec) {
    if (!vec || !vec.data) {
        return false;
    }
    return vec.data.some(word => word !== 0);
}

function bitVecSetIndices(vec, maxIndex) {
    if (!vec || !vec.data) {
        return [];
    }
    const indices = [];
    const limit = typeof maxIndex === 'number' ? maxIndex : vec.data.length * 32;
    for (let wordIndex = 0; wordIndex < vec.data.length; wordIndex++) {
        let word = vec.data[wordIndex] >>> 0;
        while (word !== 0) {
            const bit = Math.clz32(word & -word) ^ 31;
            const index = wordIndex * 32 + bit;
            if (index < limit) {
                indices.push(index);
            }
            word &= word - 1;
        }
    }
    return indices;
}

function bitVecSetLayers(vec, layerCount) {
    if (!vec || !vec.data) {
        return [];
    }
    return Array.from(new Set(
        bitVecSetIndices(vec)
            .map(index => Math.floor(index / 5))
            .filter(layerIndex => typeof layerCount !== 'number' || layerIndex < layerCount)
    )).sort((a, b) => a - b);
}

function bitVecHasIndex(vec, index) {
    if (!vec || !vec.data || index < 0) {
        return false;
    }
    const wordIndex = index >> 5;
    const bitIndex = index & 31;
    if (wordIndex >= vec.data.length) {
        return false;
    }
    return ((vec.data[wordIndex] >>> 0) & (1 << bitIndex)) !== 0;
}

function bitVecSubsetOf(left, right) {
    const leftData = left && left.data ? left.data : [];
    const rightData = right && right.data ? right.data : [];
    const length = Math.max(leftData.length, rightData.length);
    for (let index = 0; index < length; index++) {
        const leftWord = (leftData[index] || 0) >>> 0;
        const rightWord = (rightData[index] || 0) >>> 0;
        if ((leftWord & ~rightWord) !== 0) {
            return false;
        }
    }
    return true;
}

function bitVecIntersects(left, right) {
    const leftData = left && left.data ? left.data : [];
    const rightData = right && right.data ? right.data : [];
    const length = Math.max(leftData.length, rightData.length);
    for (let index = 0; index < length; index++) {
        if ((((leftData[index] || 0) >>> 0) & ((rightData[index] || 0) >>> 0)) !== 0) {
            return true;
        }
    }
    return false;
}

function objectLayerById(state) {
    const layers = [];
    for (const objectEntry of Object.values(state.objects || {})) {
        layers[objectEntry.id] = objectEntry.layer;
    }
    return layers;
}

function patternImpossible(pattern, state, objectLayers) {
    if (!pattern || pattern === ellipsisPattern) {
        return false;
    }
    if (bitVecIntersects(pattern.objectsPresent, pattern.objectsMissing) ||
        bitVecIntersects(pattern.movementsPresent, pattern.movementsMissing)) {
        return true;
    }
    const layersSeen = new Set();
    for (const objectId of bitVecSetIndices(pattern.objectsPresent, state.objectCount)) {
        const layer = objectLayers[objectId];
        if (layer === undefined || layer < 0) {
            continue;
        }
        if (layersSeen.has(layer)) {
            return true;
        }
        layersSeen.add(layer);
    }
    return false;
}

function ruleImpossible(rule, state, objectLayers) {
    return rule.patterns.some(cellRow =>
        cellRow.some(pattern => patternImpossible(pattern, state, objectLayers))
    );
}

function replacementGuaranteedNoop(replacement, pattern) {
    if (!replacement || !pattern || pattern === ellipsisPattern) {
        return false;
    }
    if (bitVecHasBits(replacement.objectsSet) ||
        bitVecHasBits(replacement.movementsSet) ||
        bitVecHasBits(replacement.movementsLayerMask) ||
        bitVecHasBits(replacement.randomEntityMask) ||
        bitVecHasBits(replacement.randomDirMask)) {
        return false;
    }
    return bitVecSubsetOf(replacement.objectsClear, pattern.objectsMissing) &&
        bitVecSubsetOf(replacement.movementsClear, pattern.movementsMissing);
}

function movementBitPairs(vec, layerCount) {
    const pairs = [];
    const layers = typeof layerCount === 'number' ? layerCount : 0;
    for (let layerIndex = 0; layerIndex < layers; layerIndex++) {
        for (let movementBit = 0; movementBit < 5; movementBit++) {
            if (bitVecHasIndex(vec, layerIndex * 5 + movementBit)) {
                pairs.push([layerIndex, movementBit]);
            }
        }
    }
    return pairs;
}

function orBitVecArrays(vecs) {
    const result = [];
    for (const vec of vecs || []) {
        if (!vec || !vec.data) {
            continue;
        }
        for (let index = 0; index < vec.data.length; index++) {
            result[index] = (result[index] || 0) | vec.data[index];
        }
    }
    return result;
}

function directionDeltaHint(direction) {
    switch (direction) {
        case 1:
            return { dx: 0, dy: -1 };
        case 2:
            return { dx: 0, dy: 1 };
        case 4:
            return { dx: -1, dy: 0 };
        case 8:
            return { dx: 1, dy: 0 };
        default:
            return { dx: 0, dy: 0 };
    }
}

function serializeReplacement(replacement) {
    if (!replacement) {
        return null;
    }
    return {
        objects_clear: bitVecToArray(replacement.objectsClear),
        objects_set: bitVecToArray(replacement.objectsSet),
        movements_clear: bitVecToArray(replacement.movementsClear),
        movements_set: bitVecToArray(replacement.movementsSet),
        movements_layer_mask: bitVecToArray(replacement.movementsLayerMask),
        random_entity_mask: bitVecToArray(replacement.randomEntityMask),
        random_dir_mask: bitVecToArray(replacement.randomDirMask),
    };
}

function serializeRulePlanReplacement(rule, rowIndex, cellIndex, pattern, state) {
    const replacement = pattern.replacement;
    const layerCount = state.LAYER_COUNT || state.collisionLayers.length;
    const touchesObjects = bitVecHasBits(replacement.objectsClear) ||
        bitVecHasBits(replacement.objectsSet) ||
        bitVecHasBits(replacement.randomEntityMask);
    const touchesMovements = bitVecHasBits(replacement.movementsClear) ||
        bitVecHasBits(replacement.movementsSet) ||
        bitVecHasBits(replacement.movementsLayerMask) ||
        bitVecHasBits(replacement.randomDirMask);
    const touchesRandom = bitVecHasBits(replacement.randomEntityMask) ||
        bitVecHasBits(replacement.randomDirMask);
    const touchesRigid = Boolean(rule.rigid && touchesMovements);

    return {
        row_index: rowIndex,
        cell_index: cellIndex,
        touches_objects: touchesObjects,
        touches_movements: touchesMovements,
        touches_movements_layer: bitVecHasBits(replacement.movementsLayerMask),
        touches_random: touchesRandom,
        touches_random_entity: bitVecHasBits(replacement.randomEntityMask),
        touches_random_dir: bitVecHasBits(replacement.randomDirMask),
        touches_rigid: touchesRigid,
        simple_direct_mask: !touchesRandom && !touchesRigid,
        objects_clear_ids: bitVecSetIndices(replacement.objectsClear, state.objectCount),
        objects_set_ids: bitVecSetIndices(replacement.objectsSet, state.objectCount),
        movements_clear_bits: movementBitPairs(replacement.movementsClear, layerCount),
        movements_set_bits: movementBitPairs(replacement.movementsSet, layerCount),
        movements_layer_bits: movementBitPairs(replacement.movementsLayerMask, layerCount),
        random_dir_bits: movementBitPairs(replacement.randomDirMask, layerCount),
        random_entity_object_ids: bitVecSetIndices(replacement.randomEntityMask, state.objectCount),
        random_entity_choices: bitVecSetIndices(replacement.randomEntityMask, state.objectCount),
        random_dir_layers: bitVecSetLayers(replacement.randomDirMask, layerCount),
        movement_layers: bitVecSetLayers(replacement.movementsLayerMask, layerCount),
    };
}

function serializeRulePlanRow(rule, rowIndex, state) {
    const cellRow = rule.patterns[rowIndex] || [];
    const concreteObjectIds = [];
    const anyAnchorObjectIds = [];
    let concreteCellCount = 0;
    let lastEllipsisIndex = -1;

    for (let cellIndex = 0; cellIndex < cellRow.length; cellIndex++) {
        const pattern = cellRow[cellIndex];
        if (pattern === ellipsisPattern) {
            lastEllipsisIndex = cellIndex;
            continue;
        }
        concreteCellCount++;
        concreteObjectIds.push(...bitVecSetIndices(pattern.objectsPresent, state.objectCount));
        for (const anyMask of pattern.anyObjectsPresent || []) {
            anyAnchorObjectIds.push(bitVecSetIndices(anyMask, state.objectCount));
        }
    }

    return {
        row_index: rowIndex,
        ellipsis_count: rule.ellipsisCount[rowIndex] || 0,
        object_ids: bitVecSetIndices(rule.cellRowMasks[rowIndex], state.objectCount),
        movement_bits: movementBitPairs(rule.cellRowMasks_Movements[rowIndex], state.LAYER_COUNT || state.collisionLayers.length),
        concrete_anchor_object_ids: Array.from(new Set(concreteObjectIds)).sort((a, b) => a - b),
        any_anchor_object_ids: anyAnchorObjectIds,
        concrete_cell_count: concreteCellCount,
        min_concrete_suffix: lastEllipsisIndex < 0
            ? concreteCellCount
            : cellRow.slice(lastEllipsisIndex + 1).filter(pattern => pattern !== ellipsisPattern).length,
        scan_order: 'x_major',
    };
}

function serializeRulePlan(rule, groupIndex, ruleIndex, late, state) {
    const replacements = [];
    const commandNames = Array.isArray(rule.commands)
        ? rule.commands.map(command => String(Array.isArray(command) ? command[0] : command))
        : [];
    const hasEllipsis = Array.isArray(rule.ellipsisCount)
        ? rule.ellipsisCount.some(count => count > 0)
        : false;
    const simpleDeterministicRowRule = !rule.isRandom &&
        !rule.rigid &&
        !hasEllipsis &&
        rule.patterns.length === 1 &&
        commandNames.length === 0;
    for (let rowIndex = 0; rowIndex < rule.patterns.length; rowIndex++) {
        const cellRow = rule.patterns[rowIndex] || [];
        for (let cellIndex = 0; cellIndex < cellRow.length; cellIndex++) {
            const pattern = cellRow[cellIndex];
            if (pattern !== ellipsisPattern && pattern.replacement && !replacementGuaranteedNoop(pattern.replacement, pattern)) {
                replacements.push(serializeRulePlanReplacement(rule, rowIndex, cellIndex, pattern, state));
            }
        }
    }

    return {
        rule_index: ruleIndex,
        group_index: groupIndex,
        late: Boolean(late),
        direction: rule.direction,
        line_number: rule.lineNumber,
        group_number: rule.groupNumber,
        rigid: Boolean(rule.rigid),
        is_random: Boolean(rule.isRandom),
        has_replacements: Boolean(rule.hasReplacements),
        has_ellipsis: hasEllipsis,
        row_count: rule.patterns.length,
        has_commands: commandNames.length > 0,
        command_names: commandNames,
        simple_deterministic_row_rule: simpleDeterministicRowRule,
        delta_hint: directionDeltaHint(rule.direction),
        rule_object_ids: bitVecSetIndices(rule.ruleMask, state.objectCount),
        rule_movement_bits: movementBitPairs(
            rule.ruleMovementMask
                ? rule.ruleMovementMask
                : { data: orBitVecArrays(rule.cellRowMasks_Movements) },
            state.LAYER_COUNT || state.collisionLayers.length
        ),
        rows: rule.patterns.map((_, rowIndex) => serializeRulePlanRow(rule, rowIndex, state)),
        replacements,
    };
}

function serializeRulePlanGroups(ruleGroups, late, state) {
    const objectLayers = objectLayerById(state);
    const groups = [];
    for (const ruleGroup of ruleGroups) {
        const group = ruleGroup
            .filter(rule => !ruleImpossible(rule, state, objectLayers))
            .map((rule, ruleIndex) => serializeRulePlan(rule, groups.length, ruleIndex, late, state));
        if (group.length > 0) {
            groups.push(group);
        }
    }
    return groups;
}

function serializePattern(pattern) {
    if (pattern === ellipsisPattern) {
        return { kind: 'ellipsis' };
    }
    return {
        kind: 'cell_pattern',
        objects_present: bitVecToArray(pattern.objectsPresent),
        objects_missing: bitVecToArray(pattern.objectsMissing),
        any_objects_present: pattern.anyObjectsPresent.map(bitVecToArray),
        movements_present: bitVecToArray(pattern.movementsPresent),
        movements_missing: bitVecToArray(pattern.movementsMissing),
        replacement: serializeReplacement(pattern.replacement),
    };
}

function serializeRule(rule) {
    return {
        direction: rule.direction,
        has_replacements: rule.hasReplacements,
        line_number: rule.lineNumber,
        ellipsis_count: Array.from(rule.ellipsisCount),
        group_number: rule.groupNumber,
        rigid: Boolean(rule.rigid),
        commands: Array.isArray(rule.commands) ? rule.commands.slice() : [],
        is_random: Boolean(rule.isRandom),
        cell_row_masks: rule.cellRowMasks.map(bitVecToArray),
        cell_row_masks_movements: rule.cellRowMasks_Movements.map(bitVecToArray),
        rule_mask: bitVecToArray(rule.ruleMask),
        patterns: rule.patterns.map(cellRow => cellRow.map(serializePattern)),
    };
}

function serializeLevel(levelDat) {
    if (Object.prototype.hasOwnProperty.call(levelDat, 'message')) {
        return {
            kind: 'message',
            message: levelDat.message,
        };
    }
    return {
        kind: 'level',
        line_number: levelDat.lineNumber,
        width: levelDat.width,
        height: levelDat.height,
        layer_count: levelDat.layerCount,
        objects: Array.from(levelDat.objects || []),
    };
}

function metadataPairsToMap(metadataPairs) {
    const map = {};
    for (let index = 0; index < metadataPairs.length; index += 2) {
        map[metadataPairs[index]] = metadataPairs[index + 1];
    }
    return map;
}

function metadataToPairs(metadata) {
    if (Array.isArray(metadata)) {
        return Array.from(metadata);
    }
    if (!metadata || typeof metadata !== 'object') {
        return [];
    }
    const normalizeValue = value => {
        if (Array.isArray(value)) {
            return value.join('x');
        }
        if (value === null || value === undefined) {
            return '';
        }
        return String(value);
    };
    const pairs = [];
    for (const [key, value] of Object.entries(metadata)) {
        pairs.push(String(key), normalizeValue(value));
    }
    return pairs;
}

function serializeObjectEntry(name, objectEntry) {
    return {
        name,
        line_number: objectEntry.lineNumber,
        id: objectEntry.id,
        layer: objectEntry.layer,
        colors: objectEntry.colors ? objectEntry.colors.slice() : [],
        spritematrix: objectEntry.spritematrix,
    };
}

function serializeSfxEntry(entry) {
    const result = {};
    for (const [key, value] of Object.entries(entry)) {
        if (value && value.data) {
            result[key] = bitVecToArray(value);
        } else if (Array.isArray(value)) {
            result[key] = value.slice();
        } else {
            result[key] = value;
        }
    }
    return result;
}

function serializeBitVecMap(map) {
    return Object.fromEntries(
        Object.entries(map || {}).map(([key, value]) => [key, bitVecToArray(value)])
    );
}

function serializeNumericLookup(value) {
    if (Array.isArray(value)) {
        return value.slice();
    }
    return { ...(value || {}) };
}

function serializeGameState(state) {
    const objectNames = Object.keys(state.objects).sort();
    const metadataPairs = metadataToPairs(state.metadata);

    return {
        schema_version: 1,
        strides: {
            object: state.STRIDE_OBJ,
            movement: state.STRIDE_MOV,
            layers: state.LAYER_COUNT,
        },
        colors: {
            foreground: state.fgcolor,
            background: state.bgcolor,
        },
        background: {
            id: state.backgroundid,
            layer: state.backgroundlayer,
        },
        object_count: state.objectCount,
        id_dict: state.idDict.slice(),
        glyph_order: state.glyphOrder.slice(),
        glyph_dict: Object.fromEntries(
            Object.entries(state.glyphDict).map(([key, value]) => [key, Array.from(value)])
        ),
        metadata_pairs: metadataPairs,
        metadata_map: metadataPairsToMap(metadataPairs),
        metadata_lines: { ...state.metadata_lines },
        objects: objectNames.map(name => serializeObjectEntry(name, state.objects[name])),
        collision_layers: state.collisionLayers.map(layer => layer.slice()),
        layer_masks: state.layerMasks.map(bitVecToArray),
        object_masks: serializeBitVecMap(state.objectMasks),
        aggregate_masks: serializeBitVecMap(state.aggregateMasks),
        player_mask: Array.isArray(state.playerMask) ? {
            aggregate: Boolean(state.playerMask[0]),
            mask: bitVecToArray(state.playerMask[1]),
        } : {
            aggregate: false,
            mask: bitVecToArray(state.playerMask),
        },
        properties_single_layer: serializeNumericLookup(state.propertiesSingleLayer),
        rigid: Boolean(state.rigid),
        rigid_groups: state.rigidGroups.slice(),
        rigid_group_index_to_group_index: state.rigidGroupIndex_to_GroupIndex.slice(),
        group_index_to_rigid_group_index: state.groupIndex_to_RigidGroupIndex.slice(),
        group_number_to_rigid_group_index: state.groupNumber_to_RigidGroupIndex.slice(),
        rules: state.rules.map(ruleGroup => ruleGroup.map(serializeRule)),
        late_rules: state.lateRules.map(ruleGroup => ruleGroup.map(serializeRule)),
        rule_plan_v1: {
            schema_version: 1,
            rules: serializeRulePlanGroups(state.rules, false, state),
            late_rules: serializeRulePlanGroups(state.lateRules, true, state),
        },
        loop_point: serializeNumericLookup(state.loopPoint),
        late_loop_point: serializeNumericLookup(state.lateLoopPoint),
        winconditions: state.winconditions.map(condition => ({
            quantifier: condition[0],
            filter1: bitVecToArray(condition[1]),
            filter2: bitVecToArray(condition[2]),
            line_number: condition[3],
            aggr1: Boolean(condition[4]),
            aggr2: Boolean(condition[5]),
        })),
        levels: state.levels.map(serializeLevel),
        sfx_events: { ...state.sfx_Events },
        sfx_creation_masks: state.sfx_CreationMasks.map(serializeSfxEntry),
        sfx_destruction_masks: state.sfx_DestructionMasks.map(serializeSfxEntry),
        sfx_movement_masks: state.sfx_MovementMasks.map(layer => layer.map(serializeSfxEntry)),
        sfx_movement_failure_masks: state.sfx_MovementFailureMasks.map(serializeSfxEntry),
        sounds: Array.isArray(state.sounds) ? state.sounds.slice() : [],
    };
}

function serializePreparedSession() {
    const currentLevel = level;
    const randomState = RandomGen && RandomGen._state ? RandomGen._state : null;
    return {
        current_level_index: typeof curlevel === 'number' ? curlevel : 0,
        current_level_target: typeof curlevelTarget === 'number' ? curlevelTarget : null,
        title_screen: Boolean(titleScreen),
        text_mode: Boolean(textMode),
        title_mode: typeof titleMode === 'number' ? titleMode : 0,
        title_selection: typeof titleSelection === 'number' ? titleSelection : 0,
        title_selected: Boolean(titleSelected),
        message_selected: Boolean(messageselected),
        winning: Boolean(winning),
        loaded_level_seed: typeof loadedLevelSeed === 'string' ? loadedLevelSeed : null,
        random_state: randomState ? {
            valid: true,
            i: typeof randomState.i === 'number' ? randomState.i : 0,
            j: typeof randomState.j === 'number' ? randomState.j : 0,
            s: Array.isArray(randomState.s) ? Array.from(randomState.s) : [],
        } : null,
        old_flickscreen_dat: Array.isArray(oldflickscreendat) ? oldflickscreendat.slice() : [],
        level: {
            line_number: currentLevel.lineNumber,
            width: currentLevel.width,
            height: currentLevel.height,
            layer_count: currentLevel.layerCount,
            objects: Array.from(currentLevel.objects || []),
        },
        restart_target: restartTarget ? {
            width: restartTarget.width,
            height: restartTarget.height,
            old_flickscreen_dat: Array.isArray(restartTarget.oldflickscreendat) ? restartTarget.oldflickscreendat.slice() : [],
            objects: Array.from(restartTarget.dat || []),
        } : null,
        serialized_level: typeof convertLevelToString === 'function' ? convertLevelToString() : '',
    };
}

function buildCompiledIr(document) {
    return {
        schema_version: 1,
        document,
        game: serializeGameState(state),
        prepared_session: serializePreparedSession(),
    };
}

module.exports = {
    buildCompiledIr,
};
