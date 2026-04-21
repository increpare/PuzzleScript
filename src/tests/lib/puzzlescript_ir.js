'use strict';

function bitVecToArray(vec) {
    if (!vec || !vec.data) {
        return [];
    }
    return Array.from(vec.data);
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
    const metadataPairs = Array.from(state.metadata || []);

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
