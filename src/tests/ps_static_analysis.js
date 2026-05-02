#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { compileSemanticSource } = require('../canonicalize');

const SCHEMA = 'ps-static-analysis-v1';

function uniqueSorted(values) {
    return Array.from(new Set(values)).sort((left, right) =>
        left.localeCompare(right, undefined, { numeric: true })
    );
}

function displayName(state, name) {
    return (state.original_case_names && state.original_case_names[name]) || name;
}

function objectInternalNamesFromMask(state, mask) {
    const bitMask = Array.isArray(mask) ? mask[1] : mask;
    const names = [];
    if (!bitMask || typeof bitMask.get !== 'function') {
        return names;
    }
    for (const [name, object] of Object.entries(state.objects)) {
        if (bitMask.get(object.id)) {
            names.push(name);
        }
    }
    return uniqueSorted(names);
}

function objectNamesFromMask(state, mask) {
    return objectInternalNamesFromMask(state, mask).map(name => displayName(state, name));
}

function buildObjects(state) {
    return Object.keys(state.objects)
        .map(name => ({
            id: state.objects[name].id,
            name: displayName(state, name),
            canonical_name: name,
            layer: state.objects[name].layer,
            tags: {},
        }))
        .sort((left, right) => left.id - right.id);
}

function buildProperties(state) {
    const properties = [];
    for (const [name, members] of Object.entries(state.propertiesDict || {})) {
        properties.push({
            name: displayName(state, name),
            canonical_name: name,
            kind: 'property',
            members: uniqueSorted(Array.from(members, member => displayName(state, member))),
            tags: {},
        });
    }
    for (const [name, target] of Object.entries(state.synonymsDict || {})) {
        properties.push({
            name: displayName(state, name),
            canonical_name: name,
            kind: 'synonym',
            members: [displayName(state, target)],
            tags: {},
        });
    }
    return properties.sort((left, right) =>
        left.name.localeCompare(right.name, undefined, { numeric: true })
    );
}

function buildCollisionLayers(state) {
    return Array.from(state.collisionLayers, (objects, id) => ({
        id,
        objects: Array.from(objects, name => displayName(state, name)),
        canonical_objects: Array.from(objects),
        tags: {},
    }));
}

function buildWinconditions(state) {
    return Array.from(state.winconditions, (condition, index) => ({
        id: `win_${index}`,
        quantifier: condition[0],
        subjects: objectNamesFromMask(state, condition[1]),
        targets: condition[2] ? objectNamesFromMask(state, condition[2]) : [],
        tags: {},
    }));
}

function buildLevels(state) {
    return Array.from(state.levels, (level, index) => {
        if (level.message !== undefined) {
            return { index, kind: 'message', objects_present: [], layers_present: [], tags: {} };
        }
        const objects = new Set();
        const layers = new Set();
        for (let cellIndex = 0; cellIndex < level.n_tiles; cellIndex++) {
            const cell = level.getCell(cellIndex);
            for (const name of objectInternalNamesFromMask(state, cell)) {
                objects.add(displayName(state, name));
                layers.add(state.objects[name].layer);
            }
        }
        return {
            index,
            kind: 'level',
            width: level.width,
            height: level.height,
            objects_present: uniqueSorted(Array.from(objects)),
            layers_present: Array.from(layers).sort((left, right) => left - right),
            tags: {},
        };
    });
}

function tagObjectLevelPresence(psTagged) {
    const playableLevels = psTagged.levels.filter(level => level.kind === 'level');
    for (const object of psTagged.objects) {
        const presentCount = playableLevels.filter(level => level.objects_present.includes(object.name)).length;
        object.tags.present_in_all_levels = playableLevels.length > 0 && presentCount === playableLevels.length;
        object.tags.present_in_some_levels = presentCount > 0 && presentCount < playableLevels.length;
        object.tags.present_in_no_levels = presentCount === 0;
    }
}

function buildPsTagged(state, options = {}) {
    const psTagged = {
        game: {
            title: state.metadata && state.metadata.title,
            source_path: options.sourcePath || '<memory>',
            tags: {},
        },
        objects: buildObjects(state),
        properties: buildProperties(state),
        collision_layers: buildCollisionLayers(state),
        winconditions: buildWinconditions(state),
        levels: buildLevels(state),
        rule_sections: [],
    };
    tagObjectLevelPresence(psTagged);
    return psTagged;
}

function emptyFacts() {
    return {
        mergeability: [],
        movement_action: [],
        count_layer_invariants: [],
        transient_boundary: [],
    };
}

function analyzeSource(source, options = {}) {
    const sourcePath = options.sourcePath || '<memory>';
    const compiled = compileSemanticSource(source, {
        includeWinConditions: true,
        throwOnError: false,
    });

    if (compiled.errorCount > 0 || compiled.state === null || compiled.state.invalid) {
        return {
            schema: SCHEMA,
            source: { path: sourcePath },
            status: 'compile_error',
            errors: compiled.errorStrings.slice(),
            ps_tagged: null,
            facts: emptyFacts(),
            summary: { proved: 0, candidate: 0, rejected: 0 },
        };
    }

    const psTagged = buildPsTagged(compiled.state, { sourcePath });
    const report = {
        schema: SCHEMA,
        source: { path: sourcePath },
        status: 'ok',
        ps_tagged: psTagged,
        facts: emptyFacts(),
        summary: { proved: 0, candidate: 0, rejected: 0 },
    };

    if (options.includePsTagged === false) {
        delete report.ps_tagged;
    }
    return report;
}

function analyzeFile(filePath, options = {}) {
    const resolved = path.resolve(filePath);
    const source = fs.readFileSync(resolved, 'utf8');
    return analyzeSource(source, Object.assign({}, options, { sourcePath: filePath }));
}

module.exports = {
    SCHEMA,
    analyzeFile,
    analyzeSource,
};
