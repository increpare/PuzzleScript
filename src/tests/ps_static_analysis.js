#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { compileSemanticSource } = require('../canonicalize');

const SCHEMA = 'ps-static-analysis-v1';
const INERT_COMMANDS = new Set(['message', 'sfx0', 'sfx1', 'sfx2', 'sfx3', 'sfx4', 'sfx5', 'sfx6', 'sfx7', 'sfx8', 'sfx9', 'sfx10']);
const SEMANTIC_COMMANDS = new Set(['cancel', 'again', 'restart', 'win', 'checkpoint']);
const DIRECTIONAL_MOVEMENTS = new Set(['up', 'down', 'left', 'right', 'moving', 'randomdir']);
const CARDINAL_MOVEMENTS = ['up', 'down', 'left', 'right'];

function uniqueSorted(values) {
    return Array.from(new Set(values)).sort((left, right) =>
        left.localeCompare(right, undefined, { numeric: true })
    );
}

function uniqueInOrder(values) {
    return Array.from(new Set(values));
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
    return Array.from(state.collisionLayers, (objects, id) => {
        const canonicalObjects = uniqueInOrder(Array.from(objects));
        return {
            id,
            objects: canonicalObjects.map(name => displayName(state, name)),
            canonical_objects: canonicalObjects,
            tags: {},
        };
    });
}

function buildWinconditions(state) {
    const objectCount = Object.keys(state.objects).length;
    return Array.from(state.winconditions, (condition, index) => {
        const targetNames = condition[2] ? objectInternalNamesFromMask(state, condition[2]) : [];
        const plainCondition = targetNames.length === objectCount;
        return {
            id: `win_${index}`,
            quantifier: condition[0],
            subjects: objectNamesFromMask(state, condition[1]),
            targets: plainCondition ? [] : targetNames.map(name => displayName(state, name)),
            tags: { plain: plainCondition },
        };
    });
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

function directionName(direction) {
    if (typeof direction === 'string') return direction;
    if (direction === 1) return 'up';
    if (direction === 2) return 'down';
    if (direction === 4) return 'left';
    if (direction === 8) return 'right';
    if (direction === 15) return 'orthogonal';
    if (direction === 16) return 'action';
    return String(direction);
}

function refForName(state, name) {
    if (state.objects[name]) {
        return { type: 'object', name: displayName(state, name), canonical_name: name };
    }
    if (state.propertiesDict && state.propertiesDict[name]) {
        return {
            type: 'property',
            name: displayName(state, name),
            canonical_name: name,
            members: uniqueSorted(Array.from(state.propertiesDict[name], member => displayName(state, member))),
        };
    }
    if (state.synonymsDict && state.synonymsDict[name]) {
        return {
            type: 'synonym',
            name: displayName(state, name),
            canonical_name: name,
            members: [displayName(state, state.synonymsDict[name])],
        };
    }
    return { type: 'unknown', name };
}

function termFromPair(state, direction, name) {
    if (direction === '...' && name === '...') {
        return { kind: 'present', ref: { type: 'ellipsis' }, movement: null };
    }
    if (direction === 'no') {
        return { kind: 'absent', ref: refForName(state, name), movement: null };
    }
    if (direction === 'random') {
        return { kind: 'random_object', ref: refForName(state, name), movement: null };
    }
    if (direction === '') {
        return { kind: 'present', ref: refForName(state, name), movement: null };
    }
    return { kind: 'present', ref: refForName(state, name), movement: directionName(direction) };
}

function termsFromCell(state, cell) {
    const terms = [];
    for (let index = 0; index < cell.length; index += 2) {
        terms.push(termFromPair(state, cell[index], cell[index + 1]));
    }
    return terms;
}

function termsFromSide(state, side) {
    return Array.from(side || [], row =>
        Array.from(row, cell => termsFromCell(state, cell))
    );
}

function flattenTerms(side) {
    return side.flat(2);
}

function termHasInputMovementRequirement(term) {
    return term.movement !== null && term.movement !== 'stationary';
}

function summarizeRule(rule) {
    const lhsTerms = flattenTerms(rule.lhs);
    const rhsTerms = flattenTerms(rule.rhs);
    return {
        lhs_terms: lhsTerms,
        rhs_terms: rhsTerms,
        lhs_present: lhsTerms.filter(term => term.kind === 'present'),
        lhs_absent: lhsTerms.filter(term => term.kind === 'absent'),
        lhs_movement: lhsTerms.filter(term => term.movement !== null),
        rhs_present: rhsTerms.filter(term => term.kind === 'present'),
        rhs_absent: rhsTerms.filter(term => term.kind === 'absent'),
        rhs_random_objects: rhsTerms.filter(term => term.kind === 'random_object'),
        rhs_movement: rhsTerms.filter(term => term.movement !== null),
        semantic_commands: rule.commands.map(command => command[0]).filter(command => SEMANTIC_COMMANDS.has(command)),
        inert_commands: rule.commands.map(command => command[0]).filter(command => INERT_COMMANDS.has(command)),
    };
}

function objectTermSignature(terms) {
    return JSON.stringify(terms
        .filter(term => term.kind === 'present')
        .map(term => JSON.stringify(term.ref))
        .sort());
}

function movementTermSignature(terms) {
    return JSON.stringify(terms
        .filter(term => term.kind === 'present' && term.movement !== null)
        .map(term => `${JSON.stringify(term.ref)}:${term.movement}`)
        .sort());
}

function tagRule(rule) {
    rule.summary = summarizeRule(rule);
    const commandNames = rule.commands.map(command => command[0]);
    const hasOnlyInertCommands = commandNames.length > 0
        && commandNames.every(command => INERT_COMMANDS.has(command));
    const hasReplacement = rule.rhs.length > 0;
    const objectMutating = hasReplacement
        && (objectTermSignature(rule.summary.lhs_terms) !== objectTermSignature(rule.summary.rhs_terms)
        || rule.summary.rhs_absent.length > 0
        || rule.summary.rhs_random_objects.length > 0);
    const writesMovement = hasReplacement
        && movementTermSignature(rule.summary.lhs_terms) !== movementTermSignature(rule.summary.rhs_terms);
    const hasSemanticCommand = rule.summary.semantic_commands.length > 0;

    rule.tags.command_only = commandNames.length > 0 && !objectMutating && !writesMovement;
    rule.tags.inert_command_only = hasOnlyInertCommands && rule.tags.command_only;
    rule.tags.object_mutating = objectMutating;
    rule.tags.writes_movement = writesMovement;
    rule.tags.movement_only = writesMovement && !objectMutating && !hasSemanticCommand;
    rule.tags.reads_action = rule.summary.lhs_movement.some(term => term.movement === 'action');
    rule.tags.has_again = rule.summary.semantic_commands.includes('again');
    const hasNonInertEffect = objectMutating || writesMovement || hasSemanticCommand;
    rule.tags.solver_state_active = !rule.tags.inert_command_only && hasNonInertEffect;
    if (rule.rigid && hasNonInertEffect) {
        rule.tags.rigid_active = true;
    }
}

function buildRuleSections(state) {
    const rules = Array.from(state.rules || []);
    return [
        buildRuleSection(state, 'early', rules.filter(rule => !rule.late)),
        buildRuleSection(state, 'late', rules.filter(rule => rule.late)),
    ];
}

function buildRuleSection(state, name, rules) {
    const groupNumbers = [];
    const groupMap = new Map();
    for (const rule of rules) {
        if (!groupMap.has(rule.groupNumber)) {
            groupMap.set(rule.groupNumber, []);
            groupNumbers.push(rule.groupNumber);
        }
        groupMap.get(rule.groupNumber).push(rule);
    }
    const groups = groupNumbers.map((groupNumber, index) =>
        buildRuleGroup(state, name, index, groupNumber, groupMap.get(groupNumber))
    );
    return {
        name,
        loops: buildLoopSummaries(state, groups),
        groups,
    };
}

function buildLoopSummaries(state, groups) {
    const loops = [];
    const stack = [];
    for (const loop of Array.from(state.loops || [])) {
        const line = loop[0];
        const bracket = loop[1];
        if (bracket === 1) {
            stack.push({ start_line: line, end_line: null });
        } else if (bracket === -1 && stack.length > 0) {
            const active = stack.pop();
            active.end_line = line;
            active.group_ids = groups
                .filter(group => group.source_line_min > active.start_line && group.source_line_max < active.end_line)
                .map(group => group.id);
            if (active.group_ids.length > 0) {
                active.id = `loop_${loops.length}`;
                loops.push(active);
            }
        }
    }
    return loops;
}

function buildRuleGroup(state, sectionName, groupIndex, groupNumber, sourceRules) {
    const rules = sourceRules.map((rule, ruleIndex) => buildRuleIr(state, sectionName, groupIndex, rule, ruleIndex));
    const group = {
        id: `${sectionName}_group_${groupIndex}`,
        group_index: groupIndex,
        group_number: groupNumber,
        source_line_min: Math.min(...sourceRules.map(rule => rule.lineNumber)),
        source_line_max: Math.max(...sourceRules.map(rule => rule.lineNumber)),
        random: sourceRules.some(rule => rule.randomRule),
        tags: {},
        rules,
    };
    for (const rule of rules) {
        tagRule(rule);
    }
    tagGroup(group);
    return group;
}

function buildRuleIr(state, sectionName, groupIndex, rule, ruleIndex) {
    return {
        id: `${sectionName}_group_${groupIndex}_rule_${ruleIndex}`,
        source_line: rule.lineNumber,
        direction: directionName(rule.direction),
        late: !!rule.late || sectionName === 'late',
        rigid: !!rule.rigid,
        random_rule: !!rule.randomRule,
        tags: {},
        commands: Array.from(rule.commands || [], command => Array.from(command)),
        lhs: termsFromSide(state, rule.lhs),
        rhs: termsFromSide(state, rule.rhs),
        summary: {},
    };
}

function tagGroup(group) {
    group.tags.has_again = group.rules.some(rule => rule.tags.has_again);
    group.tags.object_mutating = group.rules.some(rule => rule.tags.object_mutating);
    group.tags.movement_only = group.rules.some(rule => rule.tags.movement_only) && !group.tags.object_mutating;
    group.tags.command_only = group.rules.every(rule => rule.tags.command_only);
    group.tags.solver_state_active = group.rules.some(rule => rule.tags.solver_state_active);
}

function allRuleEntries(psTagged) {
    return psTagged.rule_sections.flatMap(section =>
        section.groups.flatMap(group =>
            group.rules.map(rule => ({ section, group, rule }))
        )
    );
}

function tagGame(psTagged) {
    const rules = allRuleEntries(psTagged).map(entry => entry.rule);
    psTagged.game.tags.has_again = rules.some(rule => rule.tags.has_again);
    psTagged.game.tags.has_random = rules.some(rule => rule.random_rule || rule.summary.rhs_random_objects.length > 0);
    psTagged.game.tags.has_rigid = rules.some(rule => rule.rigid);
    psTagged.game.tags.has_action_rules = rules.some(rule => rule.tags.reads_action);
    psTagged.game.tags.has_autonomous_tick_rules = rules.some(rule =>
        rule.tags.solver_state_active && !rule.summary.lhs_movement.some(termHasInputMovementRequirement)
    );
}

function membersForRef(psTagged, ref) {
    if (ref.type === 'object') return [ref.name];
    if (ref.type === 'object_set') return ref.objects.slice();
    if (ref.type === 'property' || ref.type === 'synonym' || ref.type === 'unknown') {
        const property = psTagged.properties.find(item => item.name === ref.name || item.canonical_name === ref.canonical_name);
        return property ? property.members.slice() : [ref.name];
    }
    return [];
}

function refForObjectName(psTagged, objectName) {
    const object = psTagged.objects.find(item => item.name === objectName);
    if (!object) return { type: 'object', name: objectName };
    return { type: 'object', name: object.name, canonical_name: object.canonical_name };
}

function normalizeTermRefs(psTagged) {
    for (const { rule } of allRuleEntries(psTagged)) {
        for (const term of rule.summary.lhs_terms.concat(rule.summary.rhs_terms)) {
            const members = membersForRef(psTagged, term.ref);
            if (members.length === 1 && psTagged.objects.some(object => object.name === members[0])) {
                term.expanded_objects = members;
                term.ref = refForObjectName(psTagged, members[0]);
            } else if (members.length > 1) {
                term.expanded_objects = uniqueSorted(members);
                term.ref = { type: 'object_set', objects: uniqueSorted(members), source: term.ref.name || term.ref.type };
            } else {
                term.expanded_objects = [];
            }
        }
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
        rule_sections: buildRuleSections(state),
    };
    tagObjectLevelPresence(psTagged);
    tagGame(psTagged);
    normalizeTermRefs(psTagged);
    return psTagged;
}

function emptyFacts() {
    return {
        mergeability: [],
        movement_action: [],
        count_layer_invariants: [],
        transient_boundary: [],
        rulegroup_flow: [],
    };
}

function fact(family, id, status, fields) {
    return Object.assign({
        family,
        id,
        status,
        subjects: {},
        tags: {},
        proof: [],
        blockers: [],
        evidence: [],
    }, fields);
}

function sameArray(left, right) {
    return JSON.stringify(left) === JSON.stringify(right);
}

function winRoleForObject(psTagged, objectName) {
    return psTagged.winconditions.map(win => ({
        id: win.id,
        in_subjects: win.subjects.includes(objectName),
        in_targets: win.targets.includes(objectName),
    }));
}

function directObservationsForObjects(psTagged, objects) {
    const observations = [];
    for (const { rule } of allRuleEntries(psTagged)) {
        if (rule.tags.inert_command_only) continue;
        for (const row of rule.lhs) {
            for (const cell of row) {
                const directTerms = cell.filter(term =>
                    term.ref.type === 'object' && objects.includes(term.ref.name)
                );
                if (directTerms.length === 0) continue;
                const observedObjects = uniqueSorted(directTerms.map(term => term.ref.name));
                const sameObservation = new Set(directTerms.map(term => `${term.kind}:${term.movement}`)).size === 1;
                if (sameArray(observedObjects, objects) && sameObservation) continue;
                observations.push(...directTerms.map(term => ({
                    rule_id: rule.id,
                    source_line: rule.source_line,
                    kind: term.kind,
                    movement: term.movement,
                    object: term.ref.name,
                })));
            }
        }
    }
    return observations;
}

function groupObservationIsShared(psTagged, objects) {
    for (const { rule } of allRuleEntries(psTagged)) {
        if (rule.tags.inert_command_only) continue;
        for (const term of rule.summary.lhs_terms) {
            if (term.ref.type !== 'object_set') continue;
            const overlap = objects.filter(objectName => term.expanded_objects.includes(objectName));
            if (overlap.length > 0 && overlap.length !== objects.length) {
                return false;
            }
        }
    }
    return true;
}

function deriveMergeabilityFacts(psTagged) {
    const results = [];
    for (const layer of psTagged.collision_layers) {
        if (layer.objects.length < 2) continue;
        for (let left = 0; left < layer.objects.length; left++) {
            for (let right = left + 1; right < layer.objects.length; right++) {
                const objects = [layer.objects[left], layer.objects[right]].sort();
                const blockers = [];
                const directObservations = directObservationsForObjects(psTagged, objects);
                if (directObservations.length > 0) blockers.push('individual_lhs_read');
                if (!sameArray(winRoleForObject(psTagged, objects[0]), winRoleForObject(psTagged, objects[1]))) {
                    blockers.push('different_win_roles');
                }
                if (!groupObservationIsShared(psTagged, objects)) {
                    blockers.push('partial_property_observation');
                }
                results.push(fact('mergeability', `merge_${objects.join('_')}`, blockers.length === 0 ? 'candidate' : 'rejected', {
                    subjects: { objects },
                    proof: blockers.length === 0 ? ['same_collision_layer', 'observed_only_through_shared_sets', 'same_win_roles'] : ['same_collision_layer'],
                    blockers,
                    evidence: directObservations.map(item => item.rule_id).concat(`layer_${layer.id}`),
                }));
            }
        }
    }
    return results;
}

function layerForObject(psTagged, objectName) {
    const object = psTagged.objects.find(item => item.name === objectName);
    return object ? object.layer : null;
}

function playerLayers(psTagged) {
    const playerObjects = playerObjectNameSet(psTagged);
    return uniqueSorted(Array.from(playerObjects)
        .map(objectName => layerForObject(psTagged, objectName))
        .filter(layer => layer !== null)
        .map(layer => String(layer)));
}

function movementRequirementsFromTerms(psTagged, terms) {
    const requirements = [];
    for (const term of terms) {
        if (term.kind !== 'present' || !termHasInputMovementRequirement(term)) continue;
        for (const objectName of term.expanded_objects || []) {
            const layer = layerForObject(psTagged, objectName);
            if (layer !== null) requirements.push({ layer: String(layer), movement: term.movement });
        }
    }
    return requirements;
}

function movementEffectsFromTerms(psTagged, terms) {
    const pairs = [];
    for (const term of terms) {
        if (term.kind !== 'present' || term.movement === null || term.movement === 'stationary') continue;
        for (const objectName of term.expanded_objects || []) {
            const layer = layerForObject(psTagged, objectName);
            if (layer === null) continue;
            const layerName = String(layer);
            if (term.movement === 'randomdir') {
                for (const movement of CARDINAL_MOVEMENTS) pairs.push(`${layerName}:${movement}`);
                pairs.push(`${layerName}:moving`);
            } else if (CARDINAL_MOVEMENTS.includes(term.movement)) {
                pairs.push(`${layerName}:${term.movement}`);
                pairs.push(`${layerName}:moving`);
            } else {
                pairs.push(`${layerName}:${term.movement}`);
            }
        }
    }
    return pairs;
}

function movementRequirementReachable(requirement, possibleMovements) {
    if (requirement.movement === 'moving' || requirement.movement === 'randomdir' || requirement.movement === 'orthogonal') {
        return CARDINAL_MOVEMENTS.some(movement => possibleMovements.has(`${requirement.layer}:${movement}`))
            || possibleMovements.has(`${requirement.layer}:moving`)
            || possibleMovements.has(`${requirement.layer}:randomdir`);
    }
    return possibleMovements.has(`${requirement.layer}:${requirement.movement}`);
}

function ruleMovementRequirementsReachable(psTagged, rule, possibleMovements) {
    const requirements = movementRequirementsFromTerms(psTagged, rule.summary.lhs_terms);
    if (requirements.length === 0) return true;
    return requirements.every(requirement => movementRequirementReachable(requirement, possibleMovements));
}

function deriveMovementActionFacts(psTagged) {
    const activeRules = allRuleEntries(psTagged).map(entry => entry.rule).filter(rule => rule.tags.solver_state_active);
    const possibleMovements = new Set(playerLayers(psTagged).map(layer => `${layer}:action`));
    const blockers = [];
    let changed = true;
    while (changed) {
        changed = false;
        for (const rule of activeRules) {
            if (!ruleMovementRequirementsReachable(psTagged, rule, possibleMovements)) continue;
            if (rule.tags.reads_action) blockers.push('reads_action');
            if (rule.tags.has_again) blockers.push('queues_again');
            if (rule.rigid) blockers.push('rigid_rule');
            if (!rule.summary.lhs_movement.some(termHasInputMovementRequirement)) blockers.push('autonomous_solver_active_rule');
            if (rule.tags.object_mutating) blockers.push('action_may_mutate_objects');
            for (const pair of movementEffectsFromTerms(psTagged, rule.summary.rhs_terms)) {
                const movement = pair.split(':')[1];
                if (DIRECTIONAL_MOVEMENTS.has(movement)) blockers.push('action_may_create_directional_movement');
                if (!possibleMovements.has(pair)) {
                    possibleMovements.add(pair);
                    changed = true;
                }
            }
        }
    }
    const uniqueBlockers = uniqueSorted(blockers);
    return [
        fact('movement_action', 'movement_pairs', 'proved', {
            value: Array.from(possibleMovements).sort(),
            proof: ['conservative_movement_reachability_fixpoint'],
        }),
        fact('movement_action', 'action_noop', uniqueBlockers.length === 0 ? 'proved' : 'rejected', {
            value: uniqueBlockers.length === 0,
            blockers: uniqueBlockers,
            proof: uniqueBlockers.length === 0 ? ['no_reachable_action_effects'] : [],
            evidence: activeRules.map(rule => rule.id),
        }),
    ];
}

function termMentionsObject(term, objectName) {
    return (term.expanded_objects || []).includes(objectName);
}

function ruleMentionsObject(rule, objectName) {
    return rule.summary.lhs_terms.concat(rule.summary.rhs_terms).some(term => termMentionsObject(term, objectName));
}

function ruleWritesCollisionLayerObject(psTagged, rule, objectName) {
    const layer = layerForObject(psTagged, objectName);
    if (layer === null) return false;
    return rule.summary.rhs_terms.some(term =>
        (term.kind === 'present' || term.kind === 'random_object')
        && (term.expanded_objects || []).some(termObject => layerForObject(psTagged, termObject) === layer)
    );
}

function ruleMayAffectObject(psTagged, rule, objectName) {
    if (!rule.tags.solver_state_active || !rule.tags.object_mutating) return false;
    return ruleMentionsObject(rule, objectName) || ruleWritesCollisionLayerObject(psTagged, rule, objectName);
}

function playerObjectNameSet(psTagged) {
    const playerProperty = psTagged.properties.find(item =>
        item.canonical_name === 'player' || item.name.toLowerCase() === 'player'
    );
    if (playerProperty) return new Set(playerProperty.members);
    const playerObject = psTagged.objects.find(item =>
        item.canonical_name === 'player' || item.name.toLowerCase() === 'player'
    );
    return new Set(playerObject ? [playerObject.name] : []);
}

function ruleMovementMentionsObject(rule, objectName) {
    return rule.summary.lhs_terms.concat(rule.summary.rhs_terms).some(term =>
        term.movement !== null && termMentionsObject(term, objectName)
    );
}

function positiveObjectNamesFromTerms(terms) {
    return new Set(terms
        .filter(term => term.kind === 'present')
        .flatMap(term => term.expanded_objects || []));
}

function ruleMayCreateCollisionLayerObject(psTagged, rule, layer) {
    if (!rule.tags.solver_state_active || !rule.tags.object_mutating) return false;
    const lhsObjects = positiveObjectNamesFromTerms(rule.summary.lhs_terms);
    return rule.summary.rhs_terms.some(term => {
        if (term.kind !== 'present' && term.kind !== 'random_object') return false;
        return (term.expanded_objects || []).some(objectName =>
            layerForObject(psTagged, objectName) === layer
            && (term.kind === 'random_object' || !lhsObjects.has(objectName))
        );
    });
}

function deriveCountLayerInvariantFacts(psTagged) {
    const activeRules = allRuleEntries(psTagged).map(entry => entry.rule).filter(rule => rule.tags.solver_state_active);
    const playerObjects = playerObjectNameSet(psTagged);
    const results = [];
    for (const object of psTagged.objects) {
        const writers = activeRules.filter(rule => ruleMayAffectObject(psTagged, rule, object.name));
        const movementRules = activeRules.filter(rule => ruleMovementMentionsObject(rule, object.name));
        const layerCreators = activeRules.filter(rule => ruleMayCreateCollisionLayerObject(psTagged, rule, object.layer));
        const staticBlockers = [];
        if (writers.length > 0) staticBlockers.push('object_written_by_solver_active_rule');
        if (movementRules.length > 0 || playerObjects.has(object.name)) staticBlockers.push('object_may_receive_movement');
        if (layerCreators.length > 0) staticBlockers.push('collision_layer_object_may_be_created');
        object.tags.may_be_created = writers.length > 0;
        object.tags.may_be_destroyed = writers.length > 0;
        object.tags.count_invariant = writers.length === 0;
        object.tags.static = staticBlockers.length === 0;
        results.push(fact('count_layer_invariants', `object_${object.name}_count_preserved`, writers.length === 0 ? 'proved' : 'rejected', {
            subjects: { objects: [object.name] },
            proof: writers.length === 0 ? ['no_solver_active_rule_writes_object'] : [],
            blockers: writers.length === 0 ? [] : ['object_written_by_solver_active_rule'],
            evidence: writers.map(rule => rule.id),
        }));
        results.push(fact('count_layer_invariants', `object_${object.name}_static`, staticBlockers.length === 0 ? 'proved' : 'rejected', {
            subjects: { objects: [object.name] },
            proof: staticBlockers.length === 0
                ? ['no_solver_active_rule_writes_object', 'no_movement_applied_to_object', 'no_collision_layer_object_creation']
                : [],
            blockers: uniqueSorted(staticBlockers),
            evidence: uniqueSorted(writers.concat(movementRules, layerCreators).map(rule => rule.id)),
        }));
    }
    for (const layer of psTagged.collision_layers) {
        const layerWriterIds = uniqueSorted(layer.objects.flatMap(objectName =>
            activeRules.filter(rule => ruleMayAffectObject(psTagged, rule, objectName)).map(rule => rule.id)
        ));
        const nonStaticObjects = uniqueSorted(layer.objects.filter(objectName => {
            const object = psTagged.objects.find(item => item.name === objectName);
            return !object || !object.tags.static;
        }));
        const layerBlockers = [];
        if (layerWriterIds.length > 0) layerBlockers.push('layer_objects_may_change');
        if (nonStaticObjects.length > 0) layerBlockers.push('layer_contains_nonstatic_object');
        layer.tags.static = layerBlockers.length === 0;
        results.push(fact('count_layer_invariants', `layer_${layer.id}_static`, layerBlockers.length === 0 ? 'proved' : 'candidate', {
            subjects: { layers: [layer.id] },
            proof: layerBlockers.length === 0 ? ['all_layer_objects_static'] : [],
            blockers: layerBlockers,
            evidence: uniqueSorted(layerWriterIds.concat(nonStaticObjects)),
        }));
    }
    return results;
}

function rulesInSection(psTagged, sectionName) {
    const section = psTagged.rule_sections.find(item => item.name === sectionName);
    return section ? section.groups.flatMap(group => group.rules.map(rule => ({ section, group, rule }))) : [];
}

function ruleHasPositiveLhsObject(rule, objectName) {
    return rule.summary.lhs_terms.some(term => term.kind === 'present' && termMentionsObject(term, objectName));
}

function rulePresentsObjectOnRhs(rule, objectName) {
    return rule.summary.rhs_terms.some(term =>
        (term.kind === 'present' || term.kind === 'random_object') && termMentionsObject(term, objectName)
    );
}

function ruleCreatesObject(rule, objectName) {
    return rule.tags.solver_state_active
        && rule.tags.object_mutating
        && rulePresentsObjectOnRhs(rule, objectName)
        && !ruleHasPositiveLhsObject(rule, objectName);
}

function creatorsForObject(psTagged, objectName) {
    return allRuleEntries(psTagged).filter(entry => ruleCreatesObject(entry.rule, objectName));
}

function lateClearersForObject(psTagged, objectName) {
    return rulesInSection(psTagged, 'late')
        .filter(entry => entry.rule.tags.solver_state_active)
        .filter(entry => ruleUnconditionallyClearsObject(entry.rule, objectName));
}

function ruleUnconditionallyClearsObject(rule, objectName) {
    const lhsTerms = rule.summary.lhs_terms;
    if (lhsTerms.length !== 1) return false;
    const lhsTerm = lhsTerms[0];
    if (lhsTerm.kind !== 'present' || lhsTerm.movement !== null || !termMentionsObject(lhsTerm, objectName)) {
        return false;
    }
    return !rule.summary.rhs_terms.some(term => term.kind === 'present' && termMentionsObject(term, objectName));
}

function ruleEntryBefore(left, right) {
    if (left.section.name === 'early' && right.section.name === 'late') return true;
    if (left.section.name === right.section.name) return left.group.group_index < right.group.group_index;
    return false;
}

function allCreatorsHaveLaterClearer(creators, clearers) {
    return creators.every(creator => clearers.some(clearer => ruleEntryBefore(creator, clearer)));
}

function objectInWincondition(psTagged, objectName) {
    return psTagged.winconditions.some(win => win.subjects.includes(objectName) || win.targets.includes(objectName));
}

function deriveTransientBoundaryFacts(psTagged) {
    const results = [];
    for (const object of psTagged.objects) {
        const creators = creatorsForObject(psTagged, object.name);
        const clearers = lateClearersForObject(psTagged, object.name);
        const blockers = [];
        if (creators.length === 0) blockers.push('not_created_before_end_cleanup');
        if (clearers.length === 0) blockers.push('no_late_cleanup_clear');
        if (creators.length > 0 && clearers.length > 0 && !allCreatorsHaveLaterClearer(creators, clearers)) {
            blockers.push('creator_not_followed_by_late_cleanup');
        }
        if (!object.tags.present_in_no_levels) blockers.push('present_in_some_initial_levels');
        if (objectInWincondition(psTagged, object.name)) blockers.push('appears_in_wincondition');
        if (creators.some(entry => entry.group.tags.has_again || entry.rule.tags.has_again)) blockers.push('has_again_taint');
        if (creators.some(entry => entry.rule.rigid) || clearers.some(entry => entry.rule.rigid)) blockers.push('rigid_rule');
        const status = blockers.length === 0 ? 'proved' : 'rejected';
        results.push(fact('transient_boundary', `object_${object.name}_end_turn_transient`, status, {
            subjects: { objects: [object.name] },
            tags: { single_turn_only: true },
            proof: status === 'proved' ? ['created_before_late_cleanup', 'cleared_in_late_rules', 'absent_from_initial_levels_and_winconditions'] : [],
            blockers,
            evidence: creators.concat(clearers).map(entry => entry.rule.id),
        }));
    }
    return results;
}

function addValues(target, values) {
    for (const value of values) target.add(value);
}

function termObjects(term) {
    return term.expanded_objects || [];
}

function movementExpansions(movement) {
    if (movement === 'randomdir') return CARDINAL_MOVEMENTS.concat(['moving', 'randomdir']);
    if (CARDINAL_MOVEMENTS.includes(movement)) return [movement, 'moving'];
    return [movement];
}

function ruleFlowReads(rule) {
    const objectPresent = new Set();
    const objectAbsent = new Set();
    const movement = [];
    for (const term of rule.summary.lhs_terms) {
        if (term.kind === 'present') {
            addValues(objectPresent, termObjects(term));
            if (term.movement !== null) {
                for (const objectName of termObjects(term)) {
                    movement.push({ object: objectName, movement: term.movement });
                }
            }
        } else if (term.kind === 'absent') {
            addValues(objectAbsent, termObjects(term));
        }
    }
    return { object_present: objectPresent, object_absent: objectAbsent, movement };
}

function movementTermKeys(terms) {
    const keys = new Set();
    for (const term of terms) {
        if (term.kind !== 'present' || term.movement === null) continue;
        for (const objectName of termObjects(term)) {
            keys.add(`${objectName}:${term.movement}`);
        }
    }
    return keys;
}

function presentObjectSet(terms) {
    const objects = new Set();
    for (const term of terms) {
        if (term.kind === 'present' || term.kind === 'random_object') {
            addValues(objects, termObjects(term));
        }
    }
    return objects;
}

function layerObjectsForObject(psTagged, objectName) {
    const layer = layerForObject(psTagged, objectName);
    const collisionLayer = psTagged.collision_layers.find(item => item.id === layer);
    return collisionLayer ? collisionLayer.objects : [];
}

function ruleFlowWrites(psTagged, rule) {
    const objectPresent = new Set();
    const objectAbsent = new Set();
    const movement = [];
    const lhsPresent = presentObjectSet(rule.summary.lhs_terms.filter(term => term.kind === 'present'));
    const rhsPresent = presentObjectSet(rule.summary.rhs_terms);

    if (rule.tags.object_mutating) {
        for (const term of rule.summary.rhs_terms) {
            if (term.kind === 'present' || term.kind === 'random_object') {
                const writtenObjects = termObjects(term).filter(objectName =>
                    term.kind === 'random_object' || !lhsPresent.has(objectName)
                );
                addValues(objectPresent, writtenObjects);
                for (const objectName of writtenObjects) {
                    for (const sibling of layerObjectsForObject(psTagged, objectName)) {
                        if (sibling !== objectName && !rhsPresent.has(sibling)) {
                            objectAbsent.add(sibling);
                        }
                    }
                }
            } else if (term.kind === 'absent') {
                addValues(objectAbsent, termObjects(term));
            }
        }
        for (const objectName of lhsPresent) {
            if (!rhsPresent.has(objectName)) {
                objectAbsent.add(objectName);
            }
        }
    }

    if (rule.tags.writes_movement) {
        const lhsMovementKeys = movementTermKeys(rule.summary.lhs_terms);
        const rhsMovementKeys = movementTermKeys(rule.summary.rhs_terms);
        for (const term of rule.summary.rhs_terms) {
            if (term.kind !== 'present' || term.movement === null) continue;
            for (const objectName of termObjects(term)) {
                if (lhsMovementKeys.has(`${objectName}:${term.movement}`)) continue;
                for (const movementName of movementExpansions(term.movement)) {
                    movement.push({ object: objectName, movement: movementName });
                }
            }
        }
        for (const key of lhsMovementKeys) {
            if (rhsMovementKeys.has(key)) continue;
            const objectName = key.slice(0, key.lastIndexOf(':'));
            if (rhsPresent.has(objectName)) {
                movement.push({ object: objectName, movement: 'stationary' });
            }
        }
    }

    return { object_present: objectPresent, object_absent: objectAbsent, movement };
}

function movementWriteMayEnableRead(write, read) {
    if (write.object !== read.object) return false;
    if (write.movement === read.movement) return true;
    if (read.movement === 'moving' || read.movement === 'randomdir' || read.movement === 'orthogonal') {
        return CARDINAL_MOVEMENTS.includes(write.movement) || write.movement === 'moving' || write.movement === 'randomdir';
    }
    return false;
}

function ruleMayEnableRule(writes, reads) {
    const reasons = [];
    if (Array.from(writes.object_present).some(objectName => reads.object_present.has(objectName))) {
        reasons.push('object_presence');
    }
    if (Array.from(writes.object_absent).some(objectName => reads.object_absent.has(objectName))) {
        reasons.push('object_absence');
    }
    if (writes.movement.some(write => reads.movement.some(read => movementWriteMayEnableRead(write, read)))) {
        reasons.push('movement');
    }
    return reasons;
}

function connectedComponents(ruleIds, edges) {
    const parent = new Map(ruleIds.map(id => [id, id]));
    function find(id) {
        const current = parent.get(id);
        if (current === id) return id;
        const root = find(current);
        parent.set(id, root);
        return root;
    }
    function union(left, right) {
        const leftRoot = find(left);
        const rightRoot = find(right);
        if (leftRoot !== rightRoot) parent.set(rightRoot, leftRoot);
    }
    for (const edge of edges) {
        union(edge.from, edge.to);
    }
    const components = new Map();
    for (const id of ruleIds) {
        const root = find(id);
        if (!components.has(root)) components.set(root, []);
        components.get(root).push(id);
    }
    return Array.from(components.values()).sort((left, right) =>
        ruleIds.indexOf(left[0]) - ruleIds.indexOf(right[0])
    );
}

function deriveRulegroupFlowFacts(psTagged) {
    const results = [];
    for (const section of psTagged.rule_sections) {
        for (const group of section.groups) {
            if (group.rules.length <= 1) continue;
            const ruleIds = group.rules.map(rule => rule.id);
            const readsByRule = new Map(group.rules.map(rule => [rule.id, ruleFlowReads(rule)]));
            const writesByRule = new Map(group.rules.map(rule => [rule.id, ruleFlowWrites(psTagged, rule)]));
            const interactionEdges = [];
            const rerunMasks = Object.fromEntries(ruleIds.map(id => [id, []]));
            for (let fromIndex = 0; fromIndex < group.rules.length; fromIndex++) {
                const fromRule = group.rules[fromIndex];
                const writes = writesByRule.get(fromRule.id);
                for (let toIndex = 0; toIndex < group.rules.length; toIndex++) {
                    const toRule = group.rules[toIndex];
                    const reasons = ruleMayEnableRule(writes, readsByRule.get(toRule.id));
                    if (reasons.length === 0) continue;
                    interactionEdges.push({ from: fromRule.id, to: toRule.id, reasons });
                    if (toIndex <= fromIndex) {
                        rerunMasks[fromRule.id].push(toRule.id);
                    }
                }
            }
            for (const ruleId of ruleIds) {
                rerunMasks[ruleId] = uniqueSorted(rerunMasks[ruleId]);
            }
            const components = connectedComponents(ruleIds, interactionEdges);
            const blockers = [];
            if (components.length <= 1) blockers.push('single_component');
            if (group.random) blockers.push('random_rule_group');
            if (group.rules.some(rule => rule.rigid)) blockers.push('rigid_rule');
            if (group.rules.some(rule => rule.summary.semantic_commands.length > 0)) blockers.push('semantic_command');
            const status = blockers.length === 0 ? 'candidate' : 'rejected';
            results.push(fact('rulegroup_flow', `${group.id}_flow`, status, {
                subjects: { groups: [group.id] },
                value: {
                    rule_ids: ruleIds,
                    interaction_edges: interactionEdges,
                    rerun_masks: rerunMasks,
                    components,
                    split_candidate: status === 'candidate',
                },
                proof: status === 'candidate' ? ['multiple_independent_rule_components'] : [],
                blockers,
                evidence: ruleIds,
            }));
        }
    }
    return results;
}

function deriveFacts(psTagged) {
    return {
        mergeability: deriveMergeabilityFacts(psTagged),
        movement_action: deriveMovementActionFacts(psTagged),
        count_layer_invariants: deriveCountLayerInvariantFacts(psTagged),
        transient_boundary: deriveTransientBoundaryFacts(psTagged),
        rulegroup_flow: deriveRulegroupFlowFacts(psTagged),
    };
}

function filterFacts(facts, familyFilter) {
    if (!familyFilter) return facts;
    return { [familyFilter]: facts[familyFilter] || [] };
}

function summarizeFacts(facts) {
    const summary = { proved: 0, candidate: 0, rejected: 0 };
    for (const familyFacts of Object.values(facts)) {
        for (const item of familyFacts) {
            if (Object.prototype.hasOwnProperty.call(summary, item.status)) {
                summary[item.status]++;
            }
        }
    }
    return summary;
}

function analyzeSource(source, options = {}) {
    const sourcePath = options.sourcePath || '<memory>';
    let compiled;
    try {
        compiled = compileSemanticSource(source, {
            includeWinConditions: true,
            throwOnError: false,
        });
    } catch (error) {
        return {
            schema: SCHEMA,
            source: { path: sourcePath },
            status: 'compile_error',
            errors: [error && error.message ? error.message : String(error)],
            ps_tagged: null,
            facts: emptyFacts(),
            summary: { proved: 0, candidate: 0, rejected: 0 },
        };
    }

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
    const facts = filterFacts(deriveFacts(psTagged), options.familyFilter);
    const report = {
        schema: SCHEMA,
        source: { path: sourcePath },
        status: 'ok',
        ps_tagged: psTagged,
        facts,
        summary: summarizeFacts(facts),
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

function discoverInputFiles(inputs) {
    const files = [];
    for (const input of inputs) {
        const stat = fs.statSync(input);
        if (stat.isDirectory()) {
            for (const entry of fs.readdirSync(input).sort()) {
                const fullPath = path.join(input, entry);
                if (fs.statSync(fullPath).isFile() && fullPath.endsWith('.txt')) {
                    files.push(fullPath);
                }
            }
        } else {
            files.push(input);
        }
    }
    return files;
}

function analyzePaths(inputs, options = {}) {
    return discoverInputFiles(inputs)
        .filter(filePath => !options.gameFilter || filePath.toLowerCase().includes(options.gameFilter.toLowerCase()))
        .map(filePath => analyzeFile(filePath, options));
}

module.exports = {
    SCHEMA,
    analyzeFile,
    analyzePaths,
    analyzeSource,
    discoverInputFiles,
};
