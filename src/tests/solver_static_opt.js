'use strict';

/**
 * Solver-only static optimizations (runs from pluginOptimizationHook in compiler).
 * After structural edits, re-syncs level bitmasks with new object ids / stride
 * (requires compiler globals: BitVec, STRIDE_OBJ, generateExtraMembers, etc.).
 */

function takeSolverStructuralSnapshot(state) {
    const strideObj = typeof STRIDE_OBJ !== 'undefined' ? STRIDE_OBJ : state.STRIDE_OBJ;
    const idDict = state.idDict ? state.idDict.slice() : [];
    const objectCount = state.objectCount | 0;
    const levels = [];
    if (Array.isArray(state.levels)) {
        for (let i = 0; i < state.levels.length; i++) {
            const level = state.levels[i];
            if (!level || level.message !== undefined || !level.objects) {
                levels.push(null);
            } else {
                levels.push(new Int32Array(level.objects));
            }
        }
    }
    return { strideObj, idDict, objectCount, levels };
}

function remapLevelObjectsAfterIdRewrite(state, snapshot, nameRedirect) {
    const oldStride = snapshot.strideObj;
    const oldDict = snapshot.idDict;
    const oldCount = snapshot.objectCount | 0;
    if (!Array.isArray(state.levels) || !Array.isArray(snapshot.levels)) {
        return;
    }
    const newStride = typeof STRIDE_OBJ !== 'undefined' ? STRIDE_OBJ : state.STRIDE_OBJ;
    for (let i = 0; i < state.levels.length; i++) {
        const level = state.levels[i];
        const oldBuf = snapshot.levels[i];
        if (!level || level.message !== undefined || !level.objects || !oldBuf) {
            continue;
        }
        const next = new Int32Array(level.n_tiles * newStride);
        for (let t = 0; t < level.n_tiles; t++) {
            const newCell = new BitVec(newStride);
            for (let oid = 0; oid < oldCount; oid++) {
                const word = (oid / 32) | 0;
                const bit = oid & 31;
                if ((oldBuf[t * oldStride + word] & (1 << bit)) !== 0) {
                    const name = oldDict[oid];
                    if (!name) {
                        continue;
                    }
                    let targetName = name;
                    if (typeof nameRedirect === 'function') {
                        const mapped = nameRedirect(name);
                        if (mapped === null) {
                            continue;
                        }
                        if (mapped !== undefined) {
                            targetName = mapped;
                        }
                    }
                    if (targetName && state.objects[targetName]) {
                        newCell.ibitset(state.objects[targetName].id);
                    }
                }
            }
            for (let w = 0; w < newStride; w++) {
                next[t * newStride + w] = newCell.data[w];
            }
        }
        level.objects = next;
        level.layerCount = state.collisionLayers.length;
    }
}

function rebuildCompiledStateAfterSolverStructuralEdit(state, nameRedirect) {
    if (!state || state.invalid > 0) {
        return;
    }
    const snapshot = takeSolverStructuralSnapshot(state);
    generateExtraMembers(state);
    remapLevelObjectsAfterIdRewrite(state, snapshot, nameRedirect);
    generateMasks(state);
    cacheAllRuleNames(state);
    removeDuplicateRules(state);
}

const INERT_COMMAND_NAMES = new Set([
    'message',
    'sfx0', 'sfx1', 'sfx2', 'sfx3', 'sfx4', 'sfx5',
    'sfx6', 'sfx7', 'sfx8', 'sfx9', 'sfx10',
]);

function defaultTelemetry() {
    return {
        removed_inert_rules: 0,
        removed_cosmetic_objects: 0,
        removed_collision_layers: 0,
        merged_object_aliases: 0,
        merged_object_groups: 0,
        ms_inert: 0,
        ms_cosmetic: 0,
        ms_merge: 0,
    };
}

function inertCommandOnlyRuleSourceLines(report) {
    const lines = new Set();
    if (!report || !report.ps_tagged || !Array.isArray(report.ps_tagged.rule_sections)) {
        return lines;
    }
    for (const section of report.ps_tagged.rule_sections) {
        if (!section || !Array.isArray(section.groups)) continue;
        for (const group of section.groups) {
            if (!group || !Array.isArray(group.rules)) continue;
            for (const rule of group.rules) {
                if (rule && rule.tags && rule.tags.inert_command_only && Number.isFinite(rule.source_line)) {
                    lines.add(rule.source_line);
                }
            }
        }
    }
    return lines;
}

function isInertCommandOnlyCompiledRule(rule, inertSourceLines) {
    if (!rule || !inertSourceLines.has(rule.lineNumber)) return false;
    if (rule.randomRule) return false;
    if (rule.hasReplacements) return false;
    if (!Array.isArray(rule.commands) || rule.commands.length === 0) return false;
    return rule.commands.every(command => INERT_COMMAND_NAMES.has(command[0]));
}

function dropSolverInertCommandOnlyRules(state, inertSourceLines) {
    let removed = 0;
    if (!Array.isArray(state.rules)) {
        return removed;
    }
    const next = [];
    for (const rule of state.rules) {
        if (isInertCommandOnlyCompiledRule(rule, inertSourceLines)) {
            removed++;
            continue;
        }
        next.push(rule);
    }
    state.rules = next;
    return removed;
}

function scanCellForNames(cell, out) {
    if (!Array.isArray(cell)) return;
    for (let i = 0; i + 1 < cell.length; i += 2) {
        const nm = cell[i + 1];
        if (nm && nm !== '...' && nm !== 'random') {
            out.add(nm);
        }
    }
}

function collectReferencedObjectNamesFromFlatRules(rules) {
    const out = new Set();
    if (!Array.isArray(rules)) return out;
    for (const rule of rules) {
        if (!rule) continue;
        for (const row of rule.lhs || []) {
            for (const cell of row) scanCellForNames(cell, out);
        }
        for (const row of rule.rhs || []) {
            for (const cell of row) scanCellForNames(cell, out);
        }
    }
    return out;
}

function collectWinconditionLegendRefs(state) {
    const out = new Set();
    for (const wc of state.winconditions || []) {
        const n = wc.length;
        for (let i = 0; i < n - 1; i++) {
            const tok = wc[i];
            if (typeof tok !== 'string') continue;
            if (tok in state.objects) {
                out.add(tok);
                continue;
            }
            if (state.aggregatesDict && Object.prototype.hasOwnProperty.call(state.aggregatesDict, tok)) {
                out.add(tok);
                continue;
            }
            if (state.propertiesDict && Object.prototype.hasOwnProperty.call(state.propertiesDict, tok)) {
                out.add(tok);
            }
        }
    }
    return out;
}

/** Object names that appear on any compiled non-message level map (bitset cells). */
function collectObjectNamesFromCompiledLevels(state) {
    const out = new Set();
    if (!state) {
        return out;
    }
    const strideObj = typeof STRIDE_OBJ !== 'undefined' ? STRIDE_OBJ : state.STRIDE_OBJ;
    const objectCount = state.objectCount | 0;
    const idDict = state.idDict;
    if (!strideObj || objectCount <= 0 || !idDict) {
        return out;
    }
    if (!Array.isArray(state.levels)) {
        return out;
    }
    for (const level of state.levels) {
        if (!level || level.message !== undefined || !level.objects) {
            continue;
        }
        const buf = level.objects;
        const nTiles = level.n_tiles | 0;
        for (let t = 0; t < nTiles; t++) {
            for (let oid = 0; oid < objectCount; oid++) {
                const word = (oid / 32) | 0;
                const bit = oid & 31;
                const idx = t * strideObj + word;
                if (idx < 0 || idx >= buf.length) {
                    continue;
                }
                if ((buf[idx] & (1 << bit)) !== 0) {
                    const name = idDict[oid];
                    if (name) {
                        out.add(name);
                    }
                }
            }
        }
    }
    return out;
}

function expandLegendRefsToConcreteObjectNames(state, rawNames) {
    const concrete = new Set();
    const visiting = new Set();
    function walk(name) {
        if (!name || typeof name !== 'string') return;
        if (visiting.has(name)) return;
        if (state.objects && Object.prototype.hasOwnProperty.call(state.objects, name)) {
            concrete.add(name);
            return;
        }
        visiting.add(name);
        try {
            const agg = state.aggregatesDict && state.aggregatesDict[name];
            if (agg) {
                for (let j = 0; j < agg.length; j++) walk(agg[j]);
                return;
            }
            const prop = state.propertiesDict && state.propertiesDict[name];
            if (prop) {
                for (let j = 0; j < prop.length; j++) walk(prop[j]);
            }
        } finally {
            visiting.delete(name);
        }
    }
    for (const n of rawNames) walk(n);
    return concrete;
}

function legendMentionsObject(state, objectName) {
    for (const row of state.legend_synonyms || []) {
        if (row[0] === objectName || row[1] === objectName) return true;
    }
    for (const row of state.legend_aggregates || []) {
        for (let j = 1; j < row.length; j++) {
            if (row[j] === objectName) return true;
        }
    }
    for (const row of state.legend_properties || []) {
        for (let j = 1; j < row.length; j++) {
            if (row[j] === objectName) return true;
        }
    }
    return false;
}

function collectCosmeticNames(report) {
    const set = new Set();
    const pt = report && report.ps_tagged;
    if (!pt || !Array.isArray(pt.objects)) return set;
    for (const o of pt.objects) {
        if (o && o.tags && o.tags.cosmetic) {
            set.add(o.name);
        }
    }
    return set;
}

function applyNameSubstitutionToCell(cell, renameFn) {
    if (!Array.isArray(cell)) return;
    for (let i = 0; i + 1 < cell.length; i += 2) {
        const nm = cell[i + 1];
        if (!nm || nm === '...' || nm === 'random') continue;
        const next = renameFn(nm);
        if (next !== undefined && next !== nm) {
            cell[i + 1] = next;
        }
    }
}

function applyNameSubstitutionToFlatRules(rules, renameFn) {
    if (!Array.isArray(rules)) return;
    for (const rule of rules) {
        if (!rule) continue;
        for (const row of rule.lhs || []) {
            for (const cell of row) applyNameSubstitutionToCell(cell, renameFn);
        }
        for (const row of rule.rhs || []) {
            for (const cell of row) applyNameSubstitutionToCell(cell, renameFn);
        }
    }
}

function applyNameSubstitutionToWinconditions(state, renameFn) {
    for (const wc of state.winconditions || []) {
        const n = wc.length;
        for (let i = 0; i < n - 1; i++) {
            const tok = wc[i];
            if (typeof tok !== 'string') continue;
            const mapped = renameFn(tok);
            if (mapped === null) {
                continue;
            }
            if (mapped !== undefined && mapped !== tok) {
                wc[i] = mapped;
            }
        }
    }
}

function passCosmeticPrune(state, report, telemetry) {
    const cosmetic = collectCosmeticNames(report);
    const rawReferenced = new Set();
    for (const n of collectReferencedObjectNamesFromFlatRules(state.rules)) rawReferenced.add(n);
    for (const n of collectWinconditionLegendRefs(state)) rawReferenced.add(n);
    for (const n of collectObjectNamesFromCompiledLevels(state)) rawReferenced.add(n);
    const referencedObjects = expandLegendRefsToConcreteObjectNames(state, rawReferenced);

    const toRemove = new Set();
    for (const name of cosmetic) {
        if (name === 'player' || name === 'background') continue;
        if (!state.objects[name]) continue;
        if (referencedObjects.has(name)) continue;
        if (legendMentionsObject(state, name)) continue;
        toRemove.add(name);
    }
    if (toRemove.size === 0) {
        return;
    }
    for (const name of toRemove) {
        delete state.objects[name];
        telemetry.removed_cosmetic_objects++;
    }
    for (let li = 0; li < state.collisionLayers.length; li++) {
        state.collisionLayers[li] = state.collisionLayers[li].filter(n => !toRemove.has(n));
    }
    const beforeLayers = state.collisionLayers.length;
    state.collisionLayers = state.collisionLayers.filter(layer => layer.length > 0);
    telemetry.removed_collision_layers += beforeLayers - state.collisionLayers.length;

    state.legend_synonyms = (state.legend_synonyms || []).filter(row => !toRemove.has(row[0]) && !toRemove.has(row[1]));
    state.legend_aggregates = (state.legend_aggregates || []).filter(row => {
        for (let j = 1; j < row.length; j++) {
            if (toRemove.has(row[j])) return false;
        }
        return true;
    });
    state.legend_properties = (state.legend_properties || []).filter(row => {
        for (let j = 1; j < row.length; j++) {
            if (toRemove.has(row[j])) return false;
        }
        return true;
    });

    pruneDegenerateLegendRows(state);

    function nameRedirect(nm) {
        return toRemove.has(nm) ? null : undefined;
    }
    rebuildCompiledStateAfterSolverStructuralEdit(state, nameRedirect);
}

function mergeabilityCandidates(report) {
    const facts = report && report.facts && report.facts.mergeability;
    if (!Array.isArray(facts)) return [];
    const pairs = [];
    for (const f of facts) {
        if (!f || f.status !== 'candidate' || !f.subjects || !Array.isArray(f.subjects.objects)) continue;
        if (f.subjects.objects.length !== 2) continue;
        pairs.push([f.subjects.objects[0], f.subjects.objects[1]]);
    }
    return pairs;
}

function buildMergeAliasMap(pairs, cosmeticNames, state) {
    const namesInPairs = new Set();
    for (const [a, b] of pairs) {
        if (cosmeticNames.has(a) || cosmeticNames.has(b)) continue;
        if (!state.objects[a] || !state.objects[b]) continue;
        namesInPairs.add(a);
        namesInPairs.add(b);
    }
    const parent = new Map();
    for (const n of namesInPairs) {
        parent.set(n, n);
    }
    function find(x) {
        const p = parent.get(x);
        if (p === undefined) return x;
        if (p === x) return x;
        const r = find(p);
        parent.set(x, r);
        return r;
    }
    function union(a, b) {
        let ra = find(a);
        let rb = find(b);
        if (ra === rb) return;
        parent.set(rb, ra);
    }
    for (const [a, b] of pairs) {
        if (!namesInPairs.has(a) || !namesInPairs.has(b)) continue;
        union(a, b);
    }
    const comp = new Map();
    for (const n of namesInPairs) {
        const r = find(n);
        if (!comp.has(r)) comp.set(r, []);
        comp.get(r).push(n);
    }
    const alias = {};
    let groups = 0;
    for (const members of comp.values()) {
        if (members.length < 2) continue;
        members.sort();
        const canon = members[0];
        groups++;
        for (let i = 1; i < members.length; i++) {
            alias[members[i]] = canon;
        }
    }
    return { alias, groups };
}

function applyMergePass(state, report, telemetry) {
    const cosmetic = collectCosmeticNames(report);
    const pairs = mergeabilityCandidates(report);
    const { alias, groups } = buildMergeAliasMap(pairs, cosmetic, state);
    telemetry.merged_object_groups = groups;
    const keys = Object.keys(alias);
    if (keys.length === 0) {
        return;
    }
    function renameFn(nm) {
        return Object.prototype.hasOwnProperty.call(alias, nm) ? alias[nm] : undefined;
    }
    applyNameSubstitutionToFlatRules(state.rules, renameFn);
    applyNameSubstitutionToWinconditions(state, renameFn);

    for (const layer of state.collisionLayers) {
        for (let j = 0; j < layer.length; j++) {
            if (alias[layer[j]]) layer[j] = alias[layer[j]];
        }
    }
    for (let li = 0; li < state.collisionLayers.length; li++) {
        const seen = new Set();
        const out = [];
        for (const n of state.collisionLayers[li]) {
            if (seen.has(n)) continue;
            seen.add(n);
            out.push(n);
        }
        state.collisionLayers[li] = out;
    }

    for (const row of state.legend_synonyms || []) {
        if (alias[row[0]]) row[0] = alias[row[0]];
        if (alias[row[1]]) row[1] = alias[row[1]];
    }
    for (const row of state.legend_aggregates || []) {
        for (let j = 1; j < row.length; j++) {
            if (alias[row[j]]) row[j] = alias[row[j]];
        }
        const uniq = [row[0]];
        const s = new Set();
        for (let j = 1; j < row.length; j++) {
            if (!s.has(row[j])) {
                s.add(row[j]);
                uniq.push(row[j]);
            }
        }
        for (let j = 0; j < uniq.length; j++) row[j] = uniq[j];
        row.length = uniq.length;
    }
    for (const row of state.legend_properties || []) {
        for (let j = 1; j < row.length; j++) {
            if (alias[row[j]]) row[j] = alias[row[j]];
        }
        const uniq = [row[0]];
        const s = new Set();
        for (let j = 1; j < row.length; j++) {
            if (!s.has(row[j])) {
                s.add(row[j]);
                uniq.push(row[j]);
            }
        }
        for (let j = 0; j < uniq.length; j++) row[j] = uniq[j];
        row.length = uniq.length;
    }

    for (const victim of keys) {
        delete state.objects[victim];
        telemetry.merged_object_aliases++;
    }

    sanitizeLegendSynonymsAfterMerge(state);
    pruneDegenerateLegendRows(state);

    function nameRedirect(nm) {
        return Object.prototype.hasOwnProperty.call(alias, nm) ? alias[nm] : undefined;
    }
    rebuildCompiledStateAfterSolverStructuralEdit(state, nameRedirect);
}

function pruneDegenerateLegendRows(state) {
    function rowOk(row) {
        return Array.isArray(row) && row.length >= 2 && row[0] !== undefined && row[0] !== '';
    }
    state.legend_aggregates = (state.legend_aggregates || []).filter(rowOk);
    state.legend_properties = (state.legend_properties || []).filter(rowOk);
}

function sanitizeLegendSynonymsAfterMerge(state) {
    state.legend_synonyms = (state.legend_synonyms || []).filter(row => Array.isArray(row) && row.length >= 2 && row[0] !== row[1]);
    dedupeLegendSynonymKeysKeepFirst(state);
}

function dedupeLegendSynonymKeysKeepFirst(state) {
    const seen = new Set();
    const rows = state.legend_synonyms || [];
    const out = [];
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        if (!Array.isArray(row) || row.length < 2) continue;
        const key = row[0];
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(row);
    }
    state.legend_synonyms = out;
}

function parseSolverOptPassList(arg) {
    const passes = { inert: false, cosmetic: false, merge: false };
    const parts = String(arg || '')
        .split(',')
        .map(s => s.trim().toLowerCase())
        .filter(Boolean);
    if (parts.includes('all')) {
        passes.inert = true;
        passes.cosmetic = true;
        passes.merge = true;
        return passes;
    }
    for (const p of parts) {
        if (p === 'inert') passes.inert = true;
        else if (p === 'cosmetic') passes.cosmetic = true;
        else if (p === 'merge') passes.merge = true;
        else throw new Error(`Unknown solver optimization pass: ${p}`);
    }
    return passes;
}

function resolveSolverPasses(options) {
    if (options.solverOptParityBaseline) {
        return { inert: false, cosmetic: false, merge: false };
    }
    const passes = { inert: false, cosmetic: false, merge: false };
    if (options.solverOptPasses) {
        Object.assign(passes, options.solverOptPasses);
    }
    if (options.solverOptimizeStatic) {
        passes.inert = true;
    }
    return passes;
}

function solverPassesNeedFullStaticReport(passes) {
    return !!(passes && (passes.cosmetic || passes.merge));
}

function effectiveSolverPassesForHook(staticAnalysisReport, passes) {
    const p = passes || { inert: false, cosmetic: false, merge: false };
    if (!staticAnalysisReport || staticAnalysisReport.status !== 'ok') {
        return { inert: !!p.inert, cosmetic: false, merge: false };
    }
    return { inert: !!p.inert, cosmetic: !!p.cosmetic, merge: !!p.merge };
}

function formatSolverOptimizationHumanSuffixFromTotals(t) {
    if (!t) return '';
    const parts = [];
    if ((t.static_optimization_removed_rules || 0) > 0) {
        parts.push(`inert_rules=${t.static_optimization_removed_rules}`);
    }
    if ((t.removed_cosmetic_objects || 0) > 0) {
        parts.push(`cosmetic_objs=${t.removed_cosmetic_objects}`);
    }
    if ((t.removed_collision_layers || 0) > 0) {
        parts.push(`empty_layers=${t.removed_collision_layers}`);
    }
    if ((t.merged_object_aliases || 0) > 0) {
        parts.push(`merge_aliases=${t.merged_object_aliases}`);
    }
    if ((t.merged_object_groups || 0) > 0) {
        parts.push(`merge_groups=${t.merged_object_groups}`);
    }
    const msInert = t.solver_opt_ms_inert || 0;
    const msCos = t.solver_opt_ms_cosmetic || 0;
    const msMrg = t.solver_opt_ms_merge || 0;
    const hookMs = msInert + msCos + msMrg;
    if (msInert > 0) {
        parts.push(`opt_ms_inert=${msInert.toFixed(3)}`);
    }
    if (msCos > 0) {
        parts.push(`opt_ms_cosmetic=${msCos.toFixed(3)}`);
    }
    if (msMrg > 0) {
        parts.push(`opt_ms_merge=${msMrg.toFixed(3)}`);
    }
    if (hookMs > 0) {
        parts.push(`opt_hook_ms=${hookMs.toFixed(3)}`);
    }
    if (t.solver_optimization_gated) {
        parts.push('opt_gated=1');
    }
    if (parts.length === 0) return '';
    return `solver_optimization: ${parts.join(' ')}`;
}

function buildSolverOptimizationJsonTotals(t) {
    if (!t) return null;
    const msInert = t.solver_opt_ms_inert || 0;
    const msCos = t.solver_opt_ms_cosmetic || 0;
    const msMrg = t.solver_opt_ms_merge || 0;
    const hookMs = msInert + msCos + msMrg;
    const removedInert = t.static_optimization_removed_rules || 0;
    const removedCos = t.removed_cosmetic_objects || 0;
    const removedLayers = t.removed_collision_layers || 0;
    const mergedAliases = t.merged_object_aliases || 0;
    const mergedGroups = t.merged_object_groups || 0;
    const gated = !!t.solver_optimization_gated;
    if (
        removedInert === 0
        && removedCos === 0
        && removedLayers === 0
        && mergedAliases === 0
        && mergedGroups === 0
        && hookMs === 0
        && !gated
    ) {
        return null;
    }
    const out = {
        removed_inert_rules: removedInert,
        removed_cosmetic_objects: removedCos,
        removed_collision_layers: removedLayers,
        merged_object_aliases: mergedAliases,
        merged_object_groups: mergedGroups,
        ms_inert: msInert,
        ms_cosmetic: msCos,
        ms_merge: msMrg,
        ms_hook: hookMs,
    };
    if (gated) {
        out.gated = true;
    }
    return out;
}

function createSolverOptimizationHook(staticAnalysisReport, passes) {
    return (compiledState) => {
        const telemetry = defaultTelemetry();
        if (!compiledState || compiledState.invalid > 0) {
            compiledState.solverOptimizationTelemetry = telemetry;
            return;
        }
        const p = passes || {};
        if (p.inert) {
            const t0 = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
            const lines = inertCommandOnlyRuleSourceLines(staticAnalysisReport);
            if (lines.size > 0) {
                telemetry.removed_inert_rules += dropSolverInertCommandOnlyRules(compiledState, lines);
            }
            const t1 = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
            telemetry.ms_inert += t1 - t0;
        }
        if (p.cosmetic) {
            const t0 = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
            passCosmeticPrune(compiledState, staticAnalysisReport, telemetry);
            const t1 = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
            telemetry.ms_cosmetic += t1 - t0;
        }
        if (p.merge) {
            const t0 = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
            applyMergePass(compiledState, staticAnalysisReport, telemetry);
            const t1 = typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();
            telemetry.ms_merge += t1 - t0;
        }
        compiledState.solverOptimizationTelemetry = telemetry;
    };
}

module.exports = {
    INERT_COMMAND_NAMES,
    defaultTelemetry,
    inertCommandOnlyRuleSourceLines,
    isInertCommandOnlyCompiledRule,
    dropSolverInertCommandOnlyRules,
    applyNameSubstitutionToWinconditions,
    parseSolverOptPassList,
    resolveSolverPasses,
    solverPassesNeedFullStaticReport,
    effectiveSolverPassesForHook,
    formatSolverOptimizationHumanSuffixFromTotals,
    buildSolverOptimizationJsonTotals,
    createSolverOptimizationHook,
    collectWinconditionLegendRefs,
    collectObjectNamesFromCompiledLevels,
    expandLegendRefsToConcreteObjectNames,
};
