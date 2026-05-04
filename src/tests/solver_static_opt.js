'use strict';

/**
 * Solver-only static optimizations for compiled PuzzleScript state.
 *
 * Three independent passes, all opt-in via createSolverOptimizationHook(report, passes):
 *
 *   inert    — drop rules whose only effect is firing a sound or showing a
 *              message (sfx0..sfx10, message). These can't change game state,
 *              so the solver doesn't need to evaluate them.
 *   cosmetic — delete objects flagged "cosmetic" by the static analyzer that
 *              don't appear in any rule, wincondition, or sound row.
 *   merge    — fold mergeability-candidate object pairs into a single canonical
 *              name (alphabetically first). Identical objects are equivalent
 *              for the solver.
 *
 * Hook timing: pluginOptimizationHook fires from compiler.js loadFile() after
 * rule compilation but before generateSoundData / processWinConditions. Cosmetic
 * and merge passes mutate object identity (delete or rename), which invalidates
 * the compiled level bitmasks and object IDs — so each runs a structural rebuild
 * (rebuildAfterStructuralEdit) that re-runs generateExtraMembers and remaps
 * each level's bitmask buffer to the new IDs.
 *
 * Globals consumed (provided by combined_sources.js / compiler.js):
 *   STRIDE_OBJ, BitVec, generateExtraMembers, generateMasks,
 *   cacheAllRuleNames, removeDuplicateRules, commandwords_sfx
 */

// ---------- Constants ----------

const SFX_COMMAND_NAMES = (typeof commandwords_sfx !== 'undefined' && Array.isArray(commandwords_sfx))
    ? new Set(commandwords_sfx)
    : new Set(['sfx0', 'sfx1', 'sfx2', 'sfx3', 'sfx4', 'sfx5', 'sfx6', 'sfx7', 'sfx8', 'sfx9', 'sfx10']);
const INERT_COMMAND_NAMES = new Set([...SFX_COMMAND_NAMES, 'message']);

const TELEMETRY_KEYS = [
    'removed_inert_rules',
    'removed_cosmetic_objects',
    'removed_collision_layers',
    'merged_object_aliases',
    'merged_object_groups',
    'ms_inert',
    'ms_cosmetic',
    'ms_merge',
];

function defaultTelemetry() {
    const t = {};
    for (const k of TELEMETRY_KEYS) t[k] = 0;
    return t;
}

const nowFn = (typeof performance !== 'undefined' && performance.now)
    ? () => performance.now()
    : () => Date.now();

// ---------- Rule-cell walking ----------
// Rule cells are flat arrays of [direction, name, direction, name, ...].
// '...' and 'random' are wildcards, not object names.

function isObjectNameTok(name) {
    return typeof name === 'string' && name !== '' && name !== '...' && name !== 'random';
}

function forEachRuleCellName(rule, fn) {
    if (!rule) return;
    for (const side of ['lhs', 'rhs']) {
        const rows = rule[side];
        if (!Array.isArray(rows)) continue;
        for (const row of rows) {
            for (const cell of row) {
                if (!Array.isArray(cell)) continue;
                for (let i = 1; i < cell.length; i += 2) {
                    if (isObjectNameTok(cell[i])) fn(cell, i);
                }
            }
        }
    }
}

function collectNamesFromRules(rules, out) {
    if (!Array.isArray(rules)) return;
    for (const rule of rules) {
        forEachRuleCellName(rule, (cell, i) => out.add(cell[i]));
    }
}

// ---------- Reference collection ----------

function collectWinconditionLegendRefs(state) {
    // Wincondition rows look like: [verb, subject, ...optional 'on' object, lineNumber].
    // The trailing element is always a line number, not a name.
    const out = new Set();
    for (const wc of state.winconditions || []) {
        for (let i = 0; i < wc.length - 1; i++) {
            const tok = wc[i];
            if (typeof tok !== 'string') continue;
            if (tok in (state.objects || {})) { out.add(tok); continue; }
            if (state.aggregatesDict && Object.prototype.hasOwnProperty.call(state.aggregatesDict, tok)) {
                out.add(tok); continue;
            }
            if (state.propertiesDict && Object.prototype.hasOwnProperty.call(state.propertiesDict, tok)) {
                out.add(tok);
            }
        }
    }
    return out;
}

function collectSoundTargetNames(state) {
    const out = new Set();
    for (const sound of state.sounds || []) {
        if (Array.isArray(sound) && Array.isArray(sound[0]) && typeof sound[0][0] === 'string') {
            out.add(sound[0][0]);
        }
    }
    return out;
}

/** Object names that appear on any compiled non-message level map (bitset cells). */
function collectObjectNamesFromCompiledLevels(state) {
    const out = new Set();
    if (!state || !Array.isArray(state.levels)) return out;
    const strideObj = typeof STRIDE_OBJ !== 'undefined' ? STRIDE_OBJ : state.STRIDE_OBJ;
    const objectCount = state.objectCount | 0;
    const idDict = state.idDict;
    if (!strideObj || objectCount <= 0 || !idDict) return out;
    for (const level of state.levels) {
        if (!level || level.message !== undefined || !level.objects) continue;
        const buf = level.objects;
        const nTiles = level.n_tiles | 0;
        for (let t = 0; t < nTiles; t++) {
            const base = t * strideObj;
            for (let w = 0; w < strideObj; w++) {
                let word = buf[base + w] | 0;
                if (word === 0) continue;
                while (word !== 0) {
                    const bitIdx = 31 - Math.clz32(word & -word);
                    const oid = (w << 5) + bitIdx;
                    word &= word - 1;
                    if (oid >= objectCount) break;
                    const name = idDict[oid];
                    if (name) out.add(name);
                }
            }
        }
    }
    return out;
}

function expandLegendRefsToConcreteObjectNames(state, rawNames) {
    const out = new Set();
    const visiting = new Set();
    const objects = state.objects || {};
    const aggDict = state.aggregatesDict || {};
    const propDict = state.propertiesDict || {};
    function walk(name) {
        if (typeof name !== 'string' || visiting.has(name)) return;
        if (Object.prototype.hasOwnProperty.call(objects, name)) {
            out.add(name);
            return;
        }
        visiting.add(name);
        const expansion = aggDict[name] || propDict[name];
        if (Array.isArray(expansion)) for (const child of expansion) walk(child);
        visiting.delete(name);
    }
    for (const n of rawNames) walk(n);
    return out;
}

// ---------- Rename-map application ----------
// rename: Map<oldName, newName | null>
//   string newName → rewrite all references to newName, delete state.objects[old]
//   null           → drop references entirely (caller must guarantee object isn't
//                    referenced from rule cells, winconditions, or sound rows)
//
// Returns the number of collision layers that became empty and were dropped.

function rewriteRulesByRename(rules, rename) {
    if (!Array.isArray(rules)) return;
    for (const rule of rules) {
        forEachRuleCellName(rule, (cell, i) => {
            const target = rename.get(cell[i]);
            if (typeof target === 'string') cell[i] = target;
        });
    }
}

function rewriteWinconditionsByRename(state, rename) {
    for (const wc of state.winconditions || []) {
        for (let i = 0; i < wc.length - 1; i++) {
            if (typeof wc[i] !== 'string') continue;
            const target = rename.get(wc[i]);
            if (typeof target === 'string') wc[i] = target;
        }
    }
}

function rewriteSoundsByRename(state, rename) {
    if (!Array.isArray(state.sounds)) return;
    const next = [];
    for (const sound of state.sounds) {
        if (!Array.isArray(sound) || !Array.isArray(sound[0]) || typeof sound[0][0] !== 'string') {
            next.push(sound);
            continue;
        }
        const old = sound[0][0];
        if (!rename.has(old)) {
            next.push(sound);
            continue;
        }
        const target = rename.get(old);
        if (target === null) continue; // sound row's target was deleted — drop the row
        sound[0][0] = target;
        next.push(sound);
    }
    state.sounds = next;
}

function rewriteCollisionLayersByRename(state, rename) {
    const before = state.collisionLayers.length;
    for (let li = 0; li < state.collisionLayers.length; li++) {
        const seen = new Set();
        const out = [];
        for (let n of state.collisionLayers[li]) {
            if (rename.has(n)) {
                const target = rename.get(n);
                if (target === null) continue;
                n = target;
            }
            if (seen.has(n)) continue;
            seen.add(n);
            out.push(n);
        }
        state.collisionLayers[li] = out;
    }
    state.collisionLayers = state.collisionLayers.filter(layer => layer.length > 0);
    return before - state.collisionLayers.length;
}

function rewriteLegendListByRename(rows, rename) {
    // legend_aggregates / legend_properties rows: [name, member1, member2, ...].
    // Drop the entire row if any member is null-rename or if dedup leaves <2 entries.
    // Mutate rows in place so attached properties (e.g. row.lineNumber, used by
    // generateMasks's sort) are preserved.
    const out = [];
    for (const row of rows || []) {
        if (!Array.isArray(row) || row.length < 2) continue;
        let dropped = false;
        for (let j = 1; j < row.length; j++) {
            if (!rename.has(row[j])) continue;
            const target = rename.get(row[j]);
            if (target === null) { dropped = true; break; }
            row[j] = target;
        }
        if (dropped) continue;
        const seen = new Set();
        let writeIdx = 1;
        for (let j = 1; j < row.length; j++) {
            if (seen.has(row[j])) continue;
            seen.add(row[j]);
            row[writeIdx++] = row[j];
        }
        row.length = writeIdx;
        if (row.length < 2) continue;
        out.push(row);
    }
    return out;
}

function rewriteLegendSynonymsByRename(state, rename) {
    // legend_synonyms rows: [alias, target]. Drop self-aliases and dedup-by-key.
    const out = [];
    const seenKeys = new Set();
    for (const row of state.legend_synonyms || []) {
        if (!Array.isArray(row) || row.length < 2) continue;
        let key = row[0], val = row[1];
        if (rename.has(key)) {
            const target = rename.get(key);
            if (target === null) continue;
            key = row[0] = target;
        }
        if (rename.has(val)) {
            const target = rename.get(val);
            if (target === null) continue;
            val = row[1] = target;
        }
        if (key === val || seenKeys.has(key)) continue;
        seenKeys.add(key);
        out.push(row);
    }
    state.legend_synonyms = out;
}

function applyRenameMap(state, rename) {
    if (rename.size === 0) return 0;
    rewriteRulesByRename(state.rules, rename);
    rewriteRulesByRename(state.lateRules, rename);
    rewriteWinconditionsByRename(state, rename);
    rewriteSoundsByRename(state, rename);
    const droppedLayers = rewriteCollisionLayersByRename(state, rename);
    state.legend_aggregates = rewriteLegendListByRename(state.legend_aggregates, rename);
    state.legend_properties = rewriteLegendListByRename(state.legend_properties, rename);
    rewriteLegendSynonymsByRename(state, rename);
    for (const oldName of rename.keys()) delete state.objects[oldName];
    return droppedLayers;
}

// ---------- Compiled-state rebuild after structural edits ----------
// generateExtraMembers expects spritematrix rows as parser glyph strings;
// the first compile pass rewrites them to numeric 5×5 grids. Convert back
// before re-running.

function restoreGlyphSpriteRows(state) {
    if (!state || !state.objects) return;
    for (const o of Object.values(state.objects)) {
        if (!o || !Array.isArray(o.spritematrix) || o.spritematrix.length === 0) continue;
        if (typeof o.spritematrix[0] === 'string') continue;
        o.spritematrix = o.spritematrix.map(row => {
            if (typeof row === 'string') return row;
            const chars = new Array(row.length);
            for (let c = 0; c < row.length; c++) {
                const cell = row[c];
                chars[c] = (cell === -1 || cell === undefined) ? '.' : String(Number(cell));
            }
            return chars.join('');
        });
    }
}

function snapshotLevels(state) {
    const strideObj = typeof STRIDE_OBJ !== 'undefined' ? STRIDE_OBJ : state.STRIDE_OBJ;
    const objectCount = state.objectCount | 0;
    const idDict = state.idDict ? state.idDict.slice() : [];
    const levels = [];
    for (const level of state.levels || []) {
        levels.push(level && level.objects && level.message === undefined
            ? new Int32Array(level.objects)
            : null);
    }
    return { strideObj, objectCount, idDict, levels };
}

function remapLevelsToNewIds(state, snapshot, rename) {
    if (!Array.isArray(state.levels)) return;
    const { strideObj: oldStride, objectCount: oldCount, idDict: oldDict, levels: oldLevels } = snapshot;
    const newStride = typeof STRIDE_OBJ !== 'undefined' ? STRIDE_OBJ : state.STRIDE_OBJ;
    const objects = state.objects || {};
    // Precompute per-old-id new bit position (-1 if dropped).
    const newIdForOldId = new Int32Array(oldCount).fill(-1);
    for (let oid = 0; oid < oldCount; oid++) {
        let name = oldDict[oid];
        if (!name) continue;
        if (rename.has(name)) {
            const target = rename.get(name);
            if (target === null) continue;
            name = target;
        }
        const obj = objects[name];
        if (obj && typeof obj.id === 'number') newIdForOldId[oid] = obj.id;
    }
    for (let i = 0; i < state.levels.length; i++) {
        const level = state.levels[i];
        const oldBuf = oldLevels[i];
        if (!level || !level.objects || level.message !== undefined || !oldBuf) continue;
        const next = new Int32Array(level.n_tiles * newStride);
        for (let t = 0; t < level.n_tiles; t++) {
            const oldBase = t * oldStride;
            const newBase = t * newStride;
            for (let w = 0; w < oldStride; w++) {
                let word = oldBuf[oldBase + w] | 0;
                while (word !== 0) {
                    const lsb = word & -word;
                    const bitIdx = 31 - Math.clz32(lsb >>> 0);
                    word ^= lsb;
                    const oid = (w << 5) + bitIdx;
                    if (oid >= oldCount) continue;
                    const newId = newIdForOldId[oid];
                    if (newId < 0) continue;
                    next[newBase + ((newId / 32) | 0)] |= 1 << (newId & 31);
                }
            }
        }
        level.objects = next;
        level.layerCount = state.collisionLayers.length;
    }
}

function rebuildAfterStructuralEdit(state, rename) {
    if (!state || state.invalid > 0) return;
    const snapshot = snapshotLevels(state);
    restoreGlyphSpriteRows(state);
    generateExtraMembers(state);
    remapLevelsToNewIds(state, snapshot, rename);
    generateMasks(state);
    cacheAllRuleNames(state);
    removeDuplicateRules(state);
}

// ---------- Inert pass ----------

function inertCommandOnlyRuleSourceLines(report) {
    const lines = new Set();
    if (!report || !report.ps_tagged || !Array.isArray(report.ps_tagged.rule_sections)) return lines;
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

function dropInertCommandOnlyRulesFrom(rules, inertSourceLines) {
    if (!Array.isArray(rules)) return { kept: rules, removed: 0 };
    const kept = [];
    let removed = 0;
    for (const rule of rules) {
        if (isInertCommandOnlyCompiledRule(rule, inertSourceLines)) {
            removed++;
        } else {
            kept.push(rule);
        }
    }
    return { kept, removed };
}

function dropSolverInertCommandOnlyRules(state, inertSourceLines) {
    const main = dropInertCommandOnlyRulesFrom(state.rules, inertSourceLines);
    state.rules = main.kept;
    const late = dropInertCommandOnlyRulesFrom(state.lateRules, inertSourceLines);
    state.lateRules = late.kept;
    return main.removed + late.removed;
}

// ---------- Cosmetic pass ----------

function collectCosmeticNames(report) {
    const set = new Set();
    const objects = report && report.ps_tagged && report.ps_tagged.objects;
    if (!Array.isArray(objects)) return set;
    for (const o of objects) {
        if (o && o.tags && o.tags.cosmetic) set.add(o.name);
    }
    return set;
}

function passCosmeticPrune(state, report, telemetry) {
    const cosmetic = collectCosmeticNames(report);
    if (cosmetic.size === 0) return;

    const raw = new Set();
    collectNamesFromRules(state.rules, raw);
    collectNamesFromRules(state.lateRules, raw);
    for (const n of collectWinconditionLegendRefs(state)) raw.add(n);
    for (const n of collectSoundTargetNames(state)) raw.add(n);
    // Intentionally omit compiled level maps: cosmetic objects may sit on tiles
    // only for display; remapLevelsToNewIds drops those bits during rebuild.
    const referenced = expandLegendRefsToConcreteObjectNames(state, raw);

    const rename = new Map();
    for (const name of cosmetic) {
        if (name === 'player') continue;
        if (!state.objects[name]) continue;
        if (referenced.has(name)) continue;
        rename.set(name, null);
    }
    if (rename.size === 0) return;

    telemetry.removed_cosmetic_objects += rename.size;
    telemetry.removed_collision_layers += applyRenameMap(state, rename);
    rebuildAfterStructuralEdit(state, rename);
}

// ---------- Merge pass ----------

function mergeabilityCandidatePairs(report) {
    const pairs = [];
    const facts = report && report.facts && report.facts.mergeability;
    if (!Array.isArray(facts)) return pairs;
    for (const f of facts) {
        if (!f || f.status !== 'candidate') continue;
        if (!f.subjects || !Array.isArray(f.subjects.objects)) continue;
        if (f.subjects.objects.length !== 2) continue;
        pairs.push([f.subjects.objects[0], f.subjects.objects[1]]);
    }
    return pairs;
}

function buildMergeAliasMap(pairs, cosmeticNames, state) {
    // Union-find over pairs whose endpoints both survive (not cosmetic-removed,
    // still present in state.objects). Each component's alphabetically-first
    // member is the canonical name; everyone else becomes an alias to it.
    const present = new Set();
    for (const [a, b] of pairs) {
        if (cosmeticNames.has(a) || cosmeticNames.has(b)) continue;
        if (!state.objects[a] || !state.objects[b]) continue;
        present.add(a);
        present.add(b);
    }
    const parent = new Map();
    for (const n of present) parent.set(n, n);
    function find(x) {
        let p = parent.get(x);
        if (p === x) return x;
        const r = find(p);
        parent.set(x, r);
        return r;
    }
    for (const [a, b] of pairs) {
        if (!present.has(a) || !present.has(b)) continue;
        const ra = find(a), rb = find(b);
        if (ra !== rb) parent.set(rb, ra);
    }
    const components = new Map();
    for (const n of present) {
        const r = find(n);
        if (!components.has(r)) components.set(r, []);
        components.get(r).push(n);
    }
    const alias = new Map();
    let groups = 0;
    for (const members of components.values()) {
        if (members.length < 2) continue;
        members.sort();
        const canon = members[0];
        groups++;
        for (let i = 1; i < members.length; i++) alias.set(members[i], canon);
    }
    return { alias, groups };
}

function passMerge(state, report, telemetry) {
    const cosmetic = collectCosmeticNames(report);
    const pairs = mergeabilityCandidatePairs(report);
    const { alias, groups } = buildMergeAliasMap(pairs, cosmetic, state);
    telemetry.merged_object_groups = groups;
    if (alias.size === 0) return;

    telemetry.merged_object_aliases += alias.size;
    telemetry.removed_collision_layers += applyRenameMap(state, alias);
    rebuildAfterStructuralEdit(state, alias);
}

// ---------- Pass orchestration ----------

function createSolverOptimizationHook(staticAnalysisReport, passes) {
    const p = passes || {};
    return (compiledState) => {
        const telemetry = defaultTelemetry();
        if (!compiledState || compiledState.invalid > 0) {
            if (compiledState) compiledState.solverOptimizationTelemetry = telemetry;
            return;
        }
        if (p.inert) {
            const t0 = nowFn();
            const lines = inertCommandOnlyRuleSourceLines(staticAnalysisReport);
            if (lines.size > 0) {
                telemetry.removed_inert_rules += dropSolverInertCommandOnlyRules(compiledState, lines);
            }
            telemetry.ms_inert += nowFn() - t0;
        }
        if (p.cosmetic) {
            const t0 = nowFn();
            passCosmeticPrune(compiledState, staticAnalysisReport, telemetry);
            telemetry.ms_cosmetic += nowFn() - t0;
        }
        if (p.merge) {
            const t0 = nowFn();
            passMerge(compiledState, staticAnalysisReport, telemetry);
            telemetry.ms_merge += nowFn() - t0;
        }
        compiledState.solverOptimizationTelemetry = telemetry;
    };
}

// ---------- CLI / pass-list parsing ----------

function parseSolverOptPassList(arg) {
    const passes = { inert: false, cosmetic: false, merge: false };
    const parts = String(arg || '').split(',').map(s => s.trim().toLowerCase()).filter(Boolean);
    if (parts.includes('all')) {
        passes.inert = passes.cosmetic = passes.merge = true;
        return passes;
    }
    for (const p of parts) {
        if (!(p in passes)) throw new Error(`Unknown solver optimization pass: ${p}`);
        passes[p] = true;
    }
    return passes;
}

function resolveSolverPasses(options) {
    if (options.solverOptParityBaseline) {
        return { inert: false, cosmetic: false, merge: false };
    }
    const passes = { inert: false, cosmetic: false, merge: false };
    if (options.solverOptPasses) Object.assign(passes, options.solverOptPasses);
    if (options.solverOptimizeStatic) passes.inert = true;
    return passes;
}

function solverPassesNeedFullStaticReport(passes) {
    return !!(passes && (passes.cosmetic || passes.merge));
}

function effectiveSolverPassesForHook(staticAnalysisReport, passes) {
    const p = passes || { inert: false, cosmetic: false, merge: false };
    const reportOk = !!(staticAnalysisReport && staticAnalysisReport.status === 'ok');
    return {
        inert: !!p.inert,
        cosmetic: !!p.cosmetic && reportOk,
        merge: !!p.merge && reportOk,
    };
}

// ---------- Telemetry formatters ----------
// These read the runner's renamed totals (run_solver_tests_js.js maps
// telemetry.removed_inert_rules → totals.static_optimization_removed_rules,
// telemetry.ms_inert → totals.solver_opt_ms_inert, etc).

const FORMATTER_FIELDS = [
    ['static_optimization_removed_rules', 'inert_rules'],
    ['removed_cosmetic_objects',          'cosmetic_objs'],
    ['removed_collision_layers',          'empty_layers'],
    ['merged_object_aliases',             'merge_aliases'],
    ['merged_object_groups',              'merge_groups'],
];
const FORMATTER_MS_FIELDS = [
    ['solver_opt_ms_inert',    'opt_ms_inert'],
    ['solver_opt_ms_cosmetic', 'opt_ms_cosmetic'],
    ['solver_opt_ms_merge',    'opt_ms_merge'],
];

function totalHookMs(t) {
    return (t.solver_opt_ms_inert || 0) + (t.solver_opt_ms_cosmetic || 0) + (t.solver_opt_ms_merge || 0);
}

function formatSolverOptimizationHumanSuffixFromTotals(t) {
    if (!t) return '';
    const parts = [];
    for (const [key, label] of FORMATTER_FIELDS) {
        if ((t[key] || 0) > 0) parts.push(`${label}=${t[key]}`);
    }
    for (const [key, label] of FORMATTER_MS_FIELDS) {
        if ((t[key] || 0) > 0) parts.push(`${label}=${t[key].toFixed(3)}`);
    }
    const hookMs = totalHookMs(t);
    if (hookMs > 0) parts.push(`opt_hook_ms=${hookMs.toFixed(3)}`);
    if (t.solver_optimization_gated) parts.push('opt_gated=1');
    return parts.length > 0 ? `solver_optimization: ${parts.join(' ')}` : '';
}

function buildSolverOptimizationJsonTotals(t) {
    if (!t) return null;
    const removedInert = t.static_optimization_removed_rules || 0;
    const removedCos = t.removed_cosmetic_objects || 0;
    const removedLayers = t.removed_collision_layers || 0;
    const mergedAliases = t.merged_object_aliases || 0;
    const mergedGroups = t.merged_object_groups || 0;
    const msInert = t.solver_opt_ms_inert || 0;
    const msCos = t.solver_opt_ms_cosmetic || 0;
    const msMrg = t.solver_opt_ms_merge || 0;
    const hookMs = msInert + msCos + msMrg;
    const gated = !!t.solver_optimization_gated;
    if (removedInert + removedCos + removedLayers + mergedAliases + mergedGroups + hookMs === 0 && !gated) {
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
    if (gated) out.gated = true;
    return out;
}

module.exports = {
    INERT_COMMAND_NAMES,
    defaultTelemetry,
    inertCommandOnlyRuleSourceLines,
    isInertCommandOnlyCompiledRule,
    dropSolverInertCommandOnlyRules,
    applyNameSubstitutionToWinconditions: (state, renameFn) => {
        // Backwards-compatible shim for legacy callers (test runner).
        const rename = new Map();
        for (const wc of state.winconditions || []) {
            for (let i = 0; i < wc.length - 1; i++) {
                if (typeof wc[i] !== 'string') continue;
                const next = renameFn(wc[i]);
                if (next === null || next === undefined || next === wc[i]) continue;
                rename.set(wc[i], next);
            }
        }
        rewriteWinconditionsByRename(state, rename);
    },
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
