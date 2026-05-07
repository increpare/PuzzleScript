#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const { analyzeFile, discoverInputFiles } = require('./ps_static_analysis');

const DEFAULT_OUT = 'build/static-analysis-explorer/index.html';
const PARTITION_CLASSES = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'];
const MAX_RULEGROUPS_PER_GAME = 80;
const MAX_RULES_PER_GROUP = 120;
const MAX_RULE_TEXT = 260;
const MAX_INTERACTION_EDGES = 80;
const MAX_RERUN_MASKS = 80;
const MAX_RERUN_MASK_ENTRIES = 80;
const MAX_COMPONENT_RULE_IDS = 600;

function usage(exitCode = 1) {
    const text = [
        'Usage: node src/tests/build_static_analysis_explorer.js <file-or-dir> [more paths]',
        '  [--out PATH] [--game SUBSTRING]',
    ].join('\n');
    (exitCode === 0 ? process.stdout : process.stderr).write(`${text}\n`);
    process.exit(exitCode);
}

function parseArgs(argv) {
    const inputs = [];
    const options = {
        outPath: path.resolve(DEFAULT_OUT),
        gameFilter: null,
        repoRoot: path.resolve(__dirname, '..', '..'),
    };
    const args = argv.slice(2);
    if (args.length === 0 || args.includes('--help') || args.includes('-h')) usage(args.length === 0 ? 1 : 0);
    for (let index = 0; index < args.length; index++) {
        const arg = args[index];
        if (arg === '--out' && index + 1 < args.length) {
            options.outPath = path.resolve(args[++index]);
        } else if (arg === '--game' && index + 1 < args.length) {
            options.gameFilter = args[++index];
        } else if (arg.startsWith('--')) {
            throw new Error(`Unsupported argument: ${arg}`);
        } else {
            inputs.push(arg);
        }
    }
    if (inputs.length === 0) usage(1);
    return { inputs, options };
}

function relativeSourcePath(sourcePath, repoRoot) {
    const absolute = path.resolve(sourcePath);
    return path.relative(repoRoot, absolute).split(path.sep).join('/');
}

function editorHrefForSource(sourcePath, options = {}) {
    const repoRoot = options.repoRoot || path.resolve(__dirname, '..', '..');
    const srcRoot = path.join(repoRoot, 'src');
    const relToSrc = path.relative(srcRoot, path.resolve(sourcePath)).split(path.sep).join('/');
    return `/src/editor.html?file=${encodeURIComponent(relToSrc)}`;
}

function facts(report, family) {
    return (report.facts && report.facts[family]) || [];
}

function truncateText(value, maxLength = MAX_RULE_TEXT) {
    const text = String(value);
    if (text.length <= maxLength) return text;
    return `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

function allRules(report) {
    const sections = report.ps_tagged ? report.ps_tagged.rule_sections || [] : [];
    return sections.flatMap(section =>
        section.groups.flatMap(group =>
            group.rules.map(rule => ({ section, group, rule }))
        )
    );
}

function termName(term) {
    const ref = term.ref || {};
    if (ref.type === 'object_set') return `{${truncateText(ref.source || `${(ref.objects || []).length} objects`, 80)}}`;
    if (ref.type === 'ellipsis') return '...';
    return ref.name || ref.canonical_name || '?';
}

function termText(term) {
    const parts = [];
    if (term.kind === 'absent') parts.push('no');
    if (term.kind === 'random_object') parts.push('random');
    if (term.movement) parts.push(term.movement);
    parts.push(termName(term));
    return parts.join(' ');
}

function sideText(side) {
    if (!Array.isArray(side) || side.length === 0) return '';
    return side.map(row => `[ ${row.map(cell => cell.map(termText).join(' ')).join(' | ')} ]`).join(' ');
}

function ruleText(rule) {
    const prefix = [];
    if (rule.late) prefix.push('late');
    if (rule.rigid) prefix.push('rigid');
    if (rule.random_rule) prefix.push('random');
    if (rule.direction && rule.direction !== '0') prefix.push(rule.direction);
    const replacement = rule.rhs && rule.rhs.length > 0 ? ` -> ${sideText(rule.rhs)}` : '';
    const commands = (rule.commands || []).map(command => command[0]).join(' ');
    return `${prefix.join(' ')} ${sideText(rule.lhs)}${replacement}${commands ? ` ${commands}` : ''}`.trim();
}

function flowForGroup(report, groupId) {
    return facts(report, 'rulegroup_flow').find(item => item.subjects.groups[0] === groupId) || null;
}

function mergeableGroupsFromPairs(pairs) {
    const parent = new Map();

    function find(name) {
        if (!parent.has(name)) parent.set(name, name);
        const current = parent.get(name);
        if (current === name) return name;
        const root = find(current);
        parent.set(name, root);
        return root;
    }

    function union(left, right) {
        const leftRoot = find(left);
        const rightRoot = find(right);
        if (leftRoot === rightRoot) return;
        const [first, second] = [leftRoot, rightRoot].sort();
        parent.set(second, first);
    }

    for (const pair of pairs) {
        const objects = pair.objects || [];
        if (objects.length < 2) continue;
        for (const object of objects) find(object);
        union(objects[0], objects[1]);
    }

    const groupsByRoot = new Map();
    for (const object of parent.keys()) {
        const root = find(object);
        if (!groupsByRoot.has(root)) groupsByRoot.set(root, []);
        groupsByRoot.get(root).push(object);
    }

    return Array.from(groupsByRoot.values())
        .map(objects => objects.sort())
        .filter(objects => objects.length > 1)
        .sort((left, right) => left[0].localeCompare(right[0]) || left.length - right.length)
        .map((objects, index) => ({ id: `merge_group_${index}`, objects, label: objects.join(', ') }));
}

function componentIndexByRule(flow) {
    const index = new Map();
    if (!flow || !flow.value || !Array.isArray(flow.value.components)) return index;
    flow.value.components.forEach((component, componentIndex) => {
        for (const ruleId of component) index.set(ruleId, componentIndex);
    });
    return index;
}

function hasNonEmptyRerunMask(flow) {
    return Object.values((flow && flow.value && flow.value.rerun_masks) || {}).some(mask => mask.length > 0);
}

function compactComponents(components) {
    const ruleIdCount = components.reduce((sum, component) => sum + component.length, 0);
    if (ruleIdCount > MAX_COMPONENT_RULE_IDS) return [];
    return components;
}

function compactRerunMasks(rerunMasks) {
    const entries = Object.entries(rerunMasks || {})
        .filter(([, mask]) => Array.isArray(mask) && mask.length > 0)
        .slice(0, MAX_RERUN_MASKS)
        .map(([ruleId, mask]) => [ruleId, mask.slice(0, MAX_RERUN_MASK_ENTRIES)]);
    return Object.fromEntries(entries);
}

function sortRulegroupInterest(left, right) {
    return Number(right.split_candidate) - Number(left.split_candidate)
        || right.component_count - left.component_count
        || right.rules_total - left.rules_total
        || right.interaction_edge_count - left.interaction_edge_count
        || left.id.localeCompare(right.id);
}

function summarizeRuleGroups(report) {
    if (!report.ps_tagged) return { groups: [], total: 0, splitTotal: 0, omitted: 0 };
    const results = [];
    for (const section of report.ps_tagged.rule_sections || []) {
        for (const group of section.groups || []) {
            const flow = flowForGroup(report, group.id);
            if (!flow || (flow.status !== 'candidate' && !hasNonEmptyRerunMask(flow))) continue;
            const partition = componentIndexByRule(flow);
            const components = flow.value.components || [];
            const interactionEdges = flow.value.interaction_edges || [];
            const rerunMasks = flow.value.rerun_masks || {};
            const nonEmptyRerunMaskCount = Object.values(rerunMasks)
                .filter(mask => Array.isArray(mask) && mask.length > 0)
                .length;
            const rules = group.rules.slice(0, MAX_RULES_PER_GROUP).map(rule => ({
                id: rule.id,
                text: truncateText(ruleText(rule)),
                component: partition.has(rule.id) ? partition.get(rule.id) : null,
                tags: rule.tags,
            }));
            results.push({
                id: group.id,
                section: section.name,
                status: flow.status,
                split_candidate: Boolean(flow.value.split_candidate),
                component_count: components.length,
                components: compactComponents(components),
                interaction_edge_count: interactionEdges.length,
                interaction_edges: interactionEdges.slice(0, MAX_INTERACTION_EDGES),
                rerun_mask_count: nonEmptyRerunMaskCount,
                rerun_masks: compactRerunMasks(rerunMasks),
                rules_total: group.rules.length,
                rules_omitted: Math.max(0, group.rules.length - rules.length),
                rules,
            });
        }
    }
    results.sort(sortRulegroupInterest);
    const splitTotal = results.filter(group => group.split_candidate).length;
    return {
        groups: results.slice(0, MAX_RULEGROUPS_PER_GAME),
        total: results.length,
        splitTotal,
        omitted: Math.max(0, results.length - MAX_RULEGROUPS_PER_GAME),
    };
}

function summarizeReport(report, options = {}) {
    const repoRoot = options.repoRoot || path.resolve(__dirname, '..', '..');
    const sourcePath = report.source && report.source.path ? report.source.path : '<memory>';
    const objects = report.ps_tagged ? report.ps_tagged.objects || [] : [];
    const layers = report.ps_tagged ? report.ps_tagged.collision_layers || [] : [];
    const rules = allRules(report);
    const mergeable = facts(report, 'mergeability')
        .filter(item => item.status === 'candidate')
        .map(item => ({ id: item.id, objects: item.subjects.objects || [] }));
    const mergeableGroups = mergeableGroupsFromPairs(mergeable);
    const transient = facts(report, 'transient_boundary')
        .filter(item => item.status === 'proved')
        .map(item => item.subjects.objects[0]);
    const staticObjects = objects.filter(object => object.tags && object.tags.static).map(object => object.name);
    const neverInitialOrCreated = objects
        .filter(object => object.tags && object.tags.present_in_no_levels && !object.tags.may_be_created)
        .map(object => object.name);
    const staticLayers = layers
        .filter(layer => layer.tags && layer.tags.static)
        .map(layer => ({ id: layer.id, objects: layer.objects || [] }));
    const inertLayers = layers
        .filter(layer => layer.tags && layer.tags.inert)
        .map(layer => ({ id: layer.id, objects: layer.objects || [] }));
    const cosmeticObjects = objects
        .filter(object => object.tags && object.tags.cosmetic)
        .map(object => object.name);
    const inertRules = rules
        .filter(entry => entry.rule.tags && entry.rule.tags.inert_command_only)
        .map(entry => ({
            id: entry.rule.id,
            group: entry.group.id,
            source_line: entry.rule.source_line,
            text: ruleText(entry.rule),
        }));
    const commandOnlyRules = rules
        .filter(entry => entry.rule.tags && entry.rule.tags.command_only && !entry.rule.tags.inert_command_only)
        .map(entry => ({
            id: entry.rule.id,
            group: entry.group.id,
            source_line: entry.rule.source_line,
            text: ruleText(entry.rule),
        }));
    const rulegroupFlowSummary = summarizeRuleGroups(report);
    const rulegroupFlow = rulegroupFlowSummary.groups;
    const score = mergeable.length * 4
        + rulegroupFlowSummary.splitTotal * 5
        + inertRules.length * 2
        + staticObjects.length
        + staticLayers.length
        + transient.length * 2
        + neverInitialOrCreated.length
        + cosmeticObjects.length;
    return {
        source_path: relativeSourcePath(sourcePath, repoRoot),
        display_name: path.basename(sourcePath),
        title: report.ps_tagged && report.ps_tagged.game ? report.ps_tagged.game.title || '' : '',
        status: report.status,
        editor_href: editorHrefForSource(sourcePath, { repoRoot }),
        score,
        mergeable,
        mergeable_groups: mergeableGroups,
        static_objects: staticObjects,
        static_objects_label: staticObjects.join(', '),
        never_initial_or_created: neverInitialOrCreated,
        static_layers: staticLayers,
        inert_layers: inertLayers,
        cosmetic_objects: cosmeticObjects,
        transient_objects: transient,
        inert_rules: inertRules,
        command_only_rules: commandOnlyRules,
        rulegroup_flow: rulegroupFlow,
        rulegroup_flow_total: rulegroupFlowSummary.total,
        rulegroup_flow_split_total: rulegroupFlowSummary.splitTotal,
        rulegroup_flow_omitted: rulegroupFlowSummary.omitted,
        summary: report.summary || {},
    };
}

function buildExplorerModel(reports, options = {}) {
    const games = reports
        .filter(report => report.status === 'ok')
        .map(report => summarizeReport(report, options))
        .sort((left, right) => right.score - left.score || left.display_name.localeCompare(right.display_name));
    return {
        schema: 'ps-static-analysis-explorer-v1',
        generated_at: new Date().toISOString(),
        games,
        totals: {
            games: games.length,
            mergeable: games.reduce((sum, game) => sum + game.mergeable.length, 0),
            merge_groups: games.reduce((sum, game) => sum + game.mergeable_groups.length, 0),
            split_groups: games.reduce((sum, game) => sum + game.rulegroup_flow_split_total, 0),
            inert_rules: games.reduce((sum, game) => sum + game.inert_rules.length, 0),
            static_objects: games.reduce((sum, game) => sum + game.static_objects.length, 0),
            static_layers: games.reduce((sum, game) => sum + game.static_layers.length, 0),
            inert_layers: games.reduce((sum, game) => sum + game.inert_layers.length, 0),
            cosmetic_objects: games.reduce((sum, game) => sum + game.cosmetic_objects.length, 0),
            transient_objects: games.reduce((sum, game) => sum + game.transient_objects.length, 0),
        },
    };
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function safeJsonForScript(value) {
    return JSON.stringify(value).replace(/</g, '\\u003c');
}

function renderExplorerHtml(model) {
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PuzzleScript Static Analysis Explorer</title>
<style>
:root { color-scheme: dark; --bg: #101318; --panel: #191e25; --line: #303945; --text: #e8edf3; --muted: #9aa7b5; --accent: #74b9ff; --ok: #81d38a; --warn: #f3c969; --bad: #ff8f8f; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }
header { position: sticky; top: 0; z-index: 2; padding: 14px 18px; border-bottom: 1px solid var(--line); background: rgba(16,19,24,.96); }
h1 { margin: 0 0 10px; font-size: 20px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
input, select { background: #0c0f13; border: 1px solid var(--line); color: var(--text); padding: 7px 9px; border-radius: 6px; }
main { display: grid; grid-template-columns: minmax(260px, 360px) 1fr; min-height: calc(100vh - 82px); }
#gameList { border-right: 1px solid var(--line); overflow: auto; max-height: calc(100vh - 82px); }
.game-row { display: block; width: 100%; border: 0; border-bottom: 1px solid var(--line); background: transparent; color: var(--text); padding: 10px 12px; text-align: left; cursor: pointer; }
.game-row:hover, .game-row.active { background: #202733; }
.game-title { font-weight: 650; }
.path { color: var(--muted); font-size: 12px; word-break: break-all; }
.badges { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 6px; }
.badge { border: 1px solid var(--line); border-radius: 999px; padding: 1px 7px; font-size: 12px; color: var(--muted); }
.badge.hot { color: #07110a; background: var(--ok); border-color: var(--ok); }
#detail { padding: 18px; overflow: auto; max-height: calc(100vh - 82px); }
.section { margin: 0 0 18px; padding: 14px; border: 1px solid var(--line); background: var(--panel); border-radius: 8px; }
details.section { padding: 0; }
.section h2 { margin: 0 0 10px; font-size: 16px; }
details.section summary { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 12px 14px; cursor: pointer; list-style: none; }
details.section summary::-webkit-details-marker { display: none; }
details.section summary::after { content: "show"; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
details.section[open] summary::after { content: "hide"; }
details.section summary h2 { margin: 0; }
.section-body { padding: 0 14px 14px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { border: 1px solid var(--line); border-radius: 6px; padding: 4px 7px; background: #11161d; }
a { color: var(--accent); }
.rule-group { margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.rule-group-head { display: flex; justify-content: space-between; gap: 8px; padding: 8px 10px; background: #11161d; border-bottom: 1px solid var(--line); }
.rule { display: grid; grid-template-columns: 86px 1fr; gap: 8px; padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,.06); }
.rule:last-child { border-bottom: 0; }
.rule code { white-space: pre-wrap; word-break: break-word; }
.p0 { border-left: 5px solid #74b9ff; } .p1 { border-left: 5px solid #81d38a; } .p2 { border-left: 5px solid #f3c969; } .p3 { border-left: 5px solid #ff8f8f; }
.p4 { border-left: 5px solid #c7a6ff; } .p5 { border-left: 5px solid #7ee7d1; } .p6 { border-left: 5px solid #ffb77a; } .p7 { border-left: 5px solid #d6e17a; }
.empty { color: var(--muted); }
@media (max-width: 850px) { main { grid-template-columns: 1fr; } #gameList, #detail { max-height: none; } #gameList { border-right: 0; } }
</style>
</head>
<body>
<header>
<h1>PuzzleScript Static Analysis Explorer</h1>
<div class="controls">
<input id="search" placeholder="Filter games or traits" autofocus>
<select id="sort"><option value="score">Most interesting</option><option value="name">Name</option><option value="split">Split groups</option><option value="merge">Mergeable</option></select>
<span id="totals"></span>
</div>
</header>
<main>
<nav id="gameList"></nav>
<section id="detail"></section>
</main>
<script id="explorer-data" type="application/json">${safeJsonForScript(model)}</script>
<script>
const model = JSON.parse(document.getElementById('explorer-data').textContent);
let games = model.games.slice();
let selected = games[0] || null;
const list = document.getElementById('gameList');
const detail = document.getElementById('detail');
const search = document.getElementById('search');
const sort = document.getElementById('sort');
document.getElementById('totals').textContent = model.totals.games + ' games | ' + model.totals.split_groups + ' split groups | ' + model.totals.merge_groups + ' merge groups | ' + model.totals.inert_layers + ' inert layers | ' + model.totals.cosmetic_objects + ' cosmetic objs';
function escapeText(value) {
  return String(value).replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
}
function countBadges(game) {
  return [
    ['merge group', game.mergeable_groups.length],
    ['static obj', game.static_objects.length],
    ['static layer', game.static_layers.length],
    ['inert layer', game.inert_layers.length],
    ['cosmetic', game.cosmetic_objects.length],
    ['transient', game.transient_objects.length],
    ['inert rule', game.inert_rules.length],
    ['split', game.rulegroup_flow_split_total],
  ].filter(item => item[1] > 0);
}
function searchable(game) {
  return [
    game.display_name, game.source_path, game.title,
    game.mergeable_groups.flatMap(group => group.objects).join(' '),
    game.static_objects.join(' '),
    game.static_layers.flatMap(layer => layer.objects).join(' '),
    game.inert_layers.flatMap(layer => layer.objects).join(' '),
    game.cosmetic_objects.join(' '),
    game.transient_objects.join(' '),
    game.rulegroup_flow.map(group => group.id).join(' '),
  ].join(' ').toLowerCase();
}
function visibleGames() {
  const q = search.value.trim().toLowerCase();
  let out = q ? games.filter(game => searchable(game).includes(q)) : games.slice();
  const mode = sort.value;
  out.sort((a, b) => {
    if (mode === 'name') return a.display_name.localeCompare(b.display_name);
    if (mode === 'split') return b.rulegroup_flow_split_total - a.rulegroup_flow_split_total || b.score - a.score;
    if (mode === 'merge') return b.mergeable_groups.length - a.mergeable_groups.length || b.score - a.score;
    return b.score - a.score || a.display_name.localeCompare(b.display_name);
  });
  return out;
}
function renderList() {
  const shown = visibleGames();
  if (!shown.includes(selected)) selected = shown[0] || null;
  list.innerHTML = shown.map(game => {
    const badges = countBadges(game).map(([label, count]) => '<span class="badge ' + (label === 'split' || label === 'merge group' ? 'hot' : '') + '">' + label + ' ' + count + '</span>').join('');
    return '<button class="game-row ' + (game === selected ? 'active' : '') + '" data-path="' + escapeText(game.source_path) + '"><div class="game-title">' + escapeText(game.display_name) + '</div><div class="path">' + escapeText(game.source_path) + '</div><div class="badges">' + badges + '</div></button>';
  }).join('');
  for (const button of list.querySelectorAll('.game-row')) {
    button.addEventListener('click', () => {
      selected = games.find(game => game.source_path === button.dataset.path);
      render();
    });
  }
}
function chipList(values) {
  return values.length ? '<div class="chips">' + values.map(value => '<span class="chip">' + escapeText(value) + '</span>').join('') + '</div>' : '<div class="empty">none</div>';
}
function analysisSection(title, content, open = true) {
  return '<details class="section" ' + (open ? 'open' : '') + '><summary><h2>' + escapeText(title) + '</h2></summary><div class="section-body">' + content + '</div></details>';
}
function renderRuleGroups(groups) {
  if (!groups.length) return '<div class="empty">none</div>';
  return groups.map(group => {
    const rules = group.rules.map(rule => {
      const cls = rule.component === null ? '' : '${PARTITION_CLASSES[0]}'.replace('p0', 'p' + (rule.component % ${PARTITION_CLASSES.length}));
      return '<div class="rule ' + cls + '"><span>' + escapeText(rule.id.replace(/^.*_rule_/, 'rule ')) + '</span><code>' + escapeText(rule.text) + '</code></div>';
    }).join('');
    const omitted = group.rules_omitted ? '<div class="rule empty"><span></span><code>' + group.rules_omitted + ' more rules omitted from this view</code></div>' : '';
    const meta = group.component_count + ' components; ' + group.interaction_edge_count + ' edges; ' + group.rerun_mask_count + ' rerun masks';
    return '<div class="rule-group"><div class="rule-group-head"><strong>' + escapeText(group.id) + ' · ' + escapeText(group.status) + '</strong><span>' + escapeText(meta) + '</span></div>' + rules + omitted + '</div>';
  }).join('');
}
function renderDetail() {
  const game = selected;
  if (!game) {
    detail.innerHTML = '<div class="section empty">No games match.</div>';
    return;
  }
  detail.innerHTML =
    '<div class="section"><h2>' + escapeText(game.display_name) + '</h2><div class="path">' + escapeText(game.source_path) + '</div><p><a target="_blank" href="' + escapeText(game.editor_href) + '">Open in editor</a></p></div>' +
    analysisSection('Mergeable Objects', chipList(game.mergeable_groups.map(group => group.label))) +
    analysisSection('Static Objects', game.static_objects_label ? chipList([game.static_objects_label]) : chipList([])) +
    analysisSection('Never Initial Or Created', chipList(game.never_initial_or_created), false) +
    analysisSection('Static Collision Layers', chipList(game.static_layers.map(layer => 'layer ' + layer.id + ': ' + layer.objects.join(', ')))) +
    analysisSection('Inert Collision Layers', '<p class="path">No object on these layers appears in any rule (LHS/RHS), win condition, or the Player aggregate.</p>' + chipList(game.inert_layers.map(layer => 'layer ' + layer.id + ': ' + layer.objects.join(', ')))) +
    analysisSection('Likely cosmetic objects', '<p class="path">Outside the static core closure: Player entities, wincondition objects, <code>win</code>-command LHS reads, plus objects reached by read→write rule edges or rules whose RHS write mask hits a layer that already holds a core object (see <code>docs/superpowers/specs/2026-05-03-cosmetic-closure-static-analysis-design.md</code>).</p>' + chipList(game.cosmetic_objects)) +
    analysisSection('Transient Objects', chipList(game.transient_objects)) +
    analysisSection('Solver-Discardable Rules', chipList(game.inert_rules.map(rule => rule.group + ': ' + rule.text)), false) +
    analysisSection('Semantic Command-Only Rules', chipList(game.command_only_rules.map(rule => rule.group + ': ' + rule.text)), false) +
    analysisSection('rulegroup_flow Split Candidates / Rerun Masks', '<p class="path">' + game.rulegroup_flow_total + ' interesting groups; ' + game.rulegroup_flow_split_total + ' split candidates' + (game.rulegroup_flow_omitted ? '; showing first ' + game.rulegroup_flow.length + ', omitted ' + game.rulegroup_flow_omitted : '') + '</p>' + renderRuleGroups(game.rulegroup_flow), false);
}
function render() { renderList(); renderDetail(); }
search.addEventListener('input', render);
sort.addEventListener('change', render);
render();
</script>
</body>
</html>
`;
}

function buildReports(inputs, options) {
    return discoverInputFiles(inputs)
        .filter(filePath => !options.gameFilter || filePath.toLowerCase().includes(options.gameFilter.toLowerCase()))
        .map(filePath => analyzeFile(filePath, { includePsTagged: true }));
}

function main() {
    const { inputs, options } = parseArgs(process.argv);
    const reports = buildReports(inputs, options);
    const model = buildExplorerModel(reports, options);
    const html = renderExplorerHtml(model);
    fs.mkdirSync(path.dirname(options.outPath), { recursive: true });
    fs.writeFileSync(options.outPath, html, 'utf8');
    process.stdout.write(`static_analysis_explorer wrote ${options.outPath} games=${model.games.length} split_groups=${model.totals.split_groups} merge_groups=${model.totals.merge_groups} merge_pairs=${model.totals.mergeable}\n`);
}

if (require.main === module) {
    main();
}

module.exports = {
    buildExplorerModel,
    editorHrefForSource,
    renderExplorerHtml,
    summarizeReport,
};
