'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const context = vm.createContext({ console, consolePrint() {} });
function load(file) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '..', file), 'utf8'), context, { filename: file });
}
for (const file of ['js/languageConstants.js', 'js/colors.js', 'js/codemirror/stringstream.js', 'js/parser.js', 'js/compiler.js', 'js/codemirror/rule-transform.js']) load(file);
vm.runInContext(`
    CodeMirror.Pos = (line, ch) => ({ line, ch });
    CodeMirror.registerHelper = (type, name, fn) => { CodeMirror.hint = fn; };
    globalThis.mode = codeMirrorFn();
`, context);
for (const file of ['js/codemirror/rule-completion.js', 'js/codemirror/rule-analogies.js']) {
    load(file);
}
load('js/codemirror/anyword-hint.js');

const objects = ['Background', 'Player', 'Crate', 'Target', 'RedKey', 'RedDoor', 'BlueKey', 'BlueDoor',
    'OpenDoor', 'OpenGate', 'ClosedDoor', 'ClosedGate', 'WoodBox', 'WoodWall', 'MetalBox', 'MetalWall',
    'Stage1', 'Stage2', 'Stage3', 'Crate_up', 'Crate_down'];
const source = 'objects\n\n' + objects.map(name => name + '\nred\n').join('\n') + `
legend
Movable = Player or Crate
Mixed = Crate or Target
Alias = Movable
Hero = Player
Pair = Player and Target
sounds
collisionlayers
Background
Player, Crate
Target
RedKey, BlueKey
RedDoor, BlueDoor
OpenDoor, ClosedDoor
OpenGate, ClosedGate
WoodBox, MetalBox
WoodWall, MetalWall
Stage1, Stage2, Stage3
Crate_up, Crate_down
`;

function hints(line, { previous = '', section = 'rules', ch = line.length, setup = source } = {}) {
    const lines = (setup + section + '\n' + (previous ? previous + '\n' : '') + line).split('\n');
    const state = context.mode.startState();
    let token = { string: '', state };
    lines.forEach((text, index) => {
        if (!text) { context.mode.blankLine(state); return; }
        const stream = new context.CodeMirror.StringStream(text);
        while (!stream.eol()) {
            stream.start = stream.pos;
            context.mode.token(stream, state);
            token = { string: text.slice(stream.start, stream.pos), state };
            if (index === lines.length - 1 && stream.pos >= ch) break;
        }
    });
    const cursor = { line: lines.length - 1, ch };
    const editor = {
        getCursor: () => cursor,
        getLine: i => lines[i] || '',
        getTokenAt: () => token
    };
    return context.CodeMirror.hint(editor, {});
}
const words = (line, options) => hints(line, options).list.map(item => item.text);
const has = (line, name, options) => assert.ok(words(line, options).some(word => word.toLowerCase() === name.toLowerCase()), `${line} should offer ${name}`);
const lacks = (line, name, options) => assert.ok(!words(line, options).some(word => word.toLowerCase() === name.toLowerCase()), `${line} should not offer ${name}`);

test('late rules only offer permitted pattern modifiers', () => {
    lacks('late [ m', 'moving');
    lacks('late [ s', 'stationary');
    has('late [ n', 'no');
    has('late [ Player ] -> [ r', 'random');
    has('late [ Player ] -> [ r', 'randomdir');
    has('[ m', 'moving');
});
test('win conditions suggest the token required at this position', () => {
    lacks('all Crate s', 'some', { section: 'winconditions' });
    has('all Crate o', 'on', { section: 'winconditions' });
    lacks('all S', 'some', { section: 'winconditions' });
    has('all C', 'Crate', { section: 'winconditions' });
    has('a', 'any', { section: 'winconditions' });
});
test('known same-cell layer conflicts are excluded, including aliases and aggregates', () => {
    lacks('[ Player C', 'Crate');
    lacks('[ Hero C', 'Crate');
    lacks('[ Pair C', 'Crate');
    has('[ Player T', 'Target');
    has('[ Player | C', 'Crate');
    has('[ Player ] -> [ C', 'Crate');
});
test('uncertain properties and negated names remain available', () => {
    has('[ Player M', 'Mixed');
    has('[ no Player C', 'Crate');
    has('[ Player no C', 'Crate');
    has('[ Unknown C', 'Crate');
});
test('right-side properties must have a corresponding or unique positive left match', () => {
    has('[ Movable ] -> [ M', 'Movable');
    has('[ Movable | ] -> [ | M', 'Movable');
    has('[ Movable | Movable ] -> [ | M', 'Movable');
    lacks('[ Movable | Movable | ] -> [ | | M', 'Movable');
    lacks('[ Player ] -> [ M', 'Movable');
    lacks('[ no Movable ] -> [ M', 'Movable');
    has('[ Player ] -> [ no M', 'Movable');
    has('[ Player ] -> [ random M', 'Movable');
});
test('property aliases have distinct bindings, with explanations for matching names', () => {
    lacks('[ Movable ] -> [ A', 'Alias');
    lacks('[ Alias ] -> [ M', 'Movable');
    const item = hints('[ Alias ] -> [ A').list.find(item => item.text.toLowerCase() === 'alias');
    assert.ok(item);
    assert.match(item.extra, /same object matched on the left/i);
    assert.match(item.extra, /cell/i);
    const moved = hints('[ Alias | Movable | ] -> [ | | M').list.find(item => item.text.toLowerCase() === 'movable');
    assert.match(moved.extra, /same object matched on the left/i);
});
test('malformed left sides do not prove properties unbound', () => {
    has('[ Unknown ] -> [ M', 'Movable');
    has('[ Movable -> [ M', 'Movable');
});
test('normal directional rule completion still works', () => {
    assert.ok(words('d', { previous: 'up [ Crate_up ] -> [ > Crate_up ]' })
        .some(text => text === 'down [ Crate_down ] -> [ > Crate_down ]'));
});

test('directional rule completion has no competing suffix-only repeat', () => {
    const items = hints('d', { previous: 'up [ Crate_up ] -> [ > Crate_up ]' }).list;
    assert.ok(items.some(item => item.text === 'down [ Crate_down ] -> [ > Crate_down ]'));
    assert.ok(!items.some(item => item.displayText === 'Repeat for down'));
    assert.equal(analogy('down', '[ Crate_up ] -> [ Crate_up ]')[0]?.text,
        '[ Crate_down ] -> [ Crate_down ]');
});

function analogy(line, previous, options = {}) {
    return hints(line, { previous, ...options }).list.filter(item => item.displayText && item.displayText.startsWith('Repeat for '));
}
test('repeat a rule for a defined colour family with exact preview', () => {
    const items = analogy('Blu', '[ RedKey | RedDoor ] -> [ | ]');
    assert.equal(items.length, 1);
    assert.equal(items[0].displayText, 'Repeat for Blue');
    assert.equal(items[0].text, '[ BlueKey | BlueDoor ] -> [ | ]');
    assert.match(items[0].extra, /RedKey → BlueKey/);
    assert.match(items[0].extra, /RedDoor → BlueDoor/);
});
test('infer states, materials, and numbered progressions', () => {
    assert.equal(analogy('Closed', '[ OpenDoor | OpenGate ] -> [ | ]')[0]?.text,
        '[ ClosedDoor | ClosedGate ] -> [ | ]');
    assert.equal(analogy('Metal', '[ WoodBox | WoodWall ] -> [ | ]')[0]?.text,
        '[ MetalBox | MetalWall ] -> [ | ]');
    assert.equal(analogy('2', '[ Stage1 ] -> [ Stage2 ]')[0]?.text, '[ Stage2 ] -> [ Stage3 ]');
});
test('every substituted name must exist', () => {
    assert.equal(analogy('Blue', '[ RedKey | RedDoor ] -> [ | ]', {
        setup: source.replace('BlueDoor\nred\n', '').replace('RedDoor, BlueDoor', 'RedDoor')
    }).length, 0);
    assert.equal(analogy('3', '[ Stage1 ] -> [ Stage2 ]').length, 0);
});
test('search earlier rules, deduplicate, and preserve message text', () => {
    const previous = '[ RedKey | RedDoor ] -> [ | ] message RedKey (yes)\n\n[ Player ] -> [ Player ]';
    assert.equal(analogy('Blue', previous)[0]?.text, '[ BlueKey | BlueDoor ] -> [ | ] message RedKey (yes)');
    assert.equal(analogy('Blue', '[ RedKey | RedDoor ] -> [ | ]\n[ RedKey | RedDoor ] -> [ | ]').length, 1);
    assert.equal(analogy('Blue', '[ RedKey | RedDoor ] -> [ | ]\n[ BlueKey | BlueDoor ] -> [ | ]').length, 0);
});
test('analogy does not replace an existing rule or text after the cursor', () => {
    const previous = '[ RedKey | RedDoor ] -> [ | ]';
    assert.equal(analogy('[ Blu', previous).length, 0);
    assert.equal(analogy('Blue other', previous, { ch: 4 }).length, 0);
    const item = analogy('  Blue', previous)[0];
    assert.ok(item);
    assert.equal(item.from.ch, 2);
    assert.equal(item.to.ch, 6);
});
test('incomplete rules and unknown source names do not generate analogies', () => {
    assert.equal(analogy('Blue', '[ RedKey | RedDoor -> [ | ]').length, 0);
    assert.equal(analogy('Blue', '[ RedKey | Missing ] -> [ | ]').length, 0);
});

test('late headers and left-side random spawning are constrained', () => {
    lacks('late r', 'rigid');
    lacks('rigid l', 'late');
    lacks('[ r', 'random');
    has('[ Player ] -> [ r', 'random');
});
test('property help remains visible after typing its complete name', () => {
    const item = hints('[ Movable ] -> [ Movable').list.find(item => item.text.toLowerCase() === 'movable');
    assert.ok(item);
    assert.match(item.extra, /same object matched on the left/);
});
test('layer checks remain conservative when layer information is missing', () => {
    has('[ Player C', 'Crate', { setup: source.replace('Player, Crate\n', 'Player\n') });
});
test('random RHS terms form a choice pool, not simultaneous occupants', () => {
    has('[ Player ] -> [ random RedKey random B', 'BlueKey');
    has('[ Player ] -> [ Player random C', 'Crate');
});
test('a property entirely on an occupied layer cannot coexist there', () => {
    lacks('[ Player M', 'Movable');
    has('[ Player M', 'Mixed');
});
test('comments and messages do not become rule analogy sources', () => {
    assert.equal(analogy('Blue', '(\n[ RedKey | RedDoor ] -> [ | ]\n)').length, 0);
    assert.equal(analogy('Blue', '[ Player ] -> message [ RedKey | RedDoor ] -> [ | ]').length, 0);
    assert.equal(analogy('Blue', '[ RedKey | RedDoor ] -> [ | ] (RedKey stays in comment)')[0]?.text,
        '[ BlueKey | BlueDoor ] -> [ | ] (RedKey stays in comment)');
});
test('underscore families and zero-padded numbered stages preserve declared spelling', () => {
    const lowerSource = source.replaceAll('RedKey', 'key_red').replaceAll('RedDoor', 'door_red')
        .replaceAll('BlueKey', 'key_BLUE').replaceAll('BlueDoor', 'door_BLUE');
    assert.equal(analogy('blu', '[ key_red | door_red ] -> [ | ]', { setup: lowerSource })[0]?.text,
        '[ key_BLUE | door_BLUE ] -> [ | ]');
    const padded = source.replaceAll('Stage1', 'Stage01').replaceAll('Stage2', 'Stage02').replaceAll('Stage3', 'Stage03');
    assert.equal(analogy('02', '[ Stage01 ] -> [ Stage02 ]', { setup: padded })[0]?.text,
        '[ Stage02 ] -> [ Stage03 ]');
});
test('analogies reject provable generated layer conflicts', () => {
    const setup = source.replace('RedKey, BlueKey\nRedDoor, BlueDoor', 'RedKey\nRedDoor\nBlueKey, BlueDoor');
    assert.equal(analogy('Blue', '[ RedKey RedDoor ] -> [ ]', { setup }).length, 0);
});
test('numbered repeats preserve other numeric components', () => {
    const setup = source.replaceAll('Stage1', 'Part1Stage1').replaceAll('Stage2', 'Part1Stage2')
        .replaceAll('Stage3', 'Part1Stage3');
    assert.equal(analogy('2', '[ Part1Stage1 ] -> [ Part1Stage2 ]', { setup })[0]?.text,
        '[ Part1Stage2 ] -> [ Part1Stage3 ]');
});
test('win-condition positions ignore leading comments', () => {
    has('(a comment) all Crate o', 'on', { section: 'winconditions' });
});
test('large games with no matching family do not stall completion', () => {
    const manyObjects = Array.from({ length: 1000 }, (_, i) => 'Item' + i + '\nred\n').join('\n');
    const setup = source.replace('legend\n', manyObjects + '\nlegend\n');
    const previous = Array.from({ length: 500 }, (_, i) => '[ ' +
        Array.from({ length: 10 }, (_, j) => 'Item' + ((i + j) % 1000)).join(' | ') + ' ] -> win').join('\n');
    const started = performance.now();
    assert.equal(analogy('zz', previous, { setup }).length, 0);
    assert.ok(performance.now() - started < 1000, 'completion must not block typing for a second');
});
