#!/usr/bin/env node
'use strict';

const assert = require('assert/strict');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const srcDir = process.env.PUZZLESCRIPT_TEST_SRC || path.join(__dirname, '..');

function editor(width = 800, height = 480, levelWidth = 3, levelHeight = 3, count = 30) {
    const calls = [];
    const ctx = {
        drawImage(...args) { calls.push(['image', ...args]); },
        fillText(...args) { calls.push(['text', ...args]); }
    };
    const canvas = {
        parentNode: { clientWidth: width, clientHeight: height },
        getContext() { return ctx; },
        relMouseCoords(event) { return event; }
    };
    const state = {
        metadata: {}, levels: [{}], fgcolor: '#fff', bgcolor: '#000',
        glyphOrder: [], glyphDict: {}, synonymsDict: {}, aggregatesDict: {}, idDict: []
    };
    const glyphs = Array.from({ length: count }, (_, i) => ({ glyph: i }));
    const names = glyphs.map((_, i) => String.fromCharCode(33 + i));
    names.forEach((name, i) => { state.glyphOrder.push(name); state.glyphDict[name] = [i]; });
    const context = vm.createContext({
        console, ctx, canvas, state, cellwidth: 0, cellheight: 0, x: 0, y: 0,
        level: { width: levelWidth, height: levelHeight, getCellInto() { return { get() { return false; } }; } },
        levelEditorOpened: true, textMode: false, IDE: true, canOpenEditor: true,
        screenwidth: 0, screenheight: 0, flickscreen: false, zoomscreen: false,
        keyRepeatInterval: 150, throttle_movement: false, deltatime: 17,
        sprites: [], textImages: {}, _o12: {},
        document: { getElementById() { return canvas; }, addEventListener() {} },
        window: { addEventListener() {}, requestAnimationFrame() {} }, navigator: {},
        HTMLCanvasElement: class {},
        setInterval() {}, clearInterval() {}, setTimeout() {},
    });
    for (const file of ['js/graphics.js', 'js/inputoutput.js']) {
        vm.runInContext(fs.readFileSync(path.join(srcDir, file), 'utf8'), context, { filename: file });
    }
    context.glyphImages = glyphs;
    context.glyphImagesCorrespondance = names;
    context.glyphPrintButton = { print: true };
    context.glyphMouseOver = { hover: true };
    context.glyphHighlight = { highlight: true };
    context.regenSpriteImages = function () {};
    context.redraw = function () {};
    context.printCalls = 0;
    context.printLevel = function () { context.printCalls++; };
    context.canvasResize();
    function draw() {
        calls.length = 0;
        context.drawEditorIcons(0, 0);
        return calls.filter(call => call[0] === 'image' && call[1] && 'glyph' in call[1]);
    }
    function point(x, y) { context.setMouseCoord({ x, y }); }
    function click(glyph, right = false) {
        const call = draw().find(call => call[1].glyph === glyph);
        point(call[2] + context.cellwidth / 2, call[3] + context.cellheight / 2);
        context[right ? 'levelEditorRightClick' : 'levelEditorClick']({}, true);
    }
    return { context, canvas, calls, names, draw, point, click };
}

const cases = [
    ['narrow levels use the canvas width for the palette', () => {
        const e = editor();
        const icons = e.draw();
        assert.equal(icons[8][3], icons[0][3], 'ninth glyph should fit in the first canvas-wide row');
        assert.ok(icons[8][2] > e.context.xoffset + e.context.level.width * e.context.cellwidth);
    }],
    ['drawing and both mouse buttons agree on every glyph', () => {
        const e = editor();
        for (let i = 0; i < 30; i++) {
            e.click(i);
            assert.equal(e.context.glyphSelectedIndex, i, 'left click on glyph ' + i);
        }
        for (let i = 29; i >= 0; i--) {
            e.click(i, true);
            assert.equal(e.context.glyphSelectedIndex, i, 'right click on glyph ' + i);
        }
    }],
    ['blank palette cells do not change selection', () => {
        const e = editor();
        const last = e.draw().at(-1);
        e.context.glyphSelectedIndex = 4;
        e.point(last[2] + 1.5 * e.context.cellwidth, last[3] + e.context.cellheight / 2);
        e.context.levelEditorClick({}, true);
        e.context.levelEditorRightClick({}, true);
        assert.equal(e.context.glyphSelectedIndex, 4);
    }],
    ['only the print icon prints and every glyph has the right tooltip', () => {
        const e = editor();
        e.draw();
        const print = e.calls.find(call => call[0] === 'image' && call[1].print);
        e.point(print[2] + e.context.cellwidth / 2, print[3] + e.context.cellheight / 2);
        e.context.levelEditorClick({}, true);
        assert.equal(e.context.printCalls, 1);
        e.point(print[2] + e.context.cellwidth / 2, print[3] + 1.5 * e.context.cellheight);
        e.context.levelEditorClick({}, true);
        assert.equal(e.context.printCalls, 1, 'empty cells below print must not print');
        for (let i = 0; i < 30; i++) {
            e.click(i);
            e.draw();
            assert.ok(e.calls.some(call => call[0] === 'text' && call[1] === e.names[i]), 'tooltip ' + i);
        }
    }],
    ['resizing keeps every glyph inside the canvas and is stable', () => {
        const e = editor();
        for (const width of [800, 360, 1200, 240, 800]) {
            e.canvas.parentNode.clientWidth = width;
            e.context.canvasResize();
            const before = [e.context.cellwidth, e.context.cellheight, e.context.editorRowCount, e.context.xoffset, e.context.yoffset];
            for (const icon of e.draw()) {
                assert.ok(icon[2] >= 0 && icon[2] + e.context.cellwidth <= width, 'horizontal fit at ' + width);
                assert.ok(icon[3] >= 0 && icon[3] + e.context.cellheight <= e.canvas.height, 'vertical fit at ' + width);
            }
            e.context.canvasResize();
            assert.deepEqual([e.context.cellwidth, e.context.cellheight, e.context.editorRowCount, e.context.xoffset, e.context.yoffset], before);
            e.click(29);
            assert.equal(e.context.glyphSelectedIndex, 29);
        }
    }],
    ['normal play keeps its existing canvas sizing', () => {
        const e = editor();
        e.context.levelEditorOpened = false;
        e.context.canvasResize();
        assert.deepEqual([e.context.cellwidth, e.context.cellheight, e.context.xoffset, e.context.yoffset], [160, 160, 160, 0]);
    }],
];

let failed = 0;
for (const [name, test] of cases) {
    try { test(); console.log('PASS ' + name); }
    catch (error) { failed++; console.error('FAIL ' + name + '\n' + error.stack); }
}
console.log(`${cases.length - failed}/${cases.length} editor regression tests passed`);
process.exitCode = failed ? 1 : 0;
