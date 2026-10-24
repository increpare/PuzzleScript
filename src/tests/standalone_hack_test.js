'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const src = path.join(__dirname, '..');
const templatePath = process.argv[2] || path.join(src, 'standalone.html');
const template = fs.readFileSync(templatePath, 'utf8');
const exporter = fs.readFileSync(path.join(src, 'js/buildStandalone.js'), 'utf8');
const original = 'title A $& game\r\n(message </script><script>throw Error("executed")</script> & 雪\u2028)';
function exportGame(metadata) {
    let output;
    const context = vm.createContext({
        XMLHttpRequest: class { open() {} send() {} },
        state: { metadata }, consolePrint() {}, saveAs: value => output = value
    });
    vm.runInContext(exporter, context);
    context.template = template;
    context.source = JSON.stringify(original);
    vm.runInContext('standalone_HTML_String = template; buildStandalone(source)', context);
    return output;
}
const html = exportGame({});
assert.match(html, /action="https:\/\/www\.puzzlescript\.net\/editor\.html"/);
assert.match(html, /method="post"/);
assert.match(html, /name="source"/);
assert.match(html, /<div class="footer-links"><a[\s\S]*<\/a><nav id="playerLinks" hidden><span class="footer-separator"[\s\S]*>\|<\/span><form[\s\S]*>hack<\/button>[\s\S]*<\/nav><\/div>/);
assert.doesNotMatch(html, /<script>throw Error/);
const script = html.match(/<script>const sourceCode=([\s\S]*?)<\/script>/)[1];
let received;
const runtime = vm.createContext({ compile: (_, source) => received = source });
vm.runInContext('const sourceCode=' + script, runtime);
assert.equal(received, original);
const submit = html.match(/onsubmit="([^"]*)"/)[1];
runtime.form = { elements: { source: { value: '' } } };
vm.runInContext('(function(){' + submit + '}).call(form)', runtime);
assert.equal(runtime.form.elements.source.value, original);
const nohackHtml = exportGame({ nohack: true });
assert.doesNotMatch(nohackHtml, /action="https:\/\/www\.puzzlescript\.net\/editor\.html"|>hack<\/button>|class="footer-separator"|<nav id="playerLinks"/);
assert.match(nohackHtml, /id="homepagelink"/);

// Import must take precedence over local saves, demos, and gist lookups.
const init = fs.readFileSync(path.join(src, 'js/editor.js'), 'utf8').split('CodeMirror.commands.swapLineUp')[0];
const code = { value: 'default' };
let diagnosticClick;
const jumps = [];
const editorContext = vm.createContext({
    window: { CodeMirror: { defineMode() {}, registerHelper() {} } }, codeMirrorFn() {},
    document: { getElementById: id => id === 'code' ? code : id === 'consoletextarea' ? {
        addEventListener: (type, handler) => { assert.equal(type, 'click'); diagnosticClick = handler; }
    } : { textContent: JSON.stringify(original), remove() {} } },
    jumpToLine: line => jumps.push(line),
    getParameterByName() { throw Error('Import should take precedence over URL loading'); },
    storage_has() { throw Error('Import must not read saves'); }
});
vm.runInContext(init, editorContext);
assert.equal(code.value, original);
assert.equal(typeof diagnosticClick, 'function', 'Imported diagnostic links need a safe click handler');
for (const [handler, expected] of [
    ['jumpToLine(12);', 12], ["jumpToLine('3');", 3], ['jumpToLine("7")', 7],
    ['jumpToLine(1);window.__importInjection=true', null], ['alert(1)', null],
    ['jumpToLine(-1);', null], ['jumpToLine(1.5);', null],
    ['jumpToLine(9007199254740992);', null], [null, null]
]) {
    let prevented = false;
    const before = jumps.length;
    diagnosticClick({
        target: { closest: selector => { assert.equal(selector, 'a'); return { getAttribute: () => handler }; } },
        preventDefault: () => prevented = true
    });
    assert.equal(prevented, expected !== null);
    assert.equal(jumps.length, before + (expected === null ? 0 : 1));
    if (expected !== null) assert.equal(jumps.at(-1), expected);
}
console.log('PASS: imported diagnostic links accept only validated numeric jump targets');
vm.runInNewContext(init, {
    window: { CodeMirror: { defineMode() {}, registerHelper() {} } }, codeMirrorFn() {},
    document: { getElementById: id => id === 'code' ? { value: 'default' } : {
        textContent: 'null', addEventListener() { throw Error('Normal editor must retain its existing handlers'); }
    } },
    getParameterByName: () => null, storage_has: () => false
});
console.log('PASS: exported Hack POST, nohack omission, safe source round-trip, import precedence (' + templatePath + ')');

// Exercise actual tokenizer and completion helper rather than matching keywords.
const parserContext = vm.createContext({ consolePrint() {}, consoleCacheDump() {}, consoleError() {} });
for (const file of ['languageConstants.js', 'codemirror/stringstream.js', 'parser.js']) {
    vm.runInContext(fs.readFileSync(path.join(src, 'js', file), 'utf8'), parserContext);
}
const metadata = vm.runInContext(`(() => {
    const mode = codeMirrorFn(), state = mode.startState();
    const stream = new CodeMirror.StringStream('nohack');
    while (!stream.eol()) { stream.start = stream.pos; if (mode.token(stream, state) !== 'METADATA') throw Error('nohack is not highlighted as metadata'); }
    return state.metadata;
})()`, parserContext);
assert.deepEqual(Array.from(metadata), ['nohack', 'true']);
let hint;
const hintContext = vm.createContext({ CodeMirror: {
    registerHelper: (_, name, helper) => { if (name === 'anyword') hint = helper; },
    Pos: (line, ch) => ({ line, ch })
} });
vm.runInContext(fs.readFileSync(path.join(src, 'js/codemirror/anyword-hint.js'), 'utf8'), hintContext);
const hints = hint({
    getCursor: () => ({ line: 0, ch: 4 }), getLine: () => 'noha',
    getTokenAt: () => ({ string: 'noha', state: { section: '', commentLevel: 0 } }),
    firstLine: () => 0, lastLine: () => 0
});
assert.ok(hints.list.some(entry => entry.text === 'nohack'));
console.log('PASS: nohack tokenizer and autocomplete');

const errors = [];
parserContext.consolePrint = value => errors.push(value);
vm.runInContext(`(() => {
    compiling = true;
    const mode = codeMirrorFn(), state = mode.startState();
    const stream = new CodeMirror.StringStream('<img src=x onerror="window.__importInjection=true">');
    while (!stream.eol()) { stream.start = stream.pos; mode.token(stream, state); }
})()`, parserContext);
assert.ok(errors.some(text => text.includes('&lt;img')));
assert.ok(errors.every(text => !text.includes('<img')));
console.log('PASS: source HTML in prelude errors stays text');
