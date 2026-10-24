#!/usr/bin/env node
'use strict';
// Feature-only hotfix for the verified older live release. A normal future
// release uses src/ + compile.js; this preserves the current site's older UI.
const fs = require('fs'), path = require('path'), zlib = require('zlib');
const { minify } = require('@swc/core');
const { TraceMap, eachMapping } = require('@jridgewell/trace-mapping');
const { GenMapping, addMapping, setSourceContent, toEncodedMap } = require('@jridgewell/gen-mapping');
const [baseline, output] = process.argv.slice(2);
if (!baseline || !output) throw Error('Usage: prepare_live_import.js BASELINE_DIR OUTPUT_DIR');
const repo = path.join(__dirname, '..');
const read = name => fs.readFileSync(path.join(baseline, name), 'utf8');
const changes = new Map();
function replaceOnce(text, before, after) {
    if (text.split(before).length !== 2) throw Error('Live baseline changed: ' + before.slice(0, 70));
    return text.replace(before, () => after);
}
function change(name, text) { changes.set(name, text); }
function source(name) { return changes.get('js/source/' + name) || read('js/source/' + name); }
const init = 'let fileToOpen=getParameterByName("demo");\nif (fileToOpen!==null&&fileToOpen.length>0) {';
const newInit = fs.readFileSync(path.join(repo, 'src/js/editor.js'), 'utf8').split('let importedSource = ')[1].split('\n\ttryLoadFile')[0];
change('js/source/editor.js', replaceOnce(source('editor.js'), init, 'let importedSource = ' + newInit));
change('js/source/buildStandalone.js', fs.readFileSync(path.join(repo, 'src/js/buildStandalone.js'), 'utf8'));
let parser = replaceOnce(source('parser.js'), "'noaction', 'norestart'", "'noaction', 'nohack', 'norestart'");
parser = replaceOnce(parser, 'logError(`Unrecognised stuff "${stream.string}" in the prelude.`, state.lineNumber);',
    "const text = stream.string.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');\n                                logError(`Unrecognised stuff \"${text}\" in the prelude.`, state.lineNumber);");
change('js/source/parser.js', parser);
change('js/source/codemirror/anyword-hint.js', replaceOnce(source('codemirror/anyword-hint.js'), '["noaction", ""],', '["noaction", ""],\n            ["nohack", ""],'));
const marker = '<script id="importedSource" type="application/json">null</script>';
let editorHtml = replaceOnce(read('editor.html'), '</head>', marker + '</head>');
editorHtml = replaceOnce(editorHtml, 'src="js/scripts_compiled.js"', 'src="js/scripts_compiled.js?v=edit-game-20261024-jumps"');
change('editor.html', editorHtml);
const doc = fs.readFileSync(path.join(repo, 'src/Documentation/prelude.html'), 'utf8');
const nohack = doc.match(/      <dt>nohack<\/dt>\n      <dd>.*?<\/dd>/)[0];
change('Documentation/prelude.html', replaceOnce(read('Documentation/prelude.html'), '<dt>noaction</dt>', nohack + '\n      <dt>noaction</dt>'));

async function bundle(mapName) {
    let offset = 0;
    const sources = JSON.parse(read('js/' + mapName + '.map')).sources.map(name => {
        const code = source(name.replace(/^source\//, ''));
        const result = { name, code, offset };
        offset += (code.match(/\n/g) || []).length + 1;
        return result;
    });
    const result = await minify(sources.map(s => s.code).join('\n'), {
        module: false, toplevel: false, compress: { toplevel: false, passes: 3 },
        mangle: { toplevel: false }, format: { comments: 'some', inlineScript: true }, sourceMap: true
    });
    const map = new GenMapping({ file: mapName });
    sources.forEach(s => setSourceContent(map, s.name, s.code));
    eachMapping(new TraceMap(result.map), m => {
        const generated = { line: m.generatedLine, column: m.generatedColumn };
        if (m.originalLine === null) return addMapping(map, { generated });
        const s = sources.find((s, i) => i === sources.length - 1 || m.originalLine <= sources[i + 1].offset);
        addMapping(map, { generated, source: s.name, original: { line: m.originalLine - s.offset, column: m.originalColumn }, name: m.name || undefined });
    });
    return { code: result.code, map: JSON.stringify(toEncodedMap(map)) };
}
(async () => {
    const editor = await bundle('scripts_compiled.js');
    change('js/scripts_compiled.js', editor.code + '\n//# sourceMappingURL=scripts_compiled.js.map\n');
    change('js/scripts_compiled.js.map', editor.map);
    const player = await bundle('scripts_play_compiled.js');
    // Only replace the inline engine in the export template. The live player
    // page and its shared bundle are deliberately left untouched.
    let template = read('standalone_inlined.txt');
    const scripts = Array.from(template.matchAll(/<script>([\s\S]*?)<\/script>/g));
    const engine = scripts.find(s => s[1].includes('sourceMappingURL=scripts_play_compiled.js.map'));
    if (!engine) throw Error('Live standalone engine script not found');
    template = replaceOnce(template, engine[0], '<script>' + player.code + '</script>');
    template = replaceOnce(template, '<div class="footer">', '<div class="footer">__HACK_LINK__');
    const css = '#playerLinks,#playerLinks form{display:inline}#playerLinks[hidden]{display:none}#playerLinks button{display:inline-block;min-height:44px;padding:8px 16px;border:0;background:transparent;color:___TEXTCOLOR___;font:inherit;font-size:14px;text-decoration:underline;cursor:pointer}.mobile-menu #playerLinks{display:block;clear:both;text-align:center}';
    template = replaceOnce(template, '</style>', css + '</style>');
    const compatibility = '<script>(function(){var links=document.getElementById("playerLinks");if(!links)return;links.addEventListener("touchstart",function(event){event.stopPropagation();});if(Mobile.hasTouch()){var original=Mobile.GestureHandler.prototype.buildMenu;Mobile.GestureHandler.prototype.buildMenu=function(){original.call(this);links.hidden=false;this.menuElem.appendChild(links);};}else{links.hidden=false;}})();</script>';
    template = replaceOnce(template, '<script>const sourceCode=', compatibility + '<script>const sourceCode=');
    change('standalone_inlined.txt', template);
    for (const [name, text] of changes) {
        const target = path.join(output, name);
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, text);
        if (!name.endsWith('.map')) fs.writeFileSync(target + '.gz', zlib.gzipSync(text, { level: 9 }));
    }
    const manifest = Array.from(changes.keys()).flatMap(name => name.endsWith('.map') ? [name] : [name, name + '.gz']);
    fs.writeFileSync(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2));
    console.log('Prepared ' + manifest.length + ' narrowly scoped live assets in ' + output);
})().catch(error => { console.error(error); process.exitCode = 1; });
