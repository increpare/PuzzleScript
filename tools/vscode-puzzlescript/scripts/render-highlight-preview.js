#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const {
    PuzzleScriptEditorIntelligence,
} = require('../src/puzzlescriptEditorIntelligence');

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
    console.error('Usage: node scripts/render-highlight-preview.js <input.puzzlescript> <output.html>');
    process.exit(2);
}

const repoRoot = path.resolve(__dirname, '..', '..', '..');
const source = fs.readFileSync(path.resolve(inputPath), 'utf8');
const ps = new PuzzleScriptEditorIntelligence({ repoRoot });
const tokens = ps.tokenize(source);
const colorDecorations = ps.colorDecorations(source);

const tokenByLine = new Map();
for (const token of tokens) {
    if (!tokenByLine.has(token.line)) {
        tokenByLine.set(token.line, []);
    }
    tokenByLine.get(token.line).push(token);
}

const colorByKey = new Map();
for (const decoration of colorDecorations) {
    colorByKey.set(`${decoration.line}:${decoration.start}:${decoration.length}`, decoration.color);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function className(tokenType) {
    return String(tokenType || '').replace(/[^a-zA-Z0-9_-]/g, '');
}

function renderLine(lineText, lineNumber) {
    const lineTokens = (tokenByLine.get(lineNumber) || [])
        .slice()
        .sort((a, b) => a.start - b.start || b.length - a.length);
    let cursor = 0;
    let out = '';

    for (const token of lineTokens) {
        if (token.start < cursor) {
            continue;
        }
        if (token.start > cursor) {
            out += escapeHtml(lineText.slice(cursor, token.start));
        }
        const text = lineText.slice(token.start, token.start + token.length);
        const color = colorByKey.get(`${token.line}:${token.start}:${token.length}`);
        const style = color ? ` style="color: ${escapeHtml(color)}"` : '';
        out += `<span class="${className(token.tokenType)}"${style}>${escapeHtml(text)}</span>`;
        cursor = token.start + token.length;
    }

    if (cursor < lineText.length) {
        out += escapeHtml(lineText.slice(cursor));
    }
    return out || ' ';
}

const lines = source.split('\n');
const body = lines.map((line, index) => {
    const gutter = String(index + 1).padStart(4, ' ');
    return `<span class="line"><span class="gutter">${escapeHtml(gutter)}</span> ${renderLine(line, index)}</span>`;
}).join('\n');

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PuzzleScript Highlight Preview</title>
<style>
:root { color-scheme: dark; }
body {
  margin: 0;
  background: #0f192a;
  color: #d1edff;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 14px;
  line-height: 1.45;
}
pre { margin: 0; padding: 18px; white-space: pre; }
.line { display: block; }
.gutter { color: #62718c; user-select: none; }
.psHeader { color: #ae81ff; font-weight: 700; }
.psMetadata { color: #ae81ff; }
.psName { color: #1dc116; font-weight: 700; }
.psColor { color: #f7e26b; font-weight: 700; }
.psAssignment, .psArrow, .psBracket { color: #c11d16; }
.psLogic, .psCommand { color: #ae81ff; }
.psSound { color: #ffb454; text-decoration: underline; }
.psDirection { color: #c11dc1; }
.psMessage { color: #ffb454; font-style: italic; }
.psLevel { color: #aaaaaa; }
.psComment { color: #428bdd; font-style: italic; }
.psError { color: #ff6666; text-decoration: underline; }
</style>
</head>
<body>
<pre>${body}</pre>
</body>
</html>
`;

fs.writeFileSync(path.resolve(outputPath), html, 'utf8');
console.log(`Wrote ${outputPath}`);
