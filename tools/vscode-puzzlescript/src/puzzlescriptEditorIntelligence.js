'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const SOURCE_FILES = [
    'js/storagewrapper.js',
    'js/bitvec.js',
    'js/level.js',
    'js/languageConstants.js',
    'js/globalVariables.js',
    'js/debug.js',
    'js/codemirror/stringstream.js',
    'js/colors.js',
    'js/parser.js',
    'js/compiler.js',
    'js/codemirror/rule-transform.js',
    'js/codemirror/anyword-hint.js',
];

const TOKEN_TYPES = [
    'psHeader',
    'psName',
    'psColor',
    'psAssignment',
    'psLogic',
    'psSound',
    'psDirection',
    'psArrow',
    'psBracket',
    'psCommand',
    'psMessage',
    'psMetadata',
    'psLevel',
    'psComment',
    'psError',
];

const TOKEN_MODIFIERS = [
    'spriteColor',
    'boldColor',
    'fadeColor',
    'metadataText',
    'soundEvent',
    'soundVerb',
    'messageVerb',
];

function defaultRepoRoot() {
    return path.resolve(__dirname, '..', '..', '..');
}

function escapeForVmString(value) {
    return JSON.stringify(String(value));
}

function stripHtml(value) {
    return String(value || '').replace(/<\/?[a-zA-Z][^>]*>/g, '').trim();
}

function tokenTypeForStyle(style) {
    const classes = String(style || '').split(/\s+/).filter(Boolean);
    if (classes.includes('HEADER') || classes.includes('EQUALSBIT')) return 'psHeader';
    if (classes.includes('NAME') || classes.includes('IDENTIFIER')) return 'psName';
    if (classes.includes('COLOR') || classes.some(c => c.startsWith('MULTICOLOR'))) return 'psColor';
    if (classes.includes('ASSIGNMENT')) return 'psAssignment';
    if (classes.includes('LOGICWORD')) return 'psLogic';
    if (classes.includes('SOUND') || classes.includes('SOUNDEVENT') || classes.includes('SOUNDVERB')) return 'psSound';
    if (classes.includes('DIRECTION')) return 'psDirection';
    if (classes.includes('ARROW')) return 'psArrow';
    if (classes.includes('BRACKET')) return 'psBracket';
    if (classes.includes('COMMAND')) return 'psCommand';
    if (classes.includes('MESSAGE') || classes.includes('MESSAGE_VERB')) return 'psMessage';
    if (classes.includes('METADATA') || classes.includes('METADATATEXT')) return 'psMetadata';
    if (classes.includes('LEVEL')) return 'psLevel';
    if (classes.includes('comment')) return 'psComment';
    if (classes.includes('ERROR')) return 'psError';
    return null;
}

function tokenModifiersForStyle(style) {
    const classes = String(style || '').split(/\s+/).filter(Boolean);
    const modifiers = [];
    if (classes.some(c => c.startsWith('COLOR-') || c.startsWith('MULTICOLOR'))) modifiers.push('spriteColor');
    if (classes.includes('BOLDCOLOR')) modifiers.push('boldColor');
    if (classes.includes('FADECOLOR')) modifiers.push('fadeColor');
    if (classes.includes('METADATATEXT')) modifiers.push('metadataText');
    if (classes.includes('SOUNDEVENT')) modifiers.push('soundEvent');
    if (classes.includes('SOUNDVERB')) modifiers.push('soundVerb');
    if (classes.includes('MESSAGE_VERB')) modifiers.push('messageVerb');
    return modifiers;
}

function trimTokenSpan(lineText, start, end) {
    let trimmedStart = start;
    let trimmedEnd = end;
    while (trimmedStart < trimmedEnd && /\s/.test(lineText.charAt(trimmedStart))) {
        trimmedStart += 1;
    }
    while (trimmedEnd > trimmedStart && /\s/.test(lineText.charAt(trimmedEnd - 1))) {
        trimmedEnd -= 1;
    }
    return { start: trimmedStart, end: trimmedEnd };
}

function parseDiagnostic(raw) {
    const message = stripHtml(raw);
    const match = message.match(/^line\s+(\d+)\s*:\s*(.*)$/i);
    const severity = /warningText/.test(String(raw)) ? 'warning' : 'error';
    if (!match) {
        return { line: null, message, severity };
    }
    return {
        line: Math.max(0, Number(match[1]) - 1),
        message: match[2] || message,
        severity,
    };
}

class PuzzleScriptEditorIntelligence {
    constructor(options = {}) {
        this.repoRoot = options.repoRoot || defaultRepoRoot();
        this.srcRoot = path.join(this.repoRoot, 'src');
        this.context = this.createContext();
        this.exports = this.loadEditorRuntime();
    }

    createContext() {
        const elements = [];
        const context = {
            console,
            setTimeout,
            clearTimeout,
            exports: undefined,
            module: undefined,
            require: undefined,
            localStorage: {
                getItem() { return null; },
                setItem() {},
                removeItem() {},
            },
        };
        context.window = context;
        context.globalThis = context;
        context.document = {
            URL: 'vscode-puzzlescript://',
            body: {
                classList: { contains() { return false; } },
                addEventListener() {},
                removeEventListener() {},
            },
            createElement(tagName) {
                const element = {
                    tagName,
                    className: '',
                    style: {},
                    children: [],
                    innerHTML: '',
                    textContent: '',
                    appendChild(child) {
                        this.children.push(child);
                        return child;
                    },
                    getContext() { return null; },
                };
                elements.push(element);
                return element;
            },
            createTextNode(text) {
                return { nodeType: 3, textContent: String(text) };
            },
            getElementById() { return null; },
        };
        context.consolePrint = function() {};
        context.consoleError = function() {};
        context.jumpToLine = function() {};
        context.UnitTestingThrow = function(error) { throw error; };
        return vm.createContext(context);
    }

    loadEditorRuntime() {
        let allCode = '';
        for (const file of SOURCE_FILES) {
            const absolute = path.join(this.srcRoot, file);
            allCode += `\n// ---- ${file} ----\n`;
            allCode += fs.readFileSync(absolute, 'utf8');
            allCode += '\n';
            if (file === 'js/codemirror/stringstream.js') {
                allCode += `
CodeMirror.helpers = CodeMirror.helpers || {};
CodeMirror.hint = CodeMirror.hint || {};
CodeMirror.Pos = function(line, ch) { return { line: line, ch: ch }; };
CodeMirror.registerHelper = function(type, name, value) {
    CodeMirror.helpers[type] = CodeMirror.helpers[type] || {};
    CodeMirror.helpers[type][name] = value;
    if (type === "hint") {
        CodeMirror.hint[name] = value;
    }
};
`;
            }
        }

        allCode += `
function __psSplitLines(source) {
    return String(source || "").split("\\n");
}

function __psRunParser(source, options) {
    options = options || {};
    resetParserErrorState();
    compiling = !!options.collectDiagnostics;
    var processor = new codeMirrorFn();
    var state = processor.startState();
    var tokens = [];
    var lines = __psSplitLines(source);
    try {
        for (var lineIndex = 0; lineIndex < lines.length; lineIndex++) {
            var line = lines[lineIndex];
            var stream = new CodeMirror.StringStream(line, 4);
            do {
                var before = stream.pos;
                var style = processor.token(stream, state) || "";
                var after = stream.pos;
                if (after === before && !stream.eol()) {
                    stream.next();
                    after = stream.pos;
                }
                if (after > before) {
                    tokens.push({
                        line: lineIndex,
                        start: before,
                        end: after,
                        text: line.slice(before, after),
                        style: style,
                    });
                }
                stream.start = stream.pos;
            } while (stream.eol() === false);
        }
        return {
            state: processor.copyState(state),
            tokens: tokens,
            diagnostics: errorStrings.slice(),
        };
    } finally {
        compiling = false;
    }
}

function __psTokenAt(source, lineNumber, ch) {
    resetParserErrorState();
    var processor = new codeMirrorFn();
    var state = processor.startState();
    var lines = __psSplitLines(source);
    var fallback = {
        start: ch,
        end: ch,
        string: "",
        state: processor.copyState(state),
    };
    for (var lineIndex = 0; lineIndex <= lineNumber && lineIndex < lines.length; lineIndex++) {
        var line = lines[lineIndex];
        var stream = new CodeMirror.StringStream(line, 4);
        do {
            var before = stream.pos;
            var style = processor.token(stream, state) || "";
            var after = stream.pos;
            if (after === before && !stream.eol()) {
                stream.next();
                after = stream.pos;
            }
            var token = {
                start: before,
                end: after,
                string: line.slice(before, after),
                type: style,
                state: processor.copyState(state),
            };
            if (lineIndex === lineNumber) {
                fallback = token;
                if (after >= ch || stream.eol()) {
                    return token;
                }
            }
            stream.start = stream.pos;
        } while (stream.eol() === false);
    }
    return fallback;
}

function __psComplete(source, lineNumber, ch) {
    var lines = __psSplitLines(source);
    var editor = {
        getCursor: function() { return { line: lineNumber, ch: ch }; },
        getLine: function(line) { return lines[line] || ""; },
        getTokenAt: function(pos) { return __psTokenAt(source, pos.line, pos.ch); },
    };
    var result = CodeMirror.hint.anyword(editor, { completeSingle: false });
    return {
        from: result && result.from ? result.from : { line: lineNumber, ch: ch },
        to: result && result.to ? result.to : { line: lineNumber, ch: ch },
        list: (result && result.list ? result.list : []).map(function(item, index) {
            return {
                text: String(item.text || ""),
                extra: String(item.extra || ""),
                tag: String(item.tag || ""),
                index: index,
            };
        }),
    };
}

function __psPaletteColor(name) {
    var normalized = String(name || "").toLowerCase();
    if (normalized === "transparent") return null;
    if (/^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(normalized)) return normalized;
    if (colorPalettes && colorPalettes.arnecolors && colorPalettes.arnecolors[normalized]) {
        return colorPalettes.arnecolors[normalized];
    }
    return null;
}

globalThis.__psEditorExports = {
    runParser: __psRunParser,
    tokenAt: __psTokenAt,
    complete: __psComplete,
    paletteColor: __psPaletteColor,
};
`;

        vm.runInContext(allCode, this.context, { filename: 'puzzlescript_editor_runtime.js' });
        return this.context.__psEditorExports;
    }

    tokenize(source) {
        const result = this.exports.runParser(String(source || ''), { collectDiagnostics: false });
        return result.tokens.map(token => {
            const span = trimTokenSpan(token.text, 0, token.text.length);
            const absoluteStart = token.start + span.start;
            const absoluteEnd = token.start + span.end;
            const style = token.style || '';
            return {
                line: token.line,
                start: absoluteStart,
                length: Math.max(0, absoluteEnd - absoluteStart),
                text: token.text.slice(span.start, span.end),
                style,
                tokenType: tokenTypeForStyle(style),
                modifiers: tokenModifiersForStyle(style),
            };
        }).filter(token => token.length > 0 && token.tokenType);
    }

    getStateAt(source, position) {
        return this.exports.tokenAt(String(source || ''), position.line, position.character || position.ch || 0).state;
    }

    complete(source, position) {
        return this.exports.complete(String(source || ''), position.line, position.character || position.ch || 0);
    }

    diagnose(source) {
        const result = this.exports.runParser(String(source || ''), { collectDiagnostics: true });
        return result.diagnostics.map(parseDiagnostic).filter(diagnostic => diagnostic.message);
    }

    colorDecorations(source) {
        const tokens = this.tokenize(source);
        return tokens.map(token => {
            const colorClass = String(token.style || '').split(/\s+/).find(part => part.startsWith('COLOR-'));
            const multiColorClass = String(token.style || '').split(/\s+/).find(part => part.startsWith('MULTICOLOR'));
            const color = colorClass
                ? this.exports.paletteColor(colorClass.slice('COLOR-'.length))
                : (multiColorClass ? multiColorClass.slice('MULTICOLOR'.length) : null);
            if (!color) {
                return null;
            }
            return {
                line: token.line,
                start: token.start,
                length: token.length,
                color,
            };
        }).filter(Boolean);
    }

    looksLikePuzzleScript(source) {
        return /(^|\n)\s*objects\s*(\n|$)/i.test(source)
            || /(^|\n)\s*collisionlayers\s*(\n|$)/i.test(source)
            || /(^|\n)\s*winconditions\s*(\n|$)/i.test(source);
    }
}

module.exports = {
    PuzzleScriptEditorIntelligence,
    TOKEN_TYPES,
    TOKEN_MODIFIERS,
    tokenTypeForStyle,
    stripHtml,
};
