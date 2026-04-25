'use strict';

const path = require('path');
const vscode = require('vscode');
const {
    PuzzleScriptEditorIntelligence,
    TOKEN_TYPES,
    TOKEN_MODIFIERS,
} = require('./puzzlescriptEditorIntelligence');
const {
    PuzzleScriptDebugAdapter,
    PuzzleScriptDebugConfigurationProvider,
    PuzzleScriptDebugPreview,
} = require('./puzzlescriptDebugAdapter');
const { openGeneratorPanel } = require('./puzzlescriptGeneratorPanel');

const DOCUMENT_SELECTOR = [
    { language: 'puzzlescript' },
    { pattern: '**/*.puzzlescript' },
    { pattern: '**/*.ps' },
    { pattern: '**/*.txt' },
];

const semanticLegend = new vscode.SemanticTokensLegend(TOKEN_TYPES, TOKEN_MODIFIERS);
const SECTION_NAMES = new Set([
    'objects',
    'legend',
    'sounds',
    'collisionlayers',
    'rules',
    'winconditions',
    'levels',
]);

function resolveRepoRoot(context) {
    const config = vscode.workspace.getConfiguration('puzzlescript');
    const configuredRoot = config.get('repoRoot');
    return configuredRoot && String(configuredRoot).trim()
        ? String(configuredRoot)
        : path.resolve(context.extensionPath, '..', '..');
}

function createIntelligence(context) {
    const repoRoot = resolveRepoRoot(context);
    return new PuzzleScriptEditorIntelligence({ repoRoot });
}

function shouldHandleDocument(document, intelligence) {
    if (!document || document.isClosed) {
        return false;
    }
    if (document.languageId === 'puzzlescript') {
        return true;
    }
    const filename = document.fileName || '';
    if (/\.(ps|puzzlescript)$/i.test(filename)) {
        return true;
    }
    return /\.txt$/i.test(filename) && intelligence.looksLikePuzzleScript(document.getText());
}

function completionKindForTag(tag) {
    if (String(tag || '').startsWith('COLOR')) {
        return vscode.CompletionItemKind.Color;
    }
    switch (tag) {
        case 'NAME':
            return vscode.CompletionItemKind.Variable;
        case 'COMMAND':
        case 'MESSAGE_VERB':
            return vscode.CompletionItemKind.Keyword;
        case 'DIRECTION':
            return vscode.CompletionItemKind.EnumMember;
        case 'SOUNDEVENT':
        case 'SOUNDVERB':
            return vscode.CompletionItemKind.Event;
        default:
            return vscode.CompletionItemKind.Text;
    }
}

function itemInsertText(text) {
    if (text.indexOf('\n') >= 0) {
        return new vscode.SnippetString(text.replace(/\$/g, '\\$'));
    }
    return text;
}

function tokenModifierMask(modifiers) {
    let mask = 0;
    for (const modifier of modifiers || []) {
        const index = TOKEN_MODIFIERS.indexOf(modifier);
        if (index >= 0) {
            mask |= (1 << index);
        }
    }
    return mask;
}

function diagnosticSeverity(value) {
    return value === 'warning'
        ? vscode.DiagnosticSeverity.Warning
        : vscode.DiagnosticSeverity.Error;
}

function stripLineComment(line) {
    const index = String(line).indexOf('(');
    return index >= 0 ? String(line).slice(0, index) : String(line);
}

function levelRows(source) {
    const rows = [];
    const lines = String(source || '').split('\n');
    let section = '';
    let levelIndex = -1;
    let currentLevelIndex = null;
    let currentLevelHasRows = false;

    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
        const rawLine = lines[lineIndex];
        const uncommented = stripLineComment(rawLine);
        const trimmed = uncommented.trim();
        const sectionName = trimmed.toLowerCase();

        if (/^=+$/.test(trimmed)) {
            continue;
        }
        if (SECTION_NAMES.has(sectionName)) {
            section = sectionName;
            currentLevelIndex = null;
            currentLevelHasRows = false;
            continue;
        }
        if (section !== 'levels') {
            continue;
        }
        if (!trimmed) {
            if (currentLevelHasRows) {
                currentLevelIndex = null;
                currentLevelHasRows = false;
            }
            continue;
        }
        if (/^message\b/i.test(trimmed)) {
            levelIndex += 1;
            currentLevelIndex = null;
            currentLevelHasRows = false;
            continue;
        }
        if (currentLevelIndex == null) {
            levelIndex += 1;
            currentLevelIndex = levelIndex;
        }
        currentLevelHasRows = true;
        rows.push({
            line: lineIndex,
            start: rawLine.search(/\S|$/),
            end: rawLine.length,
            level: currentLevelIndex,
        });
    }
    return rows;
}

class PuzzleScriptSemanticTokenProvider {
    constructor(intelligence) {
        this.intelligence = intelligence;
    }

    provideDocumentSemanticTokens(document) {
        if (!shouldHandleDocument(document, this.intelligence)) {
            return new vscode.SemanticTokensBuilder(semanticLegend).build();
        }

        const builder = new vscode.SemanticTokensBuilder(semanticLegend);
        for (const token of this.intelligence.tokenize(document.getText())) {
            const tokenTypeIndex = TOKEN_TYPES.indexOf(token.tokenType);
            if (tokenTypeIndex < 0 || token.length <= 0) {
                continue;
            }
            builder.push(
                token.line,
                token.start,
                token.length,
                tokenTypeIndex,
                tokenModifierMask(token.modifiers)
            );
        }
        return builder.build();
    }
}

class PuzzleScriptCompletionProvider {
    constructor(intelligence) {
        this.intelligence = intelligence;
    }

    provideCompletionItems(document, position) {
        if (!shouldHandleDocument(document, this.intelligence)) {
            return [];
        }

        const result = this.intelligence.complete(document.getText(), position);
        const range = new vscode.Range(
            new vscode.Position(result.from.line, result.from.ch),
            new vscode.Position(result.to.line, result.to.ch)
        );
        return result.list.map((entry, index) => {
            const item = new vscode.CompletionItem(entry.text, completionKindForTag(entry.tag));
            item.insertText = itemInsertText(entry.text);
            item.range = range;
            item.detail = entry.extra || entry.tag || 'PuzzleScript';
            item.sortText = String(index).padStart(6, '0');
            if (entry.tag) {
                item.documentation = new vscode.MarkdownString(`PuzzleScript ${entry.tag}`);
            }
            return item;
        });
    }
}

class PuzzleScriptLevelLinkProvider {
    constructor(intelligence) {
        this.intelligence = intelligence;
    }

    provideDocumentLinks(document) {
        if (!shouldHandleDocument(document, this.intelligence)) {
            return [];
        }
        return levelRows(document.getText()).map(row => {
            const args = encodeURIComponent(JSON.stringify([row.level]));
            const link = new vscode.DocumentLink(
                new vscode.Range(
                    new vscode.Position(row.line, row.start),
                    new vscode.Position(row.line, row.end)
                ),
                vscode.Uri.parse(`command:puzzlescript.runCurrentGameLevel?${args}`)
            );
            link.tooltip = `Run PuzzleScript level ${row.level}`;
            return link;
        });
    }
}

class PuzzleScriptDecorations {
    constructor(intelligence) {
        this.intelligence = intelligence;
        this.colorTypes = new Map();
        this.activeEditors = new Set();
    }

    dispose() {
        for (const decorationType of this.colorTypes.values()) {
            decorationType.dispose();
        }
        this.colorTypes.clear();
    }

    decorationTypeForColor(color) {
        const key = String(color).toLowerCase();
        if (!this.colorTypes.has(key)) {
            this.colorTypes.set(key, vscode.window.createTextEditorDecorationType({
                color,
                fontWeight: '600',
            }));
        }
        return this.colorTypes.get(key);
    }

    update(editor) {
        if (!editor || !shouldHandleDocument(editor.document, this.intelligence)) {
            return;
        }
        this.activeEditors.add(editor);
        const rangesByColor = new Map();
        for (const span of this.intelligence.colorDecorations(editor.document.getText())) {
            const key = String(span.color).toLowerCase();
            if (!rangesByColor.has(key)) {
                rangesByColor.set(key, []);
            }
            rangesByColor.get(key).push(new vscode.Range(
                new vscode.Position(span.line, span.start),
                new vscode.Position(span.line, span.start + span.length)
            ));
        }

        for (const [color, decorationType] of this.colorTypes.entries()) {
            editor.setDecorations(decorationType, rangesByColor.get(color) || []);
        }
        for (const [color, ranges] of rangesByColor.entries()) {
            editor.setDecorations(this.decorationTypeForColor(color), ranges);
        }
    }
}

function refreshDiagnostics(document, intelligence, collection) {
    if (!shouldHandleDocument(document, intelligence)) {
        collection.delete(document.uri);
        return;
    }

    const diagnostics = intelligence.diagnose(document.getText()).map(entry => {
        const line = entry.line == null ? 0 : Math.min(entry.line, Math.max(0, document.lineCount - 1));
        const range = document.lineAt(line).range;
        const diagnostic = new vscode.Diagnostic(range, entry.message, diagnosticSeverity(entry.severity));
        diagnostic.source = 'PuzzleScript';
        return diagnostic;
    });
    collection.set(document.uri, diagnostics);
}

function activate(context) {
    const intelligence = createIntelligence(context);
    const diagnostics = vscode.languages.createDiagnosticCollection('puzzlescript');
    const decorations = new PuzzleScriptDecorations(intelligence);
    const debugPreview = new PuzzleScriptDebugPreview(context);
    let activeDebugAdapter = null;
    const setCurrentDocumentLanguage = async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('Open a PuzzleScript file first.');
            return null;
        }
        if (editor.document.languageId === 'puzzlescript') {
            return editor.document;
        }
        return vscode.languages.setTextDocumentLanguage(editor.document, 'puzzlescript');
    };
    const sendDebugInput = token => {
        if (!activeDebugAdapter) {
            vscode.window.showWarningMessage('Start a PuzzleScript debug session before sending debug inputs.');
            return;
        }
        activeDebugAdapter.acceptInput(token);
    };
    const startDebugCurrentGame = async level => {
        const editor = vscode.window.activeTextEditor;
        if (!editor || !shouldHandleDocument(editor.document, intelligence) || editor.document.uri.scheme !== 'file') {
            vscode.window.showWarningMessage('Open a PuzzleScript file before starting the PuzzleScript debugger.');
            return;
        }
        const document = editor.document.languageId === 'puzzlescript'
            ? editor.document
            : await vscode.languages.setTextDocumentLanguage(editor.document, 'puzzlescript');
        const numericLevel = Number(level);
        const folder = vscode.workspace.getWorkspaceFolder(document.uri);
        const config = {
            type: 'puzzlescript',
            request: 'launch',
            name: Number.isInteger(numericLevel) ? `Debug PuzzleScript Level ${numericLevel}` : 'Debug PuzzleScript',
            program: document.uri.fsPath,
            source: document.getText(),
            stopOnEntry: false,
        };
        if (Number.isInteger(numericLevel)) {
            config.level = numericLevel;
        }
        await vscode.debug.startDebugging(folder, config);
    };
    const startGenerator = async () => openGeneratorPanel({
        context,
        repoRoot: resolveRepoRoot(context),
        intelligence,
    });

    const refreshDocument = document => {
        refreshDiagnostics(document, intelligence, diagnostics);
        for (const editor of vscode.window.visibleTextEditors) {
            if (editor.document === document) {
                decorations.update(editor);
            }
        }
    };

    context.subscriptions.push(
        diagnostics,
        decorations,
        vscode.languages.registerDocumentSemanticTokensProvider(
            DOCUMENT_SELECTOR,
            new PuzzleScriptSemanticTokenProvider(intelligence),
            semanticLegend
        ),
        vscode.languages.registerCompletionItemProvider(
            DOCUMENT_SELECTOR,
            new PuzzleScriptCompletionProvider(intelligence),
            ' ',
            '[',
            ']',
            '|',
            '>',
            '-',
            ','
        ),
        vscode.languages.registerDocumentLinkProvider(
            DOCUMENT_SELECTOR,
            new PuzzleScriptLevelLinkProvider(intelligence)
        ),
        vscode.workspace.onDidOpenTextDocument(refreshDocument),
        vscode.workspace.onDidChangeTextDocument(event => refreshDocument(event.document)),
        vscode.debug.onDidTerminateDebugSession(() => {
            activeDebugAdapter = null;
        }),
        vscode.window.onDidChangeVisibleTextEditors(editors => {
            for (const editor of editors) {
                decorations.update(editor);
            }
        }),
        debugPreview,
        vscode.debug.registerDebugConfigurationProvider(
            'puzzlescript',
            new PuzzleScriptDebugConfigurationProvider()
        ),
        vscode.debug.registerDebugAdapterDescriptorFactory('puzzlescript', {
            createDebugAdapterDescriptor() {
                activeDebugAdapter = new PuzzleScriptDebugAdapter({
                    context,
                    repoRoot: resolveRepoRoot(context),
                    preview: debugPreview,
                });
                return new vscode.DebugAdapterInlineImplementation(activeDebugAdapter);
            }
        }),
        vscode.commands.registerCommand('puzzlescript.runCurrentGame', startDebugCurrentGame),
        vscode.commands.registerCommand('puzzlescript.debugCurrentGame', startDebugCurrentGame),
        vscode.commands.registerCommand('puzzlescript.generateLevels', startGenerator),
        vscode.commands.registerCommand('puzzlescript.runCurrentGameLevel', level => startDebugCurrentGame(level)),
        vscode.commands.registerCommand('puzzlescript.setLanguageMode', setCurrentDocumentLanguage),
        vscode.commands.registerCommand('puzzlescript.debugInputUp', () => sendDebugInput('up')),
        vscode.commands.registerCommand('puzzlescript.debugInputDown', () => sendDebugInput('down')),
        vscode.commands.registerCommand('puzzlescript.debugInputLeft', () => sendDebugInput('left')),
        vscode.commands.registerCommand('puzzlescript.debugInputRight', () => sendDebugInput('right')),
        vscode.commands.registerCommand('puzzlescript.debugInputAction', () => sendDebugInput('action')),
        vscode.commands.registerCommand('puzzlescript.debugTick', () => sendDebugInput('tick')),
        vscode.commands.registerCommand('puzzlescript.debugUndo', () => sendDebugInput('undo')),
        vscode.commands.registerCommand('puzzlescript.debugRestart', () => sendDebugInput('restart'))
    );

    for (const document of vscode.workspace.textDocuments) {
        refreshDocument(document);
    }
    for (const editor of vscode.window.visibleTextEditors) {
        decorations.update(editor);
    }
}

function deactivate() {}

module.exports = {
    activate,
    deactivate,
    levelRows,
};
