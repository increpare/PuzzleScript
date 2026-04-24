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

const DOCUMENT_SELECTOR = [
    { language: 'puzzlescript' },
    { pattern: '**/*.puzzlescript' },
    { pattern: '**/*.ps' },
    { pattern: '**/*.txt' },
];

const semanticLegend = new vscode.SemanticTokensLegend(TOKEN_TYPES, TOKEN_MODIFIERS);

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
    const sendDebugInput = token => {
        if (!activeDebugAdapter) {
            vscode.window.showWarningMessage('Start a PuzzleScript debug session before sending debug inputs.');
            return;
        }
        activeDebugAdapter.acceptInput(token);
    };

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
        vscode.commands.registerCommand('puzzlescript.debugCurrentGame', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || !shouldHandleDocument(editor.document, intelligence) || editor.document.uri.scheme !== 'file') {
                vscode.window.showWarningMessage('Open a PuzzleScript file before starting the PuzzleScript debugger.');
                return;
            }
            const folder = vscode.workspace.getWorkspaceFolder(editor.document.uri);
            await vscode.debug.startDebugging(folder, {
                type: 'puzzlescript',
                request: 'launch',
                name: 'Debug PuzzleScript',
                program: editor.document.uri.fsPath,
                source: editor.document.getText(),
                stopOnEntry: false,
            });
        }),
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
};
