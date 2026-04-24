'use strict';

const fs = require('fs');
const path = require('path');
const vscode = require('vscode');
const { PuzzleScriptDebugRuntime } = require('./puzzlescriptDebugRuntime');

const THREAD_ID = 1;
const STACK_FRAME_ID = 1;
const VARIABLES_REF = 1;

function escapeHtml(value) {
    return String(value || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function valueString(value) {
    if (value == null) return '';
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    try {
        return JSON.stringify(value);
    } catch (error) {
        return String(value);
    }
}

class PuzzleScriptDebugPreview {
    constructor(context) {
        this.context = context;
        this.panel = null;
    }

    dispose() {
        if (this.panel) {
            this.panel.dispose();
            this.panel = null;
        }
    }

    update(snapshot) {
        if (!snapshot) {
            return;
        }
        if (!this.panel) {
            this.panel = vscode.window.createWebviewPanel(
                'puzzlescriptDebugCanvas',
                'PuzzleScript Debug Canvas',
                vscode.ViewColumn.Beside,
                { enableScripts: false, retainContextWhenHidden: true }
            );
            this.panel.onDidDispose(() => {
                this.panel = null;
            }, null, this.context.subscriptions);
        }

        const label = escapeHtml(snapshot.label || 'PuzzleScript');
        const line = snapshot.sourceLine ? `line ${snapshot.sourceLine}` : 'runtime phase';
        const board = escapeHtml(snapshot.serializedLevel || '');
        this.panel.webview.html = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
body { margin: 0; padding: 14px; font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--vscode-editor-foreground); background: var(--vscode-editor-background); }
.meta { display: flex; gap: 12px; margin-bottom: 12px; color: var(--vscode-descriptionForeground); }
.title { font-weight: 600; color: var(--vscode-editor-foreground); }
pre { margin: 0; padding: 12px; overflow: auto; border: 1px solid var(--vscode-panel-border); background: var(--vscode-textCodeBlock-background); line-height: 1.45; }
</style>
</head>
<body>
<div class="meta"><span class="title">${label}</span><span>${escapeHtml(line)}</span><span>input ${escapeHtml(snapshot.input || '')}</span></div>
<pre>${board}</pre>
</body>
</html>`;
    }
}

class PuzzleScriptDebugAdapter {
    constructor(options) {
        this.context = options.context;
        this.repoRoot = options.repoRoot;
        this.preview = options.preview;
        this._onDidSendMessage = new vscode.EventEmitter();
        this.onDidSendMessage = this._onDidSendMessage.event;
        this.seq = 1;
        this.breakpointId = 1;
        this.breakpoints = new Map();
        this.requestedBreakpoints = [];
        this.runtime = null;
        this.sourcePath = null;
        this.sourceText = null;
        this.sourceName = 'PuzzleScript';
        this.stopOnEntry = false;
        this.snapshots = [];
        this.snapshotIndex = -1;
        this.current = null;
        this.launched = false;
        this.disposed = false;
    }

    dispose() {
        this.disposed = true;
        this._onDidSendMessage.dispose();
    }

    handleMessage(message) {
        Promise.resolve()
            .then(() => this.dispatch(message))
            .catch(error => {
                this.sendResponse(message, false, undefined, error && error.message ? error.message : String(error));
            });
    }

    async dispatch(message) {
        switch (message.command) {
            case 'initialize':
                this.sendResponse(message, true, {
                    supportsConfigurationDoneRequest: true,
                    supportsRestartRequest: true,
                    supportsStepBack: false,
                    supportsEvaluateForHovers: false,
                });
                this.sendEvent('initialized');
                break;
            case 'launch':
                this.launch(message);
                break;
            case 'setBreakpoints':
                this.setBreakpoints(message);
                break;
            case 'configurationDone':
                this.sendResponse(message, true);
                if (this.stopOnEntry) {
                    this.stop('entry');
                }
                break;
            case 'threads':
                this.sendResponse(message, true, { threads: [{ id: THREAD_ID, name: 'PuzzleScript' }] });
                break;
            case 'stackTrace':
                this.stackTrace(message);
                break;
            case 'scopes':
                this.sendResponse(message, true, {
                    scopes: [{ name: 'PuzzleScript', variablesReference: VARIABLES_REF, expensive: false }],
                });
                break;
            case 'variables':
                this.variables(message);
                break;
            case 'continue':
                this.continue(message);
                break;
            case 'next':
                this.next(message);
                break;
            case 'pause':
                this.sendResponse(message, true);
                this.stop('pause');
                break;
            case 'restart':
                this.restart(message);
                break;
            case 'disconnect':
                this.sendResponse(message, true);
                this.sendEvent('terminated');
                break;
            default:
                this.sendResponse(message, true);
                break;
        }
    }

    launch(message) {
        const args = message.arguments || {};
        this.sourcePath = path.resolve(String(args.program || ''));
        this.sourceName = path.basename(this.sourcePath);
        this.stopOnEntry = Boolean(args.stopOnEntry);
        const source = args.source != null ? String(args.source) : fs.readFileSync(this.sourcePath, 'utf8');
        this.sourceText = source;
        this.runtime = new PuzzleScriptDebugRuntime({ repoRoot: this.repoRoot });
        const result = this.runtime.compile(source, {
            level: typeof args.level === 'number' ? args.level : undefined,
            seed: args.seed == null ? undefined : String(args.seed),
        });
        this.current = result.current || {
            label: 'Program start',
            sourceLine: 1,
            serializedLevel: '',
        };
        this.launched = true;
        this.revalidateBreakpoints(true);
        this.preview.update(this.current);
        this.sendResponse(message, true);
    }

    setBreakpoints(message) {
        const args = message.arguments || {};
        const sourcePath = args.source && args.source.path ? path.resolve(args.source.path) : this.sourcePath;
        this.requestedBreakpoints = (args.breakpoints || [])
            .map(bp => Number(bp.line))
            .filter(line => Number.isFinite(line))
            .map(line => ({
                id: this.breakpointId++,
                line,
                sourcePath,
                verified: false,
                message: 'PuzzleScript program has not been compiled yet.',
            }));
        this.revalidateBreakpoints(false);
        const responseBreakpoints = this.requestedBreakpoints.map(breakpoint => {
            return {
                id: breakpoint.id,
                verified: breakpoint.verified,
                line: breakpoint.line,
                message: breakpoint.message,
            };
        });
        this.sendResponse(message, true, { breakpoints: responseBreakpoints });
    }

    revalidateBreakpoints(sendEvents) {
        this.breakpoints.clear();
        if (!this.runtime || this.requestedBreakpoints.length === 0) {
            return;
        }
        const validations = this.runtime.validateBreakpoints(this.requestedBreakpoints.map(bp => bp.line));
        for (let index = 0; index < this.requestedBreakpoints.length; index++) {
            const breakpoint = this.requestedBreakpoints[index];
            const validation = validations[index];
            breakpoint.verified = Boolean(validation && validation.verified);
            breakpoint.message = validation && validation.message;
            if (breakpoint.verified) {
                this.breakpoints.set(breakpoint.line, breakpoint);
            }
            if (sendEvents) {
                this.sendEvent('breakpoint', {
                    reason: 'changed',
                    breakpoint: {
                        id: breakpoint.id,
                        verified: breakpoint.verified,
                        line: breakpoint.line,
                        message: breakpoint.message,
                    },
                });
            }
        }
    }

    restart(message) {
        if (!this.sourcePath) {
            this.sendResponse(message, false, undefined, 'No PuzzleScript program is loaded.');
            return;
        }
        const source = this.sourceText != null ? this.sourceText : fs.readFileSync(this.sourcePath, 'utf8');
        const result = this.runtime.compile(source, {});
        this.current = result.current;
        this.snapshots = [];
        this.snapshotIndex = -1;
        this.preview.update(this.current);
        this.sendResponse(message, true);
        this.stop('restart');
    }

    stackTrace(message) {
        const line = Math.max(1, Number(this.current && this.current.sourceLine) || 1);
        this.sendResponse(message, true, {
            stackFrames: [{
                id: STACK_FRAME_ID,
                name: this.current && this.current.label ? this.current.label : 'PuzzleScript',
                source: {
                    name: this.sourceName,
                    path: this.sourcePath,
                },
                line,
                column: 1,
            }],
            totalFrames: 1,
        });
    }

    variables(message) {
        const snapshot = this.current || {};
        const variables = [
            ['input', snapshot.input],
            ['rule line', snapshot.sourceLine || snapshot.lineNumber || ''],
            ['level index', snapshot.currentLevelIndex],
            ['title screen', snapshot.titleScreen],
            ['text mode', snapshot.textMode],
            ['winning', snapshot.winning],
            ['again', snapshot.againing],
            ['command queue', snapshot.commandQueue || []],
            ['command source rules', snapshot.commandQueueSourceRules || []],
            ['sounds', snapshot.soundHistory || []],
            ['serialized level', snapshot.serializedLevel || ''],
        ].map(([name, value]) => ({
            name,
            value: valueString(value),
            variablesReference: 0,
        }));
        this.sendResponse(message, true, { variables });
    }

    acceptInput(token) {
        if (!this.launched || !this.runtime) {
            vscode.window.showWarningMessage('Start a PuzzleScript debug session before sending debug inputs.');
            return;
        }
        this.snapshots = this.runtime.input(token);
        this.snapshotIndex = this.findNextBreakpointIndex(-1);
        if (this.snapshotIndex >= 0) {
            this.current = this.snapshots[this.snapshotIndex];
            this.preview.update(this.current);
            this.stop('breakpoint');
            return;
        }
        if (this.snapshots.length > 0) {
            this.snapshotIndex = this.snapshots.length - 1;
            this.current = this.snapshots[this.snapshotIndex];
            this.preview.update(this.current);
            this.sendEvent('output', {
                category: 'console',
                output: `PuzzleScript input ${token} completed without hitting a breakpoint.\n`,
            });
        }
    }

    continue(message) {
        const nextIndex = this.findNextBreakpointIndex(this.snapshotIndex);
        if (nextIndex >= 0) {
            this.snapshotIndex = nextIndex;
            this.current = this.snapshots[this.snapshotIndex];
            this.preview.update(this.current);
            this.sendResponse(message, true, { allThreadsContinued: true });
            this.stop('breakpoint');
            return;
        }
        if (this.snapshots.length > 0) {
            this.snapshotIndex = this.snapshots.length - 1;
            this.current = this.snapshots[this.snapshotIndex];
            this.preview.update(this.current);
        }
        this.sendResponse(message, true, { allThreadsContinued: true });
        this.sendEvent('continued', { threadId: THREAD_ID, allThreadsContinued: true });
    }

    next(message) {
        if (this.snapshots.length === 0 || this.snapshotIndex + 1 >= this.snapshots.length) {
            this.sendResponse(message, true);
            this.sendEvent('continued', { threadId: THREAD_ID, allThreadsContinued: true });
            return;
        }
        this.snapshotIndex += 1;
        this.current = this.snapshots[this.snapshotIndex];
        this.preview.update(this.current);
        this.sendResponse(message, true);
        this.stop('step');
    }

    findNextBreakpointIndex(afterIndex) {
        for (let index = afterIndex + 1; index < this.snapshots.length; index++) {
            const line = this.snapshots[index] && this.snapshots[index].sourceLine;
            if (line && this.breakpoints.has(line)) {
                return index;
            }
        }
        return -1;
    }

    stop(reason) {
        this.sendEvent('stopped', {
            reason,
            threadId: THREAD_ID,
            allThreadsStopped: true,
        });
    }

    sendResponse(request, success, body, message) {
        this._onDidSendMessage.fire({
            seq: this.seq++,
            type: 'response',
            request_seq: request.seq,
            command: request.command,
            success,
            message,
            body,
        });
    }

    sendEvent(event, body) {
        this._onDidSendMessage.fire({
            seq: this.seq++,
            type: 'event',
            event,
            body,
        });
    }
}

class PuzzleScriptDebugConfigurationProvider {
    resolveDebugConfiguration(folder, config) {
        const editor = vscode.window.activeTextEditor;
        const resolved = { ...config };
        if (!resolved.type) resolved.type = 'puzzlescript';
        if (!resolved.request) resolved.request = 'launch';
        if (!resolved.name) resolved.name = 'Debug PuzzleScript';
        if (!resolved.program && editor && editor.document && editor.document.uri.scheme === 'file') {
            resolved.program = editor.document.uri.fsPath;
        }
        if (editor && editor.document && editor.document.uri.scheme === 'file' && editor.document.uri.fsPath === resolved.program) {
            resolved.source = editor.document.getText();
        }
        if (resolved.stopOnEntry == null) {
            resolved.stopOnEntry = false;
        }
        return resolved;
    }
}

module.exports = {
    PuzzleScriptDebugAdapter,
    PuzzleScriptDebugConfigurationProvider,
    PuzzleScriptDebugPreview,
};
