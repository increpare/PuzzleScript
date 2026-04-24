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
        this.inputHandler = null;
    }

    setInputHandler(handler) {
        this.inputHandler = handler;
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
                { enableScripts: true, retainContextWhenHidden: true }
            );
            this.panel.onDidDispose(() => {
                this.panel = null;
            }, null, this.context.subscriptions);
            this.panel.webview.onDidReceiveMessage(message => {
                if (message && message.type === 'input' && this.inputHandler) {
                    this.inputHandler(String(message.token || ''));
                }
            }, null, this.context.subscriptions);
        }

        const snapshotJson = JSON.stringify(snapshot).replace(/</g, '\\u003c');
        this.panel.webview.html = `<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body { height: 100%; }
body { margin: 0; padding: 14px; box-sizing: border-box; font: 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--vscode-editor-foreground); background: var(--vscode-editor-background); }
.shell { display: grid; grid-template-columns: minmax(220px, 1fr) 230px; gap: 14px; align-items: stretch; height: calc(100% - 28px); min-height: 260px; }
.meta { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; color: var(--vscode-descriptionForeground); }
.title { font-weight: 600; color: var(--vscode-editor-foreground); }
.stage { position: relative; min-height: 220px; overflow: hidden; background: var(--vscode-editor-background); }
canvas { display: block; width: 100%; height: 100%; image-rendering: pixelated; }
.hoverbox { position: absolute; pointer-events: none; border: 2px solid var(--vscode-focusBorder); box-sizing: border-box; display: none; }
.side { display: grid; gap: 12px; }
.controls { display: grid; grid-template-columns: repeat(3, 42px); gap: 6px; justify-content: start; }
button { height: 34px; border: 1px solid var(--vscode-button-border, transparent); color: var(--vscode-button-foreground); background: var(--vscode-button-background); border-radius: 3px; cursor: pointer; }
button:hover { background: var(--vscode-button-hoverBackground); }
.wide { grid-column: span 3; width: 138px; }
.panel { border: 1px solid var(--vscode-panel-border); background: var(--vscode-editorWidget-background); padding: 10px; min-height: 92px; }
.panel h2 { font-size: 12px; margin: 0 0 8px; color: var(--vscode-descriptionForeground); font-weight: 600; }
.objects { display: grid; gap: 6px; }
.object { display: flex; align-items: center; gap: 7px; }
.swatch { width: 13px; height: 13px; border: 1px solid rgba(0,0,0,.35); box-sizing: border-box; }
.empty { color: var(--vscode-descriptionForeground); }
pre { white-space: pre-wrap; margin: 0; font-size: 11px; color: var(--vscode-descriptionForeground); }
</style>
</head>
<body>
<div class="meta" id="meta"></div>
<div class="shell">
  <div class="stage" id="stage">
    <canvas id="board"></canvas>
    <div class="hoverbox" id="hoverbox"></div>
  </div>
  <div class="side">
    <div class="controls">
      <span></span><button data-token="up" title="Up">Up</button><span></span>
      <button data-token="left" title="Left">Left</button><button data-token="action" title="Action">Act</button><button data-token="right" title="Right">Right</button>
      <span></span><button data-token="down" title="Down">Down</button><span></span>
      <button class="wide" data-token="tick" title="Tick / wait">Tick</button>
      <button class="wide" data-token="undo" title="Undo">Undo</button>
      <button class="wide" data-token="restart" title="Restart">Restart</button>
    </div>
    <div class="panel">
      <h2>Cell</h2>
      <div id="inspect" class="empty">Hover a cell.</div>
    </div>
  </div>
</div>
<script>
const vscode = acquireVsCodeApi();
const snapshot = ${snapshotJson};
const stage = document.getElementById('stage');
const board = document.getElementById('board');
const hoverbox = document.getElementById('hoverbox');
const meta = document.getElementById('meta');
const inspect = document.getElementById('inspect');
const context = board.getContext('2d');
let layout = { scale: 1, cellSize: 0, offsetX: 0, offsetY: 0, width: 0, height: 0 };

function cssColor(value, fallback) {
  const color = String(value || '').trim();
  if (!color || color.toLowerCase() === 'transparent') return fallback;
  return color;
}

function textColor(color) {
  const value = String(color || '').trim();
  if (!value.startsWith('#')) return 'white';
  const hex = value.length === 4
    ? value.slice(1).split('').map(ch => ch + ch).join('')
    : value.slice(1, 7);
  const n = Number.parseInt(hex, 16);
  if (!Number.isFinite(n)) return 'white';
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return ((r * 299 + g * 587 + b * 114) / 1000) > 135 ? '#111' : '#fff';
}

function objectsForCell(x, y) {
  const width = snapshot.width || 0;
  const height = snapshot.height || 0;
  const stride = snapshot.strideObject || 0;
  const tileIndex = x * height + y;
  if (!width || !height || !stride) return [];
  return (snapshot.objectInfos || []).filter(info => {
    const word = tileIndex * stride + (info.id >> 5);
    const bit = info.id & 31;
    return Boolean((snapshot.objects[word] || 0) & (1 << bit));
  }).sort((a, b) => (a.layer || 0) - (b.layer || 0));
}

function colorForSpriteIndex(object, index) {
  if (index == null || index < 0) return null;
  return cssColor((object.colors || [])[index], null);
}

function drawObjectSprite(object, x, y, tileSize) {
  const matrix = object.spriteMatrix || [];
  const pixelSize = tileSize / 5;
  for (let py = 0; py < 5; py++) {
    const row = matrix[py] || [];
    for (let px = 0; px < 5; px++) {
      const raw = row[px];
      const index = typeof raw === 'number' ? raw : Number(raw);
      const color = colorForSpriteIndex(object, index);
      if (!color) continue;
      context.fillStyle = color;
      context.fillRect(x + px * pixelSize, y + py * pixelSize, Math.ceil(pixelSize), Math.ceil(pixelSize));
    }
  }
}

function renderInspect(objects, x, y) {
  if (!objects.length) {
    inspect.className = 'empty';
    inspect.textContent = 'Cell ' + x + ', ' + y + ' is empty.';
    return;
  }
  inspect.className = 'objects';
  inspect.innerHTML = objects.map(object => {
    const color = cssColor((object.colors || [])[0], 'transparent');
    return '<div class="object"><span class="swatch" style="background:' + color + '"></span><span>' + object.name + '</span><span class="empty">layer ' + object.layer + '</span></div>';
  }).join('');
}

function render() {
  const width = snapshot.width || 1;
  const height = snapshot.height || 1;
  meta.innerHTML = '<span class="title">' + (snapshot.label || 'PuzzleScript') + '</span>'
    + '<span>' + (snapshot.sourceLine ? 'line ' + snapshot.sourceLine : 'runtime phase') + '</span>'
    + '<span>input ' + (snapshot.input || '') + '</span>'
    + '<span>level ' + (snapshot.currentLevelIndex ?? '') + '</span>';
  const rect = stage.getBoundingClientRect();
  const deviceScale = window.devicePixelRatio || 1;
  board.width = Math.max(1, Math.floor(rect.width * deviceScale));
  board.height = Math.max(1, Math.floor(rect.height * deviceScale));
  board.style.width = rect.width + 'px';
  board.style.height = rect.height + 'px';
  context.setTransform(deviceScale, 0, 0, deviceScale, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  context.imageSmoothingEnabled = false;
  const cellSize = Math.max(1, Math.floor(Math.min(rect.width / width, rect.height / height)));
  const drawWidth = cellSize * width;
  const drawHeight = cellSize * height;
  const offsetX = Math.floor((rect.width - drawWidth) / 2);
  const offsetY = Math.floor((rect.height - drawHeight) / 2);
  layout = { cellSize, offsetX, offsetY, width, height };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const objects = objectsForCell(x, y);
      const cellX = offsetX + x * cellSize;
      const cellY = offsetY + y * cellSize;
      context.fillStyle = 'transparent';
      context.clearRect(cellX, cellY, cellSize, cellSize);
      for (const object of objects) {
        drawObjectSprite(object, cellX, cellY, cellSize);
      }
    }
  }
}

function inspectAt(clientX, clientY) {
  const rect = stage.getBoundingClientRect();
  const x = Math.floor((clientX - rect.left - layout.offsetX) / layout.cellSize);
  const y = Math.floor((clientY - rect.top - layout.offsetY) / layout.cellSize);
  if (x < 0 || y < 0 || x >= layout.width || y >= layout.height) {
    hoverbox.style.display = 'none';
    inspect.className = 'empty';
    inspect.textContent = 'Hover a cell.';
    return;
  }
  hoverbox.style.display = 'block';
  hoverbox.style.left = (layout.offsetX + x * layout.cellSize) + 'px';
  hoverbox.style.top = (layout.offsetY + y * layout.cellSize) + 'px';
  hoverbox.style.width = layout.cellSize + 'px';
  hoverbox.style.height = layout.cellSize + 'px';
  renderInspect(objectsForCell(x, y), x, y);
}

document.querySelectorAll('button[data-token]').forEach(button => {
  button.addEventListener('click', () => vscode.postMessage({ type: 'input', token: button.dataset.token }));
});
window.addEventListener('keydown', event => {
  const map = { ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right', ' ': 'action', z: 'undo', r: 'restart', '.': 'tick' };
  const token = map[event.key];
  if (token) {
    event.preventDefault();
    vscode.postMessage({ type: 'input', token });
  }
});
stage.addEventListener('mousemove', event => inspectAt(event.clientX, event.clientY));
stage.addEventListener('mouseleave', () => {
  hoverbox.style.display = 'none';
  inspect.className = 'empty';
  inspect.textContent = 'Hover a cell.';
});
new ResizeObserver(render).observe(stage);
render();
</script>
</body>
</html>`;
    }
}

class PuzzleScriptDebugAdapter {
    constructor(options) {
        this.context = options.context;
        this.repoRoot = options.repoRoot;
        this.preview = options.preview;
        this.preview.setInputHandler(token => this.acceptInput(token));
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
        this.preview.setInputHandler(null);
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
        const line = Number(this.current && this.current.sourceLine);
        const sourceLine = Number.isInteger(line) && line > 0 ? line : null;
        if (sourceLine == null) {
            this.sendResponse(message, true, {
                stackFrames: [],
                totalFrames: 0,
            });
            return;
        }
        this.sendResponse(message, true, {
            stackFrames: [{
                id: STACK_FRAME_ID,
                name: this.current && this.current.label ? this.current.label : 'PuzzleScript',
                source: {
                    name: this.sourceName,
                    path: this.sourcePath,
                },
                line: sourceLine,
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
        const result = this.runtime.runInput(token, {
            breakpoints: Array.from(this.breakpoints.keys()),
        });
        this.handleRunResult(result, 'breakpoint', `PuzzleScript input ${token} completed without hitting a breakpoint.\n`);
    }

    continue(message) {
        const result = this.runtime.resume({
            breakpoints: Array.from(this.breakpoints.keys()),
        });
        this.sendResponse(message, true, { allThreadsContinued: true });
        this.handleRunResult(result, 'breakpoint', 'PuzzleScript input completed.\n', true);
    }

    next(message) {
        const result = this.runtime.resume({ step: true });
        this.sendResponse(message, true);
        this.handleRunResult(result, 'step', 'PuzzleScript input completed.\n', true);
    }

    handleRunResult(result, stopReason, completionOutput, alreadyResponded) {
        this.snapshots = result && result.snapshots ? result.snapshots : [];
        this.snapshotIndex = this.snapshots.length - 1;
        if (result && result.snapshot) {
            this.current = result.snapshot;
            this.preview.update(this.current);
        }
        if (result && result.paused) {
            this.stop(stopReason);
            return;
        }
        if (completionOutput) {
            this.sendEvent('output', {
                category: 'console',
                output: completionOutput,
            });
        }
        if (alreadyResponded) {
            this.sendEvent('continued', { threadId: THREAD_ID, allThreadsContinued: true });
        }
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
