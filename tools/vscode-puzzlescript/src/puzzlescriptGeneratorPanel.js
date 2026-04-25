'use strict';

const fs = require('fs');
const vscode = require('vscode');
const {
    DEFAULT_GENERATOR_OPTIONS,
    PRESETS,
    candidateToRows,
    insertionAfterLevel,
    normalizeRunOptions,
    readSidecarOrDefault,
    replacementForLevel,
    resolveGeneratorPath,
    selectedLevelForLine,
    specFromParts,
} = require('./puzzlescriptGeneratorCore');
const { PuzzleScriptGeneratorRun } = require('./puzzlescriptGeneratorRunner');

class PuzzleScriptGeneratorPanel {
    constructor({ context, repoRoot, document, level }) {
        this.context = context;
        this.repoRoot = repoRoot;
        this.document = document;
        this.level = level;
        this.currentRun = null;
        this.lastResult = null;
        this.options = { ...DEFAULT_GENERATOR_OPTIONS };
        this.sidecar = readSidecarOrDefault(document.uri.fsPath, level);
        this.panel = vscode.window.createWebviewPanel(
            'puzzlescriptGenerator',
            'PuzzleScript Generator',
            vscode.ViewColumn.Beside,
            { enableScripts: true, retainContextWhenHidden: true }
        );
        this.panel.webview.html = this.html();
        this.panel.onDidDispose(() => this.dispose(), null, context.subscriptions);
        this.panel.webview.onDidReceiveMessage(message => this.handleMessage(message), null, context.subscriptions);
    }

    dispose() {
        if (this.currentRun) {
            this.currentRun.cancel();
            this.currentRun = null;
        }
    }

    post(message) {
        this.panel.webview.postMessage(message);
    }

    async handleMessage(message) {
        try {
            switch (message && message.type) {
                case 'ready':
                    this.postState();
                    break;
                case 'run':
                    await this.run(message);
                    break;
                case 'stop':
                    this.stop();
                    break;
                case 'saveRecipe':
                    await this.saveRecipe(message.specText);
                    break;
                case 'preset':
                    this.post({ type: 'preset', specText: specFromParts(this.level ? this.level.rows : [], presetRules(message.id)) });
                    break;
                case 'adopt':
                    await this.adopt(message.index);
                    break;
                case 'insert':
                    await this.insert(message.index);
                    break;
                case 'preview':
                    this.preview(message.index);
                    break;
                default:
                    break;
            }
        } catch (error) {
            this.post({ type: 'error', message: error.message || String(error) });
        }
    }

    postState() {
        const config = vscode.workspace.getConfiguration('puzzlescript');
        const resolved = resolveGeneratorPath(config.get('generatorPath'), this.repoRoot);
        this.post({
            type: 'state',
            level: this.level,
            specText: this.sidecar.text,
            sidecarPath: this.sidecar.path,
            sidecarExisted: this.sidecar.existed,
            generator: resolved,
            options: this.options,
            presets: PRESETS.map(({ id, label }) => ({ id, label })),
        });
    }

    async saveRecipe(specText) {
        fs.writeFileSync(this.sidecar.path, String(specText || ''), 'utf8');
        this.sidecar.text = String(specText || '');
        this.sidecar.existed = true;
        this.post({ type: 'saved', sidecarPath: this.sidecar.path });
    }

    async run(message) {
        if (this.currentRun) {
            this.currentRun.cancel();
        }
        const config = vscode.workspace.getConfiguration('puzzlescript');
        const resolved = resolveGeneratorPath(config.get('generatorPath'), this.repoRoot);
        if (!resolved.exists) {
            this.post({
                type: 'error',
                message: `Native generator not found at ${resolved.path}. Build it with: make build_generator`,
            });
            return;
        }
        this.options = normalizeRunOptions(message.options || this.options);
        const specText = String(message.specText || this.sidecar.text || '');
        this.sidecar.text = specText;
        this.lastResult = null;
        this.post({ type: 'running', options: this.options });
        const startedAt = Date.now();
        this.currentRun = new PuzzleScriptGeneratorRun({
            binaryPath: resolved.path,
            sourceText: this.document.getText(),
            specText,
            runOptions: this.options,
            onProgress: progress => this.post({ type: 'progress', progress, elapsedMs: Date.now() - startedAt }),
        });
        try {
            const output = await this.currentRun.start();
            this.currentRun = null;
            if (output.cancelled) {
                this.post({ type: 'stopped' });
                return;
            }
            this.lastResult = output.result;
            this.post({
                type: 'result',
                result: output.result,
                rows: (output.result.top || []).map(candidate => candidateToRows(candidate, this.document.getText())),
                elapsedMs: Date.now() - startedAt,
            });
        } catch (error) {
            this.currentRun = null;
            this.post({ type: 'error', message: error.message || String(error) });
        }
    }

    stop() {
        if (this.currentRun) {
            this.currentRun.cancel();
            this.currentRun = null;
        }
        this.post({ type: 'stopped' });
    }

    candidateAt(index) {
        const candidates = this.lastResult && Array.isArray(this.lastResult.top) ? this.lastResult.top : [];
        const candidate = candidates[Number(index)];
        if (!candidate) {
            throw new Error('No generator candidate selected.');
        }
        return candidate;
    }

    async adopt(index) {
        const candidate = this.candidateAt(index);
        const source = this.document.getText();
        const edit = replacementForLevel(source, this.level, candidate);
        const workspaceEdit = new vscode.WorkspaceEdit();
        workspaceEdit.replace(
            this.document.uri,
            new vscode.Range(new vscode.Position(edit.startLine, 0), new vscode.Position(edit.endLine, 0)),
            edit.text + '\n'
        );
        await vscode.workspace.applyEdit(workspaceEdit);
        this.post({ type: 'applied', mode: 'replace' });
    }

    async insert(index) {
        const candidate = this.candidateAt(index);
        const source = this.document.getText();
        const edit = insertionAfterLevel(source, this.level, candidate);
        const workspaceEdit = new vscode.WorkspaceEdit();
        workspaceEdit.insert(this.document.uri, new vscode.Position(edit.line, 0), edit.text);
        await vscode.workspace.applyEdit(workspaceEdit);
        this.post({ type: 'applied', mode: 'insert' });
    }

    preview(index) {
        const candidate = this.candidateAt(index);
        const rows = candidateToRows(candidate, this.document.getText());
        this.post({ type: 'preview', index, rows });
    }

    html() {
        const nonce = String(Date.now());
        return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background);margin:0}
main{display:grid;grid-template-columns:minmax(280px,380px) 1fr;gap:12px;padding:12px}
section{border:1px solid var(--vscode-panel-border);border-radius:6px;padding:10px}
h2{font-size:13px;margin:0 0 8px;color:var(--vscode-descriptionForeground);font-weight:600;text-transform:uppercase}
textarea{box-sizing:border-box;width:100%;min-height:260px;font-family:var(--vscode-editor-font-family);font-size:12px;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border);padding:8px}
label{display:grid;gap:3px;font-size:12px;color:var(--vscode-descriptionForeground)}
.controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:10px}
input,select{background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border);padding:4px}
button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;padding:6px 9px;border-radius:3px;cursor:pointer}
button.secondary{background:var(--vscode-button-secondaryBackground);color:var(--vscode-button-secondaryForeground)}
.buttons{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.status{font-size:12px;color:var(--vscode-descriptionForeground);margin-top:8px;white-space:pre-wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
.card{border:1px solid var(--vscode-panel-border);border-radius:6px;padding:8px;background:var(--vscode-sideBar-background)}
.level{font-family:var(--vscode-editor-font-family);white-space:pre;line-height:1.15;overflow:auto;background:var(--vscode-textCodeBlock-background);padding:6px;border-radius:3px}
.meta{font-size:12px;color:var(--vscode-descriptionForeground);margin:6px 0}
.candidate-actions{display:flex;gap:6px;flex-wrap:wrap}
.preview{margin-top:10px}
.hidden{display:none}
</style>
</head>
<body>
<main>
<section>
<h2>Recipe</h2>
<textarea id="spec" spellcheck="false"></textarea>
<div class="buttons" id="presets"></div>
<div class="controls">
<label>Seed<input id="seed" type="number"></label>
<label>Time ms<input id="timeMs" type="number"></label>
<label>Samples<input id="samples" placeholder="time budget"></label>
<label>Jobs<input id="jobs"></label>
<label>Solver ms<input id="solverTimeoutMs" type="number"></label>
<label>Strategy<select id="solverStrategy"><option>portfolio</option><option>bfs</option><option>weighted-astar</option><option>greedy</option></select></label>
<label>Top K<input id="topK" type="number"></label>
</div>
<div class="buttons">
<button id="run">Run</button>
<button class="secondary" id="stop">Stop</button>
<button class="secondary" id="save">Save Recipe</button>
</div>
<div class="status" id="status"></div>
</section>
<section>
<h2>Candidates</h2>
<div id="candidates" class="grid"></div>
<div id="preview" class="preview hidden"></div>
</section>
</main>
<script nonce="${nonce}">
const vscode = acquireVsCodeApi();
let candidates = [];
let candidateRows = [];
let showSolutions = new Set();
const $ = id => document.getElementById(id);
function options(){return {
seed:$('seed').value,timeMs:$('timeMs').value,samples:$('samples').value,jobs:$('jobs').value,
solverTimeoutMs:$('solverTimeoutMs').value,solverStrategy:$('solverStrategy').value,topK:$('topK').value
};}
function setOptions(o){for(const [k,v] of Object.entries(o||{})){if($(k))$(k).value=v;}}
function setStatus(text){$('status').textContent=text||'';}
function renderCandidates(){
 const root=$('candidates'); root.innerHTML='';
 candidates.forEach((c,i)=>{
  const card=document.createElement('div'); card.className='card';
  const rows=(candidateRows[i]||[]).join('\\n');
  const solution=(c.solution||[]).join(' ');
  const hidden=!showSolutions.has(i);
  card.innerHTML='<div class="level"></div><div class="meta"></div><div class="candidate-actions"></div>';
  card.querySelector('.level').textContent=rows;
  card.querySelector('.meta').textContent='#'+c.rank+' score '+c.difficulty_score+' len '+c.solution_length+' sample '+c.sample_id+(hidden?'':'\\n'+solution);
  const actions=card.querySelector('.candidate-actions');
  for(const [label,type] of [['Preview','preview'],['Adopt','adopt'],['Insert','insert']]){
   const b=document.createElement('button'); b.textContent=label; b.className=type==='preview'?'secondary':'';
   b.onclick=()=>vscode.postMessage({type,index:i}); actions.appendChild(b);
  }
  const reveal=document.createElement('button'); reveal.className='secondary'; reveal.textContent=hidden?'Reveal solution':'Hide solution';
  reveal.onclick=()=>{hidden?showSolutions.add(i):showSolutions.delete(i); renderCandidates();}; actions.appendChild(reveal);
  root.appendChild(card);
 });
}
window.addEventListener('message', event=>{
 const msg=event.data;
 if(msg.type==='state'){
  $('spec').value=msg.specText; setOptions(msg.options);
  $('presets').innerHTML='';
  for(const preset of msg.presets){const b=document.createElement('button');b.className='secondary';b.textContent=preset.label;b.onclick=()=>vscode.postMessage({type:'preset',id:preset.id});$('presets').appendChild(b);}
  setStatus((msg.generator.exists?'Generator: ':'Missing generator: ')+msg.generator.path+'\\nRecipe: '+msg.sidecarPath);
 }
 if(msg.type==='preset'){$('spec').value=msg.specText;}
 if(msg.type==='running'){candidates=[];candidateRows=[];renderCandidates();setStatus('Running generator...');}
 if(msg.type==='progress'){setStatus('Running generator...\\n'+JSON.stringify(msg.progress));}
 if(msg.type==='result'){candidates=msg.result.top||[];candidateRows=msg.rows||[];renderCandidates();setStatus('Finished in '+Math.round(msg.elapsedMs/100)/10+'s\\n'+JSON.stringify(msg.result.totals));}
 if(msg.type==='error'){setStatus('Error: '+msg.message);}
 if(msg.type==='stopped'){setStatus('Stopped.');}
 if(msg.type==='saved'){setStatus('Saved recipe: '+msg.sidecarPath);}
 if(msg.type==='applied'){setStatus('Applied candidate ('+msg.mode+').');}
 if(msg.type==='preview'){$('preview').classList.remove('hidden');$('preview').innerHTML='<h2>Preview</h2><div class="level"></div><div class="meta"></div>';$('preview').querySelector('.level').textContent=msg.rows.join('\\n');$('preview').querySelector('.meta').textContent='Solution path is hidden until revealed on the candidate card.';}
});
$('run').onclick=()=>vscode.postMessage({type:'run',specText:$('spec').value,options:options()});
$('stop').onclick=()=>vscode.postMessage({type:'stop'});
$('save').onclick=()=>vscode.postMessage({type:'saveRecipe',specText:$('spec').value});
vscode.postMessage({type:'ready'});
</script>
</body>
</html>`;
    }
}

function presetRules(id) {
    const preset = PRESETS.find(entry => entry.id === id);
    return preset ? preset.rules : PRESETS[0].rules;
}

async function openGeneratorPanel({ context, repoRoot, intelligence }) {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.uri.scheme !== 'file') {
        vscode.window.showWarningMessage('Open a PuzzleScript file before starting the generator.');
        return null;
    }
    if (editor.document.languageId !== 'puzzlescript' && !intelligence.looksLikePuzzleScript(editor.document.getText())) {
        vscode.window.showWarningMessage('Open a PuzzleScript file before starting the generator.');
        return null;
    }
    const level = selectedLevelForLine(editor.document.getText(), editor.selection && editor.selection.active ? editor.selection.active.line : 0);
    if (!level) {
        vscode.window.showWarningMessage('The current PuzzleScript file has no playable level to use as a seed.');
        return null;
    }
    return new PuzzleScriptGeneratorPanel({
        context,
        repoRoot,
        document: editor.document,
        level,
    });
}

module.exports = {
    PuzzleScriptGeneratorPanel,
    normalizeRunOptions,
    openGeneratorPanel,
};
