# PuzzleScript VS Code Extension

This extension reuses PuzzleScript's existing JavaScript editor logic for VS Code editor features.

## Features

- Semantic tokens from `src/js/parser.js`.
- Completion items from `src/js/codemirror/anyword-hint.js`.
- Rule rotation/mirroring completions from `src/js/codemirror/rule-transform.js`.
- Dynamic decorations for PuzzleScript color tokens, including sprite matrix color indices.
- Parser diagnostics from the shared JavaScript parser error state.

## Development

By default, the extension expects to live at `tools/vscode-puzzlescript` inside the PuzzleScript checkout. If it is moved elsewhere, set `puzzlescript.repoRoot` to the PuzzleScript repository root.

```sh
npm test
```

## Testing Highlighting

Run the automated syntax highlighting checks:

```sh
npm run test:syntax
```

These tests use the same shared parser adapter as the extension and assert that representative PuzzleScript samples produce the expected semantic token classes and dynamic color decorations.

You can also render a small HTML preview of the highlighter output:

```sh
npm run preview:syntax -- ../linguist/samples/simple-block-pushing.puzzlescript /tmp/puzzlescript-highlight.html
```

Open the generated HTML file in a browser to inspect the token colors. This preview is not VS Code itself, but it is a useful smoke test for the shared highlighter layer that feeds VS Code semantic tokens and color decorations.

## Running In VS Code

### Option 1: Extension Development Host

Use this when editing or testing the extension.

1. Open `tools/vscode-puzzlescript` in VS Code.
2. Open the Run and Debug panel.
3. Choose `Run PuzzleScript Extension`, or press `F5`.
4. A second VS Code window opens with the extension loaded.
5. In that second window, open a `.puzzlescript` or `.ps` file.

If VS Code says there is no Markdown extension to debug, it is trying to debug the README instead of the extension. Make sure the whole `tools/vscode-puzzlescript` folder is open as the VS Code workspace, then pick `Run PuzzleScript Extension` from the Run and Debug dropdown.

If you opened the extension folder from inside this repository, no extra settings are needed. If VS Code cannot find the PuzzleScript source files, set this in the second window's settings JSON:

```json
{
  "puzzlescript.repoRoot": "/absolute/path/to/PuzzleScript"
}
```

### Option 2: Install Locally

Use this when you want the extension available in your normal VS Code window.

1. Install VS Code's packaging tool if you do not already have it:

```sh
npm install -g @vscode/vsce
```

2. From this folder, package the extension:

```sh
vsce package
```

3. Install the generated `.vsix` file:

```sh
code --install-extension puzzlescript-vscode-0.1.0.vsix
```

4. Restart VS Code and open a `.puzzlescript` or `.ps` file.

If the extension was packaged from this repository, it should find the shared PuzzleScript JS files via the bundled relative path. If you move the extension or source checkout, set `puzzlescript.repoRoot` to the PuzzleScript repository root.

### File Types

`.puzzlescript` and `.ps` files are registered as PuzzleScript automatically. `.txt` files are handled only when they look like PuzzleScript source, but VS Code may still show them as plain text; use `Change Language Mode` and choose `PuzzleScript` if needed.

The extension intentionally does not use Tree-sitter or the C++ compiler in v1; the browser editor's JavaScript parser and autocomplete code are the source of truth.
