# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PuzzleScript is an open-source HTML5 puzzle game engine. Users write games in a custom PuzzleScript language using the browser-based editor, and the engine compiles and runs them. Live at https://www.puzzlescript.net.

## Commands

**Install dependencies:**
```
npm install
```

**Build (compiles src/ → optimized bin/):**
```
node compile.js
```

**Run tests:**
```
node src/tests/run_tests_node.js
```

**Run tests with filter:**
```
node src/tests/run_tests_node.js "test name substring"
```

**Run tests verbose (see console output):**
```
node src/tests/run_tests_node.js --verbose
```

**Run tests with profiling (6 cold runs):**
```
node src/tests/run_tests_node.js --profile
```

**Run tests with timing breakdown:**
```
node src/tests/run_tests_node.js --breakdown
```

**Run only simulation tests (skip error message tests):**
```
node src/tests/run_tests_node.js --sim-only
```

Note: all commands run from the repository root, not `src/`.

## Architecture

### Compilation Pipeline

PuzzleScript source code flows through three stages:

1. **Parser** (`src/js/parser.js`) — tokenizes PuzzleScript game source text
2. **Compiler** (`src/js/compiler.js`) — transforms parsed output into executable game state (objects, rules, collision layers, levels)
3. **Engine** (`src/js/engine.js`) — executes the game loop: processes input, applies rules, handles movement/collision, manages win conditions

### Key Modules

- **`globalVariables.js`** — shared mutable state (current level, timing, game flags). Includes lazy function generation system for deferred CellPattern/Rule match function compilation.
- **`languageConstants.js`** — PuzzleScript language keywords and constants
- **`bitvec.js`** — bit vector operations used throughout for efficient object/cell state representation
- **`level.js`** — level data structure
- **`graphics.js`** — sprite rendering to canvas
- **`font.js`** — custom font rendering system
- **`inputoutput.js`** — handles keyboard/touch input and game I/O
- **`sfxr.js` / `riffwave.js`** — procedural sound effect synthesis
- **`colors.js`** — named color palette

### Entry Points

- **`src/editor.html`** — the PuzzleScript editor (code editor + game player)
- **`src/play.html`** — standalone game player
- **`src/standalone.html`** — template for exported standalone games (compile.js inlines resources into `standalone_inlined.txt`)

### Script Loading Order

Scripts are loaded as individual `<script>` tags in the HTML files and share global scope. The Node.js test runner (`run_tests_node.js`) concatenates them into a single script to replicate this behavior. The load order matters — see the `sourceFiles` array in `run_tests_node.js` for the canonical order.

### Build System

`compile.js` copies `src/` to `bin/`, then: optimizes images (pngcrush/gifsicle), concatenates and minifies CSS (ycssmin), minifies JS into two bundles via terser (`scripts_compiled.js` for editor, `scripts_play_compiled.js` for player), inlines resources into the standalone template (web-resource-inliner), minifies HTML, and generates gzip versions.

## Testing

Two types of tests, both using QUnit:

1. **Play session tests** (`src/tests/resources/testdata.js`, ~4MB) — records a game's source + input sequence + expected end state. Verifies the engine produces deterministic output.
2. **Error message tests** (`src/tests/resources/errormessage_testdata.js`, ~1MB) — records compilation errors/warnings for game source. New errors are OK; old errors must still be present. If you change error message wording, update the test data to match.

**Solver corpus runner** (`node src/tests/run_solver_tests_js.js`): optional solver-only static optimizations (`src/tests/solver_static_opt.js`) — `--solver-optimize-static` (inert command-only rules), `--solver-opt inert,cosmetic,merge` or `all`, and `--solver-opt-parity` to fail on baseline vs optimized solver mismatch. Quick check: `node src/tests/solver_static_opt_node.js`. Per-game HTML/JSON summary (two full corpus runs): `make static_optimizer_page` (see `STATIC_OPTIMIZER_PAGE_*` in the Makefile).

**Generating test data:** In the editor, compile/launch a game, then press Ctrl/Cmd+J to generate test data in the browser console.

## Development Notes

- Source files in `src/` are the raw, uncompressed, directly-runnable version. The `bin/` directory is generated output — edit only in `src/`.
- No linter or formatter is configured.
- Browser globals are used extensively — the codebase predates ES modules.
- The test runner provides browser shims (document, localStorage, window) to run in Node.js.
- Standalone export requires a local HTTP server when testing from `src/editor.html` directly (browser sandboxing blocks XMLHttpRequest for local files).
