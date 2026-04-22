# C++ PuzzleScript frontend (parser + compiler)

Status: design approved, pending per-phase implementation plans.
Date: 2026-04-22.
Supersedes: the "port compilation to C++" follow-on tracked in `2026-04-22-native-perf-crush-design.md` §Out-of-scope.

## 1. Goals, scope, and non-goals

### Primary goal

Eliminate the node.js dependency from the native runtime's compilation path. `ps_cli` must be able to accept a puzzlescript source file (current extension: `.txt`) and produce an in-memory `puzzlescript::Game` ready for the existing runtime — with **no subprocess, no intermediate `ir.json` on disk, no node involvement of any kind.**

### Secondary goal

Ship the parser + compiler as a reusable C++ frontend library (`libpuzzlescript-frontend`) behind a clean C API, so that future tooling (LSP server, VSCode extension shipping a native binary, TextMate grammar reference, GitHub Linguist submission) is a feasible follow-on without another rewrite.

### Fidelity bar

**Strict behavioral parity with the existing JS parser + compiler.** Specifically:

- **Success cases:** the C++ `Game` object must be byte-for-byte equivalent (after canonical serialization) to the JS `ir.json` produced from the same source, for every fixture in the repo corpus.
- **Error cases:** C++ diagnostics must match JS diagnostics in **count**, **order**, **severity**, **line number**, and **text** (after stripping HTML tags and normalizing whitespace). The JS message wording is the spec, including its bugs and quirks. Matching JS bug-for-bug is explicitly allowed and expected.

### Non-goals

- **Porting the browser editor's parser.** The JS parser in `src/js/parser.js` doubles as a CodeMirror mode and stays that way forever. The browser editor keeps its JS parser.
- **WASM.** Not a target, not a concern. The C++ frontend is a native library only. If someone later wants to run it in a browser they can port it themselves, but we don't shape the design around that possibility.
- **Making compilation dramatically faster.** Nice if it happens, not a goal. See §7.
- **Changing the runtime, the `Game` struct layout, or any observable engine semantics.** The C++ frontend is a new producer of the existing `Game` type; consumers (the engine core, `cli_main.cpp`) are untouched.
- **Adopting the C++ frontend in the browser.** See "non-goals: browser editor" above. We intentionally do not support this use case.

### In scope

- New library in `native/src/frontend/` producing `puzzlescript::Game`.
- New public header `native/include/puzzlescript/frontend.h` exposing a C API.
- New `ps_cli` subcommand(s) to drive compilation from source.
- Validation harness (differential testing against the JS reference).
- One-time instrumentation of the JS compiler to emit phase snapshots for debugging.

## 2. Architecture

### 2.1 Three-phase pipeline

The JS compiler progressively mutates one `state` object through ~15 passes. C++ splits this into three typed phase boundaries:

```
source.txt
   │
   ├── [parse]  ─────► ParserState    (syntax-level: tokens, raw rule tokens, raw level grids,
   │                                   section tables; no semantic resolution yet)
   │
   ├── [lower]  ─────► LoweredProgram (semantic: legend-resolved object references, direction-
   │                                   expanded rules, concrete properties, atomized aggregates,
   │                                   structurally canonical but not yet mask-compiled)
   │
   └── [compile]────► Game           (runtime-ready: masks, rule groups, win conditions, sound
                                      tables — the existing puzzlescript::Game struct)
```

Phase boundaries are real C++ type conversions. Within a phase, we use free functions that mutate the phase's struct — same incremental-build-up style as the JS compiler, just scoped to one phase at a time.

### 2.2 Diagnostics

Structured `Diagnostic` objects are collected in a `DiagnosticSink` passed through all three phases:

```cpp
struct Diagnostic {
    Severity severity;             // Error | Warning | Info | LogMessage
    DiagnosticCode code;           // enum, stable across JS/C++
    std::optional<int> line;       // 1-based, to match JS
    DiagnosticArgs args;           // structured, variant-of-variants
};
```

The formatter `format_for_js_compat(const Diagnostic&) -> std::string` produces the exact text the JS compiler would produce after `stripHTMLTags`. Diagnostic codes + args are the source-of-truth; text is a presentation concern.

This structure is what makes an LSP server feasible later: LSP consumers want structured data (code, range, severity, message, related info), not stripped text.

### 2.3 Public C API

`native/include/puzzlescript/frontend.h` exposes:

```c
typedef struct ps_frontend_result ps_frontend_result;

ps_frontend_result* ps_frontend_compile_source(const char* source, size_t source_len);
ps_game*            ps_frontend_result_game(const ps_frontend_result*);  // null on fatal error
size_t              ps_frontend_result_diagnostic_count(const ps_frontend_result*);
const ps_diagnostic* ps_frontend_result_diagnostic(const ps_frontend_result*, size_t index);
void                ps_frontend_result_free(ps_frontend_result*);
```

The C API hides C++ types entirely. The result owns its `ps_game` and diagnostics until freed. This shape keeps future LSP/CLI consumers unblocked.

### 2.4 Library and target layout

- New CMake target `puzzlescript-frontend` (static library). Depends on nothing outside `native/` — in particular, no simdjson dependency (simdjson stays scoped to `ir.json` loading in `cli_main.cpp`).
- `ps_cli` links `puzzlescript-frontend` and exposes new subcommands.
- The existing `puzzlescript-core` runtime library is unchanged and unaware of the frontend.

## 3. Data types

### 3.1 `ParserState` (frontend/types/parser_state.hpp)

Captures what a CodeMirror-style tokenizing pass produces: section headers, raw token streams per rule/legend line, grid literals per level, sound-line tuples. No name resolution — every identifier is still a raw string. Line numbers are preserved on every element.

Design guideline: `ParserState` should be convertible to a stable JSON form (`parser_state.json`) for the validation harness. This serializer lives in a dev-only translation unit (see §4.2).

### 3.2 `LoweredProgram` (frontend/types/lowered_program.hpp)

Semantic canonical form: every object identifier is resolved against the legend; direction-qualified patterns are expanded to concrete directions; `no` markers, random selectors, and property/aggregate references are resolved per JS semantics; rules are in a flat canonical list; levels are concrete grids of object-mask placeholders (or the final masks if convenient — implementation detail).

Design guideline: boundary between "syntax-ish" and "semantics-ish". After lowering, there are no more string-to-object lookups; everything is a typed handle.

### 3.3 `Game` (existing — frontend/types/game.hpp re-exports `puzzlescript::Game`)

The existing runtime type in `native/src/core.hpp`. The compile phase is the only producer of this type in the new pipeline. **No new fields are added for the frontend's benefit.** If the compile phase needs scratch state it lives in compile-phase locals, not on `Game`.

## 4. The pipeline — in-memory vs. serialized

### 4.1 The actual pipeline is 100% in-memory

`ps_frontend_compile_source` reads a source string and returns a `ps_game*`. There is no filesystem involvement, no JSON, no intermediate textual form. This is the whole point.

### 4.2 JSON serializers are dev-only artifacts

Three serializers exist **only for the validation harness and for debugging**:

- `parser_state_serialize.cpp` — `ParserState → parser_state.json`
- `lowered_program_serialize.cpp` — `LoweredProgram → lowered_program.json`
- `ir_serialize.cpp` — `Game → ir.json` (matching the existing `src/tests/lib/puzzlescript_ir.js` format byte-for-byte)

These live behind a CMake option (`-DPS_ENABLE_DEV_SERIALIZERS=ON`) and are not linked into shipping builds. They are not on any runtime path and have no performance budget.

The `ir.json` format continues to exist as a documented format for ad-hoc use and for cross-implementation diffing, but it is not part of the primary pipeline.

## 5. Validation harness

### 5.1 Corpus

The 730+ existing fixtures in:
- `src/tests/resources/testdata.js` (~457 simulation tests with inline game sources)
- `src/tests/resources/errormessage_testdata.js` (~273 error-message tests)
- `demo/` and `examples/` directories as a secondary corpus

...are the full parity target. We add one CI job per diff kind.

### 5.2 Diff kinds

**Final-struct diff (the headline gate).** For each successfully-compiled fixture:
1. Run JS compiler → `ir.json` (JS).
2. Run C++ compiler → serialize `Game` to `ir.json` (C++).
3. Canonicalize both (sort keys, normalize numbers, etc. — see §5.3).
4. Assert byte-for-byte identical.

**Diagnostic diff.** For each fixture that emits diagnostics (whether it compiles successfully with warnings, or fails outright):
1. Run JS compiler → capture all logged messages, strip HTML, trim trailing whitespace per line.
2. Run C++ compiler → format diagnostics with `format_for_js_compat`.
3. Canonicalize both (strip HTML, normalize whitespace).
4. Assert the two diagnostic streams are identical (count, order, text).

**Phase-snapshot diff (dev-time debugging aid only).** When a final-struct diff fails, we need to localize the divergence. The JS compiler is instrumented (once) to emit `parser_state.json` and `lowered_program.json` at matching phase boundaries, behind a `PSX_SNAPSHOT=1` env var. A script `scripts/diff_phase_snapshots.sh <source.txt>` runs both and diffs the earliest-diverging phase. This is developer tooling, not a CI gate.

### 5.3 Canonicalization rules

Identical to the rules in `2026-04-22-native-perf-crush-design.md` §canonicalization:
- JSON: object keys sorted; numbers formatted identically (integer `0`, not `0.0`; trailing-zero float rules match JS `JSON.stringify`); arrays in source order; all whitespace normalized to `\n` + 2-space indent.
- Diagnostic text: HTML tags stripped (`<br>` → `\n`, all others dropped); leading/trailing whitespace per line trimmed; blank lines collapsed.

Canonicalization happens on both sides before diffing, not only on C++ output. This protects us from JS quirks (e.g., `0 vs. 0.0`) being treated as C++ bugs.

### 5.4 Scripts

- `scripts/diff_ir_against_js.sh <source.txt>` — runs both pipelines, canonicalizes, diffs `Game`. Also works on whole corpus with `--corpus`.
- `scripts/diff_diagnostics_against_js.sh <source.txt>` — same shape for diagnostics.
- `scripts/diff_phase_snapshots.sh <source.txt>` — dev-only phase-level diff.

All three are shell scripts driving `ps_cli` and `node src/tests/export_ir_json.js`. CI wires the first two on the full corpus; the third is manual.

## 6. Delivery phases

Four phases, each a standalone spec → plan → implementation → PR. Phases 1–3 merge as dev-only tooling (behind `-DPS_ENABLE_DEV_SERIALIZERS=ON` and new dev-only `ps_cli` subcommands). P4 is the user-visible cutover.

### P1 — C++ parser + validation harness

**Builds:** `source → ParserState`, plus all shared infrastructure.

Deliverables:
- `native/src/frontend/parser.{hpp,cpp}`, `native/src/frontend/types/parser_state.hpp`, `native/src/frontend/language_constants.hpp`.
- `native/src/frontend/diagnostic.{hpp,cpp}` — `Diagnostic`, `DiagnosticCode`, `DiagnosticSink`, `format_for_js_compat`.
- Dev-only `parser_state_serialize.cpp`.
- `src/tests/export_ir_json.js` extended with `--snapshot-phase parser` to emit JS `parser_state.json`.
- `scripts/diff_parser_state_against_js.sh` (whole-corpus runner).
- `ps_cli compile-source --emit-parser-state` (dev build only).
- CMake target `puzzlescript-frontend` (static library, initially parser-only).

Gate to merge: 100% fixture parity on `ParserState`; 100% parity on parser-phase diagnostics.

Estimated size: ~1,700 LOC of JS parser logic + ~800 LOC of shared infrastructure (diagnostic sink, formatter, language constants, C API skeleton).

### P2 — C++ lowerer

**Builds:** `ParserState → LoweredProgram`. Mirrors JS `rulesToArray`, `levelsToArray`, `resolveDictionaryCrossReferences`, legend-resolution passes, direction expansion, property concretization, movement concretization, synonym rewrites, aggregate atomization, RHS-conflict / LHS-negation-trimming / coincidence checks.

Deliverables:
- `native/src/frontend/lower.{hpp,cpp}`, `native/src/frontend/types/lowered_program.hpp`.
- Each JS pass becomes a C++ free function over `LoweredProgram`.
- Dev-only `lowered_program_serialize.cpp`.
- JS `--snapshot-phase lowered` flag.
- `scripts/diff_lowered_program_against_js.sh`.
- `ps_cli compile-source --emit-lowered-program` (dev build).

Gate to merge: 100% fixture parity on `LoweredProgram`; 100% diagnostic parity cumulative across parser + lowerer.

Estimated size: largest phase. ~2,500–3,000 LOC across `rulesToArray` and dependents, direction/property/movement concretization, aggregate atomization.

### P3 — C++ compiler

**Builds:** `LoweredProgram → Game`. Mirrors JS `generateMasks`, `rulesToMask`, `arrangeRulesByGroupNumber`, `collapseRules`, `generateRigidGroupList`, `processWinConditions`, `checkObjectsAreLayered`, `twiddleMetaData`, `generateLoopPoints`, `generateSoundData`, `formatHomePage`.

Deliverables:
- `native/src/frontend/compile.{hpp,cpp}`.
- Dev-only `ir_serialize.cpp`.
- `scripts/diff_ir_against_js.sh` (the §5.1 headline gate).
- `ps_cli compile-source --emit-ir` (dev build).

Gate to merge: 100% fixture parity on final `Game` (serialized to `ir.json`, byte-identical after canonicalization); 100% cumulative diagnostic parity.

**After P3 the C++ frontend is feature-complete.** The runtime still consumes `ir.json` from disk — no user-visible change yet.

Estimated size: ~1,500 LOC, largely mechanical bit-twiddling over `state.rules` and `state.objectMasks`.

### P4 — Native integration cutover

**Delivers the kill-node promise.**

Deliverables:
- `ps_cli` subcommands promoted from dev-only to production: `ps_cli run <source.txt>`, `ps_cli check-trace <source.txt> <trace.json>`, `ps_cli check-trace-sweep <manifest>` (where manifest lists `(source.txt, trace.json)` pairs, no `ir.json`).
- Backward compat: existing `ir.json`-consuming surface continues to work on the same binary.
- `src/tests/run_native_trace_suite.js` and `src/tests/export_native_fixtures.js` updated to pass puzzlescript source directly to `ps_cli`, no round-trip through `export_ir_json.js`.
- Performance baseline regenerated; new metrics `compile_source_ms` and `build_ir_cache_ms` added to `perf_baseline.json` (see §7).
- README / CLAUDE.md / AGENTS.md updated to note native test runs no longer need node.

Gate to merge: full trace suite (simulation + error) passes on the source-direct path; perf baseline met.

Estimated size: ~300–500 LOC of wiring and script updates.

### Rollout discipline

- Each phase is its own PR landing on `main`. No long-lived feature branches.
- Phases 1–3 are shipped as dev-only tooling (behind CMake flag and dev-only subcommands). Zero user-visible behavior change.
- If any phase's parity gate drags past estimate, we stop and reassess scope before plowing ahead. The "kill node" delivery does not need to land by a specific date.

## 7. Performance goals

### Expected wins

- **Fixture pre-computation.** `npm run build-ir-cache` / `export_ir_json.js` becomes a fast native loop. Speculative estimate: 5–20× on the full corpus.
- **Eliminating `ir_miss_ms`** as a concept in the native runtime, replaced by `compile_source_ms` that also subsumes tokenization + lowering. Should beat the existing `ir.json` + simdjson path since we skip the JSON round-trip entirely.
- **CI / dev-loop bring-up** — no node required for the native test path.

### Non-goals

- **Steady-state simulation speed.** Unchanged. `fast_replay_ms` must not regress.
- **Beating the hot-cache `ir.json` path by a wide margin.** Nice if it happens.

### Perf gates added in P4

New entries in `perf_baseline.json`:

- `compile_source_ms` — mean source→`Game` latency across the corpus. **Gate: ≤ existing `ir_miss_ms` value.** Being slower than simdjson-loading a cached `ir.json` is a regression.
- `build_ir_cache_ms` — wall-clock to rebuild full `ir.json` cache from sources. **Gate: ≥ 3× speedup vs. current node-based path.** Conservative floor; realistic target is higher.

Existing metrics (`fast_replay_ms`, `trace_json_parse_ms`) keep current baselines and must not regress.

### Explicitly not gated

- Memory usage (as long as peak RSS <100 MB on the largest fixture).
- Binary size of `puzzlescript-frontend.a`.
- Internal phase timings — only aggregate `compile_source_ms` matters.

## 8. Risks and open questions

### Risks

**R1 — JS parser has implicit behavior we don't know about.** The 700-fixture corpus is the mitigation; differential testing is the gate. Parity breaks are C++ bugs, not JS bugs — we match JS bug-for-bug.

**R2 — The JS "gradual build-up" doesn't cleanly split into exactly three phases.** Phase boundaries are renegotiable during implementation; what's non-negotiable is corpus-level parity on the final `Game`. If a pass naturally belongs in a different phase than we guessed, move it.

**R3 — Diagnostic text parity is hard because JS uses HTML tags.** §5.3 canonicalization strips tags on both sides; structured `Diagnostic` objects let us produce text faithfully. Per-fixture divergences get resolved case-by-case (fix the formatter, or tighten canonicalization).

**R4 — Effort underestimated; phases drag.** Each phase is independently shippable (as dev-only tooling) and has a scope-reassessment checkpoint. Worst case: ship P1+P2, pause before P3.

**R5 — Some JS pass uses a feature we can't cleanly express in C++.** Confirmed during exploration that `ir.json` does not serialize generated JS functions — the engine is data-driven. If a counterexample appears during implementation, raise it and re-plan.

### Resolved questions

- **C++ language target:** C++17, no bump.
- **Error recovery:** match JS exactly — count, order, text. Non-negotiable (§1 fidelity bar).
- **LSP / TextMate grammar / Linguist:** remain feasible future consumers; C API stays clean; we don't build any of them here.
- **WASM:** not a target. Not now, not later.
- **Browser editor parser:** stays on JS forever. No plan to unify.

### Deferred to per-phase specs

- **Memory model for `ParserState` / `LoweredProgram`.** Pick a style in P1 (owning values, arena, or `unique_ptr` soup). P2 and P3 follow the same style.
- **Targeted unit tests in addition to the corpus.** Each phase's plan decides where corpus coverage is gappy. Default: corpus-only, add targeted tests where needed.

## 9. Plan-writing checklist (for follow-on work)

Before writing the P1 plan, the plan-writer should:
- Confirm the exact set of JS parser behaviors to mirror by re-reading `src/js/parser.js` end-to-end.
- Pick the `ParserState` memory-model style (see §8 "Deferred").
- Draft the `DiagnosticCode` enum covering every distinct error string in `src/js/compiler.js` + `src/js/parser.js` that can fire during the parser phase.
- Identify and list JS parser quirks (off-by-ones, implicit state, regex behaviors) the corpus will enforce but that deserve a comment in the C++ port.

Before writing the P2 plan:
- Inventory every JS pass between "parse complete" and "mask generation starts" in `src/js/compiler.js` and categorize each as parser/lower/compile.
- Decide on the `LoweredProgram` shape — specifically how direction expansion and property concretization are represented.

Before writing the P3 plan:
- Confirm the exact `ir.json` serialization format by reading `src/tests/lib/puzzlescript_ir.js` in full.
- Enumerate every `Game` field populated by JS compilation and confirm a C++ compile-phase source for each.

Before writing the P4 plan:
- Enumerate every current `ps_cli` subcommand that consumes `ir.json` and define the source-direct equivalent.
- Identify every shell script, npm script, and CI job that invokes `node src/tests/export_ir_json.js` and plan the migration.
