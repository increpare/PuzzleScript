# PuzzleScript Progress Dashboard

Last verified: 2026-05-07

This is the central project dashboard. Read this before choosing work. It should stay short, current, and test-backed; detailed designs belong in `docs/superpowers/specs/`, implementation plans in `docs/superpowers/plans/`, and historical notes can be deleted or ignored once this file supersedes them.

## How To Choose Work

Default priority order:

1. Fix red gates listed in this file.
2. Fix externally broken user-facing behavior.
3. Advance the active project lane.
4. Turn vague issue-tracker ideas into small, testable tasks.
5. Clean up docs only when they reduce decision friction.

Current active lane: JS solver / static analysis.

Current next task: inspect the focus optimizer output and pick the next JS solver/static-analysis improvement that can reduce search work, not just rewrite compiled state.

## Project Snapshot

The browser JS engine is still the semantic reference implementation. The active solver/generator work is split three ways:

- JS runtime: reference engine, rule-plan metadata consumer, and small runtime optimizations.
- JS test/tools layer: oracle exports, static analysis, JS solver experiments, and solver-only optimization hooks.
- Native C++: main solver, generator, compact state, compact turn codegen, and parity harness.

Current state:

- Native solver smoke is green.
- Native generator smoke is green.
- Full compact solver storage parity is green with the parity target's 1000 ms timeout.
- Compact solver storage smoke parity is green, but it did not exercise compact turn execution in the smoke run.
- Compact turn simulation oracle is green in default interpreter-backed mode.
- Static analyzer tests are green.
- Solver focus comparison is green for the current manifest.
- JS static optimization smoke/focus comparisons are green under a strict A/B gate.
- Generator benchmark is green for the current presets.
- Default compact turn coverage is bridge-only.
- Compiler-mode compact turn codegen coverage is green.
- Generator search is functional, but it is native-generator / VS Code tooling work; defer deeper generator integration while the current focus is JS solver/static analysis.
- GitHub has 59 open issues as of 2026-05-07; most are editor/input/web/runtime cleanup, not native solver/generator roadmap items.

## Project Lanes

| Lane | State | Work Rule |
| --- | --- | --- |
| JS solver / static analysis | Active, green for focused gates | Advance solver optimizer experiments with JS comparison gates before changing runtime semantics. |
| Compiler / runtime correctness | Important backlog | Prefer small repro tests from open issues before changing semantics. Watch `again`, rule expansion, winconditions, parentheticals, and duplicate elimination. |
| Editor / input / web | User-facing backlog | Prioritize issues that make existing games/editor workflows broken: GitHub gist loading, keyboard/controller repeat, paste, mobile undo, CodeMirror problems. |
| Native C++ port | Healthy but nuanced | Keep JS parity as oracle. Treat compact codegen coverage and default bridge coverage as different facts. |
| VS Code extension | Useful side lane | Keep generator/debugger/editor-intelligence changes isolated under `tools/vscode-puzzlescript/` with its own tests. |
| Tests / infra / docs | Support lane | Improve this only when it exposes regressions faster or reduces confusion. Avoid new narrative MD files. |
| Gallery / examples / language polish | Opportunistic | Batch small visible fixes; avoid letting them interrupt red gates unless the site is broken. |

## Fresh Command Evidence

Run from repo root on 2026-05-07.

| Command | Status | Evidence |
| --- | --- | --- |
| `make static_analysis_tests` | PASS | `ps_static_analysis_node: ok`; `static_analysis_explorer_node: ok`; `solver_static_opt_node: ok`; `compare_solver_static_opt_runs_node: ok`. Fixed count-invariant object-write detection, protected compiler background objects from cosmetic pruning, and made the static-optimization comparator fail on status/solution regressions. |
| `make solver_smoke_tests` | PASS | `solver_smoke_assert passed cases=7`. |
| `make solver_compact_parity` | PASS | `games=153/153 levels=2679 random_excluded=31`, `compact_timeout_regressions=0`, `compact_turn_attempts=0`, with `timeout_ms=1000`. |
| `make solver_compact_parity_smoke` | PASS | `games=5/5 levels=7`, `compact_timeout_regressions=0`. Caveat: `compact_turn_attempts=0`, so this verifies compact node storage parity, not compact turn kernel execution. |
| `make generator_smoke_tests` | PASS | All smoke checks passed, ending with `generator_smoke passed`. |
| `make compact_turn_simulation_tests` | PASS | `cpp_simulation_tests_direct passed=469 failed=0 total=469`, `compact_turn_oracle_checks=16554`, `compact_turn_oracle_failures=0`, `turn_executor=interpreter`. |
| `make solver_focus_compare` | PASS | `targets: interpreted=50 compiled=50 same=yes`; both statuses `{"solved":50}`; median elapsed compiled/interpreted `0.911x`. |
| `make generator_benchmark` | PASS | `sokoban_room_scatter.gen` solved `12` in each of 3 runs at ~46-49 samples/s; `sokoban_transform_pairs.gen` solved `1` in each of 3 runs at ~37.5 samples/s; wrote `build/native/generator_benchmark.json`. |
| `make compact_turn_coverage` | PASS | `callable_compact_backends: 452/452`, `native_compact_kernels: 0/452`, `interpreter_bridge_backends: 452/452`. |
| `make compact_turn_codegen_coverage` | PASS | `callable_compact_backends: 452/452`, `native_compact_kernels: 452/452`, `interpreter_bridge_backends: 0/452`. |
| `make compact_turn_codegen_bringup` | PASS | `observed compiler-mode attempts=18 hits=18 unhandled=0`. |
| `make js_static_optimization_comparison_solver_smoke` | PASS | Baseline and `--solver-opt all` both report `levels=7`, `solved=5`, `exhausted=1`, `skipped_message=1`, `errors=0`; optimized removed no rules/objects on this smoke corpus. |
| `make js_static_optimization_comparison_solver_focus` | PASS | Baseline and `--solver-opt all` both report `levels=50`, `solved=50`, `errors=0`; optimized removed `12` inert rules, `26` cosmetic objects, `7` empty layers, and merged `22` aliases across `7` groups. Expanded/generated counts are currently unchanged, so this validates safety more than speed. |

## Recently Cleared Gates

`make static_analysis_tests` no longer blocks the static analyzer / JS solver optimizer lane.

Cleared symptom:

- The count-invariant replay test for `Crates move when you move.txt` expects `Player`, `Wall`, and `Crate` to remain in the count-invariant replay set.
- The analyzer was rejecting them.

Local cause:

- A rule such as `[ > crate | no obstacle ] -> [ | > crate ]` expands `no obstacle` into absent `Player`, `Crate`, and `Wall` guards.
- Count preservation used broad "mentions or writes this object" logic, so absent guards and pure adjacent movement were treated as count writes.

Current fix:

- Count-invariant facts now use rule-flow object writes instead of object mentions.
- Added a regression for adjacent movement guarded by absence.

`make js_static_optimization_comparison_solver_focus` is green again and now fails if an optimized run silently drops levels or changes solved status.

Cleared symptom:

- The optimized focus run compiled only `46` level rows, solved `43`, and reported `3` compile errors while the comparator still exited successfully.
- The compile errors were `Seriously, you have to define something to be the background.`

Local cause:

- The solver-only cosmetic pruning pass deliberately omits level-map references, which is fine for decorative tiles.
- It did not separately protect the compiler's structural background object/layer, so lowercase/custom background objects could be deleted.

Current fix:

- Cosmetic pruning now reserves `player`, the resolved `backgroundid`, and every object on the resolved `backgroundlayer`.
- Added a regression where a cosmetic-tagged lowercase `background` must survive optimization.
- The A/B comparator now fails on total status mismatches, missing/extra level rows, per-level status changes, and solved solution-length changes.

## Architecture Map

Reference JS path:

- `src/js/compiler.js` compiles PuzzleScript and calls `pluginOptimizationHook(state)` before `rulesToMask(state)`.
- `src/js/engine.js` still owns rule execution, row/column masks, tuple application, movement resolution, and `again` behavior.
- `rulePlanMetadata` is present on runtime `Rule` objects, but only a small amount is consumed by the JS runtime.

JS solver/tooling path:

- `src/tests/run_solver_tests_js.js` is the JS solver experiment harness.
- `src/tests/ps_static_analysis.js` emits the static analysis report used by optimizer experiments.
- `src/tests/solver_static_opt.js` provides opt-in solver-only passes: inert rule removal, cosmetic object deletion, and merge candidates.
- Current status: focused gates are green; treat optimizer behavior as experimental until broader post-static gates pass.

Native solver path:

- `native/src/solver/main.cpp` is the main solver path.
- It supports normal full-state solving, compact node storage, compact-state hashing/equality, and a `trySpecializedCompactTurn` path.
- Solver mode uses `AgainPolicy::Drain` so graph nodes are settled end-of-turn states.

Native compact turn path:

- `native/src/runtime/compiled_rules.hpp` and `.cpp` define the compact turn ABI and interpreter bridge.
- `native/src/compiler/compact_turn_codegen.cpp` emits compiler-mode compact turn kernels.
- The default compact coverage target currently reports bridge backends only.
- The codegen coverage target reports full native compact kernel coverage for the current testdata corpus.

Native generator path:

- `native/src/generator/main.cpp` samples levels, dedupes candidates, solves/scored them, and retains top-K candidates.
- Generator smoke is green.
- Generator solving still clones/materializes `FullState` nodes in its own search loop. Keep this separate for now; the editor-facing "JS generator" is a VS Code/native-generator wrapper rather than a separate JS generation engine.

## What The Coverage Means

`compact_turn_coverage` and `compact_turn_codegen_coverage` answer different questions:

- `compact_turn_coverage`: Can the compact ABI be called for every testdata source using the default coverage mode? Yes, but currently all via interpreter bridge.
- `compact_turn_codegen_coverage`: Can compiler-mode codegen emit native compact kernels for every testdata source? Yes, for the current corpus and options.
- `make help` now spells out this distinction: default compact ABI coverage may count bridge backends, while compiler-mode coverage means native compact kernels.

So the useful statement is not "compact turn is done." The useful statement is:

> The ABI is callable for the corpus, and compiler-mode compact kernels can be emitted for the corpus, but the default coverage path remains bridge-only.

## Open Issue Alignment

Open GitHub issues do not currently track the native solver/generator roadmap well.

Directly solver/generator-adjacent:

- `#1096` "solver" test suite.

High-signal compiler/runtime/optimization issues:

- `#1144` object-major matrix with object counts.
- `#1140` `collectRuleMentionedObjects` wincondition bug.
- `#1136` multiple `X no X` cells causing unexpected rule deletion.
- `#1128` bad diagnostic for parentheticals inside rules.
- `#1123` missing cross-group duplicate elimination tests.
- `#1086` movement expansion question for `[ moving player ] -> again`.
- `#1019` unhelpful `again` loop diagnostics.
- `#945`, `#737`, `#720`, `#193`, `#163` older optimization notes.

High-signal editor/input/web issues:

- `#1117` public gist loading / gallery breakage.
- `#1137`, `#1111`, `#1102`, `#1101` keyboard/controller repeat and held-input behavior.
- `#1116` repeatable mobile undo.
- `#1125`, `#947`, `#695` CodeMirror/editor scrolling/highlighting upgrade pressure.
- `#1093`, `#984`, `#931` paste/text editing on browsers and iPad.

## Next Moves

1. Advance JS solver/static optimizer work:
   - Inspect the focus optimizer output: current passes remove objects/rules/aliases safely, but do not reduce expanded/generated counts.
   - Choose the next small optimizer/static-analysis change that should affect search work, then guard it with the strict smoke/focus A/B comparisons.

2. Pick one user-facing backlog item only after the JS solver/static optimizer lane has a useful next result.
   - First candidate: `#1117`, because broken game loading/gallery behavior is higher impact than editor niceties.

## Maintenance Rule

Keep this file ruthless:

- Update command evidence when gates change.
- Keep only the next few project moves.
- Link or name issues; do not paste full issue bodies.
- Move design arguments to specs or plans.
- Delete or demote any MD file that competes with this dashboard.
