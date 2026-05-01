# Whole-Game Compilation Plan

## Goal

The long-term goal is to compile a PuzzleScript game into a small, per-game
execution path: conceptually, a `turn(state, input) -> state` function that can
drive solvers, generators, embedded builds, and other tight loops without
repeatedly interpreting the full runtime data model.

The first target is C++ generation. C, LLVM, and assembly are interesting later
targets, but only after the C++ whole-turn path has proved parity and speed.

## Current State

Specialized rulegroups already generate C++ kernels for rulegroup application.
The interpreted runtime remains the correctness oracle and the default safe
executor when generated whole-turn execution is not explicitly selected.

The simulation corpus has 469 cases, deduplicated to 452 unique source texts for
specialized-rulegroup coverage. With:

```sh
make compiled_rules_simulation_suite_coverage COMPILED_RULES_MAX_ROWS=99
```

the current specialized-rulegroup backend covers 452/452 unique simulation sources. The
default `COMPILED_RULES_MAX_ROWS=1` is still intentionally conservative: it is
an iteration-time and generated-code-size knob, not a semantic boundary.

Solver and generator builds can already opt into specialized rulegroup code:

```sh
make solver_smoke_tests SPECIALIZE=true
make generator_smoke_tests SPECIALIZE=true
```

The solver corpus is now large and intentionally active. It is useful as a
source of candidate levels, but it is too bulky for every optimization loop. The
current solver/generator performance work therefore uses a mined focus group:

```sh
make solver_focus_mine
make solver_focus_compare
make solver_focus_perf_report
```

At the current checkpoint, the local focus manifest has 44 targets from 35
games, after excluding clang-heavy generated-source outliers. Focus
specialization defaults to `SOLVER_FOCUS_COMPILED_RULES_MAX_ROWS=99`, while the
global specialized-rulegroup default remains conservative.

Generated sources also export compact whole-turn backends. Compiler-mode
compact turns now pass the full `testdata.js` simulation corpus through the
compact oracle path: 469/469 cases pass, with zero compact-oracle failures.

The runtime state model has converged on one authoritative board:
`PersistentLevelState::board.objects`, a packed cell-major object-mask grid.
The interpreter, generated rulegroups, generated compact turns, solver keys,
C API reads, restart/checkpoint, and oracle export all use that same persistent
board. `Scratch::objectCellBits` / `Scratch::objectCellCounts` remain derived
rule-scan indexes, not another persistent board representation.

The remaining benchmark gap is that simulation-corpus compact execution is
currently exposed as oracle validation: it runs generated compact code and then
compares against interpreted execution. That is useful for correctness, but it
is not a clean compact-primary runtime benchmark.

## Terms

- **specialized rulegroups**: generated C++ rulegroup kernels used by the native
  runtime from `applyRuleGroup`.
- **specialized full-turn backend**: generated per-game full-state turn
  entrypoints found by source hash and attached to the loaded game.
- **interpreted turn path**: the reference engine path. This is the behavior
  oracle and the normal safe executor unless a generated executor is explicitly
  selected.
- **compact oracle path**: validation/debug plumbing that runs generated
  compact turn code beside interpreted execution and compares the results. It
  is intentionally more expensive than either path alone and should not be used
  as an end-to-end performance benchmark.
- **canonical board**: the single live board representation,
  `PersistentLevelState::board.objects`, laid out as
  `objects[tileIndex * strideObject + word]`.
- **whole-game compilation**: the broader project of moving the hot per-turn
  work from general runtime interpretation into generated per-game code.

Avoid private version labels like "v0" unless a milestone explicitly defines
what the label means.

## Architecture Direction

The runtime should keep a stable public `step` / `tick` interface. Those
entrypoints should be able to run either the interpreted executor or a generated
executor over the same canonical board. Oracle mode stays separate: it is a
correctness check, not a production execution strategy.

The generated C++ should eventually own the hot turn loop for one game at a
time. One-game specialization matches the production use case and keeps
iteration practical. Corpus-wide generation remains valuable as a proving rig
for coverage, parity, and performance regressions, but it should not dictate
the production build shape.

The solver and generator should continue to use the same `SPECIALIZE=true`
workflow while the backend grows more capable. As the generated path gets more
complete, solver/generator calls should pay less dynamic lookup cost and do less
work through generic runtime tables.

The next runtime harness step is a compact-primary execution mode for
simulation/replay, so benchmarks can compare:

```text
interpreter over canonical board
compiled rulegroups over canonical board
compiled whole-turn executor over canonical board
```

## Milestones

1. Keep rulegroup dispatch and generated full-turn dispatch attached through
   source-hash backends.
   The generated full-turn code should call generated semantic kernels directly
   from the per-game turn loop. Interpreter fallback belongs at the explicit
   executor-selection boundary, not as an expanding matrix of feature-level
   fallbacks inside compiler mode.

2. Specialize turn bookkeeping around rules.
   Generate the fixed early/late rulegroup loops, loop-point jumps, rigid retry
   structure, and command collection shape for the game.

3. Specialize command and transition handling.
   Move `cancel`, `restart`, `checkpoint`, `win`, `again`, message handling, and
   output command behavior into generated code one piece at a time, with parity
   checks after each step.

4. Specialize movement resolution and win checks.
   Generate code using fixed layer counts, object masks, player masks,
   collision layers, and win-condition masks.

5. Specialize game constants and state layout.
   Move masks, levels, rule metadata, sound tables, and frequently accessed
   state into generated or compact per-game structures.

6. Expose a smaller turn-game ABI.
   Once the generated path is faithful and fast, add a compact ABI suitable for
   solver/generator hot loops and eventually embedded targets.

7. Add a compact-primary simulation/replay harness.
   The existing compact oracle remains a correctness guard. A separate primary
   execution mode should mutate `GameSession::levelState` directly through the
   generated whole-turn executor so simulation corpus timing is apples-to-apples.

## Correctness Strategy

Simulation parity is the gate for every milestone:

```sh
make simulation_tests_cpp
```

Specialized solver/generator smoke tests should stay green:

```sh
make solver_smoke_tests SPECIALIZE=true
make generator_smoke_tests SPECIALIZE=true
```

The interpreter must remain available as the correctness oracle. Compiler mode
should fail loudly on unsupported or incorrect semantics instead of silently
falling back feature by feature. Debug tracing should prefer the interpreter
unless the generated path explicitly preserves the same trace behavior.

## Performance Strategy

Prefer small, measurable reductions in hot-loop work over large rewrites. The
first wins should come from removing per-turn dynamic rule dispatch and repeated
runtime table lookups for a single game.

Keep build iteration fast. `COMPILED_RULES_MAX_ROWS`, generated-source reuse,
Ninja builds, and one-game specialization are practical tools, not incidental
details. A speedup that makes linking painful should be treated skeptically
until it proves useful in the solver/generator workflow.

Use the solver focus group for tight performance iteration. The full solver
directory is a mining pool and broad regression suite, not the default benchmark
surface for every codegen edit.

Use corpus-wide specialization to answer coverage questions, then optimize the
one-game path that real users and tools will run.

Do not benchmark compact oracle wall time as if it were compact-primary
execution. Oracle mode intentionally does extra work.

## Open Questions

- What is the smallest generated state/executor boundary now that interpreter
  fallback no longer needs a second board representation?
- Which part of movement resolution pays off first: fixed layer loops, fixed
  masks, or a fuller generated mover?
- Should the compact turn-game ABI be C-facing from the start, or remain a C++
  solver/generator interface until the shape stabilizes?
- How much debug trace parity should generated whole-turn code provide before it
  is allowed to run when `PS_DEBUG_*` flags are enabled?
- Should compact-primary simulation become the default for compiled builds once
  the benchmark harness and parity gates are green?
