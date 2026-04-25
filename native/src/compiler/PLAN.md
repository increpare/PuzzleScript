# Whole-Game Compilation Plan

## Goal

The long-term goal is to compile a PuzzleScript game into a small, per-game
execution path: conceptually, a `tick(state, input) -> state` function that can
drive solvers, generators, embedded builds, and other tight loops without
repeatedly interpreting the full runtime data model.

The first target is C++ generation. C, LLVM, and assembly are interesting later
targets, but only after the C++ whole-tick path has proved parity and speed.

## Current State

Compiled rules already generate C++ kernels for rule-group application. The
generic runtime remains the fallback and correctness oracle.

The simulation corpus has 469 cases, deduplicated to 452 unique source texts for
compiled-rule coverage. With:

```sh
make compiled_rules_simulation_suite_coverage COMPILED_RULES_MAX_ROWS=99
```

the current compiled-rule backend covers 452/452 unique simulation sources. The
default `COMPILED_RULES_MAX_ROWS=1` is still intentionally conservative: it is
an iteration-time and generated-code-size knob, not a semantic boundary.

Solver and generator builds can already opt into generated rule code:

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
global compiled-rules default remains conservative.

Generated sources also export a compiled tick backend today. That backend is
only a dispatch/linkage proof: it delegates to `interpreterStep` and
`interpreterTick`, so it does not yet optimize whole-turn execution.

## Terms

- **compiled rules**: generated C++ rule-group kernels used by the native
  runtime from `applyRuleGroup`.
- **compiled tick backend**: generated per-game `step` and `tick` entrypoints
  found by source hash and attached to the loaded game.
- **interpreterStep / interpreterTick**: the reference engine path. This is the
  behavior oracle and fallback for generated code.
- **whole-game compilation**: the broader project of moving the hot per-turn
  work from general runtime interpretation into generated per-game code.

Avoid private version labels like "v0" unless a milestone explicitly defines
what the label means.

## Architecture Direction

The runtime should keep a stable public `step` / `tick` interface. Those
entrypoints try the compiled tick backend when one is attached and when debug
tracing allows it. If generated code declines to handle a case, or if no backend
is present, execution falls through to `interpreterStep` / `interpreterTick`.

The generated C++ should eventually own the hot turn loop for one game at a
time. One-game specialization matches the production use case and keeps
iteration practical. Corpus-wide generation remains valuable as a proving rig
for coverage, parity, and performance regressions, but it should not dictate
the production build shape.

The solver and generator should continue to use the same `SPECIALIZE=true`
workflow while the backend grows more capable. As the generated path gets more
complete, solver/generator calls should pay less dynamic lookup cost and do less
work through generic runtime tables.

## Milestones

1. Move rule-group dispatch inside the generated tick backend.
   The generated tick function should call the existing compiled rule kernels
   directly from the per-game turn loop, while preserving interpreter fallback.

2. Specialize turn bookkeeping around rules.
   Generate the fixed early/late rule-group loops, loop-point jumps, rigid retry
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

6. Expose a smaller tick-game ABI.
   Once the generated path is faithful and fast, add a compact ABI suitable for
   solver/generator hot loops and eventually embedded targets.

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

The interpreter must remain available as the correctness oracle. Generated code
should be allowed to handle only the cases it is ready for, and should fall back
cleanly for the rest. Debug tracing should prefer the interpreter unless the
generated path explicitly preserves the same trace behavior.

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

## Open Questions

- What is the smallest generated state representation that still keeps
  interpreter fallback cheap?
- Which part of movement resolution pays off first: fixed layer loops, fixed
  masks, or a fuller generated mover?
- Should the compact tick-game ABI be C-facing from the start, or remain a C++
  solver/generator interface until the shape stabilizes?
- How much debug trace parity should generated whole-tick code provide before it
  is allowed to run when `PS_DEBUG_*` flags are enabled?
