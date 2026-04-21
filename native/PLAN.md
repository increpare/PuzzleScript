# PuzzleScript Native Port Plan

This file tracks the implementation plan for the native PuzzleScript port.

## Goals

- Implement the port in `C++` internally with a stable plain `C` API.
- Build one deterministic semantics core and three consumers: headless library, CLI, and SDL2 desktop player.
- Use the existing JS compiler/runtime as the behavioral oracle during the bootstrap phase.
- Keep the runtime deterministic per session while making it cheap to clone, hash, and run in parallel across many sessions for future solver work.

## Current Phase

Phase 1 is hybrid-first:

- JS exports canonical IR, fixtures, and execution traces.
- Native code loads the exported IR and prepared-session state.
- Differential tooling compares native behavior against JS behavior.
- Native rule execution and native source compilation are still pending.

## Milestones

### M1. Differential Harness and IR Bootstrap

- Export canonical JSON IR from the existing JS compiler.
- Export normalized simulation and compilation fixtures from the existing JS test corpus.
- Export per-input execution traces from JS, including `again` substeps.
- Load the exported IR in native code and reproduce prepared-session state exactly.
- Add CLI commands for `run`, `bench`, `test-fixtures`, `diff-trace`, and `diff-trace-source`.

### M2. Scalar Native Runtime

- Implement scalar `ps_session_step` and `ps_session_tick` against compiled rule structures.
- Preserve exact semantics for `late`, `rigid`, `random`, `randomdir`, `again`, `cancel`, `restart`, `win`, `message`, and `startloop/endloop`.
- Match JS sound event emission and seeded RNG behavior exactly.
- Pass the existing simulation fixtures with identical results.

### M3. Native Compiler

- Lower PuzzleScript source directly to the same IR used by the JS exporter.
- Match compile diagnostics under the existing harness rules.
- Differential-test native compiler output against the JS compiler on a curated corpus.

### M4. SDL2 Player and Headless Packaging

- Provide a minimal SDL2 player over the same core session API.
- Provide a usable headless library surface and CLI workflows for tests and automation.
- Keep gameplay logic in the core, not in the renderer.

### M5. Performance and Solver Readiness

- Optimize hot scalar paths only after correctness is locked.
- Add SIMD backends for proven hot bitset operations.
- Keep cloning, hashing, and input expansion cheap for future solvers.
- Add multi-session throughput benchmarks for search-oriented workloads.

## Differential Debugging

This is a first-class part of the plan, not just a convenience tool.

- When native behavior diverges from JS, we should be able to run both runtimes with the same game, seed, and inputs and find the first divergent point automatically.
- Differential comparison should work at several granularities:
  - after each input
  - after each `again` substep
  - after each rule group
  - eventually after each rule application attempt
- Comparisons should include more than board contents:
  - serialized level
  - current level index and target
  - title/message state
  - command queue
  - audio events
  - RNG state
  - rigid rollback/restart-relevant state
- This should exist both as automated test infrastructure and as an explicit debugging mode in the CLI/runtime.

## Principles

- Correctness before optimization.
- No hidden global runtime state.
- `ps_game` is immutable and shareable across threads.
- `ps_session` is mutable, deterministic, and cheap to clone.
- Parallelism belongs across many sessions/search branches, not within a single state transition unless semantics are proven identical.
