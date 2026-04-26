# Whole-Game Compilation Implementation Checklist

This checklist turns `PLAN.md` into day-to-day engineering work. It should stay
specific enough that a person or agent can pick the next unchecked item, make a
small commit, and know how to prove it.

The north star is a generated per-game execution path: conceptually,
`tick(state, input) -> state`. The interpreter remains the reference
implementation until the generated path has earned every piece of behavior.

## How To Use This File

- Treat each checkbox as a commit-sized or review-sized task unless it says
  otherwise.
- Keep the generated path boring and truthful: handle only the cases it really
  implements, and fall back to the interpreter for everything else.
- After each behavior-moving change, run the smallest relevant parity test first,
  then a broader one before committing.
- Update this checklist when a task is completed, split, invalidated, or made
  more precise by new evidence.

Status markers:

- `[x]` done and validated.
- `[ ]` not started.
- `[?]` needs a design decision or measurement before implementation.

## Always-On Guardrails

- [ ] Preserve `interpreterStep` and `interpreterTick` as the behavior oracle.
  Generated code may call them directly, or decline handling so dispatch falls
  through to them.

- [ ] Preserve debug behavior. If a `PS_DEBUG_*` mode would lose information on
  the generated path, dispatch must choose the interpreter until trace parity is
  deliberately implemented.

- [ ] Preserve solver/generator semantics for `RuntimeStepOptions`:
  `playableUndo=false`, `emitAudio=false`, no accidental auto-settling of
  `pendingAgain`, and no hidden undo snapshot growth.

- [ ] Keep one-game specialization ergonomic. Production iteration should use a
  single generated game source where practical; corpus-wide generation remains a
  proving rig.

- [ ] Keep fallback cheap enough to be useful. A generated tick function should
  be allowed to bail out without rebuilding a whole second world around the
  interpreter.

- [ ] Prefer explicit coverage and counters to confidence by inspection. If a
  milestone changes behavior or dispatch shape, add a way to observe whether it
  is being used.

## Current Groundwork

- [x] Compiled rule-group kernels exist and attach by source hash.

- [x] `compile-rules --coverage-json` reports aggregate and per-source coverage.

- [x] `make compiled_rules_simulation_suite_coverage` writes a reusable
  simulation-suite coverage JSON file.

- [x] With `COMPILED_RULES_MAX_ROWS=99`, compiled rules cover 452/452 unique
  simulation-suite source texts.

- [x] Solver and generator can opt into generated rule code with
  `SPECIALIZE=true`.

- [x] Public `step` / `tick` dispatch can try a `CompiledTickBackend`.

- [x] `interpreterStep` / `interpreterTick` name the reference engine path.

- [x] Generated sources export a compiled tick backend that currently delegates
  to `interpreterStep` / `interpreterTick`.

## Current Push: Solver Focus 2x Performance

This section is the active near-term work queue. The immediate goal is:

```text
make solver_focus_compare
median_elapsed_ms compiled/interpreted <= 0.500x
same targets, same generated count, no worse solved/timeout status
```

The working focus group is intentionally small and curated. It is a performance
lab bench, not the full corpus.

### Current Inputs

- [x] Treat `src/tests/solver_tests` as a mining pool, not the default
  moment-to-moment benchmark.

  Current checkpoint:

  - Solver corpus size: 184 `.txt` game files.
  - Focus manifest size: 44 targets.
  - Focus candidate count when mined: 85.
  - Distinct games in focus: 35.
  - Excluded clang-heavy games: `easyenigma.txt`, `karamell.txt`,
    `paint everything everywhere.txt`.

- [x] Make the solver focus group the near-term performance north star.

  Current official goal:

  ```text
  make solver_focus_perf_report SOLVER_FOCUS_RUNS=3
  median_elapsed_ms compiled/interpreted <= 0.500x
  ```

  The full solver suite remains a regression and discovery tool. It should not
  be the default loop for every generated-kernel edit.

### Current Status

- [x] Add human-readable focus comparison output.

  Current command:

  ```sh
  make solver_focus_compare
  ```

  Done means: the target rebuilds stale benchmark JSONs, compares interpreted
  vs compiled runs, and prints median wall/elapsed/generated ratios.

- [x] Add detailed focus performance reporting with runtime counters.

  Current command:

  ```sh
  make solver_focus_perf_report
  ```

  Done means: the report can show slowest/fastest targets, compiled tick hits,
  compiled rule hits, pattern tests, candidate cells, row scans, and mask rebuild
  counters.

- [x] Add solver timing breakdown to focus benchmark outputs.

  Done means: focus benchmark JSON preserves solver-reported `step_ms`,
  `clone_ms`, `hash_ms`, and related timing fields, and the compare report
  prints median step/clone/hash ratios. This separates generated turn speed
  from end-to-end solver overhead.

- [x] Add step-time outlier tables to focus detail output.

  Done means: `--detail` prints the slowest and fastest targets by `step_ms`
  ratio directly, so generated-kernel regressions are visible even when
  end-to-end solver elapsed is dominated by other costs.

- [x] Report per-target work mismatches in focus comparison.

  Done means: `--detail` reports when interpreted and compiled runs expanded or
  generated different numbers of states, so a portfolio-search speedup is not
  mistaken for a pure per-state kernel speedup.

- [x] Surface mask rebuild outliers in the focus detail report.

  Done means: `--detail` prints a `top_mask_rebuilds` table with target,
  elapsed ratio, compiled routing bucket, usage reason, tick/rule hits, dirty
  rebuild calls, rebuilt rows/columns, row scans, candidate cells, pattern
  tests, and mask rebuild calls.

- [x] Make focus compiled benchmark use `SPECIALIZE=true`.

  Done means: the same focus benchmark runner can produce interpreted and
  compiled JSON outputs using the existing Makefile convention.

- [x] Include generated-code and runtime freshness in focus benchmark reuse.

  Done means: changes to compiler/runtime/solver files invalidate compiled
  focus benchmark outputs instead of reusing stale JSON.

- [x] Inline fixed mask checks in generated match predicates.

  Done means: generated fixed-width match functions use direct pointer
  arithmetic and literal `&` tests instead of helper calls such as
  `compiledRuleMaskPtr`, `compiledRuleBitsSet`, and
  `compiledRuleCellObjects`.

- [x] Inline literal masks in generated replacement code.

  Done means: replacement code uses direct `session.liveLevel.objects` /
  `session.liveMovements` pointers, stack arrays, and literal mask words instead
  of helper mask loads and `std::array` temporaries.

- [x] Use high row coverage for focus specialization.

  Done means: focus builds default to
  `SOLVER_FOCUS_COMPILED_RULES_MAX_ROWS=99`, while the global
  `COMPILED_RULES_MAX_ROWS=1` iteration default remains unchanged.

  Current evidence: focus row-limit misses moved from `237` to `0` for included
  focus sources, but median elapsed was still roughly flat.

- [x] Remove stale hard-coded focus exclusions from the Makefile.

  Done means: `solver_focus_mine` still supports explicit
  `SOLVER_FOCUS_EXCLUDE_GAMES`, but the Makefile default is empty. A fresh
  mining run should sample the current corpus and should not silently preserve
  yesterday's slow-compile blacklist.

  These games were temporarily excluded in one prior local focus manifest:

  - `easyenigma.txt`
  - `karamell.txt`
  - `paint everything everywhere.txt`

  Current evidence: they caused very large generated sources and compiler
  termination in an unbudgeted rows-99 build. Future exclusion should be driven
  by an explicit compile-time budget recorded in the freshly mined manifest, not
  a Makefile default.

- [x] Commit progress with metrics in commit titles.

  Recent checkpoints:

  - `Focus compiled rules 1.043x to 1.010x median elapsed`
  - `Focus rows99 coverage misses 237 to 0, median 1.003x`
  - `Focus excludes clang-heavy games, 50 to 44 targets`

### Measurement Discipline

- [ ] Define the official focus score line.

  Acceptance criteria:

  - The official score is `median_elapsed_ms compiled/interpreted`.
  - The report also prints `median_wall_ms` and `median_generated`, but they do
    not replace the elapsed score.
  - The report clearly fails or labels the run when target identities differ.

  Validation:

  ```sh
  make solver_focus_compare SOLVER_FOCUS_RUNS=1
  ```

- [ ] Define the official 2x gate.

  Acceptance criteria:

  - Official runs use `SOLVER_FOCUS_RUNS=3`.
  - `compiled/interpreted elapsed <= 0.500`.
  - All targets remain solved.
  - Target identities match.
  - `median_generated` and expanded/generated per-target medians match unless a
    solver algorithm change deliberately explains the difference.

  Validation:

  ```sh
  make solver_focus_perf_report SOLVER_FOCUS_RUNS=3
  node src/tests/compare_solver_focus_benchmarks.js \
    build/native/solver_focus_perf_interpreted.json \
    build/native/solver_focus_perf_compiled.json \
    --detail --goal-ratio 0.5
  ```

- [ ] Add a focus report summary suitable for commit messages.

  Acceptance criteria:

  - One command prints a compact before/after block that can be pasted into a
    commit message or PR.
  - It includes target count, solve/timeout status, elapsed ratio, wall ratio,
    generated ratio, and top three slowest targets.
  - It mentions rows, budget, opt level, and whether perf mode was enabled.

  Suggested command shape:

  ```sh
  make solver_focus_perf_report SOLVER_FOCUS_RUNS=1
  ```

- [ ] Keep interpreted and compiled focus outputs fresh by construction.

  Acceptance criteria:

  - Interpreted output is stale when manifest, corpus, strategy, run count, or
    profiling mode changes.
  - Compiled output is also stale when generated-code inputs or specialization
    knobs change.
  - The freshness check explains the reason in one line.

  Validation:

  ```sh
  touch native/src/cli/main.cpp
  make solver_focus_compare SOLVER_FOCUS_RUNS=1
  ```

- [ ] Add a checked-in focus measurement note after each meaningful speed
  change.

  Acceptance criteria:

  - Either `BASELINE_MEASUREMENTS.md` or the relevant commit message records
    the exact command, date, target count, ratio, and notable outliers.
  - Build time and solver elapsed time are recorded separately when compile time
    is part of the trade.

- [ ] Promote runtime counters to regression signals.

  Acceptance criteria:

  - The perf report flags or fails when `compiled_tick_hits` unexpectedly drops.
  - It flags rising `compiled_tick_fallbacks` or
    `compiled_rule_group_fallbacks`.
  - It can compare `mask_rebuild_calls`, `row_scans`, `candidate_cells_tested`,
    and `pattern_tests` against a previous compiled focus output.
  - Counter regression checks are opt-in until noise is understood.

  Validation:

  ```sh
  make solver_focus_perf_report SOLVER_FOCUS_RUNS=3
  ```

### Solver State And Graph Overhead Track

Intent: compiled rules have made the turn-step path faster, but focus runs now
show a large amount of elapsed time outside generated rule evaluation. This
track makes solver graph overhead visible first, then attacks it in low-risk
layers before introducing a compact state ABI.

The preferred order is:

```text
attribute graph cost -> no-allocation hash -> flat visited table
-> compact solver state prototype -> generated tick over compact state
```

- [x] Add explicit graph-overhead timing buckets to the solver.

  Status: implemented in `native/src/solver/main.cpp` and propagated through
  `run_solver_level_benchmark.js` / `compare_solver_focus_benchmarks.js`.
  Timers now use nanosecond accumulation internally and continue to report
  milliseconds in JSON. Portfolio results merge both phases, so timing and
  generated/expanded counts now describe the whole solve attempt rather than
  only the winning phase.

  Acceptance criteria:

  - Solver JSON reports these additional timing fields:
    - `frontier_pop_ms`
    - `frontier_push_ms`
    - `visited_lookup_ms`
    - `visited_insert_ms`
    - `node_store_ms`
    - `heuristic_ms`
    - `solved_check_ms`
    - `timeout_check_ms` if measurable without distorting the loop
    - `unattributed_ms`
  - `unattributed_ms` is computed from elapsed time minus known measured
    buckets, not guessed.
  - Focus comparison prints median ratios for the new major buckets.
  - Per-target detail can list targets where graph overhead is larger than
    `step_ms`.
  - Timing remains cheap enough for normal focus runs, or is guarded behind an
    opt-in profiling flag if needed.

  Validation:

  ```sh
  make solver_focus_compare SOLVER_FOCUS_RUNS=1
  node src/tests/compare_solver_focus_benchmarks.js \
    build/native/solver_focus_benchmark_interpreted.json \
    build/native/solver_focus_benchmark_compiled.json \
    --detail
  ```

- [x] Add a graph-overhead summary suitable for decision making.

  Status: `solver_focus_compare` prints median ratios for step, clone, hash,
  visited, frontier, node-store, heuristic, solved-check, timeout-check,
  unattributed, and total graph overhead. `--detail` also prints the slowest
  graph-overhead targets and the largest compiled graph-overhead targets.

  Current one-run focus reading after this checkpoint:

  ```text
  median_elapsed_ms compiled/interpreted=0.917x (-8.3%)
  median_step_ms compiled/interpreted=0.859x (-14.1%)
  median_clone_ms compiled/interpreted=0.999x (-0.1%)
  median_hash_ms compiled/interpreted=0.993x (-0.7%)
  median_graph_overhead_ms compiled/interpreted=0.977x (-2.3%)
  ```

  Interpretation: generated stepping is faster, but clone/hash/heuristic and
  remaining unattributed state-management cost now visibly cap end-to-end wins.

  Acceptance criteria:

  - The focus report prints a compact split like:

    ```text
    step=... clone=... hash=... visited=... frontier=... node_store=... unattributed=...
    ```

  - It identifies whether the next biggest bucket after `step_ms` is clone,
    hash, visited, frontier, heuristic, or unattributed.
  - It reports both absolute medians and compiled/interpreted ratios.
  - It makes clear when the generated rules are faster but end-to-end elapsed
    is capped by graph/state overhead.

- [?] Add no-allocation or streaming solver state hashing.

  Status: investigated before implementation. The current
  `sessionStateKey(...)` path is already streaming and allocation-free: it
  walks existing `Session` fields and object words into a 128-bit `StateKey`.
  The surrounding costs are the full `Session` clone, node storage, and visited
  table operations. Keep this item open for semantic expansion of the solver
  key, but do not expect a big win from removing a hash temporary.

  Intent: the current solver state key path builds a `StateKey` from a full
  `Session`. The first low-risk replacement should preserve semantics while
  avoiding avoidable temporary allocation and runtime traversal work.

  Acceptance criteria:

  - A new solver hash path can hash canonical solver-node state:
    quiescent object occupancy for deterministic focus games.
  - A parity test compares the new hash/key path against the existing
    `sessionStateKey` path for solver smoke states and replay-derived states.
  - Search behavior is unchanged: same solution status, same solution where the
    strategy is deterministic, and same generated/expanded counts on smoke
    targets.
  - `hash_ms` or total elapsed improves on at least one focus or target
    benchmark before committing.

  Validation:

  ```sh
  make build
  make solver_smoke_tests
  make solver_parity_smoke
  make solver_focus_compare SOLVER_FOCUS_RUNS=1
  ```

- [x] Replace solver visited storage with a flat open-addressed table.

  Status: implemented as `FlatBestDepth` in `native/src/solver/main.cpp`.
  It stores the full `StateKey` plus best depth, uses linear probing with
  explicit growth, preserves stale-pop `<` and child-duplicate `<=` semantics,
  and reports probe/growth counters in solver JSON and focus benchmark JSON.
  The normal path now treats `StateKey` as a probe accelerator only. Matching
  entries must also compare equal under the solver-semantic `Session` fields.
  `--hash-state-keys` is available only as an explicit performance experiment
  that treats the 128-bit key as identity.

  Current one-run focus reading after this checkpoint:

  ```text
  status: interpreted={"solved":50} compiled={"solved":50}
  median_elapsed_ms compiled/interpreted=0.922x (-7.8%)
  median_visited_ms compiled/interpreted=0.772x (-22.8%)
  median_graph_overhead_ms compiled/interpreted=0.927x (-7.3%)
  flat compiled graph bucket: 55.2ms, down from the prior 57.4ms reading
  ```

  Intent: `std::unordered_map<StateKey, depth>` is convenient but can be a poor
  fit for hundreds of thousands of small, fixed-shape solver keys.

  Acceptance criteria:

  - The solver uses a flat visited table for the hot search path, guarded behind
    an internal switch until parity is proven.
  - The table stores key plus best depth, preserves stale-pop behavior, and can
    distinguish hash collision from key equality.
  - Table growth is explicit and measured.
  - `visited_lookup_ms`, `visited_insert_ms`, or elapsed time improves on the
    focus group without solved-count regression.
  - Memory use does not grow unexpectedly versus `unordered_map`.

  Validation:

  ```sh
  make solver_smoke_tests
  make solver_determinism_tests
  make solver_parity_smoke
  make solver_focus_compare SOLVER_FOCUS_RUNS=1
  ```

- [?] Define the solver compact-state boundary.

  Status: first solver-side boundary is implemented. Solver nodes now carry a
  canonical compact state made from object-major occupancy bitsets. Random-rule
  games are excluded from focus mining for now, so RNG state is deliberately
  outside this compact solver identity. The interpreter `Session` is still
  retained for tick execution, fallback, and solution reconstruction.

  Recommended default: introduce a compact state only for solver/generator hot
  paths first, leaving player/API paths on `Session`.

  Acceptance criteria:

  - The compact state explicitly owns:
    - canonical object occupancy for one fixed level
  - It explicitly does not own RNG state in the current focus pipeline:
    games with `random` or `randomDir` rules are excluded before focus mining.
  - It explicitly does not own live movements at solver-node boundaries:
    movement state must be settled and zero before a state enters the graph.
  - It explicitly does not own current level index: the level is a fixed search
    parameter, and level transition is a solved signal.
  - It explicitly does not own pending-again: `again` chains are internal to the
    transition function and must be exhausted before a graph node is stored.
  - It explicitly does not own checkpoint/restart data: checkpoint is ignored
    by the solver, and restart is a game-over/dead-edge transition result.
  - It explicitly does not own flickscreen/zoomscreen/UI/debug/audio/undo data
    on solver hot paths.
  - It may omit proven-inert content such as static background/wall cells, but
    only with a conservative proof that those bytes cannot affect future turns.
  - Conversion from `Session` to compact state is defined.
  - Conversion from compact state back to `Session` is defined for fallback,
    parity checks, serialization, and debugging.
  - Unsupported games or unsupported runtime features decline the compact path
    cleanly.
  - The compact visited table uses canonical compact-state bytes as exact
    equality. Hashes may choose buckets and speed lookup, but a hash match
    alone must not merge two states.
  - Hashes are recomputed from canonical compact bytes until profiling proves
    that hash recomputation, rather than cloning or stepping, is the bottleneck.

- [?] Add canonical compact-state equality and memory accounting.

  Status: visited identity now hashes the canonical compact state and exact
  equality compares the compact state stored on solver nodes. Solver JSON
  reports `compact_state_bytes` and `compact_max_state_bytes`; for
  `pushit.txt#5` the compact state is 48 bytes per node, about 2.1 MB total
  for roughly 43k stored unique states in a 500 ms run. Static/inert omission
  analysis is not implemented yet.

  Intent: make state identity perfect before relying on specialized compact
  search. The solver may hash for speed, but equality must be canonical bytes
  or an equivalent exact compare.

  Acceptance criteria:

  - Define which object/layer/cell facts are mutable and therefore encoded.
  - Conservatively detect static/inert layers or cells only when no rule,
    command, win condition, movement, or random replacement can alter or
    observe the omitted facts differently.
  - Store compact state once in node storage; visited entries store hash/key,
    depth, and node index, not a duplicate state copy.
  - Compute the visited-table hash from the same canonical bytes used for
    equality; do not maintain an incremental hash through rule application in
    this milestone.
  - Report compact state bytes per node, visited entry bytes, and estimated
    total solver memory.
  - Any hash collision or hash-equivalent but byte-different state is handled by
    probing onward and counted.
  - A validation mode compares compact byte equality against materialized
    interpreter `Session` equality on smoke/focus states.

- [x] Avoid transient full-node storage for duplicate exact states.

  Intent: exact visited equality should not require moving every candidate
  child `Session` into solver node storage before discovering that the state is
  already dominated.

  Current behavior:

  - The solver computes compact state and key for a child candidate.
  - The flat visited table compares that compact state against existing stored
    nodes before the child is pushed into `nodes`.
  - Duplicate/dominated candidates are discarded without a push/pop of a full
    `Node`.
  - Surviving candidates are then stored once with compact identity plus the
    `Session` still needed for the current interpreter tick path.

- [x] Reduce weighted-A* heuristic allocation overhead.

  Intent: keep A* as the strong default while making its per-node heuristic
  cost cheaper. BFS avoids heuristic work but loses too much search guidance on
  the focus group.

  Current behavior:

  - `winConditionHeuristicScore` can reuse a per-search `HeuristicScratch`
    instead of allocating distance vectors on every heuristic call.
  - `puzzlescript_solver --astar-weight N` exposes the weighted-A* multiplier
    for experiments; default remains `2`.
  - `run_solver_level_benchmark.js --solver-arg ARG` can pass solver-specific
    benchmark flags such as `--astar-weight`.

  Current evidence on the 50-target focus group, one run, 500 ms timeout:

  ```text
  strategy/weight   solved   median_elapsed   median_generated   median_heuristic
  bfs               30/50    399 ms           53192              0.0 ms
  weighted w=1      44/50    306.5 ms         41387              12.6 ms
  weighted w=2      50/50    272 ms           41595.5            12.6 ms
  weighted w=3      48/50    290.5 ms         33497              13.9 ms
  weighted w=4      47/50    307 ms           34758              14.8 ms
  greedy            36/50    321 ms           32018.5            16.5 ms
  ```

  Conclusion: keep weighted A* weight `2` as the focus default; it is the best
  solve-rate/elapsed tradeoff in this sample.

- [x] Reuse compact solver child scratch during interpreter-backed edges.

  Intent: make the current compact-node path cheaper and shape the solver loop
  around a single edge boundary that a future compact tick can replace.

  Current behavior:

  - Compact-node mode keeps a reusable parent scratch `Session` for
    `CompactSolverState -> Session` materialization.
  - Compact-node mode now also keeps a reusable child scratch `Session` for
    each candidate input instead of heap-allocating a transient child session
    per generated edge.
  - Child preparation copies only semantic turn-start state, level objects,
    solver-relevant flags, and random state; it preserves scratch-vector
    capacity and marks masks dirty for the runtime step.

  Current evidence on the 50-target focus group, one run, weighted A*, 500 ms
  timeout:

  ```text
  mode                  solved   median_elapsed   median_generated   median_clone
  stored Session nodes  50/50    290.5 ms         41595.5            33.75 ms
  compact child scratch 50/50    276.5 ms         41595.5            8.12 ms
  ```

  Conclusion: compact storage is now a small end-to-end win while doing exactly
  the same search work. The remaining dominant costs are still interpreter
  stepping and compact heuristic evaluation, so the next priority is a real
  compact tick edge.

- [?] Prototype compact solver state for one simple focus game.

  Status: `puzzlescript_solver --compact-node-storage` stores solver nodes
  without retained `Session`s and materializes a scratch `Session` from compact
  occupancy when a node is expanded. Smoke-level correctness passes, but the
  prototype is not a default: on `pushit.txt#5` weighted A* it kept identical
  expanded/generated counts while increasing elapsed time because every
  materialized parent forces mask/cache rebuild work inside `step`. The
  compact-node path now also evaluates the win-condition heuristic directly
  from compact occupancy, proving the solver can ask heuristic questions of the
  compact state without a retained `Session`.

  Suggested first candidates:

  - `pushit.txt`
  - `15 push pull levels.txt`
  - a tiny `sokoban_basic`-style fixture if the focus games are too noisy

  Acceptance criteria:

  - The prototype is isolated behind a capability check or internal flag.
  - Compact clone is a direct copy of fixed-size or tightly packed state data,
    not a full `Session` copy.
  - Compact hash avoids materializing a full `Session`.
  - The solver can still materialize a scratch `Session` for the existing
    interpreter or compiled-rule path when the tick implementation requires it.
  - The prototype reports compact materialization under `clone_ms`; split it
    into a separate `materialize_ms` counter before judging the next iteration.
  - Solver parity remains green for the supported game.

  Next improvement: materialize row/column/board masks and object-cell bitsets
  directly from compact occupancy, or move generated tick to consume compact
  occupancy without rebuilding interpreter caches. The generic compact
  heuristic is slower than the current `Session` heuristic on `pushit.txt#5`;
  a generated/baked heuristic should specialize the masks and object ids rather
  than use the generic compact matcher.

  Current evidence:

  ```text
  pushit.txt#5 weighted-astar 1000ms:
    session nodes: expanded=26226 generated=131129 elapsed_ms=385 heuristic_ms=27.4
    compact nodes: expanded=26226 generated=131129 elapsed_ms=395 heuristic_ms=43.0
  ```

  Validation:

  ```sh
  make build
  make solver_smoke_tests
  make solver_parity_smoke
  build/native/puzzlescript_solver src/tests/solver_tests \
    --game "pushit.txt" --level 5 --timeout-ms 2000 \
    --strategy portfolio --json --quiet
  ```

- [ ] Store solver nodes as compact states for supported games.

  Acceptance criteria:

  - Supported-game nodes retain compact state bytes plus parent/input metadata,
    not `std::unique_ptr<Session>`.
  - A per-search scratch `Session` is reused for materialization when required.
  - `node_store_ms`, clone time, memory use, and generated/sec improve on at
    least one focus target.
  - Unsupported games continue to use the existing `Session` node path.
  - Solution reconstruction still uses parent/input links and does not require
    retained full sessions.

  Validation:

  ```sh
  make solver_smoke_tests
  make solver_determinism_tests
  make solver_parity_smoke
  make solver_focus_compare SOLVER_FOCUS_RUNS=1
  ```

- [ ] Add a generated compact tick prototype.

  Intent: once compact state exists, whole-game compilation should target it
  directly: `tick(compact_state, input) -> compact_state`, with `Session`
  retained as oracle and fallback rather than as the hot state container.

  Near-term task list:

  - [x] Add a separate compact tick backend type and weak finder instead of changing
    the existing `CompiledTickBackend` layout.
  - [x] Attach the compact backend by source hash next to compiled rules/tick.
  - [x] Add solver-side capability checks and counters that distinguish:
    `compact_tick_attempt`, `compact_tick_hit`, `compact_tick_fallback`, and
    `compact_tick_unsupported`.
  - [x] Try compact tick before the interpreter-backed scratch fallback when a
    supported compact backend is attached.
  - [x] Generate a `handled=false` compact backend stub so linkage, dispatch, and
    counters are proven before behavior moves.
  - [x] Refactor the current solver edge into a named helper with two
    implementations: compact tick first, interpreter-backed scratch fallback
    second.
  - [x] Pick one deterministic, no-random, no-again, no-restart focus fixture for
    the first `handled=true` compact tick.
    First target: `src/tests/solver_smoke_tests/one_move.txt`.
  - [x] Generate direct object-bitset input seeding and fixed movement execution
    for that fixture.
  - [ ] Compare the compact tick output against the interpreter fallback inside a
    debug/oracle mode before allowing solver use.
  - [ ] Only then fold compact heuristic/hash computation into the compact tick
    result when it can reuse touched state.

  Current first-slice behavior:

  - Supports movement-only games with no early/late rules, no rigid behavior,
    no turn-affecting metadata, no movement/audio side effects, one non-aggregate
    player object, one player in every non-message level, and one simple
    `all player on target` win condition.
  - Generated compact tick mutates object-major `objectBits` directly, handles
    directional/action inputs, reports `changed`, and evaluates the simple win
    condition.
  - Unsupported games still attach a compact backend with
    `wholeTurnSupported=false` and remain on the interpreter-backed scratch
    fallback.

  Current evidence:

  ```text
  one_move.txt specialized compact solver:
    solution=right
    compact_tick_attempts=1
    compact_tick_hits=1
    compact_tick_fallbacks=0
    compact_tick_unsupported=0
    step_ms=0.000084

  push_goal.txt specialized compact solver:
    solution=right,right
    compact_tick_attempts=6
    compact_tick_hits=6
    compact_tick_fallbacks=0
    compact_tick_unsupported=0
    interpreted_step_ms=0.021585
    compact_step_ms=0.000375
  ```

  Acceptance criteria:

  - One supported game has a generated C++ compact tick entrypoint.
  - The entrypoint reads/writes compact state directly for the supported turn
    slice.
  - The compact tick result reports changed, won, game-over/restart, and
    unsupported/fallback status without heap allocation.
  - Interpreter parity compares compact tick output against
    `interpreterStep`/`interpreterTick` materialized through `Session`.
  - Solver can choose compact tick for supported states and fall back cleanly
    before mutating state when unsupported behavior is encountered.

  Validation:

  ```sh
  make simulation_tests_cpp
  make solver_smoke_tests SPECIALIZE=true
  make solver_parity_smoke SPECIALIZE=true
  make solver_focus_compare SPECIALIZE=true SOLVER_FOCUS_RUNS=1
  ```

- [ ] Retire duplicated compact-state checklist items once this track owns
  them.

  Acceptance criteria:

  - The later `Specialized State Layout` and `Smaller Tick-Game ABI` sections
    either point back here or contain only non-duplicated architecture notes.
  - No compact-state task exists in two places with conflicting acceptance
    criteria.

### Focus Group Hygiene

- [ ] Regenerate the focus manifest intentionally after solver corpus changes.

  Acceptance criteria:

  - `make solver_focus_mine` does not apply stale hard-coded exclusions.
  - Explicit one-off exclusions still work through `SOLVER_FOCUS_EXCLUDE_GAMES`.
  - The manifest records `excluded_games`.
  - Target count and candidate count are visible in the command output.
  - If the solver directory grew substantially, the focus group is refreshed or
    deliberately left pinned with a note explaining why.

  Validation:

  ```sh
  make solver_focus_mine
  node -e 'const j=require("./build/native/solver_focus_group.json"); console.log(j.target_count, j.excluded_games)'
  ```

- [x] Add compile-time quarantine during focus mining.

  Intent: exclude games that make specialized focus builds painfully slow, but
  recompute that decision whenever `make solver_focus_mine` is run.

  Done means:

  - `SOLVER_FOCUS_COMPILE_TIMEOUT_SECONDS ?= 60` controls the threshold.
  - A value of `0` disables compile-time quarantine.
  - `make solver_focus_mine` resets/recomputes compile timings by default.
  - The manifest records `compile_excluded_games` with game name, measured
    compile seconds, threshold, row limit, budget kind, limit, observed value,
    and reason.
  - The manifest top-level `compile_probe` block records all active compile
    budget knobs, including generated line cap.
  - Compile exclusions are separate from manual `excluded_games`.
  - Compile timing probes run before solving, so levels from slow-to-compile
    games are not considered for the focus group.
  - The eligible temporary corpus is recorded as `mined_corpus`; the original
    corpus remains recorded as `corpus`.
  - Command output distinguishes rule-count budget exclusions from
    generated-line budget exclusions.

  Validation:

  ```sh
  make solver_focus_mine SOLVER_FOCUS_COMPILE_TIMEOUT_SECONDS=60
  node -e 'const j=require("./build/native/solver_focus_group.json"); console.log(j.compile_excluded_games || [])'
  ```

- [x] Parallelize focus compile probes.

  Intent: `make solver_focus_mine` should exploit the machine while it
  discovers which games are eligible for the focus set, without letting each
  individual CMake probe also consume every core.

  Current behavior:

  - `SOLVER_FOCUS_COMPILE_PROBE_JOBS ?= auto` controls how many per-game
    compile probes run concurrently.
  - `auto` resolves to roughly half the machine parallelism.
  - `SOLVER_FOCUS_COMPILE_BUILD_JOBS ?= 1` controls the inner CMake build for
    each probe, keeping outer parallelism predictable.
  - The manifest records both requested and resolved probe job counts under
    `compile_probe`.
  - Compile probe result ordering stays deterministic in the manifest even
    though logging appears as probes complete.
  - Each per-game line starts with a completion counter such as `[24/40]`.

  Validation:

  ```sh
  make -n solver_focus_mine
  node src/tests/mine_solver_focus_group.js ... --compile-probe-jobs 2
  ```

- [x] Exclude random-rule games from focus mining.

  Intent: keep the focus group deterministic while compact solver-state
  equality and generated-rule performance are being optimized. PuzzleScript
  `random` rule groups and `random`/`randomDir` replacements have semantics
  that can make visually identical boards distinct through RNG state, so they
  should not define the default iteration target.

  Done means:

  - The focus miner scans the `RULES` section for `random` and `randomDir`
    tokens.
  - Comment-only mentions and identifiers such as `randommoved` do not trigger
    the exclusion.
  - Random-rule exclusions happen before compile probing and before level
    solving.
  - The manifest records `random_excluded_games` with source lines and text.
  - The command summary prints `random_excluded=N`.

  Current evidence: the current focus manifest had one remaining random-rule
  target, `gobble_rush.txt` level 19, after `fickle fred.txt` was edited to no
  longer use `random`.

- [x] Abort specialized focus compilation when it exceeds the iteration budget.

  Done means: `make solver_focus_compare` and specialized
  `make solver_focus_benchmark SPECIALIZE=true` wrap the focus CMake build in a
  timeout.

  Current behavior:

  - `SOLVER_FOCUS_COMPILE_TIMEOUT_SECONDS ?= 60`
  - `SOLVER_FOCUS_COMPILE_TIMEOUT_SECONDS=0` disables the timeout.
  - Timeout exits with code `124` after terminating the build process group.

  Validation:

  ```sh
  make -n solver_focus_benchmark SPECIALIZE=true
  node src/tests/run_with_timeout.js 1 -- node -e 'setTimeout(()=>{}, 5000)'
  ```

- [x] Skip pathological generated source files in focus builds.

  Done means: focus compilation has a generated-line budget in addition to the
  compiled-rule-count budget, and the miner treats a line-budget skip as a
  compile exclusion rather than an eligible fallback game.

  Current behavior:

  - `SOLVER_FOCUS_MAX_GENERATED_LINES_PER_SOURCE ?= 20000`
  - A value of `0` disables the line budget.
  - `compile-rules` accepts `--max-generated-lines-per-source`.
  - Generated sharded C++ files include source path and source hash comments.

  Current evidence: the freshly mined 50-target focus manifest built under the
  60s guard with 30 compiled sources after skipping 4 generated-line outliers
  and 2 rule-count outliers; one-run median elapsed was `0.993x`.

- [ ] Add a manifest pruning tool for one-off focus surgery.

  Intent: make "drop this game from focus" explicit instead of using ad hoc
  Node snippets.

  Acceptance criteria:

  - Command can remove one or more games from an existing focus manifest.
  - It updates `target_count` and `excluded_games`.
  - It prints before/after target counts.

  Suggested command shape:

  ```sh
  node src/tests/filter_solver_focus_group.js \
    build/native/solver_focus_group.json \
    --exclude-game "some game.txt"
  ```

- [ ] Track clang-heavy excluded games separately from runtime-slow games.

  Acceptance criteria:

  - Exclusions distinguish "compiler pathological" from "runtime outlier".
  - Runtime-slow games stay in focus unless they make iteration painful.
  - Reintroducing an excluded game is an explicit measurement task.

- [ ] Add a focus compile-tail report.

  Acceptance criteria:

  - For a specialized focus build, report the slowest generated `.cpp` compile
    units or at least the largest generated sources.
  - The report maps generated source hashes back to game names.
  - It identifies when a source should be excluded, split, or optimized.

### Compile-Time Controls

- [ ] Keep daily focus builds bounded.

  Acceptance criteria:

  - Default focus builds use `SOLVER_FOCUS_COMPILED_RULES_MAX_ROWS=99`.
  - Default focus builds retain a finite
    `SOLVER_FOCUS_MAX_COMPILED_RULES_PER_SOURCE`.
  - Removing the per-source budget is opt-in, not the normal loop.

  Validation:

  ```sh
  /usr/bin/time -p make solver_focus_benchmark \
    SPECIALIZE=true \
    SOLVER_FOCUS_RUNS=1 \
    SOLVER_FOCUS_OUT=/tmp/focus_compiled.json
  ```

- [ ] Make unbudgeted focus builds safer.

  Acceptance criteria:

  - A command can try unbudgeted rows-99 generation without taking down normal
    iteration.
  - It prints generated source sizes before compiling or fails fast above a
    configured size limit.
  - It recommends exclusions or sharding for pathological files.

- [ ] Split or cap pathological generated sources.

  Acceptance criteria:

  - Large games no longer produce single translation units that clang spends
    minutes compiling or kills.
  - Splitting preserves source-hash registry behavior.
  - Link time and compile time are compared before/after.

  Candidate games from the current focus investigation:

  - `easyenigma.txt`
  - `karamell.txt`
  - `paint everything everywhere.txt`

- [ ] Keep Ninja as the default specialized build generator when available.

  Acceptance criteria:

  - The Makefile continues to pick Ninja for compiled-rules builds when
    installed.
  - The build output makes the generator choice obvious.

### Runtime Coverage And Routing

- [x] Classify focus targets by actual compiled-rule usage.

  Done means:

  - The detailed report separates:
    - `compiled_tick_hits == 0 && compiled_rule_group_hits == 0`
    - `compiled_tick_hits > 0 && compiled_rule_group_hits == 0`
    - `compiled_rule_group_hits > 0`
  - The summary prints counts for each bucket.
  - The slowest table includes the bucket label.

  Current bucket labels:

  - `no_tick_no_rules`
  - `tick_no_rules`
  - `compiled_rules`
  - `no_counters`
  - `unknown`

  Validation:

  ```sh
  make solver_focus_perf_report SOLVER_FOCUS_RUNS=1
  ```

- [x] Explain zero compiled-rule hits per target.

  Done means:

  - For every zero-hit focus target, the report can say whether the source was
    skipped by compile budget, unsupported by generated tick routing, or simply
    did not execute compiled groups on that solve path.
  - The explanation is machine-readable enough to sort by reason.

  Current report fields:

  - `compiled_rules_attached`
  - `compiled_tick_attached`
  - `compiled_usage_reasons`
  - per-target `reason=...` in the detail table

  Note: `compiled_rule_group_hits` counts the generic compiled-rule dispatch
  path. Generated tick rule loops bypass that generic counter, so
  `compiled_tick_bypassed_generic_rule_counter` means the generated tick backend
  ran but the old rule-group counter is not the right measurement point.

- [x] Fix source-hash mismatches caused by missing trailing newlines.

  Done means: `compile-rules` normalizes generated sources the same way the
  solver does before compiling a game, by appending one trailing newline when
  the file lacks one. Without this, generated backends existed for some focus
  games but were registered under a hash the runtime never asked for.

- [ ] Fix generated tick routes that call the wrapper but no rule kernels.

  Acceptance criteria:

  - Targets with `compiled_tick_hits > 0` and `compiled_rule_group_hits == 0`
    either gain rule hits or explain why no compiled groups are reachable.
  - Solver generated counts remain identical.
  - No target regresses from solved to timeout.

- [ ] Remove `interpreter_delegation` from supported focus turns.

  Acceptance criteria:

  - For the supported focus slice, generated tick performs the turn phases it
    claims instead of immediately delegating to `interpreterStep` /
    `interpreterTick`.
  - `compiled_tick_hits > 0`.
  - `compiled_tick_fallbacks == 0` for supported targets.
  - Coverage no longer reports `interpreter_delegation` for those sources.

  Validation:

  ```sh
  make compiled_tick_dispatch_smoke
  make solver_focus_perf_report SOLVER_FOCUS_RUNS=3
  ```

- [ ] Dispatch once per loaded game, not once per solver step.

  Acceptance criteria:

  - Solver/generator load paths cache compiled tick/backend support decisions
    after the game is loaded.
  - Per-state expansion does not repeat source hashing, backend lookup, or
    support scans.
  - Counters still prove generated dispatch use.

  Validation:

  ```sh
  make solver_smoke_tests SPECIALIZE=true
  make solver_focus_perf_report SOLVER_FOCUS_RUNS=3
  ```

- [ ] Decide whether compile-budget-skipped focus sources belong in the default
  focus group.

  Acceptance criteria:

  - If a game is useful for runtime performance but too costly to compile, add a
    source-splitting task instead of silently interpreting it.
  - If a game is mostly noise for the current compiler work, exclude it from the
    default focus manifest.

### Kernel Micro-Optimization Queue

- [ ] Remove no-op generated replacement stores.

  Acceptance criteria:

  - Replacement code does not allocate or fill `newObjects`, `created`,
    `destroyed`, or `newMovements` words whose clear/set masks are both zero and
    whose word cannot change.
  - Generated code still passes replacement parity for multi-word games.

  Validation:

  ```sh
  make solver_smoke_tests SPECIALIZE=true
  make simulation_tests_cpp
  ```

- [x] Generate specialized setter paths for common one-word object updates.

  Intent: `compiledRuleSetCellObjectsFromWords` still pays generic per-word and
  object-index maintenance costs.

  Acceptance criteria:

  - One-word object replacement can update live objects, row/column/board masks,
    object-cell bitsets, create/destroy masks, and dirty flags without a generic
    helper call.
  - Generic helper remains fallback for uncommon/multi-word cases.
  - Runtime counters or disassembly show the helper call is gone for hot
    one-word replacements.

  Current progress:

  - Added `compiledRuleSetCellObjectsWord1` so generated one-word object
    replacements avoid the generic pointer/stride helper.
  - The helper preserves live object words, row/column/board masks,
    object-cell bitsets, create/destroy masks, and dirty mask behavior.

- [x] Generate specialized setter paths for common movement updates.

  Acceptance criteria:

  - One- or two-word movement replacement can update live movements and movement
    masks without generic helper traffic.
  - Dirty row/column/board behavior remains identical.
  - `emitAudio=false` paths avoid sound-related work.

  Current progress:

  - Added `compiledRuleSetCellMovementsWord1` and emit it for one-word movement
    replacements.

- [ ] Reduce mask rebuild pressure.

  Current clue: focus counters show very high `mask_rebuild_calls` on several
  targets, including fast and slow compiled runs.

  Current progress:

  - Generated rule-group loops now guard `compiledRuleRebuildMasks(session)`
    with `session.anyMasksDirty || session.objectCellIndexDirty`, avoiding
    clean helper calls in the generated path while preserving dirty rebuilds.

  Acceptance criteria:

  - Distinguish unavoidable rebuild calls from calls that find no dirty masks.
  - Avoid rebuild calls after generated groups that provably did not change
    state.
  - Preserve interpreter fallback expectations.

- [x] Avoid redundant full scans after generated anchor scans.

  Intent: when a generated row uses an object-cell anchor, an exhaustive anchor
  scan that finds no matches has already ruled out the row. Falling through to
  the generic full line scan repeats work on the common no-match path.

  Current progress:

  - Generated anchored scans now jump to the shared empty-match exit even when
    they find no matches.
  - Generic full scans remain the fallback when no trustworthy anchor is
    selected.

- [ ] Add a mask correctness verifier for generated paths.

  Acceptance criteria:

  - Debug/non-release validation can full-rebuild row, column, board, movement,
    and object-cell masks after generated turns.
  - The verifier compares full rebuilds against incremental masks.
  - The verifier can be enabled for simulation and solver parity runs without
    changing release benchmark behavior.

  Validation:

  ```sh
  make simulation_tests_cpp
  make solver_parity_smoke SPECIALIZE=true
  ```

- [ ] Add per-generated-group counters for attempts, matches, changes, and
  fallback.

  Acceptance criteria:

  - Counters can identify which generated groups dominate a slow target.
  - Counter overhead is opt-in and absent from normal benchmark runs.

- [ ] Inspect generated assembly for one fast and one slow focus game.

  Suggested games:

  - Fast: `pipe puffer.txt` or `gem soketeer.txt`
  - Slow: `gobble_rush.txt`

  Acceptance criteria:

  - Object files and disassembly are written under
    `build/compiled-rules-inspect/`.
  - Hot match/replacement functions mostly contain loads, bit operations,
    branches, and stores.
  - Any remaining avoidable helper call is turned into a checklist item.

### Algorithmic Optimization Queue

- [ ] Fuse compatible rule scans within a group.

  Acceptance criteria:

  - Rules sharing direction, row length, and scan bounds scan candidate starts
    once.
  - Rule order and simultaneous-per-rule replacement semantics remain exact.
  - At least one focus target shows fewer candidate tests or lower elapsed time.

- [ ] Generate candidate bitsets for row patterns.

  Acceptance criteria:

  - Required object masks are shifted/intersected to produce candidate cells.
  - Missing-object constraints use complements safely inside board bounds.
  - Candidate iteration preserves PuzzleScript order.

- [ ] Specialize anchor scans to avoid sort/unique when uniqueness is known.

  Acceptance criteria:

  - Generated anchor scans can prove deterministic unique candidate order or
    retain sort/unique.
  - Any removed sort/unique is covered by parity tests.

  Current progress:

  - Vertical object-bitset anchor scans with one anchor object now skip
    `sort/unique`; the bitset iteration already yields unique start indices in
    PuzzleScript order.
  - Horizontal one-object anchor scans still sort to preserve row-major order,
    but skip `unique`.

- [ ] Add per-group precheck masks before entering compiled loops.

  Acceptance criteria:

  - Generated code skips impossible groups with one or a few board-mask checks.
  - Prechecks do not change random, rigid, or command semantics.

### Whole-Tick Work That Directly Affects Solver Speed

- [ ] Move more early/late rule traversal out of generic runtime fallback.

  Acceptance criteria:

  - A focus target with real compiled rule hits spends fewer steps in generic
    `applyRuleGroup`.
  - `compiled_rule_group_fallbacks` decreases or is explained.

- [ ] Generate the supported turn skeleton instead of delegating.

  Acceptance criteria:

  - Generated `step` / `tick` performs clear movement, movement seeding, early
    rules, movement resolution hook, late rules, result assembly, and final mask
    state for a narrow supported slice.
  - Unsupported title-screen, message, debug, or metadata cases fall back
    cleanly.
  - Final serialized state matches the interpreter.

  Validation:

  ```sh
  make simulation_tests_cpp
  make solver_smoke_tests SPECIALIZE=true
  make solver_parity_smoke SPECIALIZE=true
  ```

- [ ] Specialize the no-input solver tick path.

  Acceptance criteria:

  - Solver no-input expansions can call generated tick without interpreter turn
    setup for supported games.
  - Final state and generated counts match the interpreter.

- [ ] Specialize directional movement seeding for supported solver games.

  Acceptance criteria:

  - Direction/action input setup uses generated constants for player masks and
    movement words.
  - Unsupported metadata or aggregate cases fall back cleanly.

- [ ] Prototype compact solver clone state for one focus game.

  Acceptance criteria:

  - State clone/hash cost is measured against the current `Session` clone path.
  - Solver parity stays green.
  - The prototype is isolated behind capability checks.

### Correctness And Commit Gates For This Push

- [ ] Before every performance commit, run the smallest relevant smoke.

  Typical commands:

  ```sh
  make build
  make solver_smoke_tests SPECIALIZE=true
  git diff --check
  ```

- [ ] Before claiming a solver focus speedup, run a fresh comparison.

  Command:

  ```sh
  make solver_focus_compare SOLVER_FOCUS_RUNS=1
  ```

  Acceptance criteria:

  - Same targets.
  - No worse solved/timeout status.
  - Same `median_generated`.
  - Commit title includes the before/after elapsed ratio.

- [ ] Before merging a broad codegen/runtime change, run parity smokes.

  Commands:

  ```sh
  make simulation_tests_cpp
  make solver_parity_smoke SPECIALIZE=true
  make generator_smoke_tests SPECIALIZE=true
  ```

## Baseline Measurements

- [x] Record a clean no-specialization baseline for solver smoke.

  Command:

  ```sh
  /usr/bin/time -p make solver_smoke_tests SPECIALIZE=false
  ```

  Done means: elapsed time and pass count are noted in the relevant commit or
  PR description.

- [x] Record a clean specialization baseline for solver smoke.

  Command:

  ```sh
  /usr/bin/time -p make solver_smoke_tests SPECIALIZE=true
  ```

  Done means: elapsed time, generated source reuse behavior, and pass count are
  noted.

- [x] Record a generator smoke baseline.

  Command:

  ```sh
  /usr/bin/time -p make generator_smoke_tests SPECIALIZE=true
  ```

  Done means: elapsed time and pass/fail behavior are noted.

- [x] Record compiled-rule simulation coverage at both the fast default and the
  high-row proving setting.

  Commands:

  ```sh
  make compiled_rules_simulation_suite_coverage COMPILED_RULES_MAX_ROWS=1
  make compiled_rules_simulation_suite_coverage COMPILED_RULES_MAX_ROWS=99
  ```

  Done means: coverage counts are noted, including remaining miss reasons at
  `COMPILED_RULES_MAX_ROWS=1`.

- [x] Decide whether to add a small checked-in benchmark notes file, or keep
  measurement notes in commit messages and PR descriptions only.

- [x] Record the current filtered solver focus baseline.

  Current note lives in `native/src/compiler/BASELINE_MEASUREMENTS.md`.

  Done means: target count, excluded games, median interpreted elapsed, median
  compiled elapsed, and generated counts are recorded.

## Generated Tick Observability

- [x] Add runtime counters for compiled tick attempts, hits, and fallbacks.

  Intent: make it obvious whether a solver/generator run is actually entering
  generated tick dispatch.

  Acceptance criteria:

  - Counters are available through the existing runtime-counter snapshot path.
  - Existing counter callers remain source-compatible.
  - The profile output can distinguish compiled-rule hits from compiled-tick
    hits.

  Validation:

  ```sh
  make build
  build/native/puzzlescript_cpp test simulation-corpus src/tests/resources/testdata.js --case-index 1 --repeat 3 --profile-timers --jobs 1
  make solver_smoke_tests SPECIALIZE=true
  ```

- [x] Add a focused generated-tick dispatch smoke test.

  Intent: prove the generated tick backend is found by source hash and called,
  even while it still delegates to the interpreter.

  Acceptance criteria:

  - Test fails if `ps_compiled_tick_find_backend` is not linked or not found.
  - Test proves both `step` and `tick` dispatch can be attempted.
  - Test does not depend on timing.

  Suggested shape:

  - Generate C++ for `src/demo/sokoban_basic.txt`.
  - Link a specialized test binary or use an existing specialized smoke path.
  - Assert compiled tick attempt/hit counters changed.

- [ ] Add a debug-gating smoke test.

  Intent: ensure generated tick dispatch does not bypass debug traces by
  accident.

  Acceptance criteria:

  - At least one `PS_DEBUG_RULES`, `PS_DEBUG_MOVES`, or `PS_DEBUG_AGAIN` run
    proves dispatch chooses the interpreter.
  - Counter output or test plumbing makes that choice visible.

## Generated Tick Owns Rule-Group Dispatch

- [x] Move early rule-group loop selection into generated tick code.

  Intent: generated tick should iterate the game-specific early rule groups and
  call compiled group kernels directly, instead of entering the generic
  `applyRuleGroups` loop first.

  Implementation notes:

  - Start with games whose early groups are all covered by compiled rules.
  - Preserve loop-point behavior exactly.
  - Preserve banned rigid-group behavior by falling back if unsupported.
  - Keep late rules on the interpreter path for the first commit if that makes
    the change smaller.

  Acceptance criteria:

  - Generated tick handles at least one simple non-rigid game without calling
    generic rule-group dispatch.
  - It falls back for games or states outside the supported slice.
  - Simulation parity remains green.

  Validation:

  ```sh
  make build
  make simulation_tests_cpp
  make solver_smoke_tests SPECIALIZE=true
  ```

- [x] Move late rule-group loop selection into generated tick code.

  Intent: generated tick owns both early and late rule traversal for supported
  games.

  Acceptance criteria:

  - Late rule groups call compiled kernels directly.
  - Late loop points behave identically to the interpreter.
  - Unsupported late groups force a clean fallback.

  Validation:

  ```sh
  make simulation_tests_cpp
  make solver_parity_smoke SPECIALIZE=true
  ```

- [x] Generate fixed loop-point tables.

  Intent: avoid dynamic loop-point lookup for generated rule traversal.

  Acceptance criteria:

  - Generated C++ contains static per-game loop-point decisions.
  - Behavior matches `lookupLoopPoint`.
  - Infinite-loop guard behavior remains capped at the interpreter limit.

- [x] Generate direct group coverage predicates.

  Intent: a generated tick function should know at compile time whether every
  group it needs is compiled.

  Acceptance criteria:

  - No runtime scan is needed to decide rule-loop eligibility.
  - The generated backend declines handling if any required group is missing.
  - Coverage JSON and generated-code eligibility agree.

- [x] Remove redundant rule backend lookup inside generated tick.

  Intent: once generated tick owns rule traversal, it should call local generated
  functions directly rather than re-finding the same backend.

  Acceptance criteria:

  - Generated tick source compiles as a one-game unit.
  - Direct calls do not break sharded/corpus generation.
  - `CompiledRulesBackend` remains available for interpreter fallback.

## Turn Skeleton Specialization

- [x] Extract an interpreter-readable turn skeleton.

  Intent: identify the exact ordered phases currently inside `executeTurn` so
  the generated implementation can mirror them without copying mystery.

  Acceptance criteria:

  - The phase list is documented in comments or helper names.
  - No behavior changes are made in the extraction commit.
  - The generated checklist can refer to named phases.

  Current phases to preserve:

  - Clear audio state.
  - Prepare create/destroy sound masks.
  - Create or reference turn-start undo snapshot.
  - Handle `require_player_movement` starting positions.
  - Clear movement state.
  - Seed player movement.
  - Apply early rules.
  - Resolve movements.
  - Retry rigid pass if needed.
  - Apply late rules.
  - Compute modification status.
  - Process cancel / dontModify / output commands.
  - Process restart / checkpoint / win / again.
  - Fill `ps_step_result`.
  - Rebuild masks before returning.

- [x] Generate a supported-game predicate for the turn skeleton.

  Intent: generated tick should handle only games whose required skeleton pieces
  have been implemented.

  Acceptance criteria:

  - Predicate is generated from game features, not handwritten per fixture.
  - Unsupported commands or metadata choose fallback.
  - The reason for fallback can be observed in debug/profile mode.

- [ ] Implement generated no-input `tick` for the simplest supported games.

  Intent: make `tick(session, options)` do real generated work for games without
  player-input complications.

  Acceptance criteria:

  - Generated tick executes early rules, movement resolution by fallback or
    generated helper, late rules, and result assembly for a narrow slice.
  - `step(session, PS_INPUT_TICK, options)` and `tick(session, options)` agree.
  - Unsupported cases delegate to `interpreterTick`.

- [ ] Implement generated directional/action `step` entry.

  Intent: move real input processing into generated code after no-input tick is
  stable.

  Acceptance criteria:

  - Title-screen and message-mode behavior remains interpreter-handled until
    explicitly specialized.
  - Directional movement seeding matches interpreter behavior.
  - Action input works for supported games or falls back cleanly.

## Command Handling

- [x] Classify command shapes in compiled tick coverage.

  Intent: make the command tail measurable before moving behavior out of the
  interpreter.

  Acceptance criteria:

  - Coverage distinguishes games with no rule commands, generated command
    queues with interpreter tails, and unknown command shapes.
  - Aggregate counts are emitted alongside per-source feature status.
  - Classification uses stable strings that can be tracked over time.

- [x] Generate command queue shape for supported games.

  Intent: avoid dynamic string command handling where the command set is known.

  Acceptance criteria:

  - Generated code can represent `again`, `cancel`, `checkpoint`, `restart`,
    `win`, `message`, and output commands without string scans on the hot path.
  - Interpreter fallback can still consume the existing `CommandState`.
  - Unknown or unsupported command forms choose fallback.

- [ ] Specialize `cancel`.

  Acceptance criteria:

  - State restoration matches interpreter behavior.
  - Audio behavior respects `emitAudio`.
  - `dontModify` result behavior is identical.

- [ ] Specialize `restart`.

  Acceptance criteria:

  - Restart target restoration matches the interpreter.
  - `run_rules_on_level_start` behavior is preserved or forces fallback.
  - `recordRestartUndo` behavior is preserved.

- [ ] Specialize `checkpoint`.

  Acceptance criteria:

  - Restart snapshot contents match interpreter behavior.
  - Checkpoint after win is handled identically.

- [ ] Specialize `win` and explicit win command.

  Acceptance criteria:

  - Explicit `win` command and evaluated win conditions produce identical
    `won` / `transitioned` results.
  - Level transition behavior remains correct for final and non-final levels.

- [ ] Specialize `again` scheduling.

  Acceptance criteria:

  - `pendingAgain` is set only when interpreter would set it.
  - The would-again-change probe either runs generated code safely or delegates
    to the interpreter.
  - Solver/generator external settling remains unchanged.

- [ ] Specialize message and output commands.

  Acceptance criteria:

  - Message text, text mode, and output command side effects match interpreter
    behavior.
  - Unsupported output behavior falls back.

## Movement Specialization

- [ ] Add movement-resolution coverage classification.

  Intent: know which games can use generated movement before writing a large
  mover.

  Acceptance criteria:

  - Coverage output can classify movement features: layer count, aggregate
    player mask, rigid groups, movement sounds, failure sounds, and metadata
    affecting movement.
  - Classification is available per source.

- [ ] Generate fixed layer loops for movement masks.

  Intent: replace dynamic layer-count loops in the hot movement path for
  supported games.

  Acceptance criteria:

  - Generated code uses compile-time layer count and movement-word count.
  - Behavior matches interpreter movement resolution for non-rigid fixtures.
  - Unsupported aggregate/rigid cases fall back.

- [ ] Specialize player movement seeding.

  Acceptance criteria:

  - Direction masks match interpreter constants.
  - Aggregate player movement is either correctly handled or falls back.
  - `require_player_movement` behavior remains correct.

- [ ] Specialize movement success/failure sounds.

  Acceptance criteria:

  - `emitAudio=false` keeps all audio work out of solver/generator hot paths.
  - `emitAudio=true` matches event seeds and ordering.
  - Debug audio paths either match or force interpreter dispatch.

- [ ] Specialize rigid retry bookkeeping.

  Acceptance criteria:

  - Rigid retry limit matches interpreter behavior.
  - Banned groups are updated identically.
  - Existing rigid simulation cases pass under specialized builds.

- [ ] Expand movement specialization to aggregate player cases.

  Acceptance criteria:

  - Aggregate movement remains atomic across player layers.
  - Mixed aggregate/non-aggregate games pass parity.

## Win Conditions And Level Transitions

- [ ] Generate fixed win-condition checks.

  Acceptance criteria:

  - `some`, `no`, `all`, and aggregate combinations match interpreter behavior.
  - Masks are generated or referenced without dynamic name lookup.
  - Empty or unusual levels match interpreter results.

- [ ] Specialize level advancement.

  Acceptance criteria:

  - `transitioned` and `won` fields match interpreter behavior.
  - Message levels and title/text mode are either handled or fall back.
  - Prepared-session state remains consistent after transition.

- [ ] Specialize `run_rules_on_level_start`.

  Acceptance criteria:

  - Games without the metadata pay no generated overhead.
  - Games with the metadata run the correct rule path after load/restart.
  - Undo/audio options are preserved.

## Game Constants And Generated Data

- [ ] Emit per-game constants for strides, dimensions, object counts, layer
  counts, and word counts.

  Acceptance criteria:

  - Generated code no longer loads these values through `Game` for supported
    hot paths.
  - Constants match the loaded runtime game.

- [ ] Emit mask tables used by generated tick.

  Acceptance criteria:

  - Rule masks, layer masks, player masks, win masks, and sound masks needed by
    generated code are available as generated constants or stable references.
  - No generated constant duplicates large data without a measured reason.

- [ ] Emit compact rule metadata for generated traversal.

  Acceptance criteria:

  - Group order, late/early partitioning, loop points, random flags, rigid
    mapping, and command capability are represented without runtime scanning.

- [ ] Emit level metadata needed by generated transition logic.

  Acceptance criteria:

  - Level dimensions, message flags, target level indices, and restart data are
    available to generated code.
  - Loaded-level seeds remain interpreter-compatible.

## Specialized State Layout

- [?] Decide the first specialized state boundary.

  Options:

  - Keep using `Session` and specialize only code first.
  - Introduce a compact generated state alongside `Session`.
  - Introduce a compact state only for solver/generator, leaving player/API
    paths on `Session`.

  Recommended default: keep using `Session` until generated tick has meaningful
  behavior, then introduce compact solver/generator state.

- [ ] Add state-layout notes once the boundary is chosen.

  Acceptance criteria:

  - The chosen state owns canonical object occupancy for deterministic solver
    focus games.
  - Solver-node invariants state that movements are zero, pending-again is
    exhausted, current level is a search parameter, checkpoint is ignored, and
    restart is a game-over/dead-edge result.
  - Conversion to/from `Session` is defined for fallback and testing.

- [ ] Prototype compact solver/generator state for one simple game.

  Acceptance criteria:

  - State clone cost is lower than `Session` clone cost.
  - Hashing and serialization for tests remain available.
  - Interpreter fallback remains possible through a conversion path.

- [ ] Replace solver child cloning for supported games.

  Acceptance criteria:

  - Solver parity remains green.
  - Memory use and nodes/second improve on at least one benchmark.

## Smaller Tick-Game ABI

- [?] Decide whether the first compact ABI is C++-only or C-facing.

  Recommended default: C++-only until state layout stabilizes, then wrap in C.

- [ ] Define a compact tick result.

  Acceptance criteria:

  - Represents changed, won, transitioned, pending again, and optional audio.
  - Does not require heap allocation on solver/generator hot paths.

- [ ] Define compact input representation.

  Acceptance criteria:

  - Covers up, down, left, right, action, and no-input tick.
  - Maps exactly to existing `ps_input` values or has a lossless adapter.

- [ ] Add generated ABI smoke test.

  Acceptance criteria:

  - One generated game can be stepped without going through public `Session`
    dispatch.
  - Result and serialized final state match interpreter behavior.

## Solver Integration

- [ ] Add a solver flag or internal capability path for compiled tick.

  Intent: distinguish "compiled rules are linked" from "whole tick is actually
  handling solver steps".

  Acceptance criteria:

  - Solver can report compiled tick attempts/hits/fallbacks.
  - `SPECIALIZE=true` remains the user-facing Makefile entry.
  - Unsupported games still solve through the interpreter.

- [ ] Add solver parity target focused on compiled tick.

  Acceptance criteria:

  - Runs a small representative game set.
  - Fails if compiled tick silently falls back for every case when it should not.
  - Still verifies final solver results.

- [ ] Benchmark one-game solver hot loop.

  Acceptance criteria:

  - Compare `SPECIALIZE=false`, compiled rules only, and compiled tick handling.
  - Report build time separately from solve time.
  - Include at least one Sokoban-like and one rule-heavy fixture.

## Generator Integration

- [ ] Add generator capability reporting for compiled tick.

  Acceptance criteria:

  - Generator can report whether generated tick handled candidate steps.
  - Fallback behavior is visible in quiet logs or profiling output.

- [ ] Add generator parity/smoke target focused on compiled tick.

  Acceptance criteria:

  - Existing `generator_smoke_tests` remains green.
  - A compiled-tick-specific path proves generated dispatch is exercised.

- [ ] Benchmark generator candidate evaluation.

  Acceptance criteria:

  - Compare generated tick against interpreter path on the same preset and seed.
  - Separate generation/build time from candidate evaluation throughput.

## Build And Codegen Ergonomics

- [ ] Keep generated source reuse robust as tick code grows.

  Acceptance criteria:

  - Stamps include any options that change generated tick output.
  - Reuse never hides stale generated code after CLI changes.
  - `COMPILED_RULES_REUSE_SINGLE_CPP=false` still forces regeneration.

- [ ] Keep sharded corpus generation working.

  Acceptance criteria:

  - `--emit-cpp-dir` emits both rule and tick backend accessors per source.
  - `registry.cpp` links all generated rule and tick backends.
  - Corpus generation remains useful for coverage and smoke tests.

- [ ] Split generated code when compile time demands it.

  Acceptance criteria:

  - One-game generation remains simple.
  - Corpus generation can shard large tick helpers if needed.
  - Link time is measured before and after any split.

- [ ] Consider moving codegen out of `native/src/cli/main.cpp`.

  Intent: whole-game codegen will outgrow the CLI file.

  Acceptance criteria:

  - New module has a narrow interface: source/game/options in, generated C++
    out.
  - Behavior is unchanged in the extraction commit.
  - Tests and `compile-rules` output are unchanged except for intentional
    formatting.

## Coverage And Eligibility Reporting

- [x] Extend coverage JSON from compiled rules to compiled tick eligibility.

  Acceptance criteria:

  - Per source, report whether rule loops, commands, movement, win conditions,
    level transitions, and state layout are generated or fallback-only.
  - Aggregate counts are easy to read from a one-line Node or jq command.

- [x] Add human-readable miss reasons for compiled tick.

  Acceptance criteria:

  - Reasons are stable strings suitable for tracking over time.
  - Examples: `unsupported_command`, `movement_rigid`, `message_level`,
    `debug_trace`, `state_layout`.

- [x] Track "fully generated tick" count separately from "compiled rules"
  coverage.

  Acceptance criteria:

  - Coverage can say "rules compile" and "whole tick handles" independently.
  - The distinction is reflected in docs and CLI output.

## Correctness Test Matrix

- [ ] Keep the core simulation suite green after every behavior move.

  Command:

  ```sh
  make simulation_tests_cpp
  ```

- [ ] Keep solver smoke green after every generated dispatch change.

  Command:

  ```sh
  make solver_smoke_tests SPECIALIZE=true
  ```

- [ ] Keep solver determinism green after random or again changes.

  Command:

  ```sh
  make solver_determinism_tests SPECIALIZE=true
  ```

- [ ] Keep solver parity green after state, movement, command, or win changes.

  Command:

  ```sh
  make solver_parity_smoke SPECIALIZE=true
  ```

- [ ] Keep generator smoke green after solver-facing step changes.

  Command:

  ```sh
  make generator_smoke_tests SPECIALIZE=true
  ```

- [ ] Add focused fixtures as bugs are found.

  Acceptance criteria:

  - Every generated-path correctness bug gets a minimal regression fixture.
  - The fixture fails before the fix and passes after it.

## Performance Test Matrix

- [ ] Track generated-code size for one-game builds.

  Acceptance criteria:

  - Record generated `.cpp` size and object size for a simple game and a
    rule-heavy game.
  - Avoid optimizing for corpus-wide code size unless it affects iteration.

- [ ] Track compile and link time separately.

  Acceptance criteria:

  - Use `/usr/bin/time -p`.
  - Note whether generated sources were reused.
  - Note whether Ninja or another CMake generator was used.

- [ ] Track solver throughput.

  Acceptance criteria:

  - Measure nodes/second or equivalent existing solver output.
  - Compare against non-specialized and compiled-rule-only baselines.

- [ ] Track generator throughput.

  Acceptance criteria:

  - Measure candidate evaluations or existing generator throughput output.
  - Use fixed seed/preset when possible.

## Documentation Tasks

- [ ] Update `PLAN.md` when an architecture decision changes.

- [ ] Update this checklist when a milestone is completed.

- [ ] Add a short user-facing note once compiled tick does real work.

  Acceptance criteria:

  - Explain how to enable it through `SPECIALIZE=true`.
  - Explain fallback behavior.
  - Avoid implying C/LLVM/assembly support before it exists.

- [ ] Document `COMPILED_RULES_MAX_ROWS` as a build-time/code-size knob.

  Acceptance criteria:

  - Docs make clear that high-row support exists, but the default is conservative
    for iteration.

## Open Decisions To Revisit

- [?] Should generated tick and compiled rules remain under the `compile-rules`
  command, or should the CLI grow a clearer `compile-game` command once whole
  tick generation is real?

- [?] Should coverage files use one combined schema or separate
  compiled-rules/compiled-tick schemas?

- [?] How much trace parity is enough before generated tick may run under
  `PS_DEBUG_*` flags?

- [?] When compact state exists, should interpreter fallback convert compact
  state to `Session`, or should unsupported cases be rejected before entering
  compact-state execution?

- [x] Which benchmark should be the "north-star" solver benchmark for this
  project?

  Current answer: the near-term compiler performance north star is
  `make solver_focus_perf_report SOLVER_FOCUS_RUNS=3`, scored by
  `median_elapsed_ms compiled/interpreted` on the mined focus manifest. The full
  solver directory remains a regression suite and source of future focus
  targets.

## Current Recommended Implementation Slice

This is the next practical sequence from the current focus-group checkpoint:

1. Classify every focus target by compiled tick/rule hit bucket.
2. Explain zero compiled-rule hits per target.
3. Fix generated tick routes that enter the wrapper but do not reach rule
   kernels.
4. Add a mask correctness verifier for generated paths.
5. Reduce mask rebuild pressure on the slowest focus outliers.
6. Inspect generated assembly for one fast target and one slow target.
7. Commit each proven improvement with the before/after focus ratio in the
   commit title.

This slice assumes the observability groundwork already exists. The task now is
to use those counters to remove real hot-path work while keeping the interpreter
close enough to catch us.
