# Native performance: Crush-It Plan 2.0 (design)

Status: draft (needs review)
Date: 2026-04-23

This replaces (and updates) the earlier Phase-1-era design:
- `docs/superpowers/specs/2026-04-22-native-perf-crush-design.md`

Phase 1 results (context + what changed): `docs/superpowers/plans/2026-04-22-native-perf-phase1-results.md`.

## 1. Goals, scope, and success metrics

### Primary goal (“crush”)
Make native C++ **materially faster than JavaScript** on the *same* simulation suite:
- **Native:** `make simulation_tests_cpp`
- **JS:** `make simulation_tests_js`

“Crush” means: native is convincingly faster than JS (not just within noise). We’ll measure **median-of-5** runs for both sides on the same machine/session.

### Scope

In scope:
- Native runtime performance (rule matching, replacements, stepping).
- Native compile-from-source performance *as used by the C++ test runner* (see §2).
- Native IR load/parse performance (when using the JS-oracle fixture path).
- Test harness plumbing for correct, meaningful, and repeatable benchmarks.

Out of scope:
- Editor / rendering / audio output fidelity (beyond what the simulation suite already asserts).
- Making `make tests` fast in general (too many incidental tests; not a good proxy for engine speed).

### Hard invariants
- Simulation semantics must not change (movement ordering, rule firing, RNG behavior).
- The C++ simulation suite must remain deterministic.

### What “done” looks like
At minimum:
- `make simulation_tests_cpp` is clearly faster than `make simulation_tests_js` on median-of-5.
- `make profile_simulation_tests` shows the time is dominated by expected engine hotspots (not avoidable overhead like env lookups or repeated loading).

## 2. Benchmark definitions (what we optimize for)

### 2.1 North star: `make simulation_tests_cpp`
This is the headline KPI for Crush-It 2.0.

**Important current reality:** today `simulation_tests_cpp` runs the native sweep against a JS-oracle-generated manifest (`build/js-parity-data/fixtures.json`) via:

```104:106:Makefile
simulation_tests_cpp: build $(JS_PARITY_MANIFEST)
	$(NODE) src/tests/run_native_trace_suite.js $(JS_PARITY_MANIFEST) --cli $(PUZZLESCRIPT_CPP) --progress-every 1 --timeout-ms 45000
```

That means it does **not** directly depend on edits to `src/tests/resources/testdata.js` unless the JS parity data is regenerated.

**Crush-It 2.0 harness direction change (required):**
`simulation_tests_cpp` should be able to consume `src/tests/resources/testdata.js` **directly** (native compile-from-source + replay + final-state check), with the JS-oracle fixture pipeline preserved as a separate explicit path.

### 2.2 Drill-down profiler: `make profile_simulation_tests`
This is not the headline number; it exists to break down time into:
- `replay_ms` (engine stepping + matching + replacements)
- `game_load_ms` (IR load/build cost)
- `trace_json_parse_ms` (trace parse cost)

It currently runs `puzzlescript_cpp profile-simulations build/js-parity-data/fixtures.json --repeat 3` (`src/tests/profile_native_trace_suite.sh`).

## 3. Current baseline (April 23, 2026)

### 3.1 `profile_simulation_tests` breakdown (engine-focused replay workload)
From `profile_stats.txt`:
- `native_simulation_profile … wall_ms=20290 … game_load_ms=4690 … trace_json_parse_ms=471 … replay_ms=15081` (Pass 1)

Key takeaway: **replay dominates wall time**, then native load, then trace parse.

### 3.2 Hot stacks (where CPU is going)
From the same `profile_stats.txt` hot stacks:
- `collectRowMatches` is the dominant engine function.
- `applyReplacementAt` and `matchesPatternAt` are next.
- `__findenv_locked` is unexpectedly high and correlates with repeated `std::getenv()` checks in `native/src/runtime/core.cpp` (debug flag helpers and rule debug line filtering).

### 3.3 Native vs JS on the simulation suite
From the recent terminal output you captured:
- Native sweep (`simulation_tests_cpp`) around `native_trace_suite_timing total_elapsed_ms≈9075ms`
- JS sim-only (`simulation_tests_js`) around `≈9480ms`

We are currently **slightly ahead**; Crush-It 2.0 is about pushing this into “obviously faster”, not a 1–5% edge.

## 4. Test harness redesign (compiler-in-the-loop)

### 4.1 Motivation
For Crush-It 2.0, we want the default C++ tests to:
- run from the canonical test sources (`testdata.js` and `errormessage_testdata.js`)
- exercise the **native compiler** directly
- avoid “stale generated fixtures” confusion (editing `testdata.js` but the run still passes)

The JS-oracle export remains valuable for deep semantic comparisons, but it must be an explicit separate target, not the default meaning of `simulation_tests_cpp`.

### 4.2 Simulation runner: `testdata.js` direct
`src/tests/resources/testdata.js` is structurally:
- `var testdata = [ [name, [source, inputs, expectedSerializedLevel, targetLevel?, randomSeed?, expectedSounds?]], ... ]`
- Inputs include numeric directions and control tokens like `"undo"` and `"restart"`.
- Some cases include explicit RNG seed values; **RNG parity is mandatory** or tests will fail/flap.

Plan:
- Parse the file by skipping to the first `[` and JSON-parsing the array.
- For each case:
  - native compile the `source` with `targetLevel` + `randomSeed` (when provided)
  - replay the input list, including undo/restart semantics
  - compare final `serialized_level` against expected
  - if `expectedSounds` present, compare the flattened sound event sequence

### 4.3 Compilation runner: `errormessage_testdata.js` direct
`src/tests/resources/errormessage_testdata.js` is:
- `var errormessage_testdata = [ [name, [source, expectedErrors[], expectedErrorCount]], ... ]`

Plan:
- Parse as above (skip-to-`[`).
- For each case: native compile and compare diagnostics strings and counts to expected.

### 4.4 Keep the JS-oracle path, but separate it
Preserve the existing “JS oracle exports IR/trace JSON” pipeline as a separate suite/target (useful for deep semantic comparisons, traces, and IR regression tests).

## 5. Optimization strategy (post-harness)

### 5.1 Track A: remove accidental overhead (`getenv`)
Problem: `profile_stats.txt` shows `__findenv_locked` hot, and `native/src/runtime/core.cpp` calls `std::getenv()` in multiple helpers (e.g. `PS_DEBUG_RULES`, `PS_DEBUG_RULE_LINES`, etc.).

Design:
- Cache all debug flags once (process start) into a small struct, and make hot-path checks read that cached state.
- Ensure “debug disabled” is the common fast path.

Expected win: small-to-moderate but “free” speed, and it removes noise from profiling.

### 5.2 Track B: replay hot path (rule matching + replacements)
The current dominant functions are:
- `collectRowMatches` (drives calls to `matchesPatternAt`)
- `applyReplacementAt`

Design direction:
- Reduce the number of candidate positions we test (fewer `matchesPatternAt` calls).
- Avoid per-match allocations / repeated vector work.
- Only introduce more invasive tactics (template specialization, SIMD, new indexing structures) if measured profiles still show matching is the bottleneck after the above.

### 5.3 Track C: load path (when using JS-oracle fixtures)
Today `games_reused=0 games_loaded=469` during profiling; for the JS-oracle path this implies repeated loading dominates `game_load_ms`.

Design options (choose based on post-harness measurements):
- IR dedupe in exporter (multiple fixtures share one IR file when `(source, level, seed)` identical).
- Native-side cache keyed by IR content hash (reuse loaded `ps_game` objects even if filenames differ).

## 6. Measurement protocol and gates

### 6.1 Protocol (mandatory)
- Use **median-of-5** runs.
- Record both:
  - `make simulation_tests_cpp`
  - `make simulation_tests_js`
- Use `make profile_simulation_tests` to attribute wins.

### 6.2 Regression discipline
- Any optimization PR must paste before/after numbers for the north star.
- “Enabling” PRs must say so explicitly and are reverted if they don’t quickly pay off.

## 7. Risks and mitigations

### RNG parity risk (hard requirement)
If native RNG differs from JS, final-state-only tests will fail or become flaky. Treat RNG parity as a first-class invariant and gate.

### Harness correctness risk
Undo/restart and sound expectations exist in `testdata.js`. The harness must implement these semantics exactly; failures should print the fixture name and a minimal reproduction path.

## 8. Deliverables
- Implement new default paths:
  - `simulation_tests_cpp` runs directly from `src/tests/resources/testdata.js`
  - `compilation_tests_cpp` runs directly from `src/tests/resources/errormessage_testdata.js`
- Keep a separate JS-oracle parity suite target for trace-level comparisons.
- Keep `profile_simulation_tests` as the drill-down profiler (engine-focused).

