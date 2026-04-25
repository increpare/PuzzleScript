# Whole-Game Compilation Baseline Measurements

These notes capture rough local baselines for the whole-game compilation work.
They are not release benchmarks; they are iteration landmarks. Re-run them when
generated tick begins doing real work, or when build settings change.

Date: 2026-04-25
Branch: `cpp`

## Smoke Timings

Commands were run from the repository root with `/usr/bin/time -p`.

| Workload | Command | Result | real | user | sys | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Solver smoke, interpreter path | `make solver_smoke_tests SPECIALIZE=false` | passed 7 cases | 7.75s | 6.75s | 0.47s | Rebuilt `puzzlescript_solver` in `build`. |
| Solver smoke, generated path linked | `make solver_smoke_tests SPECIALIZE=true` | passed 7 cases | 6.07s | 4.89s | 0.24s | Reused generated solver-smoke sources (`wrote=0`), rebuilt specialized solver. |
| Generator smoke, generated path linked | `make generator_smoke_tests SPECIALIZE=true` | passed | 5.84s | 14.60s | 0.95s | Regenerated one game source (`wrote=1`), built specialized generator. |

## Rule Coverage

| Setting | Sources | Fully compiled | Remaining | Miss reasons |
| --- | ---: | ---: | ---: | --- |
| `COMPILED_RULES_MAX_ROWS=1` | 452 | 277 | 175 | `row_limit=3758` |
| `COMPILED_RULES_MAX_ROWS=99` | 452 | 452 | 0 | none |

`COMPILED_RULES_MAX_ROWS=1` is the fast default for iteration. The high-row run
is the current proving setting for simulation-suite rule coverage.

## Compiled Tick Status

Coverage JSON now reports compiled tick separately from compiled rules. At this
checkpoint, generated tick backend codegen is available, but whole-tick behavior
is not yet generated:

- `backend_codegen_available`: 452/452 simulation-suite sources.
- `fully_generated`: 0/452 simulation-suite sources.
- Current miss reason: `interpreter_delegation`.

The dedicated smoke target verifies dispatch/linkage:

```sh
make compiled_tick_dispatch_smoke
```

That target requires at least one generated tick backend to handle a solver
step, while the generated backend still delegates to `interpreterStep` /
`interpreterTick`.
