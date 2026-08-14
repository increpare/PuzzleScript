# Compiler Monster Garden Design

## Purpose

The Compiler Monster Garden is a local fuzzing tool for PuzzleScript's compiler and
runtime boundary. It grows deterministic mutations from the existing regression
fixtures, looks for outcomes that should never happen, and saves small, ordinary
PuzzleScript programs that can be promoted into regression tests.

The tool belongs beside the existing Node test runner. It deliberately uses the
same source files, browser shims, compiler entry point, and fixture arrays instead
of introducing a second compiler API or a new test framework.

## Command-line experience

The main command is:

```sh
node src/tests/monster_garden/run.js --seed 12345 --count 100
```

A fixed seed must reproduce the same chosen fixture, mutation, and generated source.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--seed N` | `Date.now()` | Non-negative integer seed |
| `--count N` | `100` | Mutants to attempt |
| `--timeout-ms N` | `2000` | Hard child deadline |
| `--fixture SUBSTR` | any | Case-insensitive name filter |
| `--mutator A,B` | all | Restrict to named mutators |
| `--list-mutators` | off | Print mutator names and exit 0 |
| `--output DIR` | `.build/monster_garden` | Artifact root |
| `--no-shrink` | shrink on | Skip minimization |
| `--no-replay` | replay on | Skip undo/replay check |
| `--max-inputs N` | `8` | Input-prefix length |
| `--shrink-budget N` | `200` | Max shrink candidate evaluations |
| `--max-attempts N` | `8` | Retries when a mutator does not apply |

Numeric flags except `--seed` must be integers `> 0`. `--seed` must be an integer
`>= 0`. Unknown flags, missing values, and unknown mutator names exit nonzero.

Each mutant runs in a fresh, time-bounded Node child process. The parent reports a
compact live tally for `ok`, `compiler-error`, `crash`, `timeout`, `invariant`,
`nondeterministic`, and `replay-divergence`.

A completed garden run exits 0 even when monsters are found. Only malformed
options and unexpected parent failures exit nonzero.

## Components

`garden.js` contains deterministic machinery and uses only Node built-ins: corpus
loading, seeded random selection, source mutations, result classification, failure
signatures, line-oriented shrinking, command-line parsing, invariant checks, and
artifact naming.

`worker.js` is a deliberately small adaptation of `run_tests_node.js`. It loads the
real PuzzleScript sources in browser-like globals, reads one JSON job from stdin,
emits one JSON result on stdout, and never owns persistence, timeouts, or policy.

`run.js` owns orchestration. It spawns workers with a hard timeout, classifies
their results, invokes the shrinker for interesting cases, writes artifacts, and
prints the run summary.

`tests.js` exercises deterministic mutation, corpus extraction, classification,
shrinking, worker behavior, timeout handling, and a small end-to-end garden run.

## Mutation strategy

Mutators are small named functions, not a grammar framework. They target compiler
seams represented in real games:

- `delete-rule-punctuation` / `duplicate-rule-punctuation`
- `swap-legend-operator`
- `invalid-viewport`
- `duplicate-rule-command`
- `legend-cycle`
- `swap-sections`
- `odd-whitespace`
- `unterminated-comment`

`apply(source, rng)` returns `{ source, detail }` or `null` when the fixture has
no target. `mutateFixture` picks a mutator from the allowed list and retries up to
`--max-attempts` times. If every attempt returns `null`, it throws `/inapplicable/`.

Mutation metadata records the mutator name, detail, fixture identity, and attempt
index.

## Contracts

### Corpus

`loadCorpus(resourceDir)` evaluates `testdata.js` and `errormessage_testdata.js`
with `vm.runInNewContext`. It does not `require()` those files and does not load
the compiler.

Simulation items are `[name, [source, inputs, expected, level?, randomSeed?]]`.
Compiler-message items are `[name, [source, expectedErrors, errorCount]]`.

Each corpus record is:

```js
{
  name, fixtureIndex, kind, source, inputs, level, randomSeed
}
```

`kind` is `simulation` or `compiler-message`. Simulation items come first, then
compiler-message items, each with their own `fixtureIndex` starting at 0.
Compiler-message records have `inputs: []`, `level: 0`, and `randomSeed: null`.
Missing simulation `level` defaults to `0`; missing `randomSeed` defaults to `null`.

### Worker job (stdin JSON)

```json
{
  "source": "...",
  "inputs": [0, 3, "undo"],
  "level": 0,
  "randomSeed": null,
  "replay": true,
  "maxInputs": 8
}
```

The worker never loads the fixture arrays. It compiles `source` with
`compile(["loadLevel", level], source, randomSeed)` after setting
`unitTesting = true`, `lazyFunctionGeneration = false`, and `IDE = false`.

The executed prefix is `inputs.slice(0, maxInputs)`. The input loop matches
`runTest`: `undo` / `restart` / `tick` / numeric `processInput`, then drain
`againing` with `processInput(-1)`.

### Worker result (stdout JSON)

```json
{
  "kind": "ok",
  "error": null,
  "fingerprint": "0\n...",
  "detail": "",
  "errorCount": 0
}
```

`error` is `{ "name": "...", "message": "..." }` for crashes, otherwise `null`.
The parent classifies timeout; the worker never emits `timeout`.

### Outcome kinds

PuzzleScript usually records diagnostics in `errorStrings` / `errorCount` and
returns. Thrown exceptions are the unexpected case.

| Kind | When | Saved? |
| --- | --- | --- |
| `ok` | `compile` did not throw, `errorCount === 0`, execution finished, invariants hold, two compiles match, replay matches | no |
| `compiler-error` | `compile` did not throw and `errorCount > 0` | no |
| `crash` | `compile`, input execution, or fingerprinting threw, or the child exited without valid JSON | yes |
| `timeout` | parent `SIGKILL` after `--timeout-ms` | yes |
| `invariant` | after a successful compile, level storage is internally inconsistent | yes |
| `nondeterministic` | two identical compile+execute runs in one worker produced different fingerprints | yes |
| `replay-divergence` | undo-and-replay disagreed with the forward fingerprint | yes |

If `errorCount > 0`, the worker emits `compiler-error` and does not execute
inputs, check invariants, replay, or recompile.

### Fingerprint

`errorCount + "\n" + convertLevelToString()` after the forward input prefix.
For `compiler-error`, `fingerprint` is `"compiler-error:" + errorCount`.

### Level invariants

Checked only after `errorCount === 0`. Failure detail is the first broken rule:

- `level` is a non-null object
- `level.width > 0` and `level.height > 0`
- `level.n_tiles === level.width * level.height`
- `level.objects.length === level.n_tiles * STRIDE_OBJ`
- if `level.movements` exists, `level.movements.length === level.n_tiles * STRIDE_MOV`

### Undo/replay

Skipped when `replay` is false, the prefix is empty, or the result is already
`compiler-error` / `crash` / `invariant`.

1. Run the input prefix and record fingerprint `F1`.
2. Call `DoUndo(false, true)` once per prefix entry.
3. Apply the same prefix again.
4. If the new fingerprint is not `F1`, emit `replay-divergence`.

### Nondeterminism

After the replay check, compile and execute the same job a second time in the
same worker. If that fingerprint is not `F1`, emit `nondeterministic`.

### Failure signature

Used while shrinking so a timeout cannot collapse into an unrelated parser crash.

- `crash`: `crash:` + `error.name` + `:` + first line of `error.message`
- `invariant`: `invariant:` + `detail`
- any other kind: the kind string alone

### Shrinking

Line-oriented deletion. Split on `\n`, try deleting one line at a time, keep a
deletion when the worker result has the same `failureSignature`, restart the scan
after a successful deletion, and stop after `--shrink-budget` evaluations or a
full pass with no deletions.

### Artifacts

Interesting results write a directory under `--output`:

- `original.txt`
- `minimized.txt` (same as original when shrinking is off or finds nothing smaller)
- `report.json` (seed, fixture, mutator, worker result, signature, shrink stats)
- `regression.js` (copy-pasteable named fixture containing the minimized source)

Directory names are `sanitize(signature) + "_s" + seed + "_" + 4-digit index`.
Non-alphanumeric characters become `-`, and the signature part is truncated to 80
characters.

Write to `name.tmp` first, then rename over the destination so an interrupted run
does not leave a half-written case.

`regression.js` uses `JSON.stringify` for escaping and this shape:

```js
[
    "monster garden <seed> <index>",
    [<minimized source>, [], ""]
],
```

## Style and limits

This is an opt-in developer tool. It does not change the compiler, editor, browser
tests, fixture format, or production build. It uses built-in Node modules only and
keeps all new code within `src/tests/monster_garden/`, apart from a short addition
to the development guide and the output ignore rule.
