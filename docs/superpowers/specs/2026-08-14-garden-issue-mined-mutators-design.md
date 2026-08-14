# Garden Issue-Mined and Semantics-Preserving Mutators

## Purpose

The Compiler Monster Garden has 43 mutators. Almost all of them damage a program
and ask whether the compiler errors instead of crashing. This design adds two
groups of mutators that the current set does not reach:

1. **Issue-mined mutators**, derived from families of bugs that recur in this
   project's own closed issues.
2. **Semantics-preserving mutators**, which produce a still-valid game that must
   behave identically, checked against the baseline run the garden already
   performs.

It extends `2026-08-14-compiler-monster-garden-design.md`. It does not change the
corpus, the worker protocol, or the command-line contract beyond one new result
kind.

## Background: why mine issues, and what to mine for

IssueMut extracts mutation operators from historical bug reports. MetaMut
generates semantics-aware mutators for C/C++ compilers. Both were considered.

Mining this project's issues for **reproducers** is worthless. Fixed bugs already
have fixtures in `testdata.js` and `errormessage_testdata.js`, `loadCorpus` loads
both, and the garden already mutates them. Mining for **single operators** is also
low-yield: the existing 43 mutators already harvested the obvious ones, so
`legend-cycle` is issue #243, `keyword-as-name` is #1109, `invalid-viewport` is
#885. An automated mining pipeline would mostly rediscover work already done, and
is therefore rejected.

What survives is **clusters**. Of 925 closed issues, 198 are bug-shaped and not
about the editor or UI, and within those the same regions of the language recur.
A single fixture records that one input is now fine. Four independent reports
about the same construct say the subsystem is structurally fragile, which is a
claim no fixture can make. Clusters are what this design mines, by reading, not
by pipeline.

## Group A: issue-mined mutators

Each was checked against the current implementations and is genuinely absent.

| Mutator | Behaviour | Issues |
| --- | --- | --- |
| `no-x-with-x` | Insert `no <Obj>` into a rule cell that already contains `<Obj>`; a variant emits `no X no X` in one cell | #1169, #1136, #1071, #762 |
| `relative-direction-cell` | Attach `perpendicular`, `parallel`, `vertical`, `horizontal`, `orthogonal`, `moving` or `stationary` to a cell object, alone and against a conflicting rule-level direction prefix | #682, #498, #496, #941 |
| `same-layer-cell` | Place two objects that share a collision layer into one rule cell | #735, #605, #734 |
| `property-in-concrete-slot` | Replace a concrete object reference with an aggregate or property name in rules, win conditions, sounds or collision layers | #929, #495, #812, #824 |
| `rigid-prefix` | Make a rule or rule group `rigid`, alone and combined with `late` | #952, #1118, #869 |
| `sprite-matrix-resize` | Resize a sprite matrix away from 5x5, including ragged rows | #973, #927 |
| `restart-again-message` | Combine restart, again, message and checkpoint in one rule and in the input tape | #774, #981, #341 |
| `multi-fault` | Apply 2 to 4 existing mutators to one fixture | #1012, #1002, #980 |

Notes on overlap with existing mutators:

- `injectNo` prepends a hardcoded `no Player` to the first cell of a rule. It
  never negates an object that is also present in that cell, which is what the
  `X no X` cluster is about.
- `directionPrefixSalad` prepends one fixed string to a rule. It never places a
  relative qualifier on an individual cell object.
- `layerDoubleBook` edits the COLLISIONLAYERS section only. It never puts two
  same-layer objects into a rule cell.
- `spriteMatrixNoise` flips pixels within a matrix and never changes its shape.
- `background-as-aggregate` and `sound-on-property` are two fixed instances of the
  general transformation `property-in-concrete-slot` performs.

`multi-fault` matters more than its size suggests. Every existing mutator injects
exactly one fault, so error-count thresholds are structurally unreachable: the
"Too many errors/warnings; noping out" path, the abort-versus-warn wording in
#1002, and the realtime throttle failure in #980 all require several errors at
once. It composes existing mutators and needs no new source manipulation.

`multi-fault` draws only from mutators that do not declare `equivalence`. Mixing
a damaging mutator into a semantics-preserving one would make the resulting
program neither, and there would be nothing meaningful to assert about it.

## Group B: semantics-preserving mutators

### The oracle

Every trial already runs the unmutated fixture as a baseline with identical
inputs (`run.js:186`), and both runs return a `fingerprint` covering the board,
raw `level.objects` and `level.movements`, `winning`, `curlevel`, message state
and rng state. For a semantics-preserving mutant, the baseline and mutant
fingerprints must agree. The comparison is a pure function in `garden.js` over
two values already in hand. No new worker code and no additional child process.

This turns on machinery that is currently built but unreachable. `run.js:191`
passes `baselineOracleFields(...)` to the baseline job, but `run.js:192` calls
`evaluateMutant(mutant, options)` with no oracle argument, so the mutant job
carries no `expectedOutput` and no `expectedErrors`. `checkErrorOracle` returns
at `worker.js:312` and the board comparison at `worker.js:493` is skipped. Today
`semantic-mismatch` can only fire on the baseline. That is correct while every
mutator changes behaviour; it stops being correct once mutators preserve it.

### Equivalence levels

Mutators gain an optional `equivalence` field, absent on all 43 existing ones.

- **`full`** — the entire fingerprint must be identical. For mutators that do not
  disturb object identity.
- **`board`** — only the board may be compared. Reordering the OBJECTS section
  renumbers object bits and changes `level.objects` wholesale while the rendered
  board is unchanged.

A `board` mutator may also supply `normalise`, because `convertLevelToString`
(`src/js/debug.js:16`) emits object *names* from `state.idDict`, sorted per cell.
A consistent rename therefore changes the board text even though the game is
identical. The normaliser for `rename-object` maps new names back to old and
re-sorts each cell, which is exact. The cell-numbering scheme is unaffected,
since it depends on the partition of cells rather than on names.

### The mutators

| Mutator | Behaviour | Equivalence |
| --- | --- | --- |
| `rename-object` | Consistently rename one object across every section | `board` + normaliser |
| `reorder-objects` | Permute OBJECTS entries | `board` |
| `reorder-winconditions` | Permute WINCONDITIONS lines | `full` |
| `reorder-sounds` | Permute SOUNDS lines | `full` |
| `inline-legend-synonym` | Replace uses of a simple `A = B` alias with `B` | `full` |
| `add-legend-alias` | Introduce an alias and route some references through it | `full` |
| `add-unreachable-rule` | Declare an object placed in no level, add a rule matching only it | `full` |
| `comment-reflow` | Insert a valid comment inside a rule's brackets | `full` |

`comment-reflow` is the legal counterpart to `odd-whitespace`, which injects
whitespace that must be rejected. Placing the comment inside the brackets also
probes issue #1128, "bad error if parenthetical inside rule".

`rename-object` must never rename `Background` or `Player`. The engine requires
both, so renaming either breaks the game and every resulting hit would be noise.

`inline-legend-synonym` replaces the alias definition with a blank line rather
than deleting it, so the mutant stays line-aligned with the fixture and paired
shrinking applies.

`append-unreached-level` was considered and rejected. Winning a level advances
`curlevel`, so appending a level after the last one changes what happens when
the player wins it, from game-over to level-advance. It is not reliably
semantics-preserving.

`rename-object` is the most likely to pay out. It drags the whole symbol table
through a transformation that must be invisible, and #824, #821 and #789 all
report trouble in that area.

### Randomness exclusion

Games using `random` or `randomdir` draw from object sets whose iteration order
depends on bit assignment, so `reorder-objects` can legitimately change a draw.
Semantics-preserving mutators skip any fixture whose source matches
`/\brandom(dir)?\b/i`, reported as `skipped` in the usual way.

This guard is not optional. Without it the semantics half emits a steady drip of
false monsters, and a tool whose value rests on a hit meaning something stops
being trusted.

## Result kind: `equivalence-break`

A new interesting kind. It is reported when the mutator declares `equivalence`,
the baseline is healthy, and either:

- both runs are `ok` but the fingerprints differ after normalisation; or
- the baseline is `ok` and the mutant is `compiler-error` — a valid
  transformation that broke compilation.

When the baseline is not healthy, no claim is made, consistent with the existing
harness-honesty rule that the garden must not blame a mutation for a
pre-existing failure.

Plumbing: `run.js:119` counts, `garden.js:1268` `isInteresting`, `garden.js:1394`
interesting kinds, `garden.js:1606` summary line.

## Shrinking

`shrinkInteresting` (`garden.js:1324`) reduces `mutant.source` alone while holding
`failureSignature` fixed. An equivalence-break is a relation between two sources,
so deleting a line from the mutant but not the baseline manufactures a divergence
for the wrong reason.

The shrinker carries the baseline source alongside the mutant source and deletes
the same line index from both, re-checking that the pair still diverges. This is
valid only when the mutant is line-aligned with the fixture, detected
automatically by equal line counts. It covers `rename-object`,
`inline-legend-synonym` and `comment-reflow`. For mutators that add or reorder
lines, shrinking is skipped and the artifact records that it was skipped, rather
than shrinking wrongly.

Artifacts for `equivalence-break` store both the baseline source and the mutant
source, so the pair can be reproduced.

## Testing

- Determinism and applicability tests for each of the 16 new mutators, in the
  existing `tests.js` style: a fixed seed produces a fixed mutation, and a fixture
  with no target returns `null`.
- Unit tests for the equivalence comparator: identical fingerprints yield no
  break; a renamed board matches after normalisation; a genuinely divergent board
  reports a break; an unhealthy baseline suppresses the claim.
- Unit tests for paired shrinking, including the line-alignment check and the
  skip path.
- A test that the randomness exclusion fires on a `random`-using fixture.
- An end-to-end garden run restricted to the semantics-preserving mutators that
  **reports zero equivalence-breaks against the current corpus**.

That last test inverts the usual fuzzing signal-to-noise ratio. The malformity
half produces findings that need triage; this half should be silent, and any hit
is a real bug.

The full Node test suite runs after every change, and the garden's own runtime is
checked for regression before any of this is considered done.

## Out of scope

- An automated issue-mining pipeline. Rejected above on yield.
- Metamorphic relations with a predictable delta, such as mirroring a level
  together with its input tape, or rotating the board and the directions. These
  need new oracle machinery rather than reusing the baseline comparison, and are
  a separate design.
- Adding gallery or archive games to the corpus (issue #590).
