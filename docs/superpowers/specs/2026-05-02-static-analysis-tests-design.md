# Static Analysis Test Design

## Summary

Add tests that check each static-analyzer claim at the level where that claim is meant to be trusted.

The analyzer currently emits two broad kinds of output:

- tagged structure, rule tags, game tags, and proof artifacts
- derived facts that may later be consumed by solvers or optimizers

Small static tests are enough for structural tags. Derived facts need stronger runtime checks. The main runtime technique will be metamorphic replay: solve or replay a known-solvable level, perturb or inspect the game state at full-turn boundaries according to the analyzer fact, and verify the known solution still behaves as promised.

The first implementation should stay small and deterministic. Corpus-scale fuzzing can come later after the replay harness is boring.

## Test Layers

### Static Fixture Tests

A fixture is a tiny purpose-built PuzzleScript source embedded in a node test. These tests compile the fixture, run `analyzeSource`, and assert exact tags or facts.

Use static fixtures for:

- `ps_tagged` shape
- rule tags
- group/game tags
- proof blockers for rejected facts
- edge cases that are hard to trigger reliably in real games

These tests should remain fast and deterministic. They are allowed to inspect analyzer internals such as `movement_pairs` because their purpose is to keep the proof machinery honest.

### Metamorphic Replay Tests

Replay tests use real engine state. They should:

1. Compile a game.
2. Load a known solvable level.
3. Solve it or use a checked-in known solution.
4. Replay the solution.
5. At every full-turn boundary, perturb or inspect state according to the analyzer fact.
6. Assert the same solution still reaches a win, or the inspected invariant still holds.

The full-turn boundary is after one input and its `again` drain. Perturbations should occur before the next input. This avoids testing a stronger property than the analyzer proves.

The replay harness must print `game`, `level`, `fact id`, `variant index`, and deterministic seed on failure.

### Solver-Visited-State Tests

For a few easily solvable levels, the solver can expose or callback on every generated state. That lets invariant checks cover many reachable states rather than only one solution path.

This is a stronger second tier for:

- count invariants
- layer-static invariants
- transient absence at turn boundaries

The first implementation can use solution replay only. Solver-visited-state checks should be a follow-up once the replay harness is stable.

## Claim Matrix

### `ps_tagged` Structure

Static test with one fixture containing:

- objects, synonyms, properties, and aggregate properties
- multiple collision layers
- win conditions
- early and late rules
- rule groups and loops
- terms using positive presence, `no`, movement, `action`, `stationary`, `random`, and `randomdir`
- inert commands and semantic commands

Assertions:

- original object names are preserved
- sections preserve `early`/`late`
- groups preserve source order
- rules preserve source lines and command lists
- cell terms distinguish `present`, `absent`, and `random_object`
- movement is separate from term polarity
- property/object-set expansion appears in `expanded_objects`

### Rule Tags

Static tests are enough for most rule tags.

Assertions:

- `inert_command_only`: `[ A ] -> sfx0` and `[ A ] -> message ...` are not solver-state active.
- `command_only`: command-only rules remain present even when inert.
- semantic command active: `cancel`, `again`, `restart`, `win`, and `checkpoint` are solver-state active.
- `object_mutating`: creation, deletion, random object creation, and collision-layer overwrite are active.
- `writes_movement`: directional RHS movement, movement clearing, and `randomdir` are detected.
- `movement_only`: movement writes without object mutation or semantic commands are movement-only.
- `reads_action`: a rule whose LHS contains `action`, such as `[ action Player ] -> [ Player Mark ]`, is tagged.
- `has_again`: rules and groups with `again` are tagged.
- `rigid_active`: rigid rules with non-inert effects are treated conservatively as active.

### Game And Group Tags

Static fixtures should assert:

- `has_again` is set when any rule queues `again`
- `has_random` is set for random rule groups, RHS `random X`, and `randomdir`
- `has_rigid` is set for rigid rules
- `has_action_rules` is set when a rule reads `action`
- `has_autonomous_tick_rules` is set for solver-active rules without input movement gates
- explicit `stationary` rules such as `[ stationary Robot ] -> [ randomDir Robot ]` count as autonomous, not input-gated

### Mergeability

Static tests:

- Candidate: same collision layer, same win role, no individual LHS observation, only shared property/object-set reads.
- Reject direct positive read: `[ BodyH ]`.
- Reject direct negation: `[ no BodyH ]`.
- Reject direct movement read: `[ right BodyH ]`.
- Reject partial property observation: a property term includes only one member of the candidate pair.
- Reject win distinction: `Some BodyH` versus `Some Body`.
- Deduplicate collision-layer entries so no self-merge facts are emitted.

Metamorphic replay:

- Target: `limerick`, `PlayerBodyH`/`PlayerBodyV`.
- Solve or use a known solution.
- Before each input, seeded-randomly swap objects within each mergeability class in every cell.
- Do this at turn boundaries, not inside a turn.
- Exclude any mergeability class containing a proved transient object.
- Assert the original solution still wins.

The test should not require the solver to rediscover the same solution. Solver output equality can be a later stricter check, but replay validity is the core property.

### Movement/Action Facts

`movement_pairs` is an internal proof artifact for `action_noop`. It represents reachable `(layer, movement)` pairs during an abstract action turn.

Static tests:

- initial player layers produce `layer:action`
- `action Player -> right Crate` makes crate-layer directional movement reachable
- `moving` requirements can be satisfied by cardinal movement
- `randomdir` expands to possible cardinal movement
- `stationary` does not count as an input movement gate

`action_noop` static tests:

- Prove: all solver-active changes are gated by normal directional movement and action cannot create new movement/effects.
- Reject direct action read: `[ action Player ] -> ...`
- Reject autonomous tick: `[ Robot ] -> [ randomDir Robot ]`
- Reject stationary autonomous tick: `[ stationary Robot ] -> [ randomDir Robot ]`
- Reject object mutation reachable from action.
- Reject movement creation reachable from action.
- Reject semantic commands reachable from action, especially `again`, `cancel`, `restart`, `win`, and `checkpoint`.
- Reject rigid active rules.

Runtime tests:

- For a proved `action_noop` game without explicit `noaction`, replay a known solution.
- At each turn boundary, clone state, apply `action`, drain `again`, and assert the board/session state matches the clone.
- Then continue replaying the original solution from the unmodified state.
- As a second check, inject `noaction` metadata and assert the original solution still replays to a win.

### Object Count Invariants

Static tests:

- Prove when no solver-active rule may affect the object.
- Reject deletion: `[ A ] -> []`.
- Reject explicit clear: `[ A ] -> [ no A ]`.
- Reject creation: `[ B ] -> [ B A ]`.
- Reject random object creation.
- Reject collision-layer overwrite by a sibling object.
- Reject rules that mention the object through a property when the expansion includes it.

Runtime tests:

- For proved object count facts, replay a known solution.
- Exclude objects that are also proved transient.
- Count object instances at level start.
- After every full turn plus `again` drain, assert the count is unchanged.

Follow-up solver-state tier:

- For a small set of easily solvable levels, run the solver and assert the same counts on all generated states.

### Layer Static Invariants

The current analyzer claim is only that a layer's occupancy does not change. It does not claim the layer is removable or irrelevant.

Static tests:

- Prove when no solver-active rule may affect any object in the layer.
- Do not prove when any object in the layer may be created, destroyed, randomly chosen, or overwritten.
- Cover background-like layers separately only as static occupancy, not removability.

Runtime tests:

- For proved `layer_N_static` facts, snapshot exact layer occupancy at level start.
- Replay a known solution.
- After every full turn plus `again` drain, assert the layer occupancy is bit-identical to the initial snapshot.

Future work:

- A stronger `solver_irrelevant_layer` or `removable_layer` fact could be tested by removing the layer and replaying solutions.
- That stronger fact is not part of this test plan.

### Transient Boundary Facts

Static tests:

- Prove early-created then late-cleared.
- Prove late-created then later-late-cleared.
- Prove empty RHS cleanup, e.g. `late [ Mark ] -> []`.
- Do not treat preserving rules as creators, even if they queue `again`.
- Reject objects present in any initial level.
- Reject objects used in win conditions.
- Reject creators with no later cleanup.
- Reject creators after cleanup.
- Reject creators in rigid rules.
- Reject creators in groups tainted by `again`.

Runtime tests:

- Target: `atlas shrank`, including `Shadowcrate`, `ShadowDoor`, `H_pickup`, `H_grav`, `H_step`, and `ShadowDoorO`.
- Replay a known solution.
- After every full turn plus `again` drain, assert every proved transient object is absent.
- This test should inspect turn-boundary state only.

### Level Presence Tags

Static tests:

- Multi-level fixture with one object present in all playable levels.
- One object present in only some playable levels.
- One object present in no playable levels.
- Include a message level and assert it does not distort playable-level presence counts.

## First Curated Runtime Targets

Use a small set first:

- `limerick`: mergeability perturbation for `PlayerBodyH`/`PlayerBodyV`
- `atlas shrank`: transient boundary checks
- `sokoban_basic` or `microban`: action-noop/noaction sanity, count invariants, static layers
- one synthetic action-noop game: direct clone/apply-action equality
- one synthetic layer-static game: exact layer occupancy check

If a real-game solution is expensive to find, the test may store a known compact solution string rather than solving every run. The replay harness should still be able to ask the solver for a solution when desired.

## Harness Components

### Analyzer Fact Loader

Input:

- PuzzleScript source path or inline fixture
- optional family filter

Output:

- `ps_tagged`
- facts grouped by family and status
- helper lookup functions for objects, layers, and transient exclusions

### Replay Engine

Responsibilities:

- compile and load a level with deterministic seed
- apply one input
- drain `again`
- capture and restore state
- read and write cell object masks
- recompute row/column masks and solver hashes after mutations
- compare board/session state for action-noop checks
- report precise failure context

### Deterministic PRNG

Use a small local deterministic PRNG seeded from:

```text
game path + level index + fact id + variant index
```

The seed string must be printed on failure.

### Perturbation Helpers

Mergeability:

- for each cell, if it contains one object from a merge class, replace it with a seeded-random member of the same class
- preserve the collision layer invariant by clearing the class mask before setting the replacement bit
- skip classes containing proved transient objects

Count:

- count object bits across all cells

Layer static:

- snapshot all bits for the layer across all cells
- compare exact layer bits after each turn

Transient:

- assert object bit is absent across all cells at boundary

Action noop:

- clone state at boundary
- apply action to clone
- drain `again`
- compare clone against original boundary state

## Test Commands

Add a fast default node test:

```sh
node src/tests/ps_static_analysis_metamorphic_node.js --fast
```

Add an optional slower corpus target later:

```sh
node src/tests/ps_static_analysis_metamorphic_node.js --corpus src/demo src/tests/solver_tests --seed static-analysis
```

The fast test should be suitable for `npm run test:node` once stable. The corpus mode should not be part of the default suite until runtime is predictable.

## Non-Goals

- Do not prove layer removability in this pass.
- Do not require solvers to rediscover identical solutions.
- Do not fuzz the full corpus in default tests.
- Do not perturb mid-turn or inside `again` processing.
- Do not let candidate facts drive optimizer behavior. Candidate facts may be used for exploratory tests only when explicitly marked.

## Success Criteria

- Every analyzer fact family has at least one positive and one negative static test.
- Every proved runtime-relevant fact family has at least one metamorphic replay test.
- Limerick-style spawned mergeable objects are tested by turn-boundary perturbation.
- Atlas-style transients are tested on real gameplay state.
- Action-noop is tested by direct clone/action comparison, not only by solver output.
- Tests are deterministic and print enough context to reproduce failures.
