# PuzzleScript Static Analysis Design

## Summary

Build a tag-first static analyzer for compiled PuzzleScript games. The analyzer is a reporting and exploration tool, not an optimizer. It preserves original names, source lines, collision layers, rule order, rule groups, loops, win conditions, and level summaries, then adds tags and derived facts that can be hand-checked.

The first version focuses on four connected fact families:

- mergeability
- movement/action behavior
- count and layer invariants
- end-of-turn transient facts

The tool must separate `proved`, `candidate`, and `rejected` facts. Future solver or optimizer work may consume only `proved` facts. Candidate facts are for corpus mining, tuning, and hand-checking.

## Core Contract

The analyzer starts from compiled state as the source of truth. It does not rewrite source, canonicalize names, collapse objects, delete rules, remove layers, or alter levels. It builds an analyzer-owned tagged view named `ps_tagged`.

`ps_tagged` is both the internal analysis representation and the primary JSON output for hand-checking. It contains the whole game, not just rules:

- game metadata and game-level tags
- objects with original names, ids, layers, and tags
- properties, synonyms, aggregates, member sets, and tags
- collision layers with object lists and tags
- win conditions with object/property masks and tags
- level summaries with initial object/layer presence
- early and late rule sections preserving loop -> group -> rule hierarchy

Rules are never stripped from `ps_tagged`. Inert behavior is represented with tags. For example, a rule whose only command is `sfx0` remains present and may have:

```json
{
  "tags": {
    "inert_command_only": true,
    "solver_state_active": false
  }
}
```

Semantically active commands such as `cancel`, `again`, `restart`, `win`, and `checkpoint` remain active. Rules, groups, sections, and derived facts that involve `again` use a simple `has_again` tag. The analyzer does not simulate subsequent again turns.

## Tagged Representation

`ps_tagged.rule_sections` preserves execution shape:

```json
{
  "rule_sections": [
    {
      "name": "early",
      "loop_points": [],
      "groups": [
        {
          "id": "early_group_17",
          "group_number": 17,
          "tags": {},
          "rules": []
        }
      ]
    },
    {
      "name": "late",
      "loop_points": [],
      "groups": []
    }
  ]
}
```

Each rule stores its source line, direction, group, late/rigid/random flags, tags, commands, and cell-level terms for both sides.

Rule terms keep presence, absence, movement, and random-object choice distinct. A term has:

- `kind`: `present`, `absent`, or `random_object`
- `ref`: an object, property, object set, or ellipsis reference
- `movement`: `null`, `stationary`, `up`, `down`, `left`, `right`, `moving`, `action`, or `randomdir`

`present` with `movement: null` means an unqualified object/property term. It does not imply the explicit PuzzleScript `stationary` qualifier. `stationary` is reserved for explicit stationary movement constraints or effects.

`absent` represents `no X` and must not be treated as a movement state. `random_object` represents RHS `random X`; it selects an object, while `movement: "randomdir"` selects a direction for a present object. Analyses must keep those effects separate.

For example:

```puzzlescript
[ a | right b no c ] -> [ up a | b right c ]
```

is represented as:

```json
{
  "lhs": [
    [
      [{"kind": "present", "ref": {"type": "object", "name": "a"}, "movement": null}],
      [
        {"kind": "present", "ref": {"type": "object", "name": "b"}, "movement": "right"},
        {"kind": "absent", "ref": {"type": "object", "name": "c"}, "movement": null}
      ]
    ]
  ],
  "rhs": [
    [
      [{"kind": "present", "ref": {"type": "object", "name": "a"}, "movement": "up"}],
      [
        {"kind": "present", "ref": {"type": "object", "name": "b"}, "movement": null},
        {"kind": "present", "ref": {"type": "object", "name": "c"}, "movement": "right"}
      ]
    ]
  ]
}
```

When the compiler has already expanded a source property or produced a mask that does not map cleanly to one named object/property, the term may use:

```json
{
  "kind": "present",
  "ref": {
    "type": "object_set",
    "objects": ["A", "B"]
  },
  "movement": null
}
```

The rule also includes flattened summaries for queries, but these summaries do not replace the cell-level structure.

Tags attach at multiple levels:

- rule tags: `solver_state_active`, `inert_command_only`, `reads_action`, `writes_movement`, `object_mutating`, `movement_only`, `has_again`
- group tags: `movement_only`, `object_mutating`, `command_only`, `may_churn_objects`, `has_again`
- object tags: `present_in_all_levels`, `present_in_some_levels`, `present_in_no_levels`, `appears_in_wincondition`, `may_be_created`, `may_be_destroyed`, `static`, `count_invariant`
- collision-layer tags: `movement_possible`, `object_mutating`, `static`, `cosmetic_candidate`, `inert_candidate`
- game tags: `has_again`, `has_random`, `has_rigid`, `has_action_rules`, `has_autonomous_tick_rules`

Rule tags follow these implications:

| Tag | Definition | Implies / excludes |
| --- | --- | --- |
| `inert_command_only` | The rule has no object or movement effect and only queues solver-state-inert commands. | implies `command_only`; excludes `solver_state_active`, `object_mutating`, `movement_only` |
| `command_only` | The rule queues commands and has no object or movement effect. | may still be `solver_state_active` if a command is semantic |
| `object_mutating` | The rule may set, clear, or randomly choose an object at end of its application. | implies `solver_state_active` |
| `writes_movement` | The rule may set or clear movement, including explicit `stationary` and `randomdir`. | implies `solver_state_active` unless the whole rule is otherwise inert |
| `movement_only` | The rule writes movement and does not mutate objects or queue semantic commands. | implies `writes_movement`; excludes `object_mutating` |
| `solver_state_active` | The rule may affect board state, movement state, or a semantic command relevant to turn outcome. | excludes `inert_command_only` |
| `reads_action` | The LHS contains a present term with `movement: "action"`. | contributes to `has_action_rules` |
| `has_again` | The rule queues `again`. | taints single-turn boundary facts involving the same group writes |

Solver-state-inert commands for v1 are `sfx0` through `sfx10` and `message`. Semantic commands are `cancel`, `again`, `restart`, `win`, and `checkpoint`. `checkpoint` is not board-mutating, but it mutates restart/metagame state and must not be silently stripped by the analyzer.

Rigid rules are always treated as `solver_state_active` when they have any non-inert effect. V1 does not prove inertness inside rigid groups.

## Fact Families

Every derived fact has:

- `family`: one of the four analysis families
- `status`: `proved`, `candidate`, or `rejected`
- `subjects`: referenced objects, properties, layers, groups, or rules
- `tags`: concise factual tags such as `has_again` or `single_turn_only`
- `proof`: structured proof steps for proved facts
- `blockers`: concrete reasons for candidate or rejected facts
- `evidence`: ids and source lines from `ps_tagged`

### Mergeability

Mergeability starts from same-collision-layer object buckets. Buckets are split when objects differ in solver-visible roles or rule observation.

Rule observation is defined over LHS terms:

- A direct object term observes that object individually.
- A property or object-set term observes the whole expanded set, not each member individually, if every member of the candidate group is included the same way.
- Positive and negative observations both count. `[ no A ]` distinguishes `A`; `[ no Body ]` does not distinguish `BodyH` from `BodyV` if `Body = BodyH or BodyV`.
- Movement observations count. `[ right A ]` distinguishes `A`; `[ right Body ]` does not distinguish candidate members included in `Body`.

Win-condition roles are checked both directly and through expanded masks. The report should tag whether a role came from a direct object name, a property, or an expanded mask. Direct `some A` distinguishes `A`; `some Body` does not distinguish `BodyH` from `BodyV` if the expanded role is identical for both.

Objects can remain merge candidates when:

- they share a collision layer
- they have the same player/background/win-condition role
- rules do not observe them individually on the LHS
- rules observe them only through shared properties or equivalent masks
- RHS differences map within the same candidate group

RHS-only spawning of different variants is a `candidate` in v1, not `proved`, when all variants project to the same group and no later rule or win condition distinguishes them. The analyzer rejects or downgrades the fact if persistent consequences escape the candidate group.

### Movement And Action

Movement/action analysis tags:

- which rules and groups read movement qualifiers
- which rules and groups write movement qualifiers
- which collision layers can receive movement
- whether object modifications are movement-only
- whether any solver-active rule reads `action`
- whether the game has autonomous tick rules
- whether `action` can be proved to do nothing

For v1, `action_noop` is proved only under conservative conditions:

- no solver-active rule reads `action`
- every solver-active changing rule is gated by directional movement, or is otherwise proved inert for solver state
- action input cannot create directional movement through rule propagation
- no action turn can change object masks, trigger movement resolution that moves objects, or queue `cancel`, `restart`, `win`, or `again`

Games that use action as "wait one turn while autonomous rules run" must not be marked `action_noop`.

The proof uses a conservative movement-reachability fixpoint, not concrete board enumeration. The abstract domain is a set of possible `(layer, movement)` pairs plus taint flags for object mutation and semantic commands. The initial action turn contains `(player_layer, action)` for every player layer and no directional movement. A solver-active rule is considered reachable if its positive movement requirements are already possible; object-layout requirements are treated as satisfiable. RHS directional movement, `moving`, or `randomdir` adds movement facts. Any reachable object mutation, semantic command, autonomous rule with no movement gate, or directional movement creation rejects `action_noop`.

Rules in rigid groups and rules in groups that queue `again` are conservative blockers for proved `action_noop` unless they are already `inert_command_only`.

### Count And Layer Invariants

This family summarizes object, property, and layer count behavior:

- object may be created
- object may be destroyed
- object count is preserved
- property count is preserved even if members convert within the property
- layer is static
- layer changes only within a candidate object family
- layer is cosmetic or inert candidate

V1 should prefer conservative `candidate` output for count invariants unless the proof is direct from the tagged rule effects.

Object level-presence tags are not a single boolean. The report uses per-level presence plus aggregate tags:

- `present_in_all_levels`
- `present_in_some_levels`
- `present_in_no_levels`

Level summaries include aggregate object and layer presence for every level. Per-cell contents are optional debug output and are not required for v1 facts.

### End-Of-Turn Transients

The boundary for v1 is the end of one normal non-yielding turn: after early rules, movement resolution, and late rules finish, before considering any subsequent `again` turn.

The analyzer tags `again` but does not simulate again processing. Any rule with `again`, or any rule in a group where another rule has `again`, taints transient proofs for objects that group can write.

An object or layer can be `end_turn_transient` when it may appear during a turn but is proved absent or solver-irrelevant by the end of that same turn. V1 should prove only narrow cleanup patterns:

- object is not part of player/background/win-condition roles
- object is absent from initial levels, or level-start presence is separately accounted for
- object may be created earlier in the turn
- a later cleanup group clears it unconditionally
- no later group can set it
- loop structure does not allow the turn to stop before cleanup

Facts that depend on subsequent again processing are candidates or rejected, not proved. Rules inside `startloop`/`endloop` iterate to fixpoint; transient claims must account for loop re-entry, including the same group setting an object again after an apparent cleanup.

## Output

The default console output is a compact audit summary:

```text
game: limerick.txt
objects: 18, layers: 5, groups: 23
proved:
  action_noop: false
  movement_layers: [Player, Body]
candidates:
  mergeable: [PlayerBodyH, PlayerBodyV]
  transient_end_turn: [TrailMark]
rejected:
  inert_layer Background: not proved, level-visible/cosmetic only
```

The JSON output is the main artifact:

```json
{
  "schema": "ps-static-analysis-v1",
  "source": {},
  "ps_tagged": {},
  "facts": {
    "mergeability": [],
    "movement_action": [],
    "count_layer_invariants": [],
    "transient_boundary": []
  },
  "summary": {}
}
```

Useful v1 flags:

- `--out <path>` writes JSON
- `--family <name>` filters fact families
- `--game <substring>` filters source paths or titles
- `--include-ps-tagged` includes the full tagged view
- `--no-ps-tagged` emits facts and summaries only

JSON output should use original names by default. Because the compiler normalizes identifiers internally, tagged objects/properties/refs may also carry a `canonical_name` for lookup, but `name` and member lists are the original-case hand-checking surface. The report should include stable ids for rules, groups, objects, layers, and facts so evidence can point back into `ps_tagged`.

## Validation Plan

Use small fixture games to test proof patterns and false-positive traps:

- mergeable same-property reads
- mergeable same-property negative reads
- not mergeable due to individual LHS read
- not mergeable due to individual negative LHS read
- not mergeable because win condition distinguishes objects
- mergeability candidate with RHS-only variant spawning
- movement-only push rules
- action noop due to directional gating
- action not noop because autonomous rules tick
- action not noop because explicit action rule changes state
- action not noop because action reaches directional movement
- rigid rule blocks proved action/transient simplification
- again group taints transient proof
- count-preserved conversion
- count not preserved due to spawn/destroy
- per-level present-in-all/some/no level tags
- transient cleared in late cleanup
- not transient because later group sets after cleanup
- not transient because loop re-entry can set after cleanup
- inert command-only rule tagged but not removed
- checkpoint tagged as semantic command, not inert command-only

Assertions should check both the final fact and the evidence path, such as:

- rule line has `inert_command_only`
- same rule has `solver_state_active=false`
- fact references the expected rule/group/object ids
- no `proved` fact lacks proof and evidence

Batch corpus validation should run over `src/demo` and `src/tests/solver_tests`:

- valid games produce JSON
- compile failures are reported per source, not fatal to the batch
- candidate and rejected counts are summarized
- Limerick produces the expected body-variant merge candidate

No solver benchmarks are required for v1. This is static analysis only.

## Non-Goals

- No solver integration.
- No optimization or decanonicalization.
- No source rewriting.
- No object/layer merging in runtime state.
- No proof across drained `again` chains.
- No performance tuning beyond keeping the analyzer usable on the example corpus.
