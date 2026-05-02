# Static Analysis Test Design

## Summary

Add tests for every static-analyzer output claim. The test plan is organized by analyzer output surface rather than by test style, so it can be used as an implementation checklist.

Each tag or fact entry uses this format:

```md
### `tag_or_fact_name` [current|planned]

Description: what the analyzer claims.

Static tests:
- exact tiny-game assertions needed

Runtime/metamorphic tests:
- replay/solver-state checks needed, or "none"

Notes:
- caveats, current limitations, or future optimizer relevance
```

`current` means the analyzer emits this tag or fact today. `planned` means the older static-analysis design calls for the tag/fact, but the current analyzer does not emit it yet. Planned entries are documentation for future work, not acceptance criteria for the first implementation.

## `ps_tagged` Structural Claims

### `objects` [current]

Description: `ps_tagged.objects` preserves compiled object ids, original display names, canonical names, collision layer ids, and per-object tags.

Static tests:
- Compile a fixture with multiple objects, aliases, properties, and collision layers.
- Assert object names preserve original case/display names.
- Assert canonical names match compiler-normalized names.
- Assert layer ids match collision layer membership.

Runtime/metamorphic tests:
- None. This is structural output.

Notes:
- These assertions should live in static analyzer node tests, not replay tests.

### `properties` [current]

Description: `ps_tagged.properties` exposes synonyms and properties with expanded member lists.

Static tests:
- Fixture with one synonym and one property containing at least two objects.
- Assert `kind` is `synonym` or `property` as appropriate.
- Assert `members` are expanded and display names are preserved.

Runtime/metamorphic tests:
- None.

Notes:
- Property expansion feeds mergeability and rule-term `expanded_objects`; keep at least one shared fixture with both paths.

### `collision_layers` [current]

Description: collision layer summaries preserve layer order, object membership, canonical object names, and layer tags.

Static tests:
- Fixture with three layers, including a multi-object layer.
- Assert layer ids and membership order.
- Assert duplicate object entries are deduplicated and do not create self-merge facts.

Runtime/metamorphic tests:
- None for the structure itself.

Notes:
- Layer-static facts have their own runtime tests below.

### `winconditions` [current]

Description: win conditions preserve quantifier, subject set, target set, and the `plain` tag when the condition has no target object.

Static tests:
- Fixture with `Some A`, `All A on B`, and `No A`.
- Assert plain winconditions have empty `targets` and `tags.plain === true`.
- Assert `on` winconditions expose both subject and target names.

Runtime/metamorphic tests:
- None.

Notes:
- Wincondition role equality is tested under mergeability.

### `levels` [current]

Description: level summaries expose playable/message level kind, dimensions, aggregate object presence, aggregate layer presence, and level tags.

Static tests:
- Multi-level fixture with a message level.
- Assert message levels do not count as playable levels for object presence tags.
- Assert playable levels expose width, height, `objects_present`, and `layers_present`.

Runtime/metamorphic tests:
- None.

Notes:
- Per-cell level contents are not currently required by the analyzer output.

### `rule_sections` [current]

Description: rule sections preserve early/late split, group hierarchy, loop summaries, rule ids, source lines, direction, flags, commands, LHS/RHS cells, and summaries.

Static tests:
- Fixture with early rules, late rules, plus-continuation groups, startloop/endloop, rigid rules, random rules, and commands.
- Assert section names are `early` and `late`.
- Assert group order and rule order match execution order.
- Assert loop summaries reference the expected group ids.
- Assert command arrays preserve inert and semantic commands.

Runtime/metamorphic tests:
- None.

Notes:
- Rule/group tags below should reuse this kind of fixture where practical.

### `rule terms` [current]

Description: cell terms keep polarity, reference, and movement separate.

Static tests:
- Fixture with `[ A | right B no C ] -> [ up A | B right C ]`.
- Assert positive terms use `kind: "present"`.
- Assert `no C` uses `kind: "absent"` with `movement: null`.
- Assert RHS `random D` uses `kind: "random_object"`.
- Assert `randomdir D` is a present term with `movement: "randomdir"`.
- Assert `stationary`, `moving`, cardinal movement, and `action` are represented as movements, not polarity.

Runtime/metamorphic tests:
- None.

Notes:
- This is the foundation for all higher-level tag tests.

## Rule Tags

### `command_only` [current]

Description: the rule queues commands and has no object or movement effect. The RHS may be absent, or may repeat the matched cell contents without changing objects or movements.

Static tests:
- `[ A ] -> sfx0` is command-only.
- `[ A ] -> [ A ] sfx0` is command-only.
- `[ right A ] -> [ right A ] sfx0` is command-only.
- `[ A ] -> checkpoint` is command-only and still solver-state active.
- `[ A ] -> [ B ] sfx0` is not command-only.
- `[ A ] -> [ right A ] sfx0` is not command-only.
- `[ right A ] -> [ A ] sfx0` is not command-only.

Runtime/metamorphic tests:
- None.

Notes:
- This tag does not imply inertness by itself.

### `inert_command_only` [current]

Description: the rule has no object or movement effect and queues only solver-state-inert commands.

Static tests:
- `[ A ] -> sfx0` is inert command only.
- `[ A ] -> message ...` is inert command only.
- `[ A ] -> checkpoint` is not inert command only.
- `[ A ] -> [ A B ] sfx0` is not inert command only.

Runtime/metamorphic tests:
- None.

Notes:
- Inert commands currently include `message` and `sfx0` through `sfx10`.

### `object_mutating` [current]

Description: the rule may change object occupancy by setting, clearing, randomly choosing, deleting, creating, or overwriting a collision layer object.

Static tests:
- Creation: `[ A ] -> [ A B ]`.
- Deletion: `[ A ] -> []`.
- Explicit clear: `[ A ] -> [ no A ]`.
- Random object: `[ A ] -> [ random B ]`.
- Collision-layer overwrite: `[ A ] -> [ B ]` where `A` and `B` share a layer.
- Negative case: `[ A ] -> [ A ]` is not object-mutating.

Runtime/metamorphic tests:
- None directly.

Notes:
- Count and layer facts consume this tag and get runtime checks.

### `writes_movement` [current]

Description: the rule may set or clear movement state.

Static tests:
- RHS cardinal movement: `[ A ] -> [ right A ]`.
- RHS `randomdir`: `[ A ] -> [ randomdir A ]`.
- Movement clear: `[ right A ] -> [ A ]`.
- Negative case: `[ A ] -> [ A ]`.

Runtime/metamorphic tests:
- None directly.

Notes:
- Movement/action facts consume this tag.

### `movement_only` [current]

Description: the rule writes movement, does not mutate objects, and does not queue semantic commands.

Static tests:
- `[ A ] -> [ right A ]` is movement-only.
- `[ A ] -> [ right A ] again` is not movement-only.
- `[ A ] -> [ right A B ]` is not movement-only.

Runtime/metamorphic tests:
- None directly.

Notes:
- Group-level `movement_only` depends on this rule tag and absence of group object mutation.

### `reads_action` [current]

Description: the LHS contains a present term with `movement: "action"`, meaning a rule directly observes the action button.

Static tests:
- `[ action Player ] -> [ Player Mark ]` sets `reads_action`.
- `[ Player ] -> [ action Player ]` does not set `reads_action`.
- `[ up Player ] -> [ Player ]` does not set `reads_action`.

Runtime/metamorphic tests:
- None directly.

Notes:
- `reads_action` contributes to `has_action_rules` and rejects `action_noop`.

### `has_again` [current]

Description: the rule queues `again`.

Static tests:
- `[ A ] -> [ A ] again` sets rule `has_again`.
- `[ A ] -> [ A ]` does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Group `has_again` and transient taint depend on this.

### `solver_state_active` [current]

Description: the rule may affect solver-visible board state, movement state, or semantic command state.

Static tests:
- Object mutation, movement write, `again`, `cancel`, `restart`, `win`, and `checkpoint` are active.
- Inert command-only rules are not active.
- Pure no-op replacement `[ A ] -> [ A ]` without commands is not active.

Runtime/metamorphic tests:
- None directly.

Notes:
- This is a high-impact tag. Static tests should cover every semantic command.

### `rigid_active` [current]

Description: a rigid rule with any non-inert effect is conservatively active.

Static tests:
- Rigid movement or object-mutating rule gets `rigid_active`.
- Rigid inert command-only rule does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Rigid active rules block proved `action_noop` and transient facts.

## Rule Group Tags

### `has_again` [current]

Description: the group contains at least one rule with `has_again`.

Static tests:
- Plus-continuation group with one `again` rule and one non-`again` rule sets group `has_again`.
- Group with no `again` rules does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Transient facts use group `has_again` as a conservative taint for group writes.

### `object_mutating` [current]

Description: at least one rule in the group mutates objects.

Static tests:
- Group with one object-mutating rule sets the tag.
- Movement-only group does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Used to derive group-level movement-only behavior.

### `movement_only` [current]

Description: at least one rule writes movement and no rule in the group mutates objects.

Static tests:
- Group of movement-only rules sets the tag.
- Group containing both movement and object mutation does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Semantic commands in rules keep individual rules from being movement-only.

### `command_only` [current]

Description: every rule in the group is command-only.

Static tests:
- Group with two command-only rules sets the tag.
- Group with one replacement rule does not.

Runtime/metamorphic tests:
- None.

Notes:
- A command-only group may still be solver-state active if its commands are semantic.

### `solver_state_active` [current]

Description: at least one rule in the group is solver-state active.

Static tests:
- Group with an active semantic command sets the tag.
- Group with only inert commands does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Used in future group-level optimizer decisions.

### `may_churn_objects` [planned]

Description: planned tag for groups that may repeatedly create/destroy/convert objects without changing the final solver-visible projection.

Static tests:
- Once emitted, use a loop/group fixture with temporary markers that churn and clean up.
- Include a negative fixture where object churn survives the group boundary.

Runtime/metamorphic tests:
- After emission, replay a known solution and assert the projected state is stable at the intended boundary.

Notes:
- Not emitted today. Do not add failing tests until the analyzer implements it.

## Object Tags

### `present_in_all_levels` [current]

Description: the object appears in every playable level.

Static tests:
- Multi-level fixture where object `A` appears in all playable levels.
- Include a message level and assert it does not affect the result.

Runtime/metamorphic tests:
- None.

Notes:
- This is aggregate level metadata, not a turn invariant.

### `present_in_some_levels` [current]

Description: the object appears in some but not all playable levels.

Static tests:
- Multi-level fixture where object `B` appears in exactly one of two playable levels.
- Assert message levels are ignored.

Runtime/metamorphic tests:
- None.

Notes:
- The tag should be false when an object appears in all levels or no levels.

### `present_in_no_levels` [current]

Description: the object appears in no playable levels.

Static tests:
- Multi-level fixture with declared object `C` absent from all playable levels.
- Assert message levels do not count as absence/presence evidence.

Runtime/metamorphic tests:
- None.

Notes:
- Transient facts require this tag.

### `may_be_created` [current]

Description: some solver-active rule may affect the object, so the analyzer cannot prove the object is never created.

Static tests:
- `[ A ] -> [ A B ]` marks `B` as may-be-created.
- `[ A ] -> [ B ]` where `A`/`B` share a layer marks both conservatively.
- No active writer leaves the tag false.

Runtime/metamorphic tests:
- None directly.

Notes:
- Current implementation sets `may_be_created` and `may_be_destroyed` from the same conservative writer set.

### `may_be_destroyed` [current]

Description: some solver-active rule may affect the object, so the analyzer cannot prove the object is never destroyed.

Static tests:
- `[ A ] -> []` marks `A` as may-be-destroyed.
- Collision-layer overwrite marks siblings conservatively.
- No active writer leaves the tag false.

Runtime/metamorphic tests:
- None directly.

Notes:
- Current implementation is intentionally conservative.

### `count_invariant` [current]

Description: no solver-active rule may affect the object, so the object's count is preserved.

Static tests:
- Prove when no active rule mentions or writes the object/layer.
- Reject deletion, explicit clear, creation, random creation, property expansion involving the object, and sibling collision-layer overwrite.

Runtime/metamorphic tests:
- Replay known solutions and count non-transient proved-invariant objects after each full turn plus `again` drain.
- Follow-up: check all generated states for a few easily solvable levels.

Notes:
- Exclude objects that also have proved transient facts from the runtime count check.

### `appears_in_wincondition` [planned]

Description: planned object tag for objects that appear directly or through expansion in any win condition.

Static tests:
- Once emitted, use direct object winconditions and property-based winconditions.
- Include negative object absent from all winconditions.

Runtime/metamorphic tests:
- None directly.

Notes:
- Current transient analysis checks wincondition participation internally but does not emit this object tag.

### `static` [current]

Description: object cell occupancy is unchanged across solver-active turns.

Static tests:
- Prove for a wall-like object that is unaffected by solver-active rules.
- Reject player objects because input may apply movement to them.
- Reject objects that receive movement directly, through a property/synonym, or through an object-set aggregate.
- Reject objects created, destroyed, or overwritten directly, through a property/synonym, or through an object-set aggregate.
- Reject objects on a collision layer where a solver-active rule may create or randomly choose another object on that layer.

Runtime/metamorphic tests:
- Replay a known solution.
- Snapshot proved-static object occupancy at level start.
- After the solved end state, assert each proved-static object's occupancy is identical to the initial snapshot.

Notes:
- This is stricter than `count_invariant`; a player can have invariant count while still not being static.
- The current proof is intentionally conservative around collision-layer creation.

## Collision Layer Tags

### `static` [current]

Description: no solver-active rule may affect any object in the layer, so exact layer occupancy should stay unchanged.

Static tests:
- Prove for a layer with no active writers.
- Do not prove when any layer object may be created, destroyed, randomly chosen, or overwritten.
- Include a background-like layer as static occupancy, not removability.

Runtime/metamorphic tests:
- Snapshot exact layer occupancy at level start.
- Replay a known solution.
- After every full turn plus `again` drain, assert layer bits are identical to the initial snapshot.

Notes:
- This does not mean the layer is removable or irrelevant.

### `movement_possible` [planned]

Description: planned tag for layers that can receive movement during a turn.

Static tests:
- Once emitted, use direct movement writes and propagated movement writes.
- Include negative layer that never receives movement.

Runtime/metamorphic tests:
- Optional: replay solutions and observe whether movement buffers for the layer remain clear at turn boundaries.

Notes:
- Movement buffers should be clear at turn boundaries, so runtime checks must inspect the right moment if added.

### `object_mutating` [planned]

Description: planned layer tag for layers where some object in the layer may be written by a solver-active rule.

Static tests:
- Once emitted, prove for a layer containing any created/destroyed/overwritten object.
- Negative fixture with only movement changes.

Runtime/metamorphic tests:
- None directly.

Notes:
- Current layer facts expose this indirectly through `layer_N_static` blockers.

### `cosmetic_candidate` [planned]

Description: planned tag for layers that appear likely cosmetic but are not yet proved removable.

Static tests:
- Once emitted, use a layer never read by rules/winconditions and never written.
- Negative fixture where a rule reads the layer.

Runtime/metamorphic tests:
- None until the tag has a precise semantic contract.

Notes:
- Candidate tags are exploratory and must not drive optimizer behavior.

### `inert_candidate` [planned]

Description: planned tag for layers that may be inert for solver state.

Static tests:
- Once emitted, use a layer absent from rules, winconditions, player roles, and semantic commands.
- Negative fixture where the layer participates in a rule or wincondition.

Runtime/metamorphic tests:
- Future stronger test may remove the layer and replay known solutions.

Notes:
- Not part of the current `layer.static` claim.

## Game Tags

### `has_again` [current]

Description: at least one rule queues `again`.

Static tests:
- Fixture with one `again` rule sets the tag.
- Fixture with no `again` rules does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Runtime tests for transient facts must drain `again`.

### `has_random` [current]

Description: the game uses random rule groups or RHS `random X` object selection.

Static tests:
- Random rule group sets the tag.
- RHS `random X` sets the tag.
- `randomdir` alone does not set the current tag.
- Deterministic fixture does not set the tag.

Runtime/metamorphic tests:
- None directly.

Notes:
- `randomdir` is a movement qualifier and is tested under movement/action facts.

### `has_rigid` [current]

Description: at least one rule is rigid.

Static tests:
- Rigid rule fixture sets the tag.
- Non-rigid fixture does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- Rigid rules conservatively block some proved facts.

### `has_action_rules` [current]

Description: at least one rule reads `action` on the LHS.

Static tests:
- `[ action Player ] -> ...` sets the tag.
- RHS-only `action` does not.

Runtime/metamorphic tests:
- None directly.

Notes:
- This is a direct game-level summary of rule `reads_action`.

### `has_autonomous_tick_rules` [current]

Description: at least one solver-active rule can run without input movement requirements.

Static tests:
- `[ Robot ] -> [ randomDir Robot ]` sets the tag.
- `[ stationary Robot ] -> [ randomDir Robot ]` sets the tag.
- `[ moving Robot ] -> [ Robot ]` does not count as autonomous if movement-gated.

Runtime/metamorphic tests:
- None directly.

Notes:
- Autonomous tick rules reject `action_noop`.

## Wincondition Tags

### `plain` [current]

Description: the wincondition is a plain condition such as `Some A`, `All A`, or `No A`, not an `on` condition with a target set.

Static tests:
- `Some A` has `tags.plain === true` and empty `targets`.
- `All A on B` has `tags.plain === false` and exposes target `B`.

Runtime/metamorphic tests:
- None.

Notes:
- Plain winconditions matter for solver heuristics and future fact grouping, but not for the first replay harness.

## Fact Families

### `mergeability` [current]

Description: same-layer object pairs are candidates when rules and winconditions do not distinguish them individually.

Static tests:
- Candidate: same collision layer, same win role, no individual LHS observation, shared property/object-set reads only.
- Reject direct positive read: `[ BodyH ]`.
- Reject direct negation: `[ no BodyH ]`.
- Reject direct movement read: `[ right BodyH ]`.
- Reject partial property observation.
- Reject win distinction: `Some BodyH` versus `Some Body`.
- Assert duplicate collision-layer entries do not emit self-merge facts.

Runtime/metamorphic tests:
- Target `limerick`, `PlayerBodyH`/`PlayerBodyV`.
- Solve or use a known solution.
- Before each input, seeded-randomly swap objects within each mergeability class in every cell.
- Perturb only at turn boundaries.
- Exclude classes containing proved transient objects.
- Assert the original solution still wins.

Notes:
- Do not require the solver to rediscover the same solution.

### `movement_action.movement_pairs` [current]

Description: internal proof artifact listing reachable `(layer, movement)` pairs during the abstract action-turn reachability analysis.

Static tests:
- Initial player layers produce `layer:action`.
- `action Player -> right Crate` makes crate-layer directional movement reachable.
- `moving` requirements can be satisfied by cardinal movement.
- `randomdir` expands to possible cardinal movement.
- `stationary` does not count as an input movement gate.

Runtime/metamorphic tests:
- None directly.

Notes:
- This is tested because it supports `action_noop`; it is not a standalone optimizer claim.

### `movement_action.action_noop` [current]

Description: action input can be proved to have no solver-visible effect.

Static tests:
- Prove when all solver-active changes are gated by normal directional movement and action cannot create effects.
- Reject direct action read.
- Reject autonomous tick.
- Reject stationary autonomous tick.
- Reject object mutation reachable from action.
- Reject movement creation reachable from action.
- Reject semantic commands reachable from action.
- Reject rigid active rules.

Runtime/metamorphic tests:
- For a proved `action_noop` game without explicit `noaction`, replay a known solution.
- At each boundary, clone state, apply `action`, drain `again`, and assert the clone matches the original boundary state.
- Inject `noaction` metadata and assert the original solution still replays to a win.

Notes:
- Solver output equality is optional and stricter than the core claim.

### `count_layer_invariants.object_*_count_preserved` [current]

Description: object count is preserved because no solver-active rule may affect the object.

Static tests:
- Prove when no active rule may affect the object.
- Reject deletion, explicit clear, creation, random creation, property-expanded writes, and sibling collision-layer overwrite.

Runtime/metamorphic tests:
- Replay a known solution.
- Exclude proved transient objects.
- Count object instances at level start.
- After every full turn plus `again` drain, assert the count is unchanged.
- Follow-up: assert on all generated states for a few easily solvable levels.

Notes:
- Runtime checks should use proved facts only.

### `count_layer_invariants.layer_*_static` [current]

Description: exact layer occupancy is preserved because no solver-active rule may affect any object in the layer.

Static tests:
- Prove when no active rule may affect any layer object.
- Do not prove when any layer object may be created, destroyed, randomly chosen, or overwritten.

Runtime/metamorphic tests:
- Snapshot exact layer occupancy at level start.
- Replay a known solution.
- After each full turn plus `again` drain, assert layer occupancy is bit-identical.

Notes:
- This is not a removability test.

### `count_layer_invariants.property_count_preserved` [planned]

Description: planned fact for preserving the total count of a property/object set even when members convert within that set.

Static tests:
- Once emitted, use a conversion fixture `A -> B` where both are members of property `Body`.
- Reject when rules create or destroy members outside a balanced conversion.

Runtime/metamorphic tests:
- Count all property members after each replayed turn and assert the total is unchanged.

Notes:
- Not emitted today.

### `count_layer_invariants.layer_changes_within_merge_family` [planned]

Description: planned fact for layers whose occupancy may change only by swapping objects within a proved mergeability family.

Static tests:
- Once emitted, use same-layer conversion between mergeable variants.
- Reject when conversion escapes the merge family.

Runtime/metamorphic tests:
- Replay a known solution and compare projected layer occupancy where merge-family members are treated as equivalent.

Notes:
- This is weaker than `layer.static` but still useful for optimizers.

### `transient_boundary` [current]

Description: object is absent at the end of each normal non-yielding turn after late cleanup.

Static tests:
- Prove early-created then late-cleared.
- Prove late-created then later-late-cleared.
- Prove empty RHS cleanup, e.g. `late [ Mark ] -> []`.
- Do not treat preserving rules as creators, even if they queue `again`.
- Reject objects present in any initial level.
- Reject objects used in winconditions.
- Reject creators with no later cleanup.
- Reject creators after cleanup.
- Reject creators in rigid rules.
- Reject creators in groups tainted by `again`.

Runtime/metamorphic tests:
- Target `atlas shrank`: `Shadowcrate`, `ShadowDoor`, `H_pickup`, `H_grav`, `H_step`, and `ShadowDoorO`.
- Replay a known solution.
- After each full turn plus `again` drain, assert every proved transient object is absent.

Notes:
- Inspect turn-boundary state only; do not require absence mid-turn.

### `transient_boundary.single_turn_only` [current]

Description: fact tag indicating the transient proof applies only to one normal turn boundary, not to arbitrary drained `again` chains.

Static tests:
- Every proved `transient_boundary` fact includes `tags.single_turn_only === true`.
- Rejected transient facts may still include the tag for boundary classification.

Runtime/metamorphic tests:
- Same as `transient_boundary`; no separate runtime harness.

Notes:
- The replay harness should drain `again` before checking absence, but the analyzer proof itself does not simulate future turns.

## Runtime/Metamorphic Harness

### `analyzer_fact_loader` [planned]

Description: helper that compiles source or reads a path, runs `analyzeSource`, and exposes current facts by family/status.

Static tests:
- Fixture path and inline source both load.
- Family filters return only requested families.

Runtime/metamorphic tests:
- Used by all replay checks.

Notes:
- Planned tags should not be treated as missing failures.

### `replay_engine` [planned]

Description: helper that compiles and loads a level, applies inputs, drains `again`, captures/restores state, reads/writes object masks, and compares boundary states.

Static tests:
- None.

Runtime/metamorphic tests:
- Synthetic one-move game verifies replay-to-win.
- Synthetic action-noop game verifies clone/apply-action/drain/compare.
- Synthetic mutation game verifies state comparison fails when expected.

Notes:
- The full-turn boundary is after one input and its `again` drain; perturbations occur before the next input.

### `deterministic_perturbation_seed` [planned]

Description: seeded PRNG input used for reproducible mergeability substitutions.

Static tests:
- Same seed yields same choices.
- Different variant index changes choices.

Runtime/metamorphic tests:
- Failure messages include game, level, fact id, variant index, and seed.

Notes:
- Seed format: `game path + level index + fact id + variant index`.

### `mergeability_perturbation` [planned]

Description: replace objects within a mergeability class at turn boundaries while preserving collision-layer validity.

Static tests:
- Unit-style helper test on a tiny level mask: clears class mask then sets exactly one replacement.
- Skips classes containing proved transient objects.

Runtime/metamorphic tests:
- `limerick` replay with `PlayerBodyH`/`PlayerBodyV` substitutions before every input.

Notes:
- Perturbation happens in live state, not by rewriting source.

### `action_noop_comparison` [planned]

Description: compare boundary state before and after action input for proved `action_noop` games.

Static tests:
- None beyond helper sanity.

Runtime/metamorphic tests:
- Clone state at boundary.
- Apply action to clone.
- Drain `again`.
- Compare board/session state to original boundary snapshot.
- Continue replay from unmodified state.

Notes:
- Injected `noaction` replay is a second check, not the only oracle.

### `count_invariant_check` [planned]

Description: count proved non-transient objects after each replayed turn.

Static tests:
- Count helper on a tiny board mask.

Runtime/metamorphic tests:
- Replay a known solution for a simple game and assert counts remain constant.
- Follow-up solver-visited-state mode can check all generated states for easy levels.

Notes:
- Exclude proved transients.

### `layer_static_check` [planned]

Description: compare exact layer occupancy to the initial layer snapshot after each replayed turn.

Static tests:
- Layer snapshot helper on a tiny board mask.

Runtime/metamorphic tests:
- Replay a known solution for a simple game and assert proved static layers are unchanged.

Notes:
- Does not remove layers.

### `transient_absence_check` [planned]

Description: assert proved transient objects are absent after each full turn plus `again` drain.

Static tests:
- Object absence helper on a tiny board mask.

Runtime/metamorphic tests:
- `atlas shrank` replay checks known transient objects at every boundary.

Notes:
- Transients may exist mid-turn.

## First Curated Runtime Targets

- `limerick`: spawned mergeability for `PlayerBodyH`/`PlayerBodyV`.
- `atlas shrank`: transient boundary for `Shadowcrate`, `ShadowDoor`, `H_pickup`, `H_grav`, `H_step`, and `ShadowDoorO`.
- `sokoban_basic` or `microban`: simple action/count/layer checks.
- synthetic action-noop game: direct clone/apply-action equality.
- synthetic layer-static game: exact layer occupancy check.

If a real-game solution is expensive to find, the test may store a known compact solution string rather than solving every run. The replay harness should still be able to ask the solver for a solution when desired.

## Test Commands

Fast default node test:

```sh
node src/tests/ps_static_analysis_metamorphic_node.js --fast
```

Optional slower corpus target:

```sh
node src/tests/ps_static_analysis_metamorphic_node.js --corpus src/demo src/tests/solver_tests --seed static-analysis
```

The fast test should be suitable for `npm run test:node` once stable. The corpus mode should not be part of the default suite until runtime is predictable.

## Non-Goals

- Do not implement tests in this documentation rewrite.
- Do not prove layer removability in this pass.
- Do not require solvers to rediscover identical solutions.
- Do not fuzz the full corpus in default tests.
- Do not perturb mid-turn or inside rule application.
- Do not let candidate facts drive optimizer behavior.
- Do not treat planned tags as current acceptance criteria.

## Success Criteria

- The spec is organized by output surface: structure, rule tags, group tags, object tags, collision layer tags, game tags, wincondition tags, fact families, and runtime harness.
- Every current emitted tag/fact has a concrete static test description.
- Every runtime-relevant current proved fact has a metamorphic replay test description.
- Planned tags/facts are clearly marked and excluded from first-pass acceptance criteria until emitted.
- Limerick-style spawned mergeable objects are tested by turn-boundary perturbation.
- Atlas-style transients are tested on real gameplay state.
- Layer static is tested as unchanged occupancy, not removability.
