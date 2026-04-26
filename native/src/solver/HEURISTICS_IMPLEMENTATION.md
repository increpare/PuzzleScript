# Wincondition Heuristic Implementation Plan

## Purpose

This document is a checklist for trying, measuring, and keeping or discarding
heuristics for the native solver's weighted A*/greedy search. The solver already
uses winconditions as its heuristic signal. The goal here is to make that signal
more informative without accidentally making it too expensive, too brittle, or
too Sokoban-specific for general PuzzleScript games.

The important design principle is:

> Classify each wincondition by what it asks the level state to prove, then use
> the cheapest heuristic that is valid for that shape.

PuzzleScript winconditions lower to:

```text
NO   A [ON B]
SOME A [ON B]
ALL  A [ON B]
```

In runtime terms this is `WinCondition{quantifier, filter1, filter2, aggr1,
aggr2}`. If the source omits `ON B`, `filter2` is an all-objects mask, so bare
conditions are still evaluated as cell-overlap predicates.

The current implementation lives in two places:

- `native/src/search/search_common.hpp`
  - `winConditionHeuristicScore`
  - `matchingDistanceField`
- `native/src/solver/main.cpp`
  - `heuristicScore`
  - `compactHeuristicScore`
  - `compactMatchingDistanceField`

Any heuristic that changes solver behavior should normally be implemented for
both normal `Session` scoring and compact solver-state scoring, or it should be
explicitly disabled when compact node storage is active.

## Experiment Rules

- [ ] Add every heuristic behind a named option, mode, compile-time flag, or
      clearly isolated strategy branch before making it the default.
- [ ] Keep the existing heuristic available as a baseline.
- [ ] Record for each experiment:
      `strategy`, `heuristic_name`, `astar_weight`, `solved`, `solution_depth`,
      `expanded`, `generated`, `duplicate_states`, `elapsed_ms`,
      `heuristic_ms`, and memory if available.
- [ ] Compare on at least:
      solver smoke tests, Sokoban-like tests, non-Sokoban tests with `NO`,
      tests with `SOME`, tests with properties/aggregates, and tests using
      compact node storage.
- [ ] Treat admissibility claims as suspect unless the game is classified as a
      restricted transport/placement puzzle. PuzzleScript rules can create,
      destroy, transform, teleport, and move multiple objects per input.
- [ ] Prefer "better ordering" over "perfect lower bound" for weighted A* and
      greedy search, but never let an approximate heuristic prune states unless
      the pruning condition is proven safe for the classified game shape.
- [ ] If a heuristic can return "impossible", keep that separate from the score
      until it has tests proving that it does not reject solvable states.

## Shared Support Work

These tasks make the individual heuristics easier to implement and compare.

### H0. Heuristic Selection And Reporting

- [ ] Add a solver option for choosing a heuristic family, for example:
      `--heuristic winconditions|allocation|rule-aware|regions|...`.
- [ ] Add the selected heuristic name to JSON and text result output.
- [ ] Add per-heuristic timing if multiple sub-heuristics can be combined.
- [ ] Add a debug mode that prints the classification of each wincondition for a
      game/level.
- [ ] Ensure portfolio mode reports the actual heuristic used by its weighted
      A* pass.

Implementation notes:

- The current `Result::heuristic` string is set to `"winconditions"`.
- The first low-risk step is to rename the current heuristic to something like
  `"winconditions-distance-v1"` internally while preserving output compatibility
  if needed.

### H1. Wincondition Classification

- [ ] Build a small immutable plan per game/level before search starts.
- [ ] For each `WinCondition`, classify:
      `quantifier`, `has_explicit_on`, `filter1_count`, `filter2_count`,
      `filter1_layers`, `filter2_layers`, `filter1_is_single_layer`,
      `filter2_is_single_layer`, `filter1_is_concrete_single_object`,
      `filter2_is_concrete_single_object`, `filter1_is_aggregate`,
      `filter2_is_aggregate`.
- [ ] Count matching cells in the initial level for each filter.
- [ ] Count matching cells in each current state only when the heuristic needs
      dynamic counts.
- [ ] Mark conditions that are "simple allocation candidates".
- [ ] Mark conditions that are "lazy fallback only".

Suggested classification helpers:

```text
maskObjectIds(mask) -> list<object_id>
maskLayers(mask) -> set<layer_id>
isConcreteSingleObject(mask, aggregate) -> true if !aggregate and one object bit
isSingleLayerMask(mask) -> true if all object bits live on the same collision layer
matchesCell(filter, aggregate, tile) -> existing matchesFilter behavior
countMatchingCells(filter, aggregate, state) -> scan tile cells
```

Important distinction:

- A property mask can still be "simple enough" if all member objects live on the
  same collision layer. In that case two alternatives from the property cannot
  occupy the same cell at once.
- Cross-layer properties and aggregates should usually fall back to lazy
  distance/count scoring. They can represent overlapping facts rather than a
  clean set of interchangeable instances.

### H2. Heuristic Cache Objects

- [ ] Extend `HeuristicScratch` or introduce a per-search `HeuristicContext`.
- [ ] Cache tile lists for static filters when possible.
- [ ] Cache distance fields for static destination masks when possible.
- [ ] Cache obstacle-aware distance fields per static filter and blocker mask
      once obstacle detection exists.
- [ ] Add equivalent compact-state scratch/cache support or a deliberate
      compact fallback path.

Keep these separate:

- Per-game/per-level immutable plan: classification, static mask metadata,
  static target cells.
- Per-state scratch: temporary tile lists, assignment matrices, DP buffers.
- Optional transposition cache: score by compact state key or by relevant object
  positions.

## Baseline Heuristics

### B0. Current Distance/Count Heuristic

Status: implemented.

Current behavior:

- `ALL A ON B`: for every `A` cell not already on `B`, add `10 + distance to
  nearest B`.
- `SOME A ON B`: if any `A` is already on `B`, add `0`; otherwise add the best
  distance from any `A` to nearest `B`.
- `NO A ON B`: optionally add `10` for each offending overlap.
- Optional player-distance term adds a capped distance from player to any
  `filter1`.

Checklist:

- [ ] Name this baseline explicitly in result output.
- [ ] Add tests that lock in the current score for tiny synthetic levels.
- [ ] Verify normal and compact scorers agree on the same states.
- [ ] Keep this as the fallback for all unclassified or risky conditions.

Limitations:

- Nearest-destination distance can reuse the same destination for many objects.
- It does not know which side of `ALL A ON B` is movable.
- It does not distinguish static targets from movable pieces.
- It does not know whether a missing `B` can be ignored.

### B1. Pure Count Deficit

Purpose:

Provide a very cheap fallback when distance is misleading or when objects are
created/destroyed by rules.

Condition handling:

- `ALL A ON B`: count `A` cells that do not also match `B`.
- `SOME A ON B`: score `0` if any satisfying overlap exists, otherwise `1`.
- `NO A ON B`: count satisfying overlaps.
- Bare `NO A`: count `A` cells.
- Bare `SOME A`: score `0` if any `A` exists, otherwise `1`.

Checklist:

- [ ] Implement count-only scoring as a selectable heuristic.
- [ ] Use it as a component that can be combined with stronger heuristics.
- [ ] Confirm it is cheaper than the current distance heuristic.
- [ ] Test it on lifecycle goals such as `NO Coin`, `NO Shadow`, `SOME Won`.

Notes:

- Count deficit is often a good progress signal for monotone games.
- It is weak for reachability and transport games.

## Allocation And Matching Heuristics

### A0. Simple Allocation Candidate Detection

Purpose:

Detect when `ALL A ON B` should be treated as an assignment problem: each
currently present `A` instance must end on a distinct `B` cell.

Candidate rule:

```text
quantifier == ALL
filter1 and filter2 are non-empty
filter1 is concrete single object OR filter1 is a single-layer property
filter2 is concrete single object OR filter2 is a single-layer property
filter1_count <= filter2_count, unless rule analysis says B can be created
```

Checklist:

- [ ] Implement a conservative `isSimpleAllocationCandidate(condition)` helper.
- [ ] Use current-state matching-cell counts, not only initial counts, for games
      where relevant objects may be created/destroyed.
- [ ] Fall back to baseline if either side is an aggregate or cross-layer
      property.
- [ ] Fall back if `filter1_count` or `filter2_count` exceeds a chosen limit.
- [ ] Add debug output explaining why each condition was accepted or rejected.

Same-layer property rationale:

If all alternatives in a property live on the same collision layer, one tile can
match at most one of them. Cell-level assignment is still meaningful. If the
property spans layers, one tile may satisfy several facts at once, so matching
can overstate the amount of work.

### A1. Rectangular Min-Cost Assignment

Purpose:

Replace "sum distance to nearest B" with "minimum total distance assigning every
`A` to a distinct `B` cell".

Condition handling:

```text
ALL A ON B
let sources = cells matching A
let destinations = cells matching B
if sources.empty(): score += 0
if destinations.size() < sources.size(): fallback or impossible-candidate
else score += min assignment cost from each source to a distinct destination
```

Checklist:

- [ ] Implement a tiny assignment solver for `sources <= destinations`.
- [ ] Use bitmask DP for small destination counts.
- [ ] Add a larger fallback: Hungarian algorithm, greedy lower bound, or baseline.
- [ ] Ensure already-satisfied `A ON B` pairs are represented with zero-cost
      assignment to their current tile when that tile matches `B`.
- [ ] Test with `ALL Crate ON Target`, `ALL Target ON Crate`,
      `ALL Player ON Target`, and property destinations.
- [ ] Implement both normal and compact variants.

Bitmask DP sketch:

```text
dp[mask] = best cost after assigning popcount(mask) sources
for each mask:
    source_index = popcount(mask)
    for each destination not in mask:
        next = mask | (1 << destination)
        dp[next] = min(dp[next], dp[mask] + cost[source_index][destination])
answer = min dp[mask] where popcount(mask) == source_count
```

Limits:

- Use DP only while destination count fits in the mask type, for example
  `<= 16` or `<= 20`.
- If `source_count == 1`, do not allocate DP; use the minimum cost directly.
- If `source_count == destination_count`, the answer is the full-mask DP value.

Cost options:

- First implementation: Manhattan or existing two-pass grid distance.
- Better implementation: wall-aware BFS if static blockers are known.
- Sokoban implementation: push-distance or actionability cost.

### A2. Equality Count Strengthening

Purpose:

Exploit the common case where a single wincondition implies exact cover because
the current level has the same number of `A` and `B` cells.

Example:

```text
ALL Target ON Crate
count(Target) == count(Crate)
```

This is equivalent to every target being covered by a crate, even without the
reciprocal `ALL Crate ON Target`.

Checklist:

- [ ] Detect `source_count == destination_count` for simple allocation
      candidates.
- [ ] Mark all destination cells as mandatory.
- [ ] If any mandatory destination has no finite-cost source, report an
      impossible-candidate flag, but do not prune until proven safe.
- [ ] Add optional deadlock checks that are enabled only under equality.
- [ ] Compare score quality against rectangular assignment.

Important:

The core numeric assignment is almost the same as A1. Equality mainly allows
stronger interpretation:

- no destination can be ignored;
- target coverage is meaningful;
- unreachable mandatory targets are suspicious;
- dead-square and tunnel reasoning can be more aggressive.

### A3. Movable-To-Static Direction Selection

Purpose:

Choose the assignment direction by game semantics instead of source syntax.

Examples:

- `ALL Target ON Crate`: targets are usually static, crates move. Match crates
  to targets.
- `ALL Crate ON Target`: same match direction.
- `ALL Player ON Exit`: match player to exit.

Checklist:

- [ ] Add a helper to estimate whether a filter is static or movable.
- [ ] Prefer assigning movable cells to static goal cells.
- [ ] If both sides appear movable, use syntactic `filter1 -> filter2`.
- [ ] If both sides appear static, use baseline count/distance.
- [ ] Test on games that phrase Sokoban goals both ways.

Possible static/movable signals:

- Objects named `target`, `goal`, `exit`, `hole`, `floor` are likely static but
  names alone should be only a weak hint.
- A filter appearing on the RHS of movement/replacement rules may be movable or
  transformable.
- A filter that is never written by rules and is present in level templates is
  likely static.
- A filter sharing the player layer or having explicit movement rules is likely
  movable.

Do not rely on names for correctness. Names can be used as tie-break hints, not
as proof.

### A4. Assignment Cost Caching

Purpose:

Keep allocation scoring cheap enough to evaluate at every expanded state.

Checklist:

- [ ] Reuse tile vectors and cost matrix storage from scratch.
- [ ] Cache static destination tile lists per condition.
- [ ] Cache static destination distance fields where possible.
- [ ] If destinations are dynamic, cache by sorted destination tile list for
      small lists only.
- [ ] Add counters for assignment calls, assignment fallback calls, and average
      assignment size.
- [ ] Compare `heuristic_ms / expanded` before and after assignment.

Recommended first threshold:

```text
if destination_count <= 16:
    bitmask DP
else:
    baseline distance heuristic
```

Then raise/lower based on timing.

## Reachability Heuristics

### R0. Player Reachability To Goal

Purpose:

Handle common goals like `SOME Player ON Exit`, `ALL Player ON Target`, and
single-player/single-goal `ALL` conditions with a pathing score that respects
static blockers.

Checklist:

- [ ] Detect conditions where one side matches the player mask.
- [ ] Compute shortest path from current player cell(s) to destination cells.
- [ ] First version may ignore moving blockers.
- [ ] Later version should treat static collision-layer blockers as walls.
- [ ] Fall back to baseline if there is no player or multiple aggregate players
      make the interpretation unclear.
- [ ] Compare against the current capped player-distance add-on.

Scoring:

- `SOME Player ON B`: shortest player path to any `B`.
- `ALL Player ON B`: if there is one player, same as above; if multiple players,
  assignment from player cells to `B` cells may be useful.
- `ALL B ON Player`: if there is one player and one `B`, same as above; otherwise
  classify as allocation only if simple.

### R1. Region Connectivity

Purpose:

Reject or penalize states where required objects and destinations are separated
by static walls or one-way structure.

Checklist:

- [ ] Build static connected components for each relevant movement layer.
- [ ] For each placement/reach condition, check whether at least one feasible
      source/destination pair shares a component.
- [ ] Under equality exact-cover, flag mandatory destinations with no source in
      their component.
- [ ] Add a soft penalty before adding hard pruning.
- [ ] Test on levels with enclosed targets and enclosed exits.

Notes:

- Region checks are cheap and often catch hopeless assignments.
- For pushable objects, player reachability and push direction matter; simple
  same-component checks are necessary but not sufficient.

## Sokoban-Like Transport Heuristics

These heuristics should be enabled only after classifying the condition as a
transport/placement problem with movable pieces and static goals.

### S0. Wall-Aware Distance Fields

Purpose:

Improve assignment costs by respecting static walls instead of using loose
two-pass Manhattan-like relaxation.

Checklist:

- [ ] Identify static blockers for each moving layer.
- [ ] BFS from each destination or multi-source BFS from all destinations.
- [ ] Use BFS distances in baseline and assignment costs.
- [ ] Fall back if static blockers cannot be identified safely.
- [ ] Compare on Sokoban-like tests and player-exit tests.

Blocker detection:

- A blocker for layer `L` is any object on layer `L` that is static for the
  purpose of the moving object.
- Background/floor layers should not block movement on other layers.
- Dynamic blockers should usually be ignored for admissible lower bounds, or
  treated as soft penalties for non-admissible ordering.

### S1. Push-Actionability Cost

Purpose:

Distance from crate to target is not enough; the player must stand on the
opposite side and push in a legal direction.

Checklist:

- [ ] For each movable source cell and destination, estimate minimum pushes
      ignoring other movable pieces.
- [ ] Add a small player-to-first-push-position term.
- [ ] Penalize source/destination pairs where all push sides are blocked.
- [ ] Cache push-distance tables per level when destination goals are static.
- [ ] Keep this as ordering-only until tested; do not prune on it initially.

Cost idea:

```text
cost(source, goal) =
    minimum pushes from source to goal ignoring other movable pieces
    + min(player distance to a legal first push square, cap)
```

If no push path exists, use a large fallback cost. Only treat it as a hard
deadlock after the abstraction is proven sound for the classified game.

### S2. Dead Squares

Purpose:

Detect cells where a movable object can never contribute to the placement goal.

Checklist:

- [ ] For equality placement goals, mark non-goal corners as dead squares for
      Sokoban-like pushables.
- [ ] Extend to wall corridors where no target lies on the corridor line.
- [ ] Add soft penalty first.
- [ ] Add optional hard prune only for proven Sokoban-like games.
- [ ] Test on simple impossible Sokoban states and on non-Sokoban games to avoid
      false positives.

Safe starting rule:

- If a pushable movable is in a static corner and that cell is not a destination
  cell for its condition, assign a very high heuristic penalty.

### S3. Linear Conflict

Purpose:

Improve matching when two movable pieces block each other along a corridor, row,
or column.

Checklist:

- [ ] Implement only after assignment and wall-aware distances exist.
- [ ] Detect pairs assigned to goals in the same one-tile-wide corridor.
- [ ] Add a fixed penalty for reversed order conflicts.
- [ ] Keep non-pruning and non-admissibility-labeled unless proven.
- [ ] Benchmark separately; this may be too narrow for PuzzleScript.

### S4. Macro-Move Awareness

Purpose:

Collapse or reward forced movement through tunnels and goal rooms.

Checklist:

- [ ] Detect one-wide tunnels in static geometry.
- [ ] Mark forced tunnel pushes as lower-branching successor candidates.
- [ ] Add a heuristic bonus/penalty only if successor generation is unchanged.
- [ ] Consider this a search optimization rather than pure heuristic scoring.

Notes:

- This may belong outside `winConditionHeuristicScore`.
- It can still be selected based on allocation/equality winconditions.

## Lifecycle And Rule-Aware Heuristics

### L0. Extinction Count For `NO`

Purpose:

Make `NO A` and `NO A ON B` goals useful without pretending they are placement
goals.

Checklist:

- [ ] Score bare `NO A` as the count of cells matching `A`.
- [ ] Score `NO A ON B` as the count of offending overlap cells.
- [ ] Use a higher weight for forbidden objects if this improves ordering.
- [ ] Compare against the current fixed `+10` penalty.
- [ ] Test on games whose goal is object removal, for example coin/fruit/shadow
      style conditions.

### L1. Rule Effect Graph

Purpose:

Estimate how objects relevant to winconditions are created, destroyed, or
transformed.

Checklist:

- [ ] Scan compiled/lowered rules for object writes.
- [ ] For each object/filter, record possible effects:
      created, destroyed, transformed-from, transformed-to, moved.
- [ ] Identify rules that can reduce `NO A` counts.
- [ ] Identify rules that can create `SOME A` witness objects.
- [ ] Identify rules that can create/open destination objects, such as
      `ExitOpen`.
- [ ] Expose this as debug output before using it in scoring.

Possible scoring:

- `NO A`: count remaining `A`, plus distance from each `A` to nearest trigger
  pattern that can destroy or transform it.
- `SOME A`: if no `A` exists, estimate distance to a rule trigger that can
  create `A`.
- `ALL Player ON ExitOpen`: if no `ExitOpen` exists, score distance/progress to
  creating `ExitOpen` before player-to-exit distance.

Limitations:

- PuzzleScript rules are pattern-based and can be conditional on surrounding
  cells. A coarse effect graph is an approximation.
- Use for ordering, not pruning.

### L2. Monotonic Progress Detection

Purpose:

Recognize games where a count only moves in the winning direction, making count
heuristics especially reliable.

Checklist:

- [ ] For each object in a wincondition, detect whether rules ever create it.
- [ ] For `NO A`, mark monotone if `A` can be destroyed but not created.
- [ ] For `ALL A ON B`, mark monotone coverage if covered cells cannot become
      uncovered.
- [ ] For `NO Unlit` / painting/filling games, count remaining negative markers.
- [ ] Add a monotone-progress heuristic component and compare on relevant tests.

Do not assume monotonicity from names. Prove it from rule effects where possible.

### L3. Landmark Heuristic

Purpose:

Count required intermediate facts that must become true before the wincondition
can pass.

Checklist:

- [ ] Use rule effect graph to find prerequisite objects for destination objects.
- [ ] Detect locked-exit style chains:
      `Key -> DoorGone -> ExitOpen -> PlayerOnExit`.
- [ ] Score unsatisfied landmarks as small additive penalties.
- [ ] Add debug output listing inferred landmarks.
- [ ] Keep this heuristic optional; false landmarks can mislead badly.

## Relaxed Planning Heuristics

### P0. Delete-Relaxed Object Reachability

Purpose:

Generalize beyond Sokoban by abstractly applying rules while ignoring deletes
and negative interactions.

High-level idea:

```text
possible_objects[cell] starts with current state
repeat:
    apply rule effects that could match possible_objects
    add newly possible objects/cells
until wincondition could pass or fixed point
score = number of relaxed layers/steps needed
```

Checklist:

- [ ] Prototype offline/debug first; do not put in hot path immediately.
- [ ] Limit iterations and object/cell explosion.
- [ ] Start with object existence only, then add cell positions.
- [ ] Score `SOME`, `NO`, and open-exit style goals.
- [ ] Benchmark cost carefully.

Notes:

- Delete relaxation is common in AI planning, but PuzzleScript's spatial rules
  make a precise implementation non-trivial.
- This is likely an ordering heuristic, not an admissible A* lower bound.

### P1. Predicate Pattern Database

Purpose:

Build tiny abstractions over only the predicates used by winconditions.

Examples:

- `ALL Target ON Crate`: abstract target-covered bitset.
- `NO Coin`: abstract coin count or coin bitset.
- `SOME Player ON Exit`: abstract player component and exit component.
- Equality placement: abstract covered-goal subset.

Checklist:

- [ ] Choose one tiny abstraction for one common condition shape.
- [ ] Build reverse BFS or dynamic table per level.
- [ ] Store exact costs in the abstraction only.
- [ ] Use table lookup as a heuristic component.
- [ ] Compare memory/time against assignment.

This is a later-stage experiment. Assignment and rule-aware counts are likely
cheaper first wins.

## Combining Heuristics

### C0. Max Combination

Purpose:

Combine several lower-bound-like components without inflating the score too much.

Checklist:

- [ ] Compute baseline distance.
- [ ] Compute allocation score for eligible conditions.
- [ ] Use `max(baseline, allocation)` for the same condition if both are intended
      as lower bounds.
- [ ] Sum across independent winconditions only as the current heuristic already
      does, and document the risk.
- [ ] Compare against simple sum.

### C1. Weighted Feature Sum

Purpose:

Use multiple non-admissible signals for weighted A*/greedy ordering.

Checklist:

- [ ] Define components:
      count deficit, assignment, player reachability, actionability, rule
      progress, dead-square penalty.
- [ ] Add fixed integer weights.
- [ ] Add heuristic-name output including the weight profile.
- [ ] Tune on a training subset, then validate on a held-out subset.
- [ ] Avoid hard pruning from any component in this mode.

Suggested first profile:

```text
score =
    10 * unsatisfied_count
  +  1 * assignment_or_distance
  +  min(player_distance, 16)
  + dead_square_penalty
```

### C2. Per-Condition Strategy Selection

Purpose:

Use the best available scorer for each wincondition independently.

Selection order:

```text
if condition is simple allocation:
    allocation scorer
else if condition is player reachability:
    player reachability scorer
else if condition is NO/lifecycle:
    lifecycle/count scorer
else:
    baseline distance/count scorer
```

Checklist:

- [ ] Implement selection in a plan object rather than inside the hot scoring
      loop.
- [ ] Add debug output for chosen scorer per condition.
- [ ] Verify fallback behavior is identical to baseline for unclassified games.

## Validation Checklist

Use this section after each implemented heuristic.

- [ ] Normal scorer compiles.
- [ ] Compact scorer compiles or explicitly falls back.
- [ ] Existing solver smoke tests pass.
- [ ] A synthetic test covers the new classification path.
- [ ] A synthetic test covers fallback for aggregate/cross-layer/property cases.
- [ ] Normal and compact heuristic scores agree where both support the heuristic.
- [ ] JSON output includes the heuristic name.
- [ ] Benchmark table recorded before/after.
- [ ] No hard pruning is introduced without a proof and regression tests.

Suggested benchmark groups:

- [ ] Simple reach: `ALL Player ON Target`, `SOME Player ON Exit`.
- [ ] Sokoban-like: `ALL Target ON Crate`, `ALL Crate ON Target`.
- [ ] Equal-count implicit exact cover.
- [ ] More targets than movable pieces.
- [ ] `NO` lifecycle goals.
- [ ] `SOME` witness-token goals.
- [ ] Property winconditions on one collision layer.
- [ ] Cross-layer property/aggregate winconditions.
- [ ] Compact node storage enabled.

## Recommended Implementation Order

- [ ] H0: add heuristic selection/reporting.
- [ ] H1: add wincondition classification and debug output.
- [ ] B0: lock in current baseline tests.
- [ ] B1/L0: add pure count and extinction count components.
- [ ] A0/A1: add simple allocation with bitmask DP and conservative fallback.
- [ ] A2: add equality-count strengthening as metadata and soft checks.
- [ ] A3: add movable-to-static direction selection.
- [ ] A4: cache assignment data and measure cost.
- [ ] R0/R1: add player reachability and region connectivity.
- [ ] S0/S1/S2: try Sokoban-specific wall-aware/actionability/dead-square
      scoring.
- [ ] L1/L2/L3: try rule-aware lifecycle, monotonicity, and landmarks.
- [ ] C2/C1: combine per-condition strategies and tune feature weights.

The likely first useful default is:

```text
per-condition scorer:
    simple ALL placement -> allocation
    player reach/exits -> reachability
    NO/lifecycle -> count
    everything else -> current baseline
```

Keep every more adventurous idea selectable until benchmark data shows it earns
its keep.
