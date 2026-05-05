# JS Solver — next steps

Action plan for tightening up `src/tests/run_solver_tests_js.js` after the
performance/correctness pass logged in `TODO.md`. Items are roughly ordered by
expected payoff-per-effort, but each is independently doable.

The native-side doc `native/src/solver/HEURISTICS_IMPLEMENTATION.md` is for the
C++ solver and shouldn't be treated as a JS to-do list — many of its items are
already covered here. Cross-references in parentheses point to the native doc
where the idea overlaps.

## Status / progress log

- **A3 — done.** Per-condition `staticDeadCellsCache` of `{corner, edge}`
  Uint8Arrays, built once via `inferStaticBlockerMask` + map boundaries.
  `deadPositionPenalty` and `allOnDeadlockHeuristic` rewritten as O(1) lookups.
  Bench on `--solver-heuristic all-on-dead-position` (timeout 250ms, 1341
  levels): **608 → 622 solved (+14), 689 → 675 timeouts (-14), -3s wall.**
  Default `make solver_tests_js` bench unaffected because the default
  heuristic (`'winconditions'` → `allOnClearPathHeuristic`) doesn't call dead
  detection. Switching from dynamic blocker check to static-only is also a
  semantic upgrade: dynamic "blockers" can move away and aren't true deadlocks.

- **B1 (minimal) — exploratory, neutral on aggregate.** New `'auto'` heuristic
  reuses `allOnClearPathHeuristic`'s extras plus a static-corner-deadlock
  penalty (32 per cornered unsatisfied tile, summed across every all-on
  condition rather than just `singleAllOnCondition`). Aggregate solved count is
  within noise of `'winconditions'` (615 vs 618 in one paired run), **but the
  two solve different games**: union is 622, with `winconditions` solving 7
  unique levels and `'auto'` solving 4 unique levels. This is concrete evidence
  that **C1 (multi-heuristic portfolio)** would be a real win — about 4 extra
  solved levels just from running both in parallel.

  Implication: the single-heuristic ranking is a misleading metric. We should
  evaluate heuristics by *how many union solves they enable*, not just their
  own count. Promote C1 above the new-heuristic items in the order below.

- **F1 — partially worked around.** The per-game JSON diff above was done
  manually with `--quiet --json` redirected to disk + a small node script.
  Codifying that flow into `bench_solver.js` (with N-rep variance reporting)
  would make every future experiment cheap.

## Where things stand

- One heuristic at a time. `--solver-heuristic` picks a single function from
  ~33 named heuristics, applied uniformly to every wincondition in a level.
  Default `'winconditions'` aliases to `allOnClearPathHeuristic`, which falls
  back to `winconditionDistanceHeuristic` for non-all-on conditions.
- No per-condition routing, no automatic combination, no rule-effect awareness.
- Portfolio mode varies the *priority formula* (`bfs`, `wa2`, `wa8`, `greedy`),
  not the *heuristic function*.
- Lots of per-state recomputation that could be hoisted into a per-level plan
  (target tile lists, dead-square cache, allowed masks, role analysis).

## A. Per-level plan object

Most heuristics rebuild the same view of the winconditions on every call.
A small immutable plan computed once per `loadLevel` would speed everything up
and let new heuristics share metadata cheaply. (cf. native H1)

- [ ] **A1. Wincondition classifier.** For each wincondition, precompute:
      `quantifier`, `hasExplicitOn`, `filter1Mask`/`filter2Mask`,
      `filter1IsConcrete`, `filter1IsSingleLayer`, `filter1IsAggregate`
      (same for filter2), `staticTargetTiles[]` (when filter2 is unchanging),
      `playerSideFlag` (one of {none, filter1, filter2}). Hang it off the
      specialization closure as `conditionPlan[i]`.
- [ ] **A2. Static target tile cache.** `collectMatchingTiles(condition[2], ...)`
      is called from many heuristics. Memoise by condition index when
      `filter2Mask` only matches non-mutating cells (most all-on goals against a
      static target object).
- [ ] **A3. Dead-square cache.** `deadPositionPenalty` walks corners every call.
      Build a `Uint8Array(level.n_tiles)` `isDeadCorner[i]` once per level using
      static blockers + plan target tiles, then the heuristic becomes a counting
      pass. Likely the single biggest win in this section.
- [ ] **A4. Player-side detection.** Once per level, mark whether either filter
      of an all-on/some-on condition matches the player object. Lets the
      router (B) jump straight to a reachability scorer instead of running
      `winconditionDistanceHeuristic`.

## B. Per-condition strategy routing

Today every wincondition gets the same heuristic. Most levels mix shapes, so a
bad pairing (`SOME Player ON Exit` evaluated by an all-on-clear-path scorer)
silently degrades search. (cf. native C2)

- [ ] **B1. Router heuristic.** Add `'auto'` heuristic: for each wincondition,
      pick a sub-scorer based on the plan classification:
      ```
      if quantifier == ALL && plan.playerSide:        playerReachOrAssignment
      else if quantifier == ALL && simpleAllocation:  matching/clear-path family
      else if quantifier == SOME && plan.playerSide:  someOnPlayer (BFS variant)
      else if quantifier == SOME:                     someOnClearPath
      else if quantifier == NO:                       noOnEscape
      else:                                           winconditionDistance
      ```
      Score is the sum across conditions, same as the existing baseline.
- [ ] **B2. Make `'auto'` the new default.** Keep `'winconditions'` as the
      legacy alias. JSON output should record the per-condition pick (one short
      string array under `result.heuristic_breakdown`).
- [ ] **B3. Combinator syntax.** Allow `--solver-heuristic
      all-on-clear-path+no-on-escape` to sum two named heuristics — useful for
      testing pairs without writing a new function.

## C. Heuristic-varying portfolio

The portfolio currently runs `bfs`/`wa2`/`wa8`/`greedy` on the *same*
heuristic. Different heuristics rank states very differently; the cheap
diversification is to also run several heuristics in parallel slices.

- [ ] **C1. Multi-heuristic portfolio.** Add `--portfolio-heuristics
      auto,all-on-matching,no-on-escape` (comma list). Portfolio rotates over
      `(heuristic × priorityMode)` pairs, sharing the visited bucket. Probably
      cap at 4 pairs to keep frontier overhead bounded.
- [ ] **C2. Auto-lock on best phase.** Today the portfolio locks to wa2 when
      `step_ms/generated > 0.05`. Extend the same idea to lock to whichever
      `(heuristic, mode)` pair was last to expand a depth-improving node.

## D. New heuristic ideas

Concrete heuristics worth prototyping. Each should be added as a named entry in
`HEURISTIC_FUNCTIONS` so it's selectable for benchmarking before becoming part
of `'auto'`.

- [ ] **D1. Reachability-aware `someOnPlayerBFS`.** Existing
      `someOnPlayerHeuristic` uses raw Manhattan. Reuse the
      `obstacleDistanceField` infrastructure (BFS from player, blocked by
      static blockers) and pick the min distance to a target tile. Should be a
      strict improvement on player-reach goals.
- [ ] **D2. `allOnRegionIsolation`.** Cheap connected-component check: if any
      unsatisfied source tile sits in a static-blocker component with no target
      tile, add a large soft penalty per such tile. Catches "boxed in" states
      that current heuristics rate low. Component table is per-level. (cf. R1)
- [ ] **D3. Equality cover (`allOnEqualityCover`).** When
      `count(filter1Mask matches) == count(filter2Mask matches)` and both sides
      are concrete/single-layer, score on *target* coverage instead of *source*
      satisfaction: count target tiles not currently covered. For symmetric
      Sokoban-style placement this is often more discriminating than the
      source-side count. (cf. native A2)
- [ ] **D4. `noOnLifecycle`.** If rules can never *create* an offending object
      (one-time scan of right-hand sides for the relevant mask), use the offender
      count directly with a high weight — it becomes a near-admissible monotone
      score. Otherwise fall back to `noOnEscape`. (cf. L0/L2)
- [ ] **D5. 2×2 packing deadlock.** For all-on placement against a static target
      mask, detect 2×2 blocks of unsatisfied movables in non-target cells with
      no rule that can split them. Mostly a Sokoban move; gate on the same
      "simple allocation candidate" classifier as D3.
- [ ] **D6. One-step lookahead refinement.** For frontier entries within ε of
      the best priority, expand once and re-score with the *minimum* child
      heuristic. Cheap with the existing solver ops, often defeats heuristic
      plateaus. Worth measuring `expanded` reduction vs added time.
- [ ] **D7. Push-direction feasibility on `someOnClearPath`.** `pushAccessPenalty`
      is all-on only. A `someOn` variant that picks the cheapest-pushable
      target (rather than the closest) helps on non-ALL Sokoban-ish goals.

## E. Search-side wins (not heuristics)

- [ ] **E1. No-op action skipping.** Before calling `stepSolverAction`, check
      whether any rule could possibly fire from the current cell occupancy +
      action token. If not, treat the result as identical-to-parent and skip
      the snapshot/hash work entirely. Needs the rule-effect scan from D4.
- [ ] **E2. Equivalent-action collapse.** Many puzzles have free moves that
      yield the same state as not moving (player against a wall). Detect by
      hashing only the post-step `objects` array against the parent and
      dropping the child if equal. Cheaper than full `capture` + visited
      lookup because the comparison can short-circuit on the first differing
      word.
- [ ] **E3. IDA\*** mode. For low-branching levels with deep solutions
      (lots of timeouts in current corpus are these), iterative deepening A\*
      cuts memory dramatically. Worth a `--strategy ida-star` experiment to
      see if the timeout games are fundamentally heuristic-bound or
      memory-bound.
- [ ] **E4. Adaptive `astarWeight` schedule.** Run wa2 for the first
      `T` ms; if no solution, double the weight and continue from the same
      frontier. (Re-prioritising is just a heap rebuild.) Lets one run cover
      what currently requires a portfolio.

## F. Benchmarking & introspection

We're flying half-blind: it's hard to tell which heuristic wins on which game
without a manual sweep.

- [ ] **F1. `bench_solver.js` driver.** Wrap `run_solver_tests_js.js` to run a
      list of `(strategy, heuristic, astarWeight)` triples across the corpus
      and emit CSV (`game, level, config, status, depth, expanded, elapsed_ms,
      heuristic_ms`). Single source of truth for "did this change help?".
- [ ] **F2. Per-condition heuristic breakdown in JSON.** When `'auto'` is
      active, include the per-condition picks in `result.heuristic_breakdown`
      (string array) so the bench can correlate failures with classifier picks.
- [ ] **F3. Compute-once heuristic timing.** Currently `heuristic_ms` lumps
      classifier + scorer + scratch work. Split into `heuristic_classify_ms`,
      `heuristic_score_ms` (cheap with the existing `timeBlock` helper). Helps
      decide whether A1/A2 are worth their complexity.
- [ ] **F4. "Expanded per solved" leaderboard.** In summary-only output, rank
      heuristics by `expanded / (solved + 1)` across the corpus. Cheap signal
      for "is this heuristic actually steering search or just along for the
      ride?".

## G. Suggested order of attack

Updated based on the A3/`auto` measurements above.

1. ~~**A3**~~ — done.
2. **C1** (multi-heuristic portfolio) — promoted above F1 because we already
   have evidence (`winconditions` ⊕ `auto` = 622 vs 618/615 singletons) that
   it's worth ~4 levels with zero new heuristic work. Just plumbing.
3. **F1** (bench driver) — once C1 lands, we'll be choosing *combinations* of
   heuristics, and the single-config `make solver_tests_js` becomes even less
   sufficient.
4. **A1** (plan object) — needed before B1 can do real per-condition routing;
   the current `'auto'` is a stopgap that just shares extras across conditions.
5. **B1 proper** (full per-condition router using A1 metadata).
6. **D1, D2, D4** (reach-aware some, region isolation, lifecycle no-on) —
   three different shapes of new signal, each cheap to try with F1 + C1 to
   validate.
7. **E1 + E2** (no-op / equivalent action skipping) — orthogonal to
   heuristics, measurable on `expanded`/`generated` ratios.
8. **D6 + E4** (lookahead, adaptive weight) — refinements once the foundation
   is in place.
9. **E3** (IDA\*) — only if F1 shows we have a memory ceiling, otherwise skip.

D5, D7, F3, F4 are nice-to-have polish — pull them in when their parent area
gets touched.

### C1 implementation sketch

The existing portfolio mode varies *priority formula* (`bfs`/`wa2`/`wa8`/
`greedy`) on a single `solverOps`. To vary heuristics, the cleanest path is:

- Construct one `solverOps` per heuristic in the portfolio list (each builds
  its own caches; the Zobrist hash is heuristic-independent so the visited
  bucket can be shared).
- Adapt `runAdaptivePortfolio` so each `portfolioMode` entry carries its own
  `solverOps` reference, and the per-mode heap stores priorities computed
  against that mode's heuristic.
- Existing auto-lock-to-wa2 logic still applies; gate it on
  `(solverOps, mode)` rather than just `mode`.
- Add `--portfolio-heuristics auto,winconditions` (comma list) with sensible
  default (`auto,winconditions` once both are in tree). Cap at ~4 entries to
  bound frontier overhead.
