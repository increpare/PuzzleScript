# JS Solver — next steps

Action plan for tightening up `src/tests/run_solver_tests_js.js`. Items are
roughly ordered by expected payoff-per-effort, but each is independently doable.

The native-side doc `native/src/solver/HEURISTICS_IMPLEMENTATION.md` is for the
C++ solver and shouldn't be treated as a JS to-do list — many of its items are
already covered here. Cross-references in parentheses point to the native doc
where the idea overlaps.

## TL;DR — status: paused

Heuristic pass concluded. Single-heuristic gain over baseline at 250ms timeout
landed at **+14 to +17 solves** (608 → 622-625 depending on heuristic). At
the 5s default timeout the spread compresses to ~5 solves (897 → ~902).

What landed:
- **A3** — per-condition static-dead-cell cache (corner/edge masks). +14 solves.
- **D2** — region-isolation penalty on top of A3. +1 to +4 more.
- **`auto`** — per-condition router; current `DEFAULT_SOLVER_HEURISTIC`.
- Opt-in plumbing for `--portfolio-heuristics` and `--strategy phase-split`
  (kept available; both showed neutral-to-regression at fixed budgets).

What was tried and rejected (see status log for benches):
C1, C1b, D1, D4, D7. Three consecutive negative D-results plus failed
multi-heuristic combinations indicate the heuristic-shape space at the 5s
budget is largely exhausted.

Profiling outcome (this pass): `step_ms` is **86.2%** of search time and
**38.3%** of step calls are no-ops — recoverable in principle but not safely
attackable from the heuristic layer (any static "drop this action" check
risks search corruption per engine semantics). Captured as **E1** in the
backlog; needs deeper engine-internals work (in `src/js/engine.js`), not
solver-script tweaking.

Reasonable next moves only with fresh evidence:
- D3 / D5 if a Sokoban-shape detector lands (small expected wins, 250ms-only).
- E1 reframed as engine-side `applyRules` early-out (requires care + tests).
- IDA\* (E3) only if measurement shows a memory ceiling.

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

- **D2 + B1 (revised default) — landed.** Region-isolation penalty
  (`regionIsolationPenalty`) added: for each unsatisfied tile in an all-on
  condition, +256 if its static-blocker-bounded connected component contains
  no current target tile. Component IDs are folded into the existing per-
  condition `staticDeadCellsCache`, so the per-call cost is one pass over
  current targets + one over unsatisfied tiles.

  New heuristic `'all-on-dead-isolated'` = dead-position + isolation. Bench
  (paired runs vs `'all-on-dead-position'`): **+1 to +4 solves** consistently
  (~622-625 vs ~620-624). Honest finish: real but small.

  Then on the (correct) critique that an all-on-named heuristic is the wrong
  shape for a general default, `'auto'` was rewritten as a per-condition
  router:
  - Iterates every wincondition; for every all-on condition it adds
    `deadPositionPenalty + regionIsolationPenalty`.
  - Base is `allOnClearPathHeuristic` (which falls through to
    `winconditionDistanceHeuristic` when there's no single all-on condition).
  - SOME and NO conditions contribute via base only — measured the existing
    specialized scorers (`some-on-static-blockers` 611, `no-on-escape` 610)
    and they're all *worse* than the base distance heuristic (~617). New
    SOME/NO specialized adders are still wanted (D4, D7) but plugging in the
    existing ones regresses.

  Switched `DEFAULT_SOLVER_HEURISTIC` to `'auto'`. Bench: ~615-621 solved.
  This is ~3-6 below `'all-on-dead-isolated'` because that heuristic only
  fires for *single* all-on conditions and skips per-condition iteration on
  multi-condition levels (cheaper per call). Net trade: small aggregate
  for per-shape correctness — the corpus is heavily Sokoban-skewed and
  rewarded the all-on-only specialist.

- **C1/C1b — both implemented, both **don't pay off***** at fixed wall-clock
  budgets. Decisive evidence below.

  Re-grounded baselines first (default `make solver_tests_js` strategy is
  `weighted-astar`, **not** `portfolio` — earlier C1 numbers in this log were
  vs the wrong baseline):

  | config (250ms timeout) | solved |
  |---|---|
  | `--solver-heuristic winconditions` (default) | 618 |
  | `--solver-heuristic all-on-dead-position` (post-A3) | 622 ×2 |
  | `--solver-heuristic auto` | 613 |
  | `--strategy portfolio --portfolio-heuristics all-on-dead-position` | 618 |
  | `--strategy portfolio --portfolio-heuristics all-on-dead-position,winconditions` | 613 (−5) |
  | `--strategy portfolio --portfolio-heuristics all-on-dead-position,auto` | 616 (−2) |

  At 500ms timeout the gap is even starker:

  | config (500ms timeout) | solved |
  |---|---|
  | `--solver-heuristic all-on-dead-position` | **700** |
  | `--strategy phase-split --portfolio-heuristics all-on-dead-position,winconditions` (250+250) | 631 |

  **+69 solves** for a single best heuristic with full budget over a
  phase-split that gives each heuristic the *same* 250ms budget the single
  heuristic uses at the lower timeout. Even when the orthogonality story
  (`union 622`) was technically true, putting both heuristics into one
  budgeted run sacrifices way more search than diversification recovers.

  **Actionable conclusions:**
  - **Done (and superseded):** `DEFAULT_SOLVER_HEURISTIC` was switched from
    `'winconditions'` to `'all-on-dead-position'` (623/1341), then to
    `'all-on-dead-isolated'` (~622-625) after D2 landed, then to `'auto'`
    (~615-621) on win-condition-shape grounds (see B-section below).
  - Heuristic combination work (B1's `auto`, multi-heuristic portfolios) is
    chasing a phantom. A *better single* heuristic beats any combination of
    weaker ones because per-level budget is the binding constraint, not
    diversity. Future heuristic work should target unconditional improvement
    of the leading scorer (D1, D2, D4 ideas — reach-aware some, region
    isolation, lifecycle no-on), not blending.

  Plumbing kept as opt-in: `--portfolio-heuristics A,B,…` (interleaved) and
  `--strategy phase-split --portfolio-heuristics A,B,…` (sequential) both
  remain, in case scheduling tricks (uneven budgets, locked phases) make
  them viable later.

- **(historical) Earlier C1 reading vs `--strategy portfolio` baseline.**
  Added `--portfolio-heuristics NAME[,NAME…]` to `parseArgs`; when omitted
  falls back to single-heuristic behaviour (`solverHeuristic`).
  `runAdaptivePortfolio` builds one `solverOps` spec per heuristic, primary
  spec drives state ops, secondary specs add `wa2` entries. Each child node
  evaluates every active heuristic.
  Bench (250ms timeout, 1341 levels) — variance turned out to be wider than
  the earlier 3-run sample suggested:
  - Baseline (5 runs): **613 ± 7 solved** (619, 617, 616, 606, 611)
  - Single-heuristic via portfolio plumbing (`--portfolio-heuristics
    winconditions`): 613 — refactor itself ≈ neutral
  - Multi `winconditions,auto`: 611 — within noise
  - Multi `auto,winconditions`: 610 — within noise
  - Earlier 4-mode-replaced design (`wa2:auto,wa2:winconditions,bfs`): 605 —
    likely a real regression because it dropped wa8/greedy

  Verdict: the plumbing-conserving design (primary keeps full quartet,
  secondaries contribute wa2 only) is **noise-neutral** vs baseline at 250ms
  timeouts. The expected union-of-622 win **doesn't appear** because each
  heuristic gets less effective budget per node and the extra heuristic
  computation eats into search depth. We get neither the gain nor a clear
  loss.

  Implication for the next iteration: **interleaving doesn't pay; sequencing
  might.** A phase-split portfolio (run heuristic A for `T1` ms, then on
  timeout restart with heuristic B for the remaining budget) preserves
  per-heuristic search density and is the obvious next experiment. C1 plumbing
  stays in place behind the opt-in flag; default behaviour unchanged.

- **D4 (`noOnLifecycle`) — tested and rejected.** Implemented a static rule scan
  that classifies every NO X ON Y condition as "no rule can ever introduce a
  new offender" vs "some rule might". For the no-create case, the offender
  count alone is monotone-decreasing along any solution path, so the standalone
  `'no-on-lifecycle'` heuristic returns `count * 32`; otherwise it falls back
  to `noOnEscape`'s distance-to-escape score.

  Bench (250ms, 1341 levels):
  - `--solver-heuristic no-on-escape` (control): 612
  - `--solver-heuristic no-on-lifecycle`: **615 (+3)**

  Bench (5000ms default):
  - `--solver-heuristic no-on-escape`: 897
  - `--solver-heuristic no-on-lifecycle`: 897 (no change)

  The +3 at tight timeouts disappears at the standard timeout — the levels
  that lifecycle solves faster also fit inside `noOnEscape`'s 5s budget. Wiring
  D4 into `auto` (per-condition, additive on top of base) **regressed** by ~3
  solves at 250ms because adding `+32` per offender on top of base's `+10` per
  offender over-prioritises NO destruction relative to all-on alignment in
  mixed-condition levels.

  Verdict: the win is too small at any realistic timeout to justify the new
  ~50 lines (cache + scan + heuristic + dead code path in `auto`). Code
  reverted; this log entry preserves the negative result.

- **D1 (`someOnPlayerBfs`) — tested and rejected.** Replaced raw player→source
  Manhattan with `obstacleDistanceField`-based BFS through the level
  (blocked by `cellHasBlockingObject`, which treats walls + decorations as
  blocking but lets the BFS pass through sources/targets/player).

  Bench (250ms): some-on-player (control) 612 → some-on-player-bfs **606
  (-6)**. Bench (5000ms): 898 → 896 (-2). Both worse.

  Why it doesn't help: the corpus is dominated by small-to-medium Sokoban-style
  levels where Manhattan is essentially equal to BFS-through-walls. Adding the
  per-call O(n_tiles) BFS work doesn't pay back in signal — and where BFS
  diverges from Manhattan it inflates `h` slightly, biasing weighted A*'s
  `f = depth + 2h` ordering toward shallower states in a way that misled
  search on a handful of levels (the -6 at 250ms). Reverted.

- **D7 (`someOnPushAccess`) — tested and rejected.** Implemented a some-on
  variant of `pushAccessPenalty` that returns the cheapest-pushable source's
  manhattan distance + a small surcharge for unpushable sources (12 for
  cornered, 4 for "can push but not toward closest target").

  Bench (250ms):
  - `--solver-heuristic some-on-clear-path` (control): 616
  - `--solver-heuristic some-on-push-access`: **609 (-7)**

  Bench (5000ms): both 897.

  The cheapest-pushable signal at 250ms misleads search vs the existing
  aligned-row/col bonus from `someOnClearPath` — the surcharges over-penalise
  source tiles that are temporarily blocked but where intermediate moves can
  free them. Reverted.

- **Profiling pass (250ms portfolio+auto, full corpus).** Added a
  short-lived `no_ops` counter on the `!stepResult.changed` branch and
  re-ran the full `--json` bench to see where time actually goes. Results
  reverted; raw findings:

  | bucket | ms | % accounted |
  |---|---|---|
  | `step_ms` | 173,724 | **86.2%** |
  | `heuristic_ms` | 19,526 | 9.7% |
  | `clone_ms` | 2,481 | 1.2% |
  | `hash_ms` | 2,427 | 1.2% |
  | `queue_ms` | 2,032 | 1.0% |
  | `snapshot_ms` | 1,318 | 0.7% |

  Of the 8.10M `stepSolverAction` calls:
  - **38.3% are no-ops** (`changed=false` — engine ran but didn't move state)
  - 32.3% are duplicates (changed but post-step hash already visited)
  - 29.4% are useful (pushed to frontier)

  At 0.0214 ms/step, the no-op slice alone is ≈66 seconds of search time
  per corpus run — **~33% of total solver wall-time** spendable on a
  cheaper-than-`processInput` no-op predictor. This dwarfs any further
  heuristic-shape work: D1/D4/D7 each shaved <2% even when they helped.

  Per-game distribution: no-ops are 30-75% of step calls in roughly half
  the corpus (top offenders: `The Far Away Danish Pastry…` 75.5%,
  `limerick.txt` 72.0%, `gapfiller.txt` 71.0%, `make way.txt` 65.9%).
  Not concentrated in a few games — a structural property of pressing
  movement keys against walls.

  **Conclusion: E1 (no-op skipping) is the highest-EV remaining work**,
  ahead of all D-section heuristics and ahead of E2 (post-step
  `objects`-array compare, which only saves the snapshot+hash slice =
  1.9% of time even if 100% effective). Recommend prioritising E1 with
  a cheap-and-conservative predicate (better to false-negative than
  false-positive — false-positives skip real children and would corrupt
  the search).

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

- [x] **C1. Multi-heuristic portfolio.** Plumbing landed via
      `--portfolio-heuristics NAME[,NAME…]`; primary spec drives state ops,
      secondary specs add `wa2` entries. **Tested and rejected** — see status
      log. Kept as opt-in flag, not used in defaults.
- [x] **C1b. Phase-split portfolio.** Implemented as `--strategy phase-split`
      with `--portfolio-heuristics A,B,…` slicing the budget evenly. **Tested
      and rejected** — at fixed wall-clock budget, each phase loses more
      search depth than diversification recovers. Even at 2× budget the
      single best heuristic wins by tens of solves. Kept as opt-in.
- [ ] **C2. Auto-lock on best phase.** Today the portfolio locks to wa2 when
      `step_ms/generated > 0.05`. Extend the same idea to lock to whichever
      `(heuristic, mode)` pair was last to expand a depth-improving node.

## D. New heuristic ideas

Concrete heuristics worth prototyping. Each should be added as a named entry in
`HEURISTIC_FUNCTIONS` so it's selectable for benchmarking before becoming part
of `'auto'`.

- [x] ~~**D1. Reachability-aware `someOnPlayerBFS`.**~~ Tested and rejected
      (see status log). −6 at 250ms, −2 at 5s. BFS-through-walls degenerates
      to Manhattan on the corpus's small/open levels; the per-call BFS cost
      and `h` inflation outweigh any reachability signal.
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
- [x] ~~**D4. `noOnLifecycle`.**~~ Tested and rejected (see status log). +3
      standalone at 250ms, neutral at 5s. Wiring into `auto` regressed by ~3.
- [ ] **D5. 2×2 packing deadlock.** For all-on placement against a static target
      mask, detect 2×2 blocks of unsatisfied movables in non-target cells with
      no rule that can split them. Mostly a Sokoban move; gate on the same
      "simple allocation candidate" classifier as D3.
- [ ] **D6. One-step lookahead refinement.** For frontier entries within ε of
      the best priority, expand once and re-score with the *minimum* child
      heuristic. Cheap with the existing solver ops, often defeats heuristic
      plateaus. Worth measuring `expanded` reduction vs added time.
- [x] ~~**D7. Push-direction feasibility on `someOnClearPath`.**~~ Tested and
      rejected (see status log). −7 standalone at 250ms vs `some-on-clear-path`,
      neutral at 5s. Cheapest-pushable surcharges over-penalise temporarily
      blocked sources.

## E. Search-side wins (not heuristics)

- [ ] **E1. No-op action skipping.** Before calling `stepSolverAction`, check
      whether any rule could possibly fire from the current cell occupancy +
      action token. If not, treat the result as identical-to-parent and skip
      the snapshot/hash work entirely. (D4's rule-effect scan was implemented
      and reverted; if E1 needs it, it'll need to be re-derived.)
      **Profiled (see status log): 38.3% of step calls are no-ops, accounting
      for ~33% of total search wall-time.** Highest-EV remaining work item,
      but **deferred** — no safe attack found from the solver-script layer.
      Implementation must be conservative: false-positives (flagging a real
      child as a no-op) corrupt the search; false-negatives just leave
      performance on the table. Candidate predicates considered:
      1. ~~**Drop `action` from the action list when no rule's LHS references
         the action button.**~~ Considered and rejected: not safe. Engine
         runs all rules every turn regardless of input, and direction
         presses set the player's `moving` bit (01111) which `[moving player]`
         / `[> player]` rules consume; ACTION sets the action bit (10000)
         which they don't. So even with no rule referencing `action`,
         pressing ACTION can produce a state distinct from any direction
         press purely via engine semantics (`resolveMovements`).
      2. **For movement actions: target cell contains only static-blocker
         objects AND no rule has a `LATE` or input-direction-agnostic LHS
         that could fire from the unchanged board.** Trickier; needs care
         around `[stationary]`, `[moving]`, and global-pattern rules.
         Likely needs full rule-dependency-graph analysis; out of scope for
         a heuristic pass.
      3. **Per-state per-action no-op cache** — won't help because each
         pre-step state is unique (post-step dedup ensures it).
      4. **Engine-side `applyRules` early-out.** Move the question into
         `src/js/engine.js`: short-circuit the rule loop if a cheap
         pre-pass shows no rule's LHS could match anywhere. Helps every
         action, not just `action`. Risks correctness regressions in the
         player; needs the full simulation test suite to gate it.

      Conclusion: this opportunity exists but the right home is engine
      internals, not solver heuristics.
- [ ] **E2. Equivalent-action collapse.** Many puzzles have free moves that
      yield the same state as not moving (player against a wall). Detect by
      hashing only the post-step `objects` array against the parent and
      dropping the child if equal. Cheaper than full `capture` + visited
      lookup because the comparison can short-circuit on the first differing
      word. **Already partially in place**: line ~2631 `if (!stepResult.changed) continue;`
      drops engine-level no-ops without snapshot/hash. The remaining
      headroom (only saves snapshot+hash slice = ~1.9% of time even if
      100% of changed-but-equivalent states were caught) is small. Skip
      until E1 is exhausted.
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

## G. Future work — not for this pass

This pass is concluded (see TL;DR at the top). The items below are kept as a
backlog. None of them should be attempted without (a) fresh measurement
evidence motivating them, and (b) the F1 bench driver to make experiments
cheap — running each candidate as a one-off is what produced the D1/D4/D7
false-starts.

Done in this pass:
- ~~**A3**~~ — per-condition static dead-cell cache (+14 solves).
- ~~**D2**~~ — region isolation penalty on top of A3 (+1 to +4).
- ~~**B1 (`auto`)**~~ — per-condition router, current default.
- ~~**C1 / C1b**~~ — both rejected; single best heuristic wins at fixed budget.

Tried and rejected with negative benches: **D1**, **D4**, **D7** (see status log).

Backlog (only with fresh evidence):
- **F1** — `bench_solver.js` multi-config driver. Prerequisite for any further
  experimentation; would have caught D1/D4/D7 as neutral-at-5s in one pass.
- **A1** — wincondition classifier as a real per-level plan object; needed
  before any B-section work can do proper per-condition routing.
- **D3 / D5** — equality cover, 2x2 packing deadlock. Sokoban-shape signals
  with likely small/250ms-only wins, similar to the rejected D-items.
- **E1** — no-op action skipping. 33%-of-wall-time potential, but the right
  home is engine internals (`src/js/engine.js`), not solver heuristics.
  See E1 entry for why each candidate predicate doesn't work at the solver
  layer.
- **D6 / E4** — lookahead heuristic, adaptive `astarWeight` schedule.
- **E2** — equivalent-action collapse. Already partially in place via the
  `changed=false` early-out at line ~2631 of `run_solver_tests_js.js`;
  remaining headroom is the snapshot+hash slice (~1.9% of time).
- **E3 (IDA\*)** — only if measurement shows a memory ceiling.
- **F2 / F3 / F4** — finer-grained instrumentation; pull in as needed.

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
