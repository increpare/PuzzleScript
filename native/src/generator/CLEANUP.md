# Generator Cleanup Notes

The generator is still in active development, so avoid broad rewrites unless the
owner explicitly asks for them. These are the current follow-up candidates,
roughly ordered by risk-adjusted value.

1. Replace stringly-typed `SolveResult::status` with a small typed status enum.
2. Remove or fold redundant `CounterSnapshot` fields into the atomic counters.
3. Unify `Strategy` and `SearchMode` once solver/generator strategy ownership is
   settled.
4. Remove the unused `timeoutMs` parameter from generator `runSearch`.
5. Reduce `Node::session` memory cost in the solver loop, likely by storing
   compact state deltas or a reusable scratch session.
6. Replace the current O(n^2) wincondition heuristic helper with precomputed
   matching tile sets or distance fields.
7. Make bounded dedupe eviction deterministic instead of erasing
   `unordered_set::begin()`.
8. Avoid resorting the full top-candidate vector for every solved sample.
9. Cache per-game solver metadata such as `gameUsesRandomness` and
   `buildStateHashProjection` outside the per-sample search.
