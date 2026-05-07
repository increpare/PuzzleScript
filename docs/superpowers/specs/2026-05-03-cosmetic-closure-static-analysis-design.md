# Cosmetic closure (core object graph) — static analysis design

**Date:** 2026-05-03  
**Status:** Draft for implementation  
**Scope:** `src/tests/ps_static_analysis.js` (+ explorer / tests as needed)

## Goal

Add a **static, conservative “core object” closure** so we can label objects as **cosmetic** (gameplay-irrelevant for this analysis) when they are never pulled in by rules from a small set of **seed** objects. This complements the existing **lexical inert collision layer** tag; it does **not** replace or redefine `layer.tags.inert`.

## Terminology

- **Object** — base entity from the `OBJECTS` section.
- **Property** — legend compound with **`or`** (any member matches).
- **Aggregate** — legend compound with **`and`** (all members together). In engine/docs this is a reserved meaning; this design avoids overloading “aggregate” when talking about **Player**.
- **Player entities** — the set of objects that count as the player for this analysis: same resolution as today’s `playerObjectNameSet` (the `Player` legend entry if it is a property/synonym, else a lone object named `player`). Not “player aggregate” unless the legend actually uses `and`.

## Winconditions vs `win` command

- **Winconditions** (the `WINCONDITIONS` section) are relational phrases in source; the compiler lowers them to masks. In static analysis IR (`buildWinconditions`), each row exposes `subjects` and `targets` arrays — implementation detail of that lowering, **not** “LHS/RHS of the author’s win line” as separate concepts in the manual. For **seeding core**, every object that appears in **either** `subjects` or `targets` for that row is a seed (full win line participation).
- **`win` command** (semantic command on a **rule**): seeds come **only** from that rule’s **LHS reads** (present / absent / movement patterns on the left), not from objects introduced only on the RHS of the same rule. Rationale: the command fires when the LHS pattern holds; RHS material is an effect, not the gating read set for seeding.

## Core seeds (v1)

Union of:

1. **Player entities** (see above).
2. **Winconditions:** all objects listed in `win.subjects` **or** `win.targets` for each compiled wincondition row (after display-name normalization, consistent with the rest of `ps_tagged`).
3. **`win` command on rules:** for each rule whose commands include `win`, every object that appears in **LHS** term expansion (`ruleFlowReads` present/absent/movement object sets — same notion as today’s flow analysis).

## Flow / closure (object graph + layer lifted writes)

We already derive per-rule **object read** and **object write** summaries (`ruleFlowReads`, `ruleFlowWrites` in `ps_static_analysis.js`): LHS-side presence / absence / movement reads, and RHS-side presence / absence / movement effects gated on `object_mutating` / `writes_movement` the same way as today.

Pure **object read → object write** edges on the same rule are **not** enough: a rule can introduce gameplay-relevant objects on a **collision layer** shared with core objects without naming those objects on the LHS (e.g. `[ ] -> [ STONE ]` while **CRATE** is core on the same layer). So closure adds a **layer-lifted write** phase.

### A — Object read → object write (same rule)

For each rule, add a **directed** edge **A → B** whenever object **A** appears in that rule’s **read** set and **B** appears in that rule’s **write** set (object-level masks from the same summaries as above).

### B — Collision-layer write mask (per rule)

From that rule’s **object write set** (objects the rule may place, remove, or replace on the RHS under the same summaries), compute **`layer_write`**: the set of collision **layer ids** that contain at least one of those written objects (`layerForObject` / `collision_layers` in `ps_tagged`).

This is a deliberate projection: we stop tracking *which* cell and only ask *which layers this rule’s writes can touch*.

### C — Layer-mediated core propagation (fixpoint)

Maintain:

- **`core`** — set of objects (starts as all **core seeds**).
- **`core_layers`** — `{ layer(o) | o ∈ core }`.

**Iterate to a fixpoint:** for each rule, if **`layer_write` ∩ `core_layers` ≠ ∅**, then add **every object in that rule’s object write set** into `core` (conservative: the whole write mask, not only objects on the intersecting layers, so cross-layer replacements in one rule do not slip through). Recompute `core_layers` from the enlarged `core` and repeat until no change.

This is intentionally coarse — close in spirit to “which layers can gameplay still touch?” — but it **does not** replace **`layer.tags.inert`** (still lexical). It only prevents marking **STONE** `cosmetic` when **CRATE** is core on the same layer and a rule writes **STONE** onto that layer.

### D — Combine

Initialize **`core`** to all seeds, then **`core_layers`** from **`core`**. Until neither step changes **`core`**: (1) apply step **C** (layer-write intersection pulls in whole write masks); (2) add every object reachable from **`core`** along **directed** edges from **A**; after each change, refresh **`core_layers`**. Interleaving matters: new core objects widen **`core_layers`**, which can activate more rules in **C** on the next round.

**Conservatism:** the analysis may leave some cosmetic objects unmarked (`cosmetic: false` when a human would say cosmetic), but must not mark a gameplay-relevant object as `cosmetic: true`. **B/C** exist specifically because the naive read→write-only graph violated that.

Reuse the same rule IR and `expanded_objects` normalization as mergeability / `rulegroup_flow`.

### Future refinement (non-v1)

**`layer_read`** from LHS object reads (layers referenced on the left) could tighten or loosen coupling; not required for the STONE/CRATE fix above.

## Cosmetic tag

- **`object.tags.cosmetic`** — `true` iff the object is **not** a core object (i.e. outside the closure). Seed objects are **not** cosmetic (`cosmetic === false`).
- Do **not** use `core_reachable` as a tag name in shipped JSON; the negative predicate is named **cosmetic**.

## Inert collision layers

- **`layer.tags.inert`** stays **exactly** as implemented today: lexical check (any object on the layer mentioned in any rule LHS/RHS, wincondition subject or target, or Player entity disqualifies the layer).

No second “semantic inert” on layers unless requested later.

## Explorer / facts (optional follow-ups)

- Surface a **Cosmetic objects** (or “Likely cosmetic”) section: objects with `cosmetic === true`.
- Optional `facts` family `cosmetic_closure` for regression/debug (evidence: seed lists + edge counts) — YAGNI until explorer needs it.

## Tests (fixtures)

1. Wincondition involving `Some Player on Goal` — both Player and Goal are seeds from win row; both non-cosmetic without any rules.
2. Rule with `win` command: only LHS objects seed from that rule for point 3; an object that appears **only** on the RHS of that same rule is not seeded by the command (may still become core via other paths).
3. Purely decorative object: only appears in rules that never connect to seeds via the chosen edge rules → `cosmetic === true`.
4. `layer.tags.inert` unchanged on a fixture where lexical inert still differs from “all objects cosmetic” (if such a case exists).
5. **STONE / CRATE** on one collision layer, win references **CRATE** only, rule `[ ] -> [ STONE ]` (LHS has no object reads, RHS writes **STONE**): **STONE** must **not** be `cosmetic: true` — **layer_write** for that rule intersects the layer of the seeded **CRATE**, so **STONE** enters **core** via layer-mediated propagation (**C**).

## Non-goals (v1)

- Sound complete gameplay reachability (would need simulation).
- Redefining mergeability / `rulegroup_flow` — they stay as today; this is an additional pass.

## References

- `buildWinconditions`, `playerObjectNameSet`, `ruleFlowReads`, `ruleFlowWrites`, `tagInertCollisionLayers` in `src/tests/ps_static_analysis.js`.
