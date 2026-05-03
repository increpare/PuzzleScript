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

## Flow / closure (Approach 2)

Build a conservative object graph from rules using the same primitives as **Approach 2** already discussed: **directed** edges from LHS read objects (via `ruleFlowReads`) toward objects affected on the RHS (via `ruleFlowWrites` / mutating presence), for rules where those summaries apply; take the **transitive closure** from seeds. If an implementation detail requires symmetrizing for a first pass, document it; the analysis remains conservative (may fail to mark something cosmetic, but should not mark core objects cosmetic).

Reuse the same rule IR and `expanded_objects` normalization as mergeability / `rulegroup_flow`.

**Core object** = in the closure of seeds along those edges.

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

## Non-goals (v1)

- Sound complete gameplay reachability (would need simulation).
- Redefining mergeability / `rulegroup_flow` — they stay as today; this is an additional pass.

## References

- `buildWinconditions`, `playerObjectNameSet`, `ruleFlowReads`, `ruleFlowWrites`, `tagInertCollisionLayers` in `src/tests/ps_static_analysis.js`.
