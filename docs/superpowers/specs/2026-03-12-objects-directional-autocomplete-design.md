# Objects directional autocomplete (gradual + ordered)

## Summary

In the `OBJECTS` section, directional object variants should be suggested **one object block at a time** (not as multi-block groups). Suggestion ordering should prefer “natural” direction sequences based on the most recent directional declarations **for the same stem**, using abstract directions (up/down/left/right) even if the concrete suffix family is `north/south/west/east`, etc.

## Goals

- Replace “suggest all missing directional variants as one big block” with **single-block suggestions**.
- Preserve the existing “derive sprite by transform” behavior (mirror/rotate) but make the workflow incremental so the user can control which variant to create next.
- Improve ranking by using recent per-stem direction history and matching against a small list of canonical direction sequences.

## Non-goals

- No new UI components or settings UI for this feature.
- No attempt to infer semantics from non-directional object names.
- No requirement to exactly match linguistic order for every concrete suffix family; ordering is based on **abstract directions**.

## Definitions

- **Directional family**: any suffix set in `RuleTransform.DIRECTIONAL_PAIRINGS` (e.g. `up/down/left/right`, `_up/_down/_left/_right`, `north/south/west/east`, etc.).
- **Stem**: the base object name before the directional suffix (e.g. `Crate` in `Crate_up`).
- **Abstract direction**: one of `{up, down, left, right}`.
- **Per-stem direction history**: the last \(N\) directional object declarations that share the same stem (case-insensitive match on stem).

## Behavior

### When suggestions appear

In the `OBJECTS` section, during the “object name” subsection (the same context currently used for the grouped directional suggestion), when:

- the most recently declared object name is a prefix match for the current word being typed, and
- the most recently declared object ends with a known directional suffix (from `RuleTransform.DIRECTIONAL_PAIRINGS`).

### What suggestions are generated

Given the most recently declared directional object `Stem + SuffixBase`:

- Determine its directional family in `RuleTransform.DIRECTIONAL_PAIRINGS`.
- For each “other direction” suffix in that family:
  - If `Stem + SuffixOther` is **not already declared**, create a completion that inserts **exactly one object block**:
    - object name line (`Stem + SuffixOther`)
    - color line (copied from the base object)
    - sprite lines (the base sprite transformed appropriately; if no sprite, omit sprite lines)

Grouped multi-block completions are not generated.

### Sprite transformation mapping

Transformations are defined in terms of the **abstract direction change** from the base suffix to the target suffix:

- `up → down`: mirror vertically
- `left → right`: mirror horizontally
- `up → left`: rotate CCW
- `up → right`: rotate CW
- Other transitions follow the consistent rotation/mirror mapping (i.e. rotate CW/CCW around the four directions; mirror across the appropriate axis).

For two-way directional families (horizontal/vertical), use the existing rotation behavior.

Each suggested variant is derived from the **base object’s current sprite** (the most recently declared object for that stem), enabling nuanced workflows by choosing the order in which variants are created.

## Ordering / ranking (canonical sequences)

### Canonical abstract direction sequences

Maintain a small, ordered list of canonical sequences expressed in abstract directions, e.g.:

- `up → down → left → right`
- `up → right → down → left`
- `up → left → down → right`
- `left → right`
- `right → left`

(Exact list may be adjusted but should include the above.)

### Strict prefix matching

To choose what suggestions appear first:

- Build the per-stem abstract direction history from the last \(N\) (e.g. 3) declarations for that stem.
- A canonical sequence matches if the history is an **exact prefix** of that sequence.
  - Example: history `up, down, left` matches sequence `up, down, left, right` and implies `right` as the preferred next direction.
  - Example: history `up, down, left` does **not** match `up, left, down, right`.

### Suggestion ordering rule

- If one or more sequences match, order suggestions so the **next direction** implied by the best match(es) appears first (highest priority).
- All other missing directional variants for that stem must still appear somewhere in the completion list (lower priority if they’re not the implied “next” direction).
- If no sequence matches, use a deterministic fallback ordering:
  - Prefer mirror counterpart first (e.g. `up → down`, `left → right`), then the two rotations.

### Abstract mapping across families

Even when the concrete suffix family is `north/south/west/east` (or casing/underscore variants), map it to abstract directions `{up,down,left,right}` for history tracking and sequence matching.

## Error handling / safety

- If the suffix family cannot be determined or the transformation mapping fails, avoid throwing; skip the smart suggestion rather than crashing hint generation.

## Test plan

- Declare `Crate_up` (with sprite), then type `cra…` and verify:
  - multiple completions exist (one per missing variant), each inserts a single object block
  - `Crate_down` uses vertical mirror; `Crate_left/right` use rotations
- Create variants in different orders and ensure subsequent suggestions derive from the last-declared object for that stem (so order matters).
- Repeat for underscore forms (`Crate_up`, `Crate_right`, etc.) and for `north/south/west/east` families.
- Verify no crashes in hint generation when objects lack sprites or colors.

