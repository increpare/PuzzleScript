# Static Analysis Rule Tag Testdata Design

## Purpose

Add the second small static-analysis testdata slice: individual rule tags.

This slice sits between object tags and later rule-flow/influence analysis. It exposes conservative, object-level facts about each rule: what object presence or absence can make the rule match, and what objects the rule writes or erases. It must stay simple enough to inspect by hand.

This slice does not add rule-group flow, cross-rule influence, movement flow, static object tags, or cosmetic object tags.

## File Layout

```text
src/tests/static_analysis_claim_descriptions.json
src/tests/static_analysis_testdata/
  rule_tags/
    *.txt
    *.json
src/tests/static_analysis_testdata_runner.js
```

`rule_tags/` follows the same pairing policy as `object_tags/`: every committed `.json` has a matching whole-source `.txt`, orphan `.txt` files may generate `.json`, and existing `.json` files are never overwritten automatically.

## Claim Descriptions

`src/tests/static_analysis_claim_descriptions.json` should grow a `ruleTags` array alongside `objectTags`.

The first rule-tag vocabulary is:

```json
{
  "ruleTags": [
    {
      "name": "objects_required",
      "description": "Concrete objects definitely required present by the rule LHS.",
      "specification": "A rule has objects_required entries for concrete objects whose presence is definitely required by positive LHS terms. Single-object terms are required. Objects of required aggregates are all tagged as required. Members of an OR property are not individually required because any one member may satisfy the term."
    },
    {
      "name": "objects_matched",
      "description": "Concrete objects whose presence may satisfy a positive LHS term.",
      "specification": "A rule has objects_matched entries for every concrete object that may satisfy a positive LHS term after resolving synonyms, properties, and aggregates to object names. This includes concrete required objects, OR-property alternatives, and objects in aggregates."
    },
    {
      "name": "object_absences_matched",
      "description": "Concrete objects whose absence may satisfy a negative LHS term.",
      "specification": "A rule has object_absences_matched entries for every concrete object that may satisfy a negative LHS term after resolving synonyms, properties, and aggregates to object names. For example, no Obstacle records every concrete object in Obstacle."
    },
    {
      "name": "objects_written",
      "description": "Concrete objects written present by the rule RHS.",
      "specification": "A rule has objects_written entries for concrete objects the RHS may write present in a cell where the LHS does not already require that object present in the same cell. This is cell-local: relocation writes the object at the destination even if total object count is preserved."
    },
    {
      "name": "objects_erased",
      "description": "Concrete objects written absent by the rule RHS.",
      "specification": "A rule has objects_erased entries for concrete objects the RHS may write absent from a cell where the LHS allows that object to be present in the same cell. This includes explicit no terms, removal from the original cell, and possible same-layer erasure caused by writing another object on the layer."
    }
  ]
}
```

Rule-tag values are arrays of concrete object display names. The runner compares these arrays as sets, so hand-written order is not meaningful.

Every object-bearing rule term in valid analyzed source should resolve to concrete object names. Failure to resolve such a term is an analyzer or runner error, not an omitted expectation.

## Expectation JSON Format

Each expectation JSON groups rule-tag expectations by rule.

```json
{
  "schema": "ps-static-analysis-testdata-v1",
  "ruleTag": [
    {
      "line": 48,
      "text": "right [ alpha | ] -> [ | stationary alpha ]",
      "tags": {
        "objects_required": ["Alpha"],
        "objects_matched": ["Alpha"],
        "object_absences_matched": [],
        "objects_written": ["Alpha"],
        "objects_erased": ["Alpha"]
      }
    }
  ]
}
```

Rules:

- `line` is the 1-based source line number for the rule.
- `text` is the trimmed rule source text on that line.
- `tags` contains only the rule tags that the fixture wants to check.
- Unknown rule-tag names fail against `static_analysis_claim_descriptions.json`.
- Listed tag values are compared as sets of object names.
- Unlisted rule tags are not checked.

## Rule Identity

`line` plus `text` must identify exactly one analyzed rule record.

If it matches zero analyzed rules, the runner fails. If it matches more than one analyzed rule, the runner fails. This slice does not add `index_on_line` or any other disambiguator. Fixtures should be written so their rule expectations are unambiguous.

The checks operate on analyzer rule records after compilation. The first specimens should therefore use already-simple source rules whose expected tags are invariant under basic rule compilation and rule-group decomposition. Rule-tag fixtures must use compiler-idempotent rule text, so the test is not about incidental expansion. Non-idempotent source examples belong in a separate fixture area whose explicit purpose is compiler normalization or rule decomposition.

## Semantics

The tags are conservative object-level facts, not precise rule-flow proofs.

For OR properties:

```text
Mover = Alpha or Beta
[ mover ] -> [ gamma ]
```

Expected rule tags:

```json
{
  "objects_required": [],
  "objects_matched": ["Alpha", "Beta"],
  "object_absences_matched": [],
  "objects_written": ["Gamma"],
  "objects_erased": ["Alpha", "Beta"]
}
```

For aggregates:

```text
Pair = Wall and Mark
[ wall mark ] -> [ ]
```

Expected rule tags:

```json
{
  "objects_required": ["Wall", "Mark"],
  "objects_matched": ["Wall", "Mark"],
  "object_absences_matched": [],
  "objects_written": [],
  "objects_erased": ["Wall", "Mark"]
}
```

For absent terms:

```text
Obstacle = Wall or Crate
[ no wall no crate player ] -> [ player stationary mark ]
```

Expected rule tags:

```json
{
  "objects_required": ["Player"],
  "objects_matched": ["Player"],
  "object_absences_matched": ["Wall", "Crate"],
  "objects_written": ["Mark"],
  "objects_erased": []
}
```

For relocation:

```text
right [ alpha | ] -> [ | stationary alpha ]
```

Expected rule tags:

```json
{
  "objects_required": ["Alpha"],
  "objects_matched": ["Alpha"],
  "object_absences_matched": [],
  "objects_written": ["Alpha"],
  "objects_erased": ["Alpha"]
}
```

## Generation Workflow

When the runner sees an orphan `rule_tags/foo.txt`, it may generate `foo.json` containing all analyzed rules and all known rule tags. The generated JSON must be reviewed by Stephen before commit, merge, or submission.

The runner never overwrites existing rule-tag JSON.

Generated rule expectations should use source order and the agreed `line` / `text` / `tags` structure. If a rule cannot be identified unambiguously by `line` plus `text`, generation should fail rather than inventing another locator.

Generation should also fail when the source rule text is not compiler-idempotent. The ordinary `rule_tags` suite is for already-normalized rule records, not compiler-normalization examples.

## Initial Specimen Themes

The first committed specimens should stay small and use already-simple rules:

- positive object read and deletion
- absent object read and object write
- relocation, using cell-local written/erased semantics

Only after those are green should we add property and aggregate specimens. Directional/movement decomposition belongs in a later, separately discussed slice.
