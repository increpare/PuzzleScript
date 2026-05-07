# Static Analysis Testdata Design

## Purpose

Build a careful, low-friction test battery for PuzzleScript static-analysis claims.

The immediate goal is not to add new analysis semantics. The immediate goal is to make existing and future claims easy to specify, inspect, and test one by one. Static-analysis features should only enter the testdata suite after their interpretation has been decided. There is no pending or expected-failure area.

The first implementation slice covers object tags only. Rule tags, layer tags, fact families, dashboard generation, and fixture-authoring UI are later slices.

## Design Principles

- Every test source is a complete valid PuzzleScript source file.
- Default to small, diagnosis-friendly specimens. Larger specimens are allowed when the interaction itself is the point.
- A test expectation file checks only the claims it lists. Unlisted analyzer output is not part of that test.
- Existing expectation JSON files are curated and must never be overwritten automatically.
- Adding a new analysis claim later must not break existing expectation JSON files.
- If an interpretational question arises, stop and ask before encoding or changing analyzer semantics.

## File Layout

```text
src/tests/static_analysis_claim_descriptions.json
src/tests/static_analysis_testdata/
  object_tags/
    *.txt
    *.json
src/tests/static_analysis_testdata_runner.js
```

`static_analysis_testdata/` is grouped by broad analysis area, not by individual tag. The first area is `object_tags/`.

Every committed `.json` in an area must have a matching `.txt` with the same stem. A `.json` without a `.txt` is an error.

## Claim Descriptions

`src/tests/static_analysis_claim_descriptions.json` is the shared glossary for static-analysis claims. It is consumed by tests now and can be consumed by the dashboard later.

The first version contains only `objectTags`.

```json
{
  "schema": "ps-static-analysis-claim-descriptions-v1",
  "objectTags": [
    {
      "name": "cosmetic",
      "description": "Object is outside the solver-visible core closure.",
      "specification": "An object has the cosmetic tag when the analyzer proves it is outside the current solver-visible core closure for object identity: it is not a player object, not referenced by win conditions, not read by win-command rules, and not reached by the analyzer's rule read/write closure from those core objects."
    }
  ]
}
```

Each entry has:

- `name`: the analyzer output key.
- `description`: a short human-facing explanation.
- `specification`: the precise contract the analyzer and tests are accountable to.
- `examples`: optional list of real testdata stems, such as `object_tags/cosmetic-basic`. Examples must correspond to committed `.txt`/`.json` files.

There is no separate label field, no status field, and no expectation-type field. If a claim appears in this file, it is a valid, decided claim. Planned or speculative analyses do not belong here.

Claim order in this file is the generated expectation order.

Initial object tags:

- `present_in_all_levels`
- `present_in_some_levels`
- `present_in_no_levels`
- `may_be_created`
- `may_be_destroyed`
- `count_invariant`
- `static`
- `cosmetic`

## Expectation JSON Format

Each expectation JSON uses a flat `expect` list.

```json
{
  "schema": "ps-static-analysis-testdata-v1",
  "note": "Optional human note.",
  "expect": [
    { "type": "objectTag", "object": "Background", "tag": "cosmetic", "is": true },
    { "type": "objectTag", "object": "Player", "tag": "cosmetic", "is": false }
  ]
}
```

Rules:

- `name` is not required on individual expectations.
- `note` is optional at top level and per expectation.
- `object` uses the analyzer's display object name.
- `type: "objectTag"` is the only expectation type in the first slice.
- Only listed expectations are checked.
- For a listed boolean expectation, a missing analyzer tag is interpreted as `false`.
- New claim descriptions added later do not affect existing expectation files.

Object names are sufficient locators for object-tag expectations. Later rule and rule-group expectations will need stronger locators such as source line plus source-text guard.

## Auto-Generation Workflow

The test runner supports an authoring shortcut for orphan `.txt` files.

When it finds `src/tests/static_analysis_testdata/object_tags/foo.txt` with no matching `foo.json`, it:

1. Analyzes `foo.txt`.
2. Generates `foo.json`.
3. Prints a clear message such as `generated static analysis testdata: object_tags/foo.json`.
4. Continues the test run and checks the generated JSON.

Generated object-tag JSON includes:

- all objects, including Background and Player
- object order from the analyzer/compiler
- object tags in claim-description order
- explicit `true` and `false` values for every object tag known at generation time

The runner never overwrites an existing `.json`. There is no draft status. Once generated, the file is a legitimate test and may be trimmed by hand.

## Runner Behavior

The runner scans `src/tests/static_analysis_testdata/` and handles each area according to that area's rules.

For `object_tags/`:

- valid `.txt` files are full PuzzleScript sources
- orphan `.txt` files generate matching `.json`
- orphan `.json` files fail
- invalid PuzzleScript source fails
- unknown expectation types fail
- unknown object names fail with a list of available object names
- unknown tag names fail against `static_analysis_claim_descriptions.json`

Expectation failures should be domain-specific, not generic JSON path errors.

Example failure:

```text
object_tags/static-basic.json
objectTag Wall.static expected true, got false
  object: Wall id=2 layer=1
```

The runner should be wired into `make static_analysis_tests` beside the existing node assertion files. Existing assertions can migrate later, but migration is not part of the first slice.

## First Specimen Themes

The first object-tag specimens should be small standalone games:

- level presence: all, some, and no playable levels
- created and destroyed objects
- count-invariant objects
- static objects versus player/movement-affected objects
- cosmetic objects
- Background and Player edge cases

Each specimen should be as small as possible while still being a whole valid source file.

## Later Work

Later slices may add:

- rule tag expectation types
- rule-group expectation types with line/text locators
- layer tag expectation types
- fact-family expectation types
- a browser specimen builder that can paste/open PuzzleScript source, show grouped analysis claims, and export selected expectation JSON
- dashboard support using `static_analysis_claim_descriptions.json`

These later slices should reuse the same discipline: decide the interpretation first, then add a passing testdata specimen.
