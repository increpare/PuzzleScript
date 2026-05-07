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
      "name": "is_player",
      "description": "Object is part of the resolved Player role.",
      "specification": "An object has is_player when it is the object named by the Player object, Player synonym, or a member of the Player property in the compiled game. If Player resolves to a property with multiple member objects, every member object has is_player true."
    },
    {
      "name": "is_background",
      "description": "Object is part of the resolved Background role.",
      "specification": "An object has is_background when it is the object named by the Background object, Background synonym, or a member of the Background property in the compiled game. If Background resolves to a property with multiple member objects on the same collision layer, every member object has is_background true."
    },
    {
      "name": "level_presence",
      "description": "Whether the object appears in all, some, or no playable levels.",
      "specification": "The level_presence tag is one of all, some, or none. Message levels are ignored. The value all means there is at least one playable level and the object appears in every playable level. The value some means the object appears in at least one but not every playable level. The value none means the object appears in no playable levels; if a valid source has zero playable levels, every object has level_presence none.",
      "values": ["all", "some", "none"]
    },
    {
      "name": "not_created_or_destroyed_by_rules",
      "description": "No solver-active rule creates or destroys this object.",
      "specification": "An object has not_created_or_destroyed_by_rules when the analyzer proves no solver-active rule can create or destroy an instance of that object according to rule object-write analysis. Pure movement or relocation of an existing object does not count as creation or destruction."
    }
  ]
}
```

Each entry has:

- `name`: the analyzer output key.
- `description`: a short human-facing explanation.
- `specification`: the precise contract the analyzer and tests are accountable to.
- `values`: optional list for valued tags. Omit it for boolean tags.
- `examples`: optional list of real testdata stems, such as `object_tags/roles-basic`. Examples must correspond to committed `.txt`/`.json` files.

There is no separate label field, no status field, and no expectation-type field. If a claim appears in this file, it is a valid, decided claim. Planned or speculative analyses do not belong here.

Claim order in this file is the generated expectation order.

The testdata vocabulary is author-facing. The runner may derive these claims from analyzer output if the analyzer stores them differently internally. For example, the first implementation can derive `level_presence` from existing `present_in_all_levels`, `present_in_some_levels`, and `present_in_no_levels` booleans instead of requiring the analyzer to change its raw output immediately.

The first implementation can derive `not_created_or_destroyed_by_rules` from the existing `count_invariant` analyzer tag. The testdata vocabulary uses the more precise name; the raw analyzer output can be renamed in a later semantic cleanup if desired.

Initial object tags:

- `is_player`
- `is_background`
- `level_presence`, with values `all`, `some`, and `none`
- `not_created_or_destroyed_by_rules`

`static` and `cosmetic` are intentionally not in the first slice. `static` depends on movement/write/layer-creation analysis. `cosmetic` depends on core-seed selection, rule object read/write extraction, rule/core reachability, and collision-layer closure. Those prerequisite analyses need their own testdata before these derived claims are formally added.

## Expectation JSON Format

Each expectation JSON uses a flat `expect` list.

```json
{
  "schema": "ps-static-analysis-testdata-v1",
  "note": "Optional human note.",
  "expect": [
    { "type": "objectTag", "object": "Background", "tag": "is_background", "is": true },
    { "type": "objectTag", "object": "Player", "tag": "is_player", "is": true },
    { "type": "objectTag", "object": "Background", "tag": "level_presence", "is": "all" },
    { "type": "objectTag", "object": "Player", "tag": "not_created_or_destroyed_by_rules", "is": true }
  ]
}
```

Rules:

- `name` is not required on individual expectations.
- `note` is optional at top level and per expectation.
- `object` uses the analyzer's display object name.
- `type: "objectTag"` is the only expectation type in the first slice.
- `is` is either a boolean for boolean tags or one of the tag's listed `values` for valued tags.
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
- explicit values for every object tag known at generation time: `true` and `false` for boolean tags, named values for valued tags

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
object_tags/roles-basic.json
objectTag Player.is_player expected true, got false
  object: Player id=1 layer=2
```

The runner should be wired into `make static_analysis_tests` beside the existing node assertion files. Existing assertions can migrate later, but migration is not part of the first slice.

## First Specimen Themes

The first object-tag specimens should be small standalone games:

- level presence: all, some, and no playable levels
- structural roles: player and background object/property resolution
- objects that are, and are not, created or destroyed by rules

Each specimen should be as small as possible while still being a whole valid source file.

## Later Work

Later slices may add:

- `static` object tags, after movement/write/layer-creation analysis has testdata
- `cosmetic` object tags, after core closure and rule-flow prerequisites have testdata
- rule tag expectation types
- rule-group expectation types with line/text locators
- layer tag expectation types
- fact-family expectation types
- a browser specimen builder that can paste/open PuzzleScript source, show grouped analysis claims, and export selected expectation JSON
- dashboard support using `static_analysis_claim_descriptions.json`

These later slices should reuse the same discipline: decide the interpretation first, then add a passing testdata specimen.
