# Static Analysis Testdata

Each `.txt` file is a whole valid PuzzleScript source. Each matching `.json`
file lists the static-analysis expectations that should be checked for that
source.

Object-tag expectations are grouped by object:

```json
{
  "schema": "ps-static-analysis-testdata-v1",
  "objectTag": [
    {
      "object": "Player",
      "is_player": true,
      "is_background": false,
      "level_presence": "all",
      "created_by_rules": false,
      "destroyed_by_rules": false
    }
  ]
}
```

## Adding An Object-Tag Test

1. Add a small whole-source `.txt` file under `object_tags/`.
2. Run:

   ```sh
   make static_analysis_tests
   ```

3. The runner will create the missing matching `.json` file and print a
   `generated static analysis testdata` message.
4. Read the generated JSON. Keep it as-is for a broad fixture, or delete
   object rows or tag fields until it focuses on the specific facts the file is
   meant to protect.
5. Run `make static_analysis_tests` again. Existing JSON files are never
   overwritten, and only listed expectations are checked.

## Adding A Rule-Tag Test

1. Add a small whole-source `.txt` file under `rule_tags/`.
2. Run:

   ```sh
   make static_analysis_tests
   ```

3. The runner will create the missing matching `.json` file and print a
   `generated static analysis testdata` message. If the source rule text is not
   compiler-idempotent, generation fails instead.
4. Read the generated JSON with Stephen before committing it. Keep only the
   rule rows and tag fields that express the intended test.
5. Run `make static_analysis_tests` again. Existing JSON files are never
   overwritten, and only listed rule-tag expectations are checked.

Rule-tag expectations identify rules by `line` plus trimmed source `text`.
If those two fields do not identify exactly one analyzed rule, the runner fails
instead of inventing another locator.

Movement rule tags use flat `Object:movement` strings. `movements_matched`
contains concrete object movements that may satisfy an LHS movement term.
`movements_required` is the definite subset for single-object movement terms.
`movements_written` contains concrete object movements written by the RHS, or
`Object:stationary` when a movement requirement is cleared while the object
stays present in the corresponding RHS cell. `movements_removed` contains
concrete LHS movement matches that are not preserved in the corresponding RHS
cell, including consumed `action` matches.

Rule-tag test sources must use compiler-idempotent rule text, so the fixture is
testing the analysis rather than incidental rule expansion. Non-idempotent rule
text belongs in a separate fixture area whose explicit purpose is compiler
normalization or rule decomposition.

Rule-tag tests should use the shared object/layer scaffold when the normal
PuzzleScript-ish layer shape is part of the specimen:

```text
Background
Target
Player, Wall, Crate
Alpha, Beta, Gamma
```

Use these names consistently so rule expectations can be read from the rule
text without re-learning the fixture's collision layers.

When a rule-tag fixture needs extra layer structure, name that structure
directly:

- `Orphan1`, `Orphan2`, ... are helper objects alone on their own layers.
- `Sibling1`, `Sibling2`, ... are helper objects that deliberately share one
  fixture-local layer.
- Property names should spell out their expansion, for example
  `Player_or_Crate = Player or Crate` or
  `Sibling1_or_Sibling2 = Sibling1 or Sibling2`.

Avoid names that describe the expected analyzer output. The object names should
describe the specimen's layer/property structure, not whether a tag is expected
to report an object as matched, absent, written, or erased.

## Adding A Program-Flow Test

1. Add a small whole-source `.txt` file under `program_flow/`.
2. Run `make static_analysis_tests` (or `node src/tests/static_analysis_testdata_runner.js`).
3. The runner will create the missing matching `.json` file. Each entry under
   `wakeEdges` lists one ordered (from, to) rule pair where firing `from`
   may enable a new match for `to`, with the reasons that triggered the edge
   (`object_presence`, `object_absence`, or `movement`). Each entry under
   `againRules` lists a rule whose firing semantically forces a tick restart.
4. Read the generated JSON and confirm by hand that every edge corresponds to
   an actual write-on-from / read-on-to relationship in the source.

Like rule-tag tests, edges identify rules by `line` plus trimmed source `text`,
so program-flow sources should also use compiler-idempotent rule text.

## Review Policy

Auto-generated or mechanically regenerated expectation JSON is a proposal, not
something to submit unseen. Before committing, merging, or submitting generated
testdata, show Stephen the generated expectations or diff and get explicit
approval.

If an interpretation question comes up while writing a fixture, stop and decide
the tag meaning before committing the test.
