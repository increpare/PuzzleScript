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

Rule-tag test sources must use compiler-idempotent rule text, so the fixture is
testing the analysis rather than incidental rule expansion. Non-idempotent rule
text belongs in a separate fixture area whose explicit purpose is compiler
normalization or rule decomposition.

## Review Policy

Auto-generated or mechanically regenerated expectation JSON is a proposal, not
something to submit unseen. Before committing, merging, or submitting generated
testdata, show Stephen the generated expectations or diff and get explicit
approval.

If an interpretation question comes up while writing a fixture, stop and decide
the tag meaning before committing the test.
