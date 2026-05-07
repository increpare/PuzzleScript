# Static Analysis Testdata

Each `.txt` file is a whole valid PuzzleScript source. Each matching `.json`
file lists the static-analysis expectations that should be checked for that
source.

## Adding An Object-Tag Test

1. Add a small whole-source `.txt` file under `object_tags/`.
2. Run:

   ```sh
   make static_analysis_tests
   ```

3. The runner will create the missing matching `.json` file and print a
   `generated static analysis testdata` message.
4. Read the generated JSON. Keep it as-is for a broad fixture, or delete
   expectations until it focuses on the specific object/tag facts the file is
   meant to protect.
5. Run `make static_analysis_tests` again. Existing JSON files are never
   overwritten, and only listed expectations are checked.

If an interpretation question comes up while writing a fixture, stop and decide
the tag meaning before committing the test.
