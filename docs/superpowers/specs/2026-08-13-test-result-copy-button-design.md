# Test Result Copy Button

## Goal

Add a copy control to every PuzzleScript browser-test result in `src/tests/tests.html`. The control must copy only the PuzzleScript program used by that test, so the copied text can be pasted directly into the PuzzleScript editor. It must appear for both passing and failing results.

Current QUnit does not provide a built-in control for copying application-specific test fixtures. The project is also intentionally pinned to QUnit 1.12.0 because later test identifiers conflict with its numerical-ID workflow, so this feature will extend the existing reporter rather than upgrade or fork QUnit.

## Architecture

Add a small, testable browser helper under `src/tests/resources/` and load it from `src/tests/tests.html` after QUnit and the two test-data arrays, but before `resources/tests.js`.

The page will initialize the helper with one ordered list of source programs:

1. `testdata[*][1][0]`, followed by
2. `errormessage_testdata[*][1][0]`.

The helper will subscribe to `QUnit.testDone`. While that callback runs, QUnit 1.12.0 exposes the completed test as `QUnit.config.current`. Its one-based `testNumber` identifies the matching source in the ordered list, and its DOM `id` identifies the result panel to decorate. Because QUnit assigns test numbers before applying URL filters, the association remains correct when a single test or a filtered subset is run.

QUnit itself and the individual test definitions will remain unchanged.

## Result Panel UI

Each completed top-level result panel will receive one native `<button type="button">` immediately before its assertion list. The button will display:

- an inline two-sheets copy SVG and the label **Copy text** initially;
- a check SVG and **Copied** for two seconds after success; or
- **Copy failed** for two seconds if both clipboard methods fail.

The control will use the existing QUnit palette, a compact rounded-button treatment, hover/focus states, an accessible `aria-label`, and polite status announcement. The SVG is inline so the test page gains no external asset or network dependency.

The button click will stop propagation so it cannot toggle or rerun the test result.

## Selection and Clipboard Flow

When decorating a panel, the helper will find the assertion message whose text starts with the matching PuzzleScript program. It will wrap only that prefix in a dedicated source element while preserving the message's remaining input, target-level, seed, audio, and diagnostic text unchanged.

On activation, the helper will:

1. expand the assertion list if it is collapsed;
2. create a browser selection covering only the wrapped PuzzleScript program;
3. call `navigator.clipboard.writeText(source)` when available;
4. fall back to `document.execCommand("copy")`, using the active selection, when the Clipboard API is unavailable or rejects; and
5. retain the selection if copying fails, allowing the user to copy it manually.

No input, target-level, random-seed, audio, assertion-diff, stack-trace, or status text will enter the clipboard.

## Error Handling

If a completed QUnit result has no matching source or the expected assertion message cannot be found, the helper will leave that panel unchanged rather than corrupt its reporter markup. Normal simulation and error-message tests always include the source-prefixed assertion message.

Clipboard failures will not affect the test result. The button will show its temporary failure state, the source will remain selected, and a diagnostic will be written to the browser console when available.

## Testing

Focused tests will exercise the helper with a minimal fake DOM and clipboard environment. They will verify:

- source lookup by QUnit test number, including a filtered/nonconsecutive executed test;
- decoration of both passing and failing panels;
- one button per result and no duplicate decoration;
- preservation of metadata and diagnostic text;
- selection of only the PuzzleScript program;
- Clipboard API success;
- fallback copying when the Clipboard API is absent or rejects;
- retained selection and failure feedback when all copy methods fail; and
- temporary success feedback and restoration of the normal label.

The existing Node test suite will then be run to confirm that loading the helper does not change PuzzleScript test behavior.

## Out of Scope

- Upgrading or modifying QUnit.
- Copying the test input, expected output, target level, random seed, audio expectations, diffs, or stack traces.
- Adding controls to the command-line test runner.
- Refactoring the existing test-data format or test registration loops.
