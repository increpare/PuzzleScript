# Constraint-aware and analogous rule completions

**Goal:** Make editor suggestions respect provable language restrictions and repeat existing rules for defined name variants.

**Design:** Keep semantic analysis independent of CodeMirror rendering. Resolve names from the live parser state rather than the last compiled game. Parse only the rule prefix before the edited word for contextual checks; unknown or malformed input must not cause speculative filtering. Recognise late-rule restrictions, win-condition positions, certain same-cell layer conflicts, and the compiler's corresponding-cell/unique-occurrence property bindings. Explain property matches inline, including `no` and `random` exceptions.

Analogy completion searches preceding rules when a new line contains a variant prefix. Infer one consistent name-component substitution (CamelCase, underscore-separated, or numbered names), require every replacement to exist, preserve commands and message text, and show the complete generated rule and exact name changes. Prefer recent examples, deduplicate generated rules, and limit the list. Do not add fuzzy semantic inference or invoke the compiler on each keystroke. Existing directional completion remains available.

## Implementation

- [x] Add a Node test harness using the actual parser and registered hint helper. Cover the user's examples and incomplete-input exceptions; run `node --test src/tests/autocomplete_test.js` to confirm failures.
- [x] Add `src/js/codemirror/rule-completion.js` for symbol resolution, conservative rule-prefix analysis, and contextual name/keyword decisions. Wire into `anyword-hint.js`, `src/editor.html`, and `compile.js`.
- [x] Add `src/js/codemirror/rule-analogies.js` to infer and preview consistent substitutions from preceding rules. Cover colours, states, materials, stages, unknown names, duplicates, comments, messages, and insertion ranges.
- [x] Explain the new interactions in `src/Documentation/autocomplete.html`; style and inspect the rendered suggestions.
- [x] Run the focused completion suite, existing compiler/engine suite, syntax checks, and review the final diff.

**Validation:** `node --test src/tests/autocomplete_test.js`; `node src/tests/run_tests_node.js`; `git diff --check`. Preview the editor locally and verify the menu text and completion insertion.

**Results:** 25 completion tests and 770 existing compiler/game tests passed. The 47-file editor bundle minified and parsed successfully. Browser checks verified the full-rule preview, Tab insertion, and corresponding-cell property explanation. Independent review prompted regressions for distinct property-alias bindings, numeric-slot preservation, same-layer properties, and RHS random choice pools. The 1,000-object/500-rule no-match performance test improved from about 2.3 seconds to under 25 ms including parser setup.
