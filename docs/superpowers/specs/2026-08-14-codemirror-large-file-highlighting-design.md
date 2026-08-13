# Large-file syntax-highlighting correctness

## Context

PuzzleScript issue #947 reports that fast scrolling in a large project can leave regions with incorrect syntax highlighting. The problem is reproducible in the current editor with the 1,711-line `easyenigma` demo. A fast jump into the middle of the file initially renders object definitions with prelude styles such as `cm-METADATA` and `cm-ERROR`. If the background highlighter later advances past that region while it is off-screen, revisiting it leaves those styles wrong indefinitely.

PuzzleScript's CodeMirror mode is unusually global. Its parser state includes the current section, nested-comment state, declared objects, legend entries, rules, win conditions, levels, and related name tables. CodeMirror's normal fallback of restarting near a requested viewport can therefore produce an approximate state that is not valid for PuzzleScript.

## Root cause

When CodeMirror renders a viewport beyond its precise highlighting frontier, `getContextBefore` may reconstruct mode state from a nearby line rather than from the beginning of the document. The resulting styles are cached on each rendered line.

The sequential highlighting worker eventually reaches those lines with precise state. Its visible-line branch replaces cached styles, but its off-screen branch only advances parser state. Thus, if the user leaves an approximately highlighted viewport before the worker reaches it, the worker can pass the region without repairing its cached styles. Once the frontier is beyond the region, returning to it does not schedule more work for those lines.

Parser cost makes the race easier to trigger, but it is not the root cause of the permanent result: after the frontier has passed, additional waiting does not correct the cache.

## Goals

- Every line previously rendered with approximate state must receive precise cached styles when the sequential worker reaches it.
- Revisiting a skipped region must show correct token classes, including `cm-LEVEL`, so Ctrl/Cmd-click behavior continues to work.
- Lines that were never rendered must retain the existing lightweight off-screen parsing path.
- The change must remain narrowly scoped to highlighting correctness.

## Non-goals

- Optimizing PuzzleScript parser-state copying or eliminating all long-task warnings.
- Replacing CodeMirror 5 or redesigning the PuzzleScript parser.
- Re-highlighting the entire document eagerly.

## Considered approaches

### 1. Repair cached styles as the precise worker passes

When the worker processes an off-screen line that already has cached styles, run the normal highlighting path with the worker's precise context and replace the line's styles and line classes. Continue using the lightweight `processLine` path for off-screen lines without cached styles.

This directly repairs the stale cache at the point where precise state becomes available. Its extra work is limited to lines the user previously caused CodeMirror to render.

### 2. Clear cached styles as the precise worker passes

The worker could discard styles on previously rendered off-screen lines and let CodeMirror recompute them on a later visit. This is cheaper while the line remains off-screen, but delays correction and relies on a later rendering path to reconstruct state from the worker's saved checkpoints.

### 3. Force viewport refreshes from `editor.js`

A `viewportChange` handler could invalidate or refresh visible lines. This avoids modifying vendored CodeMirror code but duplicates scheduler behavior, may cause unnecessary redraws, and still needs access to CodeMirror internals to distinguish approximate from precise state.

## Chosen design

Use approach 1 in CodeMirror's highlighting worker.

In the off-screen branch of `highlightWorker`:

1. If the line has a `styles` cache, highlight it with the worker's current precise context.
2. Preserve CodeMirror's `maxHighlightLength` state-reset behavior.
3. Replace `line.styles` and update or clear `line.styleClasses` from the precise result.
4. Save the resulting parser checkpoint on the existing cadence.
5. If the line has no cached styles, retain the current lightweight `processLine` behavior.

No display invalidation is needed for this branch because the line is outside CodeMirror's rendered view. The next visit will build its DOM from the corrected cache. The visible-line branch remains unchanged and continues to register line changes when token classes differ.

## Testing

Add a deterministic browser regression test using full CodeMirror and a stateful mode/document that reproduces the frontier race:

1. Render a viewport far beyond the precise frontier so approximate styles are cached.
2. Move away before the worker reaches it.
3. Allow the sequential worker to pass the skipped region while it is off-screen.
4. Revisit the region.
5. Assert that its token classes now match the precise state.

Also rerun the existing Node test suite and manually repeat the `easyenigma` fast-scroll reproduction. Before the fix, the revisited object block remains `cm-ERROR`/`cm-METADATA`; after the fix, it uses the expected object-name, color, and sprite token classes.

## Follow-up

Parser/highlighter latency should be profiled separately. CodeMirror's worker time slices and PuzzleScript's deep `copyState` implementation are likely contributors, but optimizing them is not required for this correctness fix and carries different risks.
