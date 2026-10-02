Surveyed all 51 open issues and their comments against `master` at commit [`4db08eb`](https://github.com/increpare/PuzzleScript/commit/4db08ebe1939411ac07eee916d3a98defa9820b8). I also checked recent history, the deployed site, and ran the current suite: 770/770 tests passed.

Difficulty: XS <½ day; S 1–2 days; M 3–5 days; L 1–2 weeks; XL architectural/multi-week; R requires reproduction/research.

## Critical and high

| Issue | Severity | Difficulty | Triage |
|---|---:|---:|---|
| [#1117 Gists unavailable without auth](https://github.com/increpare/PuzzleScript/issues/1117) | Critical, mitigated | M, done | The original total gallery/play outage justified Critical. The server-side proxy is deployed and currently working; code routes public loads through it. Monitor rate limits, then close or convert to an operational-monitoring issue. [Current proxy code](https://github.com/increpare/PuzzleScript/blob/4db08ebe1939411ac07eee916d3a98defa9820b8/src/js/github.js#L30-L59). |
| [#1175 Speculative AGAIN commits checkpoint](https://github.com/increpare/PuzzleScript/issues/1175) | High | S | Confirmed. `processInput(-1, true, true)` is a dry run, but checkpoint handling still changes `restartTarget`, `hasUsedCheckpoint`, and persistent storage. Guard all checkpoint side effects under `!dontModify`, and add a storage/restart-target regression test. [Relevant engine path](https://github.com/increpare/PuzzleScript/blob/4db08ebe1939411ac07eee916d3a98defa9820b8/src/js/engine.js#L2928-L2977). |
| [#1174 `flickscreen 0x5` crashes editor](https://github.com/increpare/PuzzleScript/issues/1174) | High | XS | Confirmed hang. The compiler warns about zero but retains it; editor grid rendering then executes a loop whose increment is zero. Reject/remove non-positive metadata dimensions. [Validation](https://github.com/increpare/PuzzleScript/blob/4db08ebe1939411ac07eee916d3a98defa9820b8/src/js/compiler.js#L2414-L2457), [infinite loop](https://github.com/increpare/PuzzleScript/blob/4db08ebe1939411ac07eee916d3a98defa9820b8/src/js/graphics.js#L412-L430). |
| [#1107 Memory leak](https://github.com/increpare/PuzzleScript/issues/1107) | High | S | Confirmed unbounded verbose-debug timeline; the reporter reached 868,000 snapshots. Bound it by turns/bytes, clear it on rebuild/load, or retain only snapshots still referenced by console entries. [Snapshot allocation](https://github.com/increpare/PuzzleScript/blob/4db08ebe1939411ac07eee916d3a98defa9820b8/src/js/editor.js#L317-L339). |

## Medium

| Issue | Difficulty | Triage |
|---|---:|---|
| [#1137 Multi-key repeat cycle](https://github.com/increpare/PuzzleScript/issues/1137) | S | Confirmed. New keys are inserted at `keyRepeatIndex`, disturbing round-robin order. Add deterministic keyboard/gamepad repeat tests and correct insertion/index adjustment. [Current insertion](https://github.com/increpare/PuzzleScript/blob/4db08ebe1939411ac07eee916d3a98defa9820b8/src/js/inputoutput.js#L446-L456). |
| [#1111 Held controller action leaks into level](https://github.com/increpare/PuzzleScript/issues/1111) | M | Strong code-level cause: message/level transitions clear the repeat buffer while the gamepad remains physically held, so the next poll treats it as a fresh press. Fix together with #1101 using separate physical-held and repeat state. |
| [#1101 One controller press skips messages](https://github.com/increpare/PuzzleScript/issues/1101) | M | Same transition/debouncing family as #1111. One shared state-machine fix and regression set should cover both. |
| [#1093 Large paste intermittently fails in Edge](https://github.com/increpare/PuzzleScript/issues/1093) | R/L | Retest first: CodeMirror was upgraded from 4.0.3 to 5.65.21 in 2026. If reproducible, capture clipboard events and document size; likely browser/editor integration rather than compiler logic. |
| [#1090 Safari sound latency](https://github.com/increpare/PuzzleScript/issues/1090) | R/M | Medium only if confirmed across current Safari. Needs device/version reproduction and WebAudio timing instrumentation before estimating a fix. |
| [#947 Syntax highlighting skips large regions](https://github.com/increpare/PuzzleScript/issues/947) | L | Still open after the CodeMirror upgrade and a 2026 investigation. It also disables Ctrl-click functionality, not merely coloring. Likely incremental-mode/parser scheduling work with substantial regression risk. |
| [#945 Object stacking/rule explosion](https://github.com/increpare/PuzzleScript/issues/945) | L | Repro generated 55,022 rules and can hang/crash compilation. Optimize the no-movement/movement-unchanged case first, guarded by benchmarks; general multi-layer property compression is core-engine work. |
| [#984 No copy/paste on iPad without keyboard](https://github.com/increpare/PuzzleScript/issues/984) | R/L | Merge into #931 as a concrete acceptance criterion, then retest after the CodeMirror upgrade. |
| [#931 iOS built-in keyboard editing is janky](https://github.com/increpare/PuzzleScript/issues/931) | R/L | Platform-specific but blocks practical authoring. Needs current iOS testing; likely constrained by CodeMirror 5/contenteditable behavior. |
| [#897 Cannot win and show message together](https://github.com/increpare/PuzzleScript/issues/897) | M | Confirmed architectural ordering: message output enters text mode before normal win checking. Define the desired command order, then add win+message tests; changing semantics may affect existing games. |
| [#619 GDPR compliance](https://github.com/increpare/PuzzleScript/issues/619) | M | Replace with a narrowly scoped privacy audit. The current deployed bundle contains no Google Analytics, so that premise is obsolete; it still uses a server proxy, GitHub OAuth, and local token storage. Add a concise privacy disclosure and obtain legal review rather than treating this as a normal code bug. [Live editor](https://www.puzzlescript.net/editor.html), [GitHub integration](https://github.com/increpare/PuzzleScript/blob/4db08ebe1939411ac07eee916d3a98defa9820b8/src/js/github.js). |
| [#1102 Xbox analogue down triggers X](https://github.com/increpare/PuzzleScript/issues/1102) | Done | Fixed by [`d0d3ad3`](https://github.com/increpare/PuzzleScript/commit/d0d3ad3766729d912c33d115d49fd1cc9b32c50b); close after a controller smoke test. |

## Low

| Issue | Difficulty | Triage |
|---|---:|---|
| [#1181 Remote editor load scrolls oddly](https://github.com/increpare/PuzzleScript/issues/1181) | XS | Reset cursor and scroll position after asynchronous `editor.setValue()`. Add a long-document load smoke test. |
| [#1179 `randomdir` allowed on LHS](https://github.com/increpare/PuzzleScript/issues/1179) | XS | Confirmed missing validation. Reject it beside the existing `random` LHS check; its current mask can produce misleading matches. |
| [#1178 Extra colors without sprite matrix](https://github.com/increpare/PuzzleScript/issues/1178) | XS | Add a compiler warning when the solid-color fallback will ignore additional colors. |
| [#1128 Bad parenthetical-in-rule error](https://github.com/increpare/PuzzleScript/issues/1128) | Done | Fixed, lost in a merge, then restored by [`247c097`](https://github.com/increpare/PuzzleScript/commit/247c097). Covered by compiler-message tests; close. |
| [#1123 No cross-group duplicate-elimination tests](https://github.com/increpare/PuzzleScript/issues/1123) | S | Test-gap issue. Current deduplication deliberately resets at group boundaries; write tests documenting whether that is semantic necessity or missed optimization. |
| [#1116 Repeatable mobile Undo](https://github.com/increpare/PuzzleScript/issues/1116) | S | Straightforward touch-hold timer with cancellation on `touchend`/`touchcancel`; reuse movement repeat rate. |
| [#1110 Synonyms/property overlap on same layer](https://github.com/increpare/PuzzleScript/issues/1110) | M | A fix was attempted and immediately reverted. Decide between warning, silent deduplication, or rejection; cover direct, synonym, and property overlaps before reimplementing. |
| [#1105 Search/replace button](https://github.com/increpare/PuzzleScript/issues/1105) | XS | The new search panel improves discoverability but still lacks a clickable entry point. Add a toolbar item invoking CodeMirror’s replace command. |
| [#1104 Directional legend autocomplete](https://github.com/increpare/PuzzleScript/issues/1104) | Done | Implemented by [`584f816`](https://github.com/increpare/PuzzleScript/commit/584f816); close after a quick editor check. |
| [#1103 Allow `_n/_s/_e/_w`](https://github.com/increpare/PuzzleScript/issues/1103) | XS | Add the four suffixes to directional pairings/transforms and autocomplete tests. |
| [#1087 Test failure count wrong](https://github.com/increpare/PuzzleScript/issues/1087) | Done/verify | The old QUnit path was replaced by a per-test runner in [`1b2e080`](https://github.com/increpare/PuzzleScript/commit/1b2e080). Recreate the screenshot scenario once, then close. |
| [#1086 `moving` rule expands four ways](https://github.com/increpare/PuzzleScript/issues/1086) | L | Merge into #720; it is the same movement concretization trade-off, not a separate bug. |
| [#1042 Right-click GIF download lacks extension](https://github.com/increpare/PuzzleScript/issues/1042) | External | Browser/data-URL limitation. The explicit “Download GIF” link already supplies a filename. Close as wontfix unless switching GIF output to Blob/Object URLs is independently worthwhile. |
| [#1027 Flickscreen map editor](https://github.com/increpare/PuzzleScript/issues/1027) | L | Valuable enhancement but requires editor-only camera/panning/input behavior; previous contributor backed out an attempt. |
| [#1019 Poor error during speculative AGAIN](https://github.com/increpare/PuzzleScript/issues/1019) | M | Fix alongside #1175 by introducing an explicit speculative-execution context: suppress or annotate diagnostics and preserve the originating AGAIN rule. |
| [#1018 Verbose logging omits loop boundaries](https://github.com/increpare/PuzzleScript/issues/1018) | M | Preserve source loop boundaries into compiled group metadata, then emit start/end/re-entry logging. Diagnostic-only. |
| [#916 `again_interval` affects key repeat](https://github.com/increpare/PuzzleScript/issues/916) | S/M | Current timers are separate, but repeat timers continue cycling while AGAIN suppresses input. Write timing tests; probably a small state-reset fix. |
| [#915 `nooverlap` keyword](https://github.com/increpare/PuzzleScript/issues/915) | L | Language/semantics design, matcher changes, documentation, and compatibility tests. |
| [#914 Multiple-ellipsis expansion order](https://github.com/increpare/PuzzleScript/issues/914) | L | Undefined public semantics; specify behavior before changing it because existing games may rely on current ordering. |
| [#913 Cache multiple-ellipsis matching](https://github.com/increpare/PuzzleScript/issues/913) | L | Plausible quadratic hot path, but benchmark first. Core generated matcher change. |
| [#910 Inline matches in ellipsis rules](https://github.com/increpare/PuzzleScript/issues/910) | L | Performance experiment; require representative benchmarks and generated-code size measurements. |
| [#753 Multi-layer property expansion](https://github.com/increpare/PuzzleScript/issues/753) | XL | Potentially large performance win, but prior attempt reduced Cyber Lasso’s rules by ~30% while breaking semantics/tests. High-risk compiler architecture project. |
| [#737 Deduplicate expanded property rules](https://github.com/increpare/PuzzleScript/issues/737) | M | First instrument the suite to measure frequency; implement only if meaningful. |
| [#720 Smarter movement collection](https://github.com/increpare/PuzzleScript/issues/720) | XL | Requires non-concrete movement matching in the engine, affecting core bitmask tests. Keep as the umbrella for #1086. |
| [#714 Alternative browser-key-repeat fix](https://github.com/increpare/PuzzleScript/issues/714) | R | Stale branch-review note. Close and reference it from #1137 if any useful reasoning remains. |
| [#676 Restart/start-level message override](https://github.com/increpare/PuzzleScript/issues/676) | M/L | Genuine but very narrow. Proper fix likely needs queued messages or explicit command precedence, with compatibility tests. |
| [#395 Random object plus random movement](https://github.com/increpare/PuzzleScript/issues/395) | M | Language limitation: `random` and `randomdir` collide as modifiers. Define grammar for independent object selection and movement before implementing. |
| [#337 Comments inside rules](https://github.com/increpare/PuzzleScript/issues/337) | M | Parser feature with source-line/error-reporting implications. Low impact after nine years without blocking reports. |

## Non-defects, editorial, or underspecified

| Issue | Difficulty | Triage |
|---|---:|---|
| [#1182 Pivot/balance-board experiment](https://github.com/increpare/PuzzleScript/issues/1182) | R | Empty design note; needs a concrete example and acceptance criteria before estimation. |
| [#1098 Games to add to gallery](https://github.com/increpare/PuzzleScript/issues/1098) | XS each | Editorial/content queue, not an engineering issue. |
| [#1085 Short vs long random SFX](https://github.com/increpare/PuzzleScript/issues/1085) | R | Empty product idea; specify desired UI/algorithm and examples. |
| [#912 Fuzz compiler with downloaded gists](https://github.com/increpare/PuzzleScript/issues/912) | Ongoing | Convert to a repeatable fuzzing job/checklist with a corpus and success criteria; recent compiler-fuzzer fixes show this work is already happening. |
| [#908 Self-host shared games](https://github.com/increpare/PuzzleScript/issues/908) | XL | Product, moderation, abuse prevention, storage, privacy, and operational ownership decision—not primarily a code fix. |
| [#844 Rethink default examples](https://github.com/increpare/PuzzleScript/issues/844) | Ongoing | Editorial project; partly addressed by [`3d0bc3a`](https://github.com/increpare/PuzzleScript/commit/3d0bc3a). Replace with a finite checklist. |
| [#702 Submit PuzzleScript to Linguist](https://github.com/increpare/PuzzleScript/issues/702) | M | External feature: create/maintain a TextMate grammar and resolve the generic `.txt` extension problem with Linguist maintainers. |

Recommended order:

1. Fix #1174, #1175, and #1107.
2. Address #1137, then combine #1101/#1111 into one controller-transition fix.
3. Take the cheap diagnostics/UI wins: #1179, #1178, #1181, #1103, #1105, #1116.
4. Close or verify-close #1128, #1104, #1102, #1087, and #1117.
5. Defer the property/movement/ellipsis optimizations until each has a benchmark demonstrating material impact.

I made no changes to the repository or issue tracker.