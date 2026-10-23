# Cold compilation profiling on Steam Deck

Handoff prepared 23 October 2026. The next task is to measure **cold, isolated compilation on the actual Steam Deck**, then choose optimizations from that evidence. No compiler optimizations have been made in this work.

Use real, unchanged gallery games. Include Cyber-Lasso despite its compiler errors. Prioritize the first compilation in a fresh process; repeated compilation is secondary.

## What cold compilation measures

A fresh game's first compilation is the right primary compiler metric. Measure the first `compile(["restart"], source, "gallery-profile")` after engine initialization in a fresh process, with no earlier game compilation, compiler-cache reuse, or warmup.

Player launch time also includes engine JavaScript loading/evaluation, source retrieval, rendering, and deferred function generation. Keep those timings separate and validate the eventual improvement in the Deck's browser. A faster compile timer alone does not establish faster launch to play.

The current profiler stops at the title-screen state and leaves lazy function generation enabled. It does not run the browser update loop. The normal player generates queued functions over title-screen frames, and entering a gameplay level can drain the queue. Moving more work into that queue must not be counted as a launch improvement without measuring the resulting wait.

Relevant code:

- `src/js/compiler.js`: `compile`, `loadFile`, and the named compilation phases.
- `src/js/globalVariables.js`: `WORKLIST_OBJECTS_TO_GENERATE_FUNCTIONS_FOR`, `tick_lazy_function_generation`, and backlog reset. The current time-sliced loop uses a **20 ms** budget; its nearby comment saying 10 ms is stale.
- `src/js/inputoutput.js`: `update` runs background function generation and waits for an empty backlog before advancing from the title screen.
- `src/js/engine.js`: `setGameState`, `loadLevelFromLevelDat`, and `loadFirstNonMessageLevel`/`loadLevel` paths.
- `src/play.html`: the normal gist player calls `compile(["restart"], code)` after retrieving the source.
- `src/standalone.html`: standalone games compile their embedded source.

## Repository and corpus state

PuzzleScript branch: `compiler-optimizations`.

Handoff checkout: `95f3443c8310e4952595a083ccfb808b75771f90`. The Mac timing metadata records `b388a9d2bf6fbd86bf54f0c4413da4e2cd5ea563`; the intervening commit removes a gallery entry and GIF. All measured engine-source hashes were verified unchanged. Keep an explicit baseline revision and record any later working-tree changes.

The original corpus and tools are **outside this repository**, at:

```text
/Users/stephenlavelle/Documents/google_gist_scraper/gallery_profiling/
```

They will not arrive with a PuzzleScript clone or pull. The raw scraper tree is much larger and is unnecessary for profiling this frozen corpus.

| Corpus status | Count | Treatment |
| --- | ---: | --- |
| Verified gallery sources | 140 | Primary benchmark, all included |
| Provisional source mappings | 2 | Seven-Layer Dip and Lock & Pin; cached gist IDs inferred from gallery thumbnails; keep separate until verified |
| Current gallery entries | 142 | All have stored source snapshots; 140 have verified provenance |
| Unresolved current entries | 0 | No missing-source work required before profiling the verified set |

Each source has a SHA-256, byte count, original title, URL, and provenance in `manifest.json`. `gallery-index.tsv` is the readable index. Source files are unchanged, including comments, levels, and diagnostics. The benchmark must retain the same source hashes across comparisons.

**Cyber-Lasso stays included.** It works despite two compiler errors. The corrected inclusion rule is a completed compile, no exception, an installed compiled state, and available levels. Error count and diagnostic text are recorded separately; a nonzero count does not exclude a game. `verify-cyber.cjs` tests this using the actual source.

Five entries and their GIFs were removed at the user's request: Aunt Flora's Mansion, The closet and the castle, My first Kishoutenketsu, Play Mini Gemini Replay (PMGRP), and Entanglement – Chapter One. Keep them removed. Entanglement has a scraped 2017 source that compiles cleanly with the 2017 engine, but receives eight errors today; the user explicitly chose to keep it removed after that finding.

The 785 regression source records remain useful for correctness checks. None is byte-identical to these 142 full gallery snapshots. Do not substitute regression fixtures or synthetic games into the performance corpus, and do not trust the scraper's historical `dumpprocessed_compiles` folder as today's inclusion filter.

## Transfer the frozen corpus

A portable archive has been prepared on the Mac:

```text
/Users/stephenlavelle/Documents/GitHub/PuzzleScript/build/profiling-handoff/puzzlescript-gallery-corpus-2026-10-23.tar.gz
```

SHA-256:

```text
7f1f17f2d819bb98f38d639a727735dd81389ee903e52b3949fe85a5e0a7604f
```

The archive is about 1.3 MB and contains `gallery_profiling/` with all 142 source snapshots, the manifest/index, runnable profiler and summarizer, Cyber-Lasso verification, and current Mac reports/raw timing and CPU data. The archive's source bytes and hashes have been checked against the manifest.

**The archive is in Git-ignored `build/`: transfer it separately from the checkout.** A matching `.sha256` file sits alongside it. Copy the archive to the Deck using the user's usual file-transfer method; no network service is needed to extract it.

Example Deck paths below are choices, not assumptions about an existing installation. Point `PUZZLESCRIPT_ROOT` at the actual checkout. This environment override is required because the frozen manifest records the original Mac path.

```bash
# After copying the archive into Downloads and checking out PuzzleScript:
sha256sum "$HOME/Downloads/puzzlescript-gallery-corpus-2026-10-23.tar.gz"
mkdir -p "$HOME/Projects/ps-profile"
tar -xzf "$HOME/Downloads/puzzlescript-gallery-corpus-2026-10-23.tar.gz" \
  -C "$HOME/Projects/ps-profile"

export PUZZLESCRIPT_ROOT="$HOME/Projects/PuzzleScript"
export PS_CORPUS_ROOT="$HOME/Projects/ps-profile/gallery_profiling"

git -C "$PUZZLESCRIPT_ROOT" status --short --branch
git -C "$PUZZLESCRIPT_ROOT" rev-parse HEAD
node --version
node -p 'JSON.stringify({v8:process.versions.v8,platform:process.platform,arch:process.arch,cpu:require("node:os").cpus()[0].model})'

# Smoke check only; this separate process also performs warm compiles.
node "$PS_CORPUS_ROOT/verify-cyber.cjs"
```

The profiling scripts use built-in Node modules and require no `npm install`. The Mac used Node `v25.8.1`, V8 `14.1.146.11-node.21`; record the actual Deck runtime and keep it consistent across A/B runs.

The archive supports profiling existing snapshots. It omits `prepare-corpus.cjs`, the full scrape, historical captures, and downloaded page evidence. Regenerating the inventory requires those original Mac-side inputs. Absolute cache/provenance paths inside the manifest are historical records; the profiler reads the packaged relative `sources/` paths.

Preserve the extracted Mac baseline before writing Deck output, for example by copying the directory to `gallery_profiling-deck-baseline` and changing `PS_CORPUS_ROOT` to that copy. The existing scripts write results into their own directory.

## What has already been measured

The Mac was an Apple M1 Max, macOS/arm64. For each game there were three fresh Node workers, each recording its first compile, then three untimed warmups and seven warm samples. The first compile is a genuinely isolated cold compile; the later samples are warm. Source reads and engine evaluation happen before the timer. Canvas, audio, and editor DOM work are stubbed; `IDE=false`, `unitTesting=true`, and normal lazy function generation is enabled.

The corrected dataset includes all 140 verified games: 420 cold samples, 2,940 warm samples, and 420 matching instrumented/control compiled-state JSON hashes and diagnostics. Cyber-Lasso's three cold runs and 21 warm samples were refreshed after correcting its inclusion policy; other games' original samples were retained after hash checks. The amendment is recorded in `timings.json`.

Current Mac cold results, for orientation:

| Measurement | Result |
| --- | ---: |
| Median across per-game cold medians | 19.26 ms |
| 95th percentile of per-game cold medians | 118.82 ms |
| Parsing share of pooled cold time | 27.82% |
| `rulesToMask` | 21.47% |
| `rulesToArray` | 15.60% |
| `collapseRules` | 14.03% |
| `levelsToArray` | 5.61% |

These are instrumented compilation timings, not navigation-to-play timings. Inspect `PROFILE.txt`, `summary.json`, and `results.tsv` for details; `timings.json` contains every sample and engine/source hashes.

Useful slow cases for the first Deck traces:

| Game | Mac cold median |
| --- | ---: |
| Vertebrae | 411.87 ms |
| Season Finale | 225.69 ms |
| Crate Assembler | 191.83 ms |
| VEXT EDIT | 163.99 ms |
| Cyber-Lasso | 163.19 ms |
| Pornography for Beginners | 153.86 ms |
| At the Hedges of Time | 143.02 ms |
| Single Screen Scream | 118.82 ms |

Include smaller games such as Microban alongside these outliers, then measure the full 140-game set. Find each source using the manifest; preserve its original body.

**The existing `compilation.cpuprofile` is warm, not cold.** It sampled five mixed-corpus sweeps after a warmup sweep: 700 compiles, including Cyber-Lasso in every sweep. Its BitVec, GC, and rule-processing hotspots are hypotheses for further investigation, not evidence of cold CPU shares.

## First task on the Deck

Extend the packaged harness for first-compilation CPU profiling before optimizing the compiler. A cold-only mode is **not implemented yet**. The existing `profile.cjs --repeats N` still performs warm work after each cold sample, and `profile.cjs --cpu` deliberately profiles warm sweeps. Do not use a process-wide CPU profile of either command as a cold-only profile.

Implement and verify the following behavior:

1. Run exactly one timed compilation per fresh child process. Read the source, load the engine, install probes, and initialize the profiler without compiling any game first. Run workers serially so measurements do not compete for CPU or memory.
2. Capture coarse phase timings around that first compile. The current parser probes bracket the initial whole tokenizer pass in `loadFile`; `rulesToArray` also parses/expands rule strings, so the phase named `parse` is not every parsing-related operation.
3. For CPU traces, start V8 sampling immediately before that first compile and stop immediately after it. Save a separate profile per game/run. Do no warmup, later compilation, control compilation, state serialization, or gameplay inside that capture window.
4. Make separate fresh-process timing runs without phase probes or CPU sampling to estimate instrumentation overhead. The current control VM runs later in an already-used process: it checks output equivalence and is not a cold timing baseline.
5. Run semantic/control checks after the timed window or in separate processes. Retain diagnostics and inclusion status, especially Cyber-Lasso's two errors. A genuine incomplete compile must remain visible in raw output rather than disappearing from the report.
6. Start with five independent cold runs per verified game. Keep every sample and a recorded run order; investigate noisy outliers with more runs. Save Deck results separately from the Mac data, including commit, source hashes, Node/V8, hardware, and capture settings.

Keep the default lazy-generation policy for the primary compile metric. Add a separate diagnostic measurement of the remaining generation backlog and its work, then check the normal title-to-first-level path in the browser. A forced `tick_lazy_function_generation(false)` drain measures that queued work synchronously; label it accordingly, since the normal title-screen update loop time-slices it.

In browser follow-up, record engine loading/evaluation, source retrieval, compile, title rendering, and first playable-level readiness as separate intervals in a performance trace. Distinguish cold JavaScript/compiler state from HTTP/file-cache state. Compare the same build and browser version; Node VM timings alone cannot establish the browser launch improvement.

## Compare and optimize

Record SteamOS, browser/runtime, AC or battery operation, current power limits, and temperature/thermal behavior. Hold those conditions steady across baseline/candidate runs and avoid other heavy work. Do not change power settings to manufacture a speedup.

Use Deck cold phase timings and per-game cold CPU profiles to choose one change at a time. Parsing, mask construction, rule expansion, and collapsing are starting hypotheses from the Mac cold pass. Warm-only memoization benefits do not satisfy the primary goal. Preserve browser compatibility and game behavior.

For each candidate, alternate baseline/candidate fresh processes under comparable conditions. Report per-game cold medians and distributions, the median and 95th percentile across per-game medians, absolute changes for slow games, and any regressions. Explain whether phase shares pool durations or average game percentages. CPU-profile node IDs are local to a capture; combine profiles by mapped function/file/line rather than concatenating trees or averaging their IDs.

Run `npm test` from the PuzzleScript checkout for functional regressions. Keep compiled-state/diagnostic comparisons on the real corpus. JSON hashes omit function bodies, so code-generation changes also need relevant runtime tests or recorded real-game play; matching JSON alone is insufficient.

The next deliverable should be a Deck cold baseline and profiles, with the worst real games and dominant cold phases identified. Then make a focused optimization, verify correctness, and demonstrate its effect on fresh-game compilation and browser launch behavior. Keep Cyber-Lasso included and the five removed gallery entries removed throughout.
