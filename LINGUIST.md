# Submitting PuzzleScript To GitHub Linguist

This document is the checklist for getting PuzzleScript recognized by GitHub's Linguist project, which is what GitHub uses for repository language stats and syntax highlighting.

## Current Repo Prep

This repository already contains the local materials needed for a future Linguist submission:

- `.gitattributes` marks generated/vendor paths and includes a future-ready `*.puzzlescript linguist-language=PuzzleScript` mapping.
- `tools/vscode-puzzlescript/syntaxes/puzzlescript.tmLanguage.json` is the candidate TextMate grammar with `scopeName: source.puzzlescript`.
- `tools/linguist/samples/*.puzzlescript` contains representative sample files adapted from existing demos.
- `tools/linguist/README.md` records sample origins, license notes, extension policy, and a compact upstream checklist.

Important: GitHub will not treat `PuzzleScript` as a language just because this repository has `linguist-language=PuzzleScript`. Linguist overrides only work for languages already known to Linguist.

## Upstream Gate

Do not open the upstream PR until these are true:

- `.puzzlescript` has enough public GitHub usage evidence.
- Search results are distributed across multiple users/repositories, not mostly one owner.
- The sample files have clear source and license statements.
- The grammar source is available from a stable repository and has an approved license.

Linguist's documented usage requirement for normal extensions is at least 2000 indexed files in the last year, excluding forks, with reasonable distribution across unique repositories. This is the main risk for a PuzzleScript submission.

## Extension Policy

Use this policy for the first upstream PR:

- Submit `.puzzlescript` as the only file extension.
- Do not submit `.txt`; it is far too broad and would cause false positives.
- Do not submit `.ps` in the first PR. `.ps` likely conflicts with PostScript and would require usage evidence plus classification heuristics.
- Keep `.ps` as a VS Code convenience only.

The first PR should optimize for being accepted cleanly. A later PR can add `.ps` if the ecosystem has enough public `.ps` PuzzleScript usage and a solid heuristic.

## Proposed `languages.yml` Entry

Use this as the starting point in Linguist's `lib/linguist/languages.yml`:

```yaml
PuzzleScript:
  type: programming
  color: "#f7e26b"
  extensions:
  - ".puzzlescript"
  tm_scope: source.puzzlescript
  ace_mode: text
  aliases:
  - puzzlescript
  - ps
```

Do not add `language_id` manually. Linguist generates it with `script/update-ids`.

Color rationale: `#f7e26b` is the Arne-palette yellow used by PuzzleScript's editor/game aesthetic. If maintainers ask for community consensus on the color, link to a public discussion in the PuzzleScript repository before pushing the color choice.

## Grammar Source

Linguist imports grammars from a grammar repository using `script/add-grammar`, rather than directly copying a random grammar file into `linguist`.

Recommended path:

1. Create a small public grammar repository, for example `puzzlescript/textmate-puzzlescript`.
2. Put the grammar from `tools/vscode-puzzlescript/syntaxes/puzzlescript.tmLanguage.json` in that repo.
3. Include an MIT `LICENSE`.
4. Include a README that states:
   - grammar name: `PuzzleScript`;
   - scope name: `source.puzzlescript`;
   - source repository: `https://github.com/increpare/PuzzleScript`;
   - license: MIT.
5. Tag a release before running `script/add-grammar` from the Linguist fork.

The VS Code extension's semantic tokens and autocomplete are richer than the TextMate grammar, but GitHub highlighting can only use the TextMate-compatible grammar.

## Samples

Candidate samples are in:

```text
tools/linguist/samples/simple-block-pushing.puzzlescript
tools/linguist/samples/push-directional-excerpt.puzzlescript
```

For the upstream PR:

- Copy them into `samples/PuzzleScript/` in the Linguist fork.
- Prefer adding more real-world public samples from independent repositories.
- For each sample, document:
  - original source URL;
  - whether it was adapted;
  - license.

The included samples are adapted from this MIT-licensed repository:

- `simple-block-pushing.puzzlescript`: adapted from `src/demo/sokoban_basic.txt`.
- `push-directional-excerpt.puzzlescript`: adapted from `src/demo/push.txt`.

## Collecting Usage Evidence

Use GitHub search links in the PR description. Suggested searches:

```text
extension:puzzlescript fork:false
filename:*.puzzlescript fork:false
"objects" "collisionlayers" "winconditions" extension:puzzlescript fork:false
```

Manually click through results and record a few representative repositories. If many results are from this repository or one prolific user, add exclusions such as:

```text
extension:puzzlescript fork:false -user:increpare
```

If the search count is below Linguist's requirement, pause. Opening a PR too early is likely to burn maintainer time and get closed.

## Linguist Fork Workflow

In your Linguist fork:

```sh
git clone https://github.com/<your-user>/linguist.git
cd linguist
script/bootstrap
```

Create a branch:

```sh
git checkout -b add-puzzlescript
```

Add the grammar:

```sh
script/add-grammar https://github.com/<grammar-owner>/<grammar-repo>
```

Edit `lib/linguist/languages.yml` and add the `PuzzleScript` entry alphabetically.

Copy sample files:

```sh
mkdir -p samples/PuzzleScript
cp /path/to/PuzzleScript/tools/linguist/samples/*.puzzlescript samples/PuzzleScript/
```

Generate the language ID:

```sh
script/update-ids
```

Run tests:

```sh
bundle exec rake test
```

Optional classifier test:

```sh
bundle exec script/cross-validation --test
```

## Local Sanity Checks

From the PuzzleScript repository:

```sh
npm run test:syntax
npm run preview:syntax -- ../linguist/samples/simple-block-pushing.puzzlescript /tmp/puzzlescript-highlight.html
```

From the Linguist fork after adding PuzzleScript:

```sh
bundle exec bin/github-linguist --breakdown /path/to/PuzzleScript
bundle exec bin/github-linguist --strategies /path/to/PuzzleScript/tools/linguist/samples/simple-block-pushing.puzzlescript
```

Expected result after upstream-style changes: `.puzzlescript` samples classify as PuzzleScript. Existing `.txt` demos should not be reclassified.

## Pull Request Body Notes

Use Linguist's PR template. Include this information:

```markdown
## Description

Adds PuzzleScript as a programming language with `.puzzlescript` files and TextMate scope `source.puzzlescript`.

## Usage Evidence

- GitHub search: <link>
- Representative repositories:
  - <repo link>
  - <repo link>
  - <repo link>

## Grammar

- Grammar source: <grammar repo link>
- Scope: `source.puzzlescript`
- License: MIT

## Samples

- `simple-block-pushing.puzzlescript`
  - Adapted from `src/demo/sokoban_basic.txt` in `increpare/PuzzleScript`
  - License: MIT
- `push-directional-excerpt.puzzlescript`
  - Adapted from `src/demo/push.txt` in `increpare/PuzzleScript`
  - License: MIT

## Notes

This PR intentionally adds only `.puzzlescript`. It does not add `.ps` because that extension conflicts with existing PostScript usage and would need separate heuristics/evidence.
```

## Reference Links

- Linguist contributing guide: https://github.com/github-linguist/linguist/blob/main/CONTRIBUTING.md
- Linguist overrides docs: https://github.com/github-linguist/linguist/blob/main/docs/overrides.md
- GitHub repository language docs: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-repository-languages
