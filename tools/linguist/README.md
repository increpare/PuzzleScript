# PuzzleScript Linguist Preparation

This directory contains repository-side materials for getting PuzzleScript recognized by GitHub Linguist.

## Local Repository Setup

The root `.gitattributes` marks generated and vendored paths so GitHub language statistics are less noisy:

- `src/standalone_inlined.txt` and generated build outputs are `linguist-generated`.
- `native/third_party/**` and `node_modules/**` are `linguist-vendored`.
- `*.puzzlescript` is mapped to `linguist-language=PuzzleScript` for future compatibility.

Important: GitHub and Linguist only honor `linguist-language=PuzzleScript` after PuzzleScript is accepted into upstream Linguist. Until then, this mapping is preparation rather than active classification.

## Extension Policy

- Use `.puzzlescript` as the primary upstream extension.
- Do not map all `.txt` files to PuzzleScript; many `.txt` files are prose, fixtures, logs, or generic text.
- Treat `.ps` as a VS Code convenience only for now. It likely conflicts with PostScript in Linguist and should be deferred until there is enough usage evidence and a heuristic plan.

## Proposed `languages.yml` Entry

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

Color rationale: `#f7e26b` is PuzzleScript's default `yellow` from the Arne palette used by the editor, and is visually close to the existing PuzzleScript editor/game aesthetic.

## Grammar Source

The candidate TextMate grammar is:

```text
tools/vscode-puzzlescript/syntaxes/puzzlescript.tmLanguage.json
```

The grammar uses `scopeName: source.puzzlescript` and is intentionally independent of VS Code semantic tokens. The VS Code extension adds richer editor features separately by reusing the browser editor's JavaScript parser and autocomplete logic.

The PuzzleScript repository is MIT licensed. If the grammar is split into a separate upstream grammar repository for `script/add-grammar`, preserve the MIT license and link back to this repository as the source.

## Samples

The sample files in `samples/` are adapted from existing PuzzleScript demo games in this repository:

- `simple-block-pushing.puzzlescript` is adapted from `src/demo/sokoban_basic.txt`.
- `push-directional-excerpt.puzzlescript` is adapted from `src/demo/push.txt`.

These demos are distributed with the MIT-licensed PuzzleScript repository. For an upstream Linguist PR, include this origin/license note and prefer adding more real-world public samples from independent repositories if available.

## Upstream PR Checklist

In a fork of `github-linguist/linguist`:

1. Confirm usage evidence:
   - Search GitHub for `.puzzlescript` files excluding forks.
   - Verify the results are distributed across multiple users/repositories.
   - Collect links for the PR description.
2. Add/import a grammar:
   - Publish or reference a grammar repository containing `source.puzzlescript`.
   - Run `script/add-grammar <grammar-repo-url>`.
3. Add the `PuzzleScript` entry to `lib/linguist/languages.yml`.
4. Copy representative samples into Linguist's `samples/PuzzleScript/`.
5. Run `script/update-ids`.
6. Add or update tests required by Linguist, especially `test/test_blob.rb` if needed.
7. Run:

```sh
script/bootstrap
bundle exec rake test
```

8. Open the PR using Linguist's template, including:
   - GitHub search links proving usage.
   - Sample origins and license statements.
   - Grammar source and license.

## Local Validation

After committing `.gitattributes`, a local Linguist install can be used to inspect this repository:

```sh
github-linguist --breakdown
github-linguist --strategies tools/linguist/samples/simple-block-pushing.puzzlescript
```

Before upstream acceptance, the sample strategy may not report `PuzzleScript` because the language is unknown to Linguist. After upstream acceptance or when testing inside a modified Linguist fork, it should classify as PuzzleScript.
