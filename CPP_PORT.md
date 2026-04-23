# PuzzleScript C++ Port

`puzzlescript_cpp` is the C++ PuzzleScript compiler/runtime/player command.

## What Lives Where

- Core compiler: `native/src/compiler/`
- Abstract engine/runtime: `native/src/runtime/`
- SDL player: `native/src/player/`
- Command-line interface: `native/src/cli/`
- Public C API: `native/include/puzzlescript/`
- JS oracle/reference tests: `src/tests/js_oracle/`
- Native test drivers and scripts: `src/tests/`, `native/tests/`, and `scripts/`

## Everyday Commands

```bash
make build
make run path/to/game.txt
make ctest
make js_parity_tests
make tests
```

Direct command examples:

```bash
build/native/puzzlescript_cpp play path/to/game.txt
build/native/puzzlescript_cpp run path/to/game.txt --headless
build/native/puzzlescript_cpp compile path/to/game.txt --diagnostics
build/native/puzzlescript_cpp compile path/to/game.txt --emit-parser-state
build/native/puzzlescript_cpp bench path/to/game.txt --iterations 10000 --threads 4
```

## Test Commands

- `make ctest`: fast C++ smoke/unit tests registered through CMake.
- `make js_parity_tests`: C++ vs original-JavaScript parity checks using generated JS parity data.
- `make tests`: full native correctness suite, currently `ctest` plus JS parity.

“JS parity data” means saved replay and diagnostic cases generated from the original JavaScript test suite files, especially `testdata.js` and `errormessage_testdata.js`.

For command details, run:

```bash
build/native/puzzlescript_cpp --help
build/native/puzzlescript_cpp help play
build/native/puzzlescript_cpp help compile
build/native/puzzlescript_cpp help test
build/native/puzzlescript_cpp help bench
```
