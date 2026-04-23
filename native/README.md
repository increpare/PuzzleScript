# PuzzleScript C++ Port

The native implementation is exposed as `puzzlescript_cpp`: a C++ PuzzleScript compiler/runtime/player toolkit with a stable C API.

Directory map:

- `native/src/compiler/`: C++ parser/compiler and JS-compatible diagnostics.
- `native/src/runtime/`: abstract engine, sessions, stepping, hashing, snapshots, and JSON IR compatibility loading.
- `native/src/player/`: SDL player integration.
- `native/src/cli/`: `puzzlescript_cpp` command implementation.
- `src/tests/js_oracle/`: JavaScript reference harness used to generate parity data from the original implementation.

Common commands:

```bash
make build
make run src/demo/sokoban_basic.txt
make ctest
make js_parity_tests
make tests
build/native/puzzlescript_cpp --help
build/native/puzzlescript_cpp help test
build/native/puzzlescript_cpp compile src/demo/sokoban_basic.txt --diagnostics
```

`make tests` runs the fast C++ checks and the JS parity suite. The JS parity suite regenerates saved replay/diagnostic data from `testdata.js` and `errormessage_testdata.js`, then checks that the C++ implementation matches it.
