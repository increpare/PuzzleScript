# PuzzleScript Native Scaffold

This directory contains the first native milestone for the PuzzleScript port:

- a `C++` implementation core with a stable plain `C` API
- a versioned JSON IR loader for artifacts exported by the existing JS engine
- immutable `ps_game` plus mutable, cloneable `ps_session`
- canonical `convertLevelToString()`-compatible serialization, hashing, restart, and benchmark hooks
- a native CLI with `run`, `test-fixtures`, `bench`, and `play`
- a native CLI with `run`, `step`, `test-fixtures`, `bench`, `diff-trace`, `diff-trace-source`, and `play`

What is implemented in this milestone:

- JS-side IR/fixture export from the existing compiler and test corpus
- native IR ingestion and prepared-session loading
- session cloning, hash64/hash128, snapshot export, initial-board serialization
- session cloning, hash64/hash128, snapshot export, initial-board serialization
- movement-state groundwork for the native executor, including player input seeding via `step`/`step-source`
- first scalar execution slices for simple movement resolution and simple single-row rule propagation
- optional SDL2 viewer for prepared sessions

What is intentionally not implemented yet:

- native rule execution
- native source compilation
- solver/search algorithms
- SIMD-accelerated hot paths beyond backend detection hooks

Typical workflow:

```bash
make tests
make run src/demo/sokoban_basic.txt
node src/tests/export_ir_json.js src/demo/sokoban_basic.txt /tmp/sokoban.json --level 0 --settle-again
cmake -S . -B build/native
cmake --build build/native
./build/native/ps_cli run /tmp/sokoban.json
./build/native/ps_cli play-source src/demo/sokoban_basic.txt
./build/native/ps_cli step /tmp/sokoban.json right
./build/native/ps_cli diff-trace-source src/demo/sokoban_basic.txt --level 0 --seed smoke --inputs-file native/tests/push_trace_inputs.json
./build/native/ps_cli bench /tmp/sokoban.json --iterations 10000 --threads 4
```

`make tests` builds the native port, regenerates the trace-backed coverage fixtures in `build/native/coverage-fixtures`, and runs the real native parity sweep against them.

`make run path/to/game.txt` builds the native player and launches that PuzzleScript source file directly through the JS-exported IR path.
