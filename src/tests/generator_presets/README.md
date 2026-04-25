# Native Generator Presets

These `.gen` files are lightweight Sokoban-oriented presets for the native
generator smoke path. They use the current V1 generation-rule subset and borrow
the spirit of PSMIS transform templates: start from a neutral room, place paired
goal objects, then apply small structural transforms such as wall scatter or
target/wall alternatives.

Run one manually with:

```sh
build/native/puzzlescript_generator src/demo/sokoban_basic.txt src/tests/generator_presets/sokoban_room_scatter.gen --samples 20 --quiet
```
