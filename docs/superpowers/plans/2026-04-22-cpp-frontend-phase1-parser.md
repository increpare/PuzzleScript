# C++ Frontend Phase 1: Parser + Validation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `src/js/parser.js` (~1,667 LOC) to C++ as the first of three phases of the PuzzleScript frontend, producing a `ParserState` object byte-for-byte equivalent to the JS parser output across the full fixture corpus (~730 fixtures).

**Architecture:** A new static library `puzzlescript-frontend` (C++17) exposes a C API (`ps_frontend_*`). Internally, a `parse(source) -> ParserState + Diagnostic[]` function mirrors the JS `codeMirrorFn().startState()` + `token(stream, state)` state machine line-by-line. Differential testing against instrumented JS (`export_ir_json.js --snapshot-phase parser`) is the gate: every task lands with the parity gate green for whatever corpus subset it targets.

**Tech Stack:** C++17, CMake 3.20+, existing repo infra. The JS side remains unchanged except for an additive `--snapshot-phase` flag on `export_ir_json.js`. No new C++ third-party dependencies.

**Parent spec:** `docs/superpowers/specs/2026-04-22-cpp-frontend-design.md` — consult §2–§5 for the architecture and validation harness contract this plan implements.

---

## Overview

This plan ports the PuzzleScript parser from JS to C++. The JS parser is a CodeMirror-style line-by-line tokenizer with a mutable `state` object and section-dispatched handlers (`OBJECTS`, `LEGEND`, `SOUNDS`, `COLLISIONLAYERS`, `RULES`, `WINCONDITIONS`, `LEVELS`, and the preamble before any section). We port each section incrementally, gated by a differential-testing harness that compares C++ output against JS output on the exact same source.

**Key insight:** the JS `state` object at "parse complete" is the authoritative `ParserState` shape. Serializing both to JSON and byte-diffing them is our correctness oracle. The C++ port is "done" when the full corpus diff is empty.

**Bug-for-bug compatibility is the spec.** If JS does something weird (off-by-one, redundant trailing whitespace in an error message, quirky regex behavior), C++ matches it. Never "fix" JS during the port. File a follow-up issue instead.

**Execution order matters:** Tasks 1-13 build infrastructure (library target, C API, diagnostic types, harness, ParserState struct, comment/line/section handling). Only after that do we port section-specific logic (Tasks 14-42). The final gate (Task 43-45) runs the full corpus.

## File Structure

New files created by this plan:

| Path | Responsibility |
|------|----------------|
| `native/src/frontend/CMakeLists.txt` | Build config for `puzzlescript-frontend` static lib (optional — could inline into `native/CMakeLists.txt` instead) |
| `native/src/frontend/diagnostic.hpp` | `Severity`, `DiagnosticCode` enum, `DiagnosticArgs` variant, `Diagnostic` struct, `DiagnosticSink` |
| `native/src/frontend/diagnostic.cpp` | `format_for_js_compat(const Diagnostic&)` — produces JS-identical text |
| `native/src/frontend/language_constants.hpp` | Keyword arrays, regex patterns, color name table (ported from `src/js/languageConstants.js`) |
| `native/src/frontend/language_constants.cpp` | Table definitions |
| `native/src/frontend/types/parser_state.hpp` | `ParserState` struct (mirror of JS `state` object at parse-complete) |
| `native/src/frontend/parser.hpp` | Public `parse(source) -> std::pair<ParserState, std::vector<Diagnostic>>` (internal C++ entrypoint) |
| `native/src/frontend/parser.cpp` | State-machine implementation; internal section-handler files dispatched from here |
| `native/src/frontend/parser_preamble.cpp` | Preamble/metadata section handler |
| `native/src/frontend/parser_objects.cpp` | OBJECTS section handler |
| `native/src/frontend/parser_legend.cpp` | LEGEND section handler |
| `native/src/frontend/parser_sounds.cpp` | SOUNDS section handler |
| `native/src/frontend/parser_collisionlayers.cpp` | COLLISIONLAYERS section handler |
| `native/src/frontend/parser_rules.cpp` | RULES section handler |
| `native/src/frontend/parser_winconditions.cpp` | WINCONDITIONS section handler |
| `native/src/frontend/parser_levels.cpp` | LEVELS section handler |
| `native/src/frontend/parser_state_serialize.cpp` | Dev-only: `ParserState → parser_state.json` canonical JSON |
| `native/include/puzzlescript/frontend.h` | Public C API (`ps_frontend_*`) |
| `native/src/c_api_frontend.cpp` | C API implementation bridging to C++ parser |
| `src/tests/lib/puzzlescript_parser_snapshot.js` | JS-side: serialize `state` object to canonical `parser_state.json` |
| `scripts/diff_parser_state_against_js.sh` | Single-fixture and corpus-level diff runner |
| `scripts/corpus_list.sh` | Enumerate all fixtures in `src/tests/resources/` for corpus runs |
| `native/tests/frontend/` | C++ unit tests for small helpers (canonicalizer, diagnostic formatter) |

Modified files:

| Path | Change |
|------|--------|
| `native/CMakeLists.txt` | Add `puzzlescript-frontend` target; add dev-only `compile-source` test; extend `ps_cli` to link frontend |
| `native/src/cli_main.cpp` | Add `compile-source --emit-parser-state <source.txt>` dev subcommand |
| `src/tests/export_ir_json.js` | Add `--snapshot-phase parser` flag emitting `parser_state.json` |

---

## Task Index (47 tasks)

Infrastructure (1-5) → Harness (6-10) → ParserState skeleton (11-13) → Preamble (14-17) → Section routing (18-19) → OBJECTS (20-23) → LEGEND (24-26) → SOUNDS (27-28) → COLLISIONLAYERS (29-30) → RULES (31-37) → WINCONDITIONS (38-39) → LEVELS (40-42) → Gates (43-44) → C API (44b) → CI + cleanup (45-46).

**Commit cadence:** every task ends with a commit. Tasks with multiple natural checkpoints may commit multiple times (marked per task). Never leave half-finished work uncommitted across task boundaries.

---

## Task 1: CMake target and empty library skeleton

**Files:**
- Create: `native/src/frontend/placeholder.cpp`
- Modify: `native/CMakeLists.txt`

- [ ] **Step 1: Write an empty placeholder.cpp**

```cpp
// native/src/frontend/placeholder.cpp
// Temporary TU so the library links. Deleted in Task 2.
namespace puzzlescript::frontend { inline void _placeholder() {} }
```

- [ ] **Step 2: Add library target in native/CMakeLists.txt**

Insert after the `puzzlescript_native` target definition (around line 80, before `add_executable(ps_cli ...)`):

```cmake
set(PUZZLESCRIPT_FRONTEND_SOURCES
  src/frontend/placeholder.cpp
)

add_library(puzzlescript_frontend STATIC
  ${PUZZLESCRIPT_FRONTEND_SOURCES}
)

target_include_directories(puzzlescript_frontend
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

option(PS_ENABLE_DEV_SERIALIZERS "Enable dev-only JSON serializers for frontend phase diffing" OFF)
target_compile_definitions(puzzlescript_frontend
  PRIVATE
    $<$<BOOL:${PS_ENABLE_DEV_SERIALIZERS}>:PS_ENABLE_DEV_SERIALIZERS=1>
)
```

Link into `ps_cli`:

```cmake
target_link_libraries(ps_cli
  PRIVATE
    puzzlescript_native
    puzzlescript_frontend
)
```

- [ ] **Step 3: Verify it builds**

Run: `cmake -B build && cmake --build build --target puzzlescript_frontend ps_cli`

Expected: success, no warnings on the new target.

- [ ] **Step 4: Commit**

```bash
git add native/CMakeLists.txt native/src/frontend/placeholder.cpp
git commit -m "build: add puzzlescript_frontend library target (empty skeleton)"
```

---

## Task 2: C API public header skeleton

**Files:**
- Create: `native/include/puzzlescript/frontend.h`
- Modify: `native/CMakeLists.txt` (add header to `PUZZLESCRIPT_FRONTEND_PUBLIC_HEADERS`)

- [ ] **Step 1: Write the public header**

```c
// native/include/puzzlescript/frontend.h
#ifndef PUZZLESCRIPT_FRONTEND_H
#define PUZZLESCRIPT_FRONTEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ps_frontend_result ps_frontend_result;

typedef enum ps_diagnostic_severity {
    PS_DIAG_ERROR = 0,
    PS_DIAG_WARNING = 1,
    PS_DIAG_INFO = 2,
    PS_DIAG_LOG = 3
} ps_diagnostic_severity;

typedef struct ps_diagnostic {
    ps_diagnostic_severity severity;
    int32_t code;
    int32_t line;           /* 1-based; -1 if absent */
    const char* message;    /* NUL-terminated, UTF-8, JS-compatible text */
} ps_diagnostic;

/* Parse only. Returns a result owning the ParserState and diagnostics.
 * On fatal parse errors the result still contains diagnostics; other accessors
 * may return sentinel/empty values. Always non-null (or OOM abort). */
ps_frontend_result* ps_frontend_parse(const char* source, size_t source_len);

size_t ps_frontend_result_diagnostic_count(const ps_frontend_result*);
const ps_diagnostic* ps_frontend_result_diagnostic(const ps_frontend_result*, size_t index);

/* Dev-only: emit canonical ParserState JSON to a caller-owned buffer. */
size_t ps_frontend_result_parser_state_json(
    const ps_frontend_result* result,
    char* out_buffer,
    size_t out_buffer_capacity
);

void ps_frontend_result_free(ps_frontend_result*);

#ifdef __cplusplus
}
#endif

#endif  /* PUZZLESCRIPT_FRONTEND_H */
```

- [ ] **Step 2: Add header to CMake install set**

In `native/CMakeLists.txt`, add after `PUZZLESCRIPT_FRONTEND_SOURCES`:

```cmake
set(PUZZLESCRIPT_FRONTEND_PUBLIC_HEADERS
  include/puzzlescript/frontend.h
)
add_library(puzzlescript_frontend STATIC
  ${PUZZLESCRIPT_FRONTEND_PUBLIC_HEADERS}
  ${PUZZLESCRIPT_FRONTEND_SOURCES}
)
```

- [ ] **Step 3: Verify C++ code that includes the header compiles as both C and C++**

Add a tiny check in `native/src/frontend/placeholder.cpp`:

```cpp
#include "puzzlescript/frontend.h"
namespace puzzlescript::frontend { inline void _placeholder() {} }
```

Run: `cmake --build build --target puzzlescript_frontend`

Expected: success, no warnings.

- [ ] **Step 4: Commit**

```bash
git add native/include/puzzlescript/frontend.h native/src/frontend/placeholder.cpp native/CMakeLists.txt
git commit -m "frontend: add public C API header skeleton"
```

---

## Task 3: Diagnostic types (Severity, DiagnosticCode, DiagnosticArgs, DiagnosticSink)

**Files:**
- Create: `native/src/frontend/diagnostic.hpp`
- Create: `native/src/frontend/diagnostic.cpp`
- Modify: `native/CMakeLists.txt` (add diagnostic.cpp to sources)

- [ ] **Step 1: Define diagnostic.hpp**

```cpp
// native/src/frontend/diagnostic.hpp
#pragma once
#include <optional>
#include <string>
#include <variant>
#include <vector>
#include <cstdint>

namespace puzzlescript::frontend {

enum class Severity : int { Error = 0, Warning = 1, Info = 2, LogMessage = 3 };

// DiagnosticCode: a stable enum over all distinct error/warning/info strings
// emitted by the JS parser+compiler. We grow this incrementally: when porting
// a JS logError/logWarning call site, add a new enum value and a matching
// case in format_for_js_compat.
enum class DiagnosticCode : int {
    // Sentinel / misc
    UnknownError = 0,
    // Parser-phase codes grow here. See diagnostic.cpp for the formatter.
    ObjectDefinedMultipleTimes,
    ObjectNameAlreadyInUse,
    NamedObjectIsKeyword,
    SpriteMustBe5By5,
    SpriteGraphicsMustBe5By5,
    UnknownJunkInObjectsSection,
    LegendIncorrectFormat,
    LegendDanglingWord,
    LegendCannotDefineSelf,
    LegendRepeatedRhs,
    LegendAggregateFromProperty,
    LegendPropertyFromAggregate,
    LegendIncorrectFormatGeneric,
    NameInUseFromLegend,
    WordNotDefined,
    UnexpectedSoundToken,
    UnrecognizedSectionHeader,
    MustStartWithObjects,
    UnrecognisedStuffInPrelude,
    GameTitleTooLong,
    MessageMissingSpace,
    RuleNoArrow,
    // ... many more added as needed per task.
};

// Structured args for diagnostics. The formatter inspects these to produce
// JS-compatible text. We use a tagged-union (variant) to avoid a bag-of-strings.
using DiagnosticArg = std::variant<std::string, int64_t, double>;
using DiagnosticArgs = std::vector<DiagnosticArg>;

struct Diagnostic {
    Severity severity{Severity::Error};
    DiagnosticCode code{DiagnosticCode::UnknownError};
    std::optional<int> line{};  // 1-based, matching JS
    DiagnosticArgs args{};
};

// Simple in-process sink; caps at MAX_ERRORS_FOR_REAL (100) to match JS behavior.
class DiagnosticSink {
public:
    static constexpr size_t MAX_ERRORS_FOR_REAL = 100;

    void push(Diagnostic d);
    const std::vector<Diagnostic>& diagnostics() const { return diagnostics_; }
    int error_count() const { return error_count_; }
    bool too_many() const { return too_many_; }

private:
    std::vector<Diagnostic> diagnostics_;
    int error_count_{0};
    bool too_many_{false};
};

// Formats a diagnostic into a string matching the JS compiler's output
// after stripHTMLTags + whitespace normalization.
std::string format_for_js_compat(const Diagnostic& d);

}  // namespace puzzlescript::frontend
```

- [ ] **Step 2: Stub diagnostic.cpp with push() and empty formatter**

```cpp
// native/src/frontend/diagnostic.cpp
#include "diagnostic.hpp"
#include <string>

namespace puzzlescript::frontend {

void DiagnosticSink::push(Diagnostic d) {
    if (too_many_) return;
    if (d.severity == Severity::Error) ++error_count_;
    diagnostics_.push_back(std::move(d));
    if (diagnostics_.size() > MAX_ERRORS_FOR_REAL) {
        too_many_ = true;
        Diagnostic abort;
        abort.severity = Severity::Error;
        abort.code = DiagnosticCode::UnknownError;
        abort.args = {std::string("Too many errors/warnings; aborting compilation.")};
        diagnostics_.push_back(std::move(abort));
    }
}

// Formatter grows one case at a time as each diagnostic code is first used.
// For now, UnknownError just returns the first string arg if any.
std::string format_for_js_compat(const Diagnostic& d) {
    switch (d.code) {
        case DiagnosticCode::UnknownError:
            if (!d.args.empty() && std::holds_alternative<std::string>(d.args[0])) {
                return std::get<std::string>(d.args[0]);
            }
            return {};
        default:
            return {};  // stubbed; grows per-task
    }
}

}  // namespace puzzlescript::frontend
```

- [ ] **Step 3: Add diagnostic.cpp to PUZZLESCRIPT_FRONTEND_SOURCES**

In `native/CMakeLists.txt`:

```cmake
set(PUZZLESCRIPT_FRONTEND_SOURCES
  src/frontend/diagnostic.cpp
  src/frontend/placeholder.cpp
)
```

- [ ] **Step 4: Write a failing unit test**

Create `native/tests/frontend/test_diagnostic_sink.cpp`:

```cpp
#include "frontend/diagnostic.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace puzzlescript::frontend;
    DiagnosticSink sink;
    Diagnostic d;
    d.severity = Severity::Error;
    d.code = DiagnosticCode::UnknownError;
    d.args = {std::string("hello")};
    sink.push(d);
    assert(sink.error_count() == 1);
    assert(sink.diagnostics().size() == 1);
    assert(format_for_js_compat(sink.diagnostics()[0]) == "hello");

    // Cap test
    for (int i = 0; i < 200; ++i) sink.push(d);
    assert(sink.too_many());
    std::cout << "OK\n";
    return 0;
}
```

Add to `native/CMakeLists.txt` below the other `add_test` blocks:

```cmake
add_executable(test_diagnostic_sink tests/frontend/test_diagnostic_sink.cpp)
target_link_libraries(test_diagnostic_sink PRIVATE puzzlescript_frontend)
target_include_directories(test_diagnostic_sink PRIVATE src)
add_test(NAME frontend_diagnostic_sink COMMAND test_diagnostic_sink)
```

- [ ] **Step 5: Run the test**

```
cmake --build build --target test_diagnostic_sink && ctest --test-dir build -R frontend_diagnostic_sink --output-on-failure
```

Expected: PASS "OK".

- [ ] **Step 6: Commit**

```bash
git add native/src/frontend/diagnostic.hpp native/src/frontend/diagnostic.cpp \
        native/tests/frontend/test_diagnostic_sink.cpp native/CMakeLists.txt
git commit -m "frontend: diagnostic types + sink with MAX_ERRORS cap"
```

---

## Task 4: Language constants port

**Files:**
- Create: `native/src/frontend/language_constants.hpp`
- Create: `native/src/frontend/language_constants.cpp`
- Reference: `src/js/languageConstants.js` (48 LOC), `src/js/parser.js` top (keyword_array etc.)

- [ ] **Step 1: Write header with constant tables**

```cpp
// native/src/frontend/language_constants.hpp
#pragma once
#include <array>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace puzzlescript::frontend::language {

// Mirror of src/js/languageConstants.js. Keep names and contents identical.
extern const std::array<std::string_view, 9>   relativedirs;
extern const std::array<std::string_view, 7>   sectionNames;
extern const std::array<std::string_view, 17>  commandwords;
extern const std::array<std::string_view, 11>  commandwords_sfx;
extern const std::array<std::string_view, 6>   soundverbs_directional;
extern const std::array<std::string_view, 6>   relativeDirections;
extern const std::array<std::string_view, 4>   simpleAbsoluteDirections;
extern const std::array<std::string_view, 4>   simpleRelativeDirections;

// Flat keyword list (JS keyword_array). Searched by linear scan in JS; a
// std::unordered_set is fine here — semantics are "is x a keyword?".
bool is_keyword(std::string_view word);

// Named colors: returns canonical hex if recognized, empty if not.
// Names and hexes from src/js/colors.js — port the same table.
std::string_view color_name_to_hex(std::string_view name);

// directionaggregates map: 'horizontal' -> {'left','right'}, etc.
const std::vector<std::string_view>* direction_aggregate(std::string_view name);

}  // namespace puzzlescript::frontend::language
```

- [ ] **Step 2: Write cpp with exact JS content**

```cpp
// native/src/frontend/language_constants.cpp
#include "language_constants.hpp"
#include <unordered_set>
#include <unordered_map>

namespace puzzlescript::frontend::language {

const std::array<std::string_view, 9> relativedirs = {
    "^", "v", "<", ">", "moving", "stationary", "parallel", "perpendicular", "no"
};

const std::array<std::string_view, 7> sectionNames = {
    "objects", "legend", "sounds", "collisionlayers", "rules", "winconditions", "levels"
};

const std::array<std::string_view, 17> commandwords = {
    "sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7","sfx8","sfx9","sfx10",
    "cancel","checkpoint","restart","win","message","again"
};

const std::array<std::string_view, 11> commandwords_sfx = {
    "sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7","sfx8","sfx9","sfx10"
};

const std::array<std::string_view, 6> soundverbs_directional = {
    "move", "cantmove", "", "", "", ""  // placeholder - real list: move, cantmove
};
// NOTE: JS soundverbs_directional is ['move','cantmove']; keeping explicit len.
// Trim when wiring up sounds in Task 27.

const std::array<std::string_view, 6> relativeDirections = {
    "^","v","<",">","perpendicular","parallel"
};

const std::array<std::string_view, 4> simpleAbsoluteDirections = {
    "up","down","left","right"
};

const std::array<std::string_view, 4> simpleRelativeDirections = {
    "^","v","<",">"
};

// keyword_array verbatim from parser.js line 23.
static const std::unordered_set<std::string_view> kKeywords = {
    "objects","collisionlayers","legend","sounds","rules","...","winconditions","levels",
    "|","[","]","up","down","left","right","late","rigid","^","v",">","<","no",
    "randomdir","random","horizontal","vertical","any","all","no","some","moving",
    "stationary","parallel","perpendicular","action","message","move","action","create",
    "destroy","cantmove","sfx0","sfx1","sfx2","sfx3","sfx4","sfx5","sfx6","sfx7",
    "sfx8","sfx9","sfx10","cancel","restart","win","message","again","undo","restart",
    "titlescreen","startgame","cancel","endgame","startlevel","endlevel",
    "showmessage","closemessage"
};

bool is_keyword(std::string_view word) { return kKeywords.count(word) != 0; }

// Color table ported from src/js/colors.js colorPalette / colorPalettesAlphabetical.
// Only the first palette (master_palette) is needed for parser.
// FILL THIS IN from src/js/colors.js when first needed (Task 22 - OBJECTS color line).
std::string_view color_name_to_hex(std::string_view name) {
    static const std::unordered_map<std::string_view, std::string_view> kColors = {
        // TODO (Task 22): populate from src/js/colors.js
    };
    auto it = kColors.find(name);
    if (it == kColors.end()) return {};
    return it->second;
}

static const std::unordered_map<std::string_view, std::vector<std::string_view>> kAggregates = {
    {"horizontal",      {"left", "right"}},
    {"horizontal_par",  {"left", "right"}},
    {"horizontal_perp", {"left", "right"}},
    {"vertical",        {"up", "down"}},
    {"vertical_par",    {"up", "down"}},
    {"vertical_perp",   {"up", "down"}},
    {"moving",          {"up", "down", "left", "right", "action"}},
    {"orthogonal",      {"up", "down", "left", "right"}},
    {"perpendicular",   {"^", "v"}},
    {"parallel",        {"<", ">"}}
};

const std::vector<std::string_view>* direction_aggregate(std::string_view name) {
    auto it = kAggregates.find(name);
    if (it == kAggregates.end()) return nullptr;
    return &it->second;
}

}  // namespace puzzlescript::frontend::language
```

- [ ] **Step 3: Fix the soundverbs_directional length (real list is ['move','cantmove'])**

Change the declaration to `std::array<std::string_view, 2>` in both .hpp and .cpp and set contents to `{"move","cantmove"}`.

- [ ] **Step 4: Add language_constants.cpp to CMake**

In `native/CMakeLists.txt`, update:

```cmake
set(PUZZLESCRIPT_FRONTEND_SOURCES
  src/frontend/diagnostic.cpp
  src/frontend/language_constants.cpp
  src/frontend/placeholder.cpp
)
```

- [ ] **Step 5: Write a unit test covering the keyword / aggregate / color lookups**

Create `native/tests/frontend/test_language_constants.cpp`:

```cpp
#include "frontend/language_constants.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace puzzlescript::frontend::language;
    assert(is_keyword("up"));
    assert(is_keyword("rigid"));
    assert(!is_keyword("player"));
    assert(direction_aggregate("horizontal") != nullptr);
    assert(direction_aggregate("nosuchthing") == nullptr);
    assert(direction_aggregate("moving")->size() == 5);
    // color_name_to_hex is populated in Task 22.
    std::cout << "OK\n";
    return 0;
}
```

Add to `native/CMakeLists.txt`:

```cmake
add_executable(test_language_constants tests/frontend/test_language_constants.cpp)
target_link_libraries(test_language_constants PRIVATE puzzlescript_frontend)
target_include_directories(test_language_constants PRIVATE src)
add_test(NAME frontend_language_constants COMMAND test_language_constants)
```

- [ ] **Step 6: Build and run**

```
cmake --build build --target test_language_constants && ctest --test-dir build -R frontend_language_constants --output-on-failure
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add native/src/frontend/language_constants.hpp native/src/frontend/language_constants.cpp \
        native/tests/frontend/test_language_constants.cpp native/CMakeLists.txt
git commit -m "frontend: port language_constants.js (keywords, aggregates, color stubs)"
```

---

## Task 5: Empty ParserState struct (scaffolding only, no behavior)

**Files:**
- Create: `native/src/frontend/types/parser_state.hpp`
- Reference: `src/js/parser.js` line 1606-1664 (the `startState` function)

- [ ] **Step 1: Mirror the JS state object shape in C++**

```cpp
// native/src/frontend/types/parser_state.hpp
#pragma once
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace puzzlescript::frontend {

// Shapes mirror the JS `state` object in src/js/parser.js startState() line 1606+.
// Field names match JS keys exactly — we preserve case and underscores so the
// JSON serializer produces identical keys to the JS snapshot.

// Object sprite definition: a name, 0-9 colors, and a 5x5 matrix of color indices.
// Some objects have no sprite (just a color). Preserve exactly what JS produces.
struct ObjectDef {
    std::string name;                 // the candname as-stored (lowercased)
    int lineNumber{0};
    std::vector<std::string> colors;  // canonical names/hex strings as emitted by JS
    std::vector<std::vector<std::string>> spritematrix;  // rows of per-cell color indices or "." for transparent
    int spritematrix_rowcount{0};
    // 'layer' is set later (in compiler phase); parser leaves it absent.
};

// Legend entries carry an extra `lineNumber` that JS attaches as a non-index property.
struct LegendEntry {
    std::vector<std::string> tokens;  // first token is the key, remainder per JS
    int lineNumber{0};
};

struct SoundRow {
    std::vector<std::string> tokens;  // trailing lineNumber appended as JS does
    int lineNumber{0};
};

struct MetadataPair {
    std::string key;
    std::string value;
};

// Levels are arrays of strings (raw grid rows) OR message entries. JS uses a
// bare array of rows for a grid level and a {message: ...} object for a message.
struct LevelDat {
    bool is_message{false};
    std::string message;           // only when is_message
    int lineNumber{0};              // only when !is_message
    std::vector<std::string> rows;  // one string per row, raw characters from source
};

// Win condition row - raw tokens as JS pushes onto state.winconditions.
struct WinCondition {
    std::vector<std::string> tokens;
    int lineNumber{0};
};

// Collision layer - flat list of object names.
struct CollisionLayer {
    std::vector<std::string> names;
    int lineNumber{0};
};

// Rule - stored as raw token stream until the lowerer. JS state.rules is an array of
// {lhs, rhs, directions, lineNumber, ...} objects. Mirror whichever shape JS emits.
// Full shape is filled in during Task 31 when porting the rules parser.
struct RawRule {
    int lineNumber{0};
    // Opaque token list matching the shape JS produces in processRuleString.
    // See parser.js line ~750+ for the precise structure.
    // Filled in during Task 31.
};

struct ParserState {
    // Mirror startState() field-for-field.

    // Objects by name (lowercased key -> ObjectDef).
    // JS: `objects: {}`.
    std::map<std::string, ObjectDef> objects;  // std::map so JSON keys are sorted

    int lineNumber{0};
    int commentLevel{0};
    std::string section{};               // "", "objects", "legend", etc.
    std::vector<std::string> visitedSections{};

    bool line_should_end{false};
    std::string line_should_end_because{};
    bool sol_after_comment{false};

    bool inside_cell{false};
    int bracket_balance{0};
    bool arrow_passed{false};
    bool rule_prelude{true};

    std::string objects_candname{};
    int objects_section{0};
    std::vector<std::vector<std::string>> objects_spritematrix{};

    std::vector<CollisionLayer> collisionLayers{};

    int tokenIndex{0};
    std::vector<std::string> current_line_wip_array{};

    std::vector<LegendEntry> legend_synonyms{};
    std::vector<LegendEntry> legend_aggregates{};
    std::vector<LegendEntry> legend_properties{};

    std::vector<SoundRow> sounds{};
    std::vector<RawRule> rules{};

    std::vector<std::string> names{};

    std::vector<WinCondition> winconditions{};
    std::vector<MetadataPair> metadata{};
    std::map<std::string, int> metadata_lines{};  // JS: metadata_lines

    std::map<std::string, std::string> original_case_names{};
    std::map<std::string, int> original_line_numbers{};

    std::vector<std::string> abbrevNames{};

    std::vector<LevelDat> levels{};  // starts with one empty LevelDat in JS (levels: [[]])

    std::string subsection{};
};

}  // namespace puzzlescript::frontend
```

- [ ] **Step 2: Write a trivial unit test that default-constructs the struct**

Create `native/tests/frontend/test_parser_state_default.cpp`:

```cpp
#include "frontend/types/parser_state.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace puzzlescript::frontend;
    ParserState s;
    assert(s.lineNumber == 0);
    assert(s.section.empty());
    assert(s.rule_prelude == true);
    assert(s.levels.empty());  // JS inits to [[]]; we fill that in Task 13.
    std::cout << "OK\n";
    return 0;
}
```

Add to CMake:

```cmake
add_executable(test_parser_state_default tests/frontend/test_parser_state_default.cpp)
target_link_libraries(test_parser_state_default PRIVATE puzzlescript_frontend)
target_include_directories(test_parser_state_default PRIVATE src)
add_test(NAME frontend_parser_state_default COMMAND test_parser_state_default)
```

- [ ] **Step 3: Build and run**

```
cmake --build build --target test_parser_state_default && ctest --test-dir build -R frontend_parser_state_default --output-on-failure
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/types/parser_state.hpp native/tests/frontend/test_parser_state_default.cpp native/CMakeLists.txt
git commit -m "frontend: ParserState struct skeleton mirroring JS state shape"
```

---

## Task 6: JS snapshot instrumentation — `--snapshot-phase parser`

**Files:**
- Modify: `src/tests/export_ir_json.js`
- Create: `src/tests/lib/puzzlescript_parser_snapshot.js`
- Reference: existing `src/tests/lib/puzzlescript_ir.js` for the canonicalization pattern

- [ ] **Step 1: Write the snapshot serializer**

Create `src/tests/lib/puzzlescript_parser_snapshot.js`:

```javascript
'use strict';

// Serializes the JS parser `state` object to a canonical JSON shape that
// matches the C++ ParserState JSON emitted by the frontend.
// Used by export_ir_json.js --snapshot-phase parser.

function canonicalizeObjects(objects) {
    const keys = Object.keys(objects).sort();
    const out = {};
    for (const key of keys) {
        const o = objects[key];
        out[key] = {
            name: o.name,
            lineNumber: o.lineNumber,
            colors: (o.colors || []).slice(),
            spritematrix: (o.spritematrix || []).map(row => (typeof row === 'string') ? row.split('') : row.slice()),
            spritematrix_rowcount: o.spritematrix_rowcount || 0,
        };
    }
    return out;
}

function legendEntryToArray(entry) {
    return {
        tokens: entry.slice(),  // JS stores legend entries as arrays with a .lineNumber extra prop
        lineNumber: entry.lineNumber || 0,
    };
}

function soundRowToObject(row) {
    // JS appends lineNumber as the last array element. Peel it off.
    const copy = row.slice();
    const ln = (typeof copy[copy.length - 1] === 'number') ? copy.pop() : 0;
    return { tokens: copy, lineNumber: ln };
}

function winConditionToObject(row) {
    const copy = row.slice();
    const ln = row.lineNumber || 0;
    return { tokens: copy, lineNumber: ln };
}

function collisionLayerToObject(row) {
    const copy = row.slice();
    const ln = row.lineNumber || 0;
    return { names: copy, lineNumber: ln };
}

function levelsToArray(levels) {
    // JS: levels is an array where each entry is either an array of row strings
    // (grid level) or an object with a .message property.
    return (levels || []).map(l => {
        if (l && typeof l === 'object' && 'message' in l) {
            return { is_message: true, message: l.message, lineNumber: 0, rows: [] };
        }
        return {
            is_message: false,
            message: '',
            lineNumber: l.lineNumber || 0,
            rows: (l || []).slice(),
        };
    });
}

function metadataPairs(state) {
    const pairs = [];
    for (let i = 0; i < (state.metadata || []).length; i += 2) {
        pairs.push({ key: state.metadata[i], value: state.metadata[i + 1] || '' });
    }
    return pairs;
}

function rulesToArray(rules) {
    // Raw parser rules are an array of {lineNumber, ...} objects. For parity,
    // serialize everything JS stores. Sort object keys to keep output stable.
    return (rules || []).map(r => {
        // Generic deep-copy with sorted keys.
        return JSON.parse(JSON.stringify(r, Object.keys(r).sort()));
    });
}

function buildParserStateSnapshot(state) {
    return {
        objects: canonicalizeObjects(state.objects),
        lineNumber: state.lineNumber,
        commentLevel: state.commentLevel,
        section: state.section,
        visitedSections: (state.visitedSections || []).slice(),
        line_should_end: !!state.line_should_end,
        line_should_end_because: state.line_should_end_because || '',
        sol_after_comment: !!state.sol_after_comment,
        inside_cell: !!state.inside_cell,
        bracket_balance: state.bracket_balance || 0,
        arrow_passed: !!state.arrow_passed,
        rule_prelude: !!state.rule_prelude,
        objects_candname: state.objects_candname || '',
        objects_section: state.objects_section || 0,
        objects_spritematrix: (state.objects_spritematrix || []).slice(),
        collisionLayers: (state.collisionLayers || []).map(collisionLayerToObject),
        tokenIndex: state.tokenIndex || 0,
        current_line_wip_array: (state.current_line_wip_array || []).slice(),
        legend_synonyms: (state.legend_synonyms || []).map(legendEntryToArray),
        legend_aggregates: (state.legend_aggregates || []).map(legendEntryToArray),
        legend_properties: (state.legend_properties || []).map(legendEntryToArray),
        sounds: (state.sounds || []).map(soundRowToObject),
        rules: rulesToArray(state.rules),
        names: (state.names || []).slice(),
        winconditions: (state.winconditions || []).map(winConditionToObject),
        metadata: metadataPairs(state),
        metadata_lines: Object.assign({}, state.metadata_lines || {}),
        original_case_names: Object.assign({}, state.original_case_names || {}),
        original_line_numbers: Object.assign({}, state.original_line_numbers || {}),
        abbrevNames: (state.abbrevNames || []).slice(),
        levels: levelsToArray(state.levels),
        subsection: state.subsection || '',
    };
}

module.exports = { buildParserStateSnapshot };
```

- [ ] **Step 2: Wire the snapshot flag into `export_ir_json.js`**

Modify `src/tests/export_ir_json.js`:

- Add `snapshotPhase: null` to the args defaults.
- Add a branch in `parseArgs`:

```javascript
} else if (arg === '--snapshot-phase') {
    result.snapshotPhase = args[++index];
}
```

- Import the snapshot builder at the top:

```javascript
const { buildParserStateSnapshot } = require('./lib/puzzlescript_parser_snapshot');
```

- In `main()`, after `compile(...)` returns, if `options.snapshotPhase === 'parser'`, serialize the global `state` variable (JS `compiler.js` exposes it) rather than computing `ir`:

```javascript
if (options.snapshotPhase === 'parser') {
    const payload = JSON.stringify(buildParserStateSnapshot(state), null, 2);
    if (outputFile) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, `${payload}\n`, 'utf8');
    } else {
        process.stdout.write(`${payload}\n`);
    }
    return;
}
```

Ensure `state` is reachable — it's declared in `parser.js` as a closure variable inside `codeMirrorFn`. Check if it's exposed via the node env shim (`src/tests/lib/puzzlescript_node_env.js`). If not, expose `state` by adding a getter there.

Check with: `rg "global\." src/tests/lib/puzzlescript_node_env.js` to see what's already exposed.

- [ ] **Step 3: Verify the snapshot runs on a small source**

Create a tiny source file `src/tests/resources/snapshot_smoke.txt`:

```
title Smoke
author tester
homepage example.com

========
OBJECTS
========

background
blue
```

Run: `node src/tests/export_ir_json.js src/tests/resources/snapshot_smoke.txt /tmp/smoke_parser.json --snapshot-phase parser`

Expected: `/tmp/smoke_parser.json` exists, contains JSON with top-level keys matching the ParserState fields (lineNumber, section, objects, metadata, etc.).

- [ ] **Step 4: Commit**

```bash
git add src/tests/export_ir_json.js src/tests/lib/puzzlescript_parser_snapshot.js \
        src/tests/resources/snapshot_smoke.txt src/tests/lib/puzzlescript_node_env.js
git commit -m "tests: add --snapshot-phase parser to export_ir_json.js"
```

---

## Task 7: C++ ParserState → JSON serializer (dev-only)

**Files:**
- Create: `native/src/frontend/parser_state_serialize.cpp`
- Modify: `native/CMakeLists.txt` (gate behind PS_ENABLE_DEV_SERIALIZERS)

- [ ] **Step 1: Write the serializer**

Produce canonical JSON matching the JS snapshot layout exactly (same keys, same order after sort, same number formatting).

```cpp
// native/src/frontend/parser_state_serialize.cpp
// Dev-only. Not linked into production builds.
#ifdef PS_ENABLE_DEV_SERIALIZERS
#include "types/parser_state.hpp"
#include <sstream>
#include <string>

namespace puzzlescript::frontend {

// Minimal JSON emitter. Designed to match what
// src/tests/lib/puzzlescript_parser_snapshot.js produces after JSON.stringify(x, null, 2).
// Specifically: 2-space indent, LF line endings, sorted object keys (we use std::map).
// Numbers are emitted as JS would stringify them — integers have no decimal point,
// floats use the shortest round-trippable representation (rare in parser output).

namespace {
void emit_string(std::ostringstream& out, const std::string& s) {
    out << '"';
    for (char c : s) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out << buf;
                } else {
                    out << c;
                }
        }
    }
    out << '"';
}

void emit_indent(std::ostringstream& out, int depth) {
    for (int i = 0; i < depth * 2; ++i) out << ' ';
}

void emit_bool(std::ostringstream& out, bool b) { out << (b ? "true" : "false"); }
void emit_int(std::ostringstream& out, long long v) { out << v; }

void emit_string_array(std::ostringstream& out, const std::vector<std::string>& arr, int depth) {
    if (arr.empty()) { out << "[]"; return; }
    out << "[\n";
    for (size_t i = 0; i < arr.size(); ++i) {
        emit_indent(out, depth + 1);
        emit_string(out, arr[i]);
        if (i + 1 < arr.size()) out << ',';
        out << '\n';
    }
    emit_indent(out, depth); out << ']';
}

// ... helpers for each compound field follow same pattern

void emit_parser_state(std::ostringstream& out, const ParserState& s) {
    // Top-level object, keys in sorted order? JS uses INSERT order in JSON.stringify
    // but buildParserStateSnapshot assigns fields in a fixed order. We match that
    // order exactly — see puzzlescript_parser_snapshot.js buildParserStateSnapshot.
    // NOTE: keys emitted in the SAME order as the JS builder, not alphabetically.
    out << "{\n";
    // Fill in every field in the exact order of buildParserStateSnapshot.
    // (See that file for the canonical list.)
    // This is tedious but mechanical.
    // [ ... expanded in the actual implementation ... ]
    out << "\n}";
}

}  // namespace

std::string serialize_parser_state(const ParserState& s) {
    std::ostringstream out;
    emit_parser_state(out, s);
    out << '\n';
    return out.str();
}

}  // namespace puzzlescript::frontend
#endif  // PS_ENABLE_DEV_SERIALIZERS
```

**Note:** the actual field-by-field emitter is mechanical but long. Port field names in the exact order the JS `buildParserStateSnapshot` emits them. This is the *single source of truth* for the JSON shape — if C++ and JS disagree on a field, that is the first bug to fix.

- [ ] **Step 2: Add to CMake conditional on PS_ENABLE_DEV_SERIALIZERS**

In `native/CMakeLists.txt`:

```cmake
if(PS_ENABLE_DEV_SERIALIZERS)
  target_sources(puzzlescript_frontend PRIVATE src/frontend/parser_state_serialize.cpp)
endif()
```

Declare the function in a dev-only header `native/src/frontend/parser_state_serialize.hpp`:

```cpp
#pragma once
#ifdef PS_ENABLE_DEV_SERIALIZERS
#include <string>
namespace puzzlescript::frontend {
struct ParserState;
std::string serialize_parser_state(const ParserState& s);
}
#endif
```

- [ ] **Step 3: Unit test: empty ParserState → matches JS output for empty source**

First generate the JS reference snapshot:

```bash
echo "" > /tmp/empty.txt
node src/tests/export_ir_json.js /tmp/empty.txt /tmp/empty_js.json --snapshot-phase parser
```

Then write a C++ test `native/tests/frontend/test_parser_state_serialize_empty.cpp`:

```cpp
#include "frontend/types/parser_state.hpp"
#include "frontend/parser_state_serialize.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

int main() {
    using namespace puzzlescript::frontend;
    ParserState s;
    // Match JS startState: levels starts with one empty grid-level.
    s.levels.push_back({false, "", 0, {}});

    std::string got = serialize_parser_state(s);
    // Expected: read /tmp/empty_js.json and compare.
    std::ifstream js_file("/tmp/empty_js.json");
    std::stringstream ss; ss << js_file.rdbuf();
    std::string want = ss.str();
    if (got != want) {
        std::cerr << "GOT:\n" << got << "\nWANT:\n" << want << "\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
```

Wire into CMake (guarded by `PS_ENABLE_DEV_SERIALIZERS`):

```cmake
if(PS_ENABLE_DEV_SERIALIZERS)
  add_executable(test_parser_state_serialize_empty
    tests/frontend/test_parser_state_serialize_empty.cpp)
  target_link_libraries(test_parser_state_serialize_empty PRIVATE puzzlescript_frontend)
  target_include_directories(test_parser_state_serialize_empty PRIVATE src)
  add_test(NAME frontend_parser_state_serialize_empty
           COMMAND test_parser_state_serialize_empty)
endif()
```

- [ ] **Step 4: Configure with dev serializers and run the test**

```
cmake -B build-dev -DPS_ENABLE_DEV_SERIALIZERS=ON
cmake --build build-dev --target test_parser_state_serialize_empty
ctest --test-dir build-dev -R frontend_parser_state_serialize_empty --output-on-failure
```

Expected: PASS (after iterating on the field order until it matches JS byte-for-byte).

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_state_serialize.hpp native/src/frontend/parser_state_serialize.cpp \
        native/tests/frontend/test_parser_state_serialize_empty.cpp native/CMakeLists.txt
git commit -m "frontend: dev-only ParserState JSON serializer matching JS snapshot"
```

---

## Task 8: `ps_cli compile-source --emit-parser-state` dev subcommand

**Files:**
- Modify: `native/src/cli_main.cpp`

- [ ] **Step 1: Add the subcommand dispatch**

Add a new subcommand branch that:
1. Reads a source file from disk.
2. (Temporarily) constructs an empty `ParserState`. Real parsing comes in later tasks.
3. Serializes via `serialize_parser_state` if `PS_ENABLE_DEV_SERIALIZERS` is defined, else errors out.
4. Writes to stdout (or `--output <path>` when provided).

Flag must be gated: the subcommand only registers when `PS_ENABLE_DEV_SERIALIZERS` is set (preprocessor `#ifdef`).

Reference `cli_main.cpp` existing `run-source` subcommand for the flag-parsing pattern.

- [ ] **Step 2: Verify it runs on the smoke source**

```
./build-dev/ps_cli compile-source --emit-parser-state src/tests/resources/snapshot_smoke.txt
```

Expected output (for now): an empty-ish JSON because parser isn't implemented yet. That's fine — Task 9 adds the harness script that uses this.

- [ ] **Step 3: Commit**

```bash
git add native/src/cli_main.cpp
git commit -m "ps_cli: add compile-source --emit-parser-state (dev-only)"
```

---

## Task 9: `scripts/diff_parser_state_against_js.sh`

**Files:**
- Create: `scripts/diff_parser_state_against_js.sh`
- Create: `scripts/corpus_list.sh`

- [ ] **Step 1: Write corpus lister**

```bash
#!/usr/bin/env bash
# scripts/corpus_list.sh — list every PuzzleScript source fixture we diff against.
# Extracts the inline sources from src/tests/resources/testdata.js and
# errormessage_testdata.js into a tmp directory, one file per fixture.
# Usage: scripts/corpus_list.sh <output_dir>
set -euo pipefail

OUT_DIR="${1:?usage: $0 <output_dir>}"
mkdir -p "$OUT_DIR"

# We invoke a small node helper that writes one .txt per fixture with stable
# filenames: 000_<name>.txt, 001_<name>.txt, ....
node - "$OUT_DIR" <<'NODE_EOF'
'use strict';
const fs = require('fs');
const path = require('path');
const outDir = process.argv[2];

const td = require(path.resolve('src/tests/resources/testdata.js'));
const emd = require(path.resolve('src/tests/resources/errormessage_testdata.js'));

function slug(s) { return s.replace(/[^a-zA-Z0-9]/g, '_').slice(0, 64); }

let idx = 0;
const write = (name, source) => {
    const filename = `${String(idx).padStart(4, '0')}_${slug(name)}.txt`;
    fs.writeFileSync(path.join(outDir, filename), source);
    idx++;
};

// testdata: [name, source, inputs, ...] or similar — check the actual shape.
// errormessage_testdata: [name, [source, errors, errorCount]]
// See src/tests/resources/*.js for exact structure.
for (const entry of testdata) {
    const [name, source] = entry;
    if (typeof source === 'string') write(name, source);
}
for (const entry of errormessage_testdata) {
    const [name, payload] = entry;
    write(name, payload[0]);
}
NODE_EOF

echo "Extracted $(ls "$OUT_DIR" | wc -l) fixtures to $OUT_DIR"
```

Note: the actual `testdata.js` / `errormessage_testdata.js` might not be Node CommonJS modules — check with `head -3 src/tests/resources/testdata.js`. If they declare globals via `var ...`, the node helper needs to eval them; use `vm` module for that, mirroring `src/tests/lib/puzzlescript_node_env.js`.

- [ ] **Step 2: Write diff runner**

```bash
#!/usr/bin/env bash
# scripts/diff_parser_state_against_js.sh
# Usage:
#   scripts/diff_parser_state_against_js.sh <source.txt>
#     - diff one fixture
#   scripts/diff_parser_state_against_js.sh --corpus
#     - diff every fixture in the corpus; print summary
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PS_CLI="${PS_CLI:-$ROOT/build-dev/ps_cli}"
EXPORT="$ROOT/src/tests/export_ir_json.js"

if [[ ! -x "$PS_CLI" ]]; then
    echo "ps_cli not found at $PS_CLI — build first with -DPS_ENABLE_DEV_SERIALIZERS=ON" >&2
    exit 2
fi

diff_one() {
    local src="$1"
    local tmpdir; tmpdir="$(mktemp -d)"
    local js_out="$tmpdir/js.json"
    local cpp_out="$tmpdir/cpp.json"
    node "$EXPORT" "$src" "$js_out" --snapshot-phase parser >/dev/null 2>&1 || true
    "$PS_CLI" compile-source --emit-parser-state "$src" --output "$cpp_out" || true
    if diff -u "$js_out" "$cpp_out" > "$tmpdir/diff" 2>&1; then
        echo "OK   $src"
        rm -rf "$tmpdir"
        return 0
    else
        echo "DIFF $src"
        echo "  (diff in $tmpdir/diff — first 20 lines)"
        head -20 "$tmpdir/diff" | sed 's/^/    /'
        return 1
    fi
}

if [[ "${1:-}" == "--corpus" ]]; then
    CORPUS_DIR="${CORPUS_DIR:-/tmp/ps_corpus}"
    rm -rf "$CORPUS_DIR"; mkdir -p "$CORPUS_DIR"
    "$ROOT/scripts/corpus_list.sh" "$CORPUS_DIR"
    ok=0; fail=0
    for src in "$CORPUS_DIR"/*.txt; do
        if diff_one "$src" >/dev/null 2>&1; then
            ok=$((ok+1))
        else
            fail=$((fail+1))
        fi
    done
    echo "Corpus: $ok ok, $fail diff"
    [[ $fail -eq 0 ]]
else
    diff_one "$1"
fi
```

- [ ] **Step 3: chmod +x and smoke test**

```
chmod +x scripts/corpus_list.sh scripts/diff_parser_state_against_js.sh
cmake -B build-dev -DPS_ENABLE_DEV_SERIALIZERS=ON && cmake --build build-dev --target ps_cli
scripts/diff_parser_state_against_js.sh src/tests/resources/snapshot_smoke.txt
```

Expected: DIFF (because C++ parser isn't implemented yet — it emits empty state). That's fine and proves the harness works.

- [ ] **Step 4: Commit**

```bash
git add scripts/corpus_list.sh scripts/diff_parser_state_against_js.sh
git commit -m "scripts: add diff_parser_state_against_js.sh + corpus_list.sh"
```

---

## Task 10: Baseline the corpus diff — capture current (expected-failing) output

**Files:**
- Create: `native/tests/frontend/parser_corpus_baseline.txt` (git-ignored in practice, but documented here)

- [ ] **Step 1: Extract the corpus and record how many fixtures fail**

Run:
```
scripts/diff_parser_state_against_js.sh --corpus 2>&1 | tail -5
```

Expected (right now): `Corpus: 0 ok, ~730 diff` — the C++ parser does nothing, so every fixture differs. **This is the starting baseline**; every future task should monotonically decrease the `diff` count.

- [ ] **Step 2: Document the baseline in a file**

Create `docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md`:

```markdown
# Phase 1 corpus progress

Target: 0 diffs across full corpus (~730 fixtures).

| Task | Date | ok | diff | Notes |
|------|------|----|----|-------|
| 10   | YYYY-MM-DD | 0 | 730 | baseline — C++ parser stub only |
```

This file is updated at the end of each parser section task (tasks 17, 19, 23, 26, 28, 30, 37, 39, 42, 43).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "docs: capture corpus baseline (0/730 passing)"
```

---

## Task 11: ParserState JS-matching initial values

**Files:**
- Modify: `native/src/frontend/types/parser_state.hpp` (no-op if defaults already match)
- Create: `native/src/frontend/parser.hpp`
- Create: `native/src/frontend/parser.cpp`

Target: an empty source string `""` produces the same ParserState as JS `startState()`.

- [ ] **Step 1: Declare `parse()` entry point**

```cpp
// native/src/frontend/parser.hpp
#pragma once
#include "types/parser_state.hpp"
#include "diagnostic.hpp"

namespace puzzlescript::frontend {

struct ParseOutput {
    ParserState state;
    std::vector<Diagnostic> diagnostics;
};

ParseOutput parse(const std::string& source);

}  // namespace puzzlescript::frontend
```

- [ ] **Step 2: Implement `parse()` to return the startState for any input**

```cpp
// native/src/frontend/parser.cpp
#include "parser.hpp"

namespace puzzlescript::frontend {

ParserState make_start_state() {
    ParserState s;
    // JS: levels: [[]]  — a single empty grid-level.
    s.levels.push_back({false, "", 0, {}});
    return s;
}

ParseOutput parse(const std::string& /*source*/) {
    ParseOutput out;
    out.state = make_start_state();
    return out;
}

}  // namespace puzzlescript::frontend
```

- [ ] **Step 3: Wire parse() into ps_cli compile-source**

Update `native/src/cli_main.cpp` so that `compile-source --emit-parser-state` reads the file and calls `parse()` then serializes.

- [ ] **Step 4: Run the harness on the empty fixture**

Create a tiny test file `/tmp/empty.txt` with just a newline, then:

```
scripts/diff_parser_state_against_js.sh /tmp/empty.txt
```

Expected: OK (JS and C++ both emit the startState).

- [ ] **Step 5: Run full corpus again**

```
scripts/diff_parser_state_against_js.sh --corpus 2>&1 | tail -1
```

Expected: `Corpus: 1 ok, 729 diff` (just the empty-source fixture passes — no non-empty source has its own startState-only fixture, but the baseline runs including `/tmp/empty.txt` if added to corpus).

- [ ] **Step 6: Update progress log and commit**

Update `phase1-progress.md` with the new row. Commit:

```bash
git add native/src/frontend/parser.hpp native/src/frontend/parser.cpp native/src/cli_main.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: parse() returns startState matching JS for empty source"
```

---

## Task 12: Comment + whitespace handling

**Files:**
- Modify: `native/src/frontend/parser.cpp`
- Reference: `src/js/parser.js` lines 1440-1555 (the token() function's comment/whitespace handling + commentLevel tracking)

A comment in PuzzleScript is `(nested parens allowed)`. Blank lines are section-sensitive (e.g., end a level in LEVELS; reset objects_section in OBJECTS). The parser walks char-by-char tracking `commentLevel`.

- [ ] **Step 1: Define a line-buffered stream helper**

Replicate CodeMirror's `StringStream` API enough to drive the JS parser's per-line token function. For our needs (whole-source in one go), a simpler abstraction suffices: split source into lines, tokenize each line with a rolling `commentLevel`.

```cpp
// In parser.cpp (anon namespace).
struct LineStream {
    std::string_view line;
    size_t pos{0};

    bool eol() const { return pos >= line.size(); }
    bool sol() const { return pos == 0; }
    char peek() const { return eol() ? '\0' : line[pos]; }
    char next() { return eol() ? '\0' : line[pos++]; }
    // match regex-equivalent: advance if the prefix matches given chars, etc.
    // Implement on-demand as regexes are replaced (see Task 4's language_constants).
};
```

- [ ] **Step 2: Implement comment tracking in the main loop**

```cpp
ParseOutput parse(const std::string& source) {
    ParseOutput out;
    out.state = make_start_state();
    ParserState& s = out.state;

    // Split on '\n' into lines, preserving line numbering (1-based like JS).
    size_t line_start = 0;
    int lineNumber = 0;
    while (line_start <= source.size()) {
        size_t line_end = source.find('\n', line_start);
        if (line_end == std::string::npos) line_end = source.size();
        ++lineNumber;
        s.lineNumber = lineNumber;
        std::string_view line{source.data() + line_start, line_end - line_start};
        LineStream stream{line, 0};

        bool is_blank = true;
        while (!stream.eol()) {
            char c = stream.peek();
            if (s.commentLevel > 0) {
                if (c == '(')      { ++s.commentLevel; stream.next(); }
                else if (c == ')') { --s.commentLevel; stream.next(); }
                else               { stream.next(); }
                continue;
            }
            if (c == '(') { ++s.commentLevel; stream.next(); continue; }
            if (c == ' ' || c == '\t') { stream.next(); continue; }
            is_blank = false;
            // Non-comment, non-whitespace char: dispatch to section handler.
            // For Task 12, just consume the rest of the line.
            // Task 14+ implement section-specific dispatch here.
            stream.pos = line.size();  // advance to EOL for now
            break;
        }

        if (is_blank) {
            // JS blankLineHandle:
            //   if section == 'levels' and last level non-empty: push new []
            //   if section == 'objects': objects_section = 0
            if (s.section == "levels" && !s.levels.empty()) {
                const auto& last = s.levels.back();
                if (!last.is_message && !last.rows.empty()) {
                    s.levels.push_back({false, "", 0, {}});
                }
            } else if (s.section == "objects") {
                s.objects_section = 0;
            }
        }
        line_start = line_end + 1;
    }
    return out;
}
```

- [ ] **Step 3: Write a small test**

Make `/tmp/comment.txt`:

```
(this is a comment)
(nested (comment) still comment)
```

Run:
```
scripts/diff_parser_state_against_js.sh /tmp/comment.txt
```

Expected: OK (both sides see lineNumber=2 or 3 at end, commentLevel=0, section=""). If not, iterate until the diff is empty.

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser.cpp
git commit -m "frontend: handle comments and blank lines (no section logic yet)"
```

---

## Task 13: `line_should_end` machinery and end-of-line post-processing

**Files:**
- Modify: `native/src/frontend/parser.cpp`
- Reference: `src/js/parser.js` lines 453-460 (endOfLineProcessing) and references to `line_should_end`.

- [ ] **Step 1: Add end-of-line dispatch scaffolding**

After the inner while loop exits, if `s.section == "legend"` call `processLegendLine(s)` (stub); if `"sounds"` call `processSoundsLine(s)` (stub). Clear `current_line_wip_array` and `line_should_end` at the start of each line.

- [ ] **Step 2: Add stubs that do nothing (real impls in Tasks 24, 27)**

```cpp
void processLegendLine(ParserState& /*s*/) { /* stub - Task 24 */ }
void processSoundsLine(ParserState& /*s*/) { /* stub - Task 27 */ }
```

- [ ] **Step 3: Verify comment test still passes**

```
scripts/diff_parser_state_against_js.sh /tmp/comment.txt
```

Expected: OK.

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser.cpp
git commit -m "frontend: end-of-line processing scaffolding (stubs)"
```

---

## Task 14: Preamble (metadata) — simple `key value` metadata

**Files:**
- Create: `native/src/frontend/parser_preamble.cpp`
- Reference: `src/js/parser.js` — search for `parsePreambleToken` (around line 1050-1250) and the `reg_prelude_metadata_keywords` / metadata key list.

The preamble runs before the first section header (`========\nOBJECTS\n========`). It accepts metadata lines like `title Some Game`, `author Someone`, `homepage example.com`, plus boolean flags like `noaction`, `noundo`, `verbose_logging`, plus numeric keys like `flickscreen 10x5`, `zoomscreen 10x10`, `key_repeat_interval 0.2`.

- [ ] **Step 1: Collect the full metadata keyword set from JS**

Search `src/js/parser.js` for `metadata_keywords` / `metadata_keywords_that_take_a_value` / similar identifier names. Record the two lists (boolean-only and key/value) with every entry preserved.

- [ ] **Step 2: Implement `parsePreambleToken` in parser_preamble.cpp**

```cpp
// native/src/frontend/parser_preamble.cpp
#include "parser.hpp"
#include "language_constants.hpp"

namespace puzzlescript::frontend {

// Called at SOL when no section has started.
// Returns true if the token was consumed; false means "this line is a section header".
bool parse_preamble_line(ParserState& s, std::string_view line);

// See parser.js parsePreambleToken (~line 1050-1250) for full behavior.
// For this task: implement the simple "key value" metadata and collect into s.metadata.
// Error messages must match JS exactly — see `diagnostic.hpp` DiagnosticCode list.

}  // namespace puzzlescript::frontend
```

The minimal subset: parse lines of the form `word` (boolean flag) or `word value` (key/value). Append to `s.metadata` as `{key, value}`. Record line number in `s.metadata_lines[key] = lineNumber`. Raise `DiagnosticCode::UnrecognisedStuffInPrelude` for anything else.

- [ ] **Step 3: Hook parse_preamble_line into parser.cpp's per-line loop**

```cpp
// When s.section is empty, call parse_preamble_line(s, line).
```

- [ ] **Step 4: Curate a small test fixture**

Create `/tmp/preamble_basic.txt`:

```
title Test Game
author Tester
homepage example.com

========
OBJECTS
========
```

- [ ] **Step 5: Run the harness**

```
scripts/diff_parser_state_against_js.sh /tmp/preamble_basic.txt
```

Expected: DIFF initially on the metadata field. Iterate on `parse_preamble_line` until OK.

**Do not proceed to Task 15 until this fixture passes.** Add to a curated-fixtures file if helpful.

- [ ] **Step 6: Commit**

```bash
git add native/src/frontend/parser_preamble.cpp native/src/frontend/parser.cpp native/CMakeLists.txt
git commit -m "frontend: preamble parser — simple key/value metadata"
```

---

## Task 15: Preamble — boolean-flag metadata

**Files:**
- Modify: `native/src/frontend/parser_preamble.cpp`
- Reference: JS metadata list for boolean-only keys (noaction, noundo, run_rules_on_level_start, etc.)

- [ ] **Step 1: Identify boolean-flag keys in JS**

These keys take no value; their presence alone is meaningful. JS pushes `[key, ""]` into `metadata`.

- [ ] **Step 2: Extend parse_preamble_line to recognize them**

When the entire remaining line (after trimming) is a known boolean-flag keyword, push `{key, ""}`. Don't require/allow a value.

- [ ] **Step 3: Fixture**

`/tmp/preamble_flags.txt`:

```
title Flags Game
noaction
noundo
verbose_logging

========
OBJECTS
========
```

- [ ] **Step 4: Run and iterate until OK**

```
scripts/diff_parser_state_against_js.sh /tmp/preamble_flags.txt
```

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_preamble.cpp
git commit -m "frontend: preamble parser — boolean flag metadata"
```

---

## Task 16: Preamble — numeric / composite value metadata

**Files:**
- Modify: `native/src/frontend/parser_preamble.cpp`

Keys like `flickscreen 10x5`, `zoomscreen 10x10`, `key_repeat_interval 0.2`, `background_color blue`. JS stores the raw value string — parsing into a number happens later.

- [ ] **Step 1: Extend parse_preamble_line to consume "rest of line" value**

The value is everything after the first whitespace, trimmed. JS does not validate the value during parse (that's the compiler phase).

- [ ] **Step 2: Fixture**

`/tmp/preamble_values.txt`:

```
title Values
flickscreen 10x5
zoomscreen 7x7
key_repeat_interval 0.25
background_color blue

========
OBJECTS
========
```

- [ ] **Step 3: Run and iterate until OK**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_preamble.cpp
git commit -m "frontend: preamble parser — value-taking metadata"
```

---

## Task 17: Preamble — `UnrecognisedStuffInPrelude` + `GameTitleTooLong` warning

**Files:**
- Modify: `native/src/frontend/parser_preamble.cpp`
- Modify: `native/src/frontend/diagnostic.cpp` (add formatter cases)

- [ ] **Step 1: Add diagnostic codes and formatter cases**

In diagnostic.cpp, handle:
- `UnrecognisedStuffInPrelude` → `"Unrecognised stuff in the prelude."`
- `GameTitleTooLong` → `"Game title is too long to fit on screen; truncating to fit."`

Consult the fixtures in `errormessage_testdata.js` (lines 44-50 of that file show the exact expected strings).

- [ ] **Step 2: Emit them from parse_preamble_line**

- [ ] **Step 3: Run error-fixture subset**

Use the diagnostic diff (write this alongside, or reuse parser_state diff + separately compare diagnostics). Since we haven't built the diagnostic diff script yet (that's Phase 1 task later — add it in Task 44), for this task manually compare against JS:

```
node src/tests/export_ir_json.js <fixture>.txt /tmp/js.json --snapshot-phase parser
./build-dev/ps_cli compile-source --emit-parser-state <fixture>.txt --output /tmp/cpp.json
# Compare diagnostics emitted by ps_cli (stderr?) against JS errorStrings.
```

For now, `compile-source` should print diagnostics to stderr in JS-compat-formatted form. Update the subcommand to do so.

- [ ] **Step 4: Progress-log update**

Run the full corpus. Expect many fixtures to still fail (LEGEND, OBJECTS etc. not yet implemented), but fixtures that only exercise preamble should pass.

```
scripts/diff_parser_state_against_js.sh --corpus 2>&1 | tail -1
```

Record count in `phase1-progress.md`.

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_preamble.cpp native/src/frontend/diagnostic.cpp \
        native/src/cli_main.cpp docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: preamble diagnostics (unrecognised/title-too-long)"
```

---

## Task 18: Section header detection and routing

**Files:**
- Modify: `native/src/frontend/parser.cpp`
- Reference: `src/js/parser.js` lines 1497-1570 (the section-header branch in token())

A section header is a line of `=` chars followed by a newline, then the section name on its own line, then another `=` line. JS uses `reg_sectionNames` + `reg_equalsrow` to match. The parser tracks `visitedSections` (ordered list) and rejects duplicates, requires OBJECTS first, and errors on unknown names.

- [ ] **Step 1: Implement section header detection**

```cpp
// In parser.cpp, at SOL when line matches /^=+$/:
//   read next non-blank line: match against reg_sectionNames.
//   validate: if visitedSections empty and section != "objects" => error.
//   push section into visitedSections.
//   set s.section = <name>.
//   expect a closing =... line next.
```

JS does this incrementally via the token() state machine (line-by-line); we can do the same because our outer loop is line-by-line too. Use a "pending section header" flag if needed.

- [ ] **Step 2: Route non-preamble, non-header lines to section handlers**

Add stubs `parse_objects_line`, `parse_legend_line`, etc. that do nothing yet. Dispatch based on `s.section`.

- [ ] **Step 3: Fixture**

`/tmp/sections.txt`:

```
title T
author A
homepage h

========
OBJECTS
========

=======
LEGEND
=======

=======
SOUNDS
=======

================
COLLISIONLAYERS
================

======
RULES
======

==============
WINCONDITIONS
==============

=======
LEVELS
=======
```

- [ ] **Step 4: Run and iterate**

```
scripts/diff_parser_state_against_js.sh /tmp/sections.txt
```

Expected: `visitedSections` lists all sections, `section` ends at `"levels"`, body fields empty. Fix until OK.

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser.cpp
git commit -m "frontend: section header detection + dispatch routing"
```

---

## Task 19: Section-related diagnostics (`MustStartWithObjects`, unknown section, duplicate section)

**Files:**
- Modify: `native/src/frontend/parser.cpp`
- Modify: `native/src/frontend/diagnostic.cpp`

- [ ] **Step 1: Emit diagnostics matching JS exact wording**

From JS:
- `'must start with section "OBJECTS"'` — when first section isn't OBJECTS.
- Other section errors: check JS for the exact wording around line 1570.

- [ ] **Step 2: Add formatter cases and fixtures**

Hand-curate 2-3 fixtures that exercise each path; iterate until JS and C++ produce identical diagnostic text.

- [ ] **Step 3: Full corpus run + progress log update**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser.cpp native/src/frontend/diagnostic.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: section header diagnostics"
```

---

## Task 20: OBJECTS section — name line

**Files:**
- Create: `native/src/frontend/parser_objects.cpp`
- Reference: `src/js/parser.js` line 472-570 (`parseObjectsToken` + `tryParseName`).

An OBJECTS section block has 3-4 lines per object:
1. Name (optionally followed by `COPY:existingobject`) — `objects_section == 0`.
2. Color list — `objects_section == 1`.
3. 5x5 sprite grid rows 1..5 — `objects_section == 2..6`.

Blank line resets `objects_section` to 0 (next object starts).

- [ ] **Step 1: Implement `parse_objects_name_line`**

Given a line at `objects_section == 0`, read the first word as `objects_candname`. Check:
- Already defined in `s.objects` → `ObjectDefinedMultipleTimes` error.
- Already in `legend_synonyms` → `NameInUseFromLegend` error.
- In keyword list → `NamedObjectIsKeyword` warning.

Add a new `ObjectDef` to `s.objects[candname_lowered]` with the line number. Advance `objects_section = 1`.

- [ ] **Step 2: Fixture**

`/tmp/objects_name.txt`:

```
title T

========
OBJECTS
========

Background

Wall

Player

(no colors/sprites yet — JS will likely error on compilation, but parser should accept.)
```

- [ ] **Step 3: Run and iterate**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_objects.cpp native/src/frontend/parser.cpp native/CMakeLists.txt
git commit -m "frontend: OBJECTS name-line parsing"
```

---

## Task 21: OBJECTS section — color line

**Files:**
- Modify: `native/src/frontend/parser_objects.cpp`
- Modify: `native/src/frontend/language_constants.cpp` (populate color table — see Task 4 step 2 TODO)
- Reference: `src/js/parser.js` line 570-650 (color parsing inside parseObjectsToken) + `src/js/colors.js` (color name → hex table).

- [ ] **Step 1: Port src/js/colors.js color name → hex table**

Populate `language::color_name_to_hex` with every named color from JS `colorPalettesAlphabetical` / `master_palette`. Keep hex strings in the exact same case as JS emits (`#FFFFFF` or `#ffffff`? — check JS — we match).

Compute expected hex for each name by consulting `src/js/colors.js`.

- [ ] **Step 2: Implement color-line parsing**

At `objects_section == 1`, split the line by whitespace/commas. For each token:
- Literal `#rgb` or `#rrggbb` hex → store as-is.
- Named color → store the hex (or the name, depending on what JS stores — check by diffing).
- Unknown → error, but JS still pushes something. Mirror the JS behavior.

Append all color values to `s.objects[candname].colors`. Advance `objects_section = 2`.

- [ ] **Step 3: Fixture**

`/tmp/objects_colors.txt`:

```
title T

========
OBJECTS
========

Background
blue

Wall
darkbrown darkbrown

Player
#ff00ff
```

- [ ] **Step 4: Run and iterate until OK**

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_objects.cpp native/src/frontend/language_constants.cpp
git commit -m "frontend: OBJECTS color-line parsing + color name table"
```

---

## Task 22: OBJECTS section — sprite matrix (5x5)

**Files:**
- Modify: `native/src/frontend/parser_objects.cpp`
- Reference: `src/js/parser.js` line 650-730.

At `objects_section ∈ [2..6]`, each line is a sprite row. Each char is either `.` (transparent), `0`-`9` (color index into the color list), or an error. After 5 rows, `objects_section = 7` (no-op until blank line).

- [ ] **Step 1: Parse sprite rows + validate width/indices**

Push row as a vector of single-char strings (or the string itself). Match JS structure exactly — check the JS state.objects_spritematrix and objects[candname].spritematrix shapes.

- [ ] **Step 2: Diagnostics**

- `SpriteMustBe5By5` — wrong width/height.
- `SpriteGraphicsMustBe5By5` — index out of range.

Consult `errormessage_testdata.js` for the exact wording.

- [ ] **Step 3: Fixture**

`/tmp/objects_sprites.txt`:

```
title T

========
OBJECTS
========

Background
blue

Player
black orange white
.0.0.
.111.
00000
.3.3.
.1.1.
```

- [ ] **Step 4: Run and iterate**

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_objects.cpp native/src/frontend/diagnostic.cpp
git commit -m "frontend: OBJECTS sprite-matrix parsing + 5x5 diagnostics"
```

---

## Task 23: OBJECTS — full section pass + error cases

**Files:**
- Modify: `native/src/frontend/parser_objects.cpp`

- [ ] **Step 1: Run the corpus filtered to fixtures that end after OBJECTS**

Write a helper that strips a source at the first non-OBJECTS section header and runs the harness on each. Focus on passing all of these before moving on.

Alternative: just run the full corpus, expect most to still fail downstream, but track which OBJECTS-ONLY problems the remaining diffs implicate.

- [ ] **Step 2: Fix edge cases**

Duplicate name within OBJECTS, name clashes with legend synonyms, keyword-as-name warnings, leading whitespace, etc.

- [ ] **Step 3: Progress log**

Update `phase1-progress.md` with new OK count.

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_objects.cpp docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: OBJECTS section — handle all corpus edge cases"
```

---

## Task 24: LEGEND section — simple synonyms (`A = B`)

**Files:**
- Create: `native/src/frontend/parser_legend.cpp`
- Reference: `src/js/parser.js` line 263-432 (`processLegendLine`).

End-of-line processing: JS splits the legend line into `current_line_wip_array` during tokenization, then `processLegendLine` classifies it as synonym/aggregate/property. We port the classification step.

- [ ] **Step 1: Tokenize legend lines into current_line_wip_array**

In `parser_legend.cpp`, add `parse_legend_line_token` called during char-by-char scan in the outer loop. Tokens are words separated by whitespace; `=`, `or`, `and` are special.

- [ ] **Step 2: Implement processLegendLine (synonym path only)**

When `splits.length === 3` and the middle element is `=`: create a legend_synonym entry with `tokens = [splits[0], splits[2]]` and `lineNumber = s.lineNumber`. Register original-case name.

- [ ] **Step 3: Fixture**

`/tmp/legend_synonym.txt`:

```
title T

========
OBJECTS
========

Background
blue

Wall
darkbrown

=======
LEGEND
=======

. = Background
# = Wall
```

- [ ] **Step 4: Run and iterate**

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_legend.cpp native/src/frontend/parser.cpp native/CMakeLists.txt
git commit -m "frontend: LEGEND synonyms"
```

---

## Task 25: LEGEND — aggregates (`A = B and C`) and properties (`A = B or C`)

**Files:**
- Modify: `native/src/frontend/parser_legend.cpp`

- [ ] **Step 1: Classify by joiner token (and/or)**

Port `splits[3] === 'and'` → aggregate and `splits[3] === 'or'` → property. For each, JS runs a recursive `substitutor`; port the same recursion.

- [ ] **Step 2: Fixture**

`/tmp/legend_compound.txt`:

```
title T

========
OBJECTS
========

Background
blue

Wall
darkbrown

Player
black

Crate
orange

=======
LEGEND
=======

Moveable = Player or Crate
StartingPlayer = Player and Crate
```

- [ ] **Step 3: Run and iterate**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_legend.cpp
git commit -m "frontend: LEGEND aggregates + properties"
```

---

## Task 26: LEGEND — diagnostics + full-section pass

**Files:**
- Modify: `native/src/frontend/parser_legend.cpp`
- Modify: `native/src/frontend/diagnostic.cpp`

Port all legend-related diagnostic strings from JS lines 263-432. Each diagnostic gets a DiagnosticCode + formatter case. Use error-fixture strings from `errormessage_testdata.js` as the acceptance criteria.

- [ ] **Step 1: Exhaustively port diagnostic cases**

- `LegendIncorrectFormat` (1-word) / `LegendDanglingWord` (even-count)
- `LegendCannotDefineSelf` (self-reference)
- `LegendRepeatedRhs` (warning)
- `LegendAggregateFromProperty`
- `LegendPropertyFromAggregate`
- `LegendIncorrectFormatGeneric` (fallback)
- `WordNotDefined` (from checkNameDefined call in legend)

- [ ] **Step 2: Full-corpus run + progress log update**

- [ ] **Step 3: Commit**

```bash
git add native/src/frontend/parser_legend.cpp native/src/frontend/diagnostic.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: LEGEND diagnostics complete"
```

---

## Task 27: SOUNDS section — event/seed rows

**Files:**
- Create: `native/src/frontend/parser_sounds.cpp`
- Reference: `src/js/parser.js` line 434-449 (`processSoundsLine`) + `parseSoundsToken` (~line 770-870).

A sound line is `<event> <seed>` e.g. `sfx1 44641500`, `startgame 26858107`, `endlevel 34292905`. Also verb rows: `<object> MOVE <seed>`.

- [ ] **Step 1: Tokenize and push to sounds array**

Port the token function + processSoundsLine. Each row pushed onto `s.sounds` is the array of tokens with `lineNumber` appended.

- [ ] **Step 2: Fixture**

`/tmp/sounds_basic.txt`:

```
title T

========
OBJECTS
========

Background
blue

Crate
orange

=======
LEGEND
=======

C = Crate

=======
SOUNDS
=======

Crate MOVE 36772507
sfx1 44641500
startgame 26858107
```

- [ ] **Step 3: Run and iterate**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_sounds.cpp native/CMakeLists.txt
git commit -m "frontend: SOUNDS basic events + verbs"
```

---

## Task 28: SOUNDS diagnostics + edge cases

**Files:**
- Modify: `native/src/frontend/parser_sounds.cpp`
- Modify: `native/src/frontend/diagnostic.cpp`

- `UnexpectedSoundToken` ("unexpected sound token \"boop\".") + related errors.

- [ ] **Step 1: Port JS sound diagnostics**

- [ ] **Step 2: Full corpus + progress log**

- [ ] **Step 3: Commit**

```bash
git add native/src/frontend/parser_sounds.cpp native/src/frontend/diagnostic.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: SOUNDS diagnostics + edge cases"
```

---

## Task 29: COLLISIONLAYERS section — basic layer rows

**Files:**
- Create: `native/src/frontend/parser_collisionlayers.cpp`
- Reference: `src/js/parser.js` — search for `parseCollisionLayersToken` (~line 870-950).

A collision layer line is `obj1, obj2, obj3` (or just `obj1`). Each line is one collision layer.

- [ ] **Step 1: Tokenize comma-separated names**

Push `{names: [...], lineNumber: s.lineNumber}` onto `s.collisionLayers`.

- [ ] **Step 2: Fixture**

`/tmp/collision_basic.txt`:

```
title T

========
OBJECTS
========

Background
blue

Target
green

Player
black

Wall
darkbrown

=======
LEGEND
=======

. = Background
T = Target
P = Player
# = Wall

================
COLLISIONLAYERS
================

Background
Target
Player, Wall
```

- [ ] **Step 3: Run and iterate**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_collisionlayers.cpp native/CMakeLists.txt
git commit -m "frontend: COLLISIONLAYERS basic parsing"
```

---

## Task 30: COLLISIONLAYERS diagnostics + full pass

**Files:**
- Modify: `native/src/frontend/parser_collisionlayers.cpp`
- Modify: `native/src/frontend/diagnostic.cpp`

- [ ] **Step 1: Enumerate every logError/logWarning in JS `parseCollisionLayersToken`**

```
rg -n "logError|logWarning" src/js/parser.js | awk -F: '$2+0 >= 870 && $2+0 <= 950'
```

For each, add a DiagnosticCode + formatter case using the exact JS string.

- [ ] **Step 2: Full corpus run; fix layer-specific residual diffs**

- [ ] **Step 3: Update progress log and commit**

```bash
git add native/src/frontend/parser_collisionlayers.cpp native/src/frontend/diagnostic.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: COLLISIONLAYERS diagnostics + corpus progress"
```

---

## Task 31: RULES — basic bracket/pipe/arrow lexing

**Files:**
- Create: `native/src/frontend/parser_rules.cpp`
- Reference: `src/js/parser.js` — search for `parseRulesToken` (~line 950-1250). This is the most complex section.

A rule is `[lhs_cells] -> [rhs_cells]` with optional prefixes (`late`, `rigid`, direction). Cells separated by `|`. Multiple bracket groups per side.

- [ ] **Step 1: Implement the per-char state machine for rules**

JS maintains `bracket_balance`, `inside_cell`, `arrow_passed`, `rule_prelude` in state. Port these fields (already in ParserState).

Start by emitting tokens for `[`, `]`, `|`, `->` and collecting them into a raw token list attached to a rule object. We'll enrich the structure in later tasks.

- [ ] **Step 2: Minimal fixture — trivial rule**

`/tmp/rules_trivial.txt`:

```
title T

========
OBJECTS
========

Background
blue

Player
black

=======
LEGEND
=======

. = Background
P = Player

================
COLLISIONLAYERS
================

Background
Player

======
RULES
======

[ Player ] -> [ Player ]
```

- [ ] **Step 3: Run and iterate**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_rules.cpp native/CMakeLists.txt
git commit -m "frontend: RULES — basic lexing (brackets/pipes/arrow)"
```

---

## Task 32: RULES — cell content tokens (object names, `no`, `moving`, `>`, `<`, etc.)

**Files:**
- Modify: `native/src/frontend/parser_rules.cpp`

Ports the token list inside a cell: direction qualifiers + object references.

- [ ] **Step 1: Extend the state machine to recognize all cell-body tokens**

Reference: JS list of direction-relative qualifiers in languageConstants (`relativedirs`, etc.). Port every case so the raw rule-token list matches JS.

- [ ] **Step 2: Fixture — moveable rule**

```
[ > Player | Crate ] -> [ > Player | > Crate ]
```

- [ ] **Step 3: Iterate, commit**

```bash
git commit -m "frontend: RULES — cell content tokens"
```

---

## Task 33: RULES — direction prefixes (up/down/left/right/horizontal/vertical)

**Files:**
- Modify: `native/src/frontend/parser_rules.cpp`

- [ ] **Step 1: Port direction-prefix handling**

Before the first `[`, accept one or more direction tokens; store on the rule.

- [ ] **Step 2: Fixture + iterate**

```
up [ Player ] -> [ > Player ]
horizontal [ Crate | Crate ] -> [ Crate | Crate ]
```

- [ ] **Step 3: Commit**

```bash
git commit -m "frontend: RULES — direction prefixes"
```

---

## Task 34: RULES — `late` / `rigid` modifiers

**Files:**
- Modify: `native/src/frontend/parser_rules.cpp`
- Reference: `src/js/parser.js` — search for `'late'` and `'rigid'` handling in `parseRulesToken`.

- [ ] **Step 1: Accept `late` and/or `rigid` tokens before direction prefix / first `[`**

Store on the raw rule object with the exact field names JS uses (check `state.rules` shape in JS by inspecting a snapshot).

- [ ] **Step 2: Curate fixture `/tmp/rules_modifiers.txt`**

```
[... preamble + minimal object set from earlier fixtures ...]

======
RULES
======

late [ Player | Crate ] -> [ Player | Crate ]
rigid [ > Player ] -> [ > Player ]
late rigid [ Crate ] -> [ Crate ]
```

- [ ] **Step 3: Run `scripts/diff_parser_state_against_js.sh /tmp/rules_modifiers.txt` and iterate until OK**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_rules.cpp
git commit -m "frontend: RULES — late / rigid modifiers"
```

---

## Task 35: RULES — RHS commands (sfx0, win, message, cancel, again, ...)

**Files:**
- Modify: `native/src/frontend/parser_rules.cpp`
- Reference: `language::commandwords` (Task 4) and JS `parseRulesToken` command-handling branch.

- [ ] **Step 1: Accept command tokens after the arrow and/or after the last RHS `]`**

A rule like `[ Player ] -> [ Player ] sfx1 again` has two trailing command tokens. The `message Something here` command is special — it consumes the rest of the line as a string (see JS for exact behavior).

- [ ] **Step 2: Curate fixture `/tmp/rules_commands.txt`**

```
[... preamble + objects ...]

======
RULES
======

[ Player ] -> [ Player ] sfx1
[ Crate ] -> [ Crate ] again
[ > Player | Wall ] -> cancel
[ Player ] -> win
[ Player ] -> message you win!
```

- [ ] **Step 3: Run diff and iterate until OK**

- [ ] **Step 4: Commit**

```bash
git add native/src/frontend/parser_rules.cpp
git commit -m "frontend: RULES — RHS commands (sfx/win/message/cancel/again)"
```

---

## Task 36: RULES — `startloop` / `endloop`, ellipsis

**Files:**
- Modify: `native/src/frontend/parser_rules.cpp`
- Reference: `src/js/parser.js` `reg_loopmarker` + JS ellipsis handling in bracket interior.

- [ ] **Step 1: Parse `startloop` and `endloop` as standalone rule lines**

JS pushes them into `state.rules` as special entries (not with `[...]` brackets). Mirror the structure JS uses — inspect via snapshot diff.

- [ ] **Step 2: Parse `...` (ellipsis) as a bracket-interior token**

The ellipsis token `...` occupies a cell slot. JS stores it distinctly from cell tokens (see JS `reg_ellipsis` / ellipsis pattern handling in the rules parser).

- [ ] **Step 3: Curate fixture `/tmp/rules_loops_ellipsis.txt`**

```
[... preamble + objects ...]

======
RULES
======

startloop
[ Player ] -> [ > Player ]
endloop

[ > Block |... | Grille ] -> [ Block | | Grille ]
```

- [ ] **Step 4: Run diff and iterate until OK**

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_rules.cpp
git commit -m "frontend: RULES — loops + ellipsis"
```

---

## Task 37: RULES — all diagnostics + full-section pass

**Files:**
- Modify: `native/src/frontend/parser_rules.cpp`
- Modify: `native/src/frontend/diagnostic.cpp`

Port every `logError` / `logWarning` in `parseRulesToken`. Main ones to hit:

- `RuleNoArrow`
- (others — enumerate exhaustively by grepping `logError\|logWarning` inside the rules parse function)

- [ ] **Step 1: Enumerate every logError / logWarning in the JS rules parser**

```
rg -n "logError|logWarning" src/js/parser.js | awk -F: '$2 ~ /^[0-9]+$/ && $2 >= 950 && $2 <= 1250'
```

Port each with a unique DiagnosticCode + formatter case.

- [ ] **Step 2: Full corpus run; fix residual diffs**

- [ ] **Step 3: Commit**

```bash
git add native/src/frontend/parser_rules.cpp native/src/frontend/diagnostic.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: RULES — diagnostics + full pass"
```

---

## Task 38: WINCONDITIONS — basic parsing

**Files:**
- Create: `native/src/frontend/parser_winconditions.cpp`
- Reference: `src/js/parser.js` — `parseWinConditionsToken` (~line 1250-1350).

Syntax: `<quantifier> <object>` or `<quantifier> <object> on <object>`.

- [ ] **Step 1: Tokenize and push into s.winconditions**

- [ ] **Step 2: Fixture**

```
==============
WINCONDITIONS
==============

all Player on Target
some Player
no Crate
```

- [ ] **Step 3: Iterate, commit**

```bash
git commit -m "frontend: WINCONDITIONS basic"
```

---

## Task 39: WINCONDITIONS diagnostics + full-section pass

**Files:**
- Modify: `native/src/frontend/parser_winconditions.cpp`
- Modify: `native/src/frontend/diagnostic.cpp`

- [ ] **Step 1: Enumerate every logError/logWarning in JS `parseWinConditionsToken`**

Run: `rg -n "logError|logWarning" src/js/parser.js | awk -F: '$2+0 >= 1250 && $2+0 <= 1350'`

For each, add a DiagnosticCode + formatter case using the exact JS string.

- [ ] **Step 2: Run full corpus diff; fix winconditions-specific residual diffs**

- [ ] **Step 3: Update progress log and commit**

```bash
git add native/src/frontend/parser_winconditions.cpp native/src/frontend/diagnostic.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: WINCONDITIONS diagnostics + corpus progress"
```

---

## Task 40: LEVELS — grid row parsing

**Files:**
- Create: `native/src/frontend/parser_levels.cpp`
- Reference: `src/js/parser.js` — `parseLevelsToken` (~line 1350-1440).

Each non-blank line in LEVELS is a row. Blank line ends a level, starts a new one.

- [ ] **Step 1: Append row to `s.levels.back().rows`**

Create new LevelDat on blank line if the current one is non-empty.

Validate: row width matches first row of current level (error otherwise).

- [ ] **Step 2: Fixture + iterate + commit**

```
=======
LEVELS
=======

#####
#.P.#
#.*.#
#####
```

```bash
git commit -m "frontend: LEVELS — grid rows"
```

---

## Task 41: LEVELS — `message` lines

**Files:**
- Modify: `native/src/frontend/parser_levels.cpp`
- Reference: `src/js/parser.js` message-line handling in `parseLevelsToken`.

- [ ] **Step 1: Parse `message <text>` as a LevelDat with `is_message=true`**

The line `message hello world` becomes a new `LevelDat{is_message=true, message="hello world"}` pushed onto `s.levels`. Check JS for trimming semantics.

- [ ] **Step 2: Handle `MessageMissingSpace` warning**

When the line starts with `message` followed immediately by non-space content (e.g. `messagefoo`), JS emits a warning then treats the rest as the message text. Match exactly.

- [ ] **Step 3: Fixture `/tmp/levels_messages.txt`**

```
[... preamble + minimal objects + legend + collisionlayers + rules + winconditions ...]

=======
LEVELS
=======

message level 1

##P##

message level 2

#.P.#

messagebad (should trigger MessageMissingSpace warning)

#...#
```

- [ ] **Step 4: Run diff and iterate until OK**

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/parser_levels.cpp native/src/frontend/diagnostic.cpp
git commit -m "frontend: LEVELS — message screens + MessageMissingSpace warning"
```

---

## Task 42: LEVELS — diagnostics + full-section pass

**Files:**
- Modify: `native/src/frontend/parser_levels.cpp`
- Modify: `native/src/frontend/diagnostic.cpp`

- [ ] **Step 1: Enumerate every logError/logWarning in JS `parseLevelsToken`**

```
rg -n "logError|logWarning" src/js/parser.js | awk -F: '$2+0 >= 1350 && $2+0 <= 1440'
```

For each, add a DiagnosticCode + formatter case using the exact JS string. Common ones: rows-of-unequal-length error, unknown-character-in-level (from legend), etc.

- [ ] **Step 2: Full corpus run; fix levels-specific residual diffs**

- [ ] **Step 3: Update progress log and commit**

```bash
git add native/src/frontend/parser_levels.cpp native/src/frontend/diagnostic.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: LEVELS — diagnostics + corpus progress"
```

---

## Task 43: Full-corpus ParserState parity gate

**Files:**
- None (harness run + bug-fix loop)

- [ ] **Step 1: Run full corpus**

```
scripts/diff_parser_state_against_js.sh --corpus
```

- [ ] **Step 2: For each remaining diff, classify**

- Real parser bug → open issue, fix inline or batch.
- JS quirk we missed → adjust C++ to match.
- Canonicalization bug → fix canonicalization only (never change C++ or JS semantics for this).

- [ ] **Step 3: Iterate until 0 diffs**

If a fixture is known to have a JS bug (e.g. assertion failure in JS itself), mark it as "match JS error exactly" — don't try to make C++ "better" than JS.

- [ ] **Step 4: Commit each batch of fixes as they land**

Final commit message:

```bash
git commit -m "frontend: full corpus ParserState parity (730/730 fixtures)"
```

- [ ] **Step 5: Record in progress log as COMPLETE**

Update the progress table's final row.

---

## Task 44: Diagnostics parity gate — `scripts/diff_diagnostics_against_js.sh`

**Files:**
- Create: `scripts/diff_diagnostics_against_js.sh`
- Modify: `native/src/cli_main.cpp` — ensure `compile-source` emits diagnostics to stdout (one per line, `format_for_js_compat` output) when `--emit-diagnostics` is passed.

- [ ] **Step 1: Write diff script**

Canonicalize both sides per spec §5.3 (strip HTML tags, normalize whitespace, trim). Compare diagnostic streams, order-sensitive.

- [ ] **Step 2: Run on error-fixture subset (errormessage_testdata.js)**

Expect most fixtures to pass since diagnostic text has been matched per task. Fix any stragglers.

- [ ] **Step 3: Commit + progress log**

```bash
git add scripts/diff_diagnostics_against_js.sh native/src/cli_main.cpp \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md
git commit -m "frontend: diagnostics parity gate — corpus clean"
```

---

## Task 44b: Implement the C API (`ps_frontend_parse` and accessors)

**Files:**
- Create: `native/src/frontend/c_api_frontend.cpp`
- Modify: `native/CMakeLists.txt` (add `c_api_frontend.cpp` to `PUZZLESCRIPT_FRONTEND_SOURCES`)
- Reference: `native/include/puzzlescript/frontend.h` (from Task 2), existing `native/src/c_api.cpp` for the opaque-handle pattern.

The C header has existed since Task 2 as a forward-declared surface. Until now `ps_cli` has called `puzzlescript::frontend::parse()` directly (in C++). Now we wire the C API so future consumers (LSP, etc.) can link against `libpuzzlescript_frontend.a` without including any C++ headers.

- [ ] **Step 1: Define the opaque result struct**

```cpp
// native/src/frontend/c_api_frontend.cpp
#include "puzzlescript/frontend.h"
#include "parser.hpp"
#include "diagnostic.hpp"
#include "parser_state_serialize.hpp"
#include <string>
#include <vector>

struct ps_frontend_result {
    puzzlescript::frontend::ParserState state;
    std::vector<puzzlescript::frontend::Diagnostic> diagnostics;
    std::vector<std::string> diagnostic_texts;  // keeps C-string lifetimes alive
    std::vector<ps_diagnostic> diagnostic_c;    // parallel array with .message pointing into diagnostic_texts
};

extern "C" ps_frontend_result* ps_frontend_parse(const char* source, size_t source_len) {
    auto* r = new ps_frontend_result;
    std::string s(source, source_len);
    auto out = puzzlescript::frontend::parse(s);
    r->state = std::move(out.state);
    r->diagnostics = std::move(out.diagnostics);
    r->diagnostic_texts.reserve(r->diagnostics.size());
    r->diagnostic_c.reserve(r->diagnostics.size());
    for (const auto& d : r->diagnostics) {
        r->diagnostic_texts.push_back(puzzlescript::frontend::format_for_js_compat(d));
        ps_diagnostic c{};
        c.severity = static_cast<ps_diagnostic_severity>(static_cast<int>(d.severity));
        c.code = static_cast<int32_t>(d.code);
        c.line = d.line.has_value() ? *d.line : -1;
        c.message = r->diagnostic_texts.back().c_str();
        r->diagnostic_c.push_back(c);
    }
    return r;
}

extern "C" size_t ps_frontend_result_diagnostic_count(const ps_frontend_result* r) {
    return r ? r->diagnostic_c.size() : 0;
}

extern "C" const ps_diagnostic* ps_frontend_result_diagnostic(
    const ps_frontend_result* r, size_t index)
{
    if (!r || index >= r->diagnostic_c.size()) return nullptr;
    return &r->diagnostic_c[index];
}

extern "C" size_t ps_frontend_result_parser_state_json(
    const ps_frontend_result* r, char* buf, size_t cap)
{
#ifdef PS_ENABLE_DEV_SERIALIZERS
    if (!r) return 0;
    std::string s = puzzlescript::frontend::serialize_parser_state(r->state);
    if (buf && cap > 0) {
        size_t n = s.size() < cap - 1 ? s.size() : cap - 1;
        std::memcpy(buf, s.data(), n);
        buf[n] = '\0';
    }
    return s.size();
#else
    (void)r; (void)buf; (void)cap;
    return 0;
#endif
}

extern "C" void ps_frontend_result_free(ps_frontend_result* r) { delete r; }
```

- [ ] **Step 2: Add to CMake build**

```cmake
set(PUZZLESCRIPT_FRONTEND_SOURCES
  src/frontend/c_api_frontend.cpp
  src/frontend/diagnostic.cpp
  src/frontend/language_constants.cpp
  src/frontend/parser.cpp
  src/frontend/parser_collisionlayers.cpp
  src/frontend/parser_legend.cpp
  src/frontend/parser_levels.cpp
  src/frontend/parser_objects.cpp
  src/frontend/parser_preamble.cpp
  src/frontend/parser_rules.cpp
  src/frontend/parser_sounds.cpp
  src/frontend/parser_winconditions.cpp
)
```

(Remove `placeholder.cpp` here too — it's deleted in Task 46.)

- [ ] **Step 3: Write a C-only smoke test**

Create `native/tests/frontend/test_c_api_smoke.c`:

```c
#include "puzzlescript/frontend.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const char* src = "title Hi\n\n========\nOBJECTS\n========\n\nBackground\nblue\n";
    ps_frontend_result* r = ps_frontend_parse(src, strlen(src));
    assert(r != NULL);
    /* Diagnostic count may be nonzero (e.g. missing Player); that's fine. */
    size_t n = ps_frontend_result_diagnostic_count(r);
    for (size_t i = 0; i < n; ++i) {
        const ps_diagnostic* d = ps_frontend_result_diagnostic(r, i);
        printf("[%d] line=%d: %s\n", (int)d->severity, d->line, d->message);
    }
    ps_frontend_result_free(r);
    printf("OK\n");
    return 0;
}
```

Add to CMake:

```cmake
add_executable(test_c_api_smoke tests/frontend/test_c_api_smoke.c)
set_target_properties(test_c_api_smoke PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_c_api_smoke PRIVATE puzzlescript_frontend)
add_test(NAME frontend_c_api_smoke COMMAND test_c_api_smoke)
```

- [ ] **Step 4: Build and run**

```
cmake --build build --target test_c_api_smoke && ctest --test-dir build -R frontend_c_api_smoke --output-on-failure
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add native/src/frontend/c_api_frontend.cpp native/tests/frontend/test_c_api_smoke.c native/CMakeLists.txt
git commit -m "frontend: implement C API (ps_frontend_parse + accessors)"
```

---

## Task 45: CI wiring + performance sanity

**Files:**
- Modify: `native/CMakeLists.txt` (add `parser_corpus_diff` ctest)
- Modify: `.github/workflows/*.yml` if applicable (check with `ls .github/workflows/` first; may not exist)
- Modify: `scripts/perf_check.sh` (add sanity check that `compile-source` completes on a median fixture in <100ms)

- [ ] **Step 1: Add CI test that runs the full corpus diff**

```cmake
if(PS_ENABLE_DEV_SERIALIZERS)
  add_test(
    NAME parser_corpus_diff
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/diff_parser_state_against_js.sh --corpus
  )
  set_tests_properties(parser_corpus_diff PROPERTIES TIMEOUT 600)
endif()
```

- [ ] **Step 2: Add sanity perf check**

In `scripts/perf_check.sh`, add a single invocation on `src/demo/sokoban_basic.txt` that must complete in under some sane threshold (say 100ms). This is a smoke check, not a gate — actual performance metrics come in Phase 4.

- [ ] **Step 3: Commit**

```bash
git add native/CMakeLists.txt scripts/perf_check.sh
git commit -m "frontend: CI gate for parser corpus parity + sanity perf check"
```

---

## Task 46: Cleanup + phase completion

**Files:**
- Delete: `native/src/frontend/placeholder.cpp`
- Modify: `native/CMakeLists.txt` — remove placeholder source
- Modify: `docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md` — mark complete
- Create: `docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-results.md` — end-of-phase summary

- [ ] **Step 1: Delete placeholder.cpp and remove from CMake**

- [ ] **Step 2: Write results doc**

Include:
- Final corpus parity count (should be 730/730).
- LOC of frontend lib.
- Estimated effort vs. actual (compare to spec §6 P1 estimate of ~2,500 LOC).
- Any known deviations from JS (ideally none, but file any known deliberate divergences).
- Blockers discovered that affect P2 planning (e.g., "LEGEND pass relies on X which we didn't capture in ParserState — P2 will need to add Y").

- [ ] **Step 3: Commit**

```bash
git add native/src/frontend/placeholder.cpp native/CMakeLists.txt \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-progress.md \
        docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-results.md
git commit -m "frontend: Phase 1 complete — parser + harness live, corpus green"
```

---

## Self-review checklist (for the engineer executing this plan)

Before declaring Phase 1 complete:

- [ ] `scripts/diff_parser_state_against_js.sh --corpus` reports 0 diffs.
- [ ] `scripts/diff_diagnostics_against_js.sh --corpus` reports 0 diffs.
- [ ] All `ctest -R frontend_` tests pass.
- [ ] `cmake --build build` (non-dev config) still succeeds — dev-only serializers are truly gated and don't leak into shipping builds.
- [ ] No changes to `src/js/*` semantics (the only JS edits are additive — `--snapshot-phase` flag + `puzzlescript_parser_snapshot.js`).
- [ ] `native/src/frontend/` matches the File Structure section of this plan (directory layout is conventional).
- [ ] `docs/superpowers/plans/2026-04-22-cpp-frontend-phase1-results.md` documents final LOC, effort, and any P2-relevant findings.
- [ ] Commits are granular (one task = one commit minimum, often more). Each commit builds and tests cleanly.
- [ ] CLAUDE.md and AGENTS.md don't need updates yet — that's Phase 4 (P4 cutover).

## Known scope boundaries

Explicitly **not** in Phase 1:

- Legend cross-reference resolution — that's P2 (`LoweredProgram`).
- Rule mask compilation — that's P3.
- `ps_cli run <source.txt>` — stays on the JS-generated `ir.json` path until P4.
- Any change to engine runtime, `core.cpp`, `core.hpp`, `Game` struct.
- WASM, LSP, VSCode extension (all non-goals per spec §1).

If a task's acceptance criteria seems to require any of the above, stop and escalate — a scope boundary may have been crossed.
