#pragma once

#include <memory>

#include "compiler/types/parser_state.hpp"
#include "runtime/core.hpp"

namespace puzzlescript::compiler {

// Lower a parsed PuzzleScript program into a runnable native runtime Game.
//
// This is the missing “native compiler” stage: ParserState -> runtime::Game.
// It must preserve JS semantics (including RNG behavior and rule ordering) so
// the existing JS test corpus can be used as a correctness gate.
std::unique_ptr<puzzlescript::Error> lowerToRuntimeGame(
    const ParserState& state,
    std::shared_ptr<const puzzlescript::Game>& outGame
);

} // namespace puzzlescript::compiler

