#pragma once

#include <string>
#include <string_view>

#include "frontend/diagnostic.hpp"
#include "frontend/types/parser_state.hpp"

namespace puzzlescript::frontend {

ParserState parseSource(std::string_view source, DiagnosticSink& diagnostics);
std::string serializeParserStateJson(const ParserState& state);

} // namespace puzzlescript::frontend
