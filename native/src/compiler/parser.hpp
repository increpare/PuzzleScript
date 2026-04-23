#pragma once

#include <string>
#include <string_view>

#include "compiler/diagnostic.hpp"
#include "compiler/types/parser_state.hpp"

namespace puzzlescript::compiler {

ParserState parseSource(std::string_view source, DiagnosticSink& diagnostics);
std::string serializeParserStateJson(const ParserState& state);

} // namespace puzzlescript::compiler
