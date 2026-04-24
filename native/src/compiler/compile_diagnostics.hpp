#pragma once

#include <vector>

#include "compiler/diagnostic.hpp"
#include "compiler/types/parser_state.hpp"

namespace puzzlescript::compiler {

void runCompileDiagnostics(
    const ParserState& state,
    std::string_view source,
    const std::vector<Diagnostic>& parserDiagnostics,
    DiagnosticSink& diagnostics
);

} // namespace puzzlescript::compiler
