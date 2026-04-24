#include "compiler/compile_diagnostics.hpp"
#include "compiler/parser.hpp"
#include "puzzlescript/compiler.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct ps_compiler_result {
    puzzlescript::compiler::ParserState parserState;
    std::vector<puzzlescript::compiler::Diagnostic> diagnostics;
    std::vector<ps_diagnostic> exportedDiagnostics;
    std::vector<std::string> exportedMessages;
    std::string parserStateJson;
};

namespace {

ps_diagnostic_severity toCSeverity(puzzlescript::compiler::Severity severity) {
    switch (severity) {
        case puzzlescript::compiler::Severity::Error: return PS_DIAG_ERROR;
        case puzzlescript::compiler::Severity::Warning: return PS_DIAG_WARNING;
        case puzzlescript::compiler::Severity::Info: return PS_DIAG_INFO;
        case puzzlescript::compiler::Severity::LogMessage: return PS_DIAG_LOG;
    }
    return PS_DIAG_INFO;
}

} // namespace

ps_compiler_result* makeCompilerResult(const char* source_utf8, size_t source_size, bool fullDiagnostics) {
    auto* result = new ps_compiler_result();
    puzzlescript::compiler::DiagnosticSink sink;
    const std::string_view sourceView = source_utf8 == nullptr ? std::string_view{} : std::string_view(source_utf8, source_size);
    result->parserState = puzzlescript::compiler::parseSource(
        sourceView,
        sink
    );
    if (fullDiagnostics) {
        puzzlescript::compiler::DiagnosticSink fullSink;
        puzzlescript::compiler::runCompileDiagnostics(result->parserState, sourceView, sink.diagnostics(), fullSink);
        result->diagnostics = fullSink.diagnostics();
    } else {
        result->diagnostics = sink.diagnostics();
    }
    result->parserStateJson = puzzlescript::compiler::serializeParserStateJson(result->parserState);
    result->exportedMessages.reserve(result->diagnostics.size());
    result->exportedDiagnostics.reserve(result->diagnostics.size());
    for (const auto& diagnostic : result->diagnostics) {
        result->exportedMessages.push_back(puzzlescript::compiler::formatForJsCompat(diagnostic));
        result->exportedDiagnostics.push_back(ps_diagnostic{
            toCSeverity(diagnostic.severity),
            static_cast<int32_t>(diagnostic.code),
            diagnostic.line.value_or(-1),
            result->exportedMessages.back().c_str(),
        });
    }
    return result;
}

ps_compiler_result* ps_compiler_parse_source(const char* source_utf8, size_t source_size) {
    return makeCompilerResult(source_utf8, source_size, false);
}

ps_compiler_result* ps_compiler_compile_source_diagnostics(const char* source_utf8, size_t source_size) {
    return makeCompilerResult(source_utf8, source_size, true);
}

size_t ps_compiler_result_diagnostic_count(const ps_compiler_result* result) {
    return result == nullptr ? 0 : result->exportedDiagnostics.size();
}

const ps_diagnostic* ps_compiler_result_diagnostic(const ps_compiler_result* result, size_t index) {
    if (result == nullptr || index >= result->exportedDiagnostics.size()) {
        return nullptr;
    }
    return &result->exportedDiagnostics[index];
}

size_t ps_compiler_result_parser_state_json(
    const ps_compiler_result* result,
    char* out_buffer,
    size_t out_buffer_capacity
) {
    if (result == nullptr) {
        return 0;
    }
    const size_t required = result->parserStateJson.size() + 1;
    if (out_buffer != nullptr && out_buffer_capacity > 0) {
        const size_t copySize = std::min(required, out_buffer_capacity) - 1;
        std::memcpy(out_buffer, result->parserStateJson.data(), copySize);
        out_buffer[copySize] = '\0';
    }
    return required;
}

void ps_compiler_result_free(ps_compiler_result* result) {
    delete result;
}
