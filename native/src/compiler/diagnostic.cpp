#include "compiler/diagnostic.hpp"

namespace puzzlescript::compiler {

void DiagnosticSink::add(Severity severity, DiagnosticCode code, std::optional<int32_t> line, std::string message) {
    // parser.js logError/logWarning: identical HTML diagnostics are suppressed (see duplicate checks).
    for (const auto& existing : diagnostics_) {
        if (existing.severity == severity && existing.line == line && existing.message == message) {
            return;
        }
    }
    diagnostics_.push_back(Diagnostic{
        severity,
        code,
        line,
        std::move(message),
    });
}

void DiagnosticSink::error(DiagnosticCode code, std::optional<int32_t> line, std::string message) {
    add(Severity::Error, code, line, std::move(message));
}

void DiagnosticSink::warning(DiagnosticCode code, std::optional<int32_t> line, std::string message) {
    add(Severity::Warning, code, line, std::move(message));
}

void DiagnosticSink::info(DiagnosticCode code, std::optional<int32_t> line, std::string message) {
    add(Severity::Info, code, line, std::move(message));
}

std::string formatForJsCompat(const Diagnostic& diagnostic) {
    // Mirror `buildErrorHtml` / `buildWarningHtml` after `stripHTMLTags` in the node harness:
    // anchored diagnostics become `line N : <message>`.
    if (diagnostic.line.has_value()) {
        return "line " + std::to_string(*diagnostic.line) + " : " + diagnostic.message;
    }
    return diagnostic.message;
}

} // namespace puzzlescript::compiler
