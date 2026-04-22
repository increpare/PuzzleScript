#include "frontend/diagnostic.hpp"

namespace puzzlescript::frontend {

void DiagnosticSink::add(Severity severity, DiagnosticCode code, std::optional<int32_t> line, std::string message) {
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
    return diagnostic.message;
}

} // namespace puzzlescript::frontend
