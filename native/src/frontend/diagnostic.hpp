#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace puzzlescript::frontend {

enum class Severity : int32_t {
    Error = 0,
    Warning = 1,
    Info = 2,
    LogMessage = 3,
};

enum class DiagnosticCode : int32_t {
    GenericError = 0,
    GenericWarning = 1,
    InternalNote = 2,
};

struct Diagnostic {
    Severity severity = Severity::Info;
    DiagnosticCode code = DiagnosticCode::InternalNote;
    std::optional<int32_t> line;
    std::string message;
};

class DiagnosticSink {
public:
    void add(Severity severity, DiagnosticCode code, std::optional<int32_t> line, std::string message);
    void error(DiagnosticCode code, std::optional<int32_t> line, std::string message);
    void warning(DiagnosticCode code, std::optional<int32_t> line, std::string message);
    void info(DiagnosticCode code, std::optional<int32_t> line, std::string message);

    const std::vector<Diagnostic>& diagnostics() const { return diagnostics_; }

private:
    std::vector<Diagnostic> diagnostics_;
};

std::string formatForJsCompat(const Diagnostic& diagnostic);

} // namespace puzzlescript::frontend
