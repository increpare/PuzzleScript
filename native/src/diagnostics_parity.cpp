#include "diagnostics_parity.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "puzzlescript/frontend.h"

namespace {

std::string trimAscii(std::string_view s) {
    size_t begin = 0;
    size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin])) != 0) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])) != 0) {
        --end;
    }
    return std::string(s.substr(begin, end - begin));
}

// Mirrors scripts/compare_parser_phase_diagnostics.js canonicalizeDiagnosticText.
std::string canonicalizeDiagnosticText(std::string raw) {
    try {
        const std::regex brRe(R"(<br\s*/?>)", std::regex::icase);
        raw = std::regex_replace(raw, brRe, "\n");
        const std::regex tagRe(R"(</?[a-zA-Z][^>]*>)");
        raw = std::regex_replace(raw, tagRe, std::string(""));
    } catch (const std::regex_error& error) {
        std::cerr << "canonicalizeDiagnosticText regex error: " << error.what() << "\n";
    }

    std::vector<std::string> lines;
    {
        std::stringstream stream(raw);
        std::string line;
        while (std::getline(stream, line)) {
            lines.push_back(trimAscii(line));
        }
    }
    std::vector<std::string> out;
    for (const auto& line : lines) {
        if (line.empty() && !out.empty() && out.back().empty()) {
            continue;
        }
        out.push_back(line);
    }
    while (!out.empty() && out.back().empty()) {
        out.pop_back();
    }
    while (!out.empty() && out.front().empty()) {
        out.erase(out.begin());
    }
    std::string joined;
    for (size_t index = 0; index < out.size(); ++index) {
        if (index > 0) {
            joined.push_back('\n');
        }
        joined += out[index];
    }
    return joined;
}

std::string requireStringField(const puzzlescript::json::Value::Object& object, const char* key) {
    const auto iterator = object.find(std::string(key));
    if (iterator == object.end() || !iterator->second.isString()) {
        throw std::runtime_error(std::string("bundle record missing string field: ") + key);
    }
    return iterator->second.asString();
}

const puzzlescript::json::Value& requireArrayField(const puzzlescript::json::Value::Object& object, const char* key) {
    const auto iterator = object.find(std::string(key));
    if (iterator == object.end() || !iterator->second.isArray()) {
        throw std::runtime_error(std::string("bundle record missing array field: ") + key);
    }
    return iterator->second;
}

} // namespace

int diagnosticsParityMain(const std::filesystem::path& bundleNdjsonPath) {
    std::ifstream input(bundleNdjsonPath, std::ios::binary);
    if (!input) {
        std::cerr << "Failed to open: " << bundleNdjsonPath.string() << "\n";
        return 1;
    }

    size_t checked = 0;
    size_t passed = 0;
    size_t failed = 0;
    std::string line;
    size_t lineNumber = 0;

    while (std::getline(input, line)) {
        ++lineNumber;
        const std::string trimmed = trimAscii(line);
        if (trimmed.empty()) {
            continue;
        }

        puzzlescript::json::Value root;
        try {
            root = puzzlescript::json::parse(trimmed);
        } catch (const std::exception& error) {
            std::cerr << "bundle parse error at file line " << lineNumber << ": " << error.what() << "\n";
            return 1;
        }
        if (!root.isObject()) {
            std::cerr << "bundle line " << lineNumber << ": expected JSON object\n";
            return 1;
        }
        const auto& object = root.asObject();
        int fixtureIndex = static_cast<int>(checked);
        const auto indexIterator = object.find("index");
        if (indexIterator != object.end() && indexIterator->second.isInteger()) {
            fixtureIndex = static_cast<int>(indexIterator->second.asInteger());
        }
        const std::string name = requireStringField(object, "name");
        std::string source = requireStringField(object, "source");
        if (source.empty() || source.back() != '\n') {
            source.push_back('\n');
        }

        const puzzlescript::json::Value& referenceValue = requireArrayField(object, "reference");
        const auto& referenceArray = referenceValue.asArray();
        std::vector<std::string> expected;
        expected.reserve(referenceArray.size());
        for (const auto& item : referenceArray) {
            if (!item.isString()) {
                throw std::runtime_error("bundle reference[] entries must be strings (fixture: " + name + ")");
            }
            expected.push_back(item.asString());
        }

        std::unique_ptr<ps_frontend_result, decltype(&ps_frontend_result_free)> result(
            ps_frontend_parse(source.data(), source.size()),
            ps_frontend_result_free
        );
        if (!result) {
            std::cerr << "diag_corpus index=" << fixtureIndex << " outcome=native_parse_null name=" << name << "\n";
            ++failed;
            ++checked;
            continue;
        }

        std::vector<std::string> actual;
        const size_t diagnosticCount = ps_frontend_result_diagnostic_count(result.get());
        for (size_t diagnosticIndex = 0; diagnosticIndex < diagnosticCount; ++diagnosticIndex) {
            const ps_diagnostic* diagnostic = ps_frontend_result_diagnostic(result.get(), diagnosticIndex);
            if (diagnostic == nullptr || diagnostic->message == nullptr) {
                continue;
            }
            actual.push_back(canonicalizeDiagnosticText(std::string(diagnostic->message)));
        }

        if (actual.size() != expected.size()) {
            std::cerr << "diag_corpus index=" << fixtureIndex << " outcome=mismatch name=" << name << "\n";
            std::cerr << "diagnostic_count_mismatch reference=" << expected.size() << " native=" << actual.size() << "\n";
            ++failed;
            ++checked;
            continue;
        }
        bool mismatch = false;
        for (size_t index = 0; index < actual.size(); ++index) {
            if (actual[index] != expected[index]) {
                std::cerr << "diag_corpus index=" << fixtureIndex << " outcome=mismatch name=" << name << "\n";
                std::cerr << "diagnostic_mismatch index=" << index << "\n";
                std::cerr << "--- reference (JS parser export)\n";
                std::cerr << expected[index] << "\n";
                std::cerr << "--- native (C++ frontend)\n";
                std::cerr << actual[index] << "\n";
                mismatch = true;
                break;
            }
        }
        if (mismatch) {
            ++failed;
        } else {
            ++passed;
        }
        ++checked;
    }

    std::cout << "diag_corpus checked=" << checked << " passed=" << passed << " failed=" << failed << "\n";
    return failed > 0 ? 1 : 0;
}
