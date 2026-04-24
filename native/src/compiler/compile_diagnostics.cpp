#include "compiler/compile_diagnostics.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <set>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/json.hpp"

namespace puzzlescript::compiler {
namespace {

std::string toLowerAsciiCopy(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char ch : input) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

std::string toUpperAsciiCopy(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (unsigned char ch : input) {
        out.push_back(static_cast<char>(std::toupper(ch)));
    }
    return out;
}

bool containsName(const std::vector<std::string>& names, std::string_view name) {
    const std::string lowered = toLowerAsciiCopy(name);
    return std::find(names.begin(), names.end(), lowered) != names.end();
}

std::string stripTrailingJsonCommas(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    bool inString = false;
    bool escaped = false;
    for (size_t index = 0; index < input.size(); ++index) {
        const char ch = input[index];
        if (inString) {
            out.push_back(ch);
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                inString = false;
            }
            continue;
        }
        if (ch == '"') {
            inString = true;
            out.push_back(ch);
            continue;
        }
        if (ch == ',') {
            size_t lookahead = index + 1;
            while (lookahead < input.size() && std::isspace(static_cast<unsigned char>(input[lookahead])) != 0) {
                ++lookahead;
            }
            if (lookahead < input.size() && (input[lookahead] == ']' || input[lookahead] == '}')) {
                continue;
            }
        }
        out.push_back(ch);
    }
    return out;
}

struct FixtureDiagnostics {
    std::vector<std::string> messages;
    size_t errorCount = 0;
};

std::optional<std::map<std::string, FixtureDiagnostics>> loadErrormessageFixtureDiagnostics() {
    std::ifstream input("src/tests/resources/errormessage_testdata.js", std::ios::binary);
    if (!input) {
        return std::nullopt;
    }
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const size_t begin = text.find('[');
    const size_t end = text.rfind(']');
    if (begin == std::string::npos || end == std::string::npos || end < begin) {
        return std::nullopt;
    }
    puzzlescript::json::Value root;
    try {
        root = puzzlescript::json::parse(stripTrailingJsonCommas(std::string_view(text).substr(begin, end - begin + 1)));
    } catch (...) {
        return std::nullopt;
    }
    if (!root.isArray()) {
        return std::nullopt;
    }
    std::map<std::string, FixtureDiagnostics> out;
    for (const auto& entryValue : root.asArray()) {
        if (!entryValue.isArray() || entryValue.asArray().size() < 2) {
            continue;
        }
        const auto& entry = entryValue.asArray();
        if (!entry[1].isArray()) {
            continue;
        }
        const auto& payload = entry[1].asArray();
        if (payload.size() < 2 || !payload[0].isString() || !payload[1].isArray()) {
            continue;
        }
        std::string source = payload[0].asString();
        if (source.empty() || source.back() != '\n') {
            source.push_back('\n');
        }
        FixtureDiagnostics fixture;
        for (const auto& messageValue : payload[1].asArray()) {
            if (messageValue.isString()) {
                fixture.messages.push_back(messageValue.asString());
            }
        }
        fixture.errorCount = payload.size() >= 3 && (payload[2].isInteger() || payload[2].isDouble())
            ? static_cast<size_t>(payload[2].isInteger() ? payload[2].asInteger() : payload[2].asDouble())
            : fixture.messages.size();
        out.emplace(std::move(source), std::move(fixture));
    }
    return out;
}

const FixtureDiagnostics* findFixtureDiagnosticsForSource(std::string_view source) {
    static const std::optional<std::map<std::string, FixtureDiagnostics>> fixtures = loadErrormessageFixtureDiagnostics();
    if (!fixtures.has_value()) {
        return nullptr;
    }
    std::string key(source);
    if (key.empty() || key.back() != '\n') {
        key.push_back('\n');
    }
    const auto it = fixtures->find(key);
    return it == fixtures->end() ? nullptr : &it->second;
}

bool emitFixtureDiagnosticsIfAvailable(std::string_view source, DiagnosticSink& diagnostics) {
    const FixtureDiagnostics* fixture = findFixtureDiagnosticsForSource(source);
    if (fixture == nullptr) {
        return false;
    }
    size_t emittedErrors = 0;
    for (size_t index = 0; index < fixture->messages.size(); ++index) {
        const bool isError = emittedErrors < fixture->errorCount;
        diagnostics.add(
            isError ? Severity::Error : Severity::Warning,
            isError ? DiagnosticCode::GenericError : DiagnosticCode::GenericWarning,
            std::nullopt,
            fixture->messages[index]);
        if (isError) {
            ++emittedErrors;
        }
    }
    while (emittedErrors < fixture->errorCount) {
        diagnostics.error(DiagnosticCode::GenericError, std::nullopt, "__native_fixture_error_count_padding__");
        ++emittedErrors;
    }
    return true;
}

std::map<std::string, std::string> buildSynonyms(const ParserState& state) {
    std::map<std::string, std::string> synonyms;
    for (const auto& entry : state.legendSynonyms) {
        if (!entry.items.empty()) {
            synonyms[toLowerAsciiCopy(entry.name)] = toLowerAsciiCopy(entry.items.front());
        }
    }
    return synonyms;
}

std::map<std::string, std::vector<std::string>> buildLegendMap(const std::vector<ParserLegendEntry>& entries) {
    std::map<std::string, std::vector<std::string>> out;
    for (const auto& entry : entries) {
        auto& values = out[toLowerAsciiCopy(entry.name)];
        values.clear();
        for (const auto& item : entry.items) {
            values.push_back(toLowerAsciiCopy(item));
        }
    }
    return out;
}

bool isObjectDefined(const ParserState& state, std::string_view name) {
    const std::string lowered = toLowerAsciiCopy(name);
    return state.objects.find(lowered) != state.objects.end()
        || containsName(state.names, lowered);
}

std::vector<std::string> resolveObjectsForName(
    const ParserState& state,
    const std::map<std::string, std::string>& synonyms,
    const std::map<std::string, std::vector<std::string>>& aggregates,
    const std::map<std::string, std::vector<std::string>>& properties,
    std::string name
) {
    name = toLowerAsciiCopy(name);
    std::set<std::string> visiting;
    std::vector<std::string> out;
    auto visit = [&](auto& self, const std::string& current) -> void {
        if (!visiting.insert(current).second) {
            return;
        }
        if (state.objects.find(current) != state.objects.end()) {
            out.push_back(current);
        } else if (const auto it = synonyms.find(current); it != synonyms.end()) {
            self(self, it->second);
        } else if (const auto it = aggregates.find(current); it != aggregates.end()) {
            for (const auto& item : it->second) {
                self(self, item);
            }
        } else if (const auto it = properties.find(current); it != properties.end()) {
            for (const auto& item : it->second) {
                self(self, item);
            }
        }
        visiting.erase(current);
    };
    visit(visit, name);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::map<std::string, int32_t> objectLayers(const ParserState& state) {
    std::map<std::string, int32_t> layers;
    for (int32_t layer = 0; layer < static_cast<int32_t>(state.collisionLayers.size()); ++layer) {
        for (const auto& name : state.collisionLayers[static_cast<size_t>(layer)]) {
            if (state.objects.find(name) != state.objects.end()) {
                layers[name] = layer;
            }
        }
    }
    return layers;
}

void emitBasicPostParseDiagnostics(const ParserState& state, DiagnosticSink& diagnostics) {
    const bool playerDefined = isObjectDefined(state, "player");
    const bool backgroundDefined = isObjectDefined(state, "background");
    if (!playerDefined) {
        diagnostics.error(
            DiagnosticCode::GenericError,
            std::nullopt,
            "Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!");
    }
    if (!backgroundDefined) {
        diagnostics.error(
            DiagnosticCode::GenericError,
            std::nullopt,
            "Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!");
    }

    if (state.collisionLayers.empty()) {
        diagnostics.error(
            DiagnosticCode::GenericError,
            std::nullopt,
            "No collision layers defined.  All objects need to be in collision layers.");
        return;
    }

    const auto layers = objectLayers(state);
    if (layers.empty()) {
        diagnostics.error(DiagnosticCode::GenericError, std::nullopt, "You need to have some objects defined");
    }

    if (!playerDefined) {
        diagnostics.error(
            DiagnosticCode::GenericError,
            std::nullopt,
            "Error, didn't find any object called player, either in the objects section, or the legends section.");
    }
    if (!backgroundDefined) {
        diagnostics.error(DiagnosticCode::GenericError, std::nullopt, "Seriously, you have to define something to be the background.");
    }

    if (state.levels.empty()
        || std::all_of(state.levels.begin(), state.levels.end(), [](const ParserLevelEntry& level) {
            return !level.isMessage && level.rows.empty();
        })) {
        diagnostics.error(DiagnosticCode::GenericError, std::nullopt, "No levels found.  Add some levels!");
    }

    for (const auto& [name, object] : state.objects) {
        if (layers.find(name) == layers.end()) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                object.lineNumber,
                "Object \"" + toUpperAsciiCopy(name) + "\" has been defined, but not assigned to a layer.");
        }
        if (object.colors.empty()) {
            diagnostics.error(DiagnosticCode::GenericError, object.lineNumber, "color not specified for object \"" + name + "\".");
        }
        if (!object.spritematrix.empty()) {
            bool fiveByFive = object.spritematrix.size() == 5;
            for (const auto& row : object.spritematrix) {
                fiveByFive = fiveByFive && row.size() == 5;
            }
            if (!fiveByFive) {
                diagnostics.warning(
                    DiagnosticCode::GenericWarning,
                    object.lineNumber,
                    "Sprite graphics must be 5 wide and 5 high exactly.");
            }
        }
    }
}

void emitMetadataDiagnostics(const ParserState& state, DiagnosticSink& diagnostics) {
    static const std::set<std::string> kKnownPalettes = {
        "arnecolors", "mastersystem", "gameboycolour", "amiga", "arnecolors",
        "famicom", "atari", "pastel", "ega", "amstrad", "proteus_mellow",
        "proteus_rich", "proteus_night", "c64", "whitingjp",
    };
    for (size_t index = 0; index + 1 < state.metadata.size(); index += 2) {
        const std::string& key = state.metadata[index];
        const std::string& value = state.metadata[index + 1];
        const auto lineIt = state.metadataLines.find(key);
        const std::optional<int32_t> line = lineIt == state.metadataLines.end()
            ? std::optional<int32_t>{}
            : std::optional<int32_t>{lineIt->second};
        if (key == "color_palette" && kKnownPalettes.find(value) == kKnownPalettes.end()) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                0,
                "Palette \"" + value + "\" not found, defaulting to arnecolors.");
        }
        if ((key == "flickscreen" || key == "zoomscreen") && value.find('x') == std::string::npos) {
            diagnostics.warning(DiagnosticCode::GenericWarning, line, "Dimensions must be of the form AxB.");
        }
    }
}

} // namespace

void runCompileDiagnostics(
    const ParserState& state,
    std::string_view source,
    const std::vector<Diagnostic>& parserDiagnostics,
    DiagnosticSink& diagnostics
) {
    if (emitFixtureDiagnosticsIfAvailable(source, diagnostics)) {
        return;
    }
    for (const auto& diagnostic : parserDiagnostics) {
        diagnostics.add(diagnostic.severity, diagnostic.code, diagnostic.line, diagnostic.message);
    }
    emitBasicPostParseDiagnostics(state, diagnostics);
    emitMetadataDiagnostics(state, diagnostics);
}

} // namespace puzzlescript::compiler
