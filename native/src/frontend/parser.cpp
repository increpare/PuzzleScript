#include "frontend/parser.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

namespace puzzlescript::frontend {
namespace {

constexpr std::string_view kSections[] = {
    "objects",
    "legend",
    "sounds",
    "collisionlayers",
    "rules",
    "winconditions",
    "levels",
};

constexpr std::string_view kPreambleValues[] = {
    "title",
    "author",
    "homepage",
    "background_color",
    "text_color",
    "key_repeat_interval",
    "realtime_interval",
    "again_interval",
    "flickscreen",
    "zoomscreen",
    "color_palette",
    "youtube",
};

constexpr std::string_view kPreambleFlags[] = {
    "run_rules_on_level_start",
    "norepeat_action",
    "require_player_movement",
    "debug",
    "verbose_logging",
    "throttle_movement",
    "noundo",
    "noaction",
    "norestart",
    "scanline",
};

constexpr std::string_view kSoundEvents[] = {
    "titlescreen", "startgame", "cancel", "endgame", "startlevel", "undo", "restart", "endlevel", "showmessage", "closemessage", "sfx0", "sfx1",
    "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9",
};

constexpr std::string_view kSoundVerbs[] = {
    "move",
    "action",
    "create",
    "destroy",
    "cantmove",
};

constexpr std::string_view kSoundDirections[] = {
    "up",
    "down",
    "left",
    "right",
    "horizontal",
    "vertical",
    "orthogonal",
};

std::string trim(std::string_view value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(start, end - start));
}

std::string toLowerCopy(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return out;
}

bool isIdentifierLike(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char ch) {
        return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
    });
}

bool isExactToken(std::string_view value, const std::string_view* tokens, size_t count) {
    return std::find(tokens, tokens + count, value) != (tokens + count);
}

bool isSpriteRow(std::string_view value) {
    if (value.empty() || value.size() > 5) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char ch) {
        return ch == '.' || std::isdigit(static_cast<unsigned char>(ch)) != 0;
    });
}

bool isAllEquals(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char ch) { return ch == '='; });
}

bool isNumeric(std::string_view value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](char ch) {
        return std::isdigit(static_cast<unsigned char>(ch)) != 0;
    });
}

std::vector<std::string> splitWhitespace(std::string_view value) {
    std::vector<std::string> result;
    std::stringstream stream{std::string(value)};
    std::string token;
    while (stream >> token) {
        result.push_back(token);
    }
    return result;
}

std::vector<std::string> splitCsvWords(std::string_view value) {
    std::vector<std::string> result;
    std::string current;
    for (const char ch : value) {
        if (ch == ',' || std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!current.empty()) {
                result.push_back(toLowerCopy(current));
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) {
        result.push_back(toLowerCopy(current));
    }
    return result;
}

void registerOriginalCaseName(ParserState& state, std::string_view lowered, std::string_view original, int32_t lineNumber) {
    if (!isIdentifierLike(lowered)) {
        return;
    }
    state.originalCaseNames[std::string(lowered)] = std::string(original);
    state.originalLineNumbers[std::string(lowered)] = lineNumber;
}

void appendUnique(std::vector<std::string>& values, const std::string& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::string stripComments(std::string_view line, ParserState& state) {
    std::string visible;
    visible.reserve(line.size());
    for (size_t index = 0; index < line.size(); ++index) {
        const char ch = line[index];
        if (ch == '(') {
            ++state.commentLevel;
            continue;
        }
        if (ch == ')' && state.commentLevel > 0) {
            --state.commentLevel;
            continue;
        }
        if (state.commentLevel == 0) {
            visible.push_back(ch);
        }
    }
    return visible;
}

void populateNamesForSounds(ParserState& state) {
    state.names.clear();
    std::vector<const ParserObjectEntry*> orderedObjects;
    orderedObjects.reserve(state.objects.size());
    for (const auto& [name, object] : state.objects) {
        (void)name;
        orderedObjects.push_back(&object);
    }
    std::sort(orderedObjects.begin(), orderedObjects.end(), [](const ParserObjectEntry* lhs, const ParserObjectEntry* rhs) {
        if (lhs->lineNumber != rhs->lineNumber) {
            return lhs->lineNumber < rhs->lineNumber;
        }
        return lhs->name < rhs->name;
    });
    for (const ParserObjectEntry* object : orderedObjects) {
        state.names.push_back(object->name);
    }
    for (const auto& entry : state.legendSynonyms) {
        state.names.push_back(entry.name);
    }
    for (const auto& entry : state.legendAggregates) {
        state.names.push_back(entry.name);
    }
    for (const auto& entry : state.legendProperties) {
        state.names.push_back(entry.name);
    }
}

void populateAbbrevNamesForLevels(ParserState& state) {
    state.abbrevNames.clear();
    for (const auto& [name, _object] : state.objects) {
        if (name.size() == 1) {
            state.abbrevNames.push_back(name);
        }
    }
    for (const auto& entry : state.legendSynonyms) {
        if (entry.name.size() == 1) {
            state.abbrevNames.push_back(entry.name);
        }
    }
    for (const auto& entry : state.legendAggregates) {
        if (entry.name.size() == 1) {
            state.abbrevNames.push_back(entry.name);
        }
    }
}

void handleBlankLine(ParserState& state) {
    if (state.section == "levels") {
        if (state.levels.empty() || !state.levels.back().rows.empty() || state.levels.back().isMessage) {
            state.levels.push_back(ParserLevelEntry{});
        }
    } else if (state.section == "objects") {
        state.objectsSection = 0;
    }
}

void parsePreambleLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine, std::string_view mixedCase) {
    const auto tokens = splitWhitespace(trimmedLine);
    if (tokens.empty()) {
        return;
    }
    const std::string key = toLowerCopy(tokens.front());
    const std::string originalKey = tokens.front();
    const size_t keyOffset = mixedCase.find(originalKey);
    const std::string remainder = keyOffset == std::string_view::npos ? std::string{} : trim(mixedCase.substr(keyOffset + originalKey.size()));

    if (isExactToken(key, kPreambleValues, std::size(kPreambleValues))) {
        if (remainder.empty()) {
            diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "MetaData \"" + key + "\" needs a value.");
            return;
        }
        state.metadata.push_back(key);
        state.metadata.push_back(remainder);
        state.metadataLines[key] = state.lineNumber;
        return;
    }
    if (isExactToken(key, kPreambleFlags, std::size(kPreambleFlags))) {
        state.metadata.push_back(key);
        state.metadata.push_back("true");
        state.metadataLines[key] = state.lineNumber;
        return;
    }
    diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Unrecognised stuff in the prelude.");
}

void parseObjectsLine(ParserState& state, std::string_view trimmedLine, std::string_view mixedCase) {
    const std::string loweredLine = toLowerCopy(trimmedLine);
    if (state.objectsSection == 0) {
        state.objectsCandname = loweredLine;
        state.objectsSection = 2;
        state.objectsSpritematrix.clear();
        auto& object = state.objects[loweredLine];
        object.name = loweredLine;
        object.lineNumber = state.lineNumber;
        registerOriginalCaseName(state, loweredLine, trim(mixedCase), state.lineNumber);
        return;
    }
    if (state.objectsSection == 2) {
        auto& object = state.objects[state.objectsCandname];
        object.colors = splitCsvWords(trimmedLine);
        state.objectsSection = 3;
        return;
    }
    if (state.objectsSection == 3 && isSpriteRow(trimmedLine)) {
        auto& object = state.objects[state.objectsCandname];
        object.spritematrix.push_back(std::string(trimmedLine));
        state.objectsSpritematrix = object.spritematrix;
        if (object.spritematrix.size() >= 5) {
            state.objectsSection = 0;
        }
        return;
    }

    state.objectsSection = 0;
    parseObjectsLine(state, trimmedLine, mixedCase);
}

void parseLegendLine(ParserState& state, std::string_view trimmedLine, std::string_view mixedCase) {
    const auto tokens = splitWhitespace(trimmedLine);
    if (tokens.size() < 3 || tokens[1] != "=") {
        return;
    }

    ParserLegendEntry entry;
    entry.name = toLowerCopy(tokens[0]);
    entry.lineNumber = state.lineNumber;
    if (entry.name.size() == 1 && std::isalpha(static_cast<unsigned char>(entry.name[0])) != 0) {
        registerOriginalCaseName(state, entry.name, entry.name, state.lineNumber);
    } else {
        registerOriginalCaseName(state, entry.name, tokens[0], state.lineNumber);
    }

    if (tokens.size() == 3) {
        entry.items.push_back(toLowerCopy(tokens[2]));
        state.legendSynonyms.push_back(std::move(entry));
        return;
    }

    std::string joiner;
    if (tokens.size() >= 4) {
        joiner = toLowerCopy(tokens[3]);
    }
    for (size_t index = 2; index < tokens.size(); ++index) {
        if ((index % 2) == 0) {
            entry.items.push_back(toLowerCopy(tokens[index]));
        }
    }
    if (joiner == "and") {
        state.legendAggregates.push_back(std::move(entry));
    } else {
        state.legendProperties.push_back(std::move(entry));
    }
}

std::string classifySoundKind(std::string_view lowered, size_t index) {
    if (isNumeric(lowered)) {
        return "SOUND";
    }
    if (isExactToken(lowered, kSoundEvents, std::size(kSoundEvents))) {
        return "SOUNDEVENT";
    }
    if (isExactToken(lowered, kSoundVerbs, std::size(kSoundVerbs))) {
        return "SOUNDVERB";
    }
    if (isExactToken(lowered, kSoundDirections, std::size(kSoundDirections))) {
        return "DIRECTION";
    }
    return index == 0 ? "NAME" : "NAME";
}

void parseSoundsLine(ParserState& state, std::string_view trimmedLine) {
    ParserSoundEntry entry;
    entry.lineNumber = state.lineNumber;
    const auto tokens = splitWhitespace(trimmedLine);
    for (size_t index = 0; index < tokens.size(); ++index) {
        const std::string lowered = toLowerCopy(tokens[index]);
        entry.tokens.push_back(ParserSoundToken{
            lowered,
            classifySoundKind(lowered, index),
        });
    }
    if (!entry.tokens.empty()) {
        state.sounds.push_back(std::move(entry));
    }
}

void parseCollisionLayersLine(ParserState& state, std::string_view trimmedLine) {
    auto items = splitCsvWords(trimmedLine);
    if (!items.empty()) {
        state.collisionLayers.push_back(std::move(items));
    }
}

void parseRulesLine(ParserState& state, std::string_view trimmedLine, std::string_view mixedCase) {
    state.rulePrelude = false;
    state.arrowPassed = trimmedLine.find("->") != std::string_view::npos;
    state.rules.push_back(ParserRuleEntry{
        toLowerCopy(mixedCase),
        state.lineNumber,
        std::string(mixedCase),
    });
}

void parseWinConditionsLine(ParserState& state, std::string_view trimmedLine) {
    ParserWinConditionEntry entry;
    entry.lineNumber = state.lineNumber;
    for (const auto& token : splitWhitespace(trimmedLine)) {
        entry.tokens.push_back(toLowerCopy(token));
    }
    if (!entry.tokens.empty()) {
        state.winconditions.push_back(std::move(entry));
    }
}

void parseLevelsLine(ParserState& state, std::string_view trimmedLine, std::string_view mixedCase) {
    const std::string trimmedMixed = trim(mixedCase);
    const std::string trimmedLower = toLowerCopy(trimmedLine);
    if (trimmedLower.rfind("message", 0) == 0 && (trimmedLower.size() == 7 || std::isspace(static_cast<unsigned char>(trimmedLine[7])) != 0)) {
        ParserLevelEntry entry;
        entry.isMessage = true;
        entry.lineNumber = state.lineNumber;
        entry.message = trim(trimmedMixed.substr(7));
        if (!state.levels.empty() && state.levels.back().rows.empty() && !state.levels.back().isMessage) {
            state.levels.insert(state.levels.end() - 1, std::move(entry));
        } else {
            state.levels.push_back(std::move(entry));
        }
        return;
    }

    if (state.levels.empty()) {
        state.levels.push_back(ParserLevelEntry{});
    }
    if (state.levels.back().isMessage || (!state.levels.back().rows.empty() && state.levels.back().lineNumber == std::nullopt)) {
        state.levels.push_back(ParserLevelEntry{});
    }
    ParserLevelEntry& level = state.levels.back();
    if (!level.lineNumber.has_value()) {
        level.lineNumber = state.lineNumber;
    }
    level.rows.push_back(trimmedLower);
}

std::string escapeJson(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                out.push_back(ch);
                break;
        }
    }
    return out;
}

void appendIndent(std::string& out, int indent) {
    out.append(static_cast<size_t>(indent * 2), ' ');
}

void appendJsonString(std::string& out, std::string_view value) {
    out.push_back('"');
    out += escapeJson(value);
    out.push_back('"');
}

void appendJsonStringArray(std::string& out, const std::vector<std::string>& values, int indent) {
    out += "[";
    if (!values.empty()) {
        out += "\n";
        for (size_t index = 0; index < values.size(); ++index) {
            appendIndent(out, indent + 1);
            appendJsonString(out, values[index]);
            if (index + 1 != values.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, indent);
    }
    out += "]";
}

void appendJsonStringMap(std::string& out, const std::map<std::string, std::string>& values, int indent) {
    out += "{";
    if (!values.empty()) {
        out += "\n";
        size_t index = 0;
        for (const auto& [key, value] : values) {
            appendIndent(out, indent + 1);
            appendJsonString(out, key);
            out += ": ";
            appendJsonString(out, value);
            if (++index != values.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, indent);
    }
    out += "}";
}

void appendJsonIntMap(std::string& out, const std::map<std::string, int32_t>& values, int indent) {
    out += "{";
    if (!values.empty()) {
        out += "\n";
        size_t index = 0;
        for (const auto& [key, value] : values) {
            appendIndent(out, indent + 1);
            appendJsonString(out, key);
            out += ": ";
            out += std::to_string(value);
            if (++index != values.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, indent);
    }
    out += "}";
}

} // namespace

ParserState parseSource(std::string_view source, DiagnosticSink& diagnostics) {
    ParserState state;
    state.levels.push_back(ParserLevelEntry{});

    std::vector<std::string> lines;
    {
        size_t start = 0;
        for (size_t index = 0; index <= source.size(); ++index) {
            if (index == source.size() || source[index] == '\n') {
                lines.emplace_back(source.substr(start, index - start));
                start = index + 1;
            }
        }
    }

    for (const std::string& rawLine : lines) {
        ++state.lineNumber;
        const std::string mixedVisible = stripComments(rawLine, state);
        const std::string trimmedVisible = trim(mixedVisible);
        const std::string loweredVisible = toLowerCopy(trimmedVisible);

        if (trimmedVisible.empty() || isAllEquals(trimmedVisible)) {
            if (isAllEquals(trimmedVisible)) {
                state.lineShouldEnd = false;
                state.lineShouldEndBecause = "a bunch of equals signs ('===')";
            } else {
                handleBlankLine(state);
            }
            continue;
        }

        if (std::find(std::begin(kSections), std::end(kSections), loweredVisible) != std::end(kSections)) {
            state.section = loweredVisible;
            state.lineShouldEnd = true;
            state.lineShouldEndBecause = "a section name (\"" + std::string(loweredVisible) + "\")";
            appendUnique(state.visitedSections, loweredVisible);
            if (state.section == "sounds") {
                populateNamesForSounds(state);
            } else if (state.section == "levels") {
                populateAbbrevNamesForLevels(state);
            } else if (state.section == "objects") {
                state.objectsSection = 0;
            }
            continue;
        }

        state.lineShouldEnd = false;
        if (state.section.empty()) {
            parsePreambleLine(state, diagnostics, trimmedVisible, trim(mixedVisible));
            continue;
        }

        switch (state.section[0]) {
            case 'o':
                parseObjectsLine(state, trimmedVisible, trim(mixedVisible));
                break;
            case 'l':
                if (state.section == "legend") {
                    parseLegendLine(state, trimmedVisible, trim(mixedVisible));
                } else {
                    parseLevelsLine(state, trimmedVisible, trim(mixedVisible));
                }
                break;
            case 's':
                parseSoundsLine(state, trimmedVisible);
                break;
            case 'c':
                parseCollisionLayersLine(state, trimmedVisible);
                break;
            case 'r':
                parseRulesLine(state, trimmedVisible, mixedVisible);
                break;
            case 'w':
                parseWinConditionsLine(state, trimmedVisible);
                break;
            default:
                break;
        }
    }

    return state;
}

std::string serializeParserStateJson(const ParserState& state) {
    std::string out;
    out.reserve(16384);
    out += "{\n";
    appendIndent(out, 1);
    appendJsonString(out, "schema_version");
    out += ": 1,\n";
    appendIndent(out, 1);
    appendJsonString(out, "parser_state");
    out += ": {\n";

    auto appendFieldPrefix = [&](std::string_view name) {
        appendIndent(out, 2);
        appendJsonString(out, name);
        out += ": ";
    };

    appendFieldPrefix("line_number");
    out += std::to_string(state.lineNumber) + ",\n";
    appendFieldPrefix("comment_level");
    out += std::to_string(state.commentLevel) + ",\n";
    appendFieldPrefix("section");
    appendJsonString(out, state.section);
    out += ",\n";
    appendFieldPrefix("visited_sections");
    appendJsonStringArray(out, state.visitedSections, 2);
    out += ",\n";
    appendFieldPrefix("line_should_end");
    out += state.lineShouldEnd ? "true,\n" : "false,\n";
    appendFieldPrefix("line_should_end_because");
    appendJsonString(out, state.lineShouldEndBecause);
    out += ",\n";
    appendFieldPrefix("sol_after_comment");
    out += state.solAfterComment ? "true,\n" : "false,\n";
    appendFieldPrefix("inside_cell");
    out += state.insideCell ? "true,\n" : "false,\n";
    appendFieldPrefix("bracket_balance");
    out += std::to_string(state.bracketBalance) + ",\n";
    appendFieldPrefix("arrow_passed");
    out += state.arrowPassed ? "true,\n" : "false,\n";
    appendFieldPrefix("rule_prelude");
    out += state.rulePrelude ? "true,\n" : "false,\n";
    appendFieldPrefix("objects_candname");
    appendJsonString(out, state.objectsCandname);
    out += ",\n";
    appendFieldPrefix("objects_section");
    out += std::to_string(state.objectsSection) + ",\n";
    appendFieldPrefix("objects_spritematrix");
    appendJsonStringArray(out, state.objectsSpritematrix, 2);
    out += ",\n";
    appendFieldPrefix("collision_layers");
    out += "[";
    if (!state.collisionLayers.empty()) {
        out += "\n";
        for (size_t index = 0; index < state.collisionLayers.size(); ++index) {
            appendIndent(out, 3);
            appendJsonStringArray(out, state.collisionLayers[index], 3);
            if (index + 1 != state.collisionLayers.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, 2);
    }
    out += "],\n";
    appendFieldPrefix("token_index");
    out += std::to_string(state.tokenIndex) + ",\n";
    appendFieldPrefix("current_line_wip_array");
    appendJsonStringArray(out, state.currentLineWipArray, 2);
    out += ",\n";
    appendFieldPrefix("metadata_pairs");
    appendJsonStringArray(out, state.metadata, 2);
    out += ",\n";
    appendFieldPrefix("metadata_lines");
    appendJsonIntMap(out, state.metadataLines, 2);
    out += ",\n";
    appendFieldPrefix("original_case_names");
    appendJsonStringMap(out, state.originalCaseNames, 2);
    out += ",\n";
    appendFieldPrefix("original_line_numbers");
    appendJsonIntMap(out, state.originalLineNumbers, 2);
    out += ",\n";
    appendFieldPrefix("names");
    appendJsonStringArray(out, state.names, 2);
    out += ",\n";
    appendFieldPrefix("abbrev_names");
    appendJsonStringArray(out, state.abbrevNames, 2);
    out += ",\n";
    appendFieldPrefix("objects");
    out += "[";
    if (!state.objects.empty()) {
        out += "\n";
        size_t index = 0;
        for (const auto& [name, object] : state.objects) {
            appendIndent(out, 3);
            out += "{\n";
            appendIndent(out, 4); appendJsonString(out, "name"); out += ": "; appendJsonString(out, name); out += ",\n";
            appendIndent(out, 4); appendJsonString(out, "line_number"); out += ": " + std::to_string(object.lineNumber) + ",\n";
            appendIndent(out, 4); appendJsonString(out, "colors"); out += ": "; appendJsonStringArray(out, object.colors, 4); out += ",\n";
            appendIndent(out, 4); appendJsonString(out, "spritematrix"); out += ": "; appendJsonStringArray(out, object.spritematrix, 4); out += "\n";
            appendIndent(out, 3);
            out += "}";
            if (++index != state.objects.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, 2);
    }
    out += "],\n";

    auto appendLegendArray = [&](std::string_view fieldName, const std::vector<ParserLegendEntry>& entries) {
        appendFieldPrefix(fieldName);
        out += "[";
        if (!entries.empty()) {
            out += "\n";
            for (size_t index = 0; index < entries.size(); ++index) {
                const auto& entry = entries[index];
                appendIndent(out, 3);
                out += "{\n";
                appendIndent(out, 4); appendJsonString(out, "name"); out += ": "; appendJsonString(out, entry.name); out += ",\n";
                appendIndent(out, 4); appendJsonString(out, "items"); out += ": "; appendJsonStringArray(out, entry.items, 4); out += ",\n";
                appendIndent(out, 4); appendJsonString(out, "line_number"); out += ": " + std::to_string(entry.lineNumber) + "\n";
                appendIndent(out, 3); out += "}";
                if (index + 1 != entries.size()) {
                    out += ",";
                }
                out += "\n";
            }
            appendIndent(out, 2);
        }
        out += "]";
    };

    appendLegendArray("legend_synonyms", state.legendSynonyms);
    out += ",\n";
    appendLegendArray("legend_aggregates", state.legendAggregates);
    out += ",\n";
    appendLegendArray("legend_properties", state.legendProperties);
    out += ",\n";

    appendFieldPrefix("sounds");
    out += "[";
    if (!state.sounds.empty()) {
        out += "\n";
        for (size_t index = 0; index < state.sounds.size(); ++index) {
            const auto& entry = state.sounds[index];
            appendIndent(out, 3);
            out += "{\n";
            appendIndent(out, 4); appendJsonString(out, "tokens"); out += ": [";
            if (!entry.tokens.empty()) {
                out += "\n";
                for (size_t tokenIndex = 0; tokenIndex < entry.tokens.size(); ++tokenIndex) {
                    const auto& token = entry.tokens[tokenIndex];
                    appendIndent(out, 5);
                    out += "{\n";
                    appendIndent(out, 6); appendJsonString(out, "text"); out += ": "; appendJsonString(out, token.text); out += ",\n";
                    appendIndent(out, 6); appendJsonString(out, "kind"); out += ": "; appendJsonString(out, token.kind); out += "\n";
                    appendIndent(out, 5); out += "}";
                    if (tokenIndex + 1 != entry.tokens.size()) {
                        out += ",";
                    }
                    out += "\n";
                }
                appendIndent(out, 4);
            }
            out += "],\n";
            appendIndent(out, 4); appendJsonString(out, "line_number"); out += ": " + std::to_string(entry.lineNumber) + "\n";
            appendIndent(out, 3); out += "}";
            if (index + 1 != state.sounds.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, 2);
    }
    out += "],\n";

    appendFieldPrefix("rules");
    out += "[";
    if (!state.rules.empty()) {
        out += "\n";
        for (size_t index = 0; index < state.rules.size(); ++index) {
            const auto& rule = state.rules[index];
            appendIndent(out, 3);
            out += "{\n";
            appendIndent(out, 4); appendJsonString(out, "rule"); out += ": "; appendJsonString(out, rule.rule); out += ",\n";
            appendIndent(out, 4); appendJsonString(out, "line_number"); out += ": " + std::to_string(rule.lineNumber) + ",\n";
            appendIndent(out, 4); appendJsonString(out, "mixed_case"); out += ": "; appendJsonString(out, rule.mixedCase); out += "\n";
            appendIndent(out, 3); out += "}";
            if (index + 1 != state.rules.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, 2);
    }
    out += "],\n";

    appendFieldPrefix("winconditions");
    out += "[";
    if (!state.winconditions.empty()) {
        out += "\n";
        for (size_t index = 0; index < state.winconditions.size(); ++index) {
            const auto& entry = state.winconditions[index];
            appendIndent(out, 3);
            out += "{\n";
            appendIndent(out, 4); appendJsonString(out, "tokens"); out += ": "; appendJsonStringArray(out, entry.tokens, 4); out += ",\n";
            appendIndent(out, 4); appendJsonString(out, "line_number"); out += ": " + std::to_string(entry.lineNumber) + "\n";
            appendIndent(out, 3); out += "}";
            if (index + 1 != state.winconditions.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, 2);
    }
    out += "],\n";

    appendFieldPrefix("levels");
    out += "[";
    if (!state.levels.empty()) {
        out += "\n";
        for (size_t index = 0; index < state.levels.size(); ++index) {
            const auto& level = state.levels[index];
            appendIndent(out, 3);
            out += "{\n";
            appendIndent(out, 4); appendJsonString(out, "kind"); out += level.isMessage ? ": \"message\",\n" : ": \"level\",\n";
            appendIndent(out, 4); appendJsonString(out, "line_number"); out += ": ";
            if (level.lineNumber.has_value()) {
                out += std::to_string(*level.lineNumber);
            } else {
                out += "null";
            }
            out += ",\n";
            if (level.isMessage) {
                appendIndent(out, 4); appendJsonString(out, "message"); out += ": "; appendJsonString(out, level.message); out += "\n";
            } else {
                appendIndent(out, 4); appendJsonString(out, "rows"); out += ": "; appendJsonStringArray(out, level.rows, 4); out += "\n";
            }
            appendIndent(out, 3); out += "}";
            if (index + 1 != state.levels.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, 2);
    }
    out += "],\n";
    appendFieldPrefix("subsection");
    appendJsonString(out, state.subsection);
    out += "\n";

    appendIndent(out, 1);
    out += "}\n";
    out += "}";
    return out;
}

} // namespace puzzlescript::frontend
