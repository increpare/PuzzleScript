#include "frontend/parser.hpp"

#include <utf8proc.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <regex>
#include <set>

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
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(value.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(value.size());
    utf8proc_ssize_t cursor = 0;
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            out.push_back(static_cast<char>(value[static_cast<size_t>(cursor)]));
            ++cursor;
            continue;
        }
        const utf8proc_int32_t lowered = utf8proc_tolower(codepoint);
        utf8proc_uint8_t buffer[4]{};
        const utf8proc_ssize_t encoded = utf8proc_encode_char(lowered, buffer);
        if (encoded > 0) {
            out.append(reinterpret_cast<const char*>(buffer), static_cast<size_t>(encoded));
        }
        cursor += advance;
    }
    return out;
}

std::string toUpperCopy(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(value.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(value.size());
    utf8proc_ssize_t cursor = 0;
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            out.push_back(static_cast<char>(value[static_cast<size_t>(cursor)]));
            ++cursor;
            continue;
        }
        const utf8proc_int32_t uppered = utf8proc_toupper(codepoint);
        utf8proc_uint8_t buffer[4]{};
        const utf8proc_ssize_t encoded = utf8proc_encode_char(uppered, buffer);
        if (encoded > 0) {
            out.append(reinterpret_cast<const char*>(buffer), static_cast<size_t>(encoded));
        }
        cursor += advance;
    }
    return out;
}

bool isIdentifierLike(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char ch) {
        const unsigned char byte = static_cast<unsigned char>(ch);
        return byte >= 0x80u || std::isalnum(byte) != 0 || ch == '_';
    });
}

bool isAsciiIdentifierLike(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](char ch) {
        return (ch >= '0' && ch <= '9')
            || (ch >= 'A' && ch <= 'Z')
            || (ch >= 'a' && ch <= 'z')
            || ch == '_';
    });
}

bool containsAsciiLetter(std::string_view value) {
    for (const unsigned char byte : value) {
        if (std::isalpha(byte) != 0) {
            return true;
        }
    }
    return false;
}

bool isExactToken(std::string_view value, const std::string_view* tokens, size_t count) {
    return std::find(tokens, tokens + count, value) != (tokens + count);
}

bool isSpriteRow(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    // Match parser.js section 3: stream eats /[.\\d]/ until non-sprite chars; rows may exceed
    // 5 columns (warnings are logged in JS) so do not cap line length here.
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

template <typename MapType>
std::vector<std::string> sortLikeJsObjectKeys(const MapType& values) {
    std::vector<std::string> integerKeys;
    std::vector<std::string> stringKeys;
    integerKeys.reserve(values.size());
    stringKeys.reserve(values.size());
    for (const auto& [key, _value] : values) {
        if (isNumeric(key)) {
            integerKeys.push_back(key);
        } else {
            stringKeys.push_back(key);
        }
    }
    std::sort(integerKeys.begin(), integerKeys.end(), [](const std::string& lhs, const std::string& rhs) {
        return std::stoll(lhs) < std::stoll(rhs);
    });
    std::sort(stringKeys.begin(), stringKeys.end());
    integerKeys.insert(integerKeys.end(), stringKeys.begin(), stringKeys.end());
    return integerKeys;
}

std::vector<std::string> iterateObjectKeysLikeJs(const std::map<std::string, ParserObjectEntry>& objects) {
    std::vector<std::string> integerKeys;
    std::vector<const ParserObjectEntry*> stringObjects;
    integerKeys.reserve(objects.size());
    stringObjects.reserve(objects.size());
    for (const auto& [key, object] : objects) {
        if (isNumeric(key)) {
            integerKeys.push_back(key);
        } else {
            stringObjects.push_back(&object);
        }
    }
    std::sort(integerKeys.begin(), integerKeys.end(), [](const std::string& lhs, const std::string& rhs) {
        return std::stoll(lhs) < std::stoll(rhs);
    });
    std::sort(stringObjects.begin(), stringObjects.end(), [](const ParserObjectEntry* lhs, const ParserObjectEntry* rhs) {
        if (lhs->lineNumber != rhs->lineNumber) {
            return lhs->lineNumber < rhs->lineNumber;
        }
        return lhs->name < rhs->name;
    });

    std::vector<std::string> ordered;
    ordered.reserve(objects.size());
    ordered.insert(ordered.end(), integerKeys.begin(), integerKeys.end());
    for (const ParserObjectEntry* object : stringObjects) {
        ordered.push_back(object->name);
    }
    return ordered;
}

std::vector<std::string> splitWhitespace(std::string_view value) {
    std::vector<std::string> result;
    const auto isAsciiSpace = [](char ch) {
        return std::isspace(static_cast<unsigned char>(ch)) != 0;
    };
    size_t index = 0;
    while (index < value.size()) {
        while (index < value.size() && isAsciiSpace(value[index]) != 0) {
            ++index;
        }
        if (index >= value.size()) {
            break;
        }
        const size_t start = index;
        while (index < value.size() && isAsciiSpace(value[index]) == 0) {
            ++index;
        }
        result.emplace_back(value.substr(start, index - start));
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

std::vector<std::string> splitCollisionLayerTokensPreservingCase(std::string_view value) {
    std::vector<std::string> result;
    std::string current;
    for (const char ch : value) {
        if (ch == ',' || std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!current.empty()) {
                result.push_back(trim(current));
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) {
        result.push_back(trim(current));
    }
    return result;
}

std::vector<std::string> splitObjectColorsLine(std::string_view value) {
    static const std::regex colorToken(
        "^\\s*(black|white|gray|darkgray|lightgray|grey|darkgrey|lightgrey|"
        "red|darkred|lightred|brown|darkbrown|lightbrown|orange|yellow|green|darkgreen|lightgreen|"
        "blue|lightblue|darkblue|purple|pink|transparent|#(?:[0-9a-fA-F]{3}){1,2})",
        std::regex_constants::icase);
    std::string remaining(trim(std::string(value)));
    std::vector<std::string> result;
    while (!remaining.empty()) {
        std::smatch match;
        if (!std::regex_search(remaining, match, colorToken)) {
            break;
        }
        result.push_back(toLowerCopy(trim(std::string(match[1].str()))));
        remaining = std::string(match.suffix());
    }
    return result;
}

std::vector<std::string> tokenizeLegendLine(std::string_view value) {
    std::vector<std::string> result;
    std::string current;
    for (const char ch : value) {
        if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
            continue;
        }
        if (ch == '=') {
            if (!current.empty()) {
                result.push_back(current);
                current.clear();
            }
            result.emplace_back("=");
            continue;
        }
        current.push_back(ch);
    }
    if (!current.empty()) {
        result.push_back(current);
    }
    return result;
}

bool isWordChar(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

size_t utf8CodePointCount(std::string_view value) {
    size_t count = 0;
    for (unsigned char ch : value) {
        if ((ch & 0xC0u) != 0x80u) {
            ++count;
        }
    }
    return count;
}

bool asciiSubstringEqualsInsensitive(std::string_view haystack, std::string_view needle) {
    if (haystack.size() != needle.size()) {
        return false;
    }
    for (size_t index = 0; index < haystack.size(); ++index) {
        if (std::tolower(static_cast<unsigned char>(haystack[index])) != std::tolower(static_cast<unsigned char>(needle[index]))) {
            return false;
        }
    }
    return true;
}

bool isAsciiWordCharByte(unsigned char byte) {
    return (byte >= '0' && byte <= '9')
        || (byte >= 'A' && byte <= 'Z')
        || (byte >= 'a' && byte <= 'z')
        || byte == '_';
}

void registerOriginalCaseName(ParserState& state, std::string_view lowered, std::string_view original, int32_t lineNumber) {
    if (!isAsciiIdentifierLike(lowered)) {
        return;
    }
    const std::string needle(lowered);
    for (size_t position = 0; position + needle.size() <= original.size(); ++position) {
        if (!asciiSubstringEqualsInsensitive(original.substr(position, needle.size()), needle)) {
            continue;
        }
        const bool leftBoundary = position == 0 || !isAsciiWordCharByte(static_cast<unsigned char>(original[position - 1]));
        const size_t end = position + needle.size();
        const bool rightBoundary = end >= original.size() || !isAsciiWordCharByte(static_cast<unsigned char>(original[end]));
        if (leftBoundary && rightBoundary) {
            state.originalCaseNames[needle] = std::string(original.substr(position, needle.size()));
            state.originalLineNumbers[needle] = lineNumber;
            return;
        }
    }
}

void registerLegendOriginalCaseName(ParserState& state, std::string_view lowered, int32_t lineNumber) {
    if (!isIdentifierLike(lowered)) {
        return;
    }
    if (!isWordChar(lowered.front()) || !isWordChar(lowered.back())) {
        return;
    }
    state.originalCaseNames[std::string(lowered)] = std::string(lowered);
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
            // In parser.js, '(' after "message" in a rules line does not open a comment (TOKEN_MESSAGE).
            // Line-based strip cannot track that; in rules only, treat ":( ..." as literal '(' (message tails).
            // Other sections still treat '(' after ':' as comment (OBJECTS colors, etc.).
            if (state.commentLevel == 0 && index > 0 && line[index - 1] == ':' && state.section == "rules") {
                visible.push_back(ch);
                continue;
            }
            ++state.commentLevel;
            continue;
        }
        if (ch == ')' && state.commentLevel > 0) {
            --state.commentLevel;
            if (state.commentLevel == 0) {
                state.solAfterComment = true;
            }
            continue;
        }
        if (state.commentLevel == 0) {
            visible.push_back(ch);
        }
    }
    return visible;
}

std::string takePrefixBeforeComment(std::string_view line) {
    std::string prefix;
    prefix.reserve(line.size());
    for (const char ch : line) {
        if (ch == '(') {
            break;
        }
        prefix.push_back(ch);
    }
    return prefix;
}

void populateNamesForSounds(ParserState& state) {
    state.names.clear();
    for (const auto& name : iterateObjectKeysLikeJs(state.objects)) {
        state.names.push_back(name);
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
    for (const auto& name : iterateObjectKeysLikeJs(state.objects)) {
        if (utf8CodePointCount(name) == 1) {
            state.abbrevNames.push_back(name);
        }
    }
    for (const auto& entry : state.legendSynonyms) {
        if (utf8CodePointCount(entry.name) == 1) {
            state.abbrevNames.push_back(entry.name);
        }
    }
    for (const auto& entry : state.legendAggregates) {
        if (utf8CodePointCount(entry.name) == 1) {
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

void parsePreambleLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine, std::string_view mixedCase, std::string_view rawLine) {
    const auto tokens = splitWhitespace(trimmedLine);
    if (tokens.empty()) {
        return;
    }
    const std::string key = toLowerCopy(tokens.front());
    const std::string originalKey = tokens.front();
    const size_t keyOffset = mixedCase.find(originalKey);
    const std::string loweredLine = toLowerCopy(trimmedLine);
    const size_t loweredKeyOffset = loweredLine.find(key);
    const std::string originalRemainder = keyOffset == std::string_view::npos ? std::string{} : trim(mixedCase.substr(keyOffset + originalKey.size()));
    const std::string loweredRemainder = loweredKeyOffset == std::string::npos ? std::string{} : trim(loweredLine.substr(loweredKeyOffset + key.size()));
    const bool preserveOriginalValueCase = key == "title" || key == "author" || key == "homepage";
    const std::string remainder = preserveOriginalValueCase ? originalRemainder : loweredRemainder;

    if (isExactToken(key, kPreambleValues, std::size(kPreambleValues))) {
        if (remainder.empty()) {
            diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "MetaData \"" + key + "\" needs a value.");
            return;
        }
        std::string storedValue = remainder;
        // parser.js uses reg_notcommentstart for metadata values — text after '(' is not included even if the
        // closing ')' appears later on the line (see title with parenthetical notes).
        if (preserveOriginalValueCase) {
            const std::string rawTrimmed = trim(rawLine);
            if (originalKey.size() <= rawTrimmed.size()
                && toLowerCopy(rawTrimmed.substr(0, originalKey.size())) == key) {
                size_t valueStart = originalKey.size();
                while (valueStart < rawTrimmed.size()
                    && std::isspace(static_cast<unsigned char>(rawTrimmed[valueStart])) != 0) {
                    ++valueStart;
                }
                storedValue = trim(takePrefixBeforeComment(std::string_view(rawTrimmed).substr(valueStart)));
            }
        }
        state.metadata.push_back(key);
        state.metadata.push_back(std::move(storedValue));
        state.metadataLines[key] = state.lineNumber;
        return;
    }
    if (isExactToken(key, kPreambleFlags, std::size(kPreambleFlags))) {
        state.metadata.push_back(key);
        state.metadata.push_back("true");
        return;
    }
    diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Unrecognised stuff in the prelude.");
}

void parseObjectsLine(ParserState& state, std::string_view trimmedLine, std::string_view mixedCase) {
    const std::string loweredLine = toLowerCopy(trimmedLine);
    if (state.objectsSection == 0) {
        const auto mixedTokens = splitWhitespace(mixedCase);
        if (mixedTokens.empty()) {
            return;
        }
        const std::string primaryName = toLowerCopy(mixedTokens.front());
        if (!isIdentifierLike(primaryName)) {
            return;
        }
        const std::string loweredObjectNamesLine = toLowerCopy(trim(mixedCase));
        state.objectsCandname = primaryName;
        state.objectsSection = 1;
        state.objectsSpritematrix.clear();
        auto& object = state.objects[primaryName];
        object.name = primaryName;
        object.lineNumber = state.lineNumber;
        if (isAsciiIdentifierLike(primaryName)) {
            registerOriginalCaseName(state, primaryName, trim(mixedCase), state.lineNumber);
        } else if (isIdentifierLike(primaryName) && containsAsciiLetter(primaryName)) {
            // registerOriginalCaseName only scans ASCII word boundaries; mixed ASCII/Unicode names
            // (e.g. ziehend_nördlich) need the mixed-case token. Pure-non-ASCII names (e.g. 大) omit.
            state.originalCaseNames[primaryName] = std::string(mixedTokens.front());
            state.originalLineNumbers[primaryName] = state.lineNumber;
        }
        for (size_t index = 1; index < mixedTokens.size(); ++index) {
            const std::string aliasName = toLowerCopy(mixedTokens[index]);
            if (isAsciiIdentifierLike(aliasName)) {
                registerOriginalCaseName(state, aliasName, loweredObjectNamesLine, state.lineNumber);
            } else if (isIdentifierLike(aliasName) && containsAsciiLetter(aliasName)) {
                state.originalCaseNames[aliasName] = std::string(mixedTokens[index]);
                state.originalLineNumbers[aliasName] = state.lineNumber;
            }
            state.legendSynonyms.push_back(ParserLegendEntry{
                aliasName,
                {primaryName},
                state.lineNumber,
            });
        }
        return;
    }
    if (state.objectsSection == 1) {
        auto& object = state.objects[state.objectsCandname];
        object.colors = splitObjectColorsLine(trimmedLine);
        state.objectsSection = 2;
        return;
    }
    if (state.objectsSection == 2) {
        if (!isSpriteRow(trimmedLine)) {
            state.objectsSection = 0;
            parseObjectsLine(state, trimmedLine, mixedCase);
            return;
        }
        state.objectsSection = 3;
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

void parseLegendLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine) {
    const auto tokens = tokenizeLegendLine(trimmedLine);
    if (tokens.size() < 3 || tokens[1] != "=") {
        return;
    }

    ParserLegendEntry entry;
    entry.name = toLowerCopy(tokens[0]);
    entry.lineNumber = state.lineNumber;
    registerLegendOriginalCaseName(state, entry.name, state.lineNumber);

    if (tokens.size() == 3) {
        entry.items.push_back(toLowerCopy(tokens[2]));
        state.legendSynonyms.push_back(std::move(entry));
        return;
    }

    const std::string joiner = tokens.size() >= 4 ? toLowerCopy(tokens[3]) : std::string{};
    std::function<std::vector<std::string>(const std::string&)> expandAggregate;
    std::function<std::vector<std::string>(const std::string&)> expandProperty;

    expandAggregate = [&](const std::string& name) -> std::vector<std::string> {
        const std::string lowered = toLowerCopy(name);
        if (state.objects.find(lowered) != state.objects.end()) {
            return {lowered};
        }
        for (const auto& synonym : state.legendSynonyms) {
            if (synonym.name == lowered && !synonym.items.empty()) {
                return expandAggregate(synonym.items.front());
            }
        }
        for (const auto& aggregate : state.legendAggregates) {
            if (aggregate.name == lowered) {
                std::vector<std::string> result;
                for (const auto& item : aggregate.items) {
                    const auto expanded = expandAggregate(item);
                    result.insert(result.end(), expanded.begin(), expanded.end());
                }
                return result;
            }
        }
        for (const auto& property : state.legendProperties) {
            if (property.name == lowered) {
                diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Cannot define an aggregate (using 'and') in terms of properties (something that uses 'or').");
                return {lowered};
            }
        }
        return {lowered};
    };

    expandProperty = [&](const std::string& name) -> std::vector<std::string> {
        const std::string lowered = toLowerCopy(name);
        if (state.objects.find(lowered) != state.objects.end()) {
            return {lowered};
        }
        for (const auto& synonym : state.legendSynonyms) {
            if (synonym.name == lowered && !synonym.items.empty()) {
                return expandProperty(synonym.items.front());
            }
        }
        for (const auto& aggregate : state.legendAggregates) {
            if (aggregate.name == lowered) {
                diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Cannot define a property (something defined in terms of 'or') in terms of an aggregate (something that uses 'and').  In this case, you can't define \"" + entry.name + "\" in terms of \"" + lowered + "\".");
                return {};
            }
        }
        for (const auto& property : state.legendProperties) {
            if (property.name == lowered) {
                std::vector<std::string> result;
                for (const auto& item : property.items) {
                    if (item == lowered) {
                        continue;
                    }
                    const auto expanded = expandProperty(item);
                    result.insert(result.end(), expanded.begin(), expanded.end());
                }
                return result;
            }
        }
        return {lowered};
    };

    if (joiner == "and") {
        for (size_t index = 2; index < tokens.size(); index += 2) {
            const auto expanded = expandAggregate(tokens[index]);
            entry.items.insert(entry.items.end(), expanded.begin(), expanded.end());
        }
        state.legendAggregates.push_back(std::move(entry));
    } else if (joiner == "or") {
        if (tokens.size() >= 3) {
            const auto expanded = expandProperty(tokens[2]);
            entry.items.insert(entry.items.end(), expanded.begin(), expanded.end());
        }
        if (tokens.size() >= 5) {
            const auto expanded = expandProperty(tokens[4]);
            entry.items.insert(entry.items.end(), expanded.begin(), expanded.end());
        }
        for (size_t index = 6; index < tokens.size(); index += 2) {
            entry.items.push_back(toLowerCopy(tokens[index]));
        }
        state.legendProperties.push_back(std::move(entry));
    }
}

bool soundLeadingTokenAcceptable(const ParserState& state, std::string_view lowered) {
    if (isExactToken(lowered, kSoundEvents, std::size(kSoundEvents))) {
        return true;
    }
    if (lowered.size() > 3 && lowered.rfind("sfx", 0) == 0 && isNumeric(lowered.substr(3))) {
        return true;
    }
    const std::string key(lowered);
    if (state.objects.find(key) != state.objects.end()) {
        return true;
    }
    for (const auto& entry : state.legendSynonyms) {
        if (entry.name == key) {
            return true;
        }
    }
    for (const auto& entry : state.legendAggregates) {
        if (entry.name == key) {
            return true;
        }
    }
    for (const auto& entry : state.legendProperties) {
        if (entry.name == key) {
            return true;
        }
    }
    return false;
}

std::string classifySoundKind(std::string_view lowered, size_t index) {
    if (isNumeric(lowered)) {
        return "SOUND";
    }
    if (lowered.size() > 3 && lowered.substr(0, 3) == "sfx" && isNumeric(lowered.substr(3))) {
        return "SOUNDEVENT";
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
    if (tokens.empty()) {
        return;
    }
    if (!soundLeadingTokenAcceptable(state, toLowerCopy(tokens.front()))) {
        // parser.js rejects unknown first tokens and does not push a sound row (processSoundsLine no-op).
        return;
    }
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

void parseCollisionLayersLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine) {
    const auto tokens = splitCollisionLayerTokensPreservingCase(trimmedLine);
    if (tokens.empty()) {
        return;
    }

    // parser.js parseCollisionLayersToken: each physical line starts a new layer (sol).
    state.collisionLayers.push_back({});

    std::function<std::vector<std::string>(const std::string&)> expand;
    expand = [&](const std::string& name) -> std::vector<std::string> {
        const std::string lowered = toLowerCopy(name);
        if (state.objects.find(lowered) != state.objects.end()) {
            return {lowered};
        }
        for (const auto& synonym : state.legendSynonyms) {
            if (synonym.name == lowered && !synonym.items.empty()) {
                return expand(synonym.items.front());
            }
        }
        for (const auto& aggregate : state.legendAggregates) {
            if (aggregate.name == lowered) {
                diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "\"" + lowered + "\" is an aggregate (defined using \"and\"), and cannot be added to a single layer because its constituent objects must be able to coexist.");
                return {};
            }
        }
        for (const auto& property : state.legendProperties) {
            if (property.name == lowered) {
                std::vector<std::string> result;
                for (const auto& item : property.items) {
                    if (item == lowered) {
                        continue;
                    }
                    const auto expanded = expand(item);
                    result.insert(result.end(), expanded.begin(), expanded.end());
                }
                return result;
            }
        }
        diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Cannot add \"" + toUpperCopy(name) + "\" to a collision layer; it has not been declared.");
        return {};
    };

    for (const std::string& candname : tokens) {
        const std::string loweredCand = toLowerCopy(candname);
        if (loweredCand == "background") {
            if (!state.collisionLayers.empty() && !state.collisionLayers.back().empty()) {
                diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Background must be in a layer by itself.");
            }
            state.tokenIndex = 1;
        } else if (state.tokenIndex != 0) {
            diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Background must be in a layer by itself.");
        }

        const std::vector<std::string> ar = expand(candname);
        if (state.collisionLayers.empty()) {
            diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "no layers found.");
            continue;
        }

        std::vector<size_t> foundOthers;
        for (const std::string& tcandname : ar) {
            for (size_t j = 0; j < state.collisionLayers.size(); ++j) {
                const auto& clj = state.collisionLayers[j];
                if (std::find(clj.begin(), clj.end(), tcandname) != clj.end()) {
                    if (j + 1 != state.collisionLayers.size()) {
                        foundOthers.push_back(j);
                    }
                }
            }
        }
        if (!foundOthers.empty()) {
            std::string warningStr = "Object \"" + toUpperCopy(candname) + "\" included in multiple collision layers ( layers ";
            for (size_t i = 0; i < foundOthers.size(); ++i) {
                warningStr += "#" + std::to_string(foundOthers[i] + 1) + ", ";
            }
            warningStr += "#" + std::to_string(state.collisionLayers.size());
            diagnostics.warning(DiagnosticCode::GenericWarning, state.lineNumber, warningStr + " ). You should fix this!");
        }

        if (std::find(state.currentLineWipArray.begin(), state.currentLineWipArray.end(), candname)
            != state.currentLineWipArray.end()) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "Object \"" + toUpperCopy(candname) + "\" included explicitly multiple times in the same layer. Don't do that innit.");
        }
        state.currentLineWipArray.push_back(candname);

        state.collisionLayers.back().insert(state.collisionLayers.back().end(), ar.begin(), ar.end());
    }
}

void parseRulesLine(ParserState& state, std::string_view trimmedLine, std::string_view mixedCaseRaw) {
    bool arrowPassed = false;
    bool insideCell = false;
    bool rulePrelude = true;
    int32_t bracketBalance = 0;
    for (size_t index = 0; index < trimmedLine.size(); ++index) {
        if ((index + 1) < trimmedLine.size() && trimmedLine[index] == '-' && trimmedLine[index + 1] == '>') {
            arrowPassed = true;
            ++index;
            continue;
        }
        if (trimmedLine[index] == '[') {
            insideCell = true;
            rulePrelude = false;
            continue;
        }
        if (trimmedLine[index] == ']') {
            insideCell = false;
            bracketBalance += arrowPassed ? -1 : 1;
        }
    }
    state.rulePrelude = rulePrelude;
    state.arrowPassed = arrowPassed;
    state.insideCell = insideCell;
    state.bracketBalance = bracketBalance;
    state.rules.push_back(ParserRuleEntry{
        toLowerCopy(takePrefixBeforeComment(mixedCaseRaw)),
        state.lineNumber,
        std::string(mixedCaseRaw),
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

void parseLevelsLine(ParserState& state, std::string_view trimmedLine, std::string_view rawLine) {
    const std::string trimmedMixed = trim(rawLine);
    const std::string trimmedLower = toLowerCopy(trimmedLine);
    // parser.js uses /\bmessage\b/ — "message: foo" matches (':' is a non-word boundary after "message").
    if (trimmedLower.rfind("message", 0) == 0
        && (trimmedLower.size() == 7
            || std::isspace(static_cast<unsigned char>(trimmedLine[7])) != 0
            || trimmedLine[7] == ':')) {
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
        const auto orderedKeys = sortLikeJsObjectKeys(values);
        for (size_t index = 0; index < orderedKeys.size(); ++index) {
            const auto& key = orderedKeys[index];
            appendIndent(out, indent + 1);
            appendJsonString(out, key);
            out += ": ";
            appendJsonString(out, values.at(key));
            if (index + 1 != orderedKeys.size()) {
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
        const auto orderedKeys = sortLikeJsObjectKeys(values);
        for (size_t index = 0; index < orderedKeys.size(); ++index) {
            const auto& key = orderedKeys[index];
            appendIndent(out, indent + 1);
            appendJsonString(out, key);
            out += ": ";
            out += std::to_string(values.at(key));
            if (index + 1 != orderedKeys.size()) {
                out += ",";
            }
            out += "\n";
        }
        appendIndent(out, indent);
    }
    out += "}";
}

} // namespace

struct PhysicalBlankInOpenCommentFooter {
    ParserState* state = nullptr;
    std::string_view rawLine{};

    ~PhysicalBlankInOpenCommentFooter() {
        if (state == nullptr) {
            return;
        }
        if (!trim(rawLine).empty()) {
            return;
        }
        if (state->commentLevel > 0) {
            state->solAfterComment = true;
        }
    }
};

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

    for (size_t lineIndex = 0; lineIndex < lines.size(); ++lineIndex) {
        const std::string& rawLine = lines[lineIndex];
        PhysicalBlankInOpenCommentFooter blankCommentFooter;
        blankCommentFooter.state = &state;
        blankCommentFooter.rawLine = rawLine;

        ++state.lineNumber;
        // parser.js token(): on sol, clears current_line_wip_array and tokenIndex before section logic.
        state.currentLineWipArray.clear();
        state.tokenIndex = 0;
        // parser.js clears line_should_end at the start of each physical line (sol block).
        state.lineShouldEnd = false;
        if (state.solAfterComment) {
            state.solAfterComment = false;
        }
        // Always strip comments (parser.js processes '(' / ')' before section logic).
        // A levels "message ..." line can contain '(' — skipping strip would desync commentLevel.
        const std::string mixedVisible = stripComments(rawLine, state);
        const std::string trimmedVisible = trim(mixedVisible);
        const std::string loweredVisible = toLowerCopy(trimmedVisible);

        if (trimmedVisible == ")") {
            continue;
        }

        if (trimmedVisible.empty() || isAllEquals(trimmedVisible)) {
            if (isAllEquals(trimmedVisible)) {
                state.lineShouldEnd = true;
                state.lineShouldEndBecause = "a bunch of equals signs ('===')";
            } else {
                const bool rawHasContent = !trim(rawLine).empty();
                if (rawHasContent && state.section == "objects"
                    && (state.objectsSection == 2 || state.objectsSection == 3)) {
                    // Comment-only physical line while reading sprite matrix: JS still has
                    // non-empty stream.string so blankLineHandle does not run; do not reset.
                } else {
                    handleBlankLine(state);
                }
            }
            continue;
        }

        // parser.js #976: lines that look like section headers are still level map rows inside LEVELS.
        if (state.section != "levels"
            && std::find(std::begin(kSections), std::end(kSections), loweredVisible) != std::end(kSections)) {
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

        if (state.section.empty()) {
            parsePreambleLine(state, diagnostics, trimmedVisible, trim(mixedVisible), rawLine);
            continue;
        }

        switch (state.section[0]) {
            case 'o':
                parseObjectsLine(state, trimmedVisible, trim(mixedVisible));
                break;
            case 'l':
                if (state.section == "legend") {
                    parseLegendLine(state, diagnostics, trimmedVisible);
                } else {
                    parseLevelsLine(state, trimmedVisible, rawLine);
                }
                break;
            case 's':
                parseSoundsLine(state, trimmedVisible);
                break;
            case 'c':
                parseCollisionLayersLine(state, diagnostics, trimmedVisible);
                break;
            case 'r':
                parseRulesLine(state, trimmedVisible, rawLine);
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
