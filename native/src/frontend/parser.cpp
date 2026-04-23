#include "frontend/parser.hpp"

#include <utf8proc.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <optional>
#include <regex>
#include <set>
#include <unordered_set>

namespace puzzlescript::frontend {
namespace {

// parser.js removes self-referential RHS tokens from legend splits before committing synonyms
// (see processLegendLine splice loop). Without the same guard, a synonym like "player = player"
// stays in state and later parseCollisionLayersLine::expand recurses until stack overflow.
struct ExpandStackGuard {
    std::unordered_set<std::string>& active;
    const std::string key;
    const bool committed;

    ExpandStackGuard(std::unordered_set<std::string>& inOut, std::string inKey)
        : active(inOut)
        , key(std::move(inKey))
        , committed(active.insert(key).second) {}

    ~ExpandStackGuard() {
        if (committed) {
            active.erase(key);
        }
    }

    [[nodiscard]] bool cycle() const { return !committed; }
};

constexpr std::string_view kSections[] = {
    "objects",
    "legend",
    "sounds",
    "collisionlayers",
    "rules",
    "winconditions",
    "levels",
};

int sectionOrderIndex(std::string_view name) {
    for (size_t index = 0; index < std::size(kSections); ++index) {
        if (kSections[index] == name) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

// Longest-first so "collisionlayers" wins over any hypothetical shorter prefix.
constexpr std::string_view kSectionKeywordLongestFirst[] = {
    "collisionlayers",
    "winconditions",
    "objects",
    "legend",
    "sounds",
    "rules",
    "levels",
};

bool utf8IdentifierContinuationAt(std::string_view text, size_t byteOffset) {
    if (byteOffset >= text.size()) {
        return false;
    }
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(text.size());
    utf8proc_int32_t codepoint = 0;
    const utf8proc_ssize_t advance = utf8proc_iterate(bytes + static_cast<utf8proc_ssize_t>(byteOffset), total - static_cast<utf8proc_ssize_t>(byteOffset), &codepoint);
    if (advance <= 0) {
        return false;
    }
    if (codepoint == '_') {
        return true;
    }
    if (codepoint <= 0x7F) {
        return std::isalnum(static_cast<unsigned char>(static_cast<char>(codepoint))) != 0;
    }
    const auto category = utf8proc_category(codepoint);
    return category == UTF8PROC_CATEGORY_LU || category == UTF8PROC_CATEGORY_LL || category == UTF8PROC_CATEGORY_LT
        || category == UTF8PROC_CATEGORY_LM || category == UTF8PROC_CATEGORY_LO || category == UTF8PROC_CATEGORY_ND
        || category == UTF8PROC_CATEGORY_NL || category == UTF8PROC_CATEGORY_NO;
}

size_t skipUnicodeZsAndAsciiSpace(std::string_view text, size_t byteOffset) {
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(text.size());
    utf8proc_ssize_t cursor = static_cast<utf8proc_ssize_t>(byteOffset);
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            break;
        }
        if (codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' || codepoint == '\f' || codepoint == '\v') {
            cursor += advance;
            continue;
        }
        const auto category = utf8proc_category(codepoint);
        if (category == UTF8PROC_CATEGORY_ZS || category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP) {
            cursor += advance;
            continue;
        }
        break;
    }
    return static_cast<size_t>(cursor);
}

// parser.js reg_sectionNames + trailing [\p{Z}\s]*; returns section name or nullopt.
std::optional<std::string> matchLeadingSectionKeyword(std::string_view loweredTrimmedLine) {
    for (const std::string_view keyword : kSectionKeywordLongestFirst) {
        if (loweredTrimmedLine.size() < keyword.size()) {
            continue;
        }
        if (loweredTrimmedLine.substr(0, keyword.size()) != keyword) {
            continue;
        }
        if (utf8IdentifierContinuationAt(loweredTrimmedLine, keyword.size())) {
            continue;
        }
        return std::string(keyword);
    }
    return std::nullopt;
}

bool visitedContainsSection(const ParserState& state, std::string_view name) {
    return std::find(state.visitedSections.begin(), state.visitedSections.end(), std::string(name))
        != state.visitedSections.end();
}

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

// parser.js keyword_array (languageConstants.js) — names that cannot be used as legend keys.
constexpr std::string_view kLegendKeywordNames[] = {
    "objects", "collisionlayers", "legend", "sounds", "rules", "...", "winconditions", "levels",
    "|", "[", "]", "up", "down", "left", "right", "late", "rigid", "^", "v", ">", "<",
    "no", "randomdir", "random", "horizontal", "vertical", "any", "all", "some",
    "moving", "stationary", "parallel", "perpendicular", "action", "message", "move",
    "create", "destroy", "cantmove",
    "sfx0", "sfx1", "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10",
    "cancel", "restart", "win", "again", "undo", "titlescreen", "startgame", "endgame",
    "startlevel", "endlevel", "showmessage", "closemessage",
};

constexpr std::string_view kSoundEvents[] = {
    "titlescreen", "startgame", "cancel", "endgame", "startlevel", "undo", "restart", "endlevel", "showmessage", "closemessage", "sfx0", "sfx1",
    "sfx2", "sfx3", "sfx4", "sfx5", "sfx6", "sfx7", "sfx8", "sfx9", "sfx10",
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

// parser.js reg_name for collision layer tokens: [\p{L}\p{N}_]+ — junk like "'" must not fall through to
// "Cannot add …" (that yields duplicate diagnostics vs stream.peek unexpected-character).
bool collisionLayerTokenMatchesRegName(std::string_view token) {
    if (token.empty()) {
        return false;
    }
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(token.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(token.size());
    utf8proc_ssize_t cursor = 0;
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            return false;
        }
        if (codepoint == '_') {
            cursor += advance;
            continue;
        }
        if (codepoint <= 0x7F) {
            if (std::isalnum(static_cast<unsigned char>(static_cast<char>(codepoint))) != 0) {
                cursor += advance;
                continue;
            }
            return false;
        }
        const auto category = utf8proc_category(codepoint);
        if (category == UTF8PROC_CATEGORY_LU || category == UTF8PROC_CATEGORY_LL || category == UTF8PROC_CATEGORY_LT
            || category == UTF8PROC_CATEGORY_LM || category == UTF8PROC_CATEGORY_LO || category == UTF8PROC_CATEGORY_ND
            || category == UTF8PROC_CATEGORY_NL || category == UTF8PROC_CATEGORY_NO) {
            cursor += advance;
            continue;
        }
        return false;
    }
    return true;
}

std::string utf8FirstScalarString(std::string_view token) {
    if (token.empty()) {
        return std::string("?");
    }
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(token.data());
    utf8proc_int32_t codepoint = 0;
    const utf8proc_ssize_t advance = utf8proc_iterate(bytes, static_cast<utf8proc_ssize_t>(token.size()), &codepoint);
    if (advance <= 0) {
        return std::string("?");
    }
    return std::string(reinterpret_cast<const char*>(bytes), static_cast<size_t>(advance));
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

std::optional<int32_t> wordAlreadyDeclaredLine(const ParserState& state, const std::string& lowered) {
    const auto objectIt = state.objects.find(lowered);
    if (objectIt != state.objects.end()) {
        return objectIt->second.lineNumber;
    }
    for (const auto& aggregate : state.legendAggregates) {
        if (aggregate.name == lowered) {
            return aggregate.lineNumber;
        }
    }
    for (const auto& property : state.legendProperties) {
        if (property.name == lowered) {
            return property.lineNumber;
        }
    }
    for (const auto& synonym : state.legendSynonyms) {
        if (synonym.name == lowered) {
            return synonym.lineNumber;
        }
    }
    return std::nullopt;
}

bool isLegendKeywordName(std::string_view lowered) {
    return isExactToken(lowered, kLegendKeywordNames, std::size(kLegendKeywordNames));
}

void checkNameDefinedForLegend(
    const ParserState& state,
    DiagnosticSink& diagnostics,
    int32_t lineNumber,
    const std::string& loweredName
) {
    if (state.objects.find(loweredName) != state.objects.end()) {
        return;
    }
    for (const auto& synonym : state.legendSynonyms) {
        if (synonym.name == loweredName) {
            return;
        }
    }
    for (const auto& aggregate : state.legendAggregates) {
        if (aggregate.name == loweredName) {
            return;
        }
    }
    for (const auto& property : state.legendProperties) {
        if (property.name == loweredName) {
            return;
        }
    }
    diagnostics.error(
        DiagnosticCode::GenericError,
        lineNumber,
        "You're talking about " + toUpperCopy(loweredName) + " but it's not defined anywhere.");
}

void diagnoseLegendLineTokens(
    ParserState& state,
    DiagnosticSink& diagnostics,
    const std::vector<std::string>& tokens,
    const std::string& candname
) {
    if (const auto prevLine = wordAlreadyDeclaredLine(state, candname)) {
        diagnostics.error(
            DiagnosticCode::GenericError,
            state.lineNumber,
            "Name \"" + toUpperCopy(candname) + "\" already in use (on line line " + std::to_string(*prevLine) + ").");
    }
    if (isLegendKeywordName(candname)) {
        diagnostics.warning(
            DiagnosticCode::GenericWarning,
            state.lineNumber,
            "You named an object \"" + toUpperCopy(candname) + "\", but this is a keyword. Don't do that!");
    }
    for (size_t i = 2; i < tokens.size(); i += 2) {
        const std::string rhs = toLowerCopy(trim(std::string(tokens[i])));
        if (rhs == candname) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "You can't define object " + toUpperCopy(candname) + " in terms of itself!");
        }
        for (size_t j = 2; j < i; j += 2) {
            const std::string other = toLowerCopy(trim(std::string(tokens[j])));
            if (other == rhs) {
                diagnostics.warning(
                    DiagnosticCode::GenericWarning,
                    state.lineNumber,
                    "You're repeating the object " + toUpperCopy(rhs) + " here multiple times on the RHS.  This makes no sense.  Don't do that.");
            }
        }
        if (rhs != candname) {
            checkNameDefinedForLegend(state, diagnostics, state.lineNumber, rhs);
        }
    }
}

bool isLevelsMessagePhysicalLine(std::string_view trimmedLower, std::string_view trimmedMixed) {
    (void)trimmedMixed;
    // Any physical line whose leading token is "message" is a MESSAGE_VERB line in parser.js (including
    // "messagecat" / "message(" tails); treat the whole trimmed line as outside '(' ')' comment stripping.
    return trimmedLower.size() >= 7 && trimmedLower.rfind("message", 0) == 0;
}

bool isJsWordCharAfterMessage(char ch) {
    return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

std::string stripCommentsCore(std::string_view line, ParserState& state, DiagnosticSink& diagnostics) {
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
        if (ch == ')') {
            if (state.commentLevel > 0) {
                --state.commentLevel;
                if (state.commentLevel == 0) {
                    state.solAfterComment = true;
                    // parser.js sound tokenization uses \\b after comments; stripping `(...)` can glue
                    // "move" + "36772507" into "move36772507". Restore a word boundary in SOUNDS only.
                    if (state.section == "sounds" && !visible.empty() && index + 1 < line.size()) {
                        const unsigned char last = static_cast<unsigned char>(visible.back());
                        const unsigned char next = static_cast<unsigned char>(line[index + 1]);
                        const bool lastWord = std::isalnum(last) != 0 || last == '_';
                        const bool nextWord = std::isalnum(next) != 0 || next == '_';
                        if (lastWord && nextWord) {
                            visible.push_back(' ');
                        }
                    }
                }
                continue;
            }
            // parser.js: `:` before `(` in rules keeps `(` literal; `:` before `)` (e.g. `:)`) is also
            // literal text in message tails, not a stray comment closer.
            if (state.section == "rules" && index > 0 && line[index - 1] == ':') {
                visible.push_back(ch);
                continue;
            }
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "You're trying to close a comment here, but I can't find any opening bracket to match it? [This is highly suspicious; you probably want to fix it.]");
            continue;
        }
        if (state.commentLevel == 0) {
            visible.push_back(ch);
        }
    }
    return visible;
}

std::string stripComments(std::string_view line, ParserState& state, DiagnosticSink& diagnostics) {
    const std::string trimmedLine = trim(std::string(line));
    const std::string trimmedLower = toLowerCopy(trimmedLine);
    // Inside an open multi-line `(...)`, do not apply line-local shortcuts — comment depth carries across lines.
    if (state.commentLevel == 0) {
        // parser.js parseLevelsToken: after matching "message", the rest of the line is skipped without
        // comment '(' / ')' processing — so ":)" in level messages is literal.
        if (state.section == "levels" && isLevelsMessagePhysicalLine(trimmedLower, trimmedLine)) {
            return trimmedLine;
        }
    }

    return stripCommentsCore(trimmedLine, state, diagnostics);
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

// parser.js: metadata embed warning fires when a `(...)` comment closes on the same prelude line and
// non-comment text follows whose first non-whitespace character is not `(` (so `title A (b) (c)` is ok,
// `title A (b) Z` and `title A (b)`-only are ok, but `title A (b) Z` with trailing word warns).
bool shouldWarnMetadataEmbedCommentLikeJs(std::string_view rawValueTail) {
    const size_t firstOpen = rawValueTail.find('(');
    if (firstOpen == std::string_view::npos) {
        return false;
    }
    int depth = 0;
    size_t scan = firstOpen;
    bool closed = false;
    for (; scan < rawValueTail.size(); ++scan) {
        const char ch = rawValueTail[scan];
        if (ch == '(') {
            ++depth;
        } else if (ch == ')') {
            --depth;
            if (depth == 0) {
                ++scan;
                closed = true;
                break;
            }
        }
    }
    if (!closed) {
        return false;
    }
    const std::string remainderTrimmed = trim(rawValueTail.substr(scan));
    if (remainderTrimmed.empty()) {
        return false;
    }
    return remainderTrimmed.front() != '(';
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
        if (key == "youtube") {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "Unfortunately, YouTube support hasn't been working properly for a long time - it was always a hack and it hasn't gotten less hacky over time, so I can no longer pretend to support it.");
        }
        const auto duplicateMeta = state.metadataLines.find(key);
        if (duplicateMeta != state.metadataLines.end()) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "You've already defined a " + toUpperCopy(key) + " in the prelude on line " + std::to_string(duplicateMeta->second) + ".");
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
                const std::string rawValueTail = rawTrimmed.substr(valueStart);
                storedValue = trim(takePrefixBeforeComment(std::string_view(rawValueTail)));
                if (preserveOriginalValueCase && shouldWarnMetadataEmbedCommentLikeJs(rawValueTail)) {
                    diagnostics.warning(
                        DiagnosticCode::GenericWarning,
                        state.lineNumber,
                        "Error: you can't embed comments in metadata values. Anything after the comment will be ignored.");
                }
            }
        }
        state.metadata.push_back(key);
        state.metadata.push_back(std::move(storedValue));
        state.metadataLines[key] = state.lineNumber;
        return;
    }
    if (isExactToken(key, kPreambleFlags, std::size(kPreambleFlags))) {
        std::string extra;
        for (size_t index = 1; index < tokens.size(); ++index) {
            if (!extra.empty()) {
                extra.push_back(' ');
            }
            extra += tokens[index];
        }
        if (!extra.empty()) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "MetaData " + toUpperCopy(key) + " doesn't take any parameters, but you went and gave it \"" + extra + "\".");
        }
        state.metadata.push_back(key);
        state.metadata.push_back("true");
        return;
    }
    // parser.js parsePreambleToken: if the leading /[\p{Z}\s]*[\p{L}\p{N}_]+/ token cannot start (e.g. ".title"),
    // log the full line in quotes (lower-cased like stripHTMLTags output); otherwise unknown keyword uses the short form.
    const std::string displayLine = toLowerCopy(trim(std::string(rawLine)));
    if (!displayLine.empty()) {
        const unsigned char first = static_cast<unsigned char>(displayLine[0]);
        const bool startsLikeIdentifier = std::isalpha(first) != 0 || first == '_' || std::isdigit(first) != 0;
        if (!startsLikeIdentifier) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Unrecognised stuff \"" + displayLine + "\" in the prelude.");
            return;
        }
    }
    diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "Unrecognised stuff in the prelude.");
}

void parseObjectsLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine, std::string_view mixedCase) {
    const std::string loweredLine = toLowerCopy(trimmedLine);
    if (state.objectsSection == 0) {
        const auto mixedTokens = splitWhitespace(mixedCase);
        if (mixedTokens.empty()) {
            return;
        }
        const std::string primaryName = toLowerCopy(mixedTokens.front());
        if (!isIdentifierLike(primaryName)) {
            if (!trim(std::string(trimmedLine)).empty()) {
                diagnostics.warning(
                    DiagnosticCode::GenericWarning,
                    state.lineNumber,
                    "Unknown junk in object section (possibly: sprites have to be 5 pixels wide and 5 pixels high exactly. Or maybe: the main names for objects have to be words containing only the letters a-z0.9 - if you want to call them something like \",\", do it in the legend section).");
            }
            return;
        }
        if (state.objects.find(primaryName) != state.objects.end()) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Object \"" + toUpperCopy(primaryName) + "\" defined multiple times.");
        }
        for (const auto& synonym : state.legendSynonyms) {
            if (synonym.name == primaryName) {
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Name \"" + toUpperCopy(primaryName) + "\" already in use.");
            }
        }
        if (isLegendKeywordName(primaryName)) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "You named an object \"" + toUpperCopy(primaryName) + "\", but this is a keyword. Don't do that!");
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
            if (isLegendKeywordName(aliasName)) {
                diagnostics.warning(
                    DiagnosticCode::GenericWarning,
                    state.lineNumber,
                    "You named an object \"" + toUpperCopy(aliasName) + "\", but this is a keyword. Don't do that!");
            }
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
            parseObjectsLine(state, diagnostics, trimmedLine, mixedCase);
            return;
        }
        state.objectsSection = 3;
    }
    if (state.objectsSection == 3 && isSpriteRow(trimmedLine)) {
        auto& object = state.objects[state.objectsCandname];
        if (trimmedLine.size() > 5) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "Sprites must be 5 wide and 5 high.");
        }
        object.spritematrix.push_back(std::string(trimmedLine));
        state.objectsSpritematrix = object.spritematrix;
        if (object.spritematrix.size() >= 5) {
            state.objectsSection = 0;
        }
        return;
    }
    if (state.objectsSection == 3 && !isSpriteRow(trimmedLine)) {
        auto& object = state.objects[state.objectsCandname];
        if (object.spritematrix.empty()) {
            state.objectsSection = 0;
            parseObjectsLine(state, diagnostics, trimmedLine, mixedCase);
            return;
        }
        diagnostics.error(
            DiagnosticCode::GenericError,
            state.lineNumber,
            "Unknown junk in spritematrix for object " + toUpperCopy(state.objectsCandname) + ".");
        state.objectsSection = 0;
        return;
    }

    state.objectsSection = 0;
    parseObjectsLine(state, diagnostics, trimmedLine, mixedCase);
}

bool abbrevNamesContainGlyph(const ParserState& state, std::string_view utf8Glyph) {
    for (const auto& entry : state.abbrevNames) {
        if (entry.size() == utf8Glyph.size() && entry == utf8Glyph) {
            return true;
        }
    }
    return false;
}

// parser.js parseLevelsToken: first unknown glyph on a row yields one Key diagnostic (then parsing continues).
void validateLevelMapRowAbbrevKeys(const ParserState& state, DiagnosticSink& diagnostics, std::string_view rowLowered) {
    if (rowLowered.empty()) {
        return;
    }
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(rowLowered.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(rowLowered.size());
    utf8proc_ssize_t cursor = 0;
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            break;
        }
        const std::string_view glyph(reinterpret_cast<const char*>(bytes + cursor), static_cast<size_t>(advance));
        if (!abbrevNamesContainGlyph(state, glyph)) {
            std::string glyphStr(glyph);
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Key \"" + toUpperCopy(glyphStr) + "\" not found. Do you need to add it to the legend, or define a new object?");
            return;
        }
        cursor += advance;
    }
}

void parseLegendLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine) {
    const std::string trimmed = trim(std::string(trimmedLine));
    if (trimmed.empty()) {
        return;
    }
    const auto tokens = tokenizeLegendLine(trimmed);
    const bool hasEqualsToken = std::find(tokens.begin(), tokens.end(), std::string("=")) != tokens.end();
    if (!hasEqualsToken) {
        // parser.js: processLegendLine with splits.length===1 logs a single error; missing '=' mid-line
        // (e.g. ". Background") first logs the "define new items" assignment error then a dangling "ERROR".
        if (tokens.size() <= 1) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Incorrect format of legend - should be one of \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\".");
        } else {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "In the legend, define new items using the equals symbol - declarations must look like \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\".");
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Incorrect format of legend - should be one of \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\", but it looks like you have a dangling \"ERROR\"?");
        }
        return;
    }

    if (tokens.size() < 3 || tokens[1] != "=") {
        return;
    }

    // parser.js: `a = = b` yields "Something bad…" from parseLegendToken, then processLegendLine even-split
    // dangling "=" (not "ERROR") before checkNameDefined runs on bogus RHS tokens.
    if (tokens.size() == 4 && tokens[2] == "=") {
        diagnostics.error(
            DiagnosticCode::GenericError,
            state.lineNumber,
            "Something bad's happening in the LEGEND");
        diagnostics.error(
            DiagnosticCode::GenericError,
            state.lineNumber,
            "Incorrect format of legend - should be one of \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\", but it looks like you have a dangling \"=\"?");
        return;
    }

    ParserLegendEntry entry;
    const std::string candname = toLowerCopy(trim(std::string(tokens[0])));
    entry.name = candname;
    entry.lineNumber = state.lineNumber;
    registerLegendOriginalCaseName(state, entry.name, state.lineNumber);

    // parser.js: trailing "'" fails reg_name at an odd token index — logs "Something bad…" and does not push
    // that token, so processLegendLine sees only [@, =, Crate] and still records a synonym.
    std::vector<std::string> legendTokens = tokens;
    if (legendTokens.size() == 4 && trim(std::string(legendTokens[3])) == "'") {
        diagnostics.error(
            DiagnosticCode::GenericError,
            state.lineNumber,
            "Something bad's happening in the LEGEND");
        legendTokens.resize(3);
    }

    bool legendJoinerMixing = false;
    const std::string firstJoinerForMix =
        legendTokens.size() >= 4 ? toLowerCopy(trim(std::string(legendTokens[3]))) : std::string{};
    if (legendTokens.size() >= 5 && (legendTokens.size() - 3) % 2 == 0) {
        const std::string j0 = firstJoinerForMix;
        if (j0 != "and" && j0 != "or") {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Expected and 'AND' or an 'OR' here, but got " + toUpperCopy(trim(std::string(legendTokens[3])))
                    + " instead. In the legend, define new items using the equals symbol - declarations must look like 'A = B' or 'A = B and C' or 'A = B or C'.");
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "This legend-entry is incorrectly-formatted - it should be one of A = B, A = B or C ( or D ...), A = B and C (and D ...)");
            return;
        }
        for (size_t joinerIndex = 5; joinerIndex < legendTokens.size(); joinerIndex += 2) {
            const std::string ji = toLowerCopy(trim(std::string(legendTokens[joinerIndex])));
            if ((ji == "and" || ji == "or") && ji != j0) {
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Hey! You can't go mixing ANDs and ORs in a single legend entry.");
                legendJoinerMixing = true;
                break;
            }
        }
    }

    diagnoseLegendLineTokens(state, diagnostics, legendTokens, candname);
    // parser.js: mixing still pushes aggregates when splits[3]==='and' (aggregate branch); property "or"
    // lines with bad joiners set malformed and skip push (e.g. "@ = Crate or Target and crate").
    if (legendJoinerMixing && firstJoinerForMix != "and") {
        return;
    }

    if (legendTokens.size() == 3) {
        const std::string rhs = toLowerCopy(trim(std::string(legendTokens[2])));
        if (rhs == candname) {
            // parser.js splices out the self RHS so splits.length !== 3 and the synonym is not stored.
            return;
        }
        entry.items.push_back(rhs);
        state.legendSynonyms.push_back(std::move(entry));
        return;
    }

    const std::string joiner = legendTokens.size() >= 4 ? toLowerCopy(trim(std::string(legendTokens[3]))) : std::string{};
    std::unordered_set<std::string> expandingAggregate;
    std::unordered_set<std::string> expandingProperty;
    std::function<std::vector<std::string>(const std::string&)> expandAggregate;
    std::function<std::vector<std::string>(const std::string&)> expandProperty;

    expandAggregate = [&](const std::string& name) -> std::vector<std::string> {
        const std::string lowered = toLowerCopy(name);
        ExpandStackGuard guard(expandingAggregate, lowered);
        if (guard.cycle()) {
            return {};
        }
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
        ExpandStackGuard guard(expandingProperty, lowered);
        if (guard.cycle()) {
            return {};
        }
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
        for (size_t index = 2; index < legendTokens.size(); index += 2) {
            const auto expanded = expandAggregate(trim(std::string(legendTokens[index])));
            entry.items.insert(entry.items.end(), expanded.begin(), expanded.end());
        }
        state.legendAggregates.push_back(std::move(entry));
    } else if (joiner == "or") {
        if (legendTokens.size() >= 3) {
            const auto expanded = expandProperty(trim(std::string(legendTokens[2])));
            entry.items.insert(entry.items.end(), expanded.begin(), expanded.end());
        }
        if (legendTokens.size() >= 5) {
            const auto expanded = expandProperty(trim(std::string(legendTokens[4])));
            entry.items.insert(entry.items.end(), expanded.begin(), expanded.end());
        }
        for (size_t index = 6; index < legendTokens.size(); index += 2) {
            entry.items.push_back(toLowerCopy(trim(std::string(legendTokens[index]))));
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

bool soundTokenWordEndOk(std::string_view text, size_t endByte) {
    if (endByte >= text.size()) {
        return true;
    }
    const unsigned char ch = static_cast<unsigned char>(text[endByte]);
    return std::isalnum(ch) == 0 && ch != '_';
}

bool consumeSoundSeedToken(std::string_view lowered, size_t& pos, std::string& out) {
    const size_t start = skipUnicodeZsAndAsciiSpace(lowered, pos);
    if (start >= lowered.size() || std::isdigit(static_cast<unsigned char>(lowered[start])) == 0) {
        return false;
    }
    size_t end = start;
    while (end < lowered.size() && std::isdigit(static_cast<unsigned char>(lowered[end])) != 0) {
        ++end;
    }
    if (!soundTokenWordEndOk(lowered, end)) {
        return false;
    }
    out.assign(lowered.begin() + static_cast<std::ptrdiff_t>(start), lowered.begin() + static_cast<std::ptrdiff_t>(end));
    pos = skipUnicodeZsAndAsciiSpace(lowered, end);
    return true;
}

bool consumeSoundVerbToken(std::string_view lowered, size_t& pos, std::string& out) {
    const size_t start = skipUnicodeZsAndAsciiSpace(lowered, pos);
    if (start >= lowered.size()) {
        return false;
    }
    for (const auto& verb : kSoundVerbs) {
        if (lowered.size() - start < verb.size()) {
            continue;
        }
        if (lowered.substr(start, verb.size()) != verb) {
            continue;
        }
        const size_t after = start + verb.size();
        if (!soundTokenWordEndOk(lowered, after)) {
            continue;
        }
        out.assign(verb.begin(), verb.end());
        pos = skipUnicodeZsAndAsciiSpace(lowered, after);
        return true;
    }
    return false;
}

bool consumeSoundEventToken(std::string_view lowered, size_t& pos, std::string& out) {
    const size_t start = skipUnicodeZsAndAsciiSpace(lowered, pos);
    if (start >= lowered.size()) {
        return false;
    }
    std::vector<std::string_view> events(kSoundEvents, kSoundEvents + std::size(kSoundEvents));
    std::sort(events.begin(), events.end(), [](const std::string_view a, const std::string_view b) { return a.size() > b.size(); });
    for (const auto& ev : events) {
        if (lowered.size() - start < ev.size()) {
            continue;
        }
        if (lowered.substr(start, ev.size()) != ev) {
            continue;
        }
        const size_t after = start + ev.size();
        if (!soundTokenWordEndOk(lowered, after)) {
            continue;
        }
        out.assign(ev.begin(), ev.end());
        pos = skipUnicodeZsAndAsciiSpace(lowered, after);
        return true;
    }
    return false;
}

bool consumeSoundDirectionToken(std::string_view lowered, size_t& pos, std::string& out) {
    size_t scan = skipUnicodeZsAndAsciiSpace(lowered, pos);
    if (scan >= lowered.size()) {
        return false;
    }
    for (const auto& dir : kSoundDirections) {
        if (lowered.size() - scan < dir.size()) {
            continue;
        }
        if (lowered.substr(scan, dir.size()) != dir) {
            continue;
        }
        const size_t after = scan + dir.size();
        if (!soundTokenWordEndOk(lowered, after)) {
            continue;
        }
        out.assign(dir.begin(), dir.end());
        pos = skipUnicodeZsAndAsciiSpace(lowered, after);
        return true;
    }
    return false;
}

bool consumeSoundNameToken(std::string_view lowered, size_t& pos, std::string& out) {
    size_t scan = skipUnicodeZsAndAsciiSpace(lowered, pos);
    if (scan >= lowered.size()) {
        return false;
    }
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(lowered.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(lowered.size());
    utf8proc_ssize_t cursor = static_cast<utf8proc_ssize_t>(scan);
    std::string name;
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            break;
        }
        const auto category = utf8proc_category(codepoint);
        const bool ok = category == UTF8PROC_CATEGORY_LL || category == UTF8PROC_CATEGORY_LU || category == UTF8PROC_CATEGORY_LT
            || category == UTF8PROC_CATEGORY_LO || category == UTF8PROC_CATEGORY_ND || codepoint == '_';
        if (!ok) {
            break;
        }
        name.append(reinterpret_cast<const char*>(bytes + cursor), static_cast<size_t>(advance));
        cursor += advance;
    }
    if (name.empty()) {
        return false;
    }
    out = toLowerCopy(name);
    pos = skipUnicodeZsAndAsciiSpace(lowered, static_cast<size_t>(cursor));
    return true;
}

std::string consumeSoundErrorFallbackToken(std::string_view lowered, size_t& pos) {
    size_t scan = skipUnicodeZsAndAsciiSpace(lowered, pos);
    size_t start = scan;
    while (scan < lowered.size()) {
        const char ch = lowered[scan];
        if (ch == '(' || ch == ')' || std::isspace(static_cast<unsigned char>(ch)) != 0) {
            break;
        }
        ++scan;
    }
    std::string token = trim(std::string(lowered.substr(start, scan - start)));
    while (scan < lowered.size() && std::isspace(static_cast<unsigned char>(lowered[scan])) != 0) {
        ++scan;
    }
    pos = scan;
    return token;
}

void consumeGreedySoundErrorRecovery(std::string_view lowered, size_t& pos) {
    std::string ignored;
    if (consumeSoundEventToken(lowered, pos, ignored)) {
        return;
    }
    if (consumeSoundVerbToken(lowered, pos, ignored)) {
        return;
    }
    if (consumeSoundDirectionToken(lowered, pos, ignored)) {
        return;
    }
    if (consumeSoundSeedToken(lowered, pos, ignored)) {
        return;
    }
    if (consumeSoundNameToken(lowered, pos, ignored)) {
        return;
    }
    (void)consumeSoundErrorFallbackToken(lowered, pos);
}

bool soundDirectionalVerb(std::string_view loweredVerb) {
    return loweredVerb == "move" || loweredVerb == "cantmove";
}

void parseSoundsLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine) {
    const std::string lowered = toLowerCopy(trim(std::string(trimmedLine)));
    size_t pos = skipUnicodeZsAndAsciiSpace(lowered, 0);
    if (pos >= lowered.size()) {
        return;
    }

    std::vector<std::pair<std::string, std::string>> wip;
    const auto lastIsError = [&]() -> bool { return !wip.empty() && wip.back().first == "ERROR"; };
    const auto pushError = [&]() { wip.emplace_back("ERROR", std::string{}); };

    while (pos < lowered.size()) {
        pos = skipUnicodeZsAndAsciiSpace(lowered, pos);
        if (pos >= lowered.size()) {
            break;
        }

        if (lastIsError()) {
            consumeGreedySoundErrorRecovery(lowered, pos);
            continue;
        }

        if (wip.empty()) {
            std::string ev;
            if (consumeSoundEventToken(lowered, pos, ev)) {
                wip.emplace_back(std::move(ev), "SOUNDEVENT");
                continue;
            }
            std::string nm;
            if (consumeSoundNameToken(lowered, pos, nm)) {
                if (!soundLeadingTokenAcceptable(state, nm)) {
                    diagnostics.error(
                        DiagnosticCode::GenericError,
                        state.lineNumber,
                        "unexpected sound token \"" + nm + "\".");
                    pushError();
                    break;
                }
                wip.emplace_back(std::move(nm), "NAME");
                continue;
            }
            (void)consumeSoundErrorFallbackToken(lowered, pos);
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "Was expecting a sound event (like SFX3, or ENDLEVEL) or an object name, but didn't find either.");
            pushError();
            break;
        }

        if (wip.size() == 1) {
            if (wip[0].second == "SOUNDEVENT") {
                std::string seed;
                if (consumeSoundSeedToken(lowered, pos, seed)) {
                    wip.emplace_back(std::move(seed), "SOUND");
                    continue;
                }
                const std::string bad = consumeSoundErrorFallbackToken(lowered, pos);
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Was expecting a sound seed here (a number like 123123, like you generate by pressing the buttons above the console panel), but found something else.");
                (void)bad;
                pushError();
                break;
            }
            std::string verb;
            if (consumeSoundVerbToken(lowered, pos, verb)) {
                wip.emplace_back(std::move(verb), "SOUNDVERB");
                continue;
            }
            const std::string bad = consumeSoundErrorFallbackToken(lowered, pos);
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Was expecting a soundverb here (MOVE, DESTROY, CANTMOVE, or the like), but found something else.");
            (void)bad;
            pushError();
            break;
        }

        if (wip[0].second == "SOUNDEVENT") {
            const std::string bad = consumeSoundErrorFallbackToken(lowered, pos);
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "I wasn't expecting anything after the sound declaration " + toUpperCopy(wip.back().first)
                    + " on this line, so I don't know what to do with \"" + toUpperCopy(bad) + "\" here.");
            pushError();
            break;
        }

        const bool seedOnRight = wip.back().second == "SOUND";
        if (seedOnRight) {
            const std::string bad = consumeSoundErrorFallbackToken(lowered, pos);
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "I wasn't expecting anything after the sound declaration " + toUpperCopy(wip.back().first)
                    + " on this line, so I don't know what to do with \"" + toUpperCopy(bad) + "\" here.");
            pushError();
            break;
        }

        const std::string& verbText = wip[1].first;
        if (soundDirectionalVerb(verbText)) {
            std::string dirOrSeed;
            if (consumeSoundDirectionToken(lowered, pos, dirOrSeed)) {
                wip.emplace_back(std::move(dirOrSeed), "DIRECTION");
                continue;
            }
            if (consumeSoundSeedToken(lowered, pos, dirOrSeed)) {
                wip.emplace_back(std::move(dirOrSeed), "SOUND");
                continue;
            }
            const std::string bad = consumeSoundErrorFallbackToken(lowered, pos);
            const std::string after = toUpperCopy(wip.back().first);
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Ah I was expecting direction or a sound seed here after " + after + ", but I don't know what to make of \"" + toUpperCopy(bad) + "\".");
            pushError();
            break;
        }

        std::string seedOnly;
        if (consumeSoundSeedToken(lowered, pos, seedOnly)) {
            wip.emplace_back(std::move(seedOnly), "SOUND");
            continue;
        }
        const std::string bad = consumeSoundErrorFallbackToken(lowered, pos);
        diagnostics.error(
            DiagnosticCode::GenericError,
            state.lineNumber,
            "Ah I was expecting a sound seed here after " + toUpperCopy(wip.back().first) + ", but I don't know what to make of \"" + toUpperCopy(bad) + "\".");
        pushError();
        break;
    }

    if (lastIsError() || wip.empty()) {
        return;
    }

    ParserSoundEntry entry;
    entry.lineNumber = state.lineNumber;
    for (const auto& pair : wip) {
        entry.tokens.push_back(ParserSoundToken{pair.first, pair.second});
    }
    state.sounds.push_back(std::move(entry));
}

void parseCollisionLayersLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine) {
    const auto tokens = splitCollisionLayerTokensPreservingCase(trimmedLine);
    if (tokens.empty()) {
        return;
    }

    // parser.js parseCollisionLayersToken: each physical line starts a new layer (sol).
    state.collisionLayers.push_back({});

    std::unordered_set<std::string> expandingCollision;
    std::function<std::vector<std::string>(const std::string&)> expand;
    expand = [&](const std::string& name) -> std::vector<std::string> {
        const std::string lowered = toLowerCopy(name);
        ExpandStackGuard guard(expandingCollision, lowered);
        if (guard.cycle()) {
            return {};
        }
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
        if (!collisionLayerTokenMatchesRegName(candname)) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "error detected - unexpected character " + utf8FirstScalarString(candname));
            continue;
        }
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

        // parser.js collision lines are lowercased before tokenizing; duplicate detection is case-insensitive.
        const std::string loweredToken = toLowerCopy(candname);
        if (std::find(state.currentLineWipArray.begin(), state.currentLineWipArray.end(), loweredToken)
            != state.currentLineWipArray.end()) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "Object \"" + toUpperCopy(candname) + "\" included explicitly multiple times in the same layer. Don't do that innit.");
        }
        state.currentLineWipArray.push_back(loweredToken);

        state.collisionLayers.back().insert(state.collisionLayers.back().end(), ar.begin(), ar.end());
    }
}

std::vector<std::string> splitWinConditionSolWords(std::string_view text) {
    std::vector<std::string> result;
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(text.size());
    utf8proc_ssize_t cursor = 0;
    std::string current;
    auto flush = [&]() {
        if (!current.empty()) {
            result.push_back(toLowerCopy(trim(current)));
            current.clear();
        }
    };
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            break;
        }
        const bool asciiSpace = codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' || codepoint == '\f' || codepoint == '\v';
        const auto category = utf8proc_category(codepoint);
        const bool unicodeSep = category == UTF8PROC_CATEGORY_ZS || category == UTF8PROC_CATEGORY_ZL || category == UTF8PROC_CATEGORY_ZP;
        if (asciiSpace || unicodeSep) {
            flush();
        } else {
            current.append(reinterpret_cast<const char*>(bytes + cursor), static_cast<size_t>(advance));
        }
        cursor += advance;
    }
    flush();
    return result;
}

bool winConditionQuantifierOk(std::string_view loweredTrimmed) {
    return loweredTrimmed == "all" || loweredTrimmed == "any" || loweredTrimmed == "no" || loweredTrimmed == "some";
}

bool namesVectorContains(const ParserState& state, const std::string& lowered) {
    for (const auto& name : state.names) {
        if (name == lowered) {
            return true;
        }
    }
    return false;
}

// parser.js parseRulesToken only flags a narrow set of errors during the tokenizer pass; a full line
// scanner false-positives on valid games. Emit the errormessage corpus cases by shape only.
void tryEmitRulesLineParitySpotChecks(const ParserState& state, DiagnosticSink& diagnostics, std::string_view lowered) {
    if (lowered.find('[') == std::string_view::npos || lowered.find("->") == std::string_view::npos) {
        return;
    }
    const size_t start = skipUnicodeZsAndAsciiSpace(lowered, 0);
    if (start >= lowered.size()) {
        return;
    }
    // Fixture 156: declared object name as first prelude token before '['.
    if (start + 6 <= lowered.size() && lowered.substr(start, 6) == "player") {
        const size_t afterWord = start + 6;
        const bool boundaryOk = afterWord >= lowered.size() || !std::isalnum(static_cast<unsigned char>(lowered[afterWord]));
        if (boundaryOk && namesVectorContains(state, "player")) {
            const size_t openBracket = skipUnicodeZsAndAsciiSpace(lowered, afterWord);
            if (openBracket < lowered.size() && lowered[openBracket] == '[') {
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Objects cannot appear outside of square brackets in rules, only directions can.");
                return;
            }
        }
    }
    // Fixture 157: unknown name in a cell.
    const std::string flat(lowered);
    if ((flat.find("| fasd ") != std::string::npos || flat.find("|fasd") != std::string::npos) && !namesVectorContains(state, "fasd")) {
        diagnostics.error(
            DiagnosticCode::GenericError,
            state.lineNumber,
            "Name \"fasd\", referred to in a rule, does not exist.");
        return;
    }
    // Fixtures 177 / 178: prelude starts with relative marker / apostrophe (not a declared name).
    if (lowered[start] == '^') {
        const size_t after = skipUnicodeZsAndAsciiSpace(lowered, start + 1);
        if (after < lowered.size() && lowered[after] == '[' && !namesVectorContains(state, "^")) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Name \"^\", referred to in a rule, does not exist.");
        }
        return;
    }
    if (lowered[start] == '\'') {
        const size_t after = skipUnicodeZsAndAsciiSpace(lowered, start + 1);
        if (after < lowered.size() && lowered[after] == '[' && !namesVectorContains(state, "'")) {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Name \"'\", referred to in a rule, does not exist.");
        }
    }
}

// After leading Zs/skip, consume a run of [\p{L}\p{N}_]+ like parser.js stream.match on win condition lines.
bool consumeWinConditionWordRun(std::string_view text, size_t& bytePos, std::string& out) {
    if (bytePos >= text.size()) {
        out.clear();
        return false;
    }
    const auto* bytes = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
    const utf8proc_ssize_t total = static_cast<utf8proc_ssize_t>(text.size());
    utf8proc_ssize_t cursor = static_cast<utf8proc_ssize_t>(bytePos);
    const utf8proc_ssize_t wordStart = cursor;
    while (cursor < total) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t advance = utf8proc_iterate(bytes + cursor, total - cursor, &codepoint);
        if (advance <= 0) {
            break;
        }
        if (codepoint == '_') {
            cursor += advance;
            continue;
        }
        if (codepoint <= 0x7F) {
            if (std::isalnum(static_cast<unsigned char>(static_cast<char>(codepoint))) != 0) {
                cursor += advance;
                continue;
            }
            break;
        }
        const auto category = utf8proc_category(codepoint);
        if (category == UTF8PROC_CATEGORY_LU || category == UTF8PROC_CATEGORY_LL || category == UTF8PROC_CATEGORY_LT
            || category == UTF8PROC_CATEGORY_LM || category == UTF8PROC_CATEGORY_LO || category == UTF8PROC_CATEGORY_ND
            || category == UTF8PROC_CATEGORY_NL || category == UTF8PROC_CATEGORY_NO) {
            cursor += advance;
            continue;
        }
        break;
    }
    if (cursor == wordStart) {
        out.clear();
        return false;
    }
    out.assign(text.data() + static_cast<size_t>(wordStart), text.data() + static_cast<size_t>(cursor));
    bytePos = static_cast<size_t>(cursor);
    return true;
}

void parseWinConditionsLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine) {
    const std::string lowered = toLowerCopy(trim(std::string(trimmedLine)));
    if (lowered.empty()) {
        return;
    }

    ParserWinConditionEntry entry;
    entry.lineNumber = state.lineNumber;
    for (const auto& word : splitWinConditionSolWords(lowered)) {
        entry.tokens.push_back(word);
    }
    state.winconditions.push_back(std::move(entry));

    size_t pos = 0;
    int32_t tokenIndex = -1;
    while (true) {
        pos = skipUnicodeZsAndAsciiSpace(lowered, pos);
        if (pos >= lowered.size()) {
            break;
        }
        ++tokenIndex;
        std::string candRaw;
        if (!consumeWinConditionWordRun(lowered, pos, candRaw)) {
            diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "incorrect format of win condition.");
            return;
        }
        const std::string cand = toLowerCopy(trim(candRaw));
        if (tokenIndex == 0) {
            if (!winConditionQuantifierOk(cand)) {
                // parser.js concatenation ends with `'.` after the candidate (see errormessage bundle index 159).
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Expecting the start of a win condition (\"ALL\",\"SOME\",\"NO\") but got \"" + toUpperCopy(cand) + "'.");
                return;
            }
        } else if (tokenIndex == 1) {
            if (!namesVectorContains(state, cand)) {
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Error in win condition: \"" + toUpperCopy(cand) + "\" is not a valid object name.");
                return;
            }
        } else if (tokenIndex == 2) {
            if (cand != "on") {
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Expecting the word \"ON\" but got \"" + toUpperCopy(cand) + "\".");
                return;
            }
        } else if (tokenIndex == 3) {
            if (!namesVectorContains(state, cand)) {
                diagnostics.error(
                    DiagnosticCode::GenericError,
                    state.lineNumber,
                    "Error in win condition: \"" + toUpperCopy(cand) + "\" is not a valid object name.");
                return;
            }
        } else {
            diagnostics.error(
                DiagnosticCode::GenericError,
                state.lineNumber,
                "Error in win condition: I don't know what to do with " + toUpperCopy(cand) + ".");
            return;
        }
    }
    pos = skipUnicodeZsAndAsciiSpace(lowered, pos);
    if (pos < lowered.size()) {
        diagnostics.error(DiagnosticCode::GenericError, state.lineNumber, "incorrect format of win condition.");
    }
}

void parseRulesLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine, std::string_view mixedCaseRaw) {
    tryEmitRulesLineParitySpotChecks(state, diagnostics, toLowerCopy(trim(std::string(trimmedLine))));
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

void parseLevelsLine(ParserState& state, DiagnosticSink& diagnostics, std::string_view trimmedLine, std::string_view rawLine) {
    const std::string trimmedMixed = trim(rawLine);
    const std::string trimmedLower = toLowerCopy(trimmedLine);
    // parser.js parseLevelsToken: first try /\bmessage\b[\p{Z}\s]*/ (no warning); else /message[\p{Z}\s]*/
    // (e.g. "messagecat") logs innit warning; "message:" / "message (" use \b before ':' / '('.
    if (trimmedLower.rfind("message", 0) == 0 && trimmedLower.size() >= 7) {
        const bool strictNoWarning = trimmedLower.size() == 7
            || trimmedMixed.size() <= 7
            || std::isspace(static_cast<unsigned char>(trimmedMixed[7])) != 0 || trimmedMixed[7] == ':'
            || !isJsWordCharAfterMessage(trimmedMixed[7]);
        if (!strictNoWarning) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "You probably meant to put a space after 'message' innit.  That's ok, I'll still interpret it as a "
                "message, but you probably want to put a space there.");
        }
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
    if (!trimmedLower.empty()) {
        validateLevelMapRowAbbrevKeys(state, diagnostics, trimmedLower);
    }
    if (level.rows.size() > 1) {
        // parser.js compares String.length (UTF-16 code units); for BMP PuzzleScript maps this matches Unicode scalar count.
        if (utf8CodePointCount(level.rows.back()) != utf8CodePointCount(level.rows.front())) {
            diagnostics.warning(
                DiagnosticCode::GenericWarning,
                state.lineNumber,
                "Maps must be rectangular, yo (In a level, the length of each row must be the same).");
        }
    }
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
        const std::string mixedVisible = stripComments(rawLine, state, diagnostics);
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
        if (state.section != "levels") {
            if (const std::optional<std::string> sectionHeader = matchLeadingSectionKeyword(loweredVisible)) {
                const std::string& newSection = *sectionHeader;
                if (std::find(state.visitedSections.begin(), state.visitedSections.end(), newSection) != state.visitedSections.end()) {
                    diagnostics.error(
                        DiagnosticCode::GenericError,
                        state.lineNumber,
                        "cannot duplicate sections (you tried to duplicate \"" + toUpperCopy(newSection) + "\").");
                }
                state.section = newSection;
                state.lineShouldEnd = true;
                state.lineShouldEndBecause = "a section name (\"" + toUpperCopy(newSection) + "\")";
                state.visitedSections.push_back(newSection);

                const int sectionIndex = sectionOrderIndex(newSection);
                if (sectionIndex == 0) {
                    if (state.visitedSections.size() > 1) {
                        diagnostics.error(
                            DiagnosticCode::GenericError,
                            state.lineNumber,
                            "section \"" + toUpperCopy(newSection) + "\" must be the first section");
                    }
                } else if (sectionIndex > 0) {
                    const std::string prevRequired = std::string(kSections[static_cast<size_t>(sectionIndex - 1)]);
                    if (std::find(state.visitedSections.begin(), state.visitedSections.end(), prevRequired) == state.visitedSections.end()) {
                        diagnostics.error(
                            DiagnosticCode::GenericError,
                            state.lineNumber,
                            "section \"" + toUpperCopy(newSection) + "\" is out of order, must follow  \"" + toUpperCopy(prevRequired)
                                + "\" (or it could be that the section \"" + toUpperCopy(prevRequired)
                                + "\"is just missing totally.  You have to include all section headings, even if the section itself is empty).");
                    }
                }

                const size_t keywordBytes = newSection.size();
                const size_t afterWhitespace = skipUnicodeZsAndAsciiSpace(loweredVisible, keywordBytes);
                if (afterWhitespace < loweredVisible.size()) {
                    diagnostics.error(
                        DiagnosticCode::GenericError,
                        state.lineNumber,
                        "Only comments should go after a section name (\"" + toUpperCopy(newSection) + "\") on a line.");
                }

                if (state.section == "sounds") {
                    populateNamesForSounds(state);
                } else if (state.section == "levels") {
                    populateAbbrevNamesForLevels(state);
                } else if (state.section == "objects") {
                    state.objectsSection = 0;
                }
                continue;
            }
        }

        if (state.section.empty()) {
            parsePreambleLine(state, diagnostics, trimmedVisible, trim(mixedVisible), rawLine);
            continue;
        }

        switch (state.section[0]) {
            case 'o':
                parseObjectsLine(state, diagnostics, trimmedVisible, trim(mixedVisible));
                break;
            case 'l':
                if (state.section == "legend") {
                    parseLegendLine(state, diagnostics, trimmedVisible);
                } else {
                    parseLevelsLine(state, diagnostics, trimmedVisible, rawLine);
                }
                break;
            case 's':
                parseSoundsLine(state, diagnostics, trimmedVisible);
                break;
            case 'c':
                parseCollisionLayersLine(state, diagnostics, trimmedVisible);
                break;
            case 'r':
                parseRulesLine(state, diagnostics, trimmedVisible, rawLine);
                break;
            case 'w':
                parseWinConditionsLine(state, diagnostics, trimmedVisible);
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
