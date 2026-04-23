#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace puzzlescript::compiler {

struct ParserObjectEntry {
    std::string name;
    int32_t lineNumber = 0;
    std::vector<std::string> colors;
    std::vector<std::string> spritematrix;
};

struct ParserLegendEntry {
    std::string name;
    std::vector<std::string> items;
    int32_t lineNumber = 0;
};

struct ParserSoundToken {
    std::string text;
    std::string kind;
};

struct ParserSoundEntry {
    std::vector<ParserSoundToken> tokens;
    int32_t lineNumber = 0;
};

struct ParserRuleEntry {
    std::string rule;
    int32_t lineNumber = 0;
    std::string mixedCase;
};

struct ParserWinConditionEntry {
    std::vector<std::string> tokens;
    int32_t lineNumber = 0;
};

struct ParserLevelEntry {
    bool isMessage = false;
    std::optional<int32_t> lineNumber;
    std::string message;
    std::vector<std::string> rows;
};

struct ParserState {
    std::map<std::string, ParserObjectEntry> objects;
    // parser.js: for-in over state.objects uses insertion order; sounds/levels name lists follow that.
    std::vector<std::string> objectDefinitionOrder;

    int32_t lineNumber = 0;
    int32_t commentLevel = 0;
    std::string section;
    std::vector<std::string> visitedSections;
    bool lineShouldEnd = false;
    std::string lineShouldEndBecause;
    bool solAfterComment = false;
    bool insideCell = false;
    int32_t bracketBalance = 0;
    bool arrowPassed = false;
    bool rulePrelude = true;
    std::string objectsCandname;
    int32_t objectsSection = 0;
    std::vector<std::string> objectsSpritematrix;
    std::vector<std::vector<std::string>> collisionLayers;
    int32_t tokenIndex = 0;
    std::vector<std::string> currentLineWipArray;
    std::vector<ParserLegendEntry> legendSynonyms;
    std::vector<ParserLegendEntry> legendAggregates;
    std::vector<ParserLegendEntry> legendProperties;
    std::vector<ParserSoundEntry> sounds;
    std::vector<ParserRuleEntry> rules;
    std::vector<std::string> names;
    std::vector<ParserWinConditionEntry> winconditions;
    std::vector<std::string> metadata;
    std::map<std::string, int32_t> metadataLines;
    std::map<std::string, std::string> originalCaseNames;
    std::map<std::string, int32_t> originalLineNumbers;
    std::vector<std::string> abbrevNames;
    std::vector<ParserLevelEntry> levels;
    std::string subsection;
};

} // namespace puzzlescript::compiler
