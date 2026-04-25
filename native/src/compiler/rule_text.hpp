#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace puzzlescript::compiler::ruletext {

struct BracketRow {
    std::vector<std::vector<std::string>> cells;
};

std::vector<std::string> tokenizeRuleLine(std::string_view line);
std::vector<std::vector<std::string>> splitTopLevelOr(const std::vector<std::string>& tokens, size_t begin);
size_t findTopLevelArrow(const std::vector<std::string>& tokens, size_t begin, size_t end);
std::vector<BracketRow> parseBracketRows(const std::vector<std::string>& tokens, size_t begin, size_t end, bool allowEllipsis);

} // namespace puzzlescript::compiler::ruletext
