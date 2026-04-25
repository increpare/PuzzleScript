#include "compiler/rule_text.hpp"

#include <cctype>
#include <stdexcept>
#include <utility>

namespace puzzlescript::compiler::ruletext {

std::vector<std::string> tokenizeRuleLine(std::string_view input) {
    std::string line(input);
    auto replaceAll = [&](const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = line.find(from, pos)) != std::string::npos) {
            line.replace(pos, from.size(), to);
            pos += to.size();
        }
    };
    replaceAll("[", " [ ");
    replaceAll("]", " ] ");
    replaceAll("|", " | ");
    replaceAll("->", " -> ");
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.front()))) {
        line.erase(line.begin());
    }
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back()))) {
        line.pop_back();
    }
    if (!line.empty() && line.front() == '+') {
        line.insert(1, " ");
    }

    std::vector<std::string> tokens;
    std::string current;
    for (const char ch : line) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }
    return tokens;
}

std::vector<std::vector<std::string>> splitTopLevelOr(const std::vector<std::string>& tokens, size_t begin) {
    std::vector<std::vector<std::string>> alternatives;
    std::vector<std::string> current;
    int bracketDepth = 0;
    for (size_t i = begin; i < tokens.size(); ++i) {
        if (tokens[i] == "[") ++bracketDepth;
        if (tokens[i] == "]") --bracketDepth;
        if (tokens[i] == "or" && bracketDepth == 0) {
            alternatives.push_back(std::move(current));
            current.clear();
            continue;
        }
        current.push_back(tokens[i]);
    }
    alternatives.push_back(std::move(current));
    return alternatives;
}

size_t findTopLevelArrow(const std::vector<std::string>& tokens, size_t begin, size_t end) {
    int bracketDepth = 0;
    for (size_t i = begin; i < end; ++i) {
        if (tokens[i] == "[") ++bracketDepth;
        if (tokens[i] == "]") --bracketDepth;
        if (tokens[i] == "->" && bracketDepth == 0) return i;
    }
    return end;
}

std::vector<BracketRow> parseBracketRows(const std::vector<std::string>& tokens, size_t begin, size_t end, bool allowEllipsis) {
    std::vector<BracketRow> rows;
    for (size_t i = begin; i < end; ++i) {
        if (tokens[i] != "[") continue;
        BracketRow row;
        std::vector<std::string> cell;
        ++i;
        while (i < end && tokens[i] != "]") {
            if (!allowEllipsis && tokens[i] == "...") {
                throw std::runtime_error("Rule cells do not support ... here");
            }
            if (tokens[i] == "|") {
                row.cells.push_back(std::move(cell));
                cell.clear();
                ++i;
                continue;
            }
            cell.push_back(tokens[i]);
            ++i;
        }
        if (i >= end) throw std::runtime_error("Unclosed [ in rule");
        row.cells.push_back(std::move(cell));
        rows.push_back(std::move(row));
    }
    return rows;
}

} // namespace puzzlescript::compiler::ruletext
