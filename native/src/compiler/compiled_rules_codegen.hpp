#pragma once

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/core.hpp"

namespace puzzlescript::compiler {

struct CompiledRulesOptions {
    size_t maxRows = 1;
};

struct SpecializedFullTurnSupport {
    // These are the generated top-level early/late rule loops: they sequence
    // rulegroups and preserve BEGIN LOOP / END LOOP loop-point jumps.
    bool earlyRuleLoopsGenerated = false;
    bool lateRuleLoopsGenerated = false;
    std::string commandStatus = "unknown_interpreter";
    bool wholeTurnSupported = false;
    std::string wholeTurnFallbackReason = "interpreter_delegation";
};

bool isCompilableReplacement(const Replacement& replacement);
std::string cppStringLiteral(std::string_view value);
std::string safeCppIdentifier(std::string_view value);
std::string compiledMaskWordLiteral(MaskWord word);
std::vector<MaskWord> compiledMaskWords(const Game& game, MaskOffset offset, uint32_t wordCount);
std::string compiledRuleMissReason(const Rule& rule, const CompiledRulesOptions& options, bool allowRandomRule = false);
bool isCompilableRule(const Rule& rule, const CompiledRulesOptions& options);
std::string compiledGroupMissReason(const std::vector<Rule>& group, const CompiledRulesOptions& options);
bool isCompilableGroup(const std::vector<Rule>& group, const CompiledRulesOptions& options);
bool areAllGroupsCompilable(const std::vector<std::vector<Rule>>& groups, const CompiledRulesOptions& options);

bool isKnownSpecializedFullTurnCommandName(std::string_view name);
std::string specializedFullTurnCommandStatusForGroups(const std::vector<std::vector<Rule>>& groups);
std::string specializedFullTurnCommandStatus(const Game& game);
std::optional<std::string_view> compiledRuleCommandKindExpression(std::string_view commandName);
bool canGenerateCompiledRuleCommandQueue(const Rule& rule);

SpecializedFullTurnSupport specializedFullTurnSupportForGame(const Game& game, const CompiledRulesOptions& options);
SpecializedFullTurnSupport specializedFullTurnSupportForMissingGame();

} // namespace puzzlescript::compiler
