#include "compiler/compiled_rules_codegen.hpp"

#include <algorithm>

namespace puzzlescript::compiler {

bool isCompilableReplacement(const Replacement& replacement) {
    return !replacement.hasRandomEntityMask && !replacement.hasRandomDirMask;
}

std::string compiledRuleMissReason(
    const Rule& rule,
    const CompiledRulesOptions& options,
    bool allowRandomRule
) {
    if (rule.isRandom && !allowRandomRule) {
        return "random_rule";
    }
    if (rule.patterns.empty()) {
        return "empty_row";
    }
    if (rule.patterns.size() > options.maxRows) {
        return "row_limit";
    }
    if (rule.ellipsisCount.size() < rule.patterns.size()) {
        return "missing_ellipsis_metadata";
    }
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        if (rule.patterns[rowIndex].empty()) {
            return "empty_row";
        }
        for (const auto& pattern : rule.patterns[rowIndex]) {
            if (pattern.kind == Pattern::Kind::Ellipsis) {
                continue;
            }
            if (pattern.kind != Pattern::Kind::CellPattern) {
                return "non_cell_pattern";
            }
        }
    }
    return {};
}

bool isCompilableRule(const Rule& rule, const CompiledRulesOptions& options) {
    return compiledRuleMissReason(rule, options).empty();
}

std::string compiledGroupMissReason(const std::vector<Rule>& group, const CompiledRulesOptions& options) {
    if (group.empty()) {
        return "empty_group";
    }
    const bool randomGroup = group[0].isRandom;
    for (const auto& rule : group) {
        const std::string reason = compiledRuleMissReason(rule, options, randomGroup);
        if (!reason.empty()) {
            return reason;
        }
    }
    return {};
}

bool isCompilableGroup(const std::vector<Rule>& group, const CompiledRulesOptions& options) {
    return compiledGroupMissReason(group, options).empty();
}

bool areAllGroupsCompilable(const std::vector<std::vector<Rule>>& groups, const CompiledRulesOptions& options) {
    return std::all_of(groups.begin(), groups.end(), [&](const std::vector<Rule>& group) {
        return isCompilableGroup(group, options);
    });
}

bool isKnownSpecializedFullTurnCommandName(std::string_view name) {
    if (name == "again"
        || name == "cancel"
        || name == "checkpoint"
        || name == "message"
        || name == "restart"
        || name == "win") {
        return true;
    }
    if (name.size() <= 3 || name.substr(0, 3) != "sfx") {
        return false;
    }
    return std::all_of(name.begin() + 3, name.end(), [](const char ch) {
        return ch >= '0' && ch <= '9';
    });
}

std::string specializedFullTurnCommandStatusForGroups(const std::vector<std::vector<Rule>>& groups) {
    bool sawCommand = false;
    for (const auto& group : groups) {
        for (const auto& rule : group) {
            for (const auto& command : rule.commands) {
                sawCommand = true;
                if (!isKnownSpecializedFullTurnCommandName(command.name)) {
                    return "unknown_interpreter";
                }
            }
        }
    }
    return sawCommand ? "generated_queue_interpreter_tail" : "none";
}

std::string specializedFullTurnCommandStatus(const Game& game) {
    const std::string earlyStatus = specializedFullTurnCommandStatusForGroups(game.rules);
    const std::string lateStatus = specializedFullTurnCommandStatusForGroups(game.lateRules);
    if (earlyStatus == "unknown_interpreter" || lateStatus == "unknown_interpreter") {
        return "unknown_interpreter";
    }
    if (earlyStatus == "generated_queue_interpreter_tail" || lateStatus == "generated_queue_interpreter_tail") {
        return "generated_queue_interpreter_tail";
    }
    return "none";
}

std::optional<std::string_view> compiledRuleCommandKindExpression(std::string_view commandName) {
    if (commandName == "again") return "CompiledRuleCommandKind::Again";
    if (commandName == "cancel") return "CompiledRuleCommandKind::Cancel";
    if (commandName == "checkpoint") return "CompiledRuleCommandKind::Checkpoint";
    if (commandName == "message") return "CompiledRuleCommandKind::Message";
    if (commandName == "restart") return "CompiledRuleCommandKind::Restart";
    if (commandName == "win") return "CompiledRuleCommandKind::Win";
    if (isKnownSpecializedFullTurnCommandName(commandName)
        && commandName.size() > 3
        && commandName.substr(0, 3) == "sfx") {
        return "CompiledRuleCommandKind::Output";
    }
    return std::nullopt;
}

bool canGenerateCompiledRuleCommandQueue(const Rule& rule) {
    return std::all_of(rule.commands.begin(), rule.commands.end(), [](const RuleCommand& command) {
        return compiledRuleCommandKindExpression(command.name).has_value();
    });
}

CompactTurnSupport compactNativeTurnSupportForGame(const Game& game) {
    (void)game;
    CompactTurnSupport support;
    support.fallbackReason = "native_compact_generator_rebuild";
    support.nativeFallbackReason = support.fallbackReason;
    return support;
}

CompactTurnSupport compactTurnSupportForGame(const Game& game) {
    CompactTurnSupport support = compactNativeTurnSupportForGame(game);
    support.nativeFallbackReason = support.fallbackReason;
    if (!support.supported) {
        support.supported = true;
        support.interpreterBridge = true;
        support.fallbackReason = "interpreter_bridge";
    }
    return support;
}

SpecializedFullTurnSupport specializedFullTurnSupportForGame(
    const Game& game,
    const CompiledRulesOptions& options
) {
    SpecializedFullTurnSupport support;
    support.earlyRuleLoopsGenerated = areAllGroupsCompilable(game.rules, options);
    support.lateRuleLoopsGenerated = areAllGroupsCompilable(game.lateRules, options);
    support.commandStatus = specializedFullTurnCommandStatus(game);

    if (!support.earlyRuleLoopsGenerated) {
        support.wholeTurnFallbackReason = "early_rule_loops_interpreter";
    } else if (!support.lateRuleLoopsGenerated) {
        support.wholeTurnFallbackReason = "late_rule_loops_interpreter";
    } else if (support.commandStatus == "unknown_interpreter") {
        support.wholeTurnFallbackReason = "unsupported_command";
    } else {
        support.wholeTurnFallbackReason = "movement_interpreter";
    }
    return support;
}

SpecializedFullTurnSupport specializedFullTurnSupportForMissingGame() {
    return SpecializedFullTurnSupport{};
}

} // namespace puzzlescript::compiler
