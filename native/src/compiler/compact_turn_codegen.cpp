#include "compiler/compact_turn_codegen.hpp"

#include "compiler/compiled_rules_codegen.hpp"

#include <algorithm>
#include <map>
#include <ostream>

namespace puzzlescript::compiler {

namespace {

std::string compactRulePatternUnsupportedReason(const Pattern& pattern) {
    if (pattern.kind == Pattern::Kind::Ellipsis) {
        return {};
    }
    return {};
}

std::string compactRuleCommandUnsupportedReason(const RuleCommand& command) {
    if (command.name == "again"
        || command.name == "message"
        || command.name == "cancel"
        || command.name == "checkpoint"
        || command.name == "restart"
        || command.name == "win"
        || command.name.rfind("sfx", 0) == 0) {
        return {};
    }
    return "command_" + command.name;
}

std::string compactRuleUnsupportedReason(const Rule& rule) {
    for (const RuleCommand& command : rule.commands) {
        const std::string reason = compactRuleCommandUnsupportedReason(command);
        if (!reason.empty()) {
            return reason;
        }
    }
    if (rule.patterns.empty()) {
        return "empty_rule";
    }
    if (rule.ellipsisCount.size() < rule.patterns.size()) {
        return "missing_ellipsis_metadata";
    }
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::vector<Pattern>& row = rule.patterns[rowIndex];
        if (row.empty()) {
            return "empty_row";
        }
        for (const Pattern& pattern : row) {
            const std::string reason = compactRulePatternUnsupportedReason(pattern);
            if (!reason.empty()) {
                return reason;
            }
        }
    }
    return {};
}

bool isCompactRuleSupported(const Rule& rule) {
    return compactRuleUnsupportedReason(rule).empty();
}

bool canCompactCompilerHandleTurn(const Game& game) {
    (void)game;
    return true;
}

bool hasAnyRulegroups(const std::vector<std::vector<Rule>>& groups) {
    return std::any_of(groups.begin(), groups.end(), [](const std::vector<Rule>& group) {
        return !group.empty();
    });
}

void emitMaskArray(
    std::ostream& out,
    const std::string& name,
    const std::vector<MaskWord>& words
) {
    out << "constexpr MaskWord " << name << "[] = {";
    for (size_t word = 0; word < words.size(); ++word) {
        if (word > 0) out << ", ";
        out << compiledMaskWordLiteral(words[word]);
    }
    out << "};\n";
}

class CompactMaskConstantEmitter {
public:
    CompactMaskConstantEmitter(std::ostream& out, std::string_view suffix, std::string_view phase)
        : out_(out)
        , suffix_(suffix)
        , phase_(phase) {}

    void emitAlias(const std::string& alias, const std::vector<MaskWord>& words) {
        const std::string& canonical = canonicalName(words);
        out_ << "constexpr const MaskWord* " << alias << " = " << canonical << ";\n";
    }

private:
    const std::string& canonicalName(const std::vector<MaskWord>& words) {
        auto existing = names_.find(words);
        if (existing != names_.end()) {
            return existing->second;
        }

        std::string name = "compact_turn_mask_data_" + std::string(suffix_) + "_" + std::string(phase_) + "_" + std::to_string(nextIndex_++);
        emitMaskArray(out_, name, words);
        auto inserted = names_.emplace(words, std::move(name));
        return inserted.first->second;
    }

    std::ostream& out_;
    std::string_view suffix_;
    std::string_view phase_;
    size_t nextIndex_ = 0;
    std::map<std::vector<MaskWord>, std::string> names_;
};

std::string compactPatternPrefix(
    std::string_view suffix,
    std::string_view phase,
    size_t groupIndex,
    size_t ruleIndex,
    size_t rowIndex,
    size_t patternIndex
) {
    return "compact_turn_rule_" + std::string(suffix)
        + "_" + std::string(phase)
        + "_" + std::to_string(groupIndex)
        + "_" + std::to_string(ruleIndex)
        + "_" + std::to_string(rowIndex)
        + "_" + std::to_string(patternIndex);
}

std::string compactRulePrefix(
    std::string_view suffix,
    std::string_view phase,
    size_t groupIndex,
    size_t ruleIndex
) {
    return "compact_turn_rule_" + std::string(suffix)
        + "_" + std::string(phase)
        + "_" + std::to_string(groupIndex)
        + "_" + std::to_string(ruleIndex);
}

std::string compactRowPrefix(
    std::string_view suffix,
    std::string_view phase,
    size_t groupIndex,
    size_t ruleIndex,
    size_t rowIndex
) {
    return compactRulePrefix(suffix, phase, groupIndex, ruleIndex)
        + "_row_" + std::to_string(rowIndex);
}

std::string compactGroupPrefix(
    std::string_view suffix,
    std::string_view phase,
    size_t groupIndex
) {
    return "compact_turn_group_" + std::string(suffix)
        + "_" + std::string(phase)
        + "_" + std::to_string(groupIndex);
}

void emitCompactRuleMaskData(
    std::ostream& out,
    const Game& game,
    std::string_view suffix,
    std::string_view phase,
    const std::vector<std::vector<Rule>>& groups
) {
    CompactMaskConstantEmitter masks(out, suffix, phase);
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const std::vector<Rule>& group = groups[groupIndex];
        for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
            const Rule& rule = group[ruleIndex];
            if (!isCompactRuleSupported(rule)) {
                continue;
            }
            for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
                const std::vector<Pattern>& row = rule.patterns[rowIndex];
                for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
                    const Pattern& pattern = row[patternIndex];
                    if (pattern.kind == Pattern::Kind::Ellipsis) {
                        continue;
                    }
                    const std::string prefix = compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex);
                    masks.emitAlias(prefix + "_objects_present", compiledMaskWords(game, pattern.objectsPresent, game.wordCount));
                    masks.emitAlias(prefix + "_objects_missing", compiledMaskWords(game, pattern.objectsMissing, game.wordCount));
                    masks.emitAlias(prefix + "_movements_present", compiledMaskWords(game, pattern.movementsPresent, game.movementWordCount));
                    masks.emitAlias(prefix + "_movements_missing", compiledMaskWords(game, pattern.movementsMissing, game.movementWordCount));
                    for (uint32_t anyIndex = 0; anyIndex < pattern.anyObjectsCount; ++anyIndex) {
                        const MaskOffset offset = game.anyObjectOffsets[pattern.anyObjectsFirst + anyIndex];
                        masks.emitAlias(
                            prefix + "_any_objects_" + std::to_string(anyIndex),
                            compiledMaskWords(game, offset, game.wordCount)
                        );
                    }
                    if (pattern.replacement.has_value()) {
                        const Replacement& replacement = *pattern.replacement;
                        masks.emitAlias(prefix + "_objects_clear", compiledMaskWords(game, replacement.objectsClear, game.wordCount));
                        masks.emitAlias(prefix + "_objects_set", compiledMaskWords(game, replacement.objectsSet, game.wordCount));
                        masks.emitAlias(prefix + "_movements_clear", compiledMaskWords(game, replacement.movementsClear, game.movementWordCount));
                        masks.emitAlias(prefix + "_movements_set", compiledMaskWords(game, replacement.movementsSet, game.movementWordCount));
                        masks.emitAlias(prefix + "_movements_layer_mask", compiledMaskWords(game, replacement.movementsLayerMask, game.movementWordCount));
                        if (replacement.hasRandomEntityMask) {
                            out << "constexpr int32_t " << prefix << "_random_entity_choices[] = {";
                            if (replacement.randomEntityChoices.empty()) {
                                out << "0";
                            } else {
                                for (size_t choiceIndex = 0; choiceIndex < replacement.randomEntityChoices.size(); ++choiceIndex) {
                                    if (choiceIndex > 0) out << ", ";
                                    out << replacement.randomEntityChoices[choiceIndex];
                                }
                            }
                            out << "};\n";
                            out << "constexpr size_t " << prefix << "_random_entity_choice_count = "
                                << replacement.randomEntityChoices.size() << ";\n";
                        }
                        if (replacement.hasRandomDirMask) {
                            out << "constexpr int32_t " << prefix << "_random_dir_layers[] = {";
                            if (replacement.randomDirLayers.empty()) {
                                out << "0";
                            } else {
                                for (size_t layerIndex = 0; layerIndex < replacement.randomDirLayers.size(); ++layerIndex) {
                                    if (layerIndex > 0) out << ", ";
                                    out << replacement.randomDirLayers[layerIndex];
                                }
                            }
                            out << "};\n";
                            out << "constexpr size_t " << prefix << "_random_dir_layer_count = "
                                << replacement.randomDirLayers.size() << ";\n";
                        }
                    }
                }
            }
        }
    }
}

void emitCompactPatternFunctions(
    std::ostream& out,
    const Pattern& pattern,
    std::string_view suffix,
    std::string_view phase,
    size_t groupIndex,
    size_t ruleIndex,
    size_t rowIndex,
    size_t patternIndex
) {
    const std::string prefix = compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex);
    if (pattern.kind == Pattern::Kind::Ellipsis) {
        return;
    }
    out << "bool " << prefix << "_matches(const PersistentLevelState& levelState, const Scratch& scratch, int32_t tileIndex) {\n"
        << "    const MaskWord* objects = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "    const MaskWord* movements = compact_turn_cell_movements_" << suffix << "(scratch, tileIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        if ((objects[word] & " << prefix << "_objects_present[word]) != " << prefix << "_objects_present[word]) return false;\n"
        << "        if ((objects[word] & " << prefix << "_objects_missing[word]) != 0) return false;\n"
        << "    }\n";
    for (uint32_t anyIndex = 0; anyIndex < pattern.anyObjectsCount; ++anyIndex) {
        out << "    if (!compact_turn_cell_any_objects_" << suffix
            << "(levelState, tileIndex, " << prefix << "_any_objects_" << anyIndex << ")) return false;\n";
    }
    out << "    for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "        if ((movements[word] & " << prefix << "_movements_present[word]) != " << prefix << "_movements_present[word]) return false;\n"
        << "        if ((movements[word] & " << prefix << "_movements_missing[word]) != 0) return false;\n"
        << "    }\n"
        << "    return true;\n"
        << "}\n\n";

    if (!pattern.replacement.has_value()) {
        return;
    }

    out << "bool " << prefix << "_apply(PersistentLevelState& levelState, Scratch& scratch, int32_t tileIndex, int32_t rigidGroupIndex) {\n";
    out << "    bool changed = false;\n"
        << "    bool rigidChange = false;\n"
        << "    MaskWord objectsClear[compact_turn_object_stride_" << suffix << "] = {};\n"
        << "    MaskWord objectsSet[compact_turn_object_stride_" << suffix << "] = {};\n"
        << "    MaskWord movementsClear[compact_turn_movement_stride_" << suffix << "] = {};\n"
        << "    MaskWord movementsSet[compact_turn_movement_stride_" << suffix << "] = {};\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        objectsClear[word] = " << prefix << "_objects_clear[word];\n"
        << "        objectsSet[word] = " << prefix << "_objects_set[word];\n"
        << "    }\n"
        << "    for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "        movementsClear[word] = " << prefix << "_movements_clear[word];\n"
        << "        movementsSet[word] = " << prefix << "_movements_set[word];\n"
        << "    }\n";
    const Replacement& replacement = *pattern.replacement;
    if (replacement.hasRandomEntityMask) {
        out << "    if (" << prefix << "_random_entity_choice_count > 0) {\n"
            << "        const double randomValue = compact_turn_random_uniform_" << suffix << "(levelState.rng);\n"
            << "        const size_t chosenIndex = std::min(" << prefix << "_random_entity_choice_count - 1, static_cast<size_t>(randomValue * static_cast<double>(" << prefix << "_random_entity_choice_count)));\n"
            << "        const int32_t objectId = " << prefix << "_random_entity_choices[chosenIndex];\n"
            << "        if (objectId >= 0 && objectId < compact_turn_object_count_" << suffix << ") {\n"
            << "            const uint32_t objectBit = static_cast<uint32_t>(objectId);\n"
            << "            objectsSet[maskWordIndex(objectBit)] |= maskBit(objectBit);\n"
            << "            const int32_t layer = compact_turn_object_layer_" << suffix << "[objectId];\n"
            << "            const MaskWord* layerMask = compact_turn_layer_mask_" << suffix << "(layer);\n"
            << "            if (layerMask != nullptr) {\n"
            << "                for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) objectsClear[word] |= layerMask[word];\n"
            << "                compact_turn_set_layer_bits_" << suffix << "(movementsClear, layer, 0x1f);\n"
            << "            }\n"
            << "        }\n"
            << "    }\n";
    }
    if (replacement.hasRandomDirMask) {
        out << "    for (size_t randomLayerIndex = 0; randomLayerIndex < " << prefix << "_random_dir_layer_count; ++randomLayerIndex) {\n"
            << "        const int32_t layer = " << prefix << "_random_dir_layers[randomLayerIndex];\n"
            << "        const double randomValue = compact_turn_random_uniform_" << suffix << "(levelState.rng);\n"
            << "        const int32_t randomDir = std::min(3, static_cast<int32_t>(randomValue * 4.0));\n"
            << "        const int32_t beforeBits = compact_turn_layer_bits_" << suffix << "(movementsSet, layer);\n"
            << "        compact_turn_set_layer_bits_" << suffix << "(movementsSet, layer, beforeBits | (1 << randomDir));\n"
            << "    }\n";
    }
    out
        << "    for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "        movementsClear[word] |= " << prefix << "_movements_layer_mask[word];\n"
        << "    }\n"
        << "    MaskWord* objects = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        const MaskWord before = objects[word];\n"
        << "        const MaskWord after = (before & ~objectsClear[word]) | objectsSet[word];\n"
        << "        objects[word] = after;\n"
        << "        changed = changed || before != after;\n"
        << "    }\n"
        << "    MaskWord* movements = compact_turn_cell_movements_" << suffix << "(scratch, tileIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "        const MaskWord before = movements[word];\n"
        << "        const MaskWord after = (before & ~movementsClear[word]) | movementsSet[word];\n"
        << "        movements[word] = after;\n"
        << "        changed = changed || before != after;\n"
        << "    }\n"
        << "    if (rigidGroupIndex > 0) {\n"
        << "        MaskWord rigidMask[compact_turn_movement_stride_" << suffix << "] = {};\n"
        << "        for (int32_t layer = 0; layer < compact_turn_layer_count_" << suffix << "; ++layer) {\n"
        << "            if (compact_turn_layer_bits_" << suffix << "(" << prefix << "_movements_layer_mask, layer) != 0) {\n"
        << "                compact_turn_set_layer_bits_" << suffix << "(rigidMask, layer, rigidGroupIndex);\n"
        << "            }\n"
        << "        }\n"
        << "        MaskWord* rigidGroupMask = compact_turn_cell_rigid_group_index_" << suffix << "(scratch, tileIndex);\n"
        << "        MaskWord* rigidAppliedMask = compact_turn_cell_rigid_movement_applied_" << suffix << "(scratch, tileIndex);\n"
        << "        bool rigidGroupAlreadySet = true;\n"
        << "        bool rigidMovementAlreadySet = true;\n"
        << "        for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "            if ((rigidGroupMask[word] & rigidMask[word]) != rigidMask[word]) rigidGroupAlreadySet = false;\n"
        << "            if ((rigidAppliedMask[word] & " << prefix << "_movements_layer_mask[word]) != " << prefix << "_movements_layer_mask[word]) rigidMovementAlreadySet = false;\n"
        << "        }\n"
        << "        if (!rigidGroupAlreadySet && !rigidMovementAlreadySet) {\n"
        << "            for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "                rigidGroupMask[word] |= rigidMask[word];\n"
        << "                rigidAppliedMask[word] |= " << prefix << "_movements_layer_mask[word];\n"
        << "            }\n"
        << "            rigidChange = true;\n"
        << "        }\n"
        << "    }\n"
        << "    changed = changed || rigidChange;\n"
        << "    if (changed) scratch.objectCellIndexDirty = true;\n"
        << "    return changed;\n"
        << "}\n\n";
}

void emitCompactRuleCommandQueue(
    std::ostream& out,
    const Rule& rule,
    const std::string& prefix
) {
    if (rule.commands.empty()) {
        return;
    }
    out << "    " << prefix << "_queue_commands(commands);\n";
}

void emitCompactRuleCommandFunction(
    std::ostream& out,
    const Rule& rule,
    const std::string& prefix,
    std::string_view suffix
) {
    out << "void " << prefix << "_queue_commands(CompactTurnCommands_" << suffix << "& commands) {\n";
    if (rule.commands.empty()) {
        out << "    (void)commands;\n"
            << "}\n\n";
        return;
    }
    const bool currentRuleCancel = std::any_of(rule.commands.begin(), rule.commands.end(), [](const RuleCommand& command) {
        return command.name == "cancel";
    });
    const bool currentRuleRestart = std::any_of(rule.commands.begin(), rule.commands.end(), [](const RuleCommand& command) {
        return command.name == "restart";
    });

    out << "    if (commands.hasCancel) {\n"
        << "        return;\n"
        << "    }\n";
    if (!currentRuleCancel) {
        out << "    if (commands.hasRestart) {\n"
            << "        return;\n"
            << "    }\n";
    }
    if (currentRuleCancel || currentRuleRestart) {
        out << "    commands = CompactTurnCommands_" << suffix << "{};\n";
    }
    out << "    commands.any = true;\n";
    for (const RuleCommand& command : rule.commands) {
        if (command.name == "again") {
            out << "    if (!commands.hasAgain) ++commands.commandCount;\n"
                << "    commands.hasAgain = true;\n";
        } else if (command.name == "cancel") {
            out << "    if (!commands.hasCancel) ++commands.commandCount;\n"
                << "    commands.hasCancel = true;\n";
        } else if (command.name == "checkpoint") {
            out << "    if (!commands.hasCheckpoint) ++commands.commandCount;\n"
                << "    commands.hasCheckpoint = true;\n";
        } else if (command.name == "restart") {
            out << "    if (!commands.hasRestart) ++commands.commandCount;\n"
                << "    commands.hasRestart = true;\n";
        } else if (command.name == "win") {
            out << "    if (!commands.hasWin) ++commands.commandCount;\n"
                << "    commands.hasWin = true;\n";
        } else if (command.name == "message") {
            if (command.argument.has_value()) {
                out << "    if (!commands.hasMessage) {\n"
                    << "        ++commands.commandCount;\n"
                    << "        commands.hasMessage = true;\n"
                    << "        commands.messageText = " << cppStringLiteral(*command.argument) << ";\n"
                    << "    }\n";
            } else {
                out << "    if (!commands.hasMessage) ++commands.commandCount;\n"
                    << "    commands.hasMessage = true;\n";
            }
        } else if (command.name.rfind("sfx", 0) == 0) {
            out << "    ++commands.commandCount;\n"
                << "    // Sound effects are command output only; board/search state is unaffected.\n";
        } else {
            out << "    static_assert(false, \"compact turn compiler command queue emitted unsupported command\");\n";
        }
    }
    out << "}\n\n";
}

void emitCompactRuleFunction(
    std::ostream& out,
    const Game& game,
    const Rule& rule,
    std::string_view suffix,
    std::string_view phase,
    size_t groupIndex,
    size_t ruleIndex
) {
    const std::string prefix = compactRulePrefix(suffix, phase, groupIndex, ruleIndex);
    const int32_t rigidGroupIndex = (rule.rigid
        && rule.groupNumber >= 0
        && static_cast<size_t>(rule.groupNumber) < game.groupNumberToRigidGroupIndex.size())
        ? game.groupNumberToRigidGroupIndex[static_cast<size_t>(rule.groupNumber)] + 1
        : 0;
    if (!isCompactRuleSupported(rule)) {
        const std::string reason = "compact turn compiler TODO: "
            + compactRuleUnsupportedReason(rule)
            + " at source rule line "
            + std::to_string(rule.lineNumber);
        out << "bool " << prefix << "_apply(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch, CompactTurnCommands_" << suffix << "& commands) {\n"
            << "    (void)dimensions;\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    (void)commands;\n"
            << "    static_assert(false, " << cppStringLiteral(reason) << ");\n"
            << "    return false;\n"
            << "}\n\n"
            << "void " << prefix << "_queue_commands(CompactTurnCommands_" << suffix << "& commands) {\n"
            << "    (void)commands;\n"
            << "    static_assert(false, " << cppStringLiteral(reason) << ");\n"
            << "}\n\n"
            << "bool " << prefix << "_collect_matches(LevelDimensions dimensions, const PersistentLevelState& levelState, const Scratch& scratch, std::vector<std::vector<std::vector<int32_t>>>& matches) {\n"
            << "    (void)dimensions;\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    (void)matches;\n"
            << "    static_assert(false, " << cppStringLiteral(reason) << ");\n"
            << "    return false;\n"
            << "}\n\n"
            << "bool " << prefix << "_apply_tuple(PersistentLevelState& levelState, Scratch& scratch, const std::vector<std::vector<std::vector<int32_t>>>& matches, const std::vector<size_t>& tupleIndex) {\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    (void)matches;\n"
            << "    (void)tupleIndex;\n"
            << "    static_assert(false, " << cppStringLiteral(reason) << ");\n"
            << "    return false;\n"
            << "}\n\n";
        return;
    }

    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::vector<Pattern>& row = rule.patterns[rowIndex];
        for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
            emitCompactPatternFunctions(out, row[patternIndex], suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex);
        }
    }

    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::vector<Pattern>& row = rule.patterns[rowIndex];
        const std::string rowPrefix = compactRowPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex);
        out << "bool " << rowPrefix << "_match_still_matches(const PersistentLevelState& levelState, const Scratch& scratch, const std::vector<int32_t>& match) {\n"
            << "    size_t positionIndex = 0;\n";
        for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
            if (row[patternIndex].kind == Pattern::Kind::Ellipsis) {
                continue;
            }
            out << "    if (positionIndex >= match.size() || !"
                << compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex)
                << "_matches(levelState, scratch, match[positionIndex])) return false;\n"
                << "    ++positionIndex;\n";
        }
        out << "    return positionIndex == match.size();\n"
            << "}\n\n";

        out << "bool " << rowPrefix << "_apply_replacements(PersistentLevelState& levelState, Scratch& scratch, const std::vector<int32_t>& match) {\n"
            << "    bool changed = false;\n"
            << "    size_t positionIndex = 0;\n";
        for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
            if (row[patternIndex].kind == Pattern::Kind::Ellipsis) {
                continue;
            }
            out << "    if (positionIndex >= match.size()) return changed;\n";
            if (row[patternIndex].replacement.has_value()) {
                out << "    changed = " << compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex)
                    << "_apply(levelState, scratch, match[positionIndex], " << rigidGroupIndex << ") || changed;\n";
            }
            out << "    ++positionIndex;\n";
        }
        out << "    return changed;\n"
            << "}\n\n";

        out << "bool " << rowPrefix << "_collect_matches(LevelDimensions dimensions, const PersistentLevelState& levelState, const Scratch& scratch, std::vector<std::vector<int32_t>>& rowMatches) {\n"
            << "    rowMatches.clear();\n"
            << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
            << "    if (tileCount <= 0) return false;\n"
            << "    constexpr bool horizontalScan = " << (rule.direction > 2 ? "true" : "false") << ";\n"
            << "    const int32_t primaryLimit = horizontalScan ? dimensions.height : dimensions.width;\n"
            << "    const int32_t secondaryLimit = horizontalScan ? dimensions.width : dimensions.height;\n";
        if (rule.ellipsisCount[rowIndex] == 0) {
            out << "    for (int32_t primary = 0; primary < primaryLimit; ++primary) {\n"
                << "    for (int32_t secondary = 0; secondary < secondaryLimit; ++secondary) {\n"
                << "        const int32_t x = horizontalScan ? secondary : primary;\n"
                << "        const int32_t y = horizontalScan ? primary : secondary;\n"
                << "        const int32_t startIndex = compact_turn_tile_index_" << suffix << "(dimensions, x, y);\n"
                << "        std::vector<int32_t> positions;\n"
                << "        bool matched = true;\n";
            for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
                out << "        int32_t tile_" << patternIndex << " = 0;\n"
                    << "        if (!compact_turn_cell_at_direction_" << suffix
                    << "(dimensions, startIndex, " << rule.direction << ", " << patternIndex << ", tile_" << patternIndex << ")) matched = false;\n"
                    << "        if (matched && !" << compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex)
                    << "_matches(levelState, scratch, tile_" << patternIndex << ")) matched = false;\n"
                    << "        if (matched) positions.push_back(tile_" << patternIndex << ");\n";
            }
            out << "        if (matched) rowMatches.push_back(positions);\n"
                << "    }\n"
                << "    }\n";
        } else {
            int32_t concreteCount = 0;
            std::vector<int32_t> concreteSuffix(row.size() + 1, 0);
            for (int32_t patternIndex = static_cast<int32_t>(row.size()) - 1; patternIndex >= 0; --patternIndex) {
                concreteSuffix[static_cast<size_t>(patternIndex)] = concreteSuffix[static_cast<size_t>(patternIndex + 1)]
                    + (row[static_cast<size_t>(patternIndex)].kind == Pattern::Kind::Ellipsis ? 0 : 1);
            }
            for (const Pattern& pattern : row) {
                if (pattern.kind != Pattern::Kind::Ellipsis) {
                    ++concreteCount;
                }
            }
            out << "    for (int32_t primary = 0; primary < primaryLimit; ++primary) {\n"
                << "    for (int32_t secondary = 0; secondary < secondaryLimit; ++secondary) {\n"
                << "        const int32_t x = horizontalScan ? secondary : primary;\n"
                << "        const int32_t y = horizontalScan ? primary : secondary;\n"
                << "        const int32_t startIndex = compact_turn_tile_index_" << suffix << "(dimensions, x, y);\n"
                << "        const int32_t available = compact_turn_available_at_direction_" << suffix
                << "(dimensions, startIndex, " << rule.direction << ");\n"
                << "        if (available < " << concreteCount << ") continue;\n"
                << "        std::vector<int32_t> positions;\n"
                << "        auto search = [&](auto&& self, size_t patternIndex, int32_t offset) -> void {\n"
                << "            if (patternIndex >= " << row.size() << ") {\n"
                << "                rowMatches.push_back(positions);\n"
                << "                return;\n"
                << "            }\n"
                << "            switch (patternIndex) {\n";
            for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
                out << "                case " << patternIndex << ":\n";
                if (row[patternIndex].kind == Pattern::Kind::Ellipsis) {
                    out << "                {\n"
                        << "                    const int32_t maxSkip = available - offset - " << concreteSuffix[patternIndex + 1] << ";\n"
                        << "                    for (int32_t skip = 0; skip <= maxSkip; ++skip) {\n"
                        << "                        self(self, patternIndex + 1, offset + skip);\n"
                        << "                    }\n"
                        << "                    return;\n"
                        << "                }\n";
                } else {
                    out << "                {\n"
                        << "                    if (offset >= available) return;\n"
                        << "                    int32_t tileIndex = 0;\n"
                        << "                    if (!compact_turn_cell_at_direction_" << suffix
                        << "(dimensions, startIndex, " << rule.direction << ", offset, tileIndex)) return;\n"
                        << "                    if (!" << compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex)
                        << "_matches(levelState, scratch, tileIndex)) return;\n"
                        << "                    positions.push_back(tileIndex);\n"
                        << "                    self(self, patternIndex + 1, offset + 1);\n"
                        << "                    positions.pop_back();\n"
                        << "                    return;\n"
                        << "                }\n";
                }
            }
            out << "                default:\n"
                << "                    return;\n"
                << "            }\n"
                << "        };\n"
                << "        search(search, 0, 0);\n"
                << "    }\n"
                << "    }\n";
        }
        out << "    return !rowMatches.empty();\n"
            << "}\n\n";
    }
    emitCompactRuleCommandFunction(out, rule, prefix, suffix);

    out << "bool " << prefix << "_collect_matches(LevelDimensions dimensions, const PersistentLevelState& levelState, const Scratch& scratch, std::vector<std::vector<std::vector<int32_t>>>& matches) {\n"
        << "    constexpr size_t rowCount = " << rule.patterns.size() << ";\n"
        << "    matches.assign(rowCount, std::vector<std::vector<int32_t>>{});\n";
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::string rowPrefix = compactRowPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex);
        out << "    if (!" << rowPrefix << "_collect_matches(dimensions, levelState, scratch, matches[" << rowIndex << "])) return false;\n";
    }
    out << "    return true;\n"
        << "}\n\n";

    out << "bool " << prefix << "_tuple_still_matches(const PersistentLevelState& levelState, const Scratch& scratch, const std::vector<std::vector<std::vector<int32_t>>>& matches, const std::vector<size_t>& tupleIndex) {\n";
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::string rowPrefix = compactRowPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex);
        out << "    if (!" << rowPrefix
            << "_match_still_matches(levelState, scratch, matches[" << rowIndex << "][tupleIndex[" << rowIndex << "]])) return false;\n";
    }
    out << "    return true;\n"
        << "}\n\n";

    out << "bool " << prefix << "_apply_tuple(PersistentLevelState& levelState, Scratch& scratch, const std::vector<std::vector<std::vector<int32_t>>>& matches, const std::vector<size_t>& tupleIndex) {\n"
        << "    bool changed = false;\n";
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::string rowPrefix = compactRowPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex);
        out << "    changed = " << rowPrefix
            << "_apply_replacements(levelState, scratch, matches[" << rowIndex << "][tupleIndex[" << rowIndex << "]]) || changed;\n";
    }
    out << "    return changed;\n"
        << "}\n\n";

    out << "bool " << prefix << "_apply(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch, CompactTurnCommands_" << suffix << "& commands) {\n"
        << "    constexpr size_t rowCount = " << rule.patterns.size() << ";\n"
        << "    std::vector<std::vector<std::vector<int32_t>>> matches;\n"
        << "    if (!" << prefix << "_collect_matches(dimensions, levelState, scratch, matches)) return false;\n";
    emitCompactRuleCommandQueue(out, rule, prefix);
    out << "    std::vector<size_t> tupleIndex(rowCount, 0);\n"
        << "    bool firstTuple = true;\n"
        << "    bool changed = false;\n"
        << "    while (true) {\n"
        << "        bool stillMatches = true;\n"
        << "        if (!firstTuple) {\n";
    out << "            stillMatches = " << prefix << "_tuple_still_matches(levelState, scratch, matches, tupleIndex);\n";
    out << "        }\n"
        << "        if (stillMatches) {\n";
    out << "            changed = " << prefix << "_apply_tuple(levelState, scratch, matches, tupleIndex) || changed;\n";
    out << "        }\n"
        << "        firstTuple = false;\n"
        << "        size_t rowToIncrement = 0;\n"
        << "        while (rowToIncrement < rowCount) {\n"
        << "            ++tupleIndex[rowToIncrement];\n"
        << "            if (tupleIndex[rowToIncrement] < matches[rowToIncrement].size()) break;\n"
        << "            tupleIndex[rowToIncrement] = 0;\n"
        << "            ++rowToIncrement;\n"
        << "        }\n"
        << "        if (rowToIncrement == rowCount) break;\n"
        << "    }\n"
        << "    return changed;\n"
        << "}\n\n";
}

void emitCompactRulegroupFunctions(
    std::ostream& out,
    const Game& game,
    const std::vector<std::vector<Rule>>& groups,
    const LoopPointTable& loopPoint,
    std::string_view suffix,
    std::string_view phase
) {
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const std::vector<Rule>& group = groups[groupIndex];
        for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
            emitCompactRuleFunction(out, game, group[ruleIndex], suffix, phase, groupIndex, ruleIndex);
        }

        const std::string groupPrefix = compactGroupPrefix(suffix, phase, groupIndex);
        out << "bool " << groupPrefix << "_apply(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch, CompactTurnCommands_" << suffix << "& commands, std::vector<bool>* bannedGroups) {\n";
        if (group.empty()) {
            out << "    (void)dimensions;\n"
                << "    (void)levelState;\n"
                << "    (void)scratch;\n"
                << "    (void)commands;\n"
                << "    (void)bannedGroups;\n"
                << "    return false;\n"
                << "}\n\n";
            continue;
        }
        out << "    if (bannedGroups != nullptr && " << groupIndex << " < bannedGroups->size() && (*bannedGroups)[" << groupIndex << "]) return false;\n";
        if (group[0].isRandom) {
            out << "    struct Candidate {\n"
                << "        size_t ruleIndex = 0;\n"
                << "        std::vector<size_t> tupleIndex;\n"
                << "    };\n"
                << "    std::vector<std::vector<std::vector<std::vector<int32_t>>>> groupMatches(" << group.size() << ");\n"
                << "    std::vector<Candidate> candidates;\n";
            for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
                const std::string rulePrefix = compactRulePrefix(suffix, phase, groupIndex, ruleIndex);
                out << "    if (" << rulePrefix << "_collect_matches(dimensions, levelState, scratch, groupMatches[" << ruleIndex << "])) {\n"
                    << "        bool hasMatchTuple = !groupMatches[" << ruleIndex << "].empty();\n"
                    << "        for (const auto& rowMatches : groupMatches[" << ruleIndex << "]) {\n"
                    << "            if (rowMatches.empty()) {\n"
                    << "                hasMatchTuple = false;\n"
                    << "                break;\n"
                    << "            }\n"
                    << "        }\n"
                    << "        if (hasMatchTuple) {\n"
                    << "            std::vector<size_t> tupleIndex(groupMatches[" << ruleIndex << "].size(), 0);\n"
                    << "            while (true) {\n"
                    << "                candidates.push_back(Candidate{" << ruleIndex << ", tupleIndex});\n"
                    << "                size_t rowToIncrement = 0;\n"
                    << "                while (rowToIncrement < groupMatches[" << ruleIndex << "].size()) {\n"
                    << "                    ++tupleIndex[rowToIncrement];\n"
                    << "                    if (tupleIndex[rowToIncrement] < groupMatches[" << ruleIndex << "][rowToIncrement].size()) break;\n"
                    << "                    tupleIndex[rowToIncrement] = 0;\n"
                    << "                    ++rowToIncrement;\n"
                    << "                }\n"
                    << "                if (rowToIncrement == groupMatches[" << ruleIndex << "].size()) break;\n"
                    << "            }\n"
                    << "        }\n"
                    << "    }\n";
            }
            out << "    if (candidates.empty()) return false;\n"
                << "    const double randomValue = compact_turn_random_uniform_" << suffix << "(levelState.rng);\n"
                << "    const size_t chosenIndex = std::min(candidates.size() - 1, static_cast<size_t>(randomValue * static_cast<double>(candidates.size())));\n"
                << "    const Candidate& chosen = candidates[chosenIndex];\n"
                << "    switch (chosen.ruleIndex) {\n";
            for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
                const std::string rulePrefix = compactRulePrefix(suffix, phase, groupIndex, ruleIndex);
                out << "        case " << ruleIndex << ":\n"
                    << "            " << rulePrefix << "_queue_commands(commands);\n"
                    << "            return " << rulePrefix << "_apply_tuple(levelState, scratch, groupMatches[" << ruleIndex << "], chosen.tupleIndex);\n";
            }
            out << "        default:\n"
                << "            return false;\n"
                << "    }\n"
                << "}\n\n";
            continue;
        }
        out << "    bool hasChanges = false;\n"
            << "    bool madeChangeThisLoop = true;\n"
            << "    int32_t loopCount = 0;\n"
            << "    while (madeChangeThisLoop && loopCount++ < 200) {\n"
            << "        madeChangeThisLoop = false;\n"
            << "        int32_t consecutiveFailures = 0;\n";
        for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
            out << "        if (" << compactRulePrefix(suffix, phase, groupIndex, ruleIndex)
                << "_apply(dimensions, levelState, scratch, commands)) {\n"
                << "            madeChangeThisLoop = true;\n"
                << "            consecutiveFailures = 0;\n"
                << "        } else {\n"
                << "            ++consecutiveFailures;\n"
                << "            if (consecutiveFailures == " << group.size() << ") break;\n"
                << "        }\n";
        }
        out << "        hasChanges = hasChanges || madeChangeThisLoop;\n"
            << "    }\n"
            << "    return hasChanges;\n"
            << "}\n\n";
    }

    out << "int32_t compact_turn_lookup_" << phase << "_loop_point_" << suffix << "(int32_t index) {\n"
        << "    switch (index) {\n";
    for (size_t index = 0; index < loopPoint.entries.size(); ++index) {
        if (loopPoint.entries[index].has_value()) {
            out << "        case " << index << ": return " << *loopPoint.entries[index] << ";\n";
        }
    }
    out << "        default: return -1;\n"
        << "    }\n"
        << "}\n\n";

    out << "bool compact_turn_apply_" << phase << "_rules_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch, CompactTurnCommands_" << suffix << "& commands, std::vector<bool>* bannedGroups) {\n";
    if (groups.empty()) {
        out << "    (void)dimensions;\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    (void)commands;\n"
            << "    (void)bannedGroups;\n"
            << "    return false;\n";
    } else {
        out << "    bool loopPropagated = false;\n"
            << "    bool changed = false;\n"
            << "    int32_t loopCount = 0;\n"
            << "    int32_t groupIndex = 0;\n"
            << "    constexpr int32_t groupCount = " << groups.size() << ";\n"
            << "    while (groupIndex < groupCount) {\n"
            << "        bool groupChanged = false;\n"
            << "        switch (groupIndex) {\n";
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            out << "            case " << groupIndex << ":\n"
                << "                groupChanged = " << compactGroupPrefix(suffix, phase, groupIndex)
                << "_apply(dimensions, levelState, scratch, commands, bannedGroups);\n"
                << "                break;\n";
        }
        out << "            default:\n"
            << "                break;\n"
            << "        }\n"
            << "        loopPropagated = groupChanged || loopPropagated;\n"
            << "        changed = groupChanged || changed;\n"
            << "        if (loopPropagated) {\n"
            << "            const int32_t target = compact_turn_lookup_" << phase << "_loop_point_" << suffix << "(groupIndex);\n"
            << "            if (target >= 0) {\n"
            << "                groupIndex = target;\n"
            << "                loopPropagated = false;\n"
            << "                if (++loopCount > 200) break;\n"
            << "                continue;\n"
            << "            }\n"
            << "        }\n"
            << "        ++groupIndex;\n"
            << "        if (groupIndex == groupCount && loopPropagated) {\n"
            << "            const int32_t target = compact_turn_lookup_" << phase << "_loop_point_" << suffix << "(groupIndex);\n"
            << "            if (target >= 0) {\n"
            << "                groupIndex = target;\n"
            << "                loopPropagated = false;\n"
            << "                if (++loopCount > 200) break;\n"
            << "            }\n"
            << "        }\n"
            << "    }\n"
            << "    return changed;\n";
    }
    out << "}\n\n";
}

void emitCompactTurnUnsupportedBody(std::ostream& out) {
    out << "    (void)game;\n"
        << "    (void)levelState;\n"
        << "    (void)scratch;\n"
        << "    (void)context;\n"
        << "    (void)options;\n"
        << "    (void)input;\n"
        << "    return {false, {}};\n";
}

void emitCompactTurnCompilerSingleBody(std::ostream& out, std::string_view suffix) {
    out << "    if (outHasAgain != nullptr) {\n"
        << "        *outHasAgain = false;\n"
        << "    }\n"
        << "    (void)options;\n"
        << "    ps_step_result result{};\n"
        << "    if (!compact_turn_prepare_state_" << suffix << "(dimensions, levelState, scratch)) {\n"
        << "        return {false, result};\n"
        << "    }\n"
        << "    if (!compact_turn_can_handle_turn_" << suffix << "()) {\n"
        << "        return {false, result};\n"
        << "    }\n"
        << "    const int32_t directionMask = compact_turn_input_direction_" << suffix << "(input);\n"
        << "    const std::vector<MaskWord> turnStartObjects = levelState.board.objects;\n"
        << "    const RandomState turnStartRng = levelState.rng;\n"
        << "    std::vector<MaskWord> turnStartMovements;\n"
        << "    std::vector<MaskWord> turnStartRigidGroupIndexMasks;\n"
        << "    std::vector<MaskWord> turnStartRigidMovementAppliedMasks;\n"
        << "    if (probeOnly) {\n"
        << "        turnStartMovements = scratch.liveMovements;\n"
        << "        if (compact_turn_has_rigid_" << suffix << ") {\n"
        << "            turnStartRigidGroupIndexMasks = scratch.rigidGroupIndexMasks;\n"
        << "            turnStartRigidMovementAppliedMasks = scratch.rigidMovementAppliedMasks;\n"
        << "        }\n"
        << "    }\n"
        << "    std::vector<int32_t> startPlayerPositions;\n"
        << "    if (directionMask != 0 && compact_turn_requires_player_movement_" << suffix << ") {\n"
        << "        startPlayerPositions = compact_turn_collect_player_positions_" << suffix << "(dimensions, levelState);\n"
        << "    }\n"
        << "    // Semantic compact turn compiler skeleton:\n"
        << "    // 1. validate level dimensions and persistent board storage\n"
        << "    // 2. decode input direction\n"
        << "    std::vector<bool> bannedGroups;\n"
        << "    CompactTurnCommands_" << suffix << " commands;\n"
        << "    bool seededInput = false;\n"
        << "    bool ruleChanged = false;\n"
        << "    bool moved = false;\n"
        << "    int32_t rigidLoopCount = 0;\n"
        << "    while (true) {\n"
        << "        commands = CompactTurnCommands_" << suffix << "{};\n"
        << "        if (rigidLoopCount > 0) {\n"
        << "            levelState.board.objects = turnStartObjects;\n"
        << "            levelState.rng = turnStartRng;\n"
        << "        }\n"
        << "        std::fill(scratch.liveMovements.begin(), scratch.liveMovements.end(), 0);\n"
        << "        if (compact_turn_has_rigid_" << suffix << ") {\n"
        << "            std::fill(scratch.rigidGroupIndexMasks.begin(), scratch.rigidGroupIndexMasks.end(), 0);\n"
        << "            std::fill(scratch.rigidMovementAppliedMasks.begin(), scratch.rigidMovementAppliedMasks.end(), 0);\n"
        << "        }\n"
        << "        seededInput = compact_turn_seed_player_movements_" << suffix << "(dimensions, levelState, scratch, directionMask);\n"
        << "        const bool ruleChangedThisPass = compact_turn_apply_early_rules_" << suffix << "(dimensions, levelState, scratch, commands, &bannedGroups);\n"
        << "    // 4. apply early rulegroups\n"
        << "        const CompactTurnMovementOutcome_" << suffix << " movementOutcome = compact_turn_resolve_movements_" << suffix << "(dimensions, levelState, scratch, &bannedGroups);\n"
        << "    // 5. resolve movement\n"
        << "        if (movementOutcome.shouldUndo && rigidLoopCount < 49) {\n"
        << "            ++rigidLoopCount;\n"
        << "            continue;\n"
        << "        }\n"
        << "        ruleChanged = ruleChangedThisPass;\n"
        << "        moved = movementOutcome.moved;\n"
        << "        break;\n"
        << "    }\n"
        << "    const bool lateRuleChanged = compact_turn_apply_late_rules_" << suffix << "(dimensions, levelState, scratch, commands, nullptr);\n"
        << "    const bool modified = levelState.board.objects != turnStartObjects;\n"
        << "    // 6. apply late rulegroups\n"
        << "    // 7. process commands and again policy\n"
        << "    if (probeOnly) {\n"
        << "        result.changed = commands.hasCancel\n"
        << "            ? commands.commandCount > 1\n"
        << "            : (modified || commands.hasWin || commands.hasRestart);\n"
        << "        levelState.board.objects = turnStartObjects;\n"
        << "        scratch.liveMovements = turnStartMovements;\n"
        << "        if (compact_turn_has_rigid_" << suffix << ") {\n"
        << "            scratch.rigidGroupIndexMasks = turnStartRigidGroupIndexMasks;\n"
        << "            scratch.rigidMovementAppliedMasks = turnStartRigidMovementAppliedMasks;\n"
        << "        }\n"
        << "        return {true, result};\n"
        << "    }\n"
        << "    if (!startPlayerPositions.empty() && !compact_turn_any_start_player_moved_" << suffix << "(levelState, startPlayerPositions)) {\n"
        << "        levelState.board.objects = turnStartObjects;\n"
        << "        levelState.rng = turnStartRng;\n"
        << "        std::fill(scratch.liveMovements.begin(), scratch.liveMovements.end(), 0);\n"
        << "        return {true, result};\n"
        << "    }\n"
        << "    if (commands.hasCancel) {\n"
        << "        levelState.board.objects = turnStartObjects;\n"
        << "        levelState.rng = turnStartRng;\n"
        << "        std::fill(scratch.liveMovements.begin(), scratch.liveMovements.end(), 0);\n"
        << "        if (compact_turn_has_rigid_" << suffix << ") {\n"
        << "            std::fill(scratch.rigidGroupIndexMasks.begin(), scratch.rigidGroupIndexMasks.end(), 0);\n"
        << "            std::fill(scratch.rigidMovementAppliedMasks.begin(), scratch.rigidMovementAppliedMasks.end(), 0);\n"
        << "        }\n"
        << "        if (outHasAgain != nullptr) {\n"
        << "            *outHasAgain = false;\n"
        << "        }\n"
        << "        result.changed = commands.any;\n"
        << "        return {true, result};\n"
        << "    }\n"
        << "    if (commands.hasRestart) {\n"
        << "        levelState.board.objects = turnStartObjects;\n"
        << "        levelState.rng = turnStartRng;\n"
        << "        std::fill(scratch.liveMovements.begin(), scratch.liveMovements.end(), 0);\n"
        << "        if (compact_turn_has_rigid_" << suffix << ") {\n"
        << "            std::fill(scratch.rigidGroupIndexMasks.begin(), scratch.rigidGroupIndexMasks.end(), 0);\n"
        << "            std::fill(scratch.rigidMovementAppliedMasks.begin(), scratch.rigidMovementAppliedMasks.end(), 0);\n"
        << "        }\n"
        << "        if (outHasAgain != nullptr) {\n"
        << "            *outHasAgain = false;\n"
        << "        }\n"
        << "        result.changed = commands.any;\n"
        << "        result.restarted = true;\n"
        << "        return {true, result};\n"
        << "    }\n"
        << "    const bool won = commands.hasWin || compact_turn_evaluate_win_" << suffix << "(dimensions, levelState);\n"
        << "    // 8. evaluate win conditions\n"
        << "    // 9. canonicalize and return result\n"
        << "    result.changed = seededInput || ruleChanged || moved || lateRuleChanged || commands.any;\n"
        << "    result.won = won;\n"
        << "    if (outHasAgain != nullptr) {\n"
        << "        bool scheduleAgain = false;\n"
        << "        if (commands.hasAgain && modified && !won) {\n"
        << "            const SpecializedCompactTurnOutcome probeOutcome = specialized_compact_turn_single_" << suffix << "(\n"
        << "                dimensions,\n"
        << "                levelState,\n"
        << "                scratch,\n"
        << "                PS_INPUT_TICK,\n"
        << "                options,\n"
        << "                nullptr,\n"
        << "                true\n"
        << "            );\n"
        << "            if (!probeOutcome.handled) {\n"
        << "                return probeOutcome;\n"
        << "            }\n"
        << "            scheduleAgain = probeOutcome.result.changed || probeOutcome.result.transitioned || probeOutcome.result.won;\n"
        << "        }\n"
        << "        *outHasAgain = scheduleAgain;\n"
        << "    }\n"
        << "    return {true, result};\n";
}

void emitCompactTurnCompilerDrainBody(std::ostream& out, std::string_view suffix) {
    out << "    bool hasAgain = false;\n"
        << "    SpecializedCompactTurnOutcome outcome = specialized_compact_turn_single_" << suffix << "(\n"
        << "        dimensions,\n"
        << "        levelState,\n"
        << "        scratch,\n"
        << "        input,\n"
        << "        options,\n"
        << "        &hasAgain,\n"
        << "        false\n"
        << "    );\n"
        << "    if (!outcome.handled || options.againPolicy != AgainPolicy::Drain) {\n"
        << "        return outcome;\n"
        << "    }\n"
        << "    constexpr int kMaxAgainIterations = 500;\n"
        << "    for (int iteration = 0; iteration < kMaxAgainIterations && hasAgain; ++iteration) {\n"
        << "        const bool terminal = outcome.result.won || outcome.result.restarted || outcome.result.transitioned;\n"
        << "        if (terminal || !outcome.result.changed) {\n"
        << "            break;\n"
        << "        }\n"
        << "        bool tickHasAgain = false;\n"
        << "        const SpecializedCompactTurnOutcome tickOutcome = specialized_compact_turn_single_" << suffix << "(\n"
        << "            dimensions,\n"
        << "            levelState,\n"
        << "            scratch,\n"
        << "            PS_INPUT_TICK,\n"
        << "            options,\n"
        << "            &tickHasAgain,\n"
        << "            false\n"
        << "        );\n"
        << "        if (!tickOutcome.handled) {\n"
        << "            return tickOutcome;\n"
        << "        }\n"
        << "        outcome.result.changed = outcome.result.changed || tickOutcome.result.changed;\n"
        << "        outcome.result.won = outcome.result.won || tickOutcome.result.won;\n"
        << "        outcome.result.restarted = outcome.result.restarted || tickOutcome.result.restarted;\n"
        << "        outcome.result.transitioned = outcome.result.transitioned || tickOutcome.result.transitioned;\n"
        << "        hasAgain = tickHasAgain;\n"
        << "        if (!tickOutcome.result.changed) {\n"
        << "            break;\n"
        << "        }\n"
        << "    }\n"
        << "    return outcome;\n";
}

std::string sourceSuffix(size_t sourceIndex) {
    return std::to_string(sourceIndex);
}

void emitCompactTurnAccessLayer(std::ostream& out, const Game& game, size_t sourceIndex) {
    const std::string suffix = sourceSuffix(sourceIndex);
    out << "constexpr int32_t compact_turn_object_stride_" << suffix << " = " << game.strideObject << ";\n"
        << "constexpr int32_t compact_turn_movement_stride_" << suffix << " = " << game.strideMovement << ";\n"
        << "constexpr int32_t compact_turn_object_count_" << suffix << " = " << game.objectCount << ";\n"
        << "constexpr int32_t compact_turn_layer_count_" << suffix << " = " << game.layerCount << ";\n"
        << "constexpr bool compact_turn_has_rigid_" << suffix << " = " << (game.rigid ? "true" : "false") << ";\n"
        << "constexpr bool compact_turn_has_player_mask_" << suffix << " = " << (game.playerMask != kNullMaskOffset ? "true" : "false") << ";\n"
        << "constexpr bool compact_turn_player_mask_aggregate_" << suffix << " = " << (game.playerMaskAggregate ? "true" : "false") << ";\n"
        << "constexpr bool compact_turn_requires_player_movement_" << suffix << " = "
        << (game.metadata.values.find("require_player_movement") != game.metadata.values.end() ? "true" : "false") << ";\n"
        << "constexpr int32_t compact_turn_win_condition_count_" << suffix << " = " << game.winConditions.size() << ";\n\n";

    const std::vector<MaskWord> playerMask = compiledMaskWords(game, game.playerMask, game.wordCount);
    emitMaskArray(out, "compact_turn_player_mask_" + suffix, playerMask);

    out << "constexpr int32_t compact_turn_rigid_group_index_to_group_index_" << suffix << "[] = {";
    for (size_t index = 0; index < game.rigidGroupIndexToGroupIndex.size(); ++index) {
        if (index > 0) out << ", ";
        out << game.rigidGroupIndexToGroupIndex[index];
    }
    out << "};\n";
    out << "constexpr int32_t compact_turn_rigid_group_index_to_group_index_count_" << suffix << " = "
        << game.rigidGroupIndexToGroupIndex.size() << ";\n";
    out << "constexpr int32_t compact_turn_group_number_to_rigid_group_index_" << suffix << "[] = {";
    for (size_t index = 0; index < game.groupNumberToRigidGroupIndex.size(); ++index) {
        if (index > 0) out << ", ";
        out << game.groupNumberToRigidGroupIndex[index];
    }
    out << "};\n";
    out << "constexpr int32_t compact_turn_group_number_to_rigid_group_index_count_" << suffix << " = "
        << game.groupNumberToRigidGroupIndex.size() << ";\n\n";

    out << "constexpr int32_t compact_turn_object_layer_" << suffix << "[] = {";
    const int32_t emittedObjectLayerCount = std::max(game.objectCount, 1);
    for (int32_t objectId = 0; objectId < emittedObjectLayerCount; ++objectId) {
        if (objectId > 0) out << ", ";
        const int32_t layer = static_cast<size_t>(objectId) < game.objectsById.size()
            ? game.objectsById[static_cast<size_t>(objectId)].layer
            : -1;
        out << layer;
    }
    out << "};\n\n";

    for (int32_t layer = 0; layer < game.layerCount; ++layer) {
        const MaskOffset offset = static_cast<size_t>(layer) < game.layerMaskOffsets.size()
            ? game.layerMaskOffsets[static_cast<size_t>(layer)]
            : kNullMaskOffset;
        const std::vector<MaskWord> layerMask = compiledMaskWords(game, offset, game.wordCount);
        emitMaskArray(out, "compact_turn_layer_mask_" + suffix + "_" + std::to_string(layer), layerMask);
    }
    if (game.layerCount > 0) {
        out << "\n";
    }

    for (size_t conditionIndex = 0; conditionIndex < game.winConditions.size(); ++conditionIndex) {
        const WinCondition& condition = game.winConditions[conditionIndex];
        const std::vector<MaskWord> filter1 = compiledMaskWords(game, condition.filter1, game.wordCount);
        const std::vector<MaskWord> filter2 = compiledMaskWords(game, condition.filter2, game.wordCount);
        emitMaskArray(out, "compact_turn_win_filter1_" + suffix + "_" + std::to_string(conditionIndex), filter1);
        emitMaskArray(out, "compact_turn_win_filter2_" + suffix + "_" + std::to_string(conditionIndex), filter2);
    }
    if (!game.winConditions.empty()) {
        out << "\n";
    }
    emitCompactRuleMaskData(out, game, suffix, "early", game.rules);
    emitCompactRuleMaskData(out, game, suffix, "late", game.lateRules);
    if (hasAnyRulegroups(game.rules) || hasAnyRulegroups(game.lateRules)) {
        out << "\n";
    }

    out << "struct CompactTurnCommands_" << suffix << " {\n"
        << "    bool any = false;\n"
        << "    int32_t commandCount = 0;\n"
        << "    bool hasAgain = false;\n"
        << "    bool hasCancel = false;\n"
        << "    bool hasCheckpoint = false;\n"
        << "    bool hasRestart = false;\n"
        << "    bool hasWin = false;\n"
        << "    bool hasMessage = false;\n"
        << "    std::string messageText;\n"
        << "};\n\n";

    out << "struct CompactTurnMovementOutcome_" << suffix << " {\n"
        << "    bool moved = false;\n"
        << "    bool shouldUndo = false;\n"
        << "};\n\n";

    out << "uint8_t compact_turn_next_random_byte_" << suffix << "(RandomState& state) {\n"
        << "    state.i = static_cast<uint8_t>((state.i + 1) % 256);\n"
        << "    state.j = static_cast<uint8_t>((state.j + state.s[static_cast<size_t>(state.i)]) % 256);\n"
        << "    std::swap(state.s[static_cast<size_t>(state.i)], state.s[static_cast<size_t>(state.j)]);\n"
        << "    const uint8_t index = static_cast<uint8_t>((state.s[static_cast<size_t>(state.i)] + state.s[static_cast<size_t>(state.j)]) % 256);\n"
        << "    return state.s[static_cast<size_t>(index)];\n"
        << "}\n\n";

    out << "double compact_turn_random_uniform_" << suffix << "(RandomState& state) {\n"
        << "    double output = 0.0;\n"
        << "    for (int32_t index = 0; index < 7; ++index) {\n"
        << "        output *= 256.0;\n"
        << "        output += compact_turn_next_random_byte_" << suffix << "(state);\n"
        << "    }\n"
        << "    return output / 72057594037927935.0;\n"
        << "}\n\n";

    out << "bool compact_turn_can_handle_turn_" << suffix << "() {\n"
        << "    return " << (canCompactCompilerHandleTurn(game) ? "true" : "false") << ";\n"
        << "}\n\n";

    out << "int32_t compact_turn_tile_count_" << suffix << "(LevelDimensions dimensions) {\n"
        << "    if (dimensions.width <= 0 || dimensions.height <= 0) return 0;\n"
        << "    return dimensions.width * dimensions.height;\n"
        << "}\n\n";

    out << "bool compact_turn_in_bounds_" << suffix << "(LevelDimensions dimensions, int32_t x, int32_t y) {\n"
        << "    return x >= 0 && y >= 0 && x < dimensions.width && y < dimensions.height;\n"
        << "}\n\n";

    out << "int32_t compact_turn_tile_index_" << suffix << "(LevelDimensions dimensions, int32_t x, int32_t y) {\n"
        << "    return x * dimensions.height + y;\n"
        << "}\n\n";

    out << "bool compact_turn_direction_delta_" << suffix << "(int32_t directionMask, int32_t& dx, int32_t& dy) {\n"
        << "    switch (directionMask) {\n"
        << "        case 1: dx = 0; dy = -1; return true;\n"
        << "        case 2: dx = 0; dy = 1; return true;\n"
        << "        case 4: dx = -1; dy = 0; return true;\n"
        << "        case 8: dx = 1; dy = 0; return true;\n"
        << "        case 16: dx = 0; dy = 0; return true;\n"
        << "        default: dx = 0; dy = 0; return false;\n"
        << "    }\n"
        << "}\n\n";

    out << "int32_t compact_turn_input_direction_" << suffix << "(ps_input input) {\n"
        << "    switch (input) {\n"
        << "        case PS_INPUT_UP: return 1;\n"
        << "        case PS_INPUT_DOWN: return 2;\n"
        << "        case PS_INPUT_LEFT: return 4;\n"
        << "        case PS_INPUT_RIGHT: return 8;\n"
        << "        case PS_INPUT_ACTION: return 16;\n"
        << "        default: return 0;\n"
        << "    }\n"
        << "}\n\n";

    out << "bool compact_turn_step_cell_" << suffix << "(LevelDimensions dimensions, int32_t& x, int32_t& y, int32_t directionMask) {\n"
        << "    int32_t dx = 0;\n"
        << "    int32_t dy = 0;\n"
        << "    if (!compact_turn_direction_delta_" << suffix << "(directionMask, dx, dy)) return false;\n"
        << "    const int32_t nextX = x + dx;\n"
        << "    const int32_t nextY = y + dy;\n"
        << "    if (!compact_turn_in_bounds_" << suffix << "(dimensions, nextX, nextY)) return false;\n"
        << "    x = nextX;\n"
        << "    y = nextY;\n"
        << "    return true;\n"
        << "}\n\n";

    out << "bool compact_turn_cell_at_direction_" << suffix << "(LevelDimensions dimensions, int32_t originTileIndex, int32_t directionMask, int32_t distance, int32_t& outTileIndex) {\n"
        << "    if (dimensions.width <= 0 || dimensions.height <= 0 || distance < 0 || originTileIndex < 0) return false;\n"
        << "    int32_t x = originTileIndex / dimensions.height;\n"
        << "    int32_t y = originTileIndex % dimensions.height;\n"
        << "    if (!compact_turn_in_bounds_" << suffix << "(dimensions, x, y)) return false;\n"
        << "    for (int32_t step = 0; step < distance; ++step) {\n"
        << "        if (!compact_turn_step_cell_" << suffix << "(dimensions, x, y, directionMask)) return false;\n"
        << "    }\n"
        << "    outTileIndex = compact_turn_tile_index_" << suffix << "(dimensions, x, y);\n"
        << "    return true;\n"
        << "}\n\n";

    out << "int32_t compact_turn_available_at_direction_" << suffix << "(LevelDimensions dimensions, int32_t originTileIndex, int32_t directionMask) {\n"
        << "    if (dimensions.width <= 0 || dimensions.height <= 0 || originTileIndex < 0) return 0;\n"
        << "    const int32_t x = originTileIndex / dimensions.height;\n"
        << "    const int32_t y = originTileIndex % dimensions.height;\n"
        << "    if (!compact_turn_in_bounds_" << suffix << "(dimensions, x, y)) return 0;\n"
        << "    switch (directionMask) {\n"
        << "        case 1: return y + 1;\n"
        << "        case 2: return dimensions.height - y;\n"
        << "        case 4: return x + 1;\n"
        << "        case 8: return dimensions.width - x;\n"
        << "        default: return 0;\n"
        << "    }\n"
        << "}\n\n";

    out << "bool compact_turn_prepare_state_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    if (tileCount <= 0) return false;\n"
        << "    const size_t objectWords = static_cast<size_t>(tileCount) * static_cast<size_t>(compact_turn_object_stride_" << suffix << ");\n"
        << "    if (levelState.board.objects.size() != objectWords) return false;\n"
        << "    const size_t movementWords = static_cast<size_t>(tileCount) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "    if (scratch.liveMovements.size() != movementWords) {\n"
        << "        scratch.liveMovements.assign(movementWords, 0);\n"
        << "    }\n"
        << "    if (compact_turn_has_rigid_" << suffix << ") {\n"
        << "        if (scratch.rigidGroupIndexMasks.size() != movementWords) scratch.rigidGroupIndexMasks.assign(movementWords, 0);\n"
        << "        if (scratch.rigidMovementAppliedMasks.size() != movementWords) scratch.rigidMovementAppliedMasks.assign(movementWords, 0);\n"
        << "    }\n"
        << "    return true;\n"
        << "}\n\n";

    out << "MaskWord* compact_turn_cell_objects_" << suffix << "(PersistentLevelState& levelState, int32_t tileIndex) {\n"
        << "    return levelState.board.objects.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_object_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "const MaskWord* compact_turn_cell_objects_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex) {\n"
        << "    return levelState.board.objects.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_object_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "MaskWord* compact_turn_cell_movements_" << suffix << "(Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.liveMovements.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "const MaskWord* compact_turn_cell_movements_" << suffix << "(const Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.liveMovements.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "MaskWord* compact_turn_cell_rigid_group_index_" << suffix << "(Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.rigidGroupIndexMasks.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "const MaskWord* compact_turn_cell_rigid_group_index_" << suffix << "(const Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.rigidGroupIndexMasks.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "MaskWord* compact_turn_cell_rigid_movement_applied_" << suffix << "(Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.rigidMovementAppliedMasks.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "const MaskWord* compact_turn_cell_rigid_movement_applied_" << suffix << "(const Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.rigidMovementAppliedMasks.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "int32_t compact_turn_layer_bits_" << suffix << "(const MaskWord* cell, int32_t layer) {\n"
        << "    if (layer < 0 || layer >= compact_turn_layer_count_" << suffix << ") return 0;\n"
        << "    const uint32_t shiftIndex = static_cast<uint32_t>(layer) * 5U;\n"
        << "    const uint32_t word = shiftIndex >> kMaskWordShift;\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) return 0;\n"
        << "    const uint32_t bit = shiftIndex & kMaskWordBitMask;\n"
        << "    MaskWordUnsigned result = static_cast<MaskWordUnsigned>(cell[word]) >> bit;\n"
        << "    if (bit > kMaskWordBits - 5U && word + 1U < static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) {\n"
        << "        result |= static_cast<MaskWordUnsigned>(cell[word + 1U]) << (kMaskWordBits - bit);\n"
        << "    }\n"
        << "    return static_cast<int32_t>(result & MaskWordUnsigned{0x1f});\n"
        << "}\n\n";

    out << "void compact_turn_set_layer_bits_" << suffix << "(MaskWord* cell, int32_t layer, int32_t value) {\n"
        << "    if (layer < 0 || layer >= compact_turn_layer_count_" << suffix << ") return;\n"
        << "    const uint32_t shiftIndex = static_cast<uint32_t>(layer) * 5U;\n"
        << "    const uint32_t word = shiftIndex >> kMaskWordShift;\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) return;\n"
        << "    const uint32_t bit = shiftIndex & kMaskWordBitMask;\n"
        << "    const MaskWordUnsigned packed = MaskWordUnsigned{static_cast<uint32_t>(value) & 0x1fU};\n"
        << "    const MaskWordUnsigned lowMask = MaskWordUnsigned{0x1f} << bit;\n"
        << "    cell[word] = static_cast<MaskWord>((static_cast<MaskWordUnsigned>(cell[word]) & ~lowMask) | (packed << bit));\n"
        << "    if (bit > kMaskWordBits - 5U && word + 1U < static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) {\n"
        << "        const uint32_t highShift = kMaskWordBits - bit;\n"
        << "        const MaskWordUnsigned highMask = MaskWordUnsigned{0x1f} >> highShift;\n"
        << "        cell[word + 1U] = static_cast<MaskWord>((static_cast<MaskWordUnsigned>(cell[word + 1U]) & ~highMask) | (packed >> highShift));\n"
        << "    }\n"
        << "}\n\n";

    out << "bool compact_turn_cell_has_object_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex, int32_t objectId) {\n"
        << "    if (objectId < 0 || objectId >= compact_turn_object_count_" << suffix << ") return false;\n"
        << "    const uint32_t bit = static_cast<uint32_t>(objectId);\n"
        << "    const uint32_t word = maskWordIndex(bit);\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_object_stride_" << suffix << ")) return false;\n"
        << "    return (compact_turn_cell_objects_" << suffix << "(levelState, tileIndex)[word] & maskBit(bit)) != 0;\n"
        << "}\n\n";

    out << "bool compact_turn_cell_any_objects_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex, const MaskWord* mask) {\n"
        << "    const MaskWord* cell = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        if ((cell[word] & mask[word]) != 0) return true;\n"
        << "    }\n"
        << "    return false;\n"
        << "}\n\n";

    out << "bool compact_turn_cell_has_all_objects_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex, const MaskWord* mask) {\n"
        << "    const MaskWord* cell = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        if ((cell[word] & mask[word]) != mask[word]) return false;\n"
        << "    }\n"
        << "    return true;\n"
        << "}\n\n";

    out << "bool compact_turn_matches_filter_" << suffix << "(const MaskWord* filter, bool aggregate, const MaskWord* cell) {\n"
        << "    if (aggregate) {\n"
        << "        for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "            if ((cell[word] & filter[word]) != filter[word]) return false;\n"
        << "        }\n"
        << "        return true;\n"
        << "    }\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        if ((cell[word] & filter[word]) != 0) return true;\n"
        << "    }\n"
        << "    return false;\n"
        << "}\n\n";

    out << "bool compact_turn_evaluate_win_condition_" << suffix << "(LevelDimensions dimensions, const PersistentLevelState& levelState, int32_t quantifier, const MaskWord* filter1, bool aggr1, const MaskWord* filter2, bool aggr2) {\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    switch (quantifier) {\n"
        << "        case -1:\n"
        << "            for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
        << "                const MaskWord* cell = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "                if (compact_turn_matches_filter_" << suffix << "(filter1, aggr1, cell) && compact_turn_matches_filter_" << suffix << "(filter2, aggr2, cell)) return false;\n"
        << "            }\n"
        << "            return true;\n"
        << "        case 0:\n"
        << "            for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
        << "                const MaskWord* cell = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "                if (compact_turn_matches_filter_" << suffix << "(filter1, aggr1, cell) && compact_turn_matches_filter_" << suffix << "(filter2, aggr2, cell)) return true;\n"
        << "            }\n"
        << "            return false;\n"
        << "        case 1:\n"
        << "            for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
        << "                const MaskWord* cell = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "                if (compact_turn_matches_filter_" << suffix << "(filter1, aggr1, cell) && !compact_turn_matches_filter_" << suffix << "(filter2, aggr2, cell)) return false;\n"
        << "            }\n"
        << "            return true;\n"
        << "        default:\n"
        << "            return false;\n"
        << "    }\n"
        << "}\n\n";

    out << "bool compact_turn_evaluate_win_" << suffix << "(LevelDimensions dimensions, const PersistentLevelState& levelState) {\n";
    if (game.winConditions.empty()) {
        out << "    (void)dimensions;\n"
            << "    (void)levelState;\n"
            << "    return false;\n";
    } else {
        for (size_t conditionIndex = 0; conditionIndex < game.winConditions.size(); ++conditionIndex) {
            const WinCondition& condition = game.winConditions[conditionIndex];
            out << "    if (!compact_turn_evaluate_win_condition_" << suffix << "(\n"
                << "            dimensions,\n"
                << "            levelState,\n"
                << "            " << condition.quantifier << ",\n"
                << "            compact_turn_win_filter1_" << suffix << "_" << conditionIndex << ",\n"
                << "            " << (condition.aggr1 ? "true" : "false") << ",\n"
                << "            compact_turn_win_filter2_" << suffix << "_" << conditionIndex << ",\n"
                << "            " << (condition.aggr2 ? "true" : "false") << ")) {\n"
                << "        return false;\n"
                << "    }\n";
        }
        out << "    return true;\n";
    }
    out << "}\n\n";

    out << "const MaskWord* compact_turn_layer_mask_" << suffix << "(int32_t layer) {\n"
        << "    switch (layer) {\n";
    for (int32_t layer = 0; layer < game.layerCount; ++layer) {
        out << "        case " << layer << ": return compact_turn_layer_mask_" << suffix << "_" << layer << ";\n";
    }
    out << "        default: return nullptr;\n"
        << "    }\n"
        << "}\n\n";

    out << "bool compact_turn_cell_has_layer_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex, int32_t layer) {\n"
        << "    const MaskWord* layerMask = compact_turn_layer_mask_" << suffix << "(layer);\n"
        << "    return layerMask != nullptr && compact_turn_cell_any_objects_" << suffix << "(levelState, tileIndex, layerMask);\n"
        << "}\n\n";

    out << "bool compact_turn_cell_matches_player_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex) {\n"
        << "    if (!compact_turn_has_player_mask_" << suffix << ") return false;\n"
        << "    if (compact_turn_player_mask_aggregate_" << suffix << ") {\n"
        << "        return compact_turn_cell_has_all_objects_" << suffix << "(levelState, tileIndex, compact_turn_player_mask_" << suffix << ");\n"
        << "    }\n"
        << "    return compact_turn_cell_any_objects_" << suffix << "(levelState, tileIndex, compact_turn_player_mask_" << suffix << ");\n"
        << "}\n\n";

    out << "std::vector<int32_t> compact_turn_collect_player_positions_" << suffix << "(LevelDimensions dimensions, const PersistentLevelState& levelState) {\n"
        << "    std::vector<int32_t> positions;\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
        << "        if (compact_turn_cell_matches_player_" << suffix << "(levelState, tileIndex)) {\n"
        << "            positions.push_back(tileIndex);\n"
        << "        }\n"
        << "    }\n"
        << "    return positions;\n"
        << "}\n\n";

    out << "bool compact_turn_any_start_player_moved_" << suffix << "(const PersistentLevelState& levelState, const std::vector<int32_t>& startPlayerPositions) {\n"
        << "    for (const int32_t tileIndex : startPlayerPositions) {\n"
        << "        if (!compact_turn_cell_matches_player_" << suffix << "(levelState, tileIndex)) return true;\n"
        << "    }\n"
        << "    return false;\n"
        << "}\n\n";

    out << "void compact_turn_set_cell_object_" << suffix << "(PersistentLevelState& levelState, int32_t tileIndex, int32_t objectId) {\n"
        << "    if (objectId < 0 || objectId >= compact_turn_object_count_" << suffix << ") return;\n"
        << "    const uint32_t bit = static_cast<uint32_t>(objectId);\n"
        << "    compact_turn_cell_objects_" << suffix << "(levelState, tileIndex)[maskWordIndex(bit)] |= maskBit(bit);\n"
        << "}\n\n";

    out << "void compact_turn_clear_cell_object_" << suffix << "(PersistentLevelState& levelState, int32_t tileIndex, int32_t objectId) {\n"
        << "    if (objectId < 0 || objectId >= compact_turn_object_count_" << suffix << ") return;\n"
        << "    const uint32_t bit = static_cast<uint32_t>(objectId);\n"
        << "    compact_turn_cell_objects_" << suffix << "(levelState, tileIndex)[maskWordIndex(bit)] &= ~maskBit(bit);\n"
        << "}\n\n";

    out << "int32_t compact_turn_layer_movement_" << suffix << "(const Scratch& scratch, int32_t tileIndex, int32_t layer) {\n"
        << "    return compact_turn_layer_bits_" << suffix << "(compact_turn_cell_movements_" << suffix << "(scratch, tileIndex), layer);\n"
        << "}\n\n";

    out << "void compact_turn_set_layer_movement_" << suffix << "(Scratch& scratch, int32_t tileIndex, int32_t layer, int32_t directionMask) {\n"
        << "    compact_turn_set_layer_bits_" << suffix << "(compact_turn_cell_movements_" << suffix << "(scratch, tileIndex), layer, directionMask);\n"
        << "}\n\n";

    out << "bool compact_turn_seed_player_movements_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch, int32_t directionMask) {\n"
        << "    if (directionMask == 0 || !compact_turn_has_player_mask_" << suffix << ") return false;\n"
        << "    bool changed = false;\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
        << "        if (!compact_turn_cell_matches_player_" << suffix << "(levelState, tileIndex)) continue;\n"
        << "        const MaskWord* cell = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "        for (int32_t objectId = 0; objectId < compact_turn_object_count_" << suffix << "; ++objectId) {\n"
        << "            const uint32_t objectBit = static_cast<uint32_t>(objectId);\n"
        << "            const uint32_t word = maskWordIndex(objectBit);\n"
        << "            const MaskWord bit = maskBit(objectBit);\n"
        << "            if ((compact_turn_player_mask_" << suffix << "[word] & bit) == 0 || (cell[word] & bit) == 0) continue;\n"
        << "            const int32_t layer = compact_turn_object_layer_" << suffix << "[objectId];\n"
        << "            if (layer < 0) continue;\n"
        << "            const int32_t before = compact_turn_layer_movement_" << suffix << "(scratch, tileIndex, layer);\n"
        << "            compact_turn_set_layer_movement_" << suffix << "(scratch, tileIndex, layer, directionMask);\n"
        << "            const int32_t after = compact_turn_layer_movement_" << suffix << "(scratch, tileIndex, layer);\n"
        << "            changed = changed || before != after;\n"
        << "        }\n"
        << "    }\n"
        << "    return changed;\n"
        << "}\n\n";

    out << "bool compact_turn_resolve_one_layer_movement_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, int32_t tileIndex, int32_t layer, int32_t directionMask) {\n"
        << "    int32_t targetIndex = 0;\n"
        << "    if (!compact_turn_cell_at_direction_" << suffix << "(dimensions, tileIndex, directionMask, 1, targetIndex)) return false;\n"
        << "    const MaskWord* layerMask = compact_turn_layer_mask_" << suffix << "(layer);\n"
        << "    if (layerMask == nullptr) return false;\n"
        << "    if (directionMask != 16 && compact_turn_cell_any_objects_" << suffix << "(levelState, targetIndex, layerMask)) return false;\n"
        << "    if (targetIndex == tileIndex) return true;\n"
        << "    MaskWord* source = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "    MaskWord* target = compact_turn_cell_objects_" << suffix << "(levelState, targetIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        const MaskWord moving = source[word] & layerMask[word];\n"
        << "        source[word] &= ~layerMask[word];\n"
        << "        target[word] |= moving;\n"
        << "    }\n"
        << "    return true;\n"
        << "}\n\n";

    out << "CompactTurnMovementOutcome_" << suffix << " compact_turn_resolve_movements_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch, std::vector<bool>* bannedGroups) {\n"
        << "    CompactTurnMovementOutcome_" << suffix << " outcome;\n"
        << "    bool movedAny = false;\n"
        << "    bool movedThisPass = true;\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    while (movedThisPass) {\n"
        << "        movedThisPass = false;\n"
        << "        for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
        << "            bool changedTile = false;\n"
        << "            for (int32_t layer = 0; layer < compact_turn_layer_count_" << suffix << "; ++layer) {\n"
        << "                const int32_t layerMovement = compact_turn_layer_movement_" << suffix << "(scratch, tileIndex, layer);\n"
        << "                if (layerMovement == 0) continue;\n"
        << "                if (compact_turn_resolve_one_layer_movement_" << suffix << "(dimensions, levelState, tileIndex, layer, layerMovement)) {\n"
        << "                    compact_turn_set_layer_movement_" << suffix << "(scratch, tileIndex, layer, 0);\n"
        << "                    movedThisPass = true;\n"
        << "                    movedAny = true;\n"
        << "                    changedTile = true;\n"
        << "                }\n"
        << "            }\n"
        << "            (void)changedTile;\n"
        << "        }\n"
        << "    }\n"
        << "    if (compact_turn_has_rigid_" << suffix << " && bannedGroups != nullptr) {\n"
        << "        for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
        << "            const MaskWord* movementMask = compact_turn_cell_movements_" << suffix << "(scratch, tileIndex);\n"
        << "            const MaskWord* rigidAppliedMask = compact_turn_cell_rigid_movement_applied_" << suffix << "(scratch, tileIndex);\n"
        << "            bool hasRigidFailure = false;\n"
        << "            for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "                if ((movementMask[word] & rigidAppliedMask[word]) != 0) hasRigidFailure = true;\n"
        << "            }\n"
        << "            if (!hasRigidFailure) continue;\n"
        << "            const MaskWord* rigidGroupMask = compact_turn_cell_rigid_group_index_" << suffix << "(scratch, tileIndex);\n"
        << "            for (int32_t layer = 0; layer < compact_turn_layer_count_" << suffix << "; ++layer) {\n"
        << "                if ((compact_turn_layer_bits_" << suffix << "(movementMask, layer) & compact_turn_layer_bits_" << suffix << "(rigidAppliedMask, layer)) == 0) continue;\n"
        << "                const int32_t rigidGroupIndex = compact_turn_layer_bits_" << suffix << "(rigidGroupMask, layer) - 1;\n"
        << "                if (rigidGroupIndex < 0 || rigidGroupIndex >= compact_turn_rigid_group_index_to_group_index_count_" << suffix << ") break;\n"
        << "                const int32_t groupIndex = compact_turn_rigid_group_index_to_group_index_" << suffix << "[rigidGroupIndex];\n"
        << "                if (groupIndex >= 0) {\n"
        << "                    if (static_cast<size_t>(groupIndex) >= bannedGroups->size()) bannedGroups->resize(static_cast<size_t>(groupIndex + 1), false);\n"
        << "                    (*bannedGroups)[static_cast<size_t>(groupIndex)] = true;\n"
        << "                    outcome.shouldUndo = true;\n"
        << "                }\n"
        << "                break;\n"
        << "            }\n"
        << "        }\n"
        << "    }\n"
        << "    std::fill(scratch.liveMovements.begin(), scratch.liveMovements.end(), 0);\n"
        << "    outcome.moved = movedAny;\n"
        << "    return outcome;\n"
        << "}\n\n";

    emitCompactRulegroupFunctions(out, game, game.rules, game.loopPoint, suffix, "early");
    emitCompactRulegroupFunctions(out, game, game.lateRules, game.lateLoopPoint, suffix, "late");
}

} // namespace

void emitCompactTurnBackend(
    std::ostream& out,
    const Game& game,
    std::string_view sourcePath,
    uint64_t sourceHash,
    size_t sourceIndex,
    CompactCodegenOptions options
) {
    const CompactTurnSupport compactTurnSupport = compactTurnSupportForGame(game, options);
    const std::string suffix = sourceSuffix(sourceIndex);
    if (compactTurnSupport.supported && !compactTurnSupport.interpreterBridge) {
        emitCompactTurnAccessLayer(out, game, sourceIndex);
        out << "SpecializedCompactTurnOutcome specialized_compact_turn_single_" << sourceIndex << "(\n"
            << "    LevelDimensions dimensions,\n"
            << "    PersistentLevelState& levelState,\n"
            << "    Scratch& scratch,\n"
            << "    ps_input input,\n"
            << "    RuntimeStepOptions options,\n"
            << "    bool* outHasAgain,\n"
            << "    bool probeOnly\n"
            << ") {\n";
        emitCompactTurnCompilerSingleBody(out, suffix);
        out << "}\n\n";
        out << "SpecializedCompactTurnOutcome specialized_compact_turn_core_" << sourceIndex << "(\n"
            << "    LevelDimensions dimensions,\n"
            << "    PersistentLevelState& levelState,\n"
            << "    Scratch& scratch,\n"
            << "    ps_input input,\n"
            << "    RuntimeStepOptions options\n"
            << ") {\n";
        emitCompactTurnCompilerDrainBody(out, suffix);
        out << "}\n\n";
    }
    out << "SpecializedCompactTurnOutcome specialized_compact_turn_source_" << sourceIndex << "(\n"
        << "    const Game& game,\n"
        << "    PersistentLevelState& levelState,\n"
        << "    Scratch& scratch,\n"
        << "    SpecializedCompactTurnContext context,\n"
        << "    ps_input input,\n"
        << "    RuntimeStepOptions options\n"
        << ") {\n";
    if (!compactTurnSupport.supported) {
        emitCompactTurnUnsupportedBody(out);
        out << "}\n\n";
    } else if (compactTurnSupport.interpreterBridge) {
        out << "    return compactStateInterpretedTurnBridge(game, levelState, scratch, context, input, options);\n"
            << "}\n\n";
    } else {
        out << "    (void)game;\n"
            << "    return specialized_compact_turn_core_" << sourceIndex << "(context.dimensions, levelState, scratch, input, options);\n"
            << "}\n\n";
    }
    out
        << "const SpecializedCompactTurnBackend specialized_compact_turn_backend_" << sourceIndex << " = {\n"
        << "    " << sourceHash << "ULL,\n"
        << "    " << cppStringLiteral(sourcePath) << ",\n"
        << "    specialized_compact_turn_source_" << sourceIndex << ",\n"
        << "    {" << (compactTurnSupport.supported ? "true" : "false")
        << ", " << cppStringLiteral(compactTurnSupport.fallbackReason) << "},\n"
        << "    " << (compactTurnSupport.supported && !compactTurnSupport.interpreterBridge ? "true" : "false") << ",\n"
        << "};\n\n";
}

CompactTurnSupport compactNativeTurnSupportForGame(const Game& game) {
    (void)game;
    CompactTurnSupport support;
    support.fallbackReason = "native_compact_generator_rebuild";
    support.nativeFallbackReason = support.fallbackReason;
    return support;
}

CompactTurnSupport compactTurnSupportForGame(const Game& game, const CompactCodegenOptions& options) {
    CompactTurnSupport support = compactNativeTurnSupportForGame(game);
    support.nativeFallbackReason = support.fallbackReason;
    if (!options.interpreterMode) {
        support.supported = true;
        support.interpreterBridge = false;
        support.fallbackReason = "compiler_mode";
        support.nativeFallbackReason = "compiler_mode";
        return support;
    }
    if (options.interpreterMode && !support.supported) {
        support.supported = true;
        support.interpreterBridge = true;
        support.fallbackReason = "interpreter_bridge";
    }
    return support;
}

CompactTurnSupport compactTurnSupportForGame(const Game& game) {
    return compactTurnSupportForGame(game, CompactCodegenOptions{});
}

} // namespace puzzlescript::compiler
