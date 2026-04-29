#include "compiler/compact_turn_codegen.hpp"

#include "compiler/compiled_rules_codegen.hpp"

#include <algorithm>
#include <ostream>

namespace puzzlescript::compiler {

namespace {

std::string compactRulePatternUnsupportedReason(const Pattern& pattern) {
    if (pattern.kind != Pattern::Kind::CellPattern) {
        return "non_cell_pattern";
    }
    if (pattern.replacement.has_value()) {
        const Replacement& replacement = *pattern.replacement;
        if (replacement.hasRandomEntityMask) {
            return "random_entity_replacement";
        }
        if (replacement.hasRandomDirMask) {
            return "random_direction_replacement";
        }
    }
    return {};
}

std::string compactRuleUnsupportedReason(const Rule& rule) {
    if (rule.isRandom) {
        return "random_rule";
    }
    if (rule.rigid) {
        return "rigid_rule";
    }
    if (!rule.commands.empty()) {
        return "commands";
    }
    if (rule.patterns.empty()) {
        return "empty_rule";
    }
    if (rule.ellipsisCount.size() < rule.patterns.size()) {
        return "missing_ellipsis_metadata";
    }
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        if (rule.ellipsisCount[rowIndex] != 0) {
            return "ellipsis";
        }
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
                    const std::string prefix = compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex);
                    emitMaskArray(out, prefix + "_objects_present", compiledMaskWords(game, pattern.objectsPresent, game.wordCount));
                    emitMaskArray(out, prefix + "_objects_missing", compiledMaskWords(game, pattern.objectsMissing, game.wordCount));
                    emitMaskArray(out, prefix + "_movements_present", compiledMaskWords(game, pattern.movementsPresent, game.movementWordCount));
                    emitMaskArray(out, prefix + "_movements_missing", compiledMaskWords(game, pattern.movementsMissing, game.movementWordCount));
                    for (uint32_t anyIndex = 0; anyIndex < pattern.anyObjectsCount; ++anyIndex) {
                        const MaskOffset offset = game.anyObjectOffsets[pattern.anyObjectsFirst + anyIndex];
                        emitMaskArray(
                            out,
                            prefix + "_any_objects_" + std::to_string(anyIndex),
                            compiledMaskWords(game, offset, game.wordCount)
                        );
                    }
                    if (pattern.replacement.has_value()) {
                        const Replacement& replacement = *pattern.replacement;
                        emitMaskArray(out, prefix + "_objects_clear", compiledMaskWords(game, replacement.objectsClear, game.wordCount));
                        emitMaskArray(out, prefix + "_objects_set", compiledMaskWords(game, replacement.objectsSet, game.wordCount));
                        emitMaskArray(out, prefix + "_movements_clear", compiledMaskWords(game, replacement.movementsClear, game.movementWordCount));
                        emitMaskArray(out, prefix + "_movements_set", compiledMaskWords(game, replacement.movementsSet, game.movementWordCount));
                        emitMaskArray(out, prefix + "_movements_layer_mask", compiledMaskWords(game, replacement.movementsLayerMask, game.movementWordCount));
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

    out << "bool " << prefix << "_apply(PersistentLevelState& levelState, Scratch& scratch, int32_t tileIndex) {\n";
    if (!pattern.replacement.has_value()) {
        out << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    (void)tileIndex;\n"
            << "    return false;\n"
            << "}\n\n";
        return;
    }

    out << "    bool changed = false;\n"
        << "    MaskWord* objects = compact_turn_cell_objects_" << suffix << "(levelState, tileIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_object_stride_" << suffix << "; ++word) {\n"
        << "        const MaskWord before = objects[word];\n"
        << "        const MaskWord after = (before & ~" << prefix << "_objects_clear[word]) | " << prefix << "_objects_set[word];\n"
        << "        objects[word] = after;\n"
        << "        changed = changed || before != after;\n"
        << "    }\n"
        << "    MaskWord* movements = compact_turn_cell_movements_" << suffix << "(scratch, tileIndex);\n"
        << "    for (int32_t word = 0; word < compact_turn_movement_stride_" << suffix << "; ++word) {\n"
        << "        const MaskWord before = movements[word];\n"
        << "        const MaskWord clear = " << prefix << "_movements_clear[word] | " << prefix << "_movements_layer_mask[word];\n"
        << "        const MaskWord after = (before & ~clear) | " << prefix << "_movements_set[word];\n"
        << "        movements[word] = after;\n"
        << "        changed = changed || before != after;\n"
        << "    }\n"
        << "    if (changed) scratch.objectCellIndexDirty = true;\n"
        << "    return changed;\n"
        << "}\n\n";
}

void emitCompactRuleFunction(
    std::ostream& out,
    const Rule& rule,
    std::string_view suffix,
    std::string_view phase,
    size_t groupIndex,
    size_t ruleIndex
) {
    const std::string prefix = compactRulePrefix(suffix, phase, groupIndex, ruleIndex);
    if (!isCompactRuleSupported(rule)) {
        const std::string reason = "compact turn compiler TODO: "
            + compactRuleUnsupportedReason(rule)
            + " at source rule line "
            + std::to_string(rule.lineNumber);
        out << "bool " << prefix << "_apply(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n"
            << "    (void)dimensions;\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
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
        out << "bool " << rowPrefix << "_matches(LevelDimensions dimensions, const PersistentLevelState& levelState, const Scratch& scratch, int32_t startIndex) {\n";
        for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
            out << "    int32_t tile_" << patternIndex << " = 0;\n"
                << "    if (!compact_turn_cell_at_direction_" << suffix
                << "(dimensions, startIndex, " << rule.direction << ", " << patternIndex << ", tile_" << patternIndex << ")) return false;\n"
                << "    if (!" << compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex)
                << "_matches(levelState, scratch, tile_" << patternIndex << ")) return false;\n";
        }
        out << "    return true;\n"
            << "}\n\n";

        out << "bool " << rowPrefix << "_apply_replacements(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch, int32_t startIndex) {\n"
            << "    bool changed = false;\n";
        for (size_t patternIndex = 0; patternIndex < row.size(); ++patternIndex) {
            out << "    int32_t tile_" << patternIndex << " = 0;\n"
                << "    if (!compact_turn_cell_at_direction_" << suffix
                << "(dimensions, startIndex, " << rule.direction << ", " << patternIndex << ", tile_" << patternIndex << ")) return changed;\n"
                << "    changed = " << compactPatternPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex, patternIndex)
                << "_apply(levelState, scratch, tile_" << patternIndex << ") || changed;\n";
        }
        out << "    return changed;\n"
            << "}\n\n";
    }

    out << "bool " << prefix << "_apply(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n"
        << "    constexpr size_t rowCount = " << rule.patterns.size() << ";\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    std::vector<std::vector<int32_t>> matches(rowCount);\n";
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::string rowPrefix = compactRowPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex);
        out << "    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {\n"
            << "        if (" << rowPrefix << "_matches(dimensions, levelState, scratch, tileIndex)) {\n"
            << "            matches[" << rowIndex << "].push_back(tileIndex);\n"
            << "        }\n"
            << "    }\n"
            << "    if (matches[" << rowIndex << "].empty()) return false;\n";
    }
    out << "    std::vector<size_t> tupleIndex(rowCount, 0);\n"
        << "    bool firstTuple = true;\n"
        << "    bool changed = false;\n"
        << "    while (true) {\n"
        << "        bool stillMatches = true;\n"
        << "        if (!firstTuple) {\n";
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::string rowPrefix = compactRowPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex);
        out << "            if (!" << rowPrefix
            << "_matches(dimensions, levelState, scratch, matches[" << rowIndex << "][tupleIndex[" << rowIndex << "]])) stillMatches = false;\n";
    }
    out << "        }\n"
        << "        if (stillMatches) {\n";
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const std::string rowPrefix = compactRowPrefix(suffix, phase, groupIndex, ruleIndex, rowIndex);
        out << "            changed = " << rowPrefix
            << "_apply_replacements(dimensions, levelState, scratch, matches[" << rowIndex << "][tupleIndex[" << rowIndex << "]]) || changed;\n";
    }
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
    const std::vector<std::vector<Rule>>& groups,
    std::string_view suffix,
    std::string_view phase
) {
    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const std::vector<Rule>& group = groups[groupIndex];
        for (size_t ruleIndex = 0; ruleIndex < group.size(); ++ruleIndex) {
            emitCompactRuleFunction(out, group[ruleIndex], suffix, phase, groupIndex, ruleIndex);
        }

        const std::string groupPrefix = compactGroupPrefix(suffix, phase, groupIndex);
        out << "bool " << groupPrefix << "_apply(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n";
        if (group.empty()) {
            out << "    (void)dimensions;\n"
                << "    (void)levelState;\n"
                << "    (void)scratch;\n"
                << "    return false;\n"
                << "}\n\n";
            continue;
        }
        if (group[0].isRandom) {
            out << "    (void)dimensions;\n"
                << "    (void)levelState;\n"
                << "    (void)scratch;\n"
                << "    static_assert(false, \"compact turn compiler TODO: random rule group\");\n"
                << "    return false;\n"
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
                << "_apply(dimensions, levelState, scratch)) {\n"
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

    out << "bool compact_turn_apply_" << phase << "_rules_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n";
    if (groups.empty()) {
        out << "    (void)dimensions;\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    return false;\n";
    } else {
        out << "    bool changed = false;\n";
        for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
            out << "    changed = " << compactGroupPrefix(suffix, phase, groupIndex)
                << "_apply(dimensions, levelState, scratch) || changed;\n";
        }
        out << "    return changed;\n";
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

void emitCompactTurnCompilerSkeletonBody(std::ostream& out, std::string_view suffix) {
    out << "    (void)options;\n"
        << "    ps_step_result result{};\n"
        << "    if (!compact_turn_prepare_state_" << suffix << "(dimensions, levelState, scratch)) {\n"
        << "        return {false, result};\n"
        << "    }\n"
        << "    if (!compact_turn_can_handle_turn_" << suffix << "()) {\n"
        << "        return {false, result};\n"
        << "    }\n"
        << "    const int32_t directionMask = compact_turn_input_direction_" << suffix << "(input);\n"
        << "    std::vector<MaskWord> turnStartObjects;\n"
        << "    std::vector<int32_t> startPlayerPositions;\n"
        << "    if (directionMask != 0 && compact_turn_requires_player_movement_" << suffix << ") {\n"
        << "        turnStartObjects = levelState.board.objects;\n"
        << "        startPlayerPositions = compact_turn_collect_player_positions_" << suffix << "(dimensions, levelState);\n"
        << "    }\n"
        << "    const bool seededInput = compact_turn_seed_player_movements_" << suffix << "(dimensions, levelState, scratch, directionMask);\n"
        << "    // Semantic compact turn compiler skeleton:\n"
        << "    // 1. validate level dimensions and persistent board storage\n"
        << "    // 2. decode input direction\n"
        << "    // 3. seed input movements\n"
        << "    const bool ruleChanged = compact_turn_apply_early_rules_" << suffix << "(dimensions, levelState, scratch);\n"
        << "    // 4. apply early rulegroups\n"
        << "    const bool moved = compact_turn_resolve_movements_" << suffix << "(dimensions, levelState, scratch);\n"
        << "    // 5. resolve movement\n"
        << "    const bool lateRuleChanged = compact_turn_apply_late_rules_" << suffix << "(dimensions, levelState, scratch);\n"
        << "    // 6. apply late rulegroups\n"
        << "    // 7. process commands and again policy\n"
        << "    if (!startPlayerPositions.empty() && !compact_turn_any_start_player_moved_" << suffix << "(levelState, startPlayerPositions)) {\n"
        << "        levelState.board.objects = turnStartObjects;\n"
        << "        std::fill(scratch.liveMovements.begin(), scratch.liveMovements.end(), 0);\n"
        << "        return {true, result};\n"
        << "    }\n"
        << "    const bool won = compact_turn_evaluate_win_" << suffix << "(dimensions, levelState);\n"
        << "    // 8. evaluate win conditions\n"
        << "    // 9. canonicalize and return result\n"
        << "    result.changed = seededInput || ruleChanged || moved || lateRuleChanged;\n"
        << "    result.won = won;\n"
        << "    return {true, result};\n";
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
        << "constexpr bool compact_turn_has_player_mask_" << suffix << " = " << (game.playerMask != kNullMaskOffset ? "true" : "false") << ";\n"
        << "constexpr bool compact_turn_player_mask_aggregate_" << suffix << " = " << (game.playerMaskAggregate ? "true" : "false") << ";\n"
        << "constexpr bool compact_turn_requires_player_movement_" << suffix << " = "
        << (game.metadata.values.find("require_player_movement") != game.metadata.values.end() ? "true" : "false") << ";\n"
        << "constexpr int32_t compact_turn_win_condition_count_" << suffix << " = " << game.winConditions.size() << ";\n\n";

    const std::vector<MaskWord> playerMask = compiledMaskWords(game, game.playerMask, game.wordCount);
    emitMaskArray(out, "compact_turn_player_mask_" + suffix, playerMask);

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

    out << "bool compact_turn_prepare_state_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    if (tileCount <= 0) return false;\n"
        << "    const size_t objectWords = static_cast<size_t>(tileCount) * static_cast<size_t>(compact_turn_object_stride_" << suffix << ");\n"
        << "    if (levelState.board.objects.size() != objectWords) return false;\n"
        << "    const size_t movementWords = static_cast<size_t>(tileCount) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "    if (scratch.liveMovements.size() != movementWords) {\n"
        << "        scratch.liveMovements.assign(movementWords, 0);\n"
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
        << "    if (layer < 0 || layer >= compact_turn_layer_count_" << suffix << ") return 0;\n"
        << "    const uint32_t layerIndex = static_cast<uint32_t>(layer);\n"
        << "    const uint32_t word = movementWordIndexForLayer(layerIndex);\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) return 0;\n"
        << "    const uint32_t shift = movementBitShiftForLayer(layerIndex);\n"
        << "    const MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(compact_turn_cell_movements_" << suffix << "(scratch, tileIndex)[word]);\n"
        << "    return static_cast<int32_t>((bits >> shift) & MaskWordUnsigned{0x1f});\n"
        << "}\n\n";

    out << "void compact_turn_set_layer_movement_" << suffix << "(Scratch& scratch, int32_t tileIndex, int32_t layer, int32_t directionMask) {\n"
        << "    if (layer < 0 || layer >= compact_turn_layer_count_" << suffix << ") return;\n"
        << "    const uint32_t layerIndex = static_cast<uint32_t>(layer);\n"
        << "    const uint32_t word = movementWordIndexForLayer(layerIndex);\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) return;\n"
        << "    const uint32_t shift = movementBitShiftForLayer(layerIndex);\n"
        << "    MaskWord& cellWord = compact_turn_cell_movements_" << suffix << "(scratch, tileIndex)[word];\n"
        << "    const MaskWord mask = static_cast<MaskWord>(MaskWordUnsigned{0x1f} << shift);\n"
        << "    const MaskWord value = static_cast<MaskWord>((MaskWordUnsigned{static_cast<uint32_t>(directionMask) & 0x1fU}) << shift);\n"
        << "    cellWord = static_cast<MaskWord>((static_cast<MaskWordUnsigned>(cellWord) & ~static_cast<MaskWordUnsigned>(mask)) | static_cast<MaskWordUnsigned>(value));\n"
        << "}\n\n";

    out << "void compact_turn_clear_layer_movement_" << suffix << "(Scratch& scratch, int32_t tileIndex, int32_t layer) {\n"
        << "    compact_turn_set_layer_movement_" << suffix << "(scratch, tileIndex, layer, 0);\n"
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

    out << "bool compact_turn_resolve_movements_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n"
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
        << "                    compact_turn_clear_layer_movement_" << suffix << "(scratch, tileIndex, layer);\n"
        << "                    movedThisPass = true;\n"
        << "                    movedAny = true;\n"
        << "                    changedTile = true;\n"
        << "                }\n"
        << "            }\n"
        << "            (void)changedTile;\n"
        << "        }\n"
        << "    }\n"
        << "    std::fill(scratch.liveMovements.begin(), scratch.liveMovements.end(), 0);\n"
        << "    return movedAny;\n"
        << "}\n\n";

    emitCompactRulegroupFunctions(out, game.rules, suffix, "early");
    emitCompactRulegroupFunctions(out, game.lateRules, suffix, "late");
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
        out << "SpecializedCompactTurnOutcome specialized_compact_turn_core_" << sourceIndex << "(\n"
            << "    LevelDimensions dimensions,\n"
            << "    PersistentLevelState& levelState,\n"
            << "    Scratch& scratch,\n"
            << "    ps_input input,\n"
            << "    RuntimeStepOptions options\n"
            << ") {\n";
        emitCompactTurnCompilerSkeletonBody(out, suffix);
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
