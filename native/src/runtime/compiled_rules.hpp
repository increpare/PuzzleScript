#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "runtime/core.hpp"

namespace puzzlescript {

struct SpecializedRulegroupOutcome {
    bool handled = false;
    bool changed = false;
};

using SpecializedRulegroupFn = SpecializedRulegroupOutcome (*)(FullState& state, int32_t groupIndex, bool late, CommandState& commands);

struct SpecializedRulegroupsBackend {
    uint64_t sourceHash = 0;
    const char* name = nullptr;
    SpecializedRulegroupFn applyGroup = nullptr;
    uint32_t specializedRuleCount = 0;
    uint32_t specializedGroupCount = 0;
};

struct SpecializedFullTurnOutcome {
    bool handled = false;
    ps_step_result result{};
};

struct SpecializedFullTurnSupportInfo {
    bool wholeTurnSupported = false;
    const char* wholeTurnFallbackReason = nullptr;
};

struct SpecializedRulegroupsForInterpretedTurnOutcome {
    bool handled = false;
    bool changed = false;
};

using SpecializedFullTurnStepFn = SpecializedFullTurnOutcome (*)(FullState& state, ps_input input, RuntimeStepOptions options);
using SpecializedFullTurnTickFn = SpecializedFullTurnOutcome (*)(FullState& state, RuntimeStepOptions options);

struct SpecializedFullTurnBackend {
    uint64_t sourceHash = 0;
    const char* name = nullptr;
    SpecializedFullTurnStepFn step = nullptr;
    SpecializedFullTurnTickFn tick = nullptr;
    SpecializedFullTurnSupportInfo support{};
};

struct CompactStateView {
    uint64_t* objectBits = nullptr;
    size_t objectBitWordCount = 0;
    MaskWord* movementWords = nullptr;
    size_t movementWordCount = 0;
    int32_t width = 0;
    int32_t height = 0;
    uint8_t* randomStateS = nullptr;
    size_t randomStateSize = 0;
    uint8_t* randomStateI = nullptr;
    uint8_t* randomStateJ = nullptr;
    bool* randomStateValid = nullptr;
    int32_t currentLevelIndex = 0;
};

struct SpecializedCompactTurnOutcome {
    bool handled = false;
    ps_step_result result{};
};

using SpecializedCompactTurnFn = SpecializedCompactTurnOutcome (*)(
    const Game& game,
    CompactStateView state,
    ps_input input,
    RuntimeStepOptions options
);

struct SpecializedCompactTurnBackend {
    uint64_t sourceHash = 0;
    const char* name = nullptr;
    SpecializedCompactTurnFn step = nullptr;
    SpecializedFullTurnSupportInfo support{};
};

enum class CompiledRuleCommandKind {
    Again,
    Cancel,
    Checkpoint,
    Message,
    Restart,
    Win,
    Output,
};

uint64_t compiledRulesHashSource(std::string_view source);
void attachLinkedCompiledRules(Game& game, std::string_view source);

const MaskWord* compiledRuleMaskPtr(const Game& game, MaskOffset offset);
SpecializedCompactTurnOutcome compactStateInterpretedTurnBridge(
    const Game& game,
    CompactStateView state,
    ps_input input,
    RuntimeStepOptions options
);
const MaskWord* compiledRuleCellObjects(const FullState& state, int32_t tileIndex);
const MaskWord* compiledRuleCellMovements(const FullState& state, int32_t tileIndex);
bool compiledRuleBitsSet(const MaskWord* required, size_t requiredCount, const MaskWord* actual, size_t actualCount);
bool compiledRuleAnyBits(const MaskWord* lhs, size_t lhsCount, const MaskWord* rhs, size_t rhsCount);
void compiledRuleSetCellObjectsFromWords(
    FullState& state,
    int32_t tileIndex,
    const MaskWord* objects,
    const MaskWord* created,
    const MaskWord* destroyed
);
void compiledRuleSetCellMovementsFromWords(FullState& state, int32_t tileIndex, const MaskWord* movements);
void compiledRuleSetCellObjectsWord1(
    FullState& state,
    int32_t tileIndex,
    MaskWord objects,
    MaskWord created,
    MaskWord destroyed
);
void compiledRuleSetCellMovementsWord1(FullState& state, int32_t tileIndex, MaskWord movements);
void compiledRuleRebuildMasks(FullState& state);
void compiledRuleQueueCommands(const Rule& rule, CommandState& commands);
bool compiledRulePrepareCommandQueue(CommandState& commands, bool currentRuleCancel, bool currentRuleRestart);
void compiledRuleQueueKnownCommand(
    CommandState& commands,
    CompiledRuleCommandKind kind,
    std::string_view name,
    std::string_view argument = {}
);
using CompiledRuleRowMatch = std::vector<int32_t>;
void compiledRuleCollectRowMatches(FullState& state, const Rule& rule, size_t rowIndex, std::vector<CompiledRuleRowMatch>& outMatches);
bool compiledRuleRowMatchStillMatches(const FullState& state, const Rule& rule, size_t rowIndex, const CompiledRuleRowMatch& match);
bool compiledRuleApplyRowMatch(FullState& state, const Rule& rule, size_t rowIndex, const CompiledRuleRowMatch& match);
bool compiledRuleApplyRandomGroup(FullState& state, const std::vector<Rule>& group, CommandState& commands);

} // namespace puzzlescript
