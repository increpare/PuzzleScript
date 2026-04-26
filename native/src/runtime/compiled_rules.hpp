#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "runtime/core.hpp"

namespace puzzlescript {

struct CompiledRuleApplyOutcome {
    bool handled = false;
    bool changed = false;
};

using CompiledRuleGroupFn = CompiledRuleApplyOutcome (*)(Session& session, int32_t groupIndex, bool late, CommandState& commands);

struct CompiledRulesBackend {
    uint64_t sourceHash = 0;
    const char* name = nullptr;
    CompiledRuleGroupFn applyGroup = nullptr;
    uint32_t compiledRuleCount = 0;
    uint32_t compiledGroupCount = 0;
};

struct CompiledTickApplyOutcome {
    bool handled = false;
    ps_step_result result{};
};

struct CompiledTickSupportInfo {
    bool wholeTurnSupported = false;
    const char* wholeTurnFallbackReason = nullptr;
};

struct CompiledTickRuleGroupsOutcome {
    bool handled = false;
    bool changed = false;
};

using CompiledTickStepFn = CompiledTickApplyOutcome (*)(Session& session, ps_input input, RuntimeStepOptions options);
using CompiledTickFn = CompiledTickApplyOutcome (*)(Session& session, RuntimeStepOptions options);

struct CompiledTickBackend {
    uint64_t sourceHash = 0;
    const char* name = nullptr;
    CompiledTickStepFn step = nullptr;
    CompiledTickFn tick = nullptr;
    CompiledTickSupportInfo support{};
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
    CompiledTickSupportInfo support{};
};

using CompiledCompactTickStateView = CompactStateView;
using CompiledCompactTickApplyOutcome = SpecializedCompactTurnOutcome;
using CompiledCompactTickStepFn = SpecializedCompactTurnFn;
using CompiledCompactTickBackend = SpecializedCompactTurnBackend;

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
inline SpecializedCompactTurnOutcome compiledCompactTickInterpreterBridge(
    const Game& game,
    CompactStateView state,
    ps_input input,
    RuntimeStepOptions options
) {
    return compactStateInterpretedTurnBridge(game, state, input, options);
}
const MaskWord* compiledRuleCellObjects(const Session& session, int32_t tileIndex);
const MaskWord* compiledRuleCellMovements(const Session& session, int32_t tileIndex);
bool compiledRuleBitsSet(const MaskWord* required, size_t requiredCount, const MaskWord* actual, size_t actualCount);
bool compiledRuleAnyBits(const MaskWord* lhs, size_t lhsCount, const MaskWord* rhs, size_t rhsCount);
void compiledRuleSetCellObjectsFromWords(
    Session& session,
    int32_t tileIndex,
    const MaskWord* objects,
    const MaskWord* created,
    const MaskWord* destroyed
);
void compiledRuleSetCellMovementsFromWords(Session& session, int32_t tileIndex, const MaskWord* movements);
void compiledRuleSetCellObjectsWord1(
    Session& session,
    int32_t tileIndex,
    MaskWord objects,
    MaskWord created,
    MaskWord destroyed
);
void compiledRuleSetCellMovementsWord1(Session& session, int32_t tileIndex, MaskWord movements);
void compiledRuleRebuildMasks(Session& session);
void compiledRuleQueueCommands(const Rule& rule, CommandState& commands);
bool compiledRulePrepareCommandQueue(CommandState& commands, bool currentRuleCancel, bool currentRuleRestart);
void compiledRuleQueueKnownCommand(
    CommandState& commands,
    CompiledRuleCommandKind kind,
    std::string_view name,
    std::string_view argument = {}
);
using CompiledRuleRowMatch = std::vector<int32_t>;
void compiledRuleCollectRowMatches(Session& session, const Rule& rule, size_t rowIndex, std::vector<CompiledRuleRowMatch>& outMatches);
bool compiledRuleRowMatchStillMatches(const Session& session, const Rule& rule, size_t rowIndex, const CompiledRuleRowMatch& match);
bool compiledRuleApplyRowMatch(Session& session, const Rule& rule, size_t rowIndex, const CompiledRuleRowMatch& match);
bool compiledRuleApplyRandomGroup(Session& session, const std::vector<Rule>& group, CommandState& commands);

} // namespace puzzlescript
