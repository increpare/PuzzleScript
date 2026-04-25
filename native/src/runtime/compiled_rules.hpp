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

using CompiledTickStepFn = CompiledTickApplyOutcome (*)(Session& session, ps_input input, RuntimeStepOptions options);
using CompiledTickFn = CompiledTickApplyOutcome (*)(Session& session, RuntimeStepOptions options);

struct CompiledTickBackend {
    uint64_t sourceHash = 0;
    const char* name = nullptr;
    CompiledTickStepFn step = nullptr;
    CompiledTickFn tick = nullptr;
};

uint64_t compiledRulesHashSource(std::string_view source);
void attachLinkedCompiledRules(Game& game, std::string_view source);

const MaskWord* compiledRuleMaskPtr(const Game& game, MaskOffset offset);
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
void compiledRuleRebuildMasks(Session& session);
void compiledRuleQueueCommands(const Rule& rule, CommandState& commands);
using CompiledRuleRowMatch = std::vector<int32_t>;
void compiledRuleCollectRowMatches(Session& session, const Rule& rule, size_t rowIndex, std::vector<CompiledRuleRowMatch>& outMatches);
bool compiledRuleRowMatchStillMatches(const Session& session, const Rule& rule, size_t rowIndex, const CompiledRuleRowMatch& match);
bool compiledRuleApplyRowMatch(Session& session, const Rule& rule, size_t rowIndex, const CompiledRuleRowMatch& match);
bool compiledRuleApplyRandomGroup(Session& session, const std::vector<Rule>& group, CommandState& commands);

} // namespace puzzlescript
