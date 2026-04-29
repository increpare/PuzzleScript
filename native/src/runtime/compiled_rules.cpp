#include "runtime/compiled_rules.hpp"

#include "runtime/hash.hpp"

#include <algorithm>
#include <memory>

extern "C" __attribute__((weak))
const puzzlescript::SpecializedRulegroupsBackend* ps_specialized_rulegroups_find_backend(uint64_t) {
    return nullptr;
}

extern "C" __attribute__((weak))
const puzzlescript::SpecializedRulegroupsBackend* ps_compiled_rules_find_backend(uint64_t) {
    return nullptr;
}

extern "C" __attribute__((weak))
const puzzlescript::SpecializedFullTurnBackend* ps_specialized_full_turn_find_backend(uint64_t) {
    return nullptr;
}

extern "C" __attribute__((weak))
const puzzlescript::SpecializedFullTurnBackend* ps_compiled_tick_find_backend(uint64_t) {
    return nullptr;
}

extern "C" __attribute__((weak))
const puzzlescript::SpecializedCompactTurnBackend* ps_specialized_compact_turn_find_backend(uint64_t) {
    return nullptr;
}

extern "C" __attribute__((weak))
const puzzlescript::SpecializedCompactTurnBackend* ps_compiled_compact_tick_find_backend(uint64_t) {
    return nullptr;
}

namespace puzzlescript {

uint64_t compiledRulesHashSource(std::string_view source) {
    return fnv1a64String(source);
}

void attachLinkedCompiledRules(Game& game, std::string_view source) {
    const uint64_t sourceHash = compiledRulesHashSource(source);
    game.specializedRulegroups = ps_specialized_rulegroups_find_backend(sourceHash);
    if (game.specializedRulegroups == nullptr) {
        game.specializedRulegroups = ps_compiled_rules_find_backend(sourceHash);
    }
    game.specializedFullTurn = ps_specialized_full_turn_find_backend(sourceHash);
    if (game.specializedFullTurn == nullptr) {
        game.specializedFullTurn = ps_compiled_tick_find_backend(sourceHash);
    }
    game.specializedCompactTurn = ps_specialized_compact_turn_find_backend(sourceHash);
    if (game.specializedCompactTurn == nullptr) {
        game.specializedCompactTurn = ps_compiled_compact_tick_find_backend(sourceHash);
    }
}

namespace {

void markCompactBridgeFullStateDirty(FullState& session) {
    std::fill(session.scratch.dirtyObjectRows.begin(), session.scratch.dirtyObjectRows.end(), 1);
    std::fill(session.scratch.dirtyObjectColumns.begin(), session.scratch.dirtyObjectColumns.end(), 1);
    std::fill(session.scratch.dirtyMovementRows.begin(), session.scratch.dirtyMovementRows.end(), 1);
    std::fill(session.scratch.dirtyMovementColumns.begin(), session.scratch.dirtyMovementColumns.end(), 1);
    session.scratch.dirtyObjectBoard = true;
    session.scratch.dirtyMovementBoard = true;
    session.scratch.objectCellIndexDirty = true;
    session.scratch.anyMasksDirty = true;
}

void materializeCompactBridgeState(
    const Game& game,
    const PersistentLevelState& levelState,
    const Scratch& scratch,
    LevelDimensions dimensions,
    FullState& session
) {
    session.meta.level.width = dimensions.width;
    session.meta.level.height = dimensions.height;
    session.meta.levelDimensions = dimensions;
    const int32_t tileCount = dimensions.width * dimensions.height;
    fillInterpreterBoardObjectsFromCompactObjectBits(
        game,
        dimensions,
        levelState.board.objectBits,
        session.scratch.interpreterBoard.objects
    );
    const size_t movementWordCount = static_cast<size_t>(std::max(tileCount, 0) * std::max(game.strideMovement, 0));
    session.scratch.liveMovements.assign(movementWordCount, 0);
    if (scratch.liveMovements.size() == movementWordCount) {
        std::copy(scratch.liveMovements.begin(), scratch.liveMovements.end(), session.scratch.liveMovements.begin());
    }
    session.scratch.rigidGroupIndexMasks.assign(session.scratch.liveMovements.size(), 0);
    session.scratch.rigidMovementAppliedMasks.assign(session.scratch.liveMovements.size(), 0);
    session.scratch.pendingCreateMask.clear();
    session.scratch.pendingDestroyMask.clear();
    session.meta.pendingAgain = false;
    session.meta.undoStack.clear();
    if (levelState.rng.s.size() == session.levelState.rng.s.size()) {
        session.levelState.rng = levelState.rng;
    }
    syncPersistentLevelStateFromScratch(session);
    markCompactBridgeFullStateDirty(session);
    compiledRuleRebuildMasks(session);
}

void copyCompactBridgeStateBack(const FullState& session, PersistentLevelState& levelState, Scratch& scratch) {
    fillCompactOccupancyBitsFromInterpreterBoard(session, levelState.board.objectBits);
    levelState.rng = session.levelState.rng;
    scratch.liveMovements = session.scratch.liveMovements;
}

} // namespace

SpecializedCompactTurnOutcome compactStateInterpretedTurnBridge(
    const Game& game,
    PersistentLevelState& levelState,
    Scratch& scratch,
    SpecializedCompactTurnContext context,
    ps_input input,
    RuntimeStepOptions options
) {
    if (levelState.board.objectBits.empty() || context.dimensions.width <= 0 || context.dimensions.height <= 0) {
        return {false, {}};
    }
    const bool profileCompactTurn = runtimeCountersEnabled();
    uint64_t profileMarkNs = profileCompactTurn ? runtimeCounterNowNs() : 0;
    auto addProfileNs = [&](RuntimeCounterId id) {
        if (!profileCompactTurn) {
            return;
        }
        const uint64_t nowNs = runtimeCounterNowNs();
        addRuntimeCounter(id, nowNs - profileMarkNs);
        profileMarkNs = nowNs;
    };
    if (profileCompactTurn) {
        addRuntimeCounter(RuntimeCounterId::CompactTurnBridgeCalls);
    }
    std::shared_ptr<const Game> gameRef(&game, [](const Game*) {});
    LoadedGame loadedGame{std::move(gameRef), MetaGameState{}};
    std::unique_ptr<FullState> session = createFullState(loadedGame);
    if (context.currentLevelIndex >= 0 && static_cast<size_t>(context.currentLevelIndex) < game.levels.size()) {
        if (auto error = loadLevel(*session, context.currentLevelIndex)) {
            (void)error;
            return {false, {}};
        }
    }
    addProfileNs(RuntimeCounterId::CompactTurnBridgeCreateNs);
    materializeCompactBridgeState(game, levelState, scratch, context.dimensions, *session);
    addProfileNs(RuntimeCounterId::CompactTurnBridgeMaterializeNs);
    RuntimeStepOptions drainOptions = options;
    drainOptions.againPolicy = AgainPolicy::Drain;
    ps_step_result result = interpretedTurn(*session, input, drainOptions);
    addProfileNs(RuntimeCounterId::CompactTurnBridgeTurnNs);
    copyCompactBridgeStateBack(*session, levelState, scratch);
    addProfileNs(RuntimeCounterId::CompactTurnBridgeCopybackNs);
    return {true, result};
}

} // namespace puzzlescript
