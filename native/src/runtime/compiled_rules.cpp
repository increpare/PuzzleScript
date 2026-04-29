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

void materializeCompactBridgeState(const Game& game, CompactStateView state, FullState& session) {
    session.levelState.liveLevel.width = state.width;
    session.levelState.liveLevel.height = state.height;
    session.levelState.liveLevel.layerCount = game.layerCount;
    const int32_t tileCount = state.width * state.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    session.levelState.liveLevel.objects.assign(static_cast<size_t>(std::max(tileCount, 0) * std::max(game.strideObject, 0)), 0);
    for (int32_t objectId = 0; objectId < game.objectCount; ++objectId) {
        const size_t objectBase = static_cast<size_t>(objectId) * cellWordCount;
        for (size_t bitWord = 0; bitWord < cellWordCount; ++bitWord) {
            uint64_t bits = objectBase + bitWord < state.objectBitWordCount
                ? state.objectBits[objectBase + bitWord]
                : 0;
            while (bits != 0) {
                const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(bits));
                const int32_t tileIndex = static_cast<int32_t>(bitWord * 64 + bit);
                if (tileIndex < tileCount) {
                    const int32_t word = objectId / static_cast<int32_t>(kMaskWordBits);
                    const uint32_t objectBit = static_cast<uint32_t>(objectId % static_cast<int32_t>(kMaskWordBits));
                    session.levelState.liveLevel.objects[static_cast<size_t>(tileIndex * game.strideObject + word)] |= maskBit(objectBit);
                }
                bits &= bits - 1;
            }
        }
    }
    const size_t movementWordCount = static_cast<size_t>(std::max(tileCount, 0) * std::max(game.strideMovement, 0));
    session.scratch.liveMovements.assign(movementWordCount, 0);
    if (state.movementWords != nullptr && state.movementWordCount == movementWordCount) {
        std::copy(state.movementWords, state.movementWords + state.movementWordCount, session.scratch.liveMovements.begin());
    }
    session.scratch.rigidGroupIndexMasks.assign(session.scratch.liveMovements.size(), 0);
    session.scratch.rigidMovementAppliedMasks.assign(session.scratch.liveMovements.size(), 0);
    session.scratch.pendingCreateMask.clear();
    session.scratch.pendingDestroyMask.clear();
    session.meta.pendingAgain = false;
    session.meta.undoStack.clear();
    if (state.randomStateS != nullptr
        && state.randomStateSize == session.levelState.rng.s.size()
        && state.randomStateI != nullptr
        && state.randomStateJ != nullptr
        && state.randomStateValid != nullptr) {
        session.levelState.rng.i = *state.randomStateI;
        session.levelState.rng.j = *state.randomStateJ;
        session.levelState.rng.valid = *state.randomStateValid;
        std::copy(state.randomStateS, state.randomStateS + state.randomStateSize, session.levelState.rng.s.begin());
    }
    resizeBoardOccupancyObjectBits(session);
    syncOccupancyRngFromAuthoritativeRandomState(session);
    markCompactBridgeFullStateDirty(session);
    compiledRuleRebuildMasks(session);
}

void copyCompactBridgeStateBack(const FullState& session, CompactStateView state) {
    const Game& game = *session.game;
    const int32_t tileCount = session.levelState.liveLevel.width * session.levelState.liveLevel.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    const size_t requiredWords = static_cast<size_t>(std::max(game.objectCount, 0)) * cellWordCount;
    if (state.objectBits != nullptr && state.objectBitWordCount >= requiredWords) {
        std::fill(state.objectBits, state.objectBits + state.objectBitWordCount, 0);
        for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            const size_t sourceBase = static_cast<size_t>(tileIndex * game.strideObject);
            const size_t bitWord = static_cast<size_t>(tileIndex >> 6);
            const uint64_t bitMask = uint64_t{1} << static_cast<uint32_t>(tileIndex & 63);
            for (int32_t word = 0; word < game.strideObject; ++word) {
                MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(session.levelState.liveLevel.objects[sourceBase + static_cast<size_t>(word)]);
                while (bits != 0) {
                    const uint32_t bit = static_cast<uint32_t>(
                        sizeof(MaskWordUnsigned) <= sizeof(unsigned int)
                            ? __builtin_ctz(static_cast<unsigned int>(bits))
                            : __builtin_ctzll(static_cast<unsigned long long>(bits))
                    );
                    const int32_t objectId = word * static_cast<int32_t>(kMaskWordBits) + static_cast<int32_t>(bit);
                    if (objectId < game.objectCount) {
                        state.objectBits[static_cast<size_t>(objectId) * cellWordCount + bitWord] |= bitMask;
                    }
                    bits &= bits - 1;
                }
            }
        }
    }
    if (state.randomStateS != nullptr
        && state.randomStateSize == session.levelState.rng.s.size()
        && state.randomStateI != nullptr
        && state.randomStateJ != nullptr
        && state.randomStateValid != nullptr) {
        *state.randomStateI = session.levelState.rng.i;
        *state.randomStateJ = session.levelState.rng.j;
        *state.randomStateValid = session.levelState.rng.valid;
        std::copy(session.levelState.rng.s.begin(), session.levelState.rng.s.end(), state.randomStateS);
    }
    if (state.movementWords != nullptr && state.movementWordCount == session.scratch.liveMovements.size()) {
        std::copy(session.scratch.liveMovements.begin(), session.scratch.liveMovements.end(), state.movementWords);
    }
}

} // namespace

SpecializedCompactTurnOutcome compactStateInterpretedTurnBridge(
    const Game& game,
    CompactStateView state,
    ps_input input,
    RuntimeStepOptions options
) {
    if (state.objectBits == nullptr || state.width <= 0 || state.height <= 0) {
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
    if (state.currentLevelIndex >= 0 && static_cast<size_t>(state.currentLevelIndex) < game.levels.size()) {
        if (auto error = loadLevel(*session, state.currentLevelIndex)) {
            (void)error;
            return {false, {}};
        }
    }
    addProfileNs(RuntimeCounterId::CompactTurnBridgeCreateNs);
    materializeCompactBridgeState(game, state, *session);
    addProfileNs(RuntimeCounterId::CompactTurnBridgeMaterializeNs);
    RuntimeStepOptions drainOptions = options;
    drainOptions.againPolicy = AgainPolicy::Drain;
    ps_step_result result = interpretedTurn(*session, input, drainOptions);
    addProfileNs(RuntimeCounterId::CompactTurnBridgeTurnNs);
    copyCompactBridgeStateBack(*session, state);
    addProfileNs(RuntimeCounterId::CompactTurnBridgeCopybackNs);
    return {true, result};
}

} // namespace puzzlescript
