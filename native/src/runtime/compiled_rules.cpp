#include "runtime/compiled_rules.hpp"

#include "runtime/hash.hpp"

#include <algorithm>
#include <memory>

extern "C" __attribute__((weak))
const puzzlescript::CompiledRulesBackend* ps_compiled_rules_find_backend(uint64_t) {
    return nullptr;
}

extern "C" __attribute__((weak))
const puzzlescript::CompiledTickBackend* ps_compiled_tick_find_backend(uint64_t) {
    return nullptr;
}

extern "C" __attribute__((weak))
const puzzlescript::CompiledCompactTickBackend* ps_compiled_compact_tick_find_backend(uint64_t) {
    return nullptr;
}

namespace puzzlescript {

uint64_t compiledRulesHashSource(std::string_view source) {
    return fnv1a64String(source);
}

void attachLinkedCompiledRules(Game& game, std::string_view source) {
    const uint64_t sourceHash = compiledRulesHashSource(source);
    game.compiledRules = ps_compiled_rules_find_backend(sourceHash);
    game.compiledTick = ps_compiled_tick_find_backend(sourceHash);
    game.compiledCompactTick = ps_compiled_compact_tick_find_backend(sourceHash);
}

namespace {

void markCompactBridgeSessionDirty(Session& session) {
    std::fill(session.dirtyObjectRows.begin(), session.dirtyObjectRows.end(), 1);
    std::fill(session.dirtyObjectColumns.begin(), session.dirtyObjectColumns.end(), 1);
    std::fill(session.dirtyMovementRows.begin(), session.dirtyMovementRows.end(), 1);
    std::fill(session.dirtyMovementColumns.begin(), session.dirtyMovementColumns.end(), 1);
    session.dirtyObjectBoard = true;
    session.dirtyMovementBoard = true;
    session.objectCellIndexDirty = true;
    session.anyMasksDirty = true;
}

void materializeCompactBridgeState(const Game& game, CompiledCompactTickStateView state, Session& session) {
    session.liveLevel.isMessage = false;
    session.liveLevel.message.clear();
    session.liveLevel.width = state.width;
    session.liveLevel.height = state.height;
    session.liveLevel.layerCount = game.layerCount;
    const int32_t tileCount = state.width * state.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    session.liveLevel.objects.assign(static_cast<size_t>(std::max(tileCount, 0) * std::max(game.strideObject, 0)), 0);
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
                    session.liveLevel.objects[static_cast<size_t>(tileIndex * game.strideObject + word)] |= maskBit(objectBit);
                }
                bits &= bits - 1;
            }
        }
    }
    session.liveMovements.assign(static_cast<size_t>(std::max(tileCount, 0) * std::max(game.strideMovement, 0)), 0);
    session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
    session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
    session.pendingCreateMask.clear();
    session.pendingDestroyMask.clear();
    session.pendingAgain = false;
    session.canUndo = false;
    session.undoStack.clear();
    session.lastAudioEvents.clear();
    session.lastUiAudioEvents.clear();
    if (state.randomStateS != nullptr
        && state.randomStateSize == session.randomState.s.size()
        && state.randomStateI != nullptr
        && state.randomStateJ != nullptr
        && state.randomStateValid != nullptr) {
        session.randomState.i = *state.randomStateI;
        session.randomState.j = *state.randomStateJ;
        session.randomState.valid = *state.randomStateValid;
        std::copy(state.randomStateS, state.randomStateS + state.randomStateSize, session.randomState.s.begin());
    }
    markCompactBridgeSessionDirty(session);
    compiledRuleRebuildMasks(session);
}

void copyCompactBridgeStateBack(const Session& session, CompiledCompactTickStateView state) {
    const Game& game = *session.game;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    const size_t requiredWords = static_cast<size_t>(std::max(game.objectCount, 0)) * cellWordCount;
    if (state.objectBits != nullptr && state.objectBitWordCount >= requiredWords) {
        std::fill(state.objectBits, state.objectBits + state.objectBitWordCount, 0);
        for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            const size_t sourceBase = static_cast<size_t>(tileIndex * game.strideObject);
            const size_t bitWord = static_cast<size_t>(tileIndex >> 6);
            const uint64_t bitMask = uint64_t{1} << static_cast<uint32_t>(tileIndex & 63);
            for (int32_t word = 0; word < game.strideObject; ++word) {
                MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(session.liveLevel.objects[sourceBase + static_cast<size_t>(word)]);
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
        && state.randomStateSize == session.randomState.s.size()
        && state.randomStateI != nullptr
        && state.randomStateJ != nullptr
        && state.randomStateValid != nullptr) {
        *state.randomStateI = session.randomState.i;
        *state.randomStateJ = session.randomState.j;
        *state.randomStateValid = session.randomState.valid;
        std::copy(session.randomState.s.begin(), session.randomState.s.end(), state.randomStateS);
    }
}

} // namespace

CompiledCompactTickApplyOutcome compiledCompactTickInterpreterBridge(
    const Game& game,
    CompiledCompactTickStateView state,
    ps_input input,
    RuntimeStepOptions options
) {
    if (state.objectBits == nullptr || state.width <= 0 || state.height <= 0) {
        return {false, {}};
    }
    std::shared_ptr<const Game> gameRef(&game, [](const Game*) {});
    std::unique_ptr<Session> session = createSession(std::move(gameRef));
    if (state.currentLevelIndex >= 0 && static_cast<size_t>(state.currentLevelIndex) < game.levels.size()) {
        if (auto error = loadLevel(*session, state.currentLevelIndex)) {
            (void)error;
            return {false, {}};
        }
    }
    materializeCompactBridgeState(game, state, *session);
    ps_step_result result = step(*session, input, options);
    settlePendingAgain(*session, options);
    copyCompactBridgeStateBack(*session, state);
    return {true, result};
}

} // namespace puzzlescript
