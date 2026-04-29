#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>

#include "runtime/core.hpp"

namespace puzzlescript::search {

enum class SearchMode {
    Bfs,
    WeightedAStar,
    Greedy,
};

struct StateKey {
    uint64_t lo = 0;
    uint64_t hi = 0;

    bool operator==(const StateKey& other) const {
        return lo == other.lo && hi == other.hi;
    }
};

struct StateKeyHash {
    size_t operator()(const StateKey& key) const {
        const uint64_t mixed = key.lo ^ (key.hi + 0x9e3779b97f4a7c15ULL + (key.lo << 6) + (key.lo >> 2));
        return static_cast<size_t>(mixed ^ (mixed >> 32));
    }
};

inline uint64_t mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

inline void appendStateKeyValue(StateKey& key, uint64_t value) {
    const uint64_t mixed = mix64(value + 0x9e3779b97f4a7c15ULL + key.lo);
    key.lo ^= mixed;
    key.lo *= 0x100000001b3ULL;
    key.hi ^= mix64(mixed + key.hi);
    key.hi *= 0x9e3779b185ebca87ULL;
}

template <typename ProjectWord>
inline StateKey fullStateKey(const FullState& session, bool includeRandomState, ProjectWord projectWord) {
    StateKey key{1469598103934665603ull, 7809847782465536322ull};
    appendStateKeyValue(key, static_cast<uint64_t>(static_cast<uint32_t>(session.meta.currentLevelIndex)));
    appendStateKeyValue(key, session.meta.titleScreen ? 1 : 0);
    appendStateKeyValue(key, session.meta.textMode ? 1 : 0);
    appendStateKeyValue(key, session.meta.winning ? 1 : 0);
    appendStateKeyValue(key, session.meta.pendingAgain ? 1 : 0);

    if (includeRandomState) {
        appendStateKeyValue(key, static_cast<uint64_t>(static_cast<uint32_t>(session.levelState.rng.i)));
        appendStateKeyValue(key, static_cast<uint64_t>(static_cast<uint32_t>(session.levelState.rng.j)));
        appendStateKeyValue(key, session.levelState.rng.valid ? 1 : 0);
        uint64_t packed = 0;
        uint32_t shift = 0;
        for (const uint8_t byte : session.levelState.rng.s) {
            packed |= static_cast<uint64_t>(byte) << shift;
            shift += 8;
            if (shift == 64) {
                appendStateKeyValue(key, packed);
                packed = 0;
                shift = 0;
            }
        }
        if (shift != 0) {
            appendStateKeyValue(key, packed);
        }
    }

#if PS_INTERPRETER_OBJECT_MAJOR
    const MaskVector objects = copyInterpreterBoardObjectsAsCellMajor(session);
#else
    const auto& objects = session.scratch.interpreterBoard.objects;
#endif
    for (size_t index = 0; index < objects.size(); ++index) {
        appendStateKeyValue(key, static_cast<uint64_t>(static_cast<MaskWordUnsigned>(projectWord(index, objects[index]))));
    }
    return key;
}

inline StateKey fullStateKey(const FullState& session, bool includeRandomState) {
    return fullStateKey(session, includeRandomState, [](size_t, MaskWord word) { return word; });
}

template <typename ProjectWord>
inline StateKey sessionStateKey(const FullState& state, bool includeRandomState, ProjectWord projectWord) {
    return fullStateKey(state, includeRandomState, projectWord);
}

inline StateKey sessionStateKey(const FullState& state, bool includeRandomState) {
    return fullStateKey(state, includeRandomState);
}

inline int32_t priorityFor(SearchMode mode, uint32_t depth, int32_t heuristic, int32_t weightedAStarWeight) {
    switch (mode) {
        case SearchMode::Bfs: return static_cast<int32_t>(depth);
        case SearchMode::WeightedAStar: return static_cast<int32_t>(depth) + heuristic * weightedAStarWeight;
        case SearchMode::Greedy: return heuristic;
    }
    return static_cast<int32_t>(depth);
}

inline int32_t priorityFor(SearchMode mode, uint32_t depth, int32_t heuristic) {
    return priorityFor(mode, depth, heuristic, 2);
}

inline const MaskWord* maskPtr(const Game& game, MaskOffset offset) {
    if (offset == kNullMaskOffset || offset >= game.maskArena.size()) {
        return nullptr;
    }
    return game.maskArena.data() + offset;
}

inline const MaskWord* cellObjects(const FullState& session, int32_t tileIndex) {
#if PS_INTERPRETER_OBJECT_MAJOR
    MaskVector& result = session.scratch.interpreterBoard.cellScratch;
    result.assign(static_cast<size_t>(session.game->strideObject), 0);
    const int32_t tileCount = currentLevelWidth(session) * currentLevelHeight(session);
    const size_t cellWordCount = static_cast<size_t>((tileCount + static_cast<int32_t>(kMaskWordBits) - 1) / static_cast<int32_t>(kMaskWordBits));
    const size_t bitWord = static_cast<size_t>(maskWordIndex(static_cast<uint32_t>(tileIndex)));
    const MaskWordUnsigned bitMask = MaskWordUnsigned{1} << maskBitIndex(static_cast<uint32_t>(tileIndex));
    for (int32_t objectId = 0; objectId < session.game->objectCount; ++objectId) {
        const size_t offset = static_cast<size_t>(objectId) * cellWordCount + bitWord;
        if (offset >= session.scratch.objectCellBits.size()
            || (session.scratch.objectCellBits[offset] & bitMask) == 0) {
            continue;
        }
        const int32_t word = objectId / static_cast<int32_t>(kMaskWordBits);
        const uint32_t bit = static_cast<uint32_t>(objectId % static_cast<int32_t>(kMaskWordBits));
        if (word >= 0 && word < session.game->strideObject) {
            result[static_cast<size_t>(word)] |= maskBit(bit);
        }
    }
    return result.data();
#else
    return session.scratch.interpreterBoard.objects.data() + static_cast<size_t>(tileIndex * session.game->strideObject);
#endif
}

inline bool anyBits(const MaskWord* lhs, uint32_t lhsCount, const MaskWord* rhs, uint32_t rhsCount) {
    const uint32_t count = std::min(lhsCount, rhsCount);
    for (uint32_t index = 0; index < count; ++index) {
        if ((lhs[index] & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}

inline bool bitsSet(const MaskWord* required, uint32_t requiredCount, const MaskWord* actual, uint32_t actualCount) {
    for (uint32_t index = 0; index < requiredCount; ++index) {
        const MaskWord actualWord = index < actualCount ? actual[index] : 0;
        if ((required[index] & actualWord) != required[index]) {
            return false;
        }
    }
    return true;
}

inline bool matchesFilter(const MaskWord* filter, uint32_t wordCount, bool aggregate, const MaskWord* cell) {
    if (filter == nullptr) {
        return false;
    }
    return aggregate ? bitsSet(filter, wordCount, cell, wordCount) : anyBits(filter, wordCount, cell, wordCount);
}

inline constexpr int32_t kNoMatchingDistance = 64;

inline int32_t distanceOrFallback(int32_t distance) {
    return distance == std::numeric_limits<int32_t>::max() ? kNoMatchingDistance : distance;
}

struct HeuristicScratch {
    std::vector<int32_t> distanceField;
    std::vector<std::vector<int32_t>> conditionDistances;
};

inline void matchingDistanceField(
    const FullState& session,
    const MaskWord* filter,
    bool aggregate,
    std::vector<int32_t>& distances
) {
    const int32_t width = currentLevelWidth(session);
    const int32_t height = currentLevelHeight(session);
    const int32_t tileCount = width * height;
    distances.assign(static_cast<size_t>(tileCount), std::numeric_limits<int32_t>::max());
    if (filter == nullptr) {
        return;
    }

    for (int32_t tile = 0; tile < tileCount; ++tile) {
        if (matchesFilter(filter, session.game->wordCount, aggregate, cellObjects(session, tile))) {
            distances[static_cast<size_t>(tile)] = 0;
        }
    }

    auto relax = [&](int32_t tile, int32_t neighbor) {
        int32_t& distance = distances[static_cast<size_t>(tile)];
        const int32_t neighborDistance = distances[static_cast<size_t>(neighbor)];
        if (neighborDistance != std::numeric_limits<int32_t>::max()) {
            distance = std::min(distance, neighborDistance + 1);
        }
    };

    for (int32_t x = 0; x < width; ++x) {
        for (int32_t y = 0; y < height; ++y) {
            const int32_t tile = x * height + y;
            if (x > 0) relax(tile, (x - 1) * height + y);
            if (y > 0) relax(tile, x * height + (y - 1));
        }
    }
    for (int32_t x = width - 1; x >= 0; --x) {
        for (int32_t y = height - 1; y >= 0; --y) {
            const int32_t tile = x * height + y;
            if (x + 1 < width) relax(tile, (x + 1) * height + y);
            if (y + 1 < height) relax(tile, x * height + (y + 1));
        }
    }
}

inline std::vector<int32_t> matchingDistanceField(const FullState& session, const MaskWord* filter, bool aggregate) {
    std::vector<int32_t> distances;
    matchingDistanceField(session, filter, aggregate, distances);
    return distances;
}

struct HeuristicOptions {
    bool includeNoQuantifierPenalty = false;
    bool includePlayerDistance = false;
};

inline int32_t winConditionHeuristicScore(const FullState& session, HeuristicOptions options, HeuristicScratch& scratch) {
    const Game& game = *session.game;
    if (game.winConditions.empty()) {
        return 0;
    }

    int32_t score = 0;
    const int32_t tileCount = currentLevelWidth(session) * currentLevelHeight(session);
    for (const auto& condition : game.winConditions) {
        const MaskWord* filter1 = maskPtr(game, condition.filter1);
        const MaskWord* filter2 = maskPtr(game, condition.filter2);
        if (filter1 == nullptr || filter2 == nullptr) {
            continue;
        }
        matchingDistanceField(session, filter2, condition.aggr2, scratch.distanceField);
        if (condition.quantifier == 1) {
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                const MaskWord* cell = cellObjects(session, tile);
                if (!matchesFilter(filter1, game.wordCount, condition.aggr1, cell)) {
                    continue;
                }
                if (matchesFilter(filter2, game.wordCount, condition.aggr2, cell)) {
                    continue;
                }
                score += 10 + distanceOrFallback(scratch.distanceField[static_cast<size_t>(tile)]);
            }
        } else if (condition.quantifier == 0) {
            bool passed = false;
            int32_t best = kNoMatchingDistance;
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                const MaskWord* cell = cellObjects(session, tile);
                if (!matchesFilter(filter1, game.wordCount, condition.aggr1, cell)) {
                    continue;
                }
                if (matchesFilter(filter2, game.wordCount, condition.aggr2, cell)) {
                    passed = true;
                    break;
                }
                best = std::min(best, distanceOrFallback(scratch.distanceField[static_cast<size_t>(tile)]));
            }
            score += passed ? 0 : best;
        } else if (options.includeNoQuantifierPenalty && condition.quantifier == -1) {
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                const MaskWord* cell = cellObjects(session, tile);
                if (matchesFilter(filter1, game.wordCount, condition.aggr1, cell)
                    && matchesFilter(filter2, game.wordCount, condition.aggr2, cell)) {
                    score += 10;
                }
            }
        }
    }

    if (options.includePlayerDistance && game.playerMask != kNullMaskOffset && score > 0) {
        const MaskWord* playerMask = maskPtr(game, game.playerMask);
        bool hasPlayer = false;
        int32_t best = kNoMatchingDistance;
        scratch.conditionDistances.resize(game.winConditions.size());
        for (size_t index = 0; index < game.winConditions.size(); ++index) {
            const auto& condition = game.winConditions[index];
            matchingDistanceField(session, maskPtr(game, condition.filter1), condition.aggr1, scratch.conditionDistances[index]);
        }
        for (int32_t player = 0; player < tileCount; ++player) {
            if (!matchesFilter(playerMask, game.wordCount, game.playerMaskAggregate, cellObjects(session, player))) {
                continue;
            }
            hasPlayer = true;
            for (const auto& distances : scratch.conditionDistances) {
                best = std::min(best, distanceOrFallback(distances[static_cast<size_t>(player)]));
            }
        }
        if (hasPlayer) {
            score += std::min(best, 16);
        }
    }

    return score;
}

inline int32_t winConditionHeuristicScore(const FullState& session, HeuristicOptions options = {}) {
    HeuristicScratch scratch;
    return winConditionHeuristicScore(session, options, scratch);
}

} // namespace puzzlescript::search
