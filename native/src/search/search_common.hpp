#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <limits>

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
inline StateKey sessionStateKey(const Session& session, bool includeRandomState, ProjectWord projectWord) {
    StateKey key{1469598103934665603ull, 7809847782465536322ull};
    appendStateKeyValue(key, static_cast<uint64_t>(static_cast<uint32_t>(session.preparedSession.currentLevelIndex)));
    appendStateKeyValue(key, session.preparedSession.titleScreen ? 1 : 0);
    appendStateKeyValue(key, session.preparedSession.textMode ? 1 : 0);
    appendStateKeyValue(key, session.preparedSession.winning ? 1 : 0);
    appendStateKeyValue(key, session.pendingAgain ? 1 : 0);

    if (includeRandomState) {
        appendStateKeyValue(key, static_cast<uint64_t>(static_cast<uint32_t>(session.randomState.i)));
        appendStateKeyValue(key, static_cast<uint64_t>(static_cast<uint32_t>(session.randomState.j)));
        appendStateKeyValue(key, session.randomState.valid ? 1 : 0);
        uint64_t packed = 0;
        uint32_t shift = 0;
        for (const uint8_t byte : session.randomState.s) {
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

    const auto& objects = session.liveLevel.objects;
    for (size_t index = 0; index < objects.size(); ++index) {
        appendStateKeyValue(key, static_cast<uint64_t>(static_cast<MaskWordUnsigned>(projectWord(index, objects[index]))));
    }
    return key;
}

inline StateKey sessionStateKey(const Session& session, bool includeRandomState) {
    return sessionStateKey(session, includeRandomState, [](size_t, MaskWord word) { return word; });
}

inline int32_t priorityFor(SearchMode mode, uint32_t depth, int32_t heuristic) {
    switch (mode) {
        case SearchMode::Bfs: return static_cast<int32_t>(depth);
        case SearchMode::WeightedAStar: return static_cast<int32_t>(depth) + heuristic * 2;
        case SearchMode::Greedy: return heuristic;
    }
    return static_cast<int32_t>(depth);
}

inline const MaskWord* maskPtr(const Game& game, MaskOffset offset) {
    if (offset == kNullMaskOffset || offset >= game.maskArena.size()) {
        return nullptr;
    }
    return game.maskArena.data() + offset;
}

inline const MaskWord* cellObjects(const Session& session, int32_t tileIndex) {
    return session.liveLevel.objects.data() + static_cast<size_t>(tileIndex * session.game->strideObject);
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

inline int32_t tileX(const Session& session, int32_t tileIndex) {
    return tileIndex / session.liveLevel.height;
}

inline int32_t tileY(const Session& session, int32_t tileIndex) {
    return tileIndex % session.liveLevel.height;
}

inline int32_t manhattan(const Session& session, int32_t a, int32_t b) {
    return std::abs(tileX(session, a) - tileX(session, b)) + std::abs(tileY(session, a) - tileY(session, b));
}

inline int32_t nearestMatchingDistance(const Session& session, int32_t tile, const MaskWord* filter, bool aggregate) {
    if (filter == nullptr) {
        return 64;
    }
    int32_t best = std::numeric_limits<int32_t>::max();
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    for (int32_t target = 0; target < tileCount; ++target) {
        if (matchesFilter(filter, session.game->wordCount, aggregate, cellObjects(session, target))) {
            best = std::min(best, manhattan(session, tile, target));
        }
    }
    return best == std::numeric_limits<int32_t>::max() ? 64 : best;
}

struct HeuristicOptions {
    bool includeNoQuantifierPenalty = false;
    bool includePlayerDistance = false;
};

inline int32_t winConditionHeuristicScore(const Session& session, HeuristicOptions options = {}) {
    const Game& game = *session.game;
    if (game.winConditions.empty()) {
        return 0;
    }

    int32_t score = 0;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    for (const auto& condition : game.winConditions) {
        const MaskWord* filter1 = maskPtr(game, condition.filter1);
        const MaskWord* filter2 = maskPtr(game, condition.filter2);
        if (filter1 == nullptr || filter2 == nullptr) {
            continue;
        }
        if (condition.quantifier == 1) {
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                const MaskWord* cell = cellObjects(session, tile);
                if (!matchesFilter(filter1, game.wordCount, condition.aggr1, cell)) {
                    continue;
                }
                if (matchesFilter(filter2, game.wordCount, condition.aggr2, cell)) {
                    continue;
                }
                score += 10 + nearestMatchingDistance(session, tile, filter2, condition.aggr2);
            }
        } else if (condition.quantifier == 0) {
            bool passed = false;
            int32_t best = 64;
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                const MaskWord* cell = cellObjects(session, tile);
                if (!matchesFilter(filter1, game.wordCount, condition.aggr1, cell)) {
                    continue;
                }
                if (matchesFilter(filter2, game.wordCount, condition.aggr2, cell)) {
                    passed = true;
                    break;
                }
                best = std::min(best, nearestMatchingDistance(session, tile, filter2, condition.aggr2));
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
        int32_t best = 64;
        for (int32_t player = 0; player < tileCount; ++player) {
            if (!matchesFilter(playerMask, game.wordCount, game.playerMaskAggregate, cellObjects(session, player))) {
                continue;
            }
            hasPlayer = true;
            for (const auto& condition : game.winConditions) {
                const MaskWord* filter1 = maskPtr(game, condition.filter1);
                if (filter1 != nullptr) {
                    best = std::min(best, nearestMatchingDistance(session, player, filter1, condition.aggr1));
                }
            }
        }
        if (hasPlayer) {
            score += std::min(best, 16);
        }
    }

    return score;
}

} // namespace puzzlescript::search
