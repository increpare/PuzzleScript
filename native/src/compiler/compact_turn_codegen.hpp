#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>

#include "runtime/core.hpp"

namespace puzzlescript::compiler {

struct CompactCodegenOptions {
    bool interpreterMode = false;
};

struct CompactTurnSupport {
    bool supported = false;
    std::string fallbackReason = "interpreter_delegation";
    bool interpreterBridge = false;
    std::string nativeFallbackReason = "interpreter_delegation";
};

CompactTurnSupport compactNativeTurnSupportForGame(const Game& game);
CompactTurnSupport compactTurnSupportForGame(const Game& game, const CompactCodegenOptions& options);
CompactTurnSupport compactTurnSupportForGame(const Game& game);

void emitCompactTurnBackend(
    std::ostream& out,
    const Game& game,
    std::string_view sourcePath,
    uint64_t sourceHash,
    size_t sourceIndex,
    CompactCodegenOptions options);

} // namespace puzzlescript::compiler
