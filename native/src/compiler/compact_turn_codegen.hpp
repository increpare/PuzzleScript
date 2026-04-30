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

enum class CompactTurnBackendKind {
    Unsupported,
    NativeKernel,
    InterpreterBridge,
};

struct CompactTurnSupport {
    CompactTurnBackendKind backendKind = CompactTurnBackendKind::Unsupported;
    std::string statusReason = "native_compact_generator_rebuild";
    std::string nativeKernelStatusReason = "native_compact_generator_rebuild";

    bool supported() const {
        return backendKind != CompactTurnBackendKind::Unsupported;
    }

    bool nativeKernel() const {
        return backendKind == CompactTurnBackendKind::NativeKernel;
    }

    bool usesInterpreterBridge() const {
        return backendKind == CompactTurnBackendKind::InterpreterBridge;
    }
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
