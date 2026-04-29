#include "compiler/compact_turn_codegen.hpp"

#include "compiler/compiled_rules_codegen.hpp"

#include <ostream>

namespace puzzlescript::compiler {

void emitCompactTurnBackend(
    std::ostream& out,
    const Game& game,
    std::string_view sourcePath,
    uint64_t sourceHash,
    size_t sourceIndex,
    CompactCodegenOptions options
) {
    const CompactTurnSupport compactTurnSupport = compactTurnSupportForGame(game, options);
    out << "SpecializedCompactTurnOutcome specialized_compact_turn_source_" << sourceIndex << "(\n"
        << "    const Game& game,\n"
        << "    PersistentLevelState& levelState,\n"
        << "    Scratch& scratch,\n"
        << "    SpecializedCompactTurnContext context,\n"
        << "    ps_input input,\n"
        << "    RuntimeStepOptions options\n"
        << ") {\n";
    if (!compactTurnSupport.supported) {
        out << "    (void)game;\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    (void)context;\n"
            << "    (void)options;\n"
            << "    (void)input;\n"
            << "    return {false, {}};\n"
            << "}\n\n";
    } else if (compactTurnSupport.interpreterBridge) {
        out << "    return compactStateInterpretedTurnBridge(game, levelState, scratch, context, input, options);\n"
            << "}\n\n";
    } else {
        out << "    (void)game;\n"
            << "    (void)levelState;\n"
            << "    (void)scratch;\n"
            << "    (void)context;\n"
            << "    (void)options;\n"
            << "    (void)input;\n"
            << "    return {false, {}};\n"
            << "}\n\n";
    }
    out
        << "const SpecializedCompactTurnBackend specialized_compact_turn_backend_" << sourceIndex << " = {\n"
        << "    " << sourceHash << "ULL,\n"
        << "    " << cppStringLiteral(sourcePath) << ",\n"
        << "    specialized_compact_turn_source_" << sourceIndex << ",\n"
        << "    {" << (compactTurnSupport.supported ? "true" : "false")
        << ", " << cppStringLiteral(compactTurnSupport.fallbackReason) << "},\n"
        << "    " << (compactTurnSupport.supported && !compactTurnSupport.interpreterBridge ? "true" : "false") << ",\n"
        << "};\n\n";
}

CompactTurnSupport compactNativeTurnSupportForGame(const Game& game) {
    (void)game;
    CompactTurnSupport support;
    support.fallbackReason = "native_compact_generator_rebuild";
    support.nativeFallbackReason = support.fallbackReason;
    return support;
}

CompactTurnSupport compactTurnSupportForGame(const Game& game, const CompactCodegenOptions& options) {
    CompactTurnSupport support = compactNativeTurnSupportForGame(game);
    support.nativeFallbackReason = support.fallbackReason;
    if (options.interpreterMode && !support.supported) {
        support.supported = true;
        support.interpreterBridge = true;
        support.fallbackReason = "interpreter_bridge";
    }
    return support;
}

CompactTurnSupport compactTurnSupportForGame(const Game& game) {
    return compactTurnSupportForGame(game, CompactCodegenOptions{});
}

} // namespace puzzlescript::compiler
