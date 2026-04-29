#include "compiler/compact_turn_codegen.hpp"

#include "compiler/compiled_rules_codegen.hpp"

#include <ostream>

namespace puzzlescript::compiler {

namespace {

void emitCompactTurnUnsupportedBody(std::ostream& out) {
    out << "    (void)game;\n"
        << "    (void)levelState;\n"
        << "    (void)scratch;\n"
        << "    (void)context;\n"
        << "    (void)options;\n"
        << "    (void)input;\n"
        << "    return {false, {}};\n";
}

void emitCompactTurnCompilerSkeletonBody(std::ostream& out, std::string_view suffix) {
    out << "    (void)levelState;\n"
        << "    (void)scratch;\n"
        << "    (void)options;\n"
        << "    (void)input;\n"
        << "    ps_step_result result{};\n"
        << "    if (!compact_turn_prepare_state_" << suffix << "(dimensions, levelState, scratch)) {\n"
        << "        return {false, result};\n"
        << "    }\n"
        << "    // Semantic compact turn compiler skeleton:\n"
        << "    // 1. validate level dimensions and persistent board storage\n"
        << "    // 2. decode input direction\n"
        << "    // 3. seed input movements\n"
        << "    // 4. apply early rulegroups\n"
        << "    // 5. resolve movement\n"
        << "    // 6. apply late rulegroups\n"
        << "    // 7. process commands and again policy\n"
        << "    // 8. evaluate win conditions\n"
        << "    // 9. canonicalize and return result\n"
        << "    return {false, result};\n";
}

std::string sourceSuffix(size_t sourceIndex) {
    return std::to_string(sourceIndex);
}

void emitCompactTurnAccessLayer(std::ostream& out, const Game& game, size_t sourceIndex) {
    const std::string suffix = sourceSuffix(sourceIndex);
    out << "constexpr int32_t compact_turn_object_stride_" << suffix << " = " << game.strideObject << ";\n"
        << "constexpr int32_t compact_turn_movement_stride_" << suffix << " = " << game.strideMovement << ";\n"
        << "constexpr int32_t compact_turn_object_count_" << suffix << " = " << game.objectCount << ";\n"
        << "constexpr int32_t compact_turn_layer_count_" << suffix << " = " << game.layerCount << ";\n\n";

    out << "int32_t compact_turn_tile_count_" << suffix << "(LevelDimensions dimensions) {\n"
        << "    if (dimensions.width <= 0 || dimensions.height <= 0) return 0;\n"
        << "    return dimensions.width * dimensions.height;\n"
        << "}\n\n";

    out << "bool compact_turn_in_bounds_" << suffix << "(LevelDimensions dimensions, int32_t x, int32_t y) {\n"
        << "    return x >= 0 && y >= 0 && x < dimensions.width && y < dimensions.height;\n"
        << "}\n\n";

    out << "int32_t compact_turn_tile_index_" << suffix << "(LevelDimensions dimensions, int32_t x, int32_t y) {\n"
        << "    return y * dimensions.width + x;\n"
        << "}\n\n";

    out << "bool compact_turn_prepare_state_" << suffix << "(LevelDimensions dimensions, PersistentLevelState& levelState, Scratch& scratch) {\n"
        << "    const int32_t tileCount = compact_turn_tile_count_" << suffix << "(dimensions);\n"
        << "    if (tileCount <= 0) return false;\n"
        << "    const size_t objectWords = static_cast<size_t>(tileCount) * static_cast<size_t>(compact_turn_object_stride_" << suffix << ");\n"
        << "    if (levelState.board.objects.size() != objectWords) return false;\n"
        << "    const size_t movementWords = static_cast<size_t>(tileCount) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "    if (scratch.liveMovements.size() != movementWords) {\n"
        << "        scratch.liveMovements.assign(movementWords, 0);\n"
        << "    }\n"
        << "    return true;\n"
        << "}\n\n";

    out << "MaskWord* compact_turn_cell_objects_" << suffix << "(PersistentLevelState& levelState, int32_t tileIndex) {\n"
        << "    return levelState.board.objects.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_object_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "const MaskWord* compact_turn_cell_objects_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex) {\n"
        << "    return levelState.board.objects.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_object_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "MaskWord* compact_turn_cell_movements_" << suffix << "(Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.liveMovements.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "const MaskWord* compact_turn_cell_movements_" << suffix << "(const Scratch& scratch, int32_t tileIndex) {\n"
        << "    return scratch.liveMovements.data() + static_cast<size_t>(tileIndex) * static_cast<size_t>(compact_turn_movement_stride_" << suffix << ");\n"
        << "}\n\n";

    out << "bool compact_turn_cell_has_object_" << suffix << "(const PersistentLevelState& levelState, int32_t tileIndex, int32_t objectId) {\n"
        << "    if (objectId < 0 || objectId >= compact_turn_object_count_" << suffix << ") return false;\n"
        << "    const uint32_t bit = static_cast<uint32_t>(objectId);\n"
        << "    const uint32_t word = maskWordIndex(bit);\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_object_stride_" << suffix << ")) return false;\n"
        << "    return (compact_turn_cell_objects_" << suffix << "(levelState, tileIndex)[word] & maskBit(bit)) != 0;\n"
        << "}\n\n";

    out << "void compact_turn_set_cell_object_" << suffix << "(PersistentLevelState& levelState, int32_t tileIndex, int32_t objectId) {\n"
        << "    if (objectId < 0 || objectId >= compact_turn_object_count_" << suffix << ") return;\n"
        << "    const uint32_t bit = static_cast<uint32_t>(objectId);\n"
        << "    compact_turn_cell_objects_" << suffix << "(levelState, tileIndex)[maskWordIndex(bit)] |= maskBit(bit);\n"
        << "}\n\n";

    out << "void compact_turn_clear_cell_object_" << suffix << "(PersistentLevelState& levelState, int32_t tileIndex, int32_t objectId) {\n"
        << "    if (objectId < 0 || objectId >= compact_turn_object_count_" << suffix << ") return;\n"
        << "    const uint32_t bit = static_cast<uint32_t>(objectId);\n"
        << "    compact_turn_cell_objects_" << suffix << "(levelState, tileIndex)[maskWordIndex(bit)] &= ~maskBit(bit);\n"
        << "}\n\n";

    out << "int32_t compact_turn_layer_movement_" << suffix << "(const Scratch& scratch, int32_t tileIndex, int32_t layer) {\n"
        << "    if (layer < 0 || layer >= compact_turn_layer_count_" << suffix << ") return 0;\n"
        << "    const uint32_t layerIndex = static_cast<uint32_t>(layer);\n"
        << "    const uint32_t word = movementWordIndexForLayer(layerIndex);\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) return 0;\n"
        << "    const uint32_t shift = movementBitShiftForLayer(layerIndex);\n"
        << "    const MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(compact_turn_cell_movements_" << suffix << "(scratch, tileIndex)[word]);\n"
        << "    return static_cast<int32_t>((bits >> shift) & MaskWordUnsigned{0x1f});\n"
        << "}\n\n";

    out << "void compact_turn_set_layer_movement_" << suffix << "(Scratch& scratch, int32_t tileIndex, int32_t layer, int32_t directionMask) {\n"
        << "    if (layer < 0 || layer >= compact_turn_layer_count_" << suffix << ") return;\n"
        << "    const uint32_t layerIndex = static_cast<uint32_t>(layer);\n"
        << "    const uint32_t word = movementWordIndexForLayer(layerIndex);\n"
        << "    if (word >= static_cast<uint32_t>(compact_turn_movement_stride_" << suffix << ")) return;\n"
        << "    const uint32_t shift = movementBitShiftForLayer(layerIndex);\n"
        << "    MaskWord& cellWord = compact_turn_cell_movements_" << suffix << "(scratch, tileIndex)[word];\n"
        << "    const MaskWord mask = static_cast<MaskWord>(MaskWordUnsigned{0x1f} << shift);\n"
        << "    const MaskWord value = static_cast<MaskWord>((MaskWordUnsigned{static_cast<uint32_t>(directionMask) & 0x1fU}) << shift);\n"
        << "    cellWord = static_cast<MaskWord>((static_cast<MaskWordUnsigned>(cellWord) & ~static_cast<MaskWordUnsigned>(mask)) | static_cast<MaskWordUnsigned>(value));\n"
        << "}\n\n";

    out << "void compact_turn_clear_layer_movement_" << suffix << "(Scratch& scratch, int32_t tileIndex, int32_t layer) {\n"
        << "    compact_turn_set_layer_movement_" << suffix << "(scratch, tileIndex, layer, 0);\n"
        << "}\n\n";
}

} // namespace

void emitCompactTurnBackend(
    std::ostream& out,
    const Game& game,
    std::string_view sourcePath,
    uint64_t sourceHash,
    size_t sourceIndex,
    CompactCodegenOptions options
) {
    const CompactTurnSupport compactTurnSupport = compactTurnSupportForGame(game, options);
    const std::string suffix = sourceSuffix(sourceIndex);
    if (compactTurnSupport.supported && !compactTurnSupport.interpreterBridge) {
        emitCompactTurnAccessLayer(out, game, sourceIndex);
        out << "SpecializedCompactTurnOutcome specialized_compact_turn_core_" << sourceIndex << "(\n"
            << "    LevelDimensions dimensions,\n"
            << "    PersistentLevelState& levelState,\n"
            << "    Scratch& scratch,\n"
            << "    ps_input input,\n"
            << "    RuntimeStepOptions options\n"
            << ") {\n";
        emitCompactTurnCompilerSkeletonBody(out, suffix);
        out << "}\n\n";
    }
    out << "SpecializedCompactTurnOutcome specialized_compact_turn_source_" << sourceIndex << "(\n"
        << "    const Game& game,\n"
        << "    PersistentLevelState& levelState,\n"
        << "    Scratch& scratch,\n"
        << "    SpecializedCompactTurnContext context,\n"
        << "    ps_input input,\n"
        << "    RuntimeStepOptions options\n"
        << ") {\n";
    if (!compactTurnSupport.supported) {
        emitCompactTurnUnsupportedBody(out);
        out << "}\n\n";
    } else if (compactTurnSupport.interpreterBridge) {
        out << "    return compactStateInterpretedTurnBridge(game, levelState, scratch, context, input, options);\n"
            << "}\n\n";
    } else {
        out << "    (void)game;\n"
            << "    return specialized_compact_turn_core_" << sourceIndex << "(context.dimensions, levelState, scratch, input, options);\n"
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
    if (!options.interpreterMode) {
        support.supported = true;
        support.interpreterBridge = false;
        support.fallbackReason = "compiler_mode";
        support.nativeFallbackReason = "compiler_mode";
        return support;
    }
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
