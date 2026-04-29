#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/hash.hpp"
#include "runtime/json.hpp"
#include "puzzlescript/puzzlescript.h"
#include "runtime/simd.hpp"

namespace puzzlescript {

struct Error {
    explicit Error(std::string messageText)
        : message(std::move(messageText)) {}

    std::string message;
};

struct CompileResult;

// (The historical `BitVector` typedef was removed after Tasks 7-10 — every
// engine struct now stores masks as MaskOffsets into Game::maskArena. A few
// hot-path functions still pass std::vector<int32_t> by value/reference as
// local scratch; those are intentionally written out as the canonical type
// rather than a typedef so that ownership vs view distinctions stay visible.)

// ---- Mask representation (Phase 1) -----------------------------------------
// Every per-game bitmask (pattern masks, replacement masks, rule masks, layer
// masks, glyph masks, aggregate masks, etc.) is stored as a run of `wordCount`
// consecutive MaskWords inside Game::maskArena. Structs that owned a BitVector
// now store a uint32_t offset into the arena.
//
#ifndef PS_MASK_WORD_BITS
#define PS_MASK_WORD_BITS 64
#endif

#if PS_MASK_WORD_BITS == 32
using MaskWord = int32_t;
using MaskWordUnsigned = uint32_t;
#elif PS_MASK_WORD_BITS == 64
using MaskWord = int64_t;
using MaskWordUnsigned = uint64_t;
#else
#error "PS_MASK_WORD_BITS must be 32 or 64"
#endif

using MaskVector = std::vector<MaskWord>;

inline constexpr uint32_t kMaskWordBits = PS_MASK_WORD_BITS;
inline constexpr uint32_t kMaskWordShift = PS_MASK_WORD_BITS == 64 ? 6U : 5U;
inline constexpr uint32_t kMaskWordBitMask = PS_MASK_WORD_BITS - 1U;

inline constexpr uint32_t maskWordIndex(uint32_t bitIndex) {
    return bitIndex >> kMaskWordShift;
}

inline constexpr uint32_t maskBitIndex(uint32_t bitIndex) {
    return bitIndex & kMaskWordBitMask;
}

inline constexpr MaskWord maskBit(uint32_t bitIndex) {
    return static_cast<MaskWord>(MaskWordUnsigned{1} << maskBitIndex(bitIndex));
}

inline constexpr uint32_t kMovementLayersPerMaskWord = PS_MASK_WORD_BITS == 64 ? 10U : 5U;

inline constexpr uint32_t movementStrideWordCount(uint32_t layerCount) {
    return (layerCount + kMovementLayersPerMaskWord - 1U) / kMovementLayersPerMaskWord;
}

inline constexpr uint32_t movementWordIndexForLayer(uint32_t layerIndex) {
    return (layerIndex * 5U) >> kMaskWordShift;
}

inline constexpr uint32_t movementBitShiftForLayer(uint32_t layerIndex) {
    return (layerIndex * 5U) & kMaskWordBitMask;
}

inline int32_t maskWordCountTrailingZeros(MaskWordUnsigned bits) {
#if PS_MASK_WORD_BITS == 64
    return __builtin_ctzll(bits);
#else
    return __builtin_ctz(bits);
#endif
}

struct MaskRef { const MaskWord* data; };
struct MaskMut { MaskWord* data; };

// Offset into Game::maskArena (in words, not bytes). `kNullMaskOffset` means
// "no mask assigned" (used for fields that are optional or vary per pattern).
using MaskOffset = uint32_t;
inline constexpr MaskOffset kNullMaskOffset = static_cast<uint32_t>(-1);

struct ObjectDef {
    std::string name;
    int32_t id = -1;
    int32_t layer = -1;
    std::vector<std::string> colors;
    std::vector<std::vector<int32_t>> sprite;
};

// Cell-major object bitmask grid used by the interpreter. Persistent board
// storage lives in PersistentLevelState::board.occupancy; Scratch::interpreterBoard
// is materialized from or synced back to that compact board at runtime boundaries.
// **Engine rule:** change per-cell interpreter occupancy only through
// `setCellObjects` / `setCellObjectsFromWords` (and compiled-rule wrappers),
// not by mutating `objects` directly.
struct LevelTemplate {
    bool isMessage = false;
    std::string message;
    int32_t lineNumber = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t layerCount = 0;
    MaskVector objects;
};

struct RandomState {
    std::array<uint8_t, 256> s{};
    uint8_t i = 0;
    uint8_t j = 0;
    bool valid = false;
};

struct LevelDimensions {
    int32_t width = 0;
    int32_t height = 0;
};

struct RestartSnapshot {
    int32_t width = 0;
    int32_t height = 0;
    /// Object-major compact occupancy (same layout as `BoardOccupancy::objectBits`).
    std::vector<uint64_t> objectBits;
    std::vector<int32_t> oldFlickscreenDat;
};

struct UndoSnapshot;

struct MetaGameState {
    int32_t currentLevelIndex = 0;
    std::optional<int32_t> currentLevelTarget;
    bool titleScreen = false;
    bool textMode = false;
    int32_t titleMode = 0;
    int32_t titleSelection = 0;
    bool titleSelected = false;
    bool messageSelected = false;
    bool winning = false;
    std::string messageText;
    std::string loadedLevelSeed;
    bool hasRandomState = false;
    bool randomStateValid = false;
    uint8_t randomStateI = 0;
    uint8_t randomStateJ = 0;
    std::vector<uint8_t> randomStateS;
    std::vector<int32_t> oldFlickscreenDat;
    LevelTemplate level;
    LevelDimensions levelDimensions;
    RestartSnapshot restart;
    std::string serializedLevel;
    std::vector<UndoSnapshot> undoStack;
    bool pendingAgain = false;
    bool suppressRuleMessages = false;
};

struct Replacement {
    // All masks live in Game::maskArena; these are offsets (in words).
    // The "objects" / "movements" / "movementsLayerMask" fields have width
    // Game::wordCount / Game::movementWordCount respectively. The two
    // "random" masks may have a different width (legacy IR format), stored
    // explicitly below.
    MaskOffset objectsClear       = kNullMaskOffset;
    MaskOffset objectsSet         = kNullMaskOffset;
    MaskOffset movementsClear     = kNullMaskOffset;
    MaskOffset movementsSet       = kNullMaskOffset;
    MaskOffset movementsLayerMask = kNullMaskOffset;
    MaskOffset randomEntityMask   = kNullMaskOffset;
    MaskOffset randomDirMask      = kNullMaskOffset;
    uint32_t randomEntityMaskWidth = 0;
    uint32_t randomDirMaskWidth    = 0;
    bool hasMovementsLayerMask = false;
    bool hasRandomEntityMask   = false;
    bool hasRandomDirMask      = false;
    std::vector<int32_t> randomEntityChoices;
    std::vector<int32_t> randomDirLayers;
};

struct Pattern {
    enum class Kind {
        Ellipsis,
        CellPattern,
    };

    Kind kind = Kind::CellPattern;

    // Fixed-width masks live in Game::maskArena. `objects*` masks have width
    // Game::wordCount; `movements*` masks have width Game::movementWordCount.
    MaskOffset objectsPresent   = kNullMaskOffset;
    MaskOffset objectsMissing   = kNullMaskOffset;
    MaskOffset movementsPresent = kNullMaskOffset;
    MaskOffset movementsMissing = kNullMaskOffset;
    bool hasObjectsPresent = false;
    bool hasObjectsMissing = false;
    bool hasMovementsPresent = false;
    bool hasMovementsMissing = false;

    // anyObjectsPresent is a variable-length list of masks of width
    // Game::wordCount. Each mask's offset is stored in
    // Game::anyObjectOffsets; this struct locates that run with
    // [anyObjectsFirst, anyObjectsFirst + anyObjectsCount).
    uint32_t anyObjectsFirst = 0;
    uint32_t anyObjectsCount = 0;

    // Object ids from objectsPresent, precomputed once so anchored scans do
    // not repeatedly walk object-mask words to find possible anchors.
    std::vector<int32_t> objectAnchorIds;
    std::vector<std::vector<int32_t>> anyObjectAnchorIds;

    std::optional<Replacement> replacement;
};

struct RuleCommand {
    std::string name;
    std::optional<std::string> argument;
};

struct Rule {
    int32_t direction = 0;
    bool hasReplacements = false;
    int32_t lineNumber = 0;
    std::vector<int32_t> ellipsisCount;
    int32_t groupNumber = 0;
    bool rigid = false;
    std::vector<RuleCommand> commands;
    bool isRandom = false;

    // cellRowMasks is a per-row list of object-width masks stored as a run
    // of offsets inside Game::cellRowMaskOffsets[first .. first+count).
    uint32_t cellRowMasksFirst = 0;
    uint32_t cellRowMasksCount = 0;
    uint32_t cellRowMasksMovementsFirst = 0;
    uint32_t cellRowMasksMovementsCount = 0;

    MaskOffset ruleMask = kNullMaskOffset;
    MaskOffset ruleMovementMask = kNullMaskOffset;
    bool hasRuleMovementMask = false;

    std::vector<std::vector<Pattern>> patterns;
};

struct WinCondition {
    int32_t quantifier = 0;
    // object-width masks in Game::maskArena
    MaskOffset filter1 = kNullMaskOffset;
    MaskOffset filter2 = kNullMaskOffset;
    int32_t lineNumber = 0;
    bool aggr1 = false;
    bool aggr2 = false;
};

struct LoopPointTable {
    std::vector<std::optional<int32_t>> entries;
};

struct CommandState {
    std::vector<std::string> queue;
    std::string messageText;
    bool hasAgain = false;
    bool hasCancel = false;
    bool hasCheckpoint = false;
    bool hasMessage = false;
    bool hasRestart = false;
    bool hasWin = false;
};

struct SpecializedRulegroupsBackend;
struct SpecializedFullTurnBackend;
struct SpecializedCompactTurnBackend;

struct SoundMaskEntry {
    // object-width mask of width Game::wordCount
    MaskOffset objectMask = kNullMaskOffset;
    // movement-mask whose legacy JSON form may be narrower than
    // Game::movementWordCount; store the actual parsed width.
    MaskOffset directionMask = kNullMaskOffset;
    uint32_t directionMaskWidth = 0;
    int32_t seed = 0;
};

struct BoardOccupancy {
    std::vector<uint64_t> objectBits;
};

struct PersistentBoardState {
    // Object-major compact board. This is the persistent board shape used by
    // solver and compact turn paths; interpreter execution materializes the
    // legacy cell-major board in Scratch::interpreterBoard.
    BoardOccupancy occupancy;
};

struct PersistentLevelState {
    PersistentBoardState board;
    RandomState rng;
};

struct InterpreterBoardScratch {
    MaskVector objects;
};

struct GameMetadata {
    std::vector<std::string> pairs;
    std::map<std::string, std::string> values;
    std::map<std::string, int32_t> lines;
};

// Intended turn-core boundary:
//   takeTurn(
//       const LevelDimensions& dimensions,
//       PersistentLevelState& levelState,
//       Scratch& scratch,
//       ps_input input,
//       const RuntimeStepOptions& options) -> TurnResult
//
// Fully specialized code can compile dimensions and game-level constants into
// the generated function. Metagame/session context stays outside turn core.

struct GameInformation {
    int32_t schemaVersion = 1;
    int32_t strideObject = 1;
    int32_t strideMovement = 1;
    uint32_t wordCount = 0;          // = strideObject; object-mask words per cell
    uint32_t movementWordCount = 0;  // = strideMovement; movement-mask words per cell
    std::vector<MaskWord> maskArena; // all per-game bitmasks concatenated
    // Offsets into maskArena for each entry of Pattern::anyObjectsPresent
    // runs. Pattern locates its entries as
    // anyObjectOffsets[anyObjectsFirst .. anyObjectsFirst+anyObjectsCount).
    std::vector<MaskOffset> anyObjectOffsets;

    // Offsets into maskArena for the per-row object / movement masks of
    // Rule. Each referenced mask has width wordCount or movementWordCount.
    std::vector<MaskOffset> cellRowMaskOffsets;
    std::vector<MaskOffset> cellRowMaskMovementsOffsets;
    int32_t layerCount = 1;
    int32_t objectCount = 0;
    int32_t backgroundId = -1;
    int32_t backgroundLayer = -1;
    std::string foregroundColor;
    std::string backgroundColor;
    GameMetadata metadata;
    std::vector<std::string> idDict;
    std::vector<std::string> glyphOrder;

    // Name-keyed mask tables. Sorted by name at load time so lookups are
    // binary-search. Each entry's mask lives in maskArena at `offset`,
    // width = wordCount unless noted.
    struct NamedMaskEntry { std::string name; MaskOffset offset; };
    std::vector<NamedMaskEntry> glyphMaskTable;
    std::vector<NamedMaskEntry> objectMaskTable;
    std::vector<NamedMaskEntry> aggregateMaskTable;

    std::vector<ObjectDef> objectsById;
    std::vector<std::vector<std::string>> collisionLayers;

    // layerMaskOffsets[layer] is the arena offset of the object-width mask
    // for that collision layer.
    std::vector<MaskOffset> layerMaskOffsets;

    bool playerMaskAggregate = false;
    MaskOffset playerMask = kNullMaskOffset;  // object-width mask; null means no player
    bool rigid = false;
    std::vector<bool> rigidGroups;
    std::vector<int32_t> rigidGroupIndexToGroupIndex;
    std::vector<int32_t> groupIndexToRigidGroupIndex;
    std::vector<int32_t> groupNumberToRigidGroupIndex;
    std::vector<std::vector<Rule>> rules;
    std::vector<std::vector<Rule>> lateRules;
    LoopPointTable loopPoint;
    LoopPointTable lateLoopPoint;
    std::vector<WinCondition> winConditions;
    std::vector<LevelTemplate> levels;
    std::map<std::string, int32_t> sfxEvents;
    std::vector<SoundMaskEntry> sfxCreationMasks;
    std::vector<SoundMaskEntry> sfxDestructionMasks;
    std::vector<std::vector<SoundMaskEntry>> sfxMovementMasks;
    std::vector<SoundMaskEntry> sfxMovementFailureMasks;
    const SpecializedRulegroupsBackend* specializedRulegroups = nullptr;
    const SpecializedFullTurnBackend* specializedFullTurn = nullptr;
    const SpecializedCompactTurnBackend* specializedCompactTurn = nullptr;
};

using Game = GameInformation;

struct Scratch {
    // Legacy interpreter cell-major board. Per-tile writes go through
    // setCellObjects / setCellObjectsFromWords.
    InterpreterBoardScratch interpreterBoard;
    MaskVector liveMovements;
    MaskVector rowMasks;
    MaskVector columnMasks;
    MaskVector boardMask;
    MaskVector rowMovementMasks;
    MaskVector columnMovementMasks;
    MaskVector boardMovementMask;
    // Per-object cell presence bitsets for anchored rule scans. Layout is
    // object-major: objectCellBits[objectId * cellWordCount + word].
    std::vector<uint64_t> objectCellBits;
    std::vector<uint32_t> objectCellCounts;
    int32_t objectCellBitTileCount = 0;
    bool objectCellIndexDirty = true;
    // Incremental rebuildMasks tracking: `setCellObjects`/`setCellMovements`
    // OR new bits into the row/column/board masks directly. When bits are
    // *cleared* (old & ~new != 0) we cannot undo the OR without re-scanning
    // the row/column, so we mark that row/column dirty and `rebuildMasks`
    // rebuilds only the dirty ones. Sized to [height] / [width] in loadLevel.
    // A non-empty `dirtyObjectRows` etc. implies the corresponding board mask
    // is also stale (tracked via dirtyObjectBoard / dirtyMovementBoard).
    std::vector<uint8_t> dirtyObjectRows;
    std::vector<uint8_t> dirtyObjectColumns;
    std::vector<uint8_t> dirtyMovementRows;
    std::vector<uint8_t> dirtyMovementColumns;
    bool dirtyObjectBoard = true;
    bool dirtyMovementBoard = true;
    // Fast-path flag: if true, at least one row/col/board entry is dirty and
    // `rebuildMasks` must do work. Set whenever we mark something dirty,
    // cleared by `rebuildMasks` at the end of a clean rebuild.
    bool anyMasksDirty = true;
    MaskVector rigidGroupIndexMasks;
    MaskVector rigidMovementAppliedMasks;
    MaskVector pendingCreateMask;
    MaskVector pendingDestroyMask;
    MaskVector replacementObjectsClearScratch;
    MaskVector replacementObjectsSetScratch;
    MaskVector replacementMovementsClearScratch;
    MaskVector replacementMovementsSetScratch;
    MaskVector replacementObjectsScratch;
    MaskVector replacementMovementsScratch;
    MaskVector replacementOldObjectsScratch;
    MaskVector replacementOldMovementsScratch;
    MaskVector replacementCreatedScratch;
    MaskVector replacementDestroyedScratch;
    MaskVector replacementRigidMaskScratch;
    std::vector<int32_t> singleRowMatchScratch;
    std::vector<uint8_t> ellipsisLinePossibleScratch;
    std::vector<int32_t> ellipsisMinConcreteSuffixScratch;
    std::vector<int32_t> ellipsisPositionsScratch;
    SimdBackend backend = SimdBackend::Scalar;
};

struct UndoSnapshot {
    MetaGameState meta;
    LevelTemplate liveLevel;
    MaskVector liveMovements;
    MaskVector rigidGroupIndexMasks;
    MaskVector rigidMovementAppliedMasks;
    RandomState randomState;
};

struct LoadedGame {
    std::shared_ptr<const GameInformation> information;
    MetaGameState initialMetaGameState;
};

struct GameSession {
    std::shared_ptr<const GameInformation> game;
    MetaGameState meta;
    PersistentLevelState levelState;
    Scratch scratch;
};

using FullState = GameSession;

inline LevelDimensions currentLevelDimensions(const FullState& session) {
    return session.meta.levelDimensions;
}

inline int32_t currentLevelWidth(const FullState& session) {
    return session.meta.levelDimensions.width;
}

inline int32_t currentLevelHeight(const FullState& session) {
    return session.meta.levelDimensions.height;
}

struct TurnResult {
    ps_step_result core{};
    std::vector<ps_audio_event> audio;
    std::vector<ps_audio_event> uiAudio;
};

/// Cell-major → object-major compact bits (same layout as `BoardOccupancy::objectBits`).
void fillCompactOccupancyBitsFromInterpreterBoard(const FullState& session, std::vector<uint64_t>& objectBits);

void fillCompactOccupancyBitsFromInterpreterBoardData(
    const Game& game,
    int32_t width,
    int32_t height,
    const MaskVector& interpreterObjects,
    std::vector<uint64_t>& objectBits);

void fillInterpreterBoardObjectsFromCompactObjectBits(
    const Game& game,
    LevelDimensions dimensions,
    const std::vector<uint64_t>& objectBits,
    MaskVector& interpreterObjects);

void canonicalizeCompactObjectBits(
    const Game& game,
    int32_t width,
    int32_t height,
    uint64_t* objectBits,
    size_t objectBitWordCount);

/// Updates persistent compact board occupancy from the interpreter scratch board.
void syncPersistentBoardOccupancyFromScratch(FullState& session);

/// Updates persistent within-level state from scratch after interpreter execution.
void syncPersistentLevelStateFromScratch(FullState& session);

struct CompileResult {
    LoadedGame loadedGame;
    std::unique_ptr<Error> error;
};

std::unique_ptr<Error> loadGameFromJson(std::string_view jsonText, LoadedGame& outGame);
std::unique_ptr<FullState> createFullState(const LoadedGame& loadedGame);
std::unique_ptr<FullState> createFullStateWithLoadedLevelSeed(const LoadedGame& loadedGame, std::string loadedLevelSeed);
std::unique_ptr<FullState> createSession(const LoadedGame& loadedGame);
std::unique_ptr<FullState> createSessionWithLoadedLevelSeed(const LoadedGame& loadedGame, std::string loadedLevelSeed);
std::unique_ptr<Error> loadLevel(FullState& state, int32_t levelIndex);
std::unique_ptr<Error> advanceLevel(FullState& state);
bool restart(FullState& state);
bool undo(FullState& state);
uint64_t hashSession64(const FullState& state);
ps_hash128 hashSession128(const FullState& state);
uint64_t hashFullState64(const FullState& state);
ps_hash128 hashFullState128(const FullState& state);
std::string serializeTestString(const FullState& state);
std::string exportSnapshot(const FullState& state);
size_t listInputs(ps_input* output, size_t capacity);
ps_step_result step(FullState& state, ps_input input);
ps_step_result tick(FullState& state);
void settlePendingAgain(FullState& state);

enum class AgainPolicy {
    Yield,
    Drain,
};

struct TurnOptions {
    bool playableUndo = true;
    bool emitAudio = true;
    // Solver / analysis mode: suppress cosmetic outputs (message/sfx/etc.) and
    // ignore non-solver-relevant commands like checkpoint.
    bool solverMode = false;
    AgainPolicy againPolicy = AgainPolicy::Yield;
};

using RuntimeStepOptions = TurnOptions;

struct SpecializedRulegroupsForInterpretedTurnOutcome;
using SpecializedRulegroupsForInterpretedTurnFn = SpecializedRulegroupsForInterpretedTurnOutcome (*)(
    FullState& state,
    CommandState& commands,
    std::vector<bool>* bannedGroups
);

std::unique_ptr<Error> loadLevelTemplate(FullState& state, const LevelTemplate& levelTemplate, int32_t levelIndex, RuntimeStepOptions options);
bool restart(FullState& state, RuntimeStepOptions options);

// Transitional wrappers retained while callers migrate to interpretedTurn/turn.
ps_step_result interpreterStep(FullState& state, ps_input input, RuntimeStepOptions options);
ps_step_result interpreterTick(FullState& state, RuntimeStepOptions options);
ps_step_result interpretedTurnWithSpecializedRulegroups(
    FullState& state,
    ps_input input,
    RuntimeStepOptions options,
    SpecializedRulegroupsForInterpretedTurnFn applyEarlyRules,
    SpecializedRulegroupsForInterpretedTurnFn applyLateRules
);
ps_step_result interpretedStepWithSpecializedRulegroups(
    FullState& state,
    ps_input input,
    RuntimeStepOptions options,
    SpecializedRulegroupsForInterpretedTurnFn applyEarlyRules,
    SpecializedRulegroupsForInterpretedTurnFn applyLateRules
);
ps_step_result interpretedTickWithSpecializedRulegroups(
    FullState& state,
    RuntimeStepOptions options,
    SpecializedRulegroupsForInterpretedTurnFn applyEarlyRules,
    SpecializedRulegroupsForInterpretedTurnFn applyLateRules
);

// Compatibility wrappers for generated code emitted before the rulegroup
// terminology rename.
ps_step_result interpretedTurnWithCompiledRuleGroups(
    FullState& state,
    ps_input input,
    RuntimeStepOptions options,
    SpecializedRulegroupsForInterpretedTurnFn applyEarlyRules,
    SpecializedRulegroupsForInterpretedTurnFn applyLateRules
);
ps_step_result interpreterStepWithCompiledRuleGroups(
    FullState& state,
    ps_input input,
    RuntimeStepOptions options,
    SpecializedRulegroupsForInterpretedTurnFn applyEarlyRules,
    SpecializedRulegroupsForInterpretedTurnFn applyLateRules
);
ps_step_result interpreterTickWithCompiledRuleGroups(
    FullState& state,
    RuntimeStepOptions options,
    SpecializedRulegroupsForInterpretedTurnFn applyEarlyRules,
    SpecializedRulegroupsForInterpretedTurnFn applyLateRules
);
ps_step_result interpretedTurn(FullState& state, ps_input input, RuntimeStepOptions options);

// Public/runtime compatibility wrappers. New internal code should prefer turn().
ps_step_result step(FullState& state, ps_input input, RuntimeStepOptions options);
ps_step_result turn(FullState& state, ps_input input, RuntimeStepOptions options);
ps_step_result tick(FullState& state, RuntimeStepOptions options);
void settlePendingAgain(FullState& state, RuntimeStepOptions options);

TurnResult turnResult(FullState& state, ps_input input, RuntimeStepOptions options);
std::unique_ptr<Error> benchmarkCloneHash(const FullState& state, uint32_t iterations, uint32_t threads, ps_benchmark_result& outResult);
enum class RuntimeCounterId {
    CompactTurnNativeCalls,
    CompactTurnBridgeCalls,
    CompactTurnSetupNs,
    CompactTurnEarlyRulesNs,
    CompactTurnMovementNs,
    CompactTurnLateRulesNs,
    CompactTurnWinNs,
    CompactTurnCanonicalizeNs,
    CompactTurnBridgeCreateNs,
    CompactTurnBridgeMaterializeNs,
    CompactTurnBridgeTurnNs,
    CompactTurnBridgeCopybackNs,
};
void setRuntimeCountersEnabled(bool enabled);
bool runtimeCountersEnabled();
uint64_t runtimeCounterNowNs();
void addRuntimeCounter(RuntimeCounterId id, uint64_t amount = 1);
void resetRuntimeCounters();
ps_runtime_counters snapshotRuntimeCounters();

} // namespace puzzlescript
