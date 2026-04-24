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
// MaskWord is kept as int32_t in Phase 1 to match the existing IR layout; the
// Phase 2 plan switches it to uint64_t after the arena is in place.
using MaskWord = int32_t;

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

struct LevelTemplate {
    bool isMessage = false;
    std::string message;
    int32_t lineNumber = 0;
    int32_t width = 0;
    int32_t height = 0;
    int32_t layerCount = 0;
    std::vector<int32_t> objects;
};

struct RestartSnapshot {
    int32_t width = 0;
    int32_t height = 0;
    std::vector<int32_t> objects;
    std::vector<int32_t> oldFlickscreenDat;
};

struct PreparedSession {
    int32_t currentLevelIndex = 0;
    std::optional<int32_t> currentLevelTarget;
    bool titleScreen = false;
    bool textMode = false;
    int32_t titleMode = 0;
    int32_t titleSelection = 0;
    bool titleSelected = false;
    bool messageSelected = false;
    bool winning = false;
    std::string loadedLevelSeed;
    bool hasRandomState = false;
    bool randomStateValid = false;
    uint8_t randomStateI = 0;
    uint8_t randomStateJ = 0;
    std::vector<uint8_t> randomStateS;
    std::vector<int32_t> oldFlickscreenDat;
    LevelTemplate level;
    RestartSnapshot restart;
    std::string serializedLevel;
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

    // anyObjectsPresent is a variable-length list of masks of width
    // Game::wordCount. Each mask's offset is stored in
    // Game::anyObjectOffsets; this struct locates that run with
    // [anyObjectsFirst, anyObjectsFirst + anyObjectsCount).
    uint32_t anyObjectsFirst = 0;
    uint32_t anyObjectsCount = 0;

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

struct SoundMaskEntry {
    // object-width mask of width Game::wordCount
    MaskOffset objectMask = kNullMaskOffset;
    // movement-mask whose legacy JSON form may be narrower than
    // Game::movementWordCount; store the actual parsed width.
    MaskOffset directionMask = kNullMaskOffset;
    uint32_t directionMaskWidth = 0;
    int32_t seed = 0;
};

struct Game {
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
    std::vector<std::string> metadataPairs;
    std::map<std::string, std::string> metadataMap;
    std::map<std::string, int32_t> metadataLines;
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
    PreparedSession preparedSession;
};

struct Session {
    struct RandomState {
        std::array<uint8_t, 256> s{};
        uint8_t i = 0;
        uint8_t j = 0;
        bool valid = false;
    };

    struct UndoSnapshot {
        PreparedSession preparedSession;
        LevelTemplate liveLevel;
        std::vector<int32_t> liveMovements;
        std::vector<int32_t> rigidGroupIndexMasks;
        std::vector<int32_t> rigidMovementAppliedMasks;
        RandomState randomState;
    };

    std::shared_ptr<const Game> game;
    PreparedSession preparedSession;
    LevelTemplate liveLevel;
    std::vector<int32_t> liveMovements;
    std::vector<int32_t> rowMasks;
    std::vector<int32_t> columnMasks;
    std::vector<int32_t> boardMask;
    std::vector<int32_t> rowMovementMasks;
    std::vector<int32_t> columnMovementMasks;
    std::vector<int32_t> boardMovementMask;
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
    std::vector<int32_t> rigidGroupIndexMasks;
    std::vector<int32_t> rigidMovementAppliedMasks;
    std::vector<int32_t> pendingCreateMask;
    std::vector<int32_t> pendingDestroyMask;
    // Scratch buffers reused across applyReplacementAt invocations to avoid
    // per-call heap allocation. Contents are overwritten on every call.
    std::vector<int32_t> replacementObjectsClearScratch;
    std::vector<int32_t> replacementObjectsSetScratch;
    std::vector<int32_t> replacementMovementsClearScratch;
    std::vector<int32_t> replacementMovementsSetScratch;
    std::vector<int32_t> replacementObjectsScratch;
    std::vector<int32_t> replacementMovementsScratch;
    std::vector<int32_t> replacementOldObjectsScratch;
    std::vector<int32_t> replacementOldMovementsScratch;
    std::vector<int32_t> replacementCreatedScratch;
    std::vector<int32_t> replacementDestroyedScratch;
    std::vector<int32_t> replacementRigidMaskScratch;
    std::vector<UndoSnapshot> undoStack;
    std::vector<ps_audio_event> lastAudioEvents;
    bool canUndo = false;
    bool pendingAgain = false;
    RandomState randomState;
    SimdBackend backend = SimdBackend::Scalar;
};

struct CompileResult {
    std::shared_ptr<const Game> game;
    std::unique_ptr<Error> error;
};

std::unique_ptr<Error> loadGameFromJson(std::string_view jsonText, std::shared_ptr<const Game>& outGame);
std::unique_ptr<Session> createSession(std::shared_ptr<const Game> game);
std::unique_ptr<Session> createSessionWithLoadedLevelSeed(std::shared_ptr<const Game> game, std::string loadedLevelSeed);
std::unique_ptr<Error> loadLevel(Session& session, int32_t levelIndex);
bool restart(Session& session);
bool undo(Session& session);
uint64_t hashSession64(const Session& session);
ps_hash128 hashSession128(const Session& session);
std::string serializeTestString(const Session& session);
std::string exportSnapshot(const Session& session);
size_t listInputs(ps_input* output, size_t capacity);
ps_step_result step(Session& session, ps_input input);
ps_step_result tick(Session& session);
void settlePendingAgain(Session& session);
std::unique_ptr<Error> benchmarkCloneHash(const Session& session, uint32_t iterations, uint32_t threads, ps_benchmark_result& outResult);
void setRuntimeCountersEnabled(bool enabled);
void resetRuntimeCounters();
ps_runtime_counters snapshotRuntimeCounters();

} // namespace puzzlescript
