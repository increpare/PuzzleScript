#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "hash.hpp"
#include "json.hpp"
#include "puzzlescript/puzzlescript.h"
#include "simd.hpp"

namespace puzzlescript {

struct Error {
    explicit Error(std::string messageText)
        : message(std::move(messageText)) {}

    std::string message;
};

struct CompileResult;

using BitVector = std::vector<int32_t>;

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
    BitVector objectsPresent;
    BitVector objectsMissing;
    std::vector<BitVector> anyObjectsPresent;
    BitVector movementsPresent;
    BitVector movementsMissing;
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
    std::vector<BitVector> cellRowMasks;
    std::vector<BitVector> cellRowMasksMovements;
    BitVector ruleMask;
    std::vector<std::vector<Pattern>> patterns;
};

struct WinCondition {
    int32_t quantifier = 0;
    BitVector filter1;
    BitVector filter2;
    int32_t lineNumber = 0;
    bool aggr1 = false;
    bool aggr2 = false;
};

struct LoopPointTable {
    std::vector<std::optional<int32_t>> entries;
};

struct SoundMaskEntry {
    BitVector objectMask;
    BitVector directionMask;
    int32_t seed = 0;
};

struct Game {
    int32_t schemaVersion = 1;
    int32_t strideObject = 1;
    int32_t strideMovement = 1;
    uint32_t wordCount = 0;          // = strideObject; object-mask words per cell
    uint32_t movementWordCount = 0;  // = strideMovement; movement-mask words per cell
    std::vector<MaskWord> maskArena; // all per-game bitmasks concatenated
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
    std::map<std::string, BitVector> glyphDict;
    std::vector<ObjectDef> objectsById;
    std::vector<std::vector<std::string>> collisionLayers;
    std::vector<BitVector> layerMasks;
    std::map<std::string, BitVector> objectMasks;
    std::map<std::string, BitVector> aggregateMasks;
    bool playerMaskAggregate = false;
    BitVector playerMask;
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
    std::vector<int32_t> rigidGroupIndexMasks;
    std::vector<int32_t> rigidMovementAppliedMasks;
    std::vector<int32_t> pendingCreateMask;
    std::vector<int32_t> pendingDestroyMask;
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
std::unique_ptr<Error> benchmarkCloneHash(const Session& session, uint32_t iterations, uint32_t threads, ps_benchmark_result& outResult);

} // namespace puzzlescript
