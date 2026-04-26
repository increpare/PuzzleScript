#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "compiler/diagnostic.hpp"
#include "compiler/lower_to_runtime.hpp"
#include "compiler/parser.hpp"
#include "puzzlescript/puzzlescript.h"
#include "runtime/compiled_rules.hpp"
#include "runtime/core.hpp"
#include "search/search_common.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using puzzlescript::Game;
using puzzlescript::MaskWordUnsigned;
using puzzlescript::Session;
using puzzlescript::kMaskWordBits;
using StateKey = puzzlescript::search::StateKey;
using StateKeyHash = puzzlescript::search::StateKeyHash;
using SearchMode = puzzlescript::search::SearchMode;
using puzzlescript::search::priorityFor;

enum class Strategy {
    Portfolio,
    Bfs,
    WeightedAStar,
    Greedy,
};

enum class TimingMode {
    None,
    Summary,
    Detailed,
};

struct ScopedTimer {
    explicit ScopedTimer(int64_t& target)
        : target(target), start(Clock::now()) {}

    ~ScopedTimer() {
        target += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    }

    int64_t& target;
    TimePoint start;
};

struct Options {
    std::filesystem::path corpusPath;
    std::filesystem::path solutionsDir = "build/solver-solutions/native";
    int64_t timeoutMs = 5000;
    size_t progressEvery = 25;
    size_t jobs = 0;
    Strategy strategy = Strategy::Portfolio;
    TimingMode timingMode = TimingMode::Summary;
    std::optional<std::string> gameFilter;
    std::optional<int32_t> levelFilter;
    bool writeSolutions = true;
    bool progressPerGame = false;
    bool json = false;
    bool quiet = false;
    bool summaryOnly = false;
    bool profileRuntimeCounters = false;
    bool requireCompiledTick = false;
    bool exactStateKeys = true;
    bool compactNodeStorage = false;
    bool compactTickOracle = false;
    int32_t astarWeight = 2;
};

struct CompactSolverState {
    std::vector<uint64_t> objectBits;
    std::array<uint8_t, 256> randomStateS{};
    uint8_t randomStateI = 0;
    uint8_t randomStateJ = 0;
    bool randomStateValid = false;

    bool operator==(const CompactSolverState& other) const {
        return objectBits == other.objectBits
            && randomStateS == other.randomStateS
            && randomStateI == other.randomStateI
            && randomStateJ == other.randomStateJ
            && randomStateValid == other.randomStateValid;
    }

    size_t byteSize() const {
        return objectBits.size() * sizeof(uint64_t)
            + randomStateS.size() * sizeof(uint8_t)
            + sizeof(randomStateI)
            + sizeof(randomStateJ)
            + sizeof(randomStateValid);
    }
};

struct Node {
    std::unique_ptr<Session> session;
    CompactSolverState compact;
    StateKey key;
    int32_t parent = -1;
    ps_input input = PS_INPUT_UP;
    uint32_t depth = 0;
    int32_t heuristic = 0;
};

struct QueueEntry {
    int32_t priority = 0;
    uint64_t tie = 0;
    uint32_t nodeIndex = 0;
};

struct QueueEntryGreater {
    bool operator()(const QueueEntry& a, const QueueEntry& b) const {
        if (a.priority != b.priority) {
            return a.priority > b.priority;
        }
        return a.tie > b.tie;
    }
};

struct Timing {
    int64_t compileNs = 0;
    int64_t loadNs = 0;
    int64_t cloneNs = 0;
    int64_t stepNs = 0;
    int64_t hashNs = 0;
    int64_t queueNs = 0;
    int64_t frontierPopNs = 0;
    int64_t frontierPushNs = 0;
    int64_t visitedLookupNs = 0;
    int64_t visitedInsertNs = 0;
    int64_t nodeStoreNs = 0;
    int64_t heuristicNs = 0;
    int64_t solvedCheckNs = 0;
    int64_t timeoutCheckNs = 0;
    int64_t reconstructNs = 0;
    uint64_t visitedLookupProbes = 0;
    uint64_t visitedInsertProbes = 0;
    uint64_t visitedGrows = 0;
    uint64_t visitedCapacity = 0;
    uint64_t visitedMaxProbe = 0;
    uint64_t visitedKeyCollisions = 0;
    uint64_t compactStateBytes = 0;
    uint64_t compactMaxStateBytes = 0;
};

struct Result {
    std::string game;
    int32_t level = -1;
    std::string status;
    std::string error;
    std::string strategy = "bfs";
    std::string heuristic = "none";
    std::vector<std::string> solution;
    int64_t elapsedMs = 0;
    uint64_t expanded = 0;
    uint64_t generated = 0;
    uint64_t uniqueStates = 0;
    uint64_t duplicates = 0;
    uint64_t maxFrontier = 0;
    int64_t timeoutMs = 0;
    uint32_t workerId = 0;
    bool compiledRulesAttached = false;
    bool compiledTickAttached = false;
    bool compiledCompactTickAttached = false;
    bool compactNodeStorage = false;
    int32_t astarWeight = 2;
    uint64_t compactTickAttempts = 0;
    uint64_t compactTickHits = 0;
    uint64_t compactTickFallbacks = 0;
    uint64_t compactTickUnsupported = 0;
    uint64_t compactTickOracleChecks = 0;
    uint64_t compactTickOracleFailures = 0;
    Timing timing;
};

struct HumanSummary {
    uint64_t solved = 0;
    uint64_t timeout = 0;
    uint64_t exhausted = 0;
    uint64_t skipped = 0;
    uint64_t errors = 0;
    uint64_t expanded = 0;
    uint64_t generated = 0;

    uint64_t playableLevels() const {
        return solved + timeout + exhausted + errors;
    }
};

struct SourceLevel {
    int32_t level = -1;
    size_t insertBeforeLine = 0;
    bool message = false;
};

struct CompiledGame {
    std::filesystem::path path;
    std::string name;
    std::string source;
    std::shared_ptr<const Game> game;
    int64_t compileNs = 0;
    std::optional<Result> compileError;
    size_t resultBegin = 0;
    size_t resultEnd = 0;
};

struct WorkItem {
    size_t gameIndex = 0;
    int32_t levelIndex = 0;
    size_t resultIndex = 0;
};

std::string readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void writeFile(const std::filesystem::path& path, const std::string& text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to write file: " + path.string());
    }
    stream << text;
}

std::string jsonString(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (ch < 0x20) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out << "\\u00" << kHex[ch >> 4] << kHex[ch & 0x0f];
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    out << '"';
    return out.str();
}

std::string trim(std::string_view value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool isDividerLine(const std::string& line) {
    const std::string stripped = trim(line);
    return !stripped.empty() && std::all_of(stripped.begin(), stripped.end(), [](char ch) {
        return ch == '=';
    });
}

bool isCommentLine(const std::string& line) {
    const std::string stripped = trim(line);
    return !stripped.empty() && stripped.front() == '(';
}

double ms(int64_t ns) {
    return static_cast<double>(ns) / 1000000.0;
}

int64_t measuredSearchNs(const Timing& timing) {
    return timing.loadNs
        + timing.cloneNs
        + timing.stepNs
        + timing.hashNs
        + timing.queueNs
        + timing.frontierPopNs
        + timing.frontierPushNs
        + timing.visitedLookupNs
        + timing.visitedInsertNs
        + timing.nodeStoreNs
        + timing.heuristicNs
        + timing.solvedCheckNs
        + timing.timeoutCheckNs
        + timing.reconstructNs;
}

double unattributedMs(const Result& result) {
    const int64_t elapsedNs = result.elapsedMs * 1000000;
    return ms(std::max<int64_t>(0, elapsedNs - measuredSearchNs(result.timing)));
}

std::string secondsString(int64_t elapsedMs) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << (static_cast<double>(elapsedMs) / 1000.0);
    return out.str();
}

std::string inputName(ps_input input) {
    switch (input) {
        case PS_INPUT_UP: return "up";
        case PS_INPUT_LEFT: return "left";
        case PS_INPUT_DOWN: return "down";
        case PS_INPUT_RIGHT: return "right";
        case PS_INPUT_ACTION: return "action";
        case PS_INPUT_TICK: return "tick";
    }
    return "unknown";
}

std::vector<ps_input> solverInputsForGame(const Game& game) {
    std::vector<ps_input> inputs{
        PS_INPUT_RIGHT,
        PS_INPUT_UP,
        PS_INPUT_DOWN,
        PS_INPUT_LEFT,
    };
    if (game.metadataMap.find("noaction") == game.metadataMap.end()) {
        inputs.push_back(PS_INPUT_ACTION);
    }
    return inputs;
}

std::string strategyName(Strategy strategy) {
    switch (strategy) {
        case Strategy::Portfolio: return "portfolio";
        case Strategy::Bfs: return "bfs";
        case Strategy::WeightedAStar: return "weighted-astar";
        case Strategy::Greedy: return "greedy";
    }
    return "unknown";
}

std::string searchModeName(SearchMode mode) {
    switch (mode) {
        case SearchMode::Bfs: return "bfs";
        case SearchMode::WeightedAStar: return "weighted-astar";
        case SearchMode::Greedy: return "greedy";
    }
    return "unknown";
}

Strategy parseStrategy(const std::string& value) {
    if (value == "portfolio") {
        return Strategy::Portfolio;
    }
    if (value == "bfs") {
        return Strategy::Bfs;
    }
    if (value == "weighted-astar") {
        return Strategy::WeightedAStar;
    }
    if (value == "greedy") {
        return Strategy::Greedy;
    }
    throw std::runtime_error("Unsupported strategy: " + value);
}

TimingMode parseTimingMode(const std::string& value) {
    if (value == "none") {
        return TimingMode::None;
    }
    if (value == "summary") {
        return TimingMode::Summary;
    }
    if (value == "detailed") {
        return TimingMode::Detailed;
    }
    throw std::runtime_error("Unsupported timing mode: " + value);
}

size_t autoJobCount() {
    const unsigned count = std::thread::hardware_concurrency();
    return std::max<size_t>(1, count == 0 ? 1 : count);
}

bool isHiddenPath(const std::filesystem::path& path) {
    for (const auto& part : path) {
        const std::string name = part.string();
        if (!name.empty() && name[0] == '.') {
            return true;
        }
    }
    return false;
}

std::vector<std::filesystem::path> discoverGames(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> games;
    if (std::filesystem::is_regular_file(root)) {
        games.push_back(root);
        return games;
    }
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto rel = std::filesystem::relative(entry.path(), root);
        if (isHiddenPath(rel)) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (ext == ".txt") {
            games.push_back(entry.path());
        }
    }
    std::sort(games.begin(), games.end());
    return games;
}

bool matchesGameFilter(const std::string& relativeName, const std::optional<std::string>& filter) {
    if (!filter) {
        return true;
    }
    const std::string loweredName = lowercase(relativeName);
    const std::string loweredFilter = lowercase(*filter);
    return loweredName == loweredFilter || lowercase(std::filesystem::path(relativeName).filename().generic_string()) == loweredFilter;
}

Options parseArgs(int argc, char** argv) {
    Options options;
    options.jobs = 1;
    constexpr const char* usage = "Usage: puzzlescript_solver <solver_tests_dir> [--timeout-ms N] [--jobs auto|N|1] [--strategy portfolio|bfs|weighted-astar|greedy] [--timing none|summary|detailed] [--game NAME] [--level N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json] [--profile-runtime-counters] [--require-compiled-tick] [--hash-state-keys] [--compact-node-storage] [--compact-tick-oracle] [--astar-weight N]";
    if (argc < 2) {
        throw std::runtime_error(usage);
    }
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            throw std::runtime_error(usage);
        }
        if (arg == "--timeout-ms" && index + 1 < argc) {
            options.timeoutMs = std::max<int64_t>(1, std::stoll(argv[++index]));
            continue;
        }
        if (arg == "--jobs" && index + 1 < argc) {
            const std::string value = argv[++index];
            options.jobs = value == "auto" ? autoJobCount() : std::max<size_t>(1, std::stoull(value));
            continue;
        }
        if (arg == "--strategy" && index + 1 < argc) {
            options.strategy = parseStrategy(argv[++index]);
            continue;
        }
        if (arg == "--timing" && index + 1 < argc) {
            options.timingMode = parseTimingMode(argv[++index]);
            continue;
        }
        if (arg == "--game" && index + 1 < argc) {
            options.gameFilter = argv[++index];
            continue;
        }
        if (arg == "--level" && index + 1 < argc) {
            options.levelFilter = static_cast<int32_t>(std::stoi(argv[++index]));
            continue;
        }
        if (arg == "--solutions-dir" && index + 1 < argc) {
            options.solutionsDir = argv[++index];
            options.writeSolutions = true;
            continue;
        }
        if (arg == "--no-solutions") {
            options.writeSolutions = false;
            continue;
        }
        if (arg == "--json") {
            options.json = true;
            continue;
        }
        if (arg == "--summary-only") {
            options.summaryOnly = true;
            continue;
        }
        if (arg == "--profile-runtime-counters") {
            options.profileRuntimeCounters = true;
            continue;
        }
        if (arg == "--require-compiled-tick") {
            options.requireCompiledTick = true;
            options.profileRuntimeCounters = true;
            continue;
        }
        if (arg == "--hash-state-keys") {
            options.exactStateKeys = false;
            continue;
        }
        if (arg == "--compact-node-storage") {
            options.compactNodeStorage = true;
            continue;
        }
        if (arg == "--compact-tick-oracle") {
            options.compactNodeStorage = true;
            options.compactTickOracle = true;
            continue;
        }
        if (arg == "--astar-weight" && index + 1 < argc) {
            options.astarWeight = std::max<int32_t>(1, std::stoi(argv[++index]));
            continue;
        }
        if (arg == "--quiet") {
            options.quiet = true;
            options.progressEvery = 0;
            continue;
        }
        if (arg == "--progress-every" && index + 1 < argc) {
            options.progressEvery = static_cast<size_t>(std::stoull(argv[++index]));
            continue;
        }
        if (arg == "--progress-per-game") {
            options.progressPerGame = true;
            continue;
        }
        if (options.corpusPath.empty()) {
            options.corpusPath = arg;
            continue;
        }
        throw std::runtime_error("Unsupported argument: " + arg);
    }
    if (options.corpusPath.empty()) {
        throw std::runtime_error("Missing solver test path");
    }
    return options;
}

uint32_t compactWordTrailingZeros(MaskWordUnsigned value) {
    if constexpr (sizeof(MaskWordUnsigned) <= sizeof(unsigned int)) {
        return static_cast<uint32_t>(__builtin_ctz(static_cast<unsigned int>(value)));
    } else {
        return static_cast<uint32_t>(__builtin_ctzll(static_cast<unsigned long long>(value)));
    }
}

CompactSolverState compactStateFromSession(const Session& session) {
    CompactSolverState state;
    state.randomStateS = session.randomState.s;
    state.randomStateI = session.randomState.i;
    state.randomStateJ = session.randomState.j;
    state.randomStateValid = session.randomState.valid;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    const int32_t objectCount = session.game ? session.game->objectCount : 0;
    state.objectBits.assign(static_cast<size_t>(std::max(objectCount, 0)) * cellWordCount, 0);
    if (objectCount > 0 && tileCount > 0 && cellWordCount > 0) {
        const int32_t stride = session.game->strideObject;
        for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            const size_t sourceBase = static_cast<size_t>(tileIndex * stride);
            const size_t bitWord = static_cast<size_t>(tileIndex >> 6);
            const uint64_t bitMask = uint64_t{1} << static_cast<uint32_t>(tileIndex & 63);
            for (int32_t word = 0; word < stride; ++word) {
                MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(session.liveLevel.objects[sourceBase + static_cast<size_t>(word)]);
                while (bits != 0) {
                    const uint32_t bit = compactWordTrailingZeros(bits);
                    const int32_t objectId = word * static_cast<int32_t>(kMaskWordBits) + static_cast<int32_t>(bit);
                    if (objectId < objectCount) {
                        state.objectBits[static_cast<size_t>(objectId) * cellWordCount + bitWord] |= bitMask;
                    }
                    bits &= bits - 1;
                }
            }
        }
    }
    return state;
}

StateKey compactStateKey(const CompactSolverState& state, Timing& timing) {
    ScopedTimer timer(timing.hashNs);
    StateKey key{1469598103934665603ull, 7809847782465536322ull};
    for (uint64_t word : state.objectBits) {
        puzzlescript::search::appendStateKeyValue(key, word);
    }
    for (uint8_t byte : state.randomStateS) {
        puzzlescript::search::appendStateKeyValue(key, byte);
    }
    puzzlescript::search::appendStateKeyValue(key, state.randomStateI);
    puzzlescript::search::appendStateKeyValue(key, state.randomStateJ);
    puzzlescript::search::appendStateKeyValue(key, state.randomStateValid);
    return key;
}

CompactSolverState compactStateWithTiming(const Session& session, Timing& timing) {
    ScopedTimer timer(timing.hashNs);
    return compactStateFromSession(session);
}

void markMaterializedSessionDirty(Session& session) {
    std::fill(session.dirtyObjectRows.begin(), session.dirtyObjectRows.end(), 1);
    std::fill(session.dirtyObjectColumns.begin(), session.dirtyObjectColumns.end(), 1);
    std::fill(session.dirtyMovementRows.begin(), session.dirtyMovementRows.end(), 1);
    std::fill(session.dirtyMovementColumns.begin(), session.dirtyMovementColumns.end(), 1);
    session.dirtyObjectBoard = true;
    session.dirtyMovementBoard = true;
    session.objectCellIndexDirty = true;
    session.anyMasksDirty = true;
}

void materializeCompactStateIntoSession(const CompactSolverState& state, const Session& base, Session& session) {
    session.game = base.game;
    session.preparedSession.currentLevelIndex = base.preparedSession.currentLevelIndex;
    session.preparedSession.currentLevelTarget = base.preparedSession.currentLevelTarget;
    session.preparedSession.titleScreen = base.preparedSession.titleScreen;
    session.preparedSession.textMode = base.preparedSession.textMode;
    session.preparedSession.titleMode = base.preparedSession.titleMode;
    session.preparedSession.titleSelection = base.preparedSession.titleSelection;
    session.preparedSession.titleSelected = base.preparedSession.titleSelected;
    session.preparedSession.messageSelected = base.preparedSession.messageSelected;
    session.preparedSession.winning = base.preparedSession.winning;
    session.preparedSession.messageText = base.preparedSession.messageText;
    session.preparedSession.loadedLevelSeed = base.preparedSession.loadedLevelSeed;
    session.liveLevel.isMessage = base.liveLevel.isMessage;
    session.liveLevel.message = base.liveLevel.message;
    session.liveLevel.lineNumber = base.liveLevel.lineNumber;
    session.liveLevel.width = base.liveLevel.width;
    session.liveLevel.height = base.liveLevel.height;
    session.liveLevel.layerCount = base.liveLevel.layerCount;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    const int32_t objectCount = session.game ? session.game->objectCount : 0;
    const int32_t stride = session.game ? session.game->strideObject : 0;
    session.liveLevel.objects.assign(static_cast<size_t>(std::max(tileCount, 0) * std::max(stride, 0)), 0);
    for (int32_t objectId = 0; objectId < objectCount; ++objectId) {
        const size_t objectBase = static_cast<size_t>(objectId) * cellWordCount;
        for (size_t bitWord = 0; bitWord < cellWordCount; ++bitWord) {
            uint64_t bits = objectBase + bitWord < state.objectBits.size() ? state.objectBits[objectBase + bitWord] : 0;
            while (bits != 0) {
                const uint32_t bit = static_cast<uint32_t>(__builtin_ctzll(bits));
                const int32_t tileIndex = static_cast<int32_t>(bitWord * 64 + bit);
                if (tileIndex < tileCount) {
                    const int32_t word = objectId / static_cast<int32_t>(kMaskWordBits);
                    const uint32_t objectBit = static_cast<uint32_t>(objectId % static_cast<int32_t>(kMaskWordBits));
                    session.liveLevel.objects[static_cast<size_t>(tileIndex * stride + word)] |= puzzlescript::maskBit(objectBit);
                }
                bits &= bits - 1;
            }
        }
    }
    const size_t movementWordCount = static_cast<size_t>(std::max(tileCount, 0) * (session.game ? session.game->strideMovement : 0));
    session.liveMovements.assign(movementWordCount, 0);
    session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
    session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
    session.pendingAgain = false;
    session.canUndo = false;
    session.undoStack.clear();
    session.lastAudioEvents.clear();
    session.lastUiAudioEvents.clear();
    session.randomState.s = state.randomStateS;
    session.randomState.i = state.randomStateI;
    session.randomState.j = state.randomStateJ;
    session.randomState.valid = state.randomStateValid;
    markMaterializedSessionDirty(session);
}

void prepareSolverChildSessionFromParent(Session& child, const Session& parent) {
    child.game = parent.game;
    child.preparedSession = parent.preparedSession;
    child.liveLevel.isMessage = parent.liveLevel.isMessage;
    child.liveLevel.message = parent.liveLevel.message;
    child.liveLevel.lineNumber = parent.liveLevel.lineNumber;
    child.liveLevel.width = parent.liveLevel.width;
    child.liveLevel.height = parent.liveLevel.height;
    child.liveLevel.layerCount = parent.liveLevel.layerCount;
    child.liveLevel.objects = parent.liveLevel.objects;

    child.liveMovements.assign(parent.liveMovements.size(), 0);
    child.rigidGroupIndexMasks.assign(parent.rigidGroupIndexMasks.size(), 0);
    child.rigidMovementAppliedMasks.assign(parent.rigidMovementAppliedMasks.size(), 0);
    child.pendingCreateMask.clear();
    child.pendingDestroyMask.clear();
    child.pendingAgain = false;
    child.canUndo = false;
    child.undoStack.clear();
    child.lastAudioEvents.clear();
    child.lastUiAudioEvents.clear();
    child.suppressRuleMessages = parent.suppressRuleMessages;
    child.randomState = parent.randomState;
    child.backend = parent.backend;
    markMaterializedSessionDirty(child);
}

void recordCompactStateStorage(Timing& timing, const CompactSolverState& state) {
    const uint64_t bytes = static_cast<uint64_t>(state.byteSize());
    timing.compactStateBytes += bytes;
    timing.compactMaxStateBytes = std::max(timing.compactMaxStateBytes, bytes);
}

struct CompactTickTryResult {
    bool attempted = false;
    bool handled = false;
    CompactSolverState compact;
    ps_step_result stepResult{};
};

struct SolverEdgeStep {
    std::unique_ptr<Session> ownedChild;
    Session* child = nullptr;
    CompactTickTryResult compactTick;
    ps_step_result stepResult{};
    bool oracleMismatch = false;
    std::string oracleError;
};

bool equivalentSolverStepResult(const ps_step_result& lhs, const ps_step_result& rhs) {
    const bool terminal = lhs.won || rhs.won || lhs.restarted || rhs.restarted || lhs.transitioned || rhs.transitioned;
    return lhs.changed == rhs.changed
        && lhs.won == rhs.won
        && lhs.restarted == rhs.restarted
        && (terminal || lhs.transitioned == rhs.transitioned);
}

std::string stepResultSummary(const ps_step_result& result) {
    std::ostringstream out;
    out << "{changed=" << (result.changed ? "true" : "false")
        << ",won=" << (result.won ? "true" : "false")
        << ",transitioned=" << (result.transitioned ? "true" : "false")
        << ",restarted=" << (result.restarted ? "true" : "false")
        << "}";
    return out.str();
}

std::string compactStateDiffSummary(const CompactSolverState& lhs, const CompactSolverState& rhs) {
    const size_t wordCount = std::max(lhs.objectBits.size(), rhs.objectBits.size());
    for (size_t index = 0; index < wordCount; ++index) {
        const uint64_t left = index < lhs.objectBits.size() ? lhs.objectBits[index] : 0;
        const uint64_t right = index < rhs.objectBits.size() ? rhs.objectBits[index] : 0;
        if (left != right) {
            std::ostringstream out;
            out << " word=" << index << " compact=" << left << " interpreter=" << right;
            return out.str();
        }
    }
    if (lhs.randomStateValid != rhs.randomStateValid
        || lhs.randomStateI != rhs.randomStateI
        || lhs.randomStateJ != rhs.randomStateJ
        || lhs.randomStateS != rhs.randomStateS) {
        std::ostringstream out;
        out << " random compact_valid=" << lhs.randomStateValid
            << " interpreter_valid=" << rhs.randomStateValid
            << " compact_i=" << static_cast<int32_t>(lhs.randomStateI)
            << " interpreter_i=" << static_cast<int32_t>(rhs.randomStateI)
            << " compact_j=" << static_cast<int32_t>(lhs.randomStateJ)
            << " interpreter_j=" << static_cast<int32_t>(rhs.randomStateJ);
        return out.str();
    }
    return " state_equal";
}

CompactTickTryResult tryCompiledCompactTick(
    const Game& game,
    const CompactSolverState& parent,
    ps_input input,
    int32_t width,
    int32_t height,
    int32_t currentLevelIndex,
    puzzlescript::RuntimeStepOptions options
) {
    CompactTickTryResult result;
    if (game.compiledCompactTick == nullptr || game.compiledCompactTick->step == nullptr) {
        return result;
    }
    result.attempted = true;
    result.compact = parent;
    puzzlescript::CompiledCompactTickStateView view{
        result.compact.objectBits.empty() ? nullptr : result.compact.objectBits.data(),
        result.compact.objectBits.size(),
        nullptr,
        0,
        width,
        height,
        result.compact.randomStateS.data(),
        result.compact.randomStateS.size(),
        &result.compact.randomStateI,
        &result.compact.randomStateJ,
        &result.compact.randomStateValid,
        currentLevelIndex,
    };
    const puzzlescript::CompiledCompactTickApplyOutcome outcome =
        game.compiledCompactTick->step(game, view, input, options);
    result.handled = outcome.handled;
    result.stepResult = outcome.result;
    return result;
}

SolverEdgeStep stepSolverEdge(
    const std::shared_ptr<const Game>& game,
    const Node& parentNode,
    const Session& parentSession,
    ps_input input,
    bool compactNodeStorage,
    int32_t width,
    int32_t height,
    Session& childScratch,
    Result& result,
    bool compactTickOracle
) {
    constexpr puzzlescript::RuntimeStepOptions solverStepOptions{
        .playableUndo = false,
        .emitAudio = false,
    };
    SolverEdgeStep edge;
    if (compactNodeStorage) {
        const puzzlescript::CompiledCompactTickBackend* compactTick = game ? game->compiledCompactTick : nullptr;
        if (compactTick != nullptr && compactTick->step != nullptr) {
            if (compactTick->support.wholeTurnSupported) {
                ++result.compactTickAttempts;
                {
                    ScopedTimer timer(result.timing.stepNs);
                    edge.compactTick = tryCompiledCompactTick(
                        *game,
                        parentNode.compact,
                        input,
                        width,
                        height,
                        parentSession.preparedSession.currentLevelIndex,
                        solverStepOptions
                    );
                }
                if (edge.compactTick.handled) {
                    ++result.compactTickHits;
                    if (compactTickOracle) {
                        ++result.compactTickOracleChecks;
                        {
                            ScopedTimer timer(result.timing.cloneNs);
                            prepareSolverChildSessionFromParent(childScratch, parentSession);
                        }
                        ps_step_result oracleStepResult{};
                        {
                            ScopedTimer timer(result.timing.stepNs);
                            oracleStepResult = puzzlescript::step(childScratch, input, solverStepOptions);
                            puzzlescript::settlePendingAgain(childScratch, solverStepOptions);
                        }
                        const bool terminalEdge = edge.compactTick.stepResult.won
                            || oracleStepResult.won
                            || edge.compactTick.stepResult.transitioned
                            || oracleStepResult.transitioned
                            || edge.compactTick.stepResult.restarted
                            || oracleStepResult.restarted;
                        CompactSolverState oracleCompact;
                        if (!terminalEdge) {
                            oracleCompact = compactStateWithTiming(childScratch, result.timing);
                        }
                        if (!equivalentSolverStepResult(edge.compactTick.stepResult, oracleStepResult)
                            || (!terminalEdge && !(edge.compactTick.compact == oracleCompact))) {
                            ++result.compactTickOracleFailures;
                            edge.oracleMismatch = true;
                            edge.oracleError = "compact tick oracle mismatch input=" + inputName(input)
                                + " depth=" + std::to_string(parentNode.depth)
                                + " compact_step=" + stepResultSummary(edge.compactTick.stepResult)
                                + " interpreter_step=" + stepResultSummary(oracleStepResult)
                                + compactStateDiffSummary(edge.compactTick.compact, oracleCompact);
                        }
                    }
                } else {
                    ++result.compactTickFallbacks;
                }
            } else {
                ++result.compactTickUnsupported;
            }
        }
    }

    if (!edge.compactTick.handled) {
        ScopedTimer timer(result.timing.cloneNs);
        if (!compactNodeStorage) {
            edge.ownedChild = std::make_unique<Session>(parentSession);
            edge.child = edge.ownedChild.get();
        } else {
            prepareSolverChildSessionFromParent(childScratch, parentSession);
            edge.child = &childScratch;
        }
    }

    if (edge.compactTick.handled) {
        edge.stepResult = edge.compactTick.stepResult;
    } else {
        ScopedTimer timer(result.timing.stepNs);
        edge.stepResult = puzzlescript::step(*edge.child, input, solverStepOptions);
        puzzlescript::settlePendingAgain(*edge.child, solverStepOptions);
    }
    return edge;
}

std::shared_ptr<const Game> compileGame(
    const std::string& source,
    std::string& errorMessage
) {
    try {
        puzzlescript::compiler::DiagnosticSink diagnostics;
        const auto state = puzzlescript::compiler::parseSource(source, diagnostics);
        std::shared_ptr<const Game> game;
        if (auto error = puzzlescript::compiler::lowerToRuntimeGame(state, game)) {
            errorMessage = error->message;
            return nullptr;
        }
        if (game) {
            puzzlescript::attachLinkedCompiledRules(*std::const_pointer_cast<Game>(game), source);
        }
        return game;
    } catch (const std::exception& error) {
        errorMessage = error.what();
        return nullptr;
    }
}

std::vector<std::string> reconstructSolution(const std::vector<Node>& nodes, uint32_t nodeIndex, ps_input finalInput, Timing& timing) {
    ScopedTimer timer(timing.reconstructNs);
    std::vector<std::string> reversed;
    reversed.push_back(inputName(finalInput));
    int32_t cursor = static_cast<int32_t>(nodeIndex);
    while (cursor >= 0) {
        const Node& node = nodes[static_cast<size_t>(cursor)];
        if (node.parent >= 0) {
            reversed.push_back(inputName(node.input));
        }
        cursor = node.parent;
    }
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

bool solvedByStep(const ps_step_result& stepResult, ps_session* session, int32_t levelIndex) {
    if (stepResult.won) {
        return true;
    }
    ps_session_status_info status{};
    ps_session_status(session, &status);
    return status.current_level_index != levelIndex;
}

int32_t heuristicScore(Session& session, puzzlescript::search::HeuristicScratch& scratch) {
    puzzlescript::search::HeuristicOptions options;
    options.includeNoQuantifierPenalty = true;
    options.includePlayerDistance = true;
    return puzzlescript::search::winConditionHeuristicScore(session, options, scratch);
}

bool compactObjectPresent(
    const CompactSolverState& state,
    int32_t objectId,
    int32_t tileIndex,
    size_t cellWordCount
) {
    const size_t word = static_cast<size_t>(tileIndex >> 6);
    const uint64_t mask = uint64_t{1} << static_cast<uint32_t>(tileIndex & 63);
    const size_t offset = static_cast<size_t>(objectId) * cellWordCount + word;
    return offset < state.objectBits.size() && (state.objectBits[offset] & mask) != 0;
}

bool compactMatchesFilter(
    const CompactSolverState& state,
    const Game& game,
    const puzzlescript::MaskWord* filter,
    bool aggregate,
    int32_t tileIndex,
    size_t cellWordCount
) {
    if (filter == nullptr) {
        return false;
    }
    bool sawFilterBit = false;
    for (int32_t word = 0; word < game.wordCount; ++word) {
        MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(filter[static_cast<size_t>(word)]);
        while (bits != 0) {
            sawFilterBit = true;
            const uint32_t bit = compactWordTrailingZeros(bits);
            const int32_t objectId = word * static_cast<int32_t>(kMaskWordBits) + static_cast<int32_t>(bit);
            const bool present = compactObjectPresent(state, objectId, tileIndex, cellWordCount);
            if (aggregate && !present) {
                return false;
            }
            if (!aggregate && present) {
                return true;
            }
            bits &= bits - 1;
        }
    }
    return aggregate && sawFilterBit;
}

std::vector<int32_t> compactMatchingDistanceField(
    const CompactSolverState& state,
    const Game& game,
    int32_t width,
    int32_t height,
    const puzzlescript::MaskWord* filter,
    bool aggregate
) {
    const int32_t tileCount = width * height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    std::vector<int32_t> distances(static_cast<size_t>(tileCount), std::numeric_limits<int32_t>::max());
    if (filter == nullptr) {
        return distances;
    }

    for (int32_t tile = 0; tile < tileCount; ++tile) {
        if (compactMatchesFilter(state, game, filter, aggregate, tile, cellWordCount)) {
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
    return distances;
}

int32_t compactHeuristicScore(
    const CompactSolverState& state,
    const Game& game,
    int32_t width,
    int32_t height
) {
    if (game.winConditions.empty()) {
        return 0;
    }

    int32_t score = 0;
    const int32_t tileCount = width * height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    for (const auto& condition : game.winConditions) {
        const puzzlescript::MaskWord* filter1 = puzzlescript::search::maskPtr(game, condition.filter1);
        const puzzlescript::MaskWord* filter2 = puzzlescript::search::maskPtr(game, condition.filter2);
        if (filter1 == nullptr || filter2 == nullptr) {
            continue;
        }
        const std::vector<int32_t> filter2Distances = compactMatchingDistanceField(state, game, width, height, filter2, condition.aggr2);
        if (condition.quantifier == 1) {
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                if (!compactMatchesFilter(state, game, filter1, condition.aggr1, tile, cellWordCount)) {
                    continue;
                }
                if (compactMatchesFilter(state, game, filter2, condition.aggr2, tile, cellWordCount)) {
                    continue;
                }
                score += 10 + puzzlescript::search::distanceOrFallback(filter2Distances[static_cast<size_t>(tile)]);
            }
        } else if (condition.quantifier == 0) {
            bool passed = false;
            int32_t best = puzzlescript::search::kNoMatchingDistance;
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                if (!compactMatchesFilter(state, game, filter1, condition.aggr1, tile, cellWordCount)) {
                    continue;
                }
                if (compactMatchesFilter(state, game, filter2, condition.aggr2, tile, cellWordCount)) {
                    passed = true;
                    break;
                }
                best = std::min(best, puzzlescript::search::distanceOrFallback(filter2Distances[static_cast<size_t>(tile)]));
            }
            score += passed ? 0 : best;
        } else if (condition.quantifier == -1) {
            for (int32_t tile = 0; tile < tileCount; ++tile) {
                if (compactMatchesFilter(state, game, filter1, condition.aggr1, tile, cellWordCount)
                    && compactMatchesFilter(state, game, filter2, condition.aggr2, tile, cellWordCount)) {
                    score += 10;
                }
            }
        }
    }

    if (game.playerMask != puzzlescript::kNullMaskOffset && score > 0) {
        const puzzlescript::MaskWord* playerMask = puzzlescript::search::maskPtr(game, game.playerMask);
        bool hasPlayer = false;
        int32_t best = puzzlescript::search::kNoMatchingDistance;
        std::vector<std::vector<int32_t>> conditionDistances;
        conditionDistances.reserve(game.winConditions.size());
        for (const auto& condition : game.winConditions) {
            conditionDistances.push_back(compactMatchingDistanceField(
                state,
                game,
                width,
                height,
                puzzlescript::search::maskPtr(game, condition.filter1),
                condition.aggr1
            ));
        }
        for (int32_t player = 0; player < tileCount; ++player) {
            if (!compactMatchesFilter(state, game, playerMask, game.playerMaskAggregate, player, cellWordCount)) {
                continue;
            }
            hasPlayer = true;
            for (const auto& distances : conditionDistances) {
                best = std::min(best, puzzlescript::search::distanceOrFallback(distances[static_cast<size_t>(player)]));
            }
        }
        if (hasPlayer) {
            score += std::min(best, 16);
        }
    }

    return score;
}

std::unique_ptr<Session> createLoadedSession(
    const std::shared_ptr<const Game>& game,
    const std::string& gameName,
    int32_t levelIndex,
    Result& result
) {
    const std::string seed = "solver:" + gameName + ":" + std::to_string(levelIndex);
    auto session = puzzlescript::createSessionWithLoadedLevelSeed(game, seed);
    session->suppressRuleMessages = true;
    if (auto error = puzzlescript::loadLevel(*session, levelIndex)) {
        result.status = "level_error";
        result.error = error->message;
        return nullptr;
    }
    return session;
}

bool solvedByStep(const ps_step_result& stepResult, const Session& session, int32_t levelIndex) {
    return stepResult.won || session.preparedSession.currentLevelIndex != levelIndex;
}

class FlatBestDepth {
public:
    FlatBestDepth(Timing& timing, bool exactStateKeys)
        : timing(timing), exactStateKeys(exactStateKeys) {}

    void reserve(size_t expected) {
        rehash(capacityForExpected(expected));
    }

    std::optional<uint32_t> find(
        const StateKey& key,
        const CompactSolverState& compact,
        const std::vector<Node>& nodes
    ) {
        if (entries.empty()) {
            return std::nullopt;
        }
        size_t probes = 0;
        const size_t slot = findSlot(key, compact, nodes, probes);
        recordLookup(probes);
        if (!entries[slot].occupied) {
            return std::nullopt;
        }
        return entries[slot].depth;
    }

    bool insertOrAssignIfBetter(
        const StateKey& key,
        const CompactSolverState& compact,
        uint32_t depth,
        uint32_t nodeIndex,
        const std::vector<Node>& nodes
    ) {
        ensureCapacityForInsert();
        size_t probes = 0;
        const size_t slot = findSlot(key, compact, nodes, probes);
        recordInsert(probes);
        Entry& entry = entries[slot];
        if (entry.occupied) {
            if (entry.depth <= depth) {
                return false;
            }
            entry.depth = depth;
            entry.nodeIndex = nodeIndex;
            return true;
        }
        entry.key = key;
        entry.depth = depth;
        entry.nodeIndex = nodeIndex;
        entry.occupied = true;
        ++entryCount;
        return true;
    }

    size_t size() const {
        return entryCount;
    }

private:
    struct Entry {
        StateKey key;
        uint32_t depth = 0;
        uint32_t nodeIndex = 0;
        bool occupied = false;
    };

    static size_t capacityForExpected(size_t expected) {
        size_t capacity = 16;
        const size_t minimum = std::max<size_t>(16, (expected * 10 + 6) / 7);
        while (capacity < minimum) {
            capacity *= 2;
        }
        return capacity;
    }

    void ensureCapacityForInsert() {
        if (entries.empty()) {
            rehash(16);
            return;
        }
        if ((entryCount + 1) * 10 >= entries.size() * 7) {
            rehash(entries.size() * 2);
            ++timing.visitedGrows;
        }
    }

    void rehash(size_t newCapacity) {
        std::vector<Entry> oldEntries = std::move(entries);
        entries.clear();
        entries.resize(newCapacity);
        entryCount = 0;
        timing.visitedCapacity = std::max<uint64_t>(timing.visitedCapacity, entries.size());
        for (const Entry& entry : oldEntries) {
            if (!entry.occupied) {
                continue;
            }
            const size_t slot = findEmptySlot(entry.key);
            entries[slot] = entry;
            ++entryCount;
        }
    }

    size_t findSlot(
        const StateKey& key,
        const CompactSolverState& compact,
        const std::vector<Node>& nodes,
        size_t& probes
    ) {
        const size_t mask = entries.size() - 1;
        size_t slot = StateKeyHash{}(key) & mask;
        while (true) {
            ++probes;
            const Entry& entry = entries[slot];
            if (!entry.occupied) {
                return slot;
            }
            if (entry.key == key) {
                if (!exactStateKeys) {
                    return slot;
                }
                if (nodes[entry.nodeIndex].compact == compact) {
                    return slot;
                }
                ++timing.visitedKeyCollisions;
            }
            slot = (slot + 1) & mask;
        }
    }

    size_t findEmptySlot(const StateKey& key) const {
        const size_t mask = entries.size() - 1;
        size_t slot = StateKeyHash{}(key) & mask;
        while (entries[slot].occupied) {
            slot = (slot + 1) & mask;
        }
        return slot;
    }

    void recordLookup(size_t probes) {
        timing.visitedLookupProbes += probes;
        timing.visitedMaxProbe = std::max<uint64_t>(timing.visitedMaxProbe, probes);
    }

    void recordInsert(size_t probes) {
        timing.visitedInsertProbes += probes;
        timing.visitedMaxProbe = std::max<uint64_t>(timing.visitedMaxProbe, probes);
    }

    Timing& timing;
    bool exactStateKeys = false;
    std::vector<Entry> entries;
    size_t entryCount = 0;
};

Result runSearch(
    const std::shared_ptr<const Game>& game,
    const std::string& gameName,
    int32_t levelIndex,
    int64_t timeoutMs,
    int64_t compileNs,
    SearchMode mode,
    TimePoint deadline,
    uint32_t workerId,
    bool exactStateKeys,
    bool compactNodeStorage,
    bool compactTickOracle,
    int32_t astarWeight
) {
    Result result;
    result.game = gameName;
    result.level = levelIndex;
    result.status = "exhausted";
    result.strategy = searchModeName(mode);
    result.heuristic = mode == SearchMode::Bfs ? "zero" : "winconditions";
    result.timeoutMs = timeoutMs;
    result.workerId = workerId;
    result.compiledRulesAttached = game && game->compiledRules != nullptr;
    result.compiledTickAttached = game && game->compiledTick != nullptr;
    result.compiledCompactTickAttached = game && game->compiledCompactTick != nullptr;
    result.compactNodeStorage = compactNodeStorage;
    result.astarWeight = astarWeight;
    result.timing.compileNs = compileNs;

    std::unique_ptr<Session> initial;
    {
        ScopedTimer timer(result.timing.loadNs);
        initial = createLoadedSession(game, gameName, levelIndex, result);
    }
    if (!initial) {
        return result;
    }
    if (initial->preparedSession.textMode || initial->preparedSession.level.isMessage) {
        result.status = "skipped_message";
        return result;
    }
    const int32_t searchWidth = initial->liveLevel.width;
    const int32_t searchHeight = initial->liveLevel.height;
    std::unique_ptr<Session> compactSessionBase;
    std::unique_ptr<Session> parentScratch;
    std::unique_ptr<Session> childScratch;
    if (compactNodeStorage) {
        compactSessionBase = std::make_unique<Session>(*initial);
        parentScratch = std::make_unique<Session>(*initial);
        childScratch = std::make_unique<Session>(*initial);
    }

    std::vector<Node> nodes;
    nodes.reserve(8192);

    FlatBestDepth bestDepth(result.timing, exactStateKeys);
    bestDepth.reserve(16384);
    result.uniqueStates = 1;
    puzzlescript::search::HeuristicScratch heuristicScratch;

    CompactSolverState initialCompact = compactStateWithTiming(*initial, result.timing);
    const StateKey initialKey = compactStateKey(initialCompact, result.timing);
    int32_t initialHeuristic = 0;
    if (mode != SearchMode::Bfs) {
        ScopedTimer timer(result.timing.heuristicNs);
        initialHeuristic = compactNodeStorage
            ? compactHeuristicScore(initialCompact, *game, searchWidth, searchHeight)
            : heuristicScore(*initial, heuristicScratch);
    }
    {
        ScopedTimer timer(result.timing.nodeStoreNs);
        nodes.push_back(Node{compactNodeStorage ? nullptr : std::move(initial), std::move(initialCompact), initialKey, -1, PS_INPUT_UP, 0, initialHeuristic});
        recordCompactStateStorage(result.timing, nodes.back().compact);
    }
    {
        ScopedTimer timer(result.timing.visitedInsertNs);
        bestDepth.insertOrAssignIfBetter(initialKey, nodes[0].compact, 0, 0, nodes);
    }
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> frontier;
    {
        ScopedTimer timer(result.timing.frontierPushNs);
        frontier.push(QueueEntry{priorityFor(mode, 0, initialHeuristic, astarWeight), 0, 0});
    }
    result.maxFrontier = 1;

    uint64_t nextTie = 1;
    const auto inputs = solverInputsForGame(*game);

    while (!frontier.empty()) {
        bool timedOut = false;
        {
            ScopedTimer timer(result.timing.timeoutCheckNs);
            timedOut = Clock::now() >= deadline;
        }
        if (timedOut) {
            result.status = "timeout";
            break;
        }

        QueueEntry entry;
        {
            ScopedTimer timer(result.timing.frontierPopNs);
            entry = frontier.top();
            frontier.pop();
        }

        const Node& parentNode = nodes[entry.nodeIndex];
        std::optional<uint32_t> best;
        {
            ScopedTimer timer(result.timing.visitedLookupNs);
            best = bestDepth.find(parentNode.key, parentNode.compact, nodes);
        }
        if (best && *best < parentNode.depth) {
            ++result.duplicates;
            continue;
        }

        const Session* parentSessionPtr = parentNode.session.get();
        if (parentSessionPtr == nullptr) {
            {
                ScopedTimer timer(result.timing.cloneNs);
                materializeCompactStateIntoSession(parentNode.compact, *compactSessionBase, *parentScratch);
            }
            parentSessionPtr = parentScratch.get();
        }
        const Session& parentSession = *parentSessionPtr;
        const uint32_t parentDepth = parentNode.depth;
        ++result.expanded;

        for (const ps_input input : inputs) {
            timedOut = false;
            {
                ScopedTimer timer(result.timing.timeoutCheckNs);
                timedOut = Clock::now() >= deadline;
            }
            if (timedOut) {
                result.status = "timeout";
                break;
            }

            SolverEdgeStep edge = stepSolverEdge(
                game,
                parentNode,
                parentSession,
                input,
                compactNodeStorage,
                searchWidth,
                searchHeight,
                *childScratch,
                result,
                compactTickOracle
            );
            if (edge.oracleMismatch) {
                result.status = "level_error";
                result.error = edge.oracleError;
                return result;
            }
            const ps_step_result& stepResult = edge.stepResult;
            ++result.generated;

            if (stepResult.restarted) {
                continue;
            }

            bool solved = false;
            {
                ScopedTimer timer(result.timing.solvedCheckNs);
                solved = edge.compactTick.handled ? stepResult.won : solvedByStep(stepResult, *edge.child, levelIndex);
            }
            if (solved) {
                result.status = "solved";
                result.solution = reconstructSolution(nodes, entry.nodeIndex, input, result.timing);
                return result;
            }
            if (!stepResult.changed) {
                continue;
            }

            CompactSolverState compact = edge.compactTick.handled
                ? std::move(edge.compactTick.compact)
                : compactStateWithTiming(*edge.child, result.timing);
            const StateKey key = compactStateKey(compact, result.timing);
            const uint32_t childDepth = parentDepth + 1;
            uint32_t childIndex = static_cast<uint32_t>(nodes.size());
            int32_t childHeuristic = 0;
            if (exactStateKeys) {
                bool shouldStore = false;
                {
                    ScopedTimer timer(result.timing.visitedInsertNs);
                    shouldStore = bestDepth.insertOrAssignIfBetter(key, compact, childDepth, childIndex, nodes);
                    result.uniqueStates = bestDepth.size();
                }
                if (!shouldStore) {
                    ++result.duplicates;
                    continue;
                }
                if (mode != SearchMode::Bfs) {
                    ScopedTimer timer(result.timing.heuristicNs);
                    childHeuristic = compactNodeStorage
                        ? compactHeuristicScore(compact, *game, searchWidth, searchHeight)
                        : heuristicScore(*edge.child, heuristicScratch);
                }
                {
                    ScopedTimer timer(result.timing.nodeStoreNs);
                    nodes.push_back(Node{compactNodeStorage ? nullptr : std::move(edge.ownedChild), std::move(compact), key, static_cast<int32_t>(entry.nodeIndex), input, childDepth, childHeuristic});
                    recordCompactStateStorage(result.timing, nodes.back().compact);
                }
            } else {
                bool shouldStore = false;
                {
                    ScopedTimer timer(result.timing.visitedInsertNs);
                    shouldStore = bestDepth.insertOrAssignIfBetter(key, compact, childDepth, 0, nodes);
                    result.uniqueStates = bestDepth.size();
                }
                if (!shouldStore) {
                    ++result.duplicates;
                    continue;
                }
                if (mode != SearchMode::Bfs) {
                    ScopedTimer timer(result.timing.heuristicNs);
                    childHeuristic = compactNodeStorage
                        ? compactHeuristicScore(compact, *game, searchWidth, searchHeight)
                        : heuristicScore(*edge.child, heuristicScratch);
                }
                childIndex = static_cast<uint32_t>(nodes.size());
                {
                    ScopedTimer timer(result.timing.nodeStoreNs);
                    nodes.push_back(Node{compactNodeStorage ? nullptr : std::move(edge.ownedChild), std::move(compact), key, static_cast<int32_t>(entry.nodeIndex), input, childDepth, childHeuristic});
                    recordCompactStateStorage(result.timing, nodes.back().compact);
                }
            }
            {
                ScopedTimer timer(result.timing.frontierPushNs);
                frontier.push(QueueEntry{priorityFor(mode, childDepth, childHeuristic, astarWeight), nextTie++, childIndex});
            }
            result.maxFrontier = std::max<uint64_t>(result.maxFrontier, frontier.size());
        }
    }

    return result;
}

void mergeStats(Result& target, const Result& source) {
    target.expanded += source.expanded;
    target.generated += source.generated;
    target.uniqueStates += source.uniqueStates;
    target.duplicates += source.duplicates;
    target.maxFrontier = std::max(target.maxFrontier, source.maxFrontier);
    target.compactTickAttempts += source.compactTickAttempts;
    target.compactTickHits += source.compactTickHits;
    target.compactTickFallbacks += source.compactTickFallbacks;
    target.compactTickUnsupported += source.compactTickUnsupported;
    target.compactTickOracleChecks += source.compactTickOracleChecks;
    target.compactTickOracleFailures += source.compactTickOracleFailures;
    target.timing.loadNs += source.timing.loadNs;
    target.timing.cloneNs += source.timing.cloneNs;
    target.timing.stepNs += source.timing.stepNs;
    target.timing.hashNs += source.timing.hashNs;
    target.timing.queueNs += source.timing.queueNs;
    target.timing.frontierPopNs += source.timing.frontierPopNs;
    target.timing.frontierPushNs += source.timing.frontierPushNs;
    target.timing.visitedLookupNs += source.timing.visitedLookupNs;
    target.timing.visitedInsertNs += source.timing.visitedInsertNs;
    target.timing.nodeStoreNs += source.timing.nodeStoreNs;
    target.timing.heuristicNs += source.timing.heuristicNs;
    target.timing.solvedCheckNs += source.timing.solvedCheckNs;
    target.timing.timeoutCheckNs += source.timing.timeoutCheckNs;
    target.timing.reconstructNs += source.timing.reconstructNs;
    target.timing.visitedLookupProbes += source.timing.visitedLookupProbes;
    target.timing.visitedInsertProbes += source.timing.visitedInsertProbes;
    target.timing.visitedGrows += source.timing.visitedGrows;
    target.timing.visitedCapacity = std::max(target.timing.visitedCapacity, source.timing.visitedCapacity);
    target.timing.visitedMaxProbe = std::max(target.timing.visitedMaxProbe, source.timing.visitedMaxProbe);
    target.timing.visitedKeyCollisions += source.timing.visitedKeyCollisions;
    target.timing.compactStateBytes += source.timing.compactStateBytes;
    target.timing.compactMaxStateBytes = std::max(target.timing.compactMaxStateBytes, source.timing.compactMaxStateBytes);
}

Result solveLevel(
    const std::shared_ptr<const Game>& game,
    const std::string& gameName,
    int32_t levelIndex,
    int64_t timeoutMs,
    int64_t compileNs,
    Strategy strategy,
    uint32_t workerId,
    bool exactStateKeys,
    bool compactNodeStorage,
    bool compactTickOracle,
    int32_t astarWeight
) {
    const TimePoint searchStart = Clock::now();
    const TimePoint deadline = searchStart + std::chrono::milliseconds(timeoutMs);

    auto finish = [&](Result result) {
        result.strategy = result.status == "solved" ? result.strategy : strategyName(strategy);
        result.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - searchStart).count();
        return result;
    };

    if (strategy == Strategy::Bfs) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::Bfs, deadline, workerId, exactStateKeys, compactNodeStorage, compactTickOracle, astarWeight));
    }
    if (strategy == Strategy::WeightedAStar) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::WeightedAStar, deadline, workerId, exactStateKeys, compactNodeStorage, compactTickOracle, astarWeight));
    }
    if (strategy == Strategy::Greedy) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::Greedy, deadline, workerId, exactStateKeys, compactNodeStorage, compactTickOracle, astarWeight));
    }

    Result combined;
    combined.game = gameName;
    combined.level = levelIndex;
    combined.status = "timeout";
    combined.strategy = "portfolio";
    combined.heuristic = "winconditions";
    combined.timeoutMs = timeoutMs;
    combined.workerId = workerId;
    combined.compiledRulesAttached = game && game->compiledRules != nullptr;
    combined.compiledTickAttached = game && game->compiledTick != nullptr;
    combined.compiledCompactTickAttached = game && game->compiledCompactTick != nullptr;
    combined.compactNodeStorage = compactNodeStorage;
    combined.astarWeight = astarWeight;
    combined.timing.compileNs = compileNs;

    const TimePoint bfsDeadline = searchStart + std::chrono::milliseconds(std::max<int64_t>(1, timeoutMs / 6));
    Result bfs = runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::Bfs, std::min(bfsDeadline, deadline), workerId, exactStateKeys, compactNodeStorage, compactTickOracle, astarWeight);
    mergeStats(combined, bfs);
    if (bfs.status == "solved" || bfs.status == "skipped_message" || bfs.status == "level_error") {
        bfs.strategy = bfs.status == "solved" ? "bfs" : "portfolio";
        return finish(bfs);
    }

    if (Clock::now() < deadline) {
        Result weighted = runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::WeightedAStar, deadline, workerId, exactStateKeys, compactNodeStorage, compactTickOracle, astarWeight);
        mergeStats(combined, weighted);
        if (weighted.status == "solved" || weighted.status == "level_error") {
            combined.status = weighted.status;
            combined.error = weighted.error;
            combined.strategy = weighted.status == "solved" ? "weighted-astar" : "portfolio";
            combined.heuristic = weighted.heuristic;
            combined.solution = std::move(weighted.solution);
            return finish(combined);
        }
        combined.status = weighted.status == "exhausted" ? "exhausted" : "timeout";
    }

    return finish(combined);
}

std::string relativeGameName(const std::filesystem::path& root, const std::filesystem::path& gamePath) {
    if (std::filesystem::is_directory(root)) {
        return std::filesystem::relative(gamePath, root).generic_string();
    }
    return gamePath.filename().generic_string();
}

std::vector<std::string> splitLines(const std::string& source) {
    std::vector<std::string> lines;
    std::istringstream stream(source);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    if (!source.empty() && source.back() == '\n') {
        return lines;
    }
    if (source.empty()) {
        lines.emplace_back();
    }
    return lines;
}

std::vector<SourceLevel> findSourceLevels(const std::vector<std::string>& lines) {
    std::vector<SourceLevel> levels;
    size_t index = 0;
    for (; index < lines.size(); ++index) {
        if (lowercase(trim(lines[index])) == "levels") {
            ++index;
            break;
        }
    }
    if (index >= lines.size()) {
        return levels;
    }

    int32_t levelIndex = 0;
    while (index < lines.size()) {
        const std::string stripped = trim(lines[index]);
        const std::string lower = lowercase(stripped);
        if (stripped.empty() || isDividerLine(lines[index]) || isCommentLine(lines[index])) {
            ++index;
            continue;
        }
        if (lower == "message" || startsWith(lower, "message ")) {
            levels.push_back(SourceLevel{levelIndex++, index, true});
            ++index;
            continue;
        }

        levels.push_back(SourceLevel{levelIndex++, index, false});
        ++index;
        while (index < lines.size() && !trim(lines[index]).empty()) {
            ++index;
        }
    }
    return levels;
}

char solutionLetter(const std::string& input) {
    if (input == "up") {
        return 'U';
    }
    if (input == "down") {
        return 'D';
    }
    if (input == "left") {
        return 'L';
    }
    if (input == "right") {
        return 'R';
    }
    if (input == "action") {
        return 'A';
    }
    return '?';
}

std::string compactSolution(const std::vector<std::string>& solution) {
    std::string out;
    for (size_t index = 0; index < solution.size(); ++index) {
        if (index > 0 && (index % 4) == 0) {
            out.push_back(' ');
        }
        out.push_back(solutionLetter(solution[index]));
    }
    return out;
}

bool writeAnnotatedSolutions(
    const Options& options,
    const std::string& gameName,
    const std::string& source,
    const std::vector<Result>& results,
    size_t begin,
    size_t end
) {
    if (!options.writeSolutions) {
        return false;
    }

    std::unordered_map<int32_t, std::string> solved;
    for (size_t index = begin; index < end; ++index) {
        const Result& result = results[index];
        if (result.status == "solved" && !result.solution.empty()) {
            solved[result.level] = compactSolution(result.solution);
        }
    }
    if (solved.empty()) {
        return false;
    }

    const std::vector<std::string> lines = splitLines(source);
    const std::vector<SourceLevel> sourceLevels = findSourceLevels(lines);
    std::unordered_map<size_t, std::vector<std::string>> commentsByLine;
    for (const SourceLevel& level : sourceLevels) {
        if (level.message) {
            continue;
        }
        const auto found = solved.find(level.level);
        if (found != solved.end()) {
            commentsByLine[level.insertBeforeLine].push_back("(" + found->second + ")");
        }
    }
    if (commentsByLine.empty()) {
        return false;
    }

    std::ostringstream annotated;
    for (size_t lineIndex = 0; lineIndex < lines.size(); ++lineIndex) {
        const auto comments = commentsByLine.find(lineIndex);
        if (comments != commentsByLine.end()) {
            for (const std::string& comment : comments->second) {
                annotated << comment << "\n";
            }
        }
        annotated << lines[lineIndex] << "\n";
    }

    writeFile(options.solutionsDir / gameName, annotated.str());
    return true;
}

void printJsonResult(const Result& result, std::ostream& out) {
    out << "{";
    out << "\"game\":" << jsonString(result.game);
    out << ",\"level\":" << result.level;
    out << ",\"status\":" << jsonString(result.status);
    out << ",\"strategy\":" << jsonString(result.strategy);
    out << ",\"heuristic\":" << jsonString(result.heuristic);
    out << ",\"worker_id\":" << result.workerId;
    if (!result.error.empty()) {
        out << ",\"error\":" << jsonString(result.error);
    }
    out << ",\"solution\":[";
    for (size_t index = 0; index < result.solution.size(); ++index) {
        if (index > 0) {
            out << ",";
        }
        out << jsonString(result.solution[index]);
    }
    out << "]";
    out << ",\"solution_length\":" << result.solution.size();
    out << ",\"elapsed_ms\":" << result.elapsedMs;
    out << ",\"expanded\":" << result.expanded;
    out << ",\"generated\":" << result.generated;
    out << ",\"unique_states\":" << result.uniqueStates;
    out << ",\"duplicates\":" << result.duplicates;
    out << ",\"max_frontier\":" << result.maxFrontier;
    out << ",\"timeout_ms\":" << result.timeoutMs;
    out << ",\"compiled_rules_attached\":" << (result.compiledRulesAttached ? "true" : "false");
    out << ",\"compiled_tick_attached\":" << (result.compiledTickAttached ? "true" : "false");
    out << ",\"compiled_compact_tick_attached\":" << (result.compiledCompactTickAttached ? "true" : "false");
    out << ",\"compact_node_storage\":" << (result.compactNodeStorage ? "true" : "false");
    out << ",\"astar_weight\":" << result.astarWeight;
    out << ",\"compact_tick_attempts\":" << result.compactTickAttempts;
    out << ",\"compact_tick_hits\":" << result.compactTickHits;
    out << ",\"compact_tick_fallbacks\":" << result.compactTickFallbacks;
    out << ",\"compact_tick_unsupported\":" << result.compactTickUnsupported;
    out << ",\"compact_tick_oracle_checks\":" << result.compactTickOracleChecks;
    out << ",\"compact_tick_oracle_failures\":" << result.compactTickOracleFailures;
    out << ",\"compile_ms\":" << ms(result.timing.compileNs);
    out << ",\"load_ms\":" << ms(result.timing.loadNs);
    out << ",\"clone_ms\":" << ms(result.timing.cloneNs);
    out << ",\"step_ms\":" << ms(result.timing.stepNs);
    out << ",\"hash_ms\":" << ms(result.timing.hashNs);
    out << ",\"queue_ms\":" << ms(result.timing.queueNs);
    out << ",\"frontier_pop_ms\":" << ms(result.timing.frontierPopNs);
    out << ",\"frontier_push_ms\":" << ms(result.timing.frontierPushNs);
    out << ",\"visited_lookup_ms\":" << ms(result.timing.visitedLookupNs);
    out << ",\"visited_insert_ms\":" << ms(result.timing.visitedInsertNs);
    out << ",\"visited_lookup_probes\":" << result.timing.visitedLookupProbes;
    out << ",\"visited_insert_probes\":" << result.timing.visitedInsertProbes;
    out << ",\"visited_grows\":" << result.timing.visitedGrows;
    out << ",\"visited_capacity\":" << result.timing.visitedCapacity;
    out << ",\"visited_max_probe\":" << result.timing.visitedMaxProbe;
    out << ",\"visited_key_collisions\":" << result.timing.visitedKeyCollisions;
    out << ",\"compact_state_bytes\":" << result.timing.compactStateBytes;
    out << ",\"compact_max_state_bytes\":" << result.timing.compactMaxStateBytes;
    out << ",\"node_store_ms\":" << ms(result.timing.nodeStoreNs);
    out << ",\"heuristic_ms\":" << ms(result.timing.heuristicNs);
    out << ",\"solved_check_ms\":" << ms(result.timing.solvedCheckNs);
    out << ",\"timeout_check_ms\":" << ms(result.timing.timeoutCheckNs);
    out << ",\"reconstruct_ms\":" << ms(result.timing.reconstructNs);
    out << ",\"unattributed_ms\":" << unattributedMs(result);
    out << "}";
}

void printJson(const std::vector<Result>& results) {
    uint64_t solved = 0;
    uint64_t timeout = 0;
    uint64_t exhausted = 0;
    uint64_t skipped = 0;
    uint64_t errors = 0;
    Timing timing{};
    uint64_t expanded = 0;
    uint64_t generated = 0;
    uint64_t compactTickAttempts = 0;
    uint64_t compactTickHits = 0;
    uint64_t compactTickFallbacks = 0;
    uint64_t compactTickUnsupported = 0;
    uint64_t compactTickOracleChecks = 0;
    uint64_t compactTickOracleFailures = 0;
    for (const auto& result : results) {
        solved += result.status == "solved";
        timeout += result.status == "timeout";
        exhausted += result.status == "exhausted";
        skipped += result.status == "skipped_message";
        errors += result.status == "compile_error" || result.status == "level_error";
        expanded += result.expanded;
        generated += result.generated;
        compactTickAttempts += result.compactTickAttempts;
        compactTickHits += result.compactTickHits;
        compactTickFallbacks += result.compactTickFallbacks;
        compactTickUnsupported += result.compactTickUnsupported;
        compactTickOracleChecks += result.compactTickOracleChecks;
        compactTickOracleFailures += result.compactTickOracleFailures;
        timing.compileNs += result.timing.compileNs;
        timing.loadNs += result.timing.loadNs;
        timing.cloneNs += result.timing.cloneNs;
        timing.stepNs += result.timing.stepNs;
        timing.hashNs += result.timing.hashNs;
        timing.queueNs += result.timing.queueNs;
        timing.frontierPopNs += result.timing.frontierPopNs;
        timing.frontierPushNs += result.timing.frontierPushNs;
        timing.visitedLookupNs += result.timing.visitedLookupNs;
        timing.visitedInsertNs += result.timing.visitedInsertNs;
        timing.visitedLookupProbes += result.timing.visitedLookupProbes;
        timing.visitedInsertProbes += result.timing.visitedInsertProbes;
        timing.visitedGrows += result.timing.visitedGrows;
        timing.visitedCapacity = std::max(timing.visitedCapacity, result.timing.visitedCapacity);
        timing.visitedMaxProbe = std::max(timing.visitedMaxProbe, result.timing.visitedMaxProbe);
        timing.visitedKeyCollisions += result.timing.visitedKeyCollisions;
        timing.compactStateBytes += result.timing.compactStateBytes;
        timing.compactMaxStateBytes = std::max(timing.compactMaxStateBytes, result.timing.compactMaxStateBytes);
        timing.nodeStoreNs += result.timing.nodeStoreNs;
        timing.heuristicNs += result.timing.heuristicNs;
        timing.solvedCheckNs += result.timing.solvedCheckNs;
        timing.timeoutCheckNs += result.timing.timeoutCheckNs;
        timing.reconstructNs += result.timing.reconstructNs;
    }

    std::cout << "{\n  \"results\":[\n";
    for (size_t index = 0; index < results.size(); ++index) {
        std::cout << "    ";
        printJsonResult(results[index], std::cout);
        std::cout << (index + 1 == results.size() ? "\n" : ",\n");
    }
    std::cout << "  ],\n  \"totals\":{";
    std::cout << "\"levels\":" << results.size();
    std::cout << ",\"solved\":" << solved;
    std::cout << ",\"timeout\":" << timeout;
    std::cout << ",\"exhausted\":" << exhausted;
    std::cout << ",\"skipped_message\":" << skipped;
    std::cout << ",\"errors\":" << errors;
    std::cout << ",\"expanded\":" << expanded;
    std::cout << ",\"generated\":" << generated;
    std::cout << ",\"compact_tick_attempts\":" << compactTickAttempts;
    std::cout << ",\"compact_tick_hits\":" << compactTickHits;
    std::cout << ",\"compact_tick_fallbacks\":" << compactTickFallbacks;
    std::cout << ",\"compact_tick_unsupported\":" << compactTickUnsupported;
    std::cout << ",\"compact_tick_oracle_checks\":" << compactTickOracleChecks;
    std::cout << ",\"compact_tick_oracle_failures\":" << compactTickOracleFailures;
    std::cout << ",\"compile_ms\":" << ms(timing.compileNs);
    std::cout << ",\"load_ms\":" << ms(timing.loadNs);
    std::cout << ",\"clone_ms\":" << ms(timing.cloneNs);
    std::cout << ",\"step_ms\":" << ms(timing.stepNs);
    std::cout << ",\"hash_ms\":" << ms(timing.hashNs);
    std::cout << ",\"queue_ms\":" << ms(timing.queueNs);
    std::cout << ",\"frontier_pop_ms\":" << ms(timing.frontierPopNs);
    std::cout << ",\"frontier_push_ms\":" << ms(timing.frontierPushNs);
    std::cout << ",\"visited_lookup_ms\":" << ms(timing.visitedLookupNs);
    std::cout << ",\"visited_insert_ms\":" << ms(timing.visitedInsertNs);
    std::cout << ",\"visited_lookup_probes\":" << timing.visitedLookupProbes;
    std::cout << ",\"visited_insert_probes\":" << timing.visitedInsertProbes;
    std::cout << ",\"visited_grows\":" << timing.visitedGrows;
    std::cout << ",\"visited_capacity\":" << timing.visitedCapacity;
    std::cout << ",\"visited_max_probe\":" << timing.visitedMaxProbe;
    std::cout << ",\"visited_key_collisions\":" << timing.visitedKeyCollisions;
    std::cout << ",\"compact_state_bytes\":" << timing.compactStateBytes;
    std::cout << ",\"compact_max_state_bytes\":" << timing.compactMaxStateBytes;
    std::cout << ",\"node_store_ms\":" << ms(timing.nodeStoreNs);
    std::cout << ",\"heuristic_ms\":" << ms(timing.heuristicNs);
    std::cout << ",\"solved_check_ms\":" << ms(timing.solvedCheckNs);
    std::cout << ",\"timeout_check_ms\":" << ms(timing.timeoutCheckNs);
    std::cout << ",\"reconstruct_ms\":" << ms(timing.reconstructNs);
    const int64_t totalElapsedNs = std::accumulate(results.begin(), results.end(), int64_t{0}, [](int64_t total, const Result& result) {
        return total + result.elapsedMs * 1000000;
    });
    std::cout << ",\"unattributed_ms\":" << ms(std::max<int64_t>(0, totalElapsedNs - measuredSearchNs(timing)));
    std::cout << "}\n}\n";
}

HumanSummary summarizeHuman(const std::vector<Result>& results, size_t begin, size_t end) {
    HumanSummary summary;
    for (size_t index = begin; index < end; ++index) {
        const Result& result = results[index];
        summary.solved += result.status == "solved";
        summary.timeout += result.status == "timeout";
        summary.exhausted += result.status == "exhausted";
        summary.skipped += result.status == "skipped_message";
        summary.errors += result.status == "compile_error" || result.status == "level_error";
        summary.expanded += result.expanded;
        summary.generated += result.generated;
    }
    return summary;
}

HumanSummary summarizeHuman(const std::vector<Result>& results) {
    return summarizeHuman(results, 0, results.size());
}

void printHumanBlock(std::ostream& out, std::string_view label, const HumanSummary& summary, int64_t elapsedMs) {
    out << "===\n";
    out << label << " (" << secondsString(elapsedMs) << " sec)\n";
    out << "Levels Solved: " << summary.solved << "/" << summary.playableLevels() << "\n";
    out << "Timeout: " << summary.timeout << "\n";
    if (summary.exhausted > 0) {
        out << "Unsolvable: " << summary.exhausted << "\n";
    }
    if (summary.errors > 0) {
        out << "Errors: " << summary.errors << "\n";
    }
}

void printSolutionsLocation(std::ostream& out, const Options& options) {
    if (options.writeSolutions) {
        out << "Solutions: " << options.solutionsDir.generic_string() << "\n";
    } else {
        out << "Solutions: disabled\n";
    }
}

void printHuman(const std::vector<Result>& results, const Options& options) {
    uint64_t solved = 0;
    uint64_t timeout = 0;
    uint64_t exhausted = 0;
    uint64_t skipped = 0;
    uint64_t errors = 0;
    for (const auto& result : results) {
        solved += result.status == "solved";
        timeout += result.status == "timeout";
        exhausted += result.status == "exhausted";
        skipped += result.status == "skipped_message";
        errors += result.status == "compile_error" || result.status == "level_error";
        std::cout << result.game << " level=" << result.level
                  << " status=" << result.status
                  << " strategy=" << result.strategy
                  << " solution_length=" << result.solution.size()
                  << " elapsed_ms=" << result.elapsedMs
                  << " expanded=" << result.expanded
                  << " generated=" << result.generated
                  << " unique_states=" << result.uniqueStates;
        if (!result.solution.empty()) {
            std::cout << " solution=";
            for (size_t index = 0; index < result.solution.size(); ++index) {
                if (index > 0) {
                    std::cout << ",";
                }
                std::cout << result.solution[index];
            }
        }
        if (!result.error.empty()) {
            std::cout << " error=" << result.error;
        }
        std::cout << "\n";
    }
    std::cout << "solver_totals levels=" << results.size()
              << " solved=" << solved
              << " timeout=" << timeout
              << " exhausted=" << exhausted
              << " skipped_message=" << skipped
              << " errors=" << errors << "\n";
    printSolutionsLocation(std::cout, options);
}

void printHumanSummary(const std::vector<Result>& results, const Options& options) {
    int64_t elapsedMs = 0;
    for (const auto& result : results) {
        elapsedMs += result.elapsedMs;
    }
    printHumanBlock(std::cout, "Totals", summarizeHuman(results), elapsedMs);
    printSolutionsLocation(std::cout, options);
}

std::vector<Result> runCorpus(const Options& options) {
    std::vector<Result> results;
    std::vector<CompiledGame> compiledGames;
    std::vector<WorkItem> workItems;
    const auto games = discoverGames(options.corpusPath);
    for (const auto& gamePath : games) {
        const std::string gameName = relativeGameName(options.corpusPath, gamePath);
        if (!matchesGameFilter(gameName, options.gameFilter)) {
            continue;
        }

        CompiledGame compiled;
        compiled.path = gamePath;
        compiled.name = gameName;
        compiled.resultBegin = results.size();

        if (!options.quiet && !options.progressPerGame) {
            std::cerr << "solver_progress game=" << gameName << " phase=compile\n";
        }
        compiled.source = readFile(gamePath);
        if (compiled.source.empty() || compiled.source.back() != '\n') {
            compiled.source.push_back('\n');
        }

        std::string compileError;
        {
            ScopedTimer timer(compiled.compileNs);
            compiled.game = compileGame(compiled.source, compileError);
        }
        if (!compiled.game) {
            Result result;
            result.game = gameName;
            result.level = -1;
            result.status = "compile_error";
            result.error = compileError;
            result.strategy = strategyName(options.strategy);
            result.timeoutMs = options.timeoutMs;
            result.timing.compileNs = compiled.compileNs;
            results.push_back(std::move(result));
            compiled.resultEnd = results.size();
            if (!options.quiet && !options.progressPerGame) {
                std::cerr << "solver_progress game=" << gameName << " level=-1 status=compile_error completed="
                          << results.size() << "\n";
            }
            compiledGames.push_back(std::move(compiled));
            continue;
        }

        const int32_t levelCount = static_cast<int32_t>(compiled.game->levels.size());
        if (!options.quiet && !options.progressPerGame) {
            std::cerr << "solver_progress game=" << gameName << " phase=levels count=" << levelCount << "\n";
        }
        for (int32_t levelIndex = 0; levelIndex < levelCount; ++levelIndex) {
            if (options.levelFilter && *options.levelFilter != levelIndex) {
                continue;
            }
            if (!options.quiet && !options.progressPerGame) {
                std::cerr << "solver_progress game=" << gameName << " level=" << levelIndex << " phase=start\n";
            }
            const size_t resultIndex = results.size();
            Result placeholder;
            placeholder.game = gameName;
            placeholder.level = levelIndex;
            placeholder.status = "pending";
            placeholder.strategy = strategyName(options.strategy);
            placeholder.timeoutMs = options.timeoutMs;
            placeholder.timing.compileNs = compiled.compileNs;
            results.push_back(std::move(placeholder));
            workItems.push_back(WorkItem{compiledGames.size(), levelIndex, resultIndex});
        }
        compiled.resultEnd = results.size();
        compiledGames.push_back(std::move(compiled));
    }

    if (!workItems.empty()) {
        const size_t threadCount = std::min(options.jobs, workItems.size());
        std::atomic<size_t> nextWork{0};
        std::atomic<size_t> completed{0};
        auto worker = [&](uint32_t workerId) {
            while (true) {
                const size_t workIndex = nextWork.fetch_add(1);
                if (workIndex >= workItems.size()) {
                    break;
                }
                const WorkItem& item = workItems[workIndex];
                const CompiledGame& compiled = compiledGames[item.gameIndex];
                Result result = solveLevel(
                    compiled.game,
                    compiled.name,
                    item.levelIndex,
                    options.timeoutMs,
                    compiled.compileNs,
                    options.strategy,
                    workerId,
                    options.exactStateKeys,
                    options.compactNodeStorage,
                    options.compactTickOracle,
                    options.astarWeight
                );
                results[item.resultIndex] = std::move(result);
                const size_t done = completed.fetch_add(1) + 1;
                if (!options.quiet && !options.progressPerGame && options.progressEvery > 0 && (done % options.progressEvery) == 0) {
                    std::cerr << "solver_progress completed=" << done << " total=" << workItems.size() << "\n";
                }
            }
        };

        if (threadCount <= 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(threadCount);
            for (size_t index = 0; index < threadCount; ++index) {
                threads.emplace_back(worker, static_cast<uint32_t>(index));
            }
            for (auto& thread : threads) {
                thread.join();
            }
        }
    }

    for (const CompiledGame& compiled : compiledGames) {
        writeAnnotatedSolutions(options, compiled.name, compiled.source, results, compiled.resultBegin, compiled.resultEnd);
        if (!options.quiet && options.progressPerGame) {
            int64_t elapsedMs = 0;
            for (size_t index = compiled.resultBegin; index < compiled.resultEnd; ++index) {
                elapsedMs += results[index].elapsedMs;
            }
            printHumanBlock(std::cerr, "Game: " + compiled.name, summarizeHuman(results, compiled.resultBegin, compiled.resultEnd), elapsedMs);
        }
    }
    return results;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);
        if (options.profileRuntimeCounters) {
            ps_runtime_counters_reset();
            ps_runtime_counters_set_enabled(true);
        }
        const auto results = runCorpus(options);
        ps_runtime_counters runtimeCounters{};
        if (options.profileRuntimeCounters) {
            ps_runtime_counters_snapshot(&runtimeCounters);
            ps_runtime_counters_set_enabled(false);
            std::cerr << "solver_runtime_counters"
                      << " rules_visited=" << runtimeCounters.rules_visited
                      << " rules_skipped_by_mask=" << runtimeCounters.rules_skipped_by_mask
                      << " candidate_cells_tested=" << runtimeCounters.candidate_cells_tested
                      << " pattern_tests=" << runtimeCounters.pattern_tests
                      << " pattern_matches=" << runtimeCounters.pattern_matches
                      << " replacements_attempted=" << runtimeCounters.replacements_attempted
                      << " replacements_applied=" << runtimeCounters.replacements_applied
                      << " row_scans=" << runtimeCounters.row_scans
                      << " ellipsis_scans=" << runtimeCounters.ellipsis_scans
                      << " mask_rebuild_calls=" << runtimeCounters.mask_rebuild_calls
                      << " mask_rebuild_dirty_calls=" << runtimeCounters.mask_rebuild_dirty_calls
                      << " mask_rebuild_rows=" << runtimeCounters.mask_rebuild_rows
                      << " mask_rebuild_columns=" << runtimeCounters.mask_rebuild_columns
                      << " compiled_rule_group_attempts=" << runtimeCounters.compiled_rule_group_attempts
                      << " compiled_rule_group_hits=" << runtimeCounters.compiled_rule_group_hits
                      << " compiled_rule_group_fallbacks=" << runtimeCounters.compiled_rule_group_fallbacks
                      << " compiled_tick_attempts=" << runtimeCounters.compiled_tick_attempts
                      << " compiled_tick_hits=" << runtimeCounters.compiled_tick_hits
                      << " compiled_tick_fallbacks=" << runtimeCounters.compiled_tick_fallbacks
                      << " compact_tick_attempts=" << std::accumulate(results.begin(), results.end(), uint64_t{0}, [](uint64_t total, const Result& result) { return total + result.compactTickAttempts; })
                      << " compact_tick_hits=" << std::accumulate(results.begin(), results.end(), uint64_t{0}, [](uint64_t total, const Result& result) { return total + result.compactTickHits; })
                      << " compact_tick_fallbacks=" << std::accumulate(results.begin(), results.end(), uint64_t{0}, [](uint64_t total, const Result& result) { return total + result.compactTickFallbacks; })
                      << " compact_tick_unsupported=" << std::accumulate(results.begin(), results.end(), uint64_t{0}, [](uint64_t total, const Result& result) { return total + result.compactTickUnsupported; })
                      << " compact_tick_oracle_checks=" << std::accumulate(results.begin(), results.end(), uint64_t{0}, [](uint64_t total, const Result& result) { return total + result.compactTickOracleChecks; })
                      << " compact_tick_oracle_failures=" << std::accumulate(results.begin(), results.end(), uint64_t{0}, [](uint64_t total, const Result& result) { return total + result.compactTickOracleFailures; })
                      << "\n";
        }
        if (options.requireCompiledTick && runtimeCounters.compiled_tick_hits == 0) {
            std::cerr << "compiled tick dispatch was required but no generated tick backend handled a step\n";
            return 1;
        }
        if (options.json) {
            printJson(results);
        } else if (options.summaryOnly) {
            printHumanSummary(results, options);
        } else {
            printHuman(results, options);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
