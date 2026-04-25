#include <algorithm>
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
using puzzlescript::Session;
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
};

struct Node {
    std::unique_ptr<Session> session;
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
    if (argc < 2) {
        throw std::runtime_error("Usage: puzzlescript_solver <solver_tests_dir> [--timeout-ms N] [--jobs auto|N|1] [--strategy portfolio|bfs|weighted-astar|greedy] [--timing none|summary|detailed] [--game NAME] [--level N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json] [--profile-runtime-counters] [--require-compiled-tick] [--hash-state-keys]");
    }
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("Usage: puzzlescript_solver <solver_tests_dir> [--timeout-ms N] [--jobs auto|N|1] [--strategy portfolio|bfs|weighted-astar|greedy] [--timing none|summary|detailed] [--game NAME] [--level N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json] [--profile-runtime-counters] [--require-compiled-tick] [--hash-state-keys]");
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

bool gameUsesRandomnessInRules(const std::vector<std::vector<puzzlescript::Rule>>& groups) {
    for (const auto& group : groups) {
        for (const auto& rule : group) {
            if (rule.isRandom) {
                return true;
            }
            for (const auto& patternRow : rule.patterns) {
                for (const auto& pattern : patternRow) {
                    if (!pattern.replacement) {
                        continue;
                    }
                    if (pattern.replacement->hasRandomEntityMask || pattern.replacement->hasRandomDirMask) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool gameUsesRandomness(const Game& game) {
    return gameUsesRandomnessInRules(game.rules) || gameUsesRandomnessInRules(game.lateRules);
}

StateKey solverStateKey(const Session& session, bool includeRandomState, Timing& timing) {
    ScopedTimer timer(timing.hashNs);
    return puzzlescript::search::sessionStateKey(session, includeRandomState);
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

int32_t heuristicScore(const Session& session) {
    puzzlescript::search::HeuristicOptions options;
    options.includeNoQuantifierPenalty = true;
    options.includePlayerDistance = true;
    return puzzlescript::search::winConditionHeuristicScore(session, options);
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

bool solverStatesEqual(const Session& lhs, const Session& rhs, bool includeRandomState) {
    if (lhs.preparedSession.currentLevelIndex != rhs.preparedSession.currentLevelIndex
        || lhs.preparedSession.titleScreen != rhs.preparedSession.titleScreen
        || lhs.preparedSession.textMode != rhs.preparedSession.textMode
        || lhs.preparedSession.winning != rhs.preparedSession.winning
        || lhs.pendingAgain != rhs.pendingAgain
        || lhs.liveLevel.width != rhs.liveLevel.width
        || lhs.liveLevel.height != rhs.liveLevel.height
        || lhs.liveLevel.objects != rhs.liveLevel.objects
        || lhs.liveMovements != rhs.liveMovements
        || lhs.preparedSession.oldFlickscreenDat != rhs.preparedSession.oldFlickscreenDat
        || lhs.preparedSession.restart.width != rhs.preparedSession.restart.width
        || lhs.preparedSession.restart.height != rhs.preparedSession.restart.height
        || lhs.preparedSession.restart.objects != rhs.preparedSession.restart.objects
        || lhs.preparedSession.restart.oldFlickscreenDat != rhs.preparedSession.restart.oldFlickscreenDat) {
        return false;
    }
    if (!includeRandomState) {
        return true;
    }
    return lhs.randomState.i == rhs.randomState.i
        && lhs.randomState.j == rhs.randomState.j
        && lhs.randomState.valid == rhs.randomState.valid
        && lhs.randomState.s == rhs.randomState.s;
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
        const Session& session,
        const std::vector<Node>& nodes,
        bool includeRandomState
    ) {
        if (entries.empty()) {
            return std::nullopt;
        }
        size_t probes = 0;
        const size_t slot = findSlot(key, session, nodes, includeRandomState, probes);
        recordLookup(probes);
        if (!entries[slot].occupied) {
            return std::nullopt;
        }
        return entries[slot].depth;
    }

    bool insertOrAssignIfBetter(
        const StateKey& key,
        const Session& session,
        uint32_t depth,
        uint32_t nodeIndex,
        const std::vector<Node>& nodes,
        bool includeRandomState
    ) {
        ensureCapacityForInsert();
        size_t probes = 0;
        const size_t slot = findSlot(key, session, nodes, includeRandomState, probes);
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
        const Session& session,
        const std::vector<Node>& nodes,
        bool includeRandomState,
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
                const Session& existing = *nodes[entry.nodeIndex].session;
                if (solverStatesEqual(existing, session, includeRandomState)) {
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
    bool exactStateKeys
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

    std::vector<Node> nodes;
    nodes.reserve(8192);

    const bool includeRandomStateInKey = gameUsesRandomness(*game);
    FlatBestDepth bestDepth(result.timing, exactStateKeys);
    bestDepth.reserve(16384);
    result.uniqueStates = 1;

    int32_t initialHeuristic = 0;
    if (mode != SearchMode::Bfs) {
        ScopedTimer timer(result.timing.heuristicNs);
        initialHeuristic = heuristicScore(*initial);
    }
    const StateKey initialKey = solverStateKey(*initial, includeRandomStateInKey, result.timing);
    {
        ScopedTimer timer(result.timing.nodeStoreNs);
        nodes.push_back(Node{std::move(initial), initialKey, -1, PS_INPUT_UP, 0, initialHeuristic});
    }
    {
        ScopedTimer timer(result.timing.visitedInsertNs);
        const Session& initialSession = *nodes[0].session;
        bestDepth.insertOrAssignIfBetter(initialKey, initialSession, 0, 0, nodes, includeRandomStateInKey);
    }
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> frontier;
    {
        ScopedTimer timer(result.timing.frontierPushNs);
        frontier.push(QueueEntry{priorityFor(mode, 0, initialHeuristic), 0, 0});
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
            best = bestDepth.find(parentNode.key, *parentNode.session, nodes, includeRandomStateInKey);
        }
        if (best && *best < parentNode.depth) {
            ++result.duplicates;
            continue;
        }

        const Session& parentSession = *parentNode.session;
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

            std::unique_ptr<Session> child;
            {
                ScopedTimer timer(result.timing.cloneNs);
                child = std::make_unique<Session>(parentSession);
            }

            ps_step_result stepResult{};
            {
                ScopedTimer timer(result.timing.stepNs);
                constexpr puzzlescript::RuntimeStepOptions solverStepOptions{
                    .playableUndo = false,
                    .emitAudio = false,
                };
                stepResult = puzzlescript::step(*child, input, solverStepOptions);
                puzzlescript::settlePendingAgain(*child, solverStepOptions);
            }
            ++result.generated;

            bool solved = false;
            {
                ScopedTimer timer(result.timing.solvedCheckNs);
                solved = solvedByStep(stepResult, *child, levelIndex);
            }
            if (solved) {
                result.status = "solved";
                result.solution = reconstructSolution(nodes, entry.nodeIndex, input, result.timing);
                return result;
            }
            if (!stepResult.changed) {
                continue;
            }

            const StateKey key = solverStateKey(*child, includeRandomStateInKey, result.timing);
            const uint32_t childDepth = parentDepth + 1;
            uint32_t childIndex = static_cast<uint32_t>(nodes.size());
            int32_t childHeuristic = 0;
            if (exactStateKeys) {
                {
                    ScopedTimer timer(result.timing.nodeStoreNs);
                    nodes.push_back(Node{std::move(child), key, static_cast<int32_t>(entry.nodeIndex), input, childDepth, 0});
                }
                bool shouldStore = false;
                {
                    ScopedTimer timer(result.timing.visitedInsertNs);
                    const Session& childSession = *nodes[childIndex].session;
                    shouldStore = bestDepth.insertOrAssignIfBetter(key, childSession, childDepth, childIndex, nodes, includeRandomStateInKey);
                    result.uniqueStates = bestDepth.size();
                }
                if (!shouldStore) {
                    {
                        ScopedTimer timer(result.timing.nodeStoreNs);
                        nodes.pop_back();
                    }
                    ++result.duplicates;
                    continue;
                }
                if (mode != SearchMode::Bfs) {
                    ScopedTimer timer(result.timing.heuristicNs);
                    childHeuristic = heuristicScore(*nodes[childIndex].session);
                    nodes[childIndex].heuristic = childHeuristic;
                }
            } else {
                bool shouldStore = false;
                {
                    ScopedTimer timer(result.timing.visitedInsertNs);
                    shouldStore = bestDepth.insertOrAssignIfBetter(key, *child, childDepth, 0, nodes, includeRandomStateInKey);
                    result.uniqueStates = bestDepth.size();
                }
                if (!shouldStore) {
                    ++result.duplicates;
                    continue;
                }
                if (mode != SearchMode::Bfs) {
                    ScopedTimer timer(result.timing.heuristicNs);
                    childHeuristic = heuristicScore(*child);
                }
                childIndex = static_cast<uint32_t>(nodes.size());
                {
                    ScopedTimer timer(result.timing.nodeStoreNs);
                    nodes.push_back(Node{std::move(child), key, static_cast<int32_t>(entry.nodeIndex), input, childDepth, childHeuristic});
                }
            }
            {
                ScopedTimer timer(result.timing.frontierPushNs);
                frontier.push(QueueEntry{priorityFor(mode, childDepth, childHeuristic), nextTie++, childIndex});
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
}

Result solveLevel(
    const std::shared_ptr<const Game>& game,
    const std::string& gameName,
    int32_t levelIndex,
    int64_t timeoutMs,
    int64_t compileNs,
    Strategy strategy,
    uint32_t workerId,
    bool exactStateKeys
) {
    const TimePoint searchStart = Clock::now();
    const TimePoint deadline = searchStart + std::chrono::milliseconds(timeoutMs);

    auto finish = [&](Result result) {
        result.strategy = result.status == "solved" ? result.strategy : strategyName(strategy);
        result.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - searchStart).count();
        return result;
    };

    if (strategy == Strategy::Bfs) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::Bfs, deadline, workerId, exactStateKeys));
    }
    if (strategy == Strategy::WeightedAStar) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::WeightedAStar, deadline, workerId, exactStateKeys));
    }
    if (strategy == Strategy::Greedy) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::Greedy, deadline, workerId, exactStateKeys));
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
    combined.timing.compileNs = compileNs;

    const TimePoint bfsDeadline = searchStart + std::chrono::milliseconds(std::max<int64_t>(1, timeoutMs / 6));
    Result bfs = runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::Bfs, std::min(bfsDeadline, deadline), workerId, exactStateKeys);
    mergeStats(combined, bfs);
    if (bfs.status == "solved" || bfs.status == "skipped_message" || bfs.status == "level_error") {
        bfs.strategy = bfs.status == "solved" ? "bfs" : "portfolio";
        return finish(bfs);
    }

    if (Clock::now() < deadline) {
        Result weighted = runSearch(game, gameName, levelIndex, timeoutMs, compileNs, SearchMode::WeightedAStar, deadline, workerId, exactStateKeys);
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
    for (const auto& result : results) {
        solved += result.status == "solved";
        timeout += result.status == "timeout";
        exhausted += result.status == "exhausted";
        skipped += result.status == "skipped_message";
        errors += result.status == "compile_error" || result.status == "level_error";
        expanded += result.expanded;
        generated += result.generated;
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
                    options.exactStateKeys
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
