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
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "compiler/diagnostic.hpp"
#include "compiler/lower_to_runtime.hpp"
#include "compiler/parser.hpp"
#include "puzzlescript/puzzlescript.h"
#include "runtime/core.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using puzzlescript::Game;
using puzzlescript::MaskOffset;
using puzzlescript::MaskWord;
using puzzlescript::MaskWordUnsigned;
using puzzlescript::Session;

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

enum class SearchMode {
    Bfs,
    WeightedAStar,
    Greedy,
};

struct ScopedTimer {
    explicit ScopedTimer(int64_t& target)
        : target(target), start(Clock::now()) {}

    ~ScopedTimer() {
        target += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
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

struct Node {
    std::unique_ptr<Session> session;
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
    int64_t compileUs = 0;
    int64_t loadUs = 0;
    int64_t cloneUs = 0;
    int64_t stepUs = 0;
    int64_t hashUs = 0;
    int64_t queueUs = 0;
    int64_t reconstructUs = 0;
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
    int64_t compileUs = 0;
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

double ms(int64_t us) {
    return static_cast<double>(us) / 1000.0;
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

std::vector<ps_input> solverInputs() {
    return {
        PS_INPUT_UP,
        PS_INPUT_LEFT,
        PS_INPUT_DOWN,
        PS_INPUT_RIGHT,
        PS_INPUT_ACTION,
    };
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
        throw std::runtime_error("Usage: puzzlescript_solver <solver_tests_dir> [--timeout-ms N] [--jobs auto|N|1] [--strategy portfolio|bfs|weighted-astar|greedy] [--timing none|summary|detailed] [--game NAME] [--level N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json]");
    }
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("Usage: puzzlescript_solver <solver_tests_dir> [--timeout-ms N] [--jobs auto|N|1] [--strategy portfolio|bfs|weighted-astar|greedy] [--timing none|summary|detailed] [--game NAME] [--level N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json]");
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

StateKey sessionKey(const Session& session, Timing& timing) {
    ScopedTimer timer(timing.hashUs);
    const ps_hash128 hash = puzzlescript::hashSession128(session);
    return StateKey{hash.lo, hash.hi};
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
        return game;
    } catch (const std::exception& error) {
        errorMessage = error.what();
        return nullptr;
    }
}

std::vector<std::string> reconstructSolution(const std::vector<Node>& nodes, uint32_t nodeIndex, ps_input finalInput, Timing& timing) {
    ScopedTimer timer(timing.reconstructUs);
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

const MaskWord* maskPtr(const Game& game, MaskOffset offset) {
    if (offset == puzzlescript::kNullMaskOffset || offset >= game.maskArena.size()) {
        return nullptr;
    }
    return game.maskArena.data() + offset;
}

const MaskWord* cellObjects(const Session& session, int32_t tileIndex) {
    return session.liveLevel.objects.data() + static_cast<size_t>(tileIndex * session.game->strideObject);
}

bool anyBits(const MaskWord* lhs, uint32_t lhsCount, const MaskWord* rhs, uint32_t rhsCount) {
    const uint32_t count = std::min(lhsCount, rhsCount);
    for (uint32_t index = 0; index < count; ++index) {
        if ((lhs[index] & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}

bool bitsSet(const MaskWord* required, uint32_t requiredCount, const MaskWord* actual, uint32_t actualCount) {
    for (uint32_t index = 0; index < requiredCount; ++index) {
        const MaskWord actualWord = index < actualCount ? actual[index] : 0;
        if ((required[index] & actualWord) != required[index]) {
            return false;
        }
    }
    return true;
}

bool matchesFilter(const MaskWord* filter, uint32_t wordCount, bool aggregate, const MaskWord* cell) {
    if (filter == nullptr) {
        return false;
    }
    return aggregate ? bitsSet(filter, wordCount, cell, wordCount) : anyBits(filter, wordCount, cell, wordCount);
}

int32_t tileX(const Session& session, int32_t tileIndex) {
    return tileIndex / session.liveLevel.height;
}

int32_t tileY(const Session& session, int32_t tileIndex) {
    return tileIndex % session.liveLevel.height;
}

int32_t manhattan(const Session& session, int32_t a, int32_t b) {
    return std::abs(tileX(session, a) - tileX(session, b)) + std::abs(tileY(session, a) - tileY(session, b));
}

std::vector<int32_t> matchingTiles(const Session& session, const MaskWord* filter, bool aggregate) {
    std::vector<int32_t> out;
    if (filter == nullptr) {
        return out;
    }
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    out.reserve(static_cast<size_t>(tileCount));
    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        if (matchesFilter(filter, session.game->wordCount, aggregate, cellObjects(session, tileIndex))) {
            out.push_back(tileIndex);
        }
    }
    return out;
}

int32_t nearestDistance(const Session& session, int32_t tile, const std::vector<int32_t>& targets) {
    if (targets.empty()) {
        return 64;
    }
    int32_t best = std::numeric_limits<int32_t>::max();
    for (const int32_t target : targets) {
        best = std::min(best, manhattan(session, tile, target));
    }
    return best == std::numeric_limits<int32_t>::max() ? 64 : best;
}

int32_t heuristicScore(const Session& session) {
    const Game& game = *session.game;
    if (game.winConditions.empty()) {
        return 0;
    }

    int32_t score = 0;
    for (const auto& condition : game.winConditions) {
        const MaskWord* filter1 = maskPtr(game, condition.filter1);
        const MaskWord* filter2 = maskPtr(game, condition.filter2);
        if (filter1 == nullptr || filter2 == nullptr) {
            continue;
        }
        const auto first = matchingTiles(session, filter1, condition.aggr1);
        const auto second = matchingTiles(session, filter2, condition.aggr2);
        if (condition.quantifier == 1) {
            for (const int32_t tile : first) {
                if (matchesFilter(filter2, game.wordCount, condition.aggr2, cellObjects(session, tile))) {
                    continue;
                }
                score += 10 + nearestDistance(session, tile, second);
            }
        } else if (condition.quantifier == 0) {
            bool passed = false;
            for (const int32_t tile : first) {
                if (matchesFilter(filter2, game.wordCount, condition.aggr2, cellObjects(session, tile))) {
                    passed = true;
                    break;
                }
            }
            if (!passed) {
                int32_t best = 64;
                for (const int32_t tile : first) {
                    best = std::min(best, nearestDistance(session, tile, second));
                }
                score += best;
            }
        } else if (condition.quantifier == -1) {
            for (const int32_t tile : first) {
                if (matchesFilter(filter2, game.wordCount, condition.aggr2, cellObjects(session, tile))) {
                    score += 10;
                }
            }
        }
    }

    if (game.playerMask != puzzlescript::kNullMaskOffset && score > 0) {
        const MaskWord* playerMask = maskPtr(game, game.playerMask);
        const auto players = matchingTiles(session, playerMask, game.playerMaskAggregate);
        if (!players.empty()) {
            int32_t best = 64;
            for (const auto& condition : game.winConditions) {
                const MaskWord* filter1 = maskPtr(game, condition.filter1);
                const auto relevant = matchingTiles(session, filter1, condition.aggr1);
                for (const int32_t player : players) {
                    best = std::min(best, nearestDistance(session, player, relevant));
                }
            }
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

int32_t priorityFor(SearchMode mode, uint32_t depth, int32_t heuristic) {
    switch (mode) {
        case SearchMode::Bfs: return static_cast<int32_t>(depth);
        case SearchMode::WeightedAStar: return static_cast<int32_t>(depth) + heuristic * 4;
        case SearchMode::Greedy: return heuristic;
    }
    return static_cast<int32_t>(depth);
}

Result runSearch(
    const std::shared_ptr<const Game>& game,
    const std::string& gameName,
    int32_t levelIndex,
    int64_t timeoutMs,
    int64_t compileUs,
    SearchMode mode,
    TimePoint deadline,
    uint32_t workerId
) {
    Result result;
    result.game = gameName;
    result.level = levelIndex;
    result.status = "exhausted";
    result.strategy = searchModeName(mode);
    result.heuristic = mode == SearchMode::Bfs ? "zero" : "winconditions";
    result.timeoutMs = timeoutMs;
    result.workerId = workerId;
    result.timing.compileUs = compileUs;

    std::unique_ptr<Session> initial;
    {
        ScopedTimer timer(result.timing.loadUs);
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

    std::unordered_map<StateKey, uint32_t, StateKeyHash> bestDepth;
    bestDepth.reserve(16384);
    bestDepth.emplace(sessionKey(*initial, result.timing), 0);
    result.uniqueStates = 1;

    const int32_t initialHeuristic = mode == SearchMode::Bfs ? 0 : heuristicScore(*initial);
    nodes.push_back(Node{std::move(initial), -1, PS_INPUT_UP, 0, initialHeuristic});
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> frontier;
    frontier.push(QueueEntry{priorityFor(mode, 0, initialHeuristic), 0, 0});
    result.maxFrontier = 1;

    uint64_t nextTie = 1;
    const auto inputs = solverInputs();

    while (!frontier.empty()) {
        if (Clock::now() >= deadline) {
            result.status = "timeout";
            break;
        }

        const QueueEntry entry = frontier.top();
        frontier.pop();

        const Session& parentSession = *nodes[entry.nodeIndex].session;
        const uint32_t parentDepth = nodes[entry.nodeIndex].depth;
        ++result.expanded;

        for (const ps_input input : inputs) {
            if (Clock::now() >= deadline) {
                result.status = "timeout";
                break;
            }

            std::unique_ptr<Session> child;
            {
                ScopedTimer timer(result.timing.cloneUs);
                child = std::make_unique<Session>(parentSession);
            }

            ps_step_result stepResult{};
            {
                ScopedTimer timer(result.timing.stepUs);
                stepResult = puzzlescript::step(*child, input);
                puzzlescript::settlePendingAgain(*child);
            }
            ++result.generated;

            if (solvedByStep(stepResult, *child, levelIndex)) {
                result.status = "solved";
                result.solution = reconstructSolution(nodes, entry.nodeIndex, input, result.timing);
                return result;
            }
            if (!stepResult.changed) {
                continue;
            }

            const StateKey key = sessionKey(*child, result.timing);
            const uint32_t childDepth = parentDepth + 1;
            const auto found = bestDepth.find(key);
            if (found != bestDepth.end() && found->second <= childDepth) {
                ++result.duplicates;
                continue;
            }
            bestDepth[key] = childDepth;
            result.uniqueStates = bestDepth.size();

            const int32_t childHeuristic = mode == SearchMode::Bfs ? 0 : heuristicScore(*child);
            const uint32_t childIndex = static_cast<uint32_t>(nodes.size());
            nodes.push_back(Node{std::move(child), static_cast<int32_t>(entry.nodeIndex), input, childDepth, childHeuristic});
            frontier.push(QueueEntry{priorityFor(mode, childDepth, childHeuristic), nextTie++, childIndex});
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
    target.timing.loadUs += source.timing.loadUs;
    target.timing.cloneUs += source.timing.cloneUs;
    target.timing.stepUs += source.timing.stepUs;
    target.timing.hashUs += source.timing.hashUs;
    target.timing.queueUs += source.timing.queueUs;
    target.timing.reconstructUs += source.timing.reconstructUs;
}

Result solveLevel(
    const std::shared_ptr<const Game>& game,
    const std::string& gameName,
    int32_t levelIndex,
    int64_t timeoutMs,
    int64_t compileUs,
    Strategy strategy,
    uint32_t workerId
) {
    const TimePoint searchStart = Clock::now();
    const TimePoint deadline = searchStart + std::chrono::milliseconds(timeoutMs);

    auto finish = [&](Result result) {
        result.strategy = result.status == "solved" ? result.strategy : strategyName(strategy);
        result.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - searchStart).count();
        return result;
    };

    if (strategy == Strategy::Bfs) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileUs, SearchMode::Bfs, deadline, workerId));
    }
    if (strategy == Strategy::WeightedAStar) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileUs, SearchMode::WeightedAStar, deadline, workerId));
    }
    if (strategy == Strategy::Greedy) {
        return finish(runSearch(game, gameName, levelIndex, timeoutMs, compileUs, SearchMode::Greedy, deadline, workerId));
    }

    Result combined;
    combined.game = gameName;
    combined.level = levelIndex;
    combined.status = "timeout";
    combined.strategy = "portfolio";
    combined.heuristic = "winconditions";
    combined.timeoutMs = timeoutMs;
    combined.workerId = workerId;
    combined.timing.compileUs = compileUs;

    const TimePoint weightedDeadline = searchStart + std::chrono::milliseconds(std::max<int64_t>(1, timeoutMs * 60 / 100));
    Result weighted = runSearch(game, gameName, levelIndex, timeoutMs, compileUs, SearchMode::WeightedAStar, std::min(weightedDeadline, deadline), workerId);
    mergeStats(combined, weighted);
    if (weighted.status == "solved" || weighted.status == "skipped_message" || weighted.status == "level_error") {
        weighted.strategy = weighted.status == "solved" ? "weighted-astar" : "portfolio";
        return finish(weighted);
    }

    if (Clock::now() < deadline) {
        const TimePoint greedyDeadline = searchStart + std::chrono::milliseconds(std::max<int64_t>(1, timeoutMs * 85 / 100));
        Result greedy = runSearch(game, gameName, levelIndex, timeoutMs, compileUs, SearchMode::Greedy, std::min(greedyDeadline, deadline), workerId);
        mergeStats(combined, greedy);
        if (greedy.status == "solved" || greedy.status == "level_error") {
            greedy.strategy = greedy.status == "solved" ? "greedy" : "portfolio";
            return finish(greedy);
        }
    }

    if (Clock::now() < deadline) {
        Result bfs = runSearch(game, gameName, levelIndex, timeoutMs, compileUs, SearchMode::Bfs, deadline, workerId);
        mergeStats(combined, bfs);
        if (bfs.status == "solved" || bfs.status == "level_error") {
            bfs.strategy = bfs.status == "solved" ? "bfs" : "portfolio";
            return finish(bfs);
        }
        combined.status = bfs.status == "exhausted" ? "exhausted" : "timeout";
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
    out << ",\"compile_ms\":" << ms(result.timing.compileUs);
    out << ",\"load_ms\":" << ms(result.timing.loadUs);
    out << ",\"clone_ms\":" << ms(result.timing.cloneUs);
    out << ",\"step_ms\":" << ms(result.timing.stepUs);
    out << ",\"hash_ms\":" << ms(result.timing.hashUs);
    out << ",\"queue_ms\":" << ms(result.timing.queueUs);
    out << ",\"reconstruct_ms\":" << ms(result.timing.reconstructUs);
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
        timing.compileUs += result.timing.compileUs;
        timing.loadUs += result.timing.loadUs;
        timing.cloneUs += result.timing.cloneUs;
        timing.stepUs += result.timing.stepUs;
        timing.hashUs += result.timing.hashUs;
        timing.queueUs += result.timing.queueUs;
        timing.reconstructUs += result.timing.reconstructUs;
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
    std::cout << ",\"compile_ms\":" << ms(timing.compileUs);
    std::cout << ",\"load_ms\":" << ms(timing.loadUs);
    std::cout << ",\"clone_ms\":" << ms(timing.cloneUs);
    std::cout << ",\"step_ms\":" << ms(timing.stepUs);
    std::cout << ",\"hash_ms\":" << ms(timing.hashUs);
    std::cout << ",\"queue_ms\":" << ms(timing.queueUs);
    std::cout << ",\"reconstruct_ms\":" << ms(timing.reconstructUs);
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
            ScopedTimer timer(compiled.compileUs);
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
            result.timing.compileUs = compiled.compileUs;
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
            placeholder.timing.compileUs = compiled.compileUs;
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
                    compiled.compileUs,
                    options.strategy,
                    workerId
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
        const auto results = runCorpus(options);
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
