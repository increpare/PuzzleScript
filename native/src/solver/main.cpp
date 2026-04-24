#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "puzzlescript/puzzlescript.h"

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

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
    std::unique_ptr<ps_session, decltype(&ps_session_destroy)> session{nullptr, ps_session_destroy};
    int32_t parent = -1;
    ps_input input = PS_INPUT_UP;
    uint32_t depth = 0;
};

struct QueueEntry {
    uint32_t priority = 0;
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
    std::vector<std::string> solution;
    int64_t elapsedMs = 0;
    uint64_t expanded = 0;
    uint64_t generated = 0;
    uint64_t uniqueStates = 0;
    uint64_t duplicates = 0;
    uint64_t maxFrontier = 0;
    int64_t timeoutMs = 0;
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

Options parseArgs(int argc, char** argv) {
    Options options;
    if (argc < 2) {
        throw std::runtime_error("Usage: puzzlescript_solver <solver_tests_dir> [--timeout-ms N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json]");
    }
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h") {
            throw std::runtime_error("Usage: puzzlescript_solver <solver_tests_dir> [--timeout-ms N] [--solutions-dir DIR] [--no-solutions] [--progress-every N] [--progress-per-game] [--summary-only] [--quiet] [--json]");
        }
        if (arg == "--timeout-ms" && index + 1 < argc) {
            options.timeoutMs = std::max<int64_t>(1, std::stoll(argv[++index]));
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

StateKey sessionKey(ps_session* session, Timing& timing) {
    ScopedTimer timer(timing.hashUs);
    const ps_hash128 hash = ps_session_hash128(session);
    return StateKey{hash.lo, hash.hi};
}

std::string compileErrorMessage(ps_compile_result* result) {
    if (!result) {
        return "failed to compile source";
    }
    const ps_error* error = ps_compile_result_error(result);
    if (!error) {
        return "failed to compile source";
    }
    return ps_error_message(error);
}

std::unique_ptr<ps_game, decltype(&ps_free_game)> compileGame(
    const std::string& source,
    std::string& errorMessage
) {
    ps_compile_result* rawResult = nullptr;
    if (!ps_compile_source(source.data(), source.size(), &rawResult) || rawResult == nullptr) {
        errorMessage = compileErrorMessage(rawResult);
        if (rawResult) {
            ps_free_compile_result(rawResult);
        }
        return {nullptr, ps_free_game};
    }

    ps_game* game = const_cast<ps_game*>(ps_compile_result_game(rawResult));
    if (!game) {
        errorMessage = compileErrorMessage(rawResult);
        ps_free_compile_result(rawResult);
        return {nullptr, ps_free_game};
    }

    ps_free_compile_result(rawResult);
    return {game, ps_free_game};
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

void settleAgain(ps_session* session) {
    for (int pass = 0; pass < 500 && ps_session_pending_again(session); ++pass) {
        (void)ps_session_tick(session);
    }
}

Result solveLevel(ps_game* game, const std::string& gameName, int32_t levelIndex, int64_t timeoutMs, int64_t compileUs) {
    Result result;
    result.game = gameName;
    result.level = levelIndex;
    result.status = "exhausted";
    result.timeoutMs = timeoutMs;
    result.timing.compileUs = compileUs;

    ps_error* error = nullptr;
    ps_session* rawSession = nullptr;
    const std::string seed = "solver:" + gameName + ":" + std::to_string(levelIndex);
    {
        ScopedTimer timer(result.timing.loadUs);
        if (!ps_session_create_with_loaded_level_seed(game, seed.c_str(), &rawSession, &error)) {
            result.status = "level_error";
            result.error = error ? ps_error_message(error) : "failed to create session";
            if (error) {
                ps_free_error(error);
            }
            return result;
        }
        if (!ps_session_load_level(rawSession, levelIndex, &error)) {
            result.status = "level_error";
            result.error = error ? ps_error_message(error) : "failed to load level";
            if (error) {
                ps_free_error(error);
            }
            ps_session_destroy(rawSession);
            return result;
        }
    }

    std::unique_ptr<ps_session, decltype(&ps_session_destroy)> initial(rawSession, ps_session_destroy);
    ps_session_set_unit_testing(initial.get(), true);
    ps_session_status_info initialStatus{};
    ps_session_status(initial.get(), &initialStatus);
    if (initialStatus.mode != PS_SESSION_MODE_LEVEL) {
        result.status = "skipped_message";
        return result;
    }

    const TimePoint searchStart = Clock::now();
    const TimePoint deadline = searchStart + std::chrono::milliseconds(timeoutMs);
    std::vector<Node> nodes;
    nodes.reserve(1024);

    std::unordered_map<StateKey, uint32_t, StateKeyHash> bestDepth;
    bestDepth.emplace(sessionKey(initial.get(), result.timing), 0);
    result.uniqueStates = 1;

    nodes.push_back(Node{std::move(initial), -1, PS_INPUT_UP, 0});
    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> frontier;
    {
        ScopedTimer timer(result.timing.queueUs);
        frontier.push(QueueEntry{0, 0, 0});
    }
    result.maxFrontier = 1;

    uint64_t nextTie = 1;
    const auto inputs = solverInputs();

    while (!frontier.empty()) {
        if (Clock::now() >= deadline) {
            result.status = "timeout";
            break;
        }

        QueueEntry entry{};
        {
            ScopedTimer timer(result.timing.queueUs);
            entry = frontier.top();
            frontier.pop();
        }

        ps_session* parentSession = nodes[entry.nodeIndex].session.get();
        const uint32_t parentDepth = nodes[entry.nodeIndex].depth;
        ++result.expanded;

        for (const ps_input input : inputs) {
            if (Clock::now() >= deadline) {
                result.status = "timeout";
                break;
            }

            ps_session* rawChild = nullptr;
            {
                ScopedTimer timer(result.timing.cloneUs);
                if (!ps_session_clone(parentSession, &rawChild, &error)) {
                    result.status = "level_error";
                    result.error = error ? ps_error_message(error) : "failed to clone session";
                    if (error) {
                        ps_free_error(error);
                    }
                    break;
                }
            }

            std::unique_ptr<ps_session, decltype(&ps_session_destroy)> child(rawChild, ps_session_destroy);
            ps_step_result stepResult{};
            {
                ScopedTimer timer(result.timing.stepUs);
                stepResult = ps_session_step(child.get(), input);
                settleAgain(child.get());
            }
            ++result.generated;

            if (solvedByStep(stepResult, child.get(), levelIndex)) {
                result.status = "solved";
                result.solution = reconstructSolution(nodes, entry.nodeIndex, input, result.timing);
                result.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - searchStart).count();
                return result;
            }

            const StateKey key = sessionKey(child.get(), result.timing);
            const uint32_t childDepth = parentDepth + 1;
            const auto found = bestDepth.find(key);
            if (found != bestDepth.end() && found->second <= childDepth) {
                ++result.duplicates;
                continue;
            }
            bestDepth[key] = childDepth;
            result.uniqueStates = bestDepth.size();

            const uint32_t childIndex = static_cast<uint32_t>(nodes.size());
            nodes.push_back(Node{std::move(child), static_cast<int32_t>(entry.nodeIndex), input, childDepth});
            {
                ScopedTimer timer(result.timing.queueUs);
                frontier.push(QueueEntry{childDepth, nextTie++, childIndex});
                result.maxFrontier = std::max<uint64_t>(result.maxFrontier, frontier.size());
            }
        }

        if (result.status == "timeout" || result.status == "level_error") {
            break;
        }
    }

    result.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - searchStart).count();
    return result;
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
    const auto games = discoverGames(options.corpusPath);
    size_t attemptedLevels = 0;
    for (const auto& gamePath : games) {
        const std::string gameName = relativeGameName(options.corpusPath, gamePath);
        const size_t gameResultBegin = results.size();
        const auto gameStartedAt = Clock::now();
        if (!options.quiet && !options.progressPerGame) {
            std::cerr << "solver_progress game=" << gameName << " phase=compile\n";
        }
        std::string source = readFile(gamePath);
        if (source.empty() || source.back() != '\n') {
            source.push_back('\n');
        }

        std::string compileError;
        int64_t compileUs = 0;
        std::unique_ptr<ps_game, decltype(&ps_free_game)> game(nullptr, ps_free_game);
        {
            ScopedTimer timer(compileUs);
            game = compileGame(source, compileError);
        }
        if (!game) {
            Result result;
            result.game = gameName;
            result.level = -1;
            result.status = "compile_error";
            result.error = compileError;
            result.timeoutMs = options.timeoutMs;
            result.timing.compileUs = compileUs;
            results.push_back(std::move(result));
            if (!options.quiet && options.progressPerGame) {
                const int64_t elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - gameStartedAt).count();
                printHumanBlock(std::cerr, "Game: " + gameName, summarizeHuman(results, gameResultBegin, results.size()), elapsedMs);
            } else if (!options.quiet) {
                std::cerr << "solver_progress game=" << gameName << " level=-1 status=compile_error completed="
                          << results.size() << "\n";
            }
            continue;
        }

        const int32_t levelCount = ps_game_level_count(game.get());
        if (!options.quiet && !options.progressPerGame) {
            std::cerr << "solver_progress game=" << gameName << " phase=levels count=" << levelCount << "\n";
        }
        for (int32_t levelIndex = 0; levelIndex < levelCount; ++levelIndex) {
            if (!options.quiet && !options.progressPerGame) {
                std::cerr << "solver_progress game=" << gameName << " level=" << levelIndex << " phase=start\n";
            }
            results.push_back(solveLevel(game.get(), gameName, levelIndex, options.timeoutMs, compileUs));
            ++attemptedLevels;
            const Result& last = results.back();
            if (!options.quiet && !options.progressPerGame && options.progressEvery > 0 && (attemptedLevels % options.progressEvery) == 0) {
                std::cerr << "solver_progress game=" << gameName
                          << " level=" << levelIndex
                          << " status=" << last.status
                          << " solution_length=" << last.solution.size()
                          << " elapsed_ms=" << last.elapsedMs
                          << " expanded=" << last.expanded
                          << " generated=" << last.generated
                          << " completed=" << attemptedLevels
                          << "\n";
            }
        }
        writeAnnotatedSolutions(options, gameName, source, results, gameResultBegin, results.size());
        if (!options.quiet && options.progressPerGame) {
            const int64_t elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - gameStartedAt).count();
            printHumanBlock(std::cerr, "Game: " + gameName, summarizeHuman(results, gameResultBegin, results.size()), elapsedMs);
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
