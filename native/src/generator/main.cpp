#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <csignal>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <unistd.h>

#include "compiler/diagnostic.hpp"
#include "compiler/lower_to_runtime.hpp"
#include "compiler/parser.hpp"
#include "compiler/rule_text.hpp"
#include "runtime/compiled_rules.hpp"
#include "runtime/core.hpp"
#include "search/search_common.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using puzzlescript::Game;
using puzzlescript::LevelTemplate;
using puzzlescript::MaskVector;
using puzzlescript::MaskWord;
using puzzlescript::MaskWordUnsigned;
using puzzlescript::Session;
using StateKey = puzzlescript::search::StateKey;
using StateKeyHash = puzzlescript::search::StateKeyHash;
using SearchMode = puzzlescript::search::SearchMode;
using puzzlescript::search::anyBits;
using puzzlescript::search::bitsSet;
using puzzlescript::search::maskPtr;
using puzzlescript::search::priorityFor;

enum class SolveStatus {
    Exhausted,
    Solved,
    Timeout,
    LevelError,
};

struct Options {
    std::filesystem::path gamePath;
    std::filesystem::path specPath;
    std::filesystem::path jsonOut;
    int64_t timeMs = 60000;
    std::optional<uint64_t> samples;
    size_t jobs = 0;
    uint64_t seed = 1;
    int64_t solverTimeoutMs = 250;
    std::optional<SearchMode> solverMode;
    size_t topK = 50;
    size_t dedupeMax = 1000000;
    bool quiet = false;
};

struct PatternTerm {
    MaskVector mask;
    bool missing = false;
    bool any = false;
};

struct ReplacementTerm {
    MaskVector clearMask;
    MaskVector setMask;
};

struct Slot {
    std::vector<PatternTerm> terms;
    std::vector<ReplacementTerm> replacements;
};

enum class Direction {
    Up,
    Down,
    Left,
    Right,
};

struct PatternGroup {
    std::vector<Slot> cells;
};

struct Alternative {
    std::vector<PatternGroup> groups;
    std::vector<Direction> directions;
    double optionProbability = 1.0;
};

struct ChooseRule {
    int32_t count = 0;
    std::vector<Alternative> alternatives;
};

struct GenerationProgram {
    std::vector<ChooseRule> rules;
};

struct Candidate {
    uint64_t score = 0;
    uint64_t uniqueStates = 0;
    uint64_t expanded = 0;
    size_t solutionLength = 0;
    uint64_t levelHash = 0;
    uint64_t sampleId = 0;
    uint64_t seed = 0;
    std::vector<std::string> solution;
    LevelTemplate level;
};

struct Counters {
    std::atomic<uint64_t> samplesAttempted{0};
    std::atomic<uint64_t> validGenerated{0};
    std::atomic<uint64_t> rejected{0};
    std::atomic<uint64_t> deduped{0};
    std::atomic<uint64_t> solverSearches{0};
    std::atomic<uint64_t> solverExpanded{0};
    std::atomic<uint64_t> solverGenerated{0};
    std::atomic<uint64_t> solverUniqueStates{0};
    std::atomic<uint64_t> solverDuplicates{0};
    std::atomic<uint64_t> solved{0};
    std::atomic<uint64_t> timeouts{0};
    std::atomic<uint64_t> exhausted{0};
    std::atomic<uint64_t> levelErrors{0};
};

struct SharedState {
    std::atomic<uint64_t> nextSample{0};
    std::atomic<bool> cancel{false};
    Counters counters;
    std::mutex topMutex;
    std::vector<Candidate> top;
    std::array<std::mutex, 64> dedupeMutexes;
    std::array<std::unordered_set<uint64_t>, 64> dedupe;
    std::array<std::deque<uint64_t>, 64> dedupeOrder;
};

struct CounterValues {
    uint64_t samplesAttempted = 0;
    uint64_t validGenerated = 0;
    uint64_t rejected = 0;
    uint64_t deduped = 0;
    uint64_t solverSearches = 0;
    uint64_t solverExpanded = 0;
    uint64_t solverGenerated = 0;
    uint64_t solverUniqueStates = 0;
    uint64_t solverDuplicates = 0;
    uint64_t solved = 0;
    uint64_t timeouts = 0;
    uint64_t exhausted = 0;
    uint64_t levelErrors = 0;
};

struct StateHashProjection {
    MaskVector objectMask;
    bool enabled = false;
};

struct SolverMetadata {
    std::vector<ps_input> inputs;
    bool includeRandomState = false;
    StateHashProjection projection;
};

struct Node {
    std::unique_ptr<Session> session;
    StateKey key;
    int32_t parent = -1;
    ps_input input = PS_INPUT_UP;
    uint32_t depth = 0;
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

struct SolveResult {
    SolveStatus status = SolveStatus::Exhausted;
    std::vector<std::string> solution;
    uint64_t expanded = 0;
    uint64_t generated = 0;
    uint64_t uniqueStates = 0;
    uint64_t duplicates = 0;
};

std::atomic<bool>* gCancelFlag = nullptr;

void handleSignal(int) {
    if (gCancelFlag != nullptr) {
        gCancelFlag->store(true, std::memory_order_relaxed);
    }
}

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
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Failed to write file: " + path.string());
    }
    stream << text;
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
    return lines;
}

std::string joinLines(const std::vector<std::string>& lines) {
    std::ostringstream out;
    for (const auto& line : lines) {
        out << line << '\n';
    }
    return out.str();
}

size_t autoJobCount() {
    const unsigned count = std::thread::hardware_concurrency();
    return std::max<size_t>(1, count == 0 ? 1 : count);
}

std::optional<SearchMode> parseSolverMode(const std::string& value) {
    if (value == "portfolio") return std::nullopt;
    if (value == "bfs") return SearchMode::Bfs;
    if (value == "weighted-astar") return SearchMode::WeightedAStar;
    if (value == "greedy") return SearchMode::Greedy;
    throw std::runtime_error("Unsupported solver strategy: " + value);
}

Options parseArgs(int argc, char** argv) {
    Options options;
    options.jobs = 1;
    if (argc < 3) {
        throw std::runtime_error("Usage: puzzlescript_generator <game.txt> <spec.gen> [--time-ms N] [--samples N] [--jobs auto|N] [--seed N] [--solver-timeout-ms N] [--solver-strategy portfolio|bfs|weighted-astar|greedy] [--top-k N] [--dedupe-max N] [--json-out PATH] [--quiet]");
    }
    options.gamePath = argv[1];
    options.specPath = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--time-ms" && index + 1 < argc) {
            options.timeMs = std::max<int64_t>(1, std::stoll(argv[++index]));
        } else if (arg == "--samples" && index + 1 < argc) {
            options.samples = static_cast<uint64_t>(std::stoull(argv[++index]));
        } else if (arg == "--jobs" && index + 1 < argc) {
            const std::string value = argv[++index];
            options.jobs = value == "auto" ? autoJobCount() : std::max<size_t>(1, std::stoull(value));
        } else if (arg == "--seed" && index + 1 < argc) {
            options.seed = static_cast<uint64_t>(std::stoull(argv[++index]));
        } else if (arg == "--solver-timeout-ms" && index + 1 < argc) {
            options.solverTimeoutMs = std::max<int64_t>(1, std::stoll(argv[++index]));
        } else if (arg == "--solver-strategy" && index + 1 < argc) {
            options.solverMode = parseSolverMode(argv[++index]);
        } else if (arg == "--top-k" && index + 1 < argc) {
            options.topK = std::max<size_t>(1, std::stoull(argv[++index]));
        } else if (arg == "--dedupe-max" && index + 1 < argc) {
            options.dedupeMax = std::max<size_t>(64, std::stoull(argv[++index]));
        } else if (arg == "--json-out" && index + 1 < argc) {
            options.jsonOut = argv[++index];
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else {
            throw std::runtime_error("Unsupported argument: " + arg);
        }
    }
    return options;
}

uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Rng {
    uint64_t state = 1;

    explicit Rng(uint64_t seed) : state(seed == 0 ? 1 : seed) {}

    uint64_t next() {
        state = splitmix64(state);
        return state;
    }

    size_t index(size_t n) {
        return n == 0 ? 0 : static_cast<size_t>(next() % n);
    }

    double unit() {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }
};

bool betterCandidate(const Candidate& a, const Candidate& b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.solutionLength != b.solutionLength) return a.solutionLength > b.solutionLength;
    if (a.expanded != b.expanded) return a.expanded > b.expanded;
    if (a.levelHash != b.levelHash) return a.levelHash < b.levelHash;
    return a.sampleId < b.sampleId;
}

MaskVector emptyMask(const Game& game) {
    return MaskVector(static_cast<size_t>(game.wordCount), 0);
}

void setMaskBit(MaskVector& words, int32_t bitIndex) {
    if (bitIndex < 0) return;
    const uint32_t word = puzzlescript::maskWordIndex(static_cast<uint32_t>(bitIndex));
    if (word >= words.size()) return;
    words[word] |= puzzlescript::maskBit(static_cast<uint32_t>(bitIndex));
}

bool maskHasBit(const MaskVector& words, int32_t bitIndex) {
    if (bitIndex < 0) return false;
    const uint32_t word = puzzlescript::maskWordIndex(static_cast<uint32_t>(bitIndex));
    return word < words.size() && (words[word] & puzzlescript::maskBit(static_cast<uint32_t>(bitIndex))) != 0;
}

void orMask(MaskVector& target, const MaskVector& source) {
    for (size_t i = 0; i < target.size() && i < source.size(); ++i) {
        target[i] |= source[i];
    }
}

void orMaskOffset(MaskVector& target, const Game& game, puzzlescript::MaskOffset offset) {
    const MaskWord* source = maskPtr(game, offset);
    if (source == nullptr) return;
    for (uint32_t i = 0; i < game.wordCount && i < target.size(); ++i) {
        target[i] |= source[i];
    }
}

std::optional<puzzlescript::MaskOffset> lookupNamedMask(const std::vector<Game::NamedMaskEntry>& table, const std::string& name) {
    auto it = std::lower_bound(table.begin(), table.end(), name,
        [](const Game::NamedMaskEntry& entry, const std::string& n) { return entry.name < n; });
    if (it == table.end() || it->name != name) return std::nullopt;
    return it->offset;
}

struct NameResolver {
    const Game& game;
    std::map<std::string, int32_t> objectIdByName;
    std::map<std::string, std::string> synonymOf;
    std::map<std::string, std::vector<std::string>> aggregateOf;
    std::map<std::string, std::vector<std::string>> propertyOf;
    std::map<std::string, MaskVector> resolved;

    NameResolver(const Game& game, const puzzlescript::compiler::ParserState& state)
        : game(game) {
        for (int32_t id = 0; id < static_cast<int32_t>(game.idDict.size()); ++id) {
            objectIdByName[game.idDict[static_cast<size_t>(id)]] = id;
        }
        for (const auto& entry : state.legendSynonyms) {
            if (!entry.items.empty()) synonymOf[lowercase(entry.name)] = lowercase(entry.items.front());
        }
        for (const auto& entry : state.legendAggregates) {
            std::vector<std::string> items;
            for (const auto& item : entry.items) items.push_back(lowercase(item));
            aggregateOf[lowercase(entry.name)] = std::move(items);
        }
        for (const auto& entry : state.legendProperties) {
            std::vector<std::string> items;
            for (const auto& item : entry.items) items.push_back(lowercase(item));
            propertyOf[lowercase(entry.name)] = std::move(items);
        }
        normalizeAliases();
    }

    void normalizeAliases() {
        bool modified = true;
        while (modified) {
            modified = false;
            for (auto& [_, value] : synonymOf) {
                if (auto it = synonymOf.find(value); it != synonymOf.end()) {
                    value = it->second;
                    modified = true;
                }
            }
            std::vector<std::string> propertyKeys;
            for (const auto& [name, _] : propertyOf) propertyKeys.push_back(name);
            for (const auto& name : propertyKeys) {
                auto& values = propertyOf[name];
                for (size_t i = 0; i < values.size(); ++i) {
                    if (auto syn = synonymOf.find(values[i]); syn != synonymOf.end()) {
                        values[i] = syn->second;
                        modified = true;
                    } else if (auto prop = propertyOf.find(values[i]); prop != propertyOf.end()) {
                        values.erase(values.begin() + static_cast<std::ptrdiff_t>(i));
                        for (const auto& item : prop->second) {
                            if (std::find(values.begin(), values.end(), item) == values.end()) values.push_back(item);
                        }
                        modified = true;
                        --i;
                    }
                }
            }
            std::vector<std::string> aggregateKeys;
            for (const auto& [name, _] : aggregateOf) aggregateKeys.push_back(name);
            for (const auto& name : aggregateKeys) {
                auto& values = aggregateOf[name];
                for (size_t i = 0; i < values.size(); ++i) {
                    if (auto syn = synonymOf.find(values[i]); syn != synonymOf.end()) {
                        values[i] = syn->second;
                        modified = true;
                    } else if (auto agg = aggregateOf.find(values[i]); agg != aggregateOf.end()) {
                        values.erase(values.begin() + static_cast<std::ptrdiff_t>(i));
                        for (const auto& item : agg->second) {
                            if (std::find(values.begin(), values.end(), item) == values.end()) values.push_back(item);
                        }
                        modified = true;
                        --i;
                    }
                }
            }
        }
    }

    MaskVector resolve(const std::string& rawName) {
        const std::string name = lowercase(rawName);
        std::set<std::string> visiting;
        return resolveInner(name, visiting);
    }

    bool isProperty(const std::string& rawName) const {
        return propertyOf.find(lowercase(rawName)) != propertyOf.end();
    }

    MaskVector resolveInner(const std::string& name, std::set<std::string>& visiting) {
        if (auto cached = resolved.find(name); cached != resolved.end()) return cached->second;
        if (!visiting.insert(name).second) {
            throw std::runtime_error("Legend cycle detected at '" + name + "'");
        }

        MaskVector mask = emptyMask(game);
        if (auto object = objectIdByName.find(name); object != objectIdByName.end()) {
            setMaskBit(mask, object->second);
        } else if (auto synonym = synonymOf.find(name); synonym != synonymOf.end()) {
            mask = resolveInner(synonym->second, visiting);
        } else if (auto aggregate = aggregateOf.find(name); aggregate != aggregateOf.end()) {
            for (const auto& item : aggregate->second) orMask(mask, resolveInner(item, visiting));
        } else if (auto property = propertyOf.find(name); property != propertyOf.end()) {
            for (const auto& item : property->second) orMask(mask, resolveInner(item, visiting));
        } else {
            throw std::runtime_error("Unknown generation rule name: " + rawDisplay(name));
        }

        visiting.erase(name);
        resolved.emplace(name, mask);
        return mask;
    }

    std::string rawDisplay(const std::string& name) const {
        return name;
    }
};

std::shared_ptr<const Game> compileGame(const std::string& source, puzzlescript::compiler::ParserState* outState = nullptr) {
    puzzlescript::compiler::DiagnosticSink diagnostics;
    auto state = puzzlescript::compiler::parseSource(source, diagnostics);
    std::shared_ptr<const Game> game;
    if (auto error = puzzlescript::compiler::lowerToRuntimeGame(state, game)) {
        throw std::runtime_error(error->message);
    }
    if (game) {
        puzzlescript::attachLinkedCompiledRules(*std::const_pointer_cast<Game>(game), source);
    }
    if (outState != nullptr) {
        *outState = std::move(state);
    }
    return game;
}

struct Spec {
    std::vector<std::string> initRows;
    std::vector<std::string> ruleLines;
};

Spec parseSpec(const std::string& text) {
    enum class Section { None, Init, Rules };
    Section section = Section::None;
    Spec spec;
    for (const auto& rawLine : splitLines(text)) {
        const std::string line = trim(rawLine);
        const std::string lowered = lowercase(line);
        if (lowered == "[ init level ]" || lowered == "[ generation rules ]") {
            throw std::runtime_error("Generation spec sections must use (INIT LEVEL) and (GENERATION RULES), not square brackets");
        }
        if (lowered == "(init level)") {
            section = Section::Init;
            continue;
        }
        if (lowered == "(generation rules)") {
            section = Section::Rules;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        if (section == Section::Init) {
            spec.initRows.push_back(line);
        } else if (section == Section::Rules) {
            spec.ruleLines.push_back(line);
        } else {
            throw std::runtime_error("Generation spec content found before (INIT LEVEL)");
        }
    }
    if (spec.initRows.empty()) throw std::runtime_error("Generation spec is missing init level rows");
    if (spec.ruleLines.empty()) throw std::runtime_error("Generation spec is missing generation rules");
    return spec;
}

std::string sourceWithInitLevel(const std::string& source, const std::vector<std::string>& rows) {
    std::vector<std::string> lines = splitLines(source);
    for (size_t i = 0; i < lines.size(); ++i) {
        if (lowercase(trim(lines[i])) != "levels") continue;
        std::vector<std::string> out(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(i + 1));
        if (i + 1 < lines.size()) {
            out.push_back(lines[i + 1]);
        } else {
            out.push_back("=======");
        }
        out.emplace_back();
        for (const auto& row : rows) out.push_back(row);
        out.emplace_back();
        return joinLines(out);
    }
    throw std::runtime_error("PuzzleScript source has no LEVELS section");
}

std::vector<std::vector<std::string>> splitAlternatives(const std::vector<std::string>& tokens, size_t begin) {
    return puzzlescript::compiler::ruletext::splitTopLevelOr(tokens, begin);
}

std::vector<std::vector<std::vector<std::string>>> parseBracketGroups(const std::vector<std::string>& tokens, size_t begin, size_t end) {
    std::vector<std::vector<std::vector<std::string>>> groups;
    for (auto row : puzzlescript::compiler::ruletext::parseBracketRows(tokens, begin, end, false)) {
        groups.push_back(std::move(row.cells));
    }
    return groups;
}

int32_t objectLayerForBit(const Game& game, int32_t objectId) {
    if (objectId < 0 || static_cast<size_t>(objectId) >= game.objectsById.size()) return -1;
    return game.objectsById[static_cast<size_t>(objectId)].layer;
}

std::vector<int32_t> objectIdsFromMask(const Game& game, const MaskVector& mask) {
    std::vector<int32_t> ids;
    for (int32_t id = 0; id < game.objectCount; ++id) {
        if (maskHasBit(mask, id)) ids.push_back(id);
    }
    return ids;
}

ReplacementTerm makeSetReplacement(const Game& game, const MaskVector& setMask) {
    ReplacementTerm term;
    term.setMask = setMask;
    term.clearMask = emptyMask(game);
    for (const int32_t objectId : objectIdsFromMask(game, setMask)) {
        const int32_t layer = objectLayerForBit(game, objectId);
        if (layer >= 0 && static_cast<size_t>(layer) < game.layerMaskOffsets.size()) {
            orMaskOffset(term.clearMask, game, game.layerMaskOffsets[static_cast<size_t>(layer)]);
        }
    }
    return term;
}

Slot compileSlot(const std::vector<std::string>& lhsCell, const std::vector<std::string>& rhsCell, const Game& game, NameResolver& resolver) {
    Slot slot;
    MaskVector matchedPresentMask = emptyMask(game);
    for (size_t i = 0; i < lhsCell.size(); ++i) {
        std::string token = lowercase(lhsCell[i]);
        if (token == "random" || token == "randomdir" || token == "late" || token == "rigid" || token == "option") {
            throw std::runtime_error("Unsupported V1 generation token in LHS: " + token);
        }
        bool missing = false;
        if (token == "no") {
            if (i + 1 >= lhsCell.size()) throw std::runtime_error("'no' must be followed by a name in generation rules");
            missing = true;
            token = lhsCell[++i];
        }
        PatternTerm term;
        term.mask = resolver.resolve(token);
        term.missing = missing;
        term.any = !missing && resolver.isProperty(token);
        if (!missing) {
            orMask(matchedPresentMask, term.mask);
        }
        slot.terms.push_back(std::move(term));
    }

    if (rhsCell.empty()) {
        ReplacementTerm term;
        term.clearMask = std::move(matchedPresentMask);
        term.setMask = emptyMask(game);
        slot.replacements.push_back(std::move(term));
        return slot;
    }

    for (size_t i = 0; i < rhsCell.size(); ++i) {
        std::string token = lowercase(rhsCell[i]);
        if (token == "random" || token == "randomdir" || token == "late" || token == "rigid" || token == "option") {
            throw std::runtime_error("Unsupported V1 generation token in RHS: " + token);
        }
        if (token == "no") {
            if (i + 1 >= rhsCell.size()) throw std::runtime_error("'no' must be followed by a name in generation rules");
            ReplacementTerm term;
            term.clearMask = resolver.resolve(rhsCell[++i]);
            term.setMask = emptyMask(game);
            slot.replacements.push_back(std::move(term));
            continue;
        }
        slot.replacements.push_back(makeSetReplacement(game, resolver.resolve(token)));
    }
    return slot;
}

Alternative compileAlternative(const std::vector<std::string>& tokens, const Game& game, NameResolver& resolver) {
    size_t cursor = 0;
    std::vector<Direction> explicitDirections;
    bool sawDirection = false;
    double optionProbability = 1.0;
    while (cursor < tokens.size() && tokens[cursor] != "[") {
        const std::string token = lowercase(tokens[cursor]);
        if (token == "up") {
            explicitDirections = {Direction::Up};
            sawDirection = true;
            ++cursor;
            continue;
        }
        if (token == "down") {
            explicitDirections = {Direction::Down};
            sawDirection = true;
            ++cursor;
            continue;
        }
        if (token == "left") {
            explicitDirections = {Direction::Left};
            sawDirection = true;
            ++cursor;
            continue;
        }
        if (token == "right") {
            explicitDirections = {Direction::Right};
            sawDirection = true;
            ++cursor;
            continue;
        }
        if (token == "horizontal") {
            explicitDirections = {Direction::Left, Direction::Right};
            sawDirection = true;
            ++cursor;
            continue;
        }
        if (token == "vertical") {
            explicitDirections = {Direction::Up, Direction::Down};
            sawDirection = true;
            ++cursor;
            continue;
        }
        if (token == "orthogonal") {
            explicitDirections = {Direction::Up, Direction::Down, Direction::Left, Direction::Right};
            sawDirection = true;
            ++cursor;
            continue;
        }
        if (token == "option") {
            if (cursor + 1 >= tokens.size()) throw std::runtime_error("option must be followed by a probability");
            optionProbability = std::stod(tokens[++cursor]);
            if (optionProbability < 0.0 || optionProbability > 1.0) {
                throw std::runtime_error("option probability must be between 0 and 1");
            }
            ++cursor;
            continue;
        }
        throw std::runtime_error("Unsupported generation rule prefix: " + tokens[cursor]);
    }
    const size_t arrowIndex = puzzlescript::compiler::ruletext::findTopLevelArrow(tokens, cursor, tokens.size());
    if (arrowIndex == tokens.size()) throw std::runtime_error("Generation rule alternative is missing ->");
    const auto lhsGroups = parseBracketGroups(tokens, cursor, arrowIndex);
    const auto rhsGroups = parseBracketGroups(tokens, arrowIndex + 1, tokens.size());
    if (lhsGroups.empty()) throw std::runtime_error("Generation rule has no LHS cells");
    if (lhsGroups.size() != rhsGroups.size()) {
        throw std::runtime_error("Generation rule LHS/RHS must have the same number of bracket groups");
    }
    Alternative alternative;
    alternative.optionProbability = optionProbability;
    bool hasDirectionalRows = false;
    for (size_t i = 0; i < lhsGroups.size(); ++i) {
        if (lhsGroups[i].size() != rhsGroups[i].size()) {
            throw std::runtime_error("Generation rule LHS/RHS row cells must have the same length");
        }
        PatternGroup group;
        for (size_t cellIndex = 0; cellIndex < lhsGroups[i].size(); ++cellIndex) {
            group.cells.push_back(compileSlot(lhsGroups[i][cellIndex], rhsGroups[i][cellIndex], game, resolver));
        }
        if (group.cells.size() > 1) hasDirectionalRows = true;
        alternative.groups.push_back(std::move(group));
    }
    if (sawDirection) {
        alternative.directions = std::move(explicitDirections);
    } else if (hasDirectionalRows) {
        alternative.directions = {Direction::Up, Direction::Down, Direction::Left, Direction::Right};
    } else {
        alternative.directions = {Direction::Right};
    }
    return alternative;
}

GenerationProgram compileGenerationProgram(const Spec& spec, const Game& game, NameResolver& resolver) {
    GenerationProgram program;
    for (const auto& line : spec.ruleLines) {
        auto tokens = puzzlescript::compiler::ruletext::tokenizeRuleLine(lowercase(line));
        if (tokens.empty()) continue;
        if (tokens[0] == "choose") {
            if (tokens.size() < 4) throw std::runtime_error("Malformed choose rule");
            ChooseRule rule;
            rule.count = std::stoi(tokens[1]);
            if (rule.count < 0) throw std::runtime_error("choose count must be non-negative");
            for (const auto& altTokens : splitAlternatives(tokens, 2)) {
                rule.alternatives.push_back(compileAlternative(altTokens, game, resolver));
            }
            program.rules.push_back(std::move(rule));
        } else if (tokens[0] == "or") {
            if (program.rules.empty()) throw std::runtime_error("or generation rule must follow a choose rule");
            for (const auto& altTokens : splitAlternatives(tokens, 1)) {
                program.rules.back().alternatives.push_back(compileAlternative(altTokens, game, resolver));
            }
        } else if (tokens[0] == "option") {
            ChooseRule rule;
            rule.count = 1;
            for (const auto& altTokens : splitAlternatives(tokens, 0)) {
                rule.alternatives.push_back(compileAlternative(altTokens, game, resolver));
            }
            program.rules.push_back(std::move(rule));
        } else {
            throw std::runtime_error("Generation rules must start with choose, or, or option");
        }
    }
    return program;
}

const MaskWord* cellPtr(const LevelTemplate& level, const Game& game, int32_t tileIndex) {
    return level.objects.data() + static_cast<size_t>(tileIndex * game.strideObject);
}

MaskWord* cellPtr(LevelTemplate& level, const Game& game, int32_t tileIndex) {
    return level.objects.data() + static_cast<size_t>(tileIndex * game.strideObject);
}

bool matchesSlot(const Slot& slot, const LevelTemplate& level, const Game& game, int32_t tileIndex) {
    const MaskWord* cell = cellPtr(level, game, tileIndex);
    for (const auto& term : slot.terms) {
        if (term.missing) {
            if (anyBits(term.mask.data(), game.wordCount, cell, game.wordCount)) return false;
        } else if (term.any) {
            if (!anyBits(term.mask.data(), game.wordCount, cell, game.wordCount)) return false;
        } else {
            if (!bitsSet(term.mask.data(), game.wordCount, cell, game.wordCount)) return false;
        }
    }
    return true;
}

void applySlot(const Slot& slot, LevelTemplate& level, const Game& game, int32_t tileIndex) {
    MaskWord* cell = cellPtr(level, game, tileIndex);
    for (const auto& replacement : slot.replacements) {
        for (uint32_t w = 0; w < game.wordCount; ++w) {
            cell[w] = (cell[w] & ~replacement.clearMask[w]) | replacement.setMask[w];
        }
    }
}

std::optional<int32_t> directedTile(const LevelTemplate& level, int32_t anchorTile, Direction direction, size_t distance) {
    const int32_t x = anchorTile / level.height;
    const int32_t y = anchorTile % level.height;
    int32_t nx = x;
    int32_t ny = y;
    const int32_t step = static_cast<int32_t>(distance);
    switch (direction) {
        case Direction::Up: ny -= step; break;
        case Direction::Down: ny += step; break;
        case Direction::Left: nx -= step; break;
        case Direction::Right: nx += step; break;
    }
    if (nx < 0 || nx >= level.width || ny < 0 || ny >= level.height) return std::nullopt;
    return nx * level.height + ny;
}

bool matchesGroup(
    const PatternGroup& group,
    const LevelTemplate& level,
    const Game& game,
    int32_t anchorTile,
    Direction direction,
    const std::vector<int32_t>& reservedTiles
) {
    for (size_t cellIndex = 0; cellIndex < group.cells.size(); ++cellIndex) {
        const auto tile = directedTile(level, anchorTile, direction, cellIndex);
        if (!tile) return false;
        if (std::find(reservedTiles.begin(), reservedTiles.end(), *tile) != reservedTiles.end()) return false;
        if (!matchesSlot(group.cells[cellIndex], level, game, *tile)) return false;
    }
    return true;
}

void applyGroup(const PatternGroup& group, LevelTemplate& level, const Game& game, int32_t anchorTile, Direction direction) {
    for (size_t cellIndex = 0; cellIndex < group.cells.size(); ++cellIndex) {
        const auto tile = directedTile(level, anchorTile, direction, cellIndex);
        if (tile) applySlot(group.cells[cellIndex], level, game, *tile);
    }
}

bool chooseGroupsForDirection(
    const Alternative& alternative,
    const LevelTemplate& level,
    const Game& game,
    Direction direction,
    Rng& rng,
    std::vector<int32_t>& candidates,
    std::vector<int32_t>& chosenAnchors,
    std::vector<int32_t>& reservedTiles
) {
    const int32_t tileCount = level.width * level.height;
    chosenAnchors.clear();
    reservedTiles.clear();
    for (const auto& group : alternative.groups) {
        candidates.clear();
        for (int32_t tile = 0; tile < tileCount; ++tile) {
            if (matchesGroup(group, level, game, tile, direction, reservedTiles)) candidates.push_back(tile);
        }
        if (candidates.empty()) return false;
        const int32_t anchor = candidates[rng.index(candidates.size())];
        chosenAnchors.push_back(anchor);
        for (size_t cellIndex = 0; cellIndex < group.cells.size(); ++cellIndex) {
            const auto tile = directedTile(level, anchor, direction, cellIndex);
            if (tile) reservedTiles.push_back(*tile);
        }
    }
    return true;
}

bool applyProgram(const GenerationProgram& program, const LevelTemplate& init, const Game& game, Rng& rng, LevelTemplate& out) {
    out = init;
    std::vector<int32_t> candidates;
    std::vector<int32_t> chosenAnchors;
    std::vector<int32_t> reservedTiles;
    for (const auto& rule : program.rules) {
        for (int32_t iteration = 0; iteration < rule.count; ++iteration) {
            if (rule.alternatives.empty()) return false;
            const Alternative& alternative = rule.alternatives[rng.index(rule.alternatives.size())];
            if (alternative.optionProbability < 1.0 && rng.unit() >= alternative.optionProbability) {
                continue;
            }
            const auto& directions = alternative.directions;
            const size_t directionCount = directions.empty() ? 0 : directions.size();
            const size_t directionStart = directionCount == 0 ? 0 : rng.index(directionCount);
            bool matched = false;
            Direction direction = Direction::Right;
            for (size_t attempt = 0; attempt < std::max<size_t>(1, directionCount); ++attempt) {
                direction = directionCount == 0
                    ? Direction::Right
                    : directions[(directionStart + attempt) % directionCount];
                if (chooseGroupsForDirection(alternative, out, game, direction, rng, candidates, chosenAnchors, reservedTiles)) {
                    matched = true;
                    break;
                }
            }
            if (!matched) return false;
            for (size_t groupIndex = 0; groupIndex < alternative.groups.size(); ++groupIndex) {
                applyGroup(alternative.groups[groupIndex], out, game, chosenAnchors[groupIndex], direction);
            }
        }
    }
    return true;
}

uint64_t hashLevel(const LevelTemplate& level) {
    uint64_t hash = 1469598103934665603ull;
    auto append = [&](uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    append(static_cast<uint64_t>(static_cast<uint32_t>(level.width)));
    append(static_cast<uint64_t>(static_cast<uint32_t>(level.height)));
    for (const MaskWord word : level.objects) {
        append(static_cast<uint64_t>(static_cast<MaskWordUnsigned>(word)));
    }
    return hash;
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
    std::vector<ps_input> inputs{PS_INPUT_RIGHT, PS_INPUT_UP, PS_INPUT_DOWN, PS_INPUT_LEFT};
    if (game.metadataMap.find("noaction") == game.metadataMap.end()) {
        inputs.push_back(PS_INPUT_ACTION);
    }
    return inputs;
}

bool gameUsesRandomnessInRules(const std::vector<std::vector<puzzlescript::Rule>>& groups) {
    for (const auto& group : groups) {
        for (const auto& rule : group) {
            if (rule.isRandom) return true;
            for (const auto& row : rule.patterns) {
                for (const auto& pattern : row) {
                    if (!pattern.replacement) continue;
                    if (pattern.replacement->hasRandomEntityMask || pattern.replacement->hasRandomDirMask) return true;
                }
            }
        }
    }
    return false;
}

bool gameUsesRandomness(const Game& game) {
    return gameUsesRandomnessInRules(game.rules) || gameUsesRandomnessInRules(game.lateRules);
}

void collectPatternReferences(MaskVector& referenced, const Game& game, const puzzlescript::Pattern& pattern) {
    orMaskOffset(referenced, game, pattern.objectsPresent);
    orMaskOffset(referenced, game, pattern.objectsMissing);
    for (uint32_t index = 0; index < pattern.anyObjectsCount; ++index) {
        const uint32_t arenaIndex = pattern.anyObjectsFirst + index;
        if (arenaIndex < game.anyObjectOffsets.size()) {
            orMaskOffset(referenced, game, game.anyObjectOffsets[arenaIndex]);
        }
    }
    if (!pattern.replacement) return;
    orMaskOffset(referenced, game, pattern.replacement->objectsClear);
    orMaskOffset(referenced, game, pattern.replacement->objectsSet);
}

void collectRuleReferences(MaskVector& referenced, const Game& game, const std::vector<std::vector<puzzlescript::Rule>>& groups) {
    for (const auto& group : groups) {
        for (const auto& rule : group) {
            orMaskOffset(referenced, game, rule.ruleMask);
            for (const auto& row : rule.patterns) {
                for (const auto& pattern : row) collectPatternReferences(referenced, game, pattern);
            }
        }
    }
}

StateHashProjection buildStateHashProjection(const Game& game) {
    StateHashProjection projection;
    projection.objectMask.assign(static_cast<size_t>(game.wordCount), 0);
    MaskVector referenced(static_cast<size_t>(game.wordCount), 0);
    collectRuleReferences(referenced, game, game.rules);
    collectRuleReferences(referenced, game, game.lateRules);
    orMaskOffset(referenced, game, game.playerMask);
    for (const auto& condition : game.winConditions) {
        orMaskOffset(referenced, game, condition.filter1);
        orMaskOffset(referenced, game, condition.filter2);
    }
    bool hasReference = false;
    for (int32_t objectId = 0; objectId < game.objectCount; ++objectId) {
        if (!maskHasBit(referenced, objectId)) continue;
        hasReference = true;
        const int32_t layer = game.objectsById[static_cast<size_t>(objectId)].layer;
        if (layer >= 0 && static_cast<size_t>(layer) < game.layerMaskOffsets.size()) {
            orMaskOffset(projection.objectMask, game, game.layerMaskOffsets[static_cast<size_t>(layer)]);
        }
    }
    if (!hasReference) return projection;
    for (int32_t objectId = 0; objectId < game.objectCount; ++objectId) {
        if (!maskHasBit(projection.objectMask, objectId)) {
            projection.enabled = true;
            break;
        }
    }
    return projection;
}

StateKey solverStateKey(const Session& session, bool includeRandomState, const StateHashProjection& projection) {
    const uint32_t stride = session.game->strideObject;
    return puzzlescript::search::sessionStateKey(session, includeRandomState, [&](size_t index, MaskWord word) {
        if (projection.enabled && stride > 0) {
            word &= projection.objectMask[index % stride];
        }
        return word;
    });
}

int32_t heuristicScore(const Session& session) {
    return puzzlescript::search::winConditionHeuristicScore(session);
}

std::vector<std::string> reconstructSolution(const std::vector<Node>& nodes, uint32_t nodeIndex, ps_input finalInput) {
    std::vector<std::string> reversed;
    reversed.push_back(inputName(finalInput));
    int32_t cursor = static_cast<int32_t>(nodeIndex);
    while (cursor >= 0) {
        const Node& node = nodes[static_cast<size_t>(cursor)];
        if (node.parent >= 0) reversed.push_back(inputName(node.input));
        cursor = node.parent;
    }
    std::reverse(reversed.begin(), reversed.end());
    return reversed;
}

bool solvedByStep(const ps_step_result& stepResult, const Session& session, int32_t levelIndex) {
    return stepResult.won || session.preparedSession.currentLevelIndex != levelIndex;
}

SolveResult runSearch(
    const std::shared_ptr<const Game>& game,
    const LevelTemplate& generatedLevel,
    const SolverMetadata& metadata,
    uint64_t sampleId,
    SearchMode mode,
    TimePoint deadline
) {
    SolveResult result;
    auto initial = puzzlescript::createSessionWithLoadedLevelSeed(game, "generator:" + std::to_string(sampleId));
    initial->suppressRuleMessages = true;
    constexpr puzzlescript::RuntimeStepOptions solverStepOptions{false, false};
    if (auto error = puzzlescript::loadLevelTemplate(*initial, generatedLevel, 0, solverStepOptions)) {
        result.status = SolveStatus::LevelError;
        return result;
    }

    std::vector<Node> nodes;
    nodes.reserve(8192);
    std::unordered_map<StateKey, uint32_t, StateKeyHash> bestDepth;
    bestDepth.reserve(16384);
    const StateKey initialKey = solverStateKey(*initial, metadata.includeRandomState, metadata.projection);
    bestDepth.emplace(initialKey, 0);
    result.uniqueStates = 1;
    const int32_t initialHeuristic = mode == SearchMode::Bfs ? 0 : heuristicScore(*initial);
    nodes.push_back(Node{std::move(initial), initialKey, -1, PS_INPUT_UP, 0});

    std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueEntryGreater> frontier;
    frontier.push(QueueEntry{priorityFor(mode, 0, initialHeuristic), 0, 0});
    uint64_t nextTie = 1;

    while (!frontier.empty()) {
        if (Clock::now() >= deadline) {
            result.status = SolveStatus::Timeout;
            break;
        }
        const QueueEntry entry = frontier.top();
        frontier.pop();
        const StateKey parentKey = nodes[entry.nodeIndex].key;
        const uint32_t parentDepth = nodes[entry.nodeIndex].depth;
        const auto best = bestDepth.find(parentKey);
        if (best != bestDepth.end() && best->second < parentDepth) {
            ++result.duplicates;
            continue;
        }
        Session* parentSession = nodes[entry.nodeIndex].session.get();
        if (parentSession == nullptr) {
            ++result.duplicates;
            continue;
        }
        ++result.expanded;
        for (const ps_input input : metadata.inputs) {
            if (Clock::now() >= deadline) {
                result.status = SolveStatus::Timeout;
                break;
            }
            auto child = std::make_unique<Session>(*parentSession);
            ps_step_result stepResult = puzzlescript::step(*child, input, solverStepOptions);
            puzzlescript::settlePendingAgain(*child, solverStepOptions);
            ++result.generated;
            if (solvedByStep(stepResult, *child, 0)) {
                result.status = SolveStatus::Solved;
                result.solution = reconstructSolution(nodes, entry.nodeIndex, input);
                return result;
            }
            if (!stepResult.changed) continue;
            const StateKey key = solverStateKey(*child, metadata.includeRandomState, metadata.projection);
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
            nodes.push_back(Node{std::move(child), key, static_cast<int32_t>(entry.nodeIndex), input, childDepth});
            frontier.push(QueueEntry{priorityFor(mode, childDepth, childHeuristic), nextTie++, childIndex});
        }
        nodes[entry.nodeIndex].session.reset();
        if (result.status == SolveStatus::Timeout) {
            break;
        }
    }
    return result;
}

void mergeStats(SolveResult& target, const SolveResult& source) {
    target.expanded += source.expanded;
    target.generated += source.generated;
    target.uniqueStates += source.uniqueStates;
    target.duplicates += source.duplicates;
}

SolveResult solveGeneratedLevel(
    const std::shared_ptr<const Game>& game,
    const LevelTemplate& generatedLevel,
    const SolverMetadata& metadata,
    uint64_t sampleId,
    int64_t timeoutMs,
    std::optional<SearchMode> mode
) {
    const TimePoint start = Clock::now();
    const TimePoint deadline = start + std::chrono::milliseconds(timeoutMs);
    if (mode) return runSearch(game, generatedLevel, metadata, sampleId, *mode, deadline);

    SolveResult combined;
    combined.status = SolveStatus::Timeout;
    const TimePoint bfsDeadline = start + std::chrono::milliseconds(std::max<int64_t>(1, timeoutMs / 6));
    SolveResult bfs = runSearch(game, generatedLevel, metadata, sampleId, SearchMode::Bfs, std::min(bfsDeadline, deadline));
    mergeStats(combined, bfs);
    if (bfs.status == SolveStatus::Solved || bfs.status == SolveStatus::LevelError) return bfs;
    if (Clock::now() < deadline) {
        SolveResult weighted = runSearch(game, generatedLevel, metadata, sampleId, SearchMode::WeightedAStar, deadline);
        mergeStats(combined, weighted);
        if (weighted.status == SolveStatus::Solved || weighted.status == SolveStatus::LevelError) return weighted;
        combined.status = weighted.status == SolveStatus::Exhausted ? SolveStatus::Exhausted : SolveStatus::Timeout;
    }
    return combined;
}

bool insertDedupe(SharedState& shared, uint64_t hash, size_t maxEntries) {
    const size_t shard = static_cast<size_t>(hash % shared.dedupe.size());
    std::lock_guard<std::mutex> lock(shared.dedupeMutexes[shard]);
    const size_t shardCap = std::max<size_t>(1, maxEntries / shared.dedupe.size());
    if (shared.dedupe[shard].find(hash) != shared.dedupe[shard].end()) {
        return false;
    }
    if (shared.dedupe[shard].size() >= shardCap) {
        const uint64_t evicted = shared.dedupeOrder[shard].front();
        shared.dedupeOrder[shard].pop_front();
        shared.dedupe[shard].erase(evicted);
    }
    shared.dedupe[shard].insert(hash);
    shared.dedupeOrder[shard].push_back(hash);
    return true;
}

void maybeInsertTop(SharedState& shared, Candidate candidate, size_t topK) {
    std::lock_guard<std::mutex> lock(shared.topMutex);
    if (shared.top.size() < topK) {
        shared.top.push_back(std::move(candidate));
        return;
    }
    auto worst = shared.top.begin();
    for (auto it = shared.top.begin() + 1; it != shared.top.end(); ++it) {
        if (betterCandidate(*worst, *it)) {
            worst = it;
        }
    }
    if (betterCandidate(candidate, *worst)) {
        *worst = std::move(candidate);
    }
}

SolverMetadata buildSolverMetadata(const Game& game) {
    SolverMetadata metadata;
    metadata.inputs = solverInputsForGame(game);
    metadata.includeRandomState = gameUsesRandomness(game);
    metadata.projection = buildStateHashProjection(game);
    return metadata;
}

void workerMain(
    const Options& options,
    const std::shared_ptr<const Game>& game,
    const SolverMetadata& solverMetadata,
    const GenerationProgram& program,
    const LevelTemplate& initLevel,
    SharedState& shared,
    TimePoint deadline
) {
    while (!shared.cancel.load(std::memory_order_relaxed)) {
        const uint64_t sampleId = shared.nextSample.fetch_add(1, std::memory_order_relaxed);
        if (options.samples && sampleId >= *options.samples) {
            shared.cancel.store(true, std::memory_order_relaxed);
            break;
        }
        if (Clock::now() >= deadline) {
            shared.cancel.store(true, std::memory_order_relaxed);
            break;
        }
        shared.counters.samplesAttempted.fetch_add(1, std::memory_order_relaxed);
        const uint64_t sampleSeed = splitmix64(options.seed ^ (sampleId + 0x9e3779b97f4a7c15ULL));
        Rng rng(sampleSeed);
        LevelTemplate candidateLevel;
        if (!applyProgram(program, initLevel, *game, rng, candidateLevel)) {
            shared.counters.rejected.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        shared.counters.validGenerated.fetch_add(1, std::memory_order_relaxed);
        const uint64_t levelHash = hashLevel(candidateLevel);
        if (!insertDedupe(shared, levelHash, options.dedupeMax)) {
            shared.counters.deduped.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        SolveResult solved = solveGeneratedLevel(game, candidateLevel, solverMetadata, sampleId, options.solverTimeoutMs, options.solverMode);
        shared.counters.solverSearches.fetch_add(1, std::memory_order_relaxed);
        shared.counters.solverExpanded.fetch_add(solved.expanded, std::memory_order_relaxed);
        shared.counters.solverGenerated.fetch_add(solved.generated, std::memory_order_relaxed);
        shared.counters.solverUniqueStates.fetch_add(solved.uniqueStates, std::memory_order_relaxed);
        shared.counters.solverDuplicates.fetch_add(solved.duplicates, std::memory_order_relaxed);
        if (solved.status == SolveStatus::Solved) {
            shared.counters.solved.fetch_add(1, std::memory_order_relaxed);
            Candidate candidate;
            candidate.score = solved.uniqueStates;
            candidate.uniqueStates = solved.uniqueStates;
            candidate.expanded = solved.expanded;
            candidate.solutionLength = solved.solution.size();
            candidate.levelHash = levelHash;
            candidate.sampleId = sampleId;
            candidate.seed = sampleSeed;
            candidate.solution = std::move(solved.solution);
            candidate.level = std::move(candidateLevel);
            maybeInsertTop(shared, std::move(candidate), options.topK);
        } else if (solved.status == SolveStatus::Timeout) {
            shared.counters.timeouts.fetch_add(1, std::memory_order_relaxed);
        } else if (solved.status == SolveStatus::LevelError) {
            shared.counters.levelErrors.fetch_add(1, std::memory_order_relaxed);
            shared.counters.exhausted.fetch_add(1, std::memory_order_relaxed);
        } else {
            shared.counters.exhausted.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

std::vector<Candidate> snapshotTop(SharedState& shared) {
    std::lock_guard<std::mutex> lock(shared.topMutex);
    auto top = shared.top;
    std::sort(top.begin(), top.end(), betterCandidate);
    return top;
}

CounterValues snapshotCounters(const SharedState& shared) {
    CounterValues counters;
    counters.samplesAttempted = shared.counters.samplesAttempted.load(std::memory_order_relaxed);
    counters.validGenerated = shared.counters.validGenerated.load(std::memory_order_relaxed);
    counters.rejected = shared.counters.rejected.load(std::memory_order_relaxed);
    counters.deduped = shared.counters.deduped.load(std::memory_order_relaxed);
    counters.solverSearches = shared.counters.solverSearches.load(std::memory_order_relaxed);
    counters.solverExpanded = shared.counters.solverExpanded.load(std::memory_order_relaxed);
    counters.solverGenerated = shared.counters.solverGenerated.load(std::memory_order_relaxed);
    counters.solverUniqueStates = shared.counters.solverUniqueStates.load(std::memory_order_relaxed);
    counters.solverDuplicates = shared.counters.solverDuplicates.load(std::memory_order_relaxed);
    counters.solved = shared.counters.solved.load(std::memory_order_relaxed);
    counters.timeouts = shared.counters.timeouts.load(std::memory_order_relaxed);
    counters.exhausted = shared.counters.exhausted.load(std::memory_order_relaxed);
    counters.levelErrors = shared.counters.levelErrors.load(std::memory_order_relaxed);
    return counters;
}

void appendCounterLabels(std::ostream& out, const CounterValues& counters) {
    out << " samples=" << counters.samplesAttempted
        << " valid=" << counters.validGenerated
        << " solved=" << counters.solved
        << " invalid_generation=" << counters.rejected
        << " duplicate=" << counters.deduped
        << " unsolved=" << counters.exhausted
        << " timeout=" << counters.timeouts;
}

std::string compactSolution(const std::vector<std::string>& solution) {
    std::string out;
    for (const auto& input : solution) {
        if (input == "up") out.push_back('U');
        else if (input == "down") out.push_back('D');
        else if (input == "left") out.push_back('L');
        else if (input == "right") out.push_back('R');
        else if (input == "action") out.push_back('A');
        else out.push_back('?');
    }
    return out;
}

void renderDashboard(const Options& options, SharedState& shared, TimePoint start, bool final) {
    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - start).count();
    const CounterValues counters = snapshotCounters(shared);
    const double rate = elapsed > 0.0 ? static_cast<double>(counters.samplesAttempted) / elapsed : 0.0;
    const auto top = snapshotTop(shared);
    std::ostringstream out;
    out << "\x1b[H\x1b[J";
    out << "PuzzleScript generator " << (final ? "finished" : "running") << "\n";
    out << "elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s"
        << " jobs=" << options.jobs
        << " rate=" << std::setprecision(1) << rate << "/s";
    appendCounterLabels(out, counters);
    out << "\n\n";
    out << "Top 5\n";
    for (size_t i = 0; i < 5; ++i) {
        if (i < top.size()) {
            const Candidate& c = top[i];
            out << std::setw(2) << (i + 1)
                << " score=" << c.score
                << " len=" << c.solutionLength
                << " states=" << c.uniqueStates
                << " sample=" << c.sampleId
                << " size=" << c.level.width << "x" << c.level.height
                << " sol=" << compactSolution(c.solution).substr(0, 32)
                << "\n";
        } else {
            out << std::setw(2) << (i + 1) << " --\n";
        }
    }
    std::cout << out.str() << std::flush;
}

void printSparseProgress(const Options& options, SharedState& shared, TimePoint start) {
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    const CounterValues counters = snapshotCounters(shared);
    std::cerr << "generator_progress elapsed_s=" << std::fixed << std::setprecision(1) << elapsed
              << " jobs=" << options.jobs
              << " top=" << snapshotTop(shared).size();
    appendCounterLabels(std::cerr, counters);
    std::cerr << "\n";
}

std::string objectNamesForCell(const Game& game, const LevelTemplate& level, int32_t tileIndex) {
    const MaskWord* cell = cellPtr(level, game, tileIndex);
    std::vector<std::string> names;
    for (int32_t id = 0; id < game.objectCount; ++id) {
        const uint32_t word = puzzlescript::maskWordIndex(static_cast<uint32_t>(id));
        if (word >= game.wordCount) continue;
        if ((cell[word] & puzzlescript::maskBit(static_cast<uint32_t>(id))) != 0) {
            if (static_cast<size_t>(id) < game.idDict.size()) names.push_back(game.idDict[static_cast<size_t>(id)]);
        }
    }
    std::sort(names.begin(), names.end());
    std::ostringstream out;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) out << " ";
        out << names[i];
    }
    return out.str();
}

std::string finalJson(const Options& options, const Game& game, SharedState& shared) {
    const auto top = snapshotTop(shared);
    const CounterValues counters = snapshotCounters(shared);
    std::ostringstream out;
    out << "{\n";
    out << "  \"totals\":{";
    out << "\"samples_attempted\":" << counters.samplesAttempted;
    out << ",\"valid_generated\":" << counters.validGenerated;
    out << ",\"solved\":" << counters.solved;
    out << ",\"rejected\":" << counters.rejected;
    out << ",\"deduped\":" << counters.deduped;
    out << ",\"timeouts\":" << counters.timeouts;
    out << ",\"exhausted\":" << counters.exhausted;
    out << ",\"invalid_generation\":" << counters.rejected;
    out << ",\"duplicate_levels\":" << counters.deduped;
    out << ",\"unsolved\":" << counters.exhausted;
    out << ",\"solver_timeouts\":" << counters.timeouts;
    out << "},\n";
    out << "  \"solver_totals\":{";
    out << "\"searches\":" << counters.solverSearches;
    out << ",\"expanded\":" << counters.solverExpanded;
    out << ",\"generated\":" << counters.solverGenerated;
    out << ",\"unique_states\":" << counters.solverUniqueStates;
    out << ",\"duplicates\":" << counters.solverDuplicates;
    out << ",\"timeouts\":" << counters.timeouts;
    out << ",\"exhausted\":" << counters.exhausted;
    out << ",\"level_errors\":" << counters.levelErrors;
    out << "},\n";
    out << "  \"top\":[\n";
    for (size_t i = 0; i < top.size(); ++i) {
        const Candidate& c = top[i];
        out << "    {";
        out << "\"rank\":" << (i + 1);
        out << ",\"difficulty_score\":" << c.score;
        out << ",\"unique_states\":" << c.uniqueStates;
        out << ",\"expanded\":" << c.expanded;
        out << ",\"solution_length\":" << c.solutionLength;
        out << ",\"level_hash\":" << c.levelHash;
        out << ",\"sample_id\":" << c.sampleId;
        out << ",\"seed\":" << c.seed;
        out << ",\"width\":" << c.level.width;
        out << ",\"height\":" << c.level.height;
        out << ",\"solution\":[";
        for (size_t j = 0; j < c.solution.size(); ++j) {
            if (j > 0) out << ",";
            out << jsonString(c.solution[j]);
        }
        out << "],\"cells\":[";
        for (int32_t y = 0; y < c.level.height; ++y) {
            if (y > 0) out << ",";
            out << "[";
            for (int32_t x = 0; x < c.level.width; ++x) {
                if (x > 0) out << ",";
                const int32_t tile = x * c.level.height + y;
                out << jsonString(objectNamesForCell(game, c.level, tile));
            }
            out << "]";
        }
        out << "]}";
        out << (i + 1 == top.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parseArgs(argc, argv);
        const std::string gameSource = readFile(options.gamePath);
        const Spec spec = parseSpec(readFile(options.specPath));
        puzzlescript::compiler::ParserState parserState;
        const std::string initSource = sourceWithInitLevel(gameSource, spec.initRows);
        auto game = compileGame(initSource, &parserState);
        if (game->levels.empty() || game->levels.front().isMessage) {
            throw std::runtime_error("Compiled init level did not produce a playable level");
        }
        NameResolver resolver(*game, parserState);
        const GenerationProgram program = compileGenerationProgram(spec, *game, resolver);
        const SolverMetadata solverMetadata = buildSolverMetadata(*game);

        SharedState shared;
        gCancelFlag = &shared.cancel;
        std::signal(SIGINT, handleSignal);
        std::signal(SIGTERM, handleSignal);

        const TimePoint start = Clock::now();
        const TimePoint deadline = start + std::chrono::milliseconds(options.timeMs);
        std::vector<std::thread> workers;
        workers.reserve(options.jobs);
        for (size_t i = 0; i < options.jobs; ++i) {
            workers.emplace_back(workerMain, std::cref(options), game, std::cref(solverMetadata), std::cref(program), std::cref(game->levels.front()), std::ref(shared), deadline);
        }

        const bool dashboard = !options.quiet && isatty(STDOUT_FILENO);
        TimePoint lastSparse = start;
        while (!shared.cancel.load(std::memory_order_relaxed)) {
            if (Clock::now() >= deadline) {
                shared.cancel.store(true, std::memory_order_relaxed);
                break;
            }
            if (dashboard) {
                renderDashboard(options, shared, start, false);
            } else if (!options.quiet && Clock::now() - lastSparse >= std::chrono::seconds(5)) {
                printSparseProgress(options, shared, start);
                lastSparse = Clock::now();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        for (auto& worker : workers) {
            worker.join();
        }
        if (dashboard) {
            renderDashboard(options, shared, start, true);
            std::cout << "\n";
        }

        const std::string json = finalJson(options, *game, shared);
        if (!options.jsonOut.empty()) {
            writeFile(options.jsonOut, json);
        } else if (options.quiet || !dashboard) {
            std::cout << json;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
