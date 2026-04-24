#include "runtime/core.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>

#include "simdjson.h" // vendored; will replace puzzlescript::json in Task 4

namespace puzzlescript {
void runRulesOnLevelStart(Session& session);
void runRulesOnLevelStart(Session& session, RuntimeStepOptions options);
namespace {

void rebuildMasks(Session& session);
void rebuildObjectCellIndex(Session& session);
void markAllMasksDirty(Session& session);
void markAllMovementMasksDirty(Session& session);
void clearMovementState(Session& session);
std::string toString(const json::Value& value);
std::vector<int32_t> parseIntVector(const json::Value& value);
std::vector<RuleCommand> parseRuleCommands(const json::Value& value);
LoopPointTable parseLoopPointTable(const json::Value& value);
std::map<std::string, int32_t> parseSoundEventMap(const json::Value& value);
SoundMaskEntry parseSoundMaskEntry(Game& game, const json::Value& value);
std::vector<SoundMaskEntry> parseSoundMaskEntries(Game& game, const json::Value& value);
std::vector<std::vector<SoundMaskEntry>> parseLayeredSoundMaskEntries(Game& game, const json::Value& value);
bool anyBitsInCommon(const MaskVector& lhs, const MaskVector& rhs);
bool anyBitsInCommon(const MaskWord* lhs, size_t lhsCount, const MaskWord* rhs, size_t rhsCount);
bool bitsSetInArray(const MaskWord* required, size_t requiredCount, const MaskWord* actual, size_t actualCount);
MaskVector getCellObjects(const Session& session, int32_t tileIndex);
MaskVector getCellMovements(const Session& session, int32_t tileIndex);
int32_t getShiftedMask5(const MaskVector& value, int32_t shift);

inline const MaskWord* maskPtr(const Game& game, MaskOffset offset);
inline MaskVector arenaCopy(const Game& game, MaskOffset offset, uint32_t wordCount);

struct CommandState {
    std::vector<std::string> queue;
    std::string messageText;
};

struct RuleApplyOutcome {
    bool matched = false;
    bool changed = false;
};

bool rowStillMatchesAt(const Session& session, const std::vector<Pattern>& row, int32_t startIndex, int32_t delta);

struct MovementResolveOutcome {
    bool moved = false;
    bool shouldUndo = false;
};

struct ExecuteTurnOptions {
    bool pushUndo = true;
    bool recordRestartUndo = true;
    bool emitAudio = true;
    bool ignoreRestartCommand = false;
    bool ignoreWin = false;
    bool dontModify = false;
    bool* observedModification = nullptr;
};

struct RuntimeCounterStorage {
    std::atomic<uint64_t> rulesVisited{0};
    std::atomic<uint64_t> rulesSkippedByMask{0};
    std::atomic<uint64_t> candidateCellsTested{0};
    std::atomic<uint64_t> patternTests{0};
    std::atomic<uint64_t> patternMatches{0};
    std::atomic<uint64_t> replacementsAttempted{0};
    std::atomic<uint64_t> replacementsApplied{0};
    std::atomic<uint64_t> rowScans{0};
    std::atomic<uint64_t> ellipsisScans{0};
    std::atomic<uint64_t> maskRebuildCalls{0};
    std::atomic<uint64_t> maskRebuildDirtyCalls{0};
    std::atomic<uint64_t> maskRebuildRows{0};
    std::atomic<uint64_t> maskRebuildColumns{0};
};

bool gRuntimeCountersEnabled = false;
RuntimeCounterStorage gRuntimeCounters;

inline void addCounter(std::atomic<uint64_t>& counter, uint64_t amount = 1) {
    if (gRuntimeCountersEnabled) {
        counter.fetch_add(amount, std::memory_order_relaxed);
    }
}

bool debugEnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
}

std::optional<uint64_t> debugEnvUint64(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(parsed);
}

struct DebugConfig {
    bool rigid = debugEnvFlag("PS_DEBUG_RIGID");
    bool again = debugEnvFlag("PS_DEBUG_AGAIN");
    bool random = debugEnvFlag("PS_DEBUG_RANDOM");
    bool rules = debugEnvFlag("PS_DEBUG_RULES");
    bool moves = debugEnvFlag("PS_DEBUG_MOVES");
    bool audio = debugEnvFlag("PS_DEBUG_AUDIO");
    std::optional<uint64_t> randomBoardHash = debugEnvUint64("PS_DEBUG_RANDOM_BOARD_HASH");
    std::optional<uint64_t> randomSessionHash = debugEnvUint64("PS_DEBUG_RANDOM_SESSION_HASH");
    std::string randomSubstring = [] {
        const char* value = std::getenv("PS_DEBUG_RANDOM_SUBSTRING");
        return value == nullptr ? std::string{} : std::string(value);
    }();
    bool ruleLinesUnrestricted = true;
    std::vector<int32_t> ruleLines = [] {
        std::vector<int32_t> lines;
        const char* value = std::getenv("PS_DEBUG_RULE_LINES");
        if (value == nullptr || value[0] == '\0') {
            return lines;
        }
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ',')) {
            if (token.empty()) {
                continue;
            }
            try {
                lines.push_back(std::stoi(token));
            } catch (const std::exception&) {
                continue;
            }
        }
        return lines;
    }();

    DebugConfig() {
        ruleLinesUnrestricted = ruleLines.empty();
    }
};

const DebugConfig& debugConfig() {
    static const DebugConfig config;
    return config;
}

const json::Value& requireField(const json::Value::Object& object, std::string_view key) {
    const auto it = object.find(std::string(key));
    if (it == object.end()) {
        throw json::ParseError("Missing required field: " + std::string(key));
    }
    return it->second;
}

int32_t toInt(const json::Value& value) {
    if (value.isInteger()) {
        return static_cast<int32_t>(value.asInteger());
    }
    if (value.isDouble()) {
        return static_cast<int32_t>(value.asDouble());
    }
    if (value.isBool()) {
        return value.asBool() ? 1 : 0;
    }
    if (value.isNull()) {
        return 0;
    }
    if (value.isString()) {
        return static_cast<int32_t>(std::stoi(value.asString()));
    }
    throw json::ParseError("Expected integer-compatible JSON value");
}

MaskWord toMaskWord(const json::Value& value) {
    if (value.isInteger()) {
        return static_cast<MaskWord>(value.asInteger());
    }
    if (value.isDouble()) {
        return static_cast<MaskWord>(value.asDouble());
    }
    if (value.isBool()) {
        return value.asBool() ? static_cast<MaskWord>(1) : static_cast<MaskWord>(0);
    }
    if (value.isNull()) {
        return 0;
    }
    if (value.isString()) {
        return static_cast<MaskWord>(std::stoll(value.asString()));
    }
    throw json::ParseError("Expected mask-word-compatible JSON value");
}

bool toBool(const json::Value& value) {
    return value.isBool() ? value.asBool() : toInt(value) != 0;
}

bool anyBitsSet(const std::vector<int32_t>& value) {
    return std::any_of(value.begin(), value.end(), [](int32_t word) { return word != 0; });
}

#if PS_MASK_WORD_BITS == 64
bool anyBitsSet(const MaskVector& value) {
    return std::any_of(value.begin(), value.end(), [](MaskWord word) { return word != 0; });
}
#endif

inline bool anyBitsSet(const int32_t* value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (value[i] != 0) return true;
    }
    return false;
}

#if PS_MASK_WORD_BITS == 64
inline bool anyBitsSet(const MaskWord* value, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (value[i] != 0) return true;
    }
    return false;
}
#endif

bool commandQueueContains(const CommandState& state, std::string_view command) {
    return std::find(state.queue.begin(), state.queue.end(), std::string(command)) != state.queue.end();
}

bool rigidDebugEnabled() {
    return debugConfig().rigid;
}

void rigidDebugLog(const std::string& message) {
    if (rigidDebugEnabled()) {
        std::cerr << "[rigid] " << message << '\n';
    }
}

bool againDebugEnabled() {
    return debugConfig().again;
}

void againDebugLog(const std::string& message) {
    if (againDebugEnabled()) {
        std::cerr << "[again] " << message << '\n';
    }
}

bool randomDebugEnabled() {
    return debugConfig().random;
}

void randomDebugLog(const std::string& message) {
    if (randomDebugEnabled()) {
        std::cerr << "[random] " << message << '\n';
    }
}

bool ruleDebugEnabled() {
    return debugConfig().rules;
}

bool ruleDebugLineFilterMatches(int32_t lineNumber) {
    if (!ruleDebugEnabled()) {
        return false;
    }
    const DebugConfig& config = debugConfig();
    if (config.ruleLinesUnrestricted) {
        return true;
    }
    return std::find(config.ruleLines.begin(), config.ruleLines.end(), lineNumber) != config.ruleLines.end();
}

void ruleDebugLog(const std::string& message) {
    if (ruleDebugEnabled()) {
        std::cerr << "[rules] " << message << '\n';
    }
}

bool movementDebugEnabled() {
    return debugConfig().moves;
}

void movementDebugLog(const std::string& message) {
    if (movementDebugEnabled()) {
        std::cerr << "[moves] " << message << '\n';
    }
}

bool audioDebugEnabled() {
    return debugConfig().audio;
}

void audioDebugLog(const std::string& message) {
    if (audioDebugEnabled()) {
        std::cerr << "[audio] " << message << '\n';
    }
}

std::optional<uint64_t> randomDebugBoardHashFilter() {
    return debugConfig().randomBoardHash;
}

std::optional<uint64_t> randomDebugSessionHashFilter() {
    return debugConfig().randomSessionHash;
}

std::string_view randomDebugSubstringFilter() {
    return debugConfig().randomSubstring;
}

void appendAudioEvent(Session& session, int32_t seed, const char* kind) {
    const std::string_view kindView = kind == nullptr ? std::string_view{} : std::string_view(kind);
    // JS dedupes movement audio seeds within each canmove/cantmove list.
    if (kindView == "canmove" || kindView == "cantmove") {
        const auto duplicate = std::find_if(session.lastAudioEvents.begin(), session.lastAudioEvents.end(), [seed, kindView](const ps_audio_event& event) {
            const std::string_view eventKind = event.kind == nullptr ? std::string_view{} : std::string_view(event.kind);
            return event.seed == seed && eventKind == kindView;
        });
        if (duplicate != session.lastAudioEvents.end()) {
            return;
        }
    }
    if (audioDebugEnabled()) {
        std::cerr << "[audio] emit seed=" << seed << " kind=" << kindView << '\n';
    }
    session.lastAudioEvents.push_back(ps_audio_event{seed, kind});
}

void appendUiAudioEvent(Session& session, int32_t seed, const char* kind) {
    session.lastUiAudioEvents.push_back(ps_audio_event{seed, kind});
}

int audioEventPriority(const ps_audio_event& event) {
    const std::string_view kind = event.kind == nullptr ? std::string_view{} : std::string_view(event.kind);
    if (kind == "cantmove") {
        return 0;
    }
    if (kind == "canmove") {
        return 1;
    }
    if (kind == "create") {
        return 2;
    }
    if (kind == "destroy") {
        return 3;
    }
    return 4;
}

void sortAudioEvents(Session& session) {
    std::stable_sort(session.lastAudioEvents.begin(), session.lastAudioEvents.end(), [](const ps_audio_event& lhs, const ps_audio_event& rhs) {
        return audioEventPriority(lhs) < audioEventPriority(rhs);
    });
}

void clearAudioEventsByKind(Session& session, std::string_view kind) {
    session.lastAudioEvents.erase(
        std::remove_if(session.lastAudioEvents.begin(), session.lastAudioEvents.end(), [kind](const ps_audio_event& event) {
            return kind == event.kind;
        }),
        session.lastAudioEvents.end());
}

void tryPlaySimpleSound(Session& session, std::string_view soundName) {
    const auto it = session.game->sfxEvents.find(std::string(soundName));
    if (it == session.game->sfxEvents.end()) {
        return;
    }
    // In the JS engine, these UI-ish "simple sounds" call playSound(seed, true),
    // which explicitly does NOT record the seed in the sound history (used by tests).
    // The native trace suite expects the same behavior: do not emit these as test audio events.
    appendUiAudioEvent(session, it->second, "ui");
}

void processOutputCommands(Session& session, const CommandState& commands, bool suppressMessages = false, bool emitAudio = true) {
    for (const auto& command : commands.queue) {
        if (command == "message") {
            if (suppressMessages || session.suppressRuleMessages) {
                continue;
            }
            session.preparedSession.messageText = commands.messageText;
            session.preparedSession.textMode = true;
            session.preparedSession.titleScreen = false;
            session.preparedSession.messageSelected = false;
            if (emitAudio) {
                tryPlaySimpleSound(session, "showmessage");
            }
        } else if (command.size() >= 3 && command[0] == 's' && command[1] == 'f' && command[2] == 'x') {
            if (emitAudio) {
                tryPlaySimpleSound(session, command);
            }
        }
    }
}

void accumulateMask(std::vector<int32_t>& target, const std::vector<int32_t>& source) {
    if (target.size() < source.size()) {
        target.resize(source.size(), 0);
    }
    for (size_t index = 0; index < source.size(); ++index) {
        target[index] |= source[index];
    }
}

#if PS_MASK_WORD_BITS == 64
void accumulateMask(MaskVector& target, const MaskVector& source) {
    if (target.size() < source.size()) {
        target.resize(source.size(), 0);
    }
    for (size_t index = 0; index < source.size(); ++index) {
        target[index] |= source[index];
    }
}
#endif

void accumulateMaskWords(MaskVector& target, const MaskWord* source, size_t sourceSize) {
    if (target.size() < sourceSize) {
        target.resize(sourceSize, 0);
    }
    for (size_t index = 0; index < sourceSize; ++index) {
        target[index] |= source[index];
    }
}

std::string describeObjects(const Session& session, const MaskVector& mask) {
    std::ostringstream stream;
    bool emitted = false;
    for (int32_t objectId = 0; objectId < session.game->objectCount; ++objectId) {
        const uint32_t word = maskWordIndex(static_cast<uint32_t>(objectId));
        if (word >= mask.size() || (mask[static_cast<size_t>(word)] & maskBit(static_cast<uint32_t>(objectId))) == 0) {
            continue;
        }
        if (!emitted) {
            stream << "[";
        } else {
            stream << ",";
        }
        emitted = true;
        if (static_cast<size_t>(objectId) < session.game->objectsById.size()
            && !session.game->objectsById[static_cast<size_t>(objectId)].name.empty()) {
            stream << session.game->objectsById[static_cast<size_t>(objectId)].name;
        } else {
            stream << "#" << objectId;
        }
    }
    if (!emitted) {
        return "[]";
    }
    stream << "]";
    return stream.str();
}

std::string describeMovements(const Session& session, const MaskVector& mask) {
    std::ostringstream stream;
    bool emitted = false;
    auto dirName = [](int32_t directionMask) -> const char* {
        switch (directionMask) {
            case 1: return "up";
            case 2: return "down";
            case 4: return "left";
            case 8: return "right";
            case 16: return "action";
            default: return "?";
        }
    };
    for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
        const int32_t directionMask = getShiftedMask5(mask, 5 * layer);
        if (directionMask == 0) {
            continue;
        }
        if (!emitted) {
            stream << "[";
        } else {
            stream << ",";
        }
        emitted = true;
        stream << "layer=" << layer << ":" << dirName(directionMask);
    }
    if (!emitted) {
        return "[]";
    }
    stream << "]";
    return stream.str();
}

std::string formatMatchList(const std::vector<int32_t>& matches, int32_t height, size_t limit = 8) {
    std::ostringstream stream;
    stream << "[";
    for (size_t index = 0; index < matches.size() && index < limit; ++index) {
        if (index > 0) {
            stream << ",";
        }
        const int32_t tileIndex = matches[index];
        stream << tileIndex << "@(" << (tileIndex / height) << "," << (tileIndex % height) << ")";
    }
    if (matches.size() > limit) {
        stream << ",...";
    }
    stream << "]";
    return stream.str();
}

void dumpActiveMovements(const Session& session, std::string_view label) {
    if (!movementDebugEnabled()) {
        return;
    }
    std::ostringstream header;
    header << std::string(label) << " hash=" << hashSession64(session);
    movementDebugLog(header.str());
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        const MaskVector movementMask = getCellMovements(session, tileIndex);
        if (!anyBitsSet(movementMask)) {
            continue;
        }
        const int32_t x = tileIndex / session.liveLevel.height;
        const int32_t y = tileIndex % session.liveLevel.height;
        std::ostringstream line;
        line << "tile=(" << x << "," << y << ")"
             << " objects=" << describeObjects(session, getCellObjects(session, tileIndex))
             << " movements=" << describeMovements(session, movementMask);
        movementDebugLog(line.str());
    }
}

void queueRuleCommands(const Rule& rule, CommandState& state) {
    if (rule.commands.empty()) {
        return;
    }

    const bool preexistingCancel = commandQueueContains(state, "cancel");
    const bool preexistingRestart = commandQueueContains(state, "restart");
    bool currentRuleCancel = false;
    bool currentRuleRestart = false;

    for (const auto& command : rule.commands) {
        const std::string& commandName = command.name;
        if (commandName == "cancel") {
            currentRuleCancel = true;
        } else if (commandName == "restart") {
            currentRuleRestart = true;
        }
    }

    if (preexistingCancel) {
        return;
    }
    if (preexistingRestart && !currentRuleCancel) {
        return;
    }
    if (currentRuleCancel || currentRuleRestart) {
        state.queue.clear();
        state.messageText.clear();
    }

    for (const auto& command : rule.commands) {
        const std::string& commandName = command.name;
        if (!commandQueueContains(state, commandName)) {
            state.queue.push_back(commandName);
        }
        if (commandName == "message" && command.argument.has_value()) {
            state.messageText = *command.argument;
        }
    }
}

void tryPlayMaskSounds(Session& session, const std::vector<SoundMaskEntry>& entries, const MaskVector& changedMask, const char* kind) {
    if (entries.empty() || !anyBitsSet(changedMask)) {
        return;
    }
    if (audioDebugEnabled()) {
        std::ostringstream stream;
        stream << "kind=" << kind << " changed=" << describeObjects(session, changedMask);
        audioDebugLog(stream.str());
    }
    const Game& game = *session.game;
    const uint32_t wordCount = game.wordCount;
    for (const auto& entry : entries) {
        const MaskWord* entryMask = maskPtr(game, entry.objectMask);
        if (entryMask == nullptr) continue;
        if (anyBitsInCommon(changedMask.data(), changedMask.size(), entryMask, wordCount)) {
            if (audioDebugEnabled()) {
                const MaskVector entryMaskCopy = arenaCopy(game, entry.objectMask, wordCount);
                std::ostringstream stream;
                stream << "matched kind=" << kind
                       << " seed=" << entry.seed
                       << " mask=" << describeObjects(session, entryMaskCopy);
                audioDebugLog(stream.str());
            }
            appendAudioEvent(session, entry.seed, kind);
        }
    }
}

void seedRandomState(Session::RandomState& state, std::string_view seed) {
    for (int idx = 0; idx < 256; ++idx) {
        state.s[static_cast<size_t>(idx)] = static_cast<uint8_t>(idx);
    }
    state.i = 0;
    state.j = 0;
    state.valid = !seed.empty();
    if (!state.valid) {
        return;
    }

    std::vector<uint8_t> input;
    input.reserve(seed.size());
    for (unsigned char ch : seed) {
        input.push_back(ch);
    }
    if (input.empty()) {
        state.valid = false;
        return;
    }

    uint32_t j = 0;
    for (size_t idx = 0; idx < state.s.size(); ++idx) {
        j = (j + state.s[idx] + input[idx % input.size()]) % 256;
        std::swap(state.s[idx], state.s[static_cast<size_t>(j)]);
    }
}

uint8_t nextRandomByte(Session::RandomState& state) {
    state.i = static_cast<uint8_t>((state.i + 1) % 256);
    state.j = static_cast<uint8_t>((state.j + state.s[static_cast<size_t>(state.i)]) % 256);
    std::swap(state.s[static_cast<size_t>(state.i)], state.s[static_cast<size_t>(state.j)]);
    const uint8_t index = static_cast<uint8_t>((state.s[static_cast<size_t>(state.i)] + state.s[static_cast<size_t>(state.j)]) % 256);
    return state.s[static_cast<size_t>(index)];
}

double randomUniform(Session::RandomState& state) {
    double output = 0.0;
    for (int idx = 0; idx < 7; ++idx) {
        output *= 256.0;
        output += nextRandomByte(state);
    }
    return output / (std::pow(2.0, 56.0) - 1.0);
}

std::vector<int32_t> previewRandomBytes(const Session::RandomState& state, int count) {
    Session::RandomState probe = state;
    std::vector<int32_t> bytes;
    bytes.reserve(static_cast<size_t>(std::max(count, 0)));
    for (int idx = 0; idx < count; ++idx) {
        bytes.push_back(static_cast<int32_t>(nextRandomByte(probe)));
    }
    return bytes;
}

bool anyBitsInCommon(const std::vector<int32_t>& lhs, const std::vector<int32_t>& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    for (size_t index = 0; index < count; ++index) {
        if ((lhs[index] & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}

#if PS_MASK_WORD_BITS == 64
bool anyBitsInCommon(const MaskVector& lhs, const MaskVector& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    for (size_t index = 0; index < count; ++index) {
        if ((lhs[index] & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}
#endif

// Raw-pointer overload so arena-stored masks can be checked without
// materializing a std::vector<int32_t>.
bool anyBitsInCommon(const int32_t* lhs, size_t lhsCount, const int32_t* rhs, size_t rhsCount) {
    const size_t count = std::min(lhsCount, rhsCount);
    for (size_t index = 0; index < count; ++index) {
        if ((lhs[index] & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}

#if PS_MASK_WORD_BITS == 64
bool anyBitsInCommon(const MaskWord* lhs, size_t lhsCount, const MaskWord* rhs, size_t rhsCount) {
    const size_t count = std::min(lhsCount, rhsCount);
    for (size_t index = 0; index < count; ++index) {
        if ((lhs[index] & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}

bool anyBitsInCommon(const int32_t* lhs, size_t lhsCount, const MaskWord* rhs, size_t rhsCount) {
    const size_t count = std::min(lhsCount, rhsCount);
    for (size_t index = 0; index < count; ++index) {
        if ((static_cast<MaskWord>(lhs[index]) & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}

bool anyBitsInCommon(const MaskWord* lhs, size_t lhsCount, const int32_t* rhs, size_t rhsCount) {
    const size_t count = std::min(lhsCount, rhsCount);
    for (size_t index = 0; index < count; ++index) {
        if ((lhs[index] & static_cast<MaskWord>(rhs[index])) != 0) {
            return true;
        }
    }
    return false;
}
#endif

bool bitsSetInArray(const std::vector<int32_t>& required, const int32_t* actual, size_t actualCount) {
    const size_t count = std::min(required.size(), actualCount);
    for (size_t index = 0; index < count; ++index) {
        if ((actual[index] & required[index]) != required[index]) {
            return false;
        }
    }
    for (size_t index = count; index < required.size(); ++index) {
        if (required[index] != 0) {
            return false;
        }
    }
    return true;
}

// Arena-pointer variant: required and actual both provided as raw pointers
// with their own word counts. Used after Rule/Pattern mask migration so we
// do not have to materialize per-row BitVectors just to compare bits.
bool bitsSetInArray(const int32_t* required, size_t requiredCount, const int32_t* actual, size_t actualCount) {
    const size_t count = std::min(requiredCount, actualCount);
    for (size_t index = 0; index < count; ++index) {
        if ((actual[index] & required[index]) != required[index]) {
            return false;
        }
    }
    for (size_t index = count; index < requiredCount; ++index) {
        if (required[index] != 0) {
            return false;
        }
    }
    return true;
}

#if PS_MASK_WORD_BITS == 64
bool bitsSetInArray(const MaskWord* required, size_t requiredCount, const MaskWord* actual, size_t actualCount) {
    const size_t count = std::min(requiredCount, actualCount);
    for (size_t index = 0; index < count; ++index) {
        if ((actual[index] & required[index]) != required[index]) {
            return false;
        }
    }
    for (size_t index = count; index < requiredCount; ++index) {
        if (required[index] != 0) {
            return false;
        }
    }
    return true;
}

bool bitsSetInArray(const MaskWord* required, size_t requiredCount, const int32_t* actual, size_t actualCount) {
    const size_t count = std::min(requiredCount, actualCount);
    for (size_t index = 0; index < count; ++index) {
        if ((static_cast<MaskWord>(actual[index]) & required[index]) != required[index]) {
            return false;
        }
    }
    for (size_t index = count; index < requiredCount; ++index) {
        if (required[index] != 0) {
            return false;
        }
    }
    return true;
}
#endif

bool bitsSetInArray(const std::vector<int32_t>& required, const std::vector<int32_t>& actual) {
    const size_t count = std::min(required.size(), actual.size());
    for (size_t index = 0; index < count; ++index) {
        if ((actual[index] & required[index]) != required[index]) {
            return false;
        }
    }
    for (size_t index = count; index < required.size(); ++index) {
        if (required[index] != 0) {
            return false;
        }
    }
    return true;
}

#if PS_MASK_WORD_BITS == 64
bool bitsSetInArray(const MaskVector& required, const MaskVector& actual) {
    const size_t count = std::min(required.size(), actual.size());
    for (size_t index = 0; index < count; ++index) {
        if ((actual[index] & required[index]) != required[index]) {
            return false;
        }
    }
    for (size_t index = count; index < required.size(); ++index) {
        if (required[index] != 0) {
            return false;
        }
    }
    return true;
}
#endif

std::string toString(const json::Value& value) {
    if (value.isString()) {
        return value.asString();
    }
    if (value.isInteger()) {
        return std::to_string(value.asInteger());
    }
    if (value.isDouble()) {
        return std::to_string(value.asDouble());
    }
    if (value.isBool()) {
        return value.asBool() ? "true" : "false";
    }
    if (value.isNull()) {
        return "";
    }
    throw json::ParseError("Expected string-compatible JSON value");
}

std::vector<int32_t> parseIntVector(const json::Value& value) {
    std::vector<int32_t> result;
    for (const auto& entry : value.asArray()) {
        result.push_back(toInt(entry));
    }
    return result;
}

MaskVector parseMaskVector(const json::Value& value) {
    MaskVector result;
    for (const auto& entry : value.asArray()) {
        result.push_back(toMaskWord(entry));
    }
    return result;
}

std::vector<int32_t> objectIdsFromMask(const MaskVector& words, int32_t objectCount) {
    std::vector<int32_t> ids;
    for (uint32_t word = 0; word < words.size(); ++word) {
        MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(words[static_cast<size_t>(word)]);
        while (bits != 0) {
            const int32_t bit = maskWordCountTrailingZeros(bits);
            const int32_t objectId = static_cast<int32_t>(word) * static_cast<int32_t>(kMaskWordBits) + bit;
            if (objectId < objectCount) {
                ids.push_back(objectId);
            }
            bits &= bits - 1;
        }
    }
    return ids;
}

// ---- Game mask-arena helpers ----------------------------------------------
// These append mask words into `game.maskArena` and return the offset (in
// words) of the first element. Used during IR parsing to replace the old
// std::vector<int32_t>-per-field layout with a single contiguous arena.

MaskOffset storeMaskWords(Game& game, const MaskVector& words) {
    MaskOffset offset = static_cast<MaskOffset>(game.maskArena.size());
    game.maskArena.insert(game.maskArena.end(), words.begin(), words.end());
    return offset;
}

// Append `game.wordCount` zero words and return the offset. Used for fields
// that are absent in the IR and need an all-zero mask at the arena's width.
[[maybe_unused]] MaskOffset storeZeroMask(Game& game) {
    MaskOffset offset = static_cast<MaskOffset>(game.maskArena.size());
    game.maskArena.insert(game.maskArena.end(), game.wordCount, 0);
    return offset;
}

[[maybe_unused]] inline MaskRef maskAt(const Game& game, MaskOffset offset) {
    return MaskRef{ game.maskArena.data() + offset };
}

[[maybe_unused]] inline MaskMut maskAt(Game& game, MaskOffset offset) {
    return MaskMut{ game.maskArena.data() + offset };
}

[[maybe_unused]] inline bool anyBitsSet(MaskRef m, uint32_t wordCount) {
    for (uint32_t w = 0; w < wordCount; ++w) {
        if (m.data[w] != 0) return true;
    }
    return false;
}

[[maybe_unused]] inline bool anyBitsInCommon(MaskRef a, MaskRef b, uint32_t wordCount) {
    for (uint32_t w = 0; w < wordCount; ++w) {
        if ((a.data[w] & b.data[w]) != 0) return true;
    }
    return false;
}

// Copy `wordCount` arena words starting at `offset` into a newly-allocated
// std::vector<int32_t>. Used at call sites that need a mutable copy of a mask
// (e.g. `std::vector<int32_t> objectsClear = replacement.objectsClear;` patterns where
// the callee modifies the copy). `kNullMaskOffset` yields an empty vector.
inline MaskVector arenaCopy(const Game& game, MaskOffset offset, uint32_t wordCount) {
    if (offset == kNullMaskOffset || wordCount == 0) return {};
    const MaskWord* begin = game.maskArena.data() + offset;
    return MaskVector(begin, begin + wordCount);
}

// Check whether any bit is set in the masked `wordCount` arena words starting
// at `offset`. `kNullMaskOffset` returns false.
inline bool arenaAnyBitsSet(const Game& game, MaskOffset offset, uint32_t wordCount) {
    if (offset == kNullMaskOffset) return false;
    const MaskWord* data = game.maskArena.data() + offset;
    for (uint32_t w = 0; w < wordCount; ++w) {
        if (data[w] != 0) return true;
    }
    return false;
}

// Return a raw pointer to the first word of an arena-stored mask, or nullptr
// if the offset is null.
inline const MaskWord* maskPtr(const Game& game, MaskOffset offset) {
    return offset == kNullMaskOffset ? nullptr : game.maskArena.data() + offset;
}

// Binary-search a sorted NamedMaskEntry table by name. Returns
// kNullMaskOffset if not found.
inline MaskOffset lookupNamedMask(const std::vector<Game::NamedMaskEntry>& table,
                                  const std::string& name) {
    auto it = std::lower_bound(table.begin(), table.end(), name,
        [](const Game::NamedMaskEntry& entry, const std::string& n) { return entry.name < n; });
    if (it == table.end() || it->name != name) return kNullMaskOffset;
    return it->offset;
}

std::vector<RuleCommand> parseRuleCommands(const json::Value& value) {
    std::vector<RuleCommand> result;
    if (!value.isArray()) {
        return result;
    }
    result.reserve(value.asArray().size());
    for (const auto& commandValue : value.asArray()) {
        if (!commandValue.isArray() || commandValue.asArray().empty()) {
            continue;
        }
        RuleCommand command;
        const auto& commandArray = commandValue.asArray();
        command.name = toString(commandArray[0]);
        if (commandArray.size() > 1 && !commandArray[1].isNull()) {
            command.argument = toString(commandArray[1]);
        }
        result.push_back(std::move(command));
    }
    return result;
}

std::vector<std::string> parseStringVector(const json::Value& value) {
    std::vector<std::string> result;
    for (const auto& entry : value.asArray()) {
        result.push_back(toString(entry));
    }
    return result;
}

std::vector<bool> parseBoolVector(const json::Value& value) {
    std::vector<bool> result;
    for (const auto& entry : value.asArray()) {
        result.push_back(toBool(entry));
    }
    return result;
}

LoopPointTable parseLoopPointTable(const json::Value& value) {
    LoopPointTable table;
    if (value.isArray()) {
        table.entries.resize(value.asArray().size());
        for (size_t index = 0; index < value.asArray().size(); ++index) {
            const auto& entry = value.asArray()[index];
            if (!entry.isNull()) {
                table.entries[index] = toInt(entry);
            }
        }
        return table;
    }
    if (value.isObject()) {
        size_t maxIndex = 0;
        bool sawAny = false;
        for (const auto& [key, entry] : value.asObject()) {
            if (entry.isNull()) {
                continue;
            }
            maxIndex = std::max(maxIndex, static_cast<size_t>(std::stoll(key)));
            sawAny = true;
        }
        if (!sawAny) {
            return table;
        }
        table.entries.resize(maxIndex + 1);
        for (const auto& [key, entry] : value.asObject()) {
            if (entry.isNull()) {
                continue;
            }
            table.entries[static_cast<size_t>(std::stoll(key))] = toInt(entry);
        }
    }
    return table;
}

std::map<std::string, std::string> parseStringMap(const json::Value& value) {
    std::map<std::string, std::string> result;
    for (const auto& [key, entry] : value.asObject()) {
        result.emplace(key, toString(entry));
    }
    return result;
}

std::map<std::string, int32_t> parseIntMap(const json::Value& value) {
    std::map<std::string, int32_t> result;
    for (const auto& [key, entry] : value.asObject()) {
        result.emplace(key, toInt(entry));
    }
    return result;
}

std::map<std::string, int32_t> parseSoundEventMap(const json::Value& value) {
    return parseIntMap(value);
}

// Parse a JSON map of { name -> int[] bitmask } into a sorted vector of
// NamedMaskEntry, with mask words stored into Game::maskArena.
std::vector<Game::NamedMaskEntry> parseNamedMaskTable(Game& game, const json::Value& value) {
    std::vector<Game::NamedMaskEntry> result;
    const auto& object = value.asObject();
    result.reserve(object.size());
    for (const auto& [key, entry] : object) {
        Game::NamedMaskEntry named;
        named.name = key;
        named.offset = storeMaskWords(game, parseMaskVector(entry));
        result.push_back(std::move(named));
    }
    std::sort(result.begin(), result.end(),
              [](const Game::NamedMaskEntry& a, const Game::NamedMaskEntry& b) { return a.name < b.name; });
    return result;
}

SoundMaskEntry parseSoundMaskEntry(Game& game, const json::Value& value) {
    const auto& object = value.asObject();
    SoundMaskEntry entry;
    if (const auto* objectMask = value.find("objectMask"); objectMask != nullptr) {
        entry.objectMask = storeMaskWords(game, parseMaskVector(*objectMask));
    }
    if (const auto* directionMask = value.find("directionMask"); directionMask != nullptr) {
        auto words = parseMaskVector(*directionMask);
        entry.directionMaskWidth = static_cast<uint32_t>(words.size());
        entry.directionMask = storeMaskWords(game, words);
    }
    entry.seed = toInt(requireField(object, "seed"));
    return entry;
}

std::vector<SoundMaskEntry> parseSoundMaskEntries(Game& game, const json::Value& value) {
    std::vector<SoundMaskEntry> result;
    if (!value.isArray()) {
        return result;
    }
    result.reserve(value.asArray().size());
    for (const auto& entry : value.asArray()) {
        if (!entry.isObject()) {
            continue;
        }
        result.push_back(parseSoundMaskEntry(game, entry));
    }
    return result;
}

std::vector<std::vector<SoundMaskEntry>> parseLayeredSoundMaskEntries(Game& game, const json::Value& value) {
    std::vector<std::vector<SoundMaskEntry>> result;
    if (!value.isArray()) {
        return result;
    }
    result.reserve(value.asArray().size());
    for (const auto& layer : value.asArray()) {
        result.push_back(parseSoundMaskEntries(game, layer));
    }
    return result;
}

std::vector<std::vector<int32_t>> parseSprite(const json::Value& value) {
    std::vector<std::vector<int32_t>> sprite;
    for (const auto& row : value.asArray()) {
        sprite.push_back(parseIntVector(row));
    }
    return sprite;
}

Replacement parseReplacement(Game& game, const json::Value& value) {
    const auto& object = value.asObject();
    Replacement replacement;
    replacement.objectsClear   = storeMaskWords(game, parseMaskVector(requireField(object, "objects_clear")));
    replacement.objectsSet     = storeMaskWords(game, parseMaskVector(requireField(object, "objects_set")));
    replacement.movementsClear = storeMaskWords(game, parseMaskVector(requireField(object, "movements_clear")));
    replacement.movementsSet   = storeMaskWords(game, parseMaskVector(requireField(object, "movements_set")));
    {
        auto words = parseMaskVector(requireField(object, "movements_layer_mask"));
        replacement.hasMovementsLayerMask = anyBitsSet(words);
        replacement.movementsLayerMask = storeMaskWords(game, words);
    }
    {
        auto words = parseMaskVector(requireField(object, "random_entity_mask"));
        replacement.randomEntityMaskWidth = static_cast<uint32_t>(words.size());
        replacement.hasRandomEntityMask = anyBitsSet(words);
        if (replacement.hasRandomEntityMask) {
            for (int32_t objectId = 0; objectId < game.objectCount; ++objectId) {
                const uint32_t word = maskWordIndex(static_cast<uint32_t>(objectId));
                if (word < replacement.randomEntityMaskWidth
                    && (words[static_cast<size_t>(word)] & maskBit(static_cast<uint32_t>(objectId))) != 0) {
                    replacement.randomEntityChoices.push_back(objectId);
                }
            }
        }
        replacement.randomEntityMask = storeMaskWords(game, words);
    }
    {
        auto words = parseMaskVector(requireField(object, "random_dir_mask"));
        replacement.randomDirMaskWidth = static_cast<uint32_t>(words.size());
        replacement.hasRandomDirMask = anyBitsSet(words);
        if (replacement.hasRandomDirMask) {
            for (int32_t layer = 0; layer < game.layerCount; ++layer) {
                const uint32_t word = movementWordIndexForLayer(static_cast<uint32_t>(layer));
                const uint32_t bit = movementBitShiftForLayer(static_cast<uint32_t>(layer));
                const int32_t dirBits = word < replacement.randomDirMaskWidth
                    ? static_cast<int32_t>((words[static_cast<size_t>(word)] >> bit) & 0x1F)
                    : 0;
                if (dirBits != 0) {
                    replacement.randomDirLayers.push_back(layer);
                }
            }
        }
        replacement.randomDirMask = storeMaskWords(game, words);
    }
    return replacement;
}

Pattern parsePattern(Game& game, const json::Value& value) {
    const auto& object = value.asObject();
    Pattern pattern;
    const std::string kind = toString(requireField(object, "kind"));
    pattern.kind = kind == "ellipsis" ? Pattern::Kind::Ellipsis : Pattern::Kind::CellPattern;
    if (pattern.kind == Pattern::Kind::Ellipsis) {
        return pattern;
    }

    const auto objectsPresent = parseMaskVector(requireField(object, "objects_present"));
    const auto objectsMissing = parseMaskVector(requireField(object, "objects_missing"));
    const auto movementsPresent = parseMaskVector(requireField(object, "movements_present"));
    const auto movementsMissing = parseMaskVector(requireField(object, "movements_missing"));
    pattern.hasObjectsPresent = anyBitsSet(objectsPresent);
    pattern.hasObjectsMissing = anyBitsSet(objectsMissing);
    pattern.hasMovementsPresent = anyBitsSet(movementsPresent);
    pattern.hasMovementsMissing = anyBitsSet(movementsMissing);
    pattern.objectsPresent   = storeMaskWords(game, objectsPresent);
    pattern.objectsMissing   = storeMaskWords(game, objectsMissing);
    pattern.movementsPresent = storeMaskWords(game, movementsPresent);
    pattern.movementsMissing = storeMaskWords(game, movementsMissing);
    pattern.objectAnchorIds = objectIdsFromMask(objectsPresent, game.objectCount);

    pattern.anyObjectsFirst = static_cast<uint32_t>(game.anyObjectOffsets.size());
    for (const auto& anyMask : requireField(object, "any_objects_present").asArray()) {
        const MaskVector words = parseMaskVector(anyMask);
        pattern.anyObjectAnchorIds.push_back(objectIdsFromMask(words, game.objectCount));
        const MaskOffset offset = storeMaskWords(game, words);
        game.anyObjectOffsets.push_back(offset);
    }
    pattern.anyObjectsCount = static_cast<uint32_t>(game.anyObjectOffsets.size()) - pattern.anyObjectsFirst;

    if (const auto* replacement = value.find("replacement"); replacement && !replacement->isNull()) {
        pattern.replacement = parseReplacement(game, *replacement);
    }
    return pattern;
}

Rule parseRule(Game& game, const json::Value& value) {
    const auto& object = value.asObject();
    Rule rule;
    rule.direction = toInt(requireField(object, "direction"));
    rule.hasReplacements = toBool(requireField(object, "has_replacements"));
    rule.lineNumber = toInt(requireField(object, "line_number"));
    rule.ellipsisCount = parseIntVector(requireField(object, "ellipsis_count"));
    rule.groupNumber = toInt(requireField(object, "group_number"));
    rule.rigid = toBool(requireField(object, "rigid"));
    rule.commands = parseRuleCommands(requireField(object, "commands"));
    rule.isRandom = toBool(requireField(object, "is_random"));
    rule.cellRowMasksFirst = static_cast<uint32_t>(game.cellRowMaskOffsets.size());
    for (const auto& rowMask : requireField(object, "cell_row_masks").asArray()) {
        game.cellRowMaskOffsets.push_back(storeMaskWords(game, parseMaskVector(rowMask)));
    }
    rule.cellRowMasksCount = static_cast<uint32_t>(game.cellRowMaskOffsets.size()) - rule.cellRowMasksFirst;

    rule.cellRowMasksMovementsFirst = static_cast<uint32_t>(game.cellRowMaskMovementsOffsets.size());
    MaskVector ruleMovementMaskWords(static_cast<size_t>(game.movementWordCount), 0);
    for (const auto& rowMask : requireField(object, "cell_row_masks_movements").asArray()) {
        const auto words = parseMaskVector(rowMask);
        for (uint32_t word = 0; word < game.movementWordCount && word < words.size(); ++word) {
            ruleMovementMaskWords[static_cast<size_t>(word)] |= words[static_cast<size_t>(word)];
        }
        game.cellRowMaskMovementsOffsets.push_back(storeMaskWords(game, words));
    }
    rule.cellRowMasksMovementsCount = static_cast<uint32_t>(game.cellRowMaskMovementsOffsets.size()) - rule.cellRowMasksMovementsFirst;
    rule.hasRuleMovementMask = anyBitsSet(ruleMovementMaskWords);
    rule.ruleMovementMask = storeMaskWords(game, ruleMovementMaskWords);

    rule.ruleMask = storeMaskWords(game, parseMaskVector(requireField(object, "rule_mask")));
    for (const auto& patternRowValue : requireField(object, "patterns").asArray()) {
        std::vector<Pattern> patternRow;
        for (const auto& patternValue : patternRowValue.asArray()) {
            patternRow.push_back(parsePattern(game, patternValue));
        }
        rule.patterns.push_back(std::move(patternRow));
    }
    return rule;
}

std::vector<std::vector<Rule>> parseRuleGroups(Game& game, const json::Value& value) {
    std::vector<std::vector<Rule>> groups;
    for (const auto& groupValue : value.asArray()) {
        std::vector<Rule> group;
        for (const auto& ruleValue : groupValue.asArray()) {
            group.push_back(parseRule(game, ruleValue));
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

WinCondition parseWinCondition(Game& game, const json::Value& value) {
    const auto& object = value.asObject();
    WinCondition condition;
    condition.quantifier = toInt(requireField(object, "quantifier"));
    condition.filter1 = storeMaskWords(game, parseMaskVector(requireField(object, "filter1")));
    condition.filter2 = storeMaskWords(game, parseMaskVector(requireField(object, "filter2")));
    condition.lineNumber = toInt(requireField(object, "line_number"));
    condition.aggr1 = toBool(requireField(object, "aggr1"));
    condition.aggr2 = toBool(requireField(object, "aggr2"));
    return condition;
}

LevelTemplate parseLevelTemplate(const json::Value& value) {
    const auto& object = value.asObject();
    LevelTemplate level;
    std::string kind = "level";
    if (const auto it = object.find("kind"); it != object.end()) {
        kind = it->second.asString();
    } else if (object.find("message") != object.end()) {
        kind = "message";
    }
    level.isMessage = kind == "message";
    if (level.isMessage) {
        level.message = requireField(object, "message").asString();
        return level;
    }
    if (object.find("width") == object.end() || object.find("height") == object.end()) {
        // JS export_ir_json can serialize the prepared text-mode placeholder
        // as {"objects":[]} for games that start on a message screen. The
        // containing prepared_session has current_level_index; after parsing
        // the game we replace this shell with the real level template.
        if (const auto objects = object.find("objects"); objects != object.end() && objects->second.isArray() && objects->second.asArray().empty()) {
            return level;
        }
    }
    if (const auto* lineNumber = value.find("line_number"); lineNumber != nullptr && !lineNumber->isNull()) {
        level.lineNumber = toInt(*lineNumber);
    }
    level.width = toInt(requireField(object, "width"));
    level.height = toInt(requireField(object, "height"));
    level.layerCount = toInt(requireField(object, "layer_count"));
    level.objects = parseMaskVector(requireField(object, "objects"));
    return level;
}

PreparedSession parsePreparedSession(const json::Value& value) {
    const auto& object = value.asObject();
    PreparedSession prepared;
    prepared.currentLevelIndex = toInt(requireField(object, "current_level_index"));
    if (const auto* target = value.find("current_level_target"); target && !target->isNull()) {
        prepared.currentLevelTarget = toInt(*target);
    }
    prepared.titleScreen = toBool(requireField(object, "title_screen"));
    prepared.textMode = toBool(requireField(object, "text_mode"));
    if (const auto* titleMode = value.find("title_mode"); titleMode && !titleMode->isNull()) {
        prepared.titleMode = toInt(*titleMode);
    }
    if (const auto* titleSelection = value.find("title_selection"); titleSelection && !titleSelection->isNull()) {
        prepared.titleSelection = toInt(*titleSelection);
    }
    prepared.titleSelected = toBool(requireField(object, "title_selected"));
    prepared.messageSelected = toBool(requireField(object, "message_selected"));
    prepared.winning = toBool(requireField(object, "winning"));
    if (const auto* messageText = value.find("message_text"); messageText && !messageText->isNull()) {
        prepared.messageText = toString(*messageText);
    }
    if (const auto* seed = value.find("loaded_level_seed"); seed && !seed->isNull()) {
        prepared.loadedLevelSeed = toString(*seed);
    }
    if (const auto* randomState = value.find("random_state"); randomState && randomState->isObject()) {
        const auto& randomStateObject = randomState->asObject();
        prepared.hasRandomState = true;
        if (const auto* valid = randomState->find("valid"); valid && !valid->isNull()) {
            prepared.randomStateValid = toBool(*valid);
        }
        if (const auto* i = randomState->find("i"); i && !i->isNull()) {
            prepared.randomStateI = static_cast<uint8_t>(toInt(*i));
        }
        if (const auto* j = randomState->find("j"); j && !j->isNull()) {
            prepared.randomStateJ = static_cast<uint8_t>(toInt(*j));
        }
        if (const auto* s = randomState->find("s"); s && s->isArray()) {
            prepared.randomStateS.reserve(s->asArray().size());
            for (const auto& entry : s->asArray()) {
                prepared.randomStateS.push_back(static_cast<uint8_t>(toInt(entry)));
            }
        }
    }
    prepared.oldFlickscreenDat = parseIntVector(requireField(object, "old_flickscreen_dat"));
    prepared.level = parseLevelTemplate(requireField(object, "level"));
    prepared.serializedLevel = requireField(object, "serialized_level").asString();

    if (const auto* restart = value.find("restart_target"); restart && restart->isObject()) {
        const auto& restartObject = restart->asObject();
        prepared.restart.width = toInt(requireField(restartObject, "width"));
        prepared.restart.height = toInt(requireField(restartObject, "height"));
        prepared.restart.objects = parseMaskVector(requireField(restartObject, "objects"));
        prepared.restart.oldFlickscreenDat = parseIntVector(requireField(restartObject, "old_flickscreen_dat"));
    }

    return prepared;
}

MaskVector getCellObjects(const Session& session, int32_t tileIndex) {
    MaskVector result(static_cast<size_t>(session.game->strideObject), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideObject);
    for (int32_t word = 0; word < session.game->strideObject; ++word) {
        result[static_cast<size_t>(word)] = session.liveLevel.objects[base + static_cast<size_t>(word)];
    }
    return result;
}

const MaskWord* getCellObjectsPtr(const Session& session, int32_t tileIndex) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideObject);
    return session.liveLevel.objects.data() + base;
}

size_t objectCellWordCount(const Session& session) {
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    return static_cast<size_t>((tileCount + 63) / 64);
}

void setObjectCellIndexBit(Session& session, int32_t objectId, int32_t tileIndex, bool present) {
    const size_t cellWordCount = objectCellWordCount(session);
    if (objectId < 0 || objectId >= session.game->objectCount || tileIndex < 0 || cellWordCount == 0) {
        return;
    }
    const size_t objectBase = static_cast<size_t>(objectId) * cellWordCount;
    const size_t bitWord = static_cast<size_t>(tileIndex >> 6);
    if (objectBase + bitWord >= session.objectCellBits.size()
        || static_cast<size_t>(objectId) >= session.objectCellCounts.size()) {
        session.objectCellIndexDirty = true;
        return;
    }
    const uint64_t bit = uint64_t{1} << static_cast<uint32_t>(tileIndex & 63);
    const bool wasPresent = (session.objectCellBits[objectBase + bitWord] & bit) != 0;
    if (present) {
        session.objectCellBits[objectBase + bitWord] |= bit;
        if (!wasPresent) {
            ++session.objectCellCounts[static_cast<size_t>(objectId)];
        }
    } else {
        session.objectCellBits[objectBase + bitWord] &= ~bit;
        if (wasPresent && session.objectCellCounts[static_cast<size_t>(objectId)] > 0) {
            --session.objectCellCounts[static_cast<size_t>(objectId)];
        }
    }
}

void setCellObjectsFromWords(Session& session, int32_t tileIndex, const MaskWord* objects) {
    const int32_t stride = session.game->strideObject;
    const size_t base = static_cast<size_t>(tileIndex * stride);
    const int32_t columnIndex = tileIndex / session.liveLevel.height;
    const int32_t rowIndex = tileIndex % session.liveLevel.height;
    const size_t columnBase = static_cast<size_t>(columnIndex * stride);
    const size_t rowBase = static_cast<size_t>(rowIndex * stride);
    MaskWord clearedAny = 0;
    for (int32_t word = 0; word < stride; ++word) {
        const MaskWord oldValue = session.liveLevel.objects[base + static_cast<size_t>(word)];
        const MaskWord value = objects[static_cast<size_t>(word)];
        MaskWordUnsigned changedBits = static_cast<MaskWordUnsigned>(oldValue ^ value);
        while (changedBits != 0) {
            const int32_t bit = maskWordCountTrailingZeros(changedBits);
            const int32_t objectId = word * static_cast<int32_t>(kMaskWordBits) + bit;
            setObjectCellIndexBit(session, objectId, tileIndex, (static_cast<MaskWordUnsigned>(value) & (MaskWordUnsigned{1} << bit)) != 0);
            changedBits &= changedBits - 1;
        }
        clearedAny |= (oldValue & ~value);
        session.liveLevel.objects[base + static_cast<size_t>(word)] = value;
        session.columnMasks[columnBase + static_cast<size_t>(word)] |= value;
        session.rowMasks[rowBase + static_cast<size_t>(word)] |= value;
        session.boardMask[static_cast<size_t>(word)] |= value;
    }
    if (clearedAny != 0) {
        if (static_cast<size_t>(rowIndex) < session.dirtyObjectRows.size())
            session.dirtyObjectRows[static_cast<size_t>(rowIndex)] = 1;
        if (static_cast<size_t>(columnIndex) < session.dirtyObjectColumns.size())
            session.dirtyObjectColumns[static_cast<size_t>(columnIndex)] = 1;
        session.dirtyObjectBoard = true;
        session.anyMasksDirty = true;
    }
}

void setCellObjects(Session& session, int32_t tileIndex, const MaskVector& objects) {
    setCellObjectsFromWords(session, tileIndex, objects.data());
}

MaskVector getCellMovements(const Session& session, int32_t tileIndex) {
    MaskVector result(static_cast<size_t>(session.game->strideMovement), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        result[static_cast<size_t>(word)] = session.liveMovements[base + static_cast<size_t>(word)];
    }
    return result;
}

const MaskWord* getCellMovementsPtr(const Session& session, int32_t tileIndex) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    return session.liveMovements.data() + base;
}

MaskVector getCellRigidGroupIndexMask(const Session& session, int32_t tileIndex) {
    MaskVector result(static_cast<size_t>(session.game->strideMovement), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        result[static_cast<size_t>(word)] = session.rigidGroupIndexMasks[base + static_cast<size_t>(word)];
    }
    return result;
}

MaskVector getCellRigidMovementAppliedMask(const Session& session, int32_t tileIndex) {
    MaskVector result(static_cast<size_t>(session.game->strideMovement), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        result[static_cast<size_t>(word)] = session.rigidMovementAppliedMasks[base + static_cast<size_t>(word)];
    }
    return result;
}

int32_t getShiftedMask5(const MaskVector& value, int32_t shift) {
    const int32_t word = shift / static_cast<int32_t>(kMaskWordBits);
    const int32_t bit = shift & static_cast<int32_t>(kMaskWordBitMask);
    MaskWordUnsigned result = 0;
    if (word < static_cast<int32_t>(value.size())) {
        result = static_cast<MaskWordUnsigned>(value[static_cast<size_t>(word)]) >> bit;
    }
    if (bit > static_cast<int32_t>(kMaskWordBits - 5U) && word + 1 < static_cast<int32_t>(value.size())) {
        result |= static_cast<MaskWordUnsigned>(value[static_cast<size_t>(word + 1)]) << (kMaskWordBits - bit);
    }
    return static_cast<int32_t>(result & 0x1F);
}

void clearShiftedMask5(MaskVector& value, int32_t shift) {
    const int32_t word = shift / static_cast<int32_t>(kMaskWordBits);
    const int32_t bit = shift & static_cast<int32_t>(kMaskWordBitMask);
    if (word >= static_cast<int32_t>(value.size())) {
        return;
    }
    const MaskWordUnsigned lowMask = MaskWordUnsigned{0x1F} << bit;
    value[static_cast<size_t>(word)] &= ~static_cast<MaskWord>(lowMask);
    if (bit > static_cast<int32_t>(kMaskWordBits - 5U) && word + 1 < static_cast<int32_t>(value.size())) {
        const MaskWordUnsigned highMask = MaskWordUnsigned{0x1F} >> (kMaskWordBits - bit);
        value[static_cast<size_t>(word + 1)] &= ~static_cast<MaskWord>(highMask);
    }
}

void setCellMovementsFromWords(Session& session, int32_t tileIndex, const MaskWord* movements) {
    const int32_t stride = session.game->strideMovement;
    const size_t base = static_cast<size_t>(tileIndex * stride);
    const int32_t columnIndex = tileIndex / session.liveLevel.height;
    const int32_t rowIndex = tileIndex % session.liveLevel.height;
    const size_t columnBase = static_cast<size_t>(columnIndex * stride);
    const size_t rowBase = static_cast<size_t>(rowIndex * stride);
    MaskWord clearedAny = 0;
    for (int32_t word = 0; word < stride; ++word) {
        const MaskWord oldValue = session.liveMovements[base + static_cast<size_t>(word)];
        const MaskWord value = movements[static_cast<size_t>(word)];
        clearedAny |= (oldValue & ~value);
        session.liveMovements[base + static_cast<size_t>(word)] = value;
        session.columnMovementMasks[columnBase + static_cast<size_t>(word)] |= value;
        session.rowMovementMasks[rowBase + static_cast<size_t>(word)] |= value;
        session.boardMovementMask[static_cast<size_t>(word)] |= value;
    }
    if (clearedAny != 0) {
        if (static_cast<size_t>(rowIndex) < session.dirtyMovementRows.size())
            session.dirtyMovementRows[static_cast<size_t>(rowIndex)] = 1;
        if (static_cast<size_t>(columnIndex) < session.dirtyMovementColumns.size())
            session.dirtyMovementColumns[static_cast<size_t>(columnIndex)] = 1;
        session.dirtyMovementBoard = true;
        session.anyMasksDirty = true;
    }
}

void setCellMovements(Session& session, int32_t tileIndex, const MaskVector& movements) {
    setCellMovementsFromWords(session, tileIndex, movements.data());
}

void setCellRigidGroupIndexMask(Session& session, int32_t tileIndex, const MaskVector& masks) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        session.rigidGroupIndexMasks[base + static_cast<size_t>(word)] = masks[static_cast<size_t>(word)];
    }
}

void setCellRigidMovementAppliedMask(Session& session, int32_t tileIndex, const MaskVector& masks) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        session.rigidMovementAppliedMasks[base + static_cast<size_t>(word)] = masks[static_cast<size_t>(word)];
    }
}

void clearRigidState(Session& session) {
    std::fill(session.rigidGroupIndexMasks.begin(), session.rigidGroupIndexMasks.end(), 0);
    std::fill(session.rigidMovementAppliedMasks.begin(), session.rigidMovementAppliedMasks.end(), 0);
}

void clearMovementState(Session& session) {
    std::fill(session.liveMovements.begin(), session.liveMovements.end(), 0);
    std::fill(session.rowMovementMasks.begin(), session.rowMovementMasks.end(), 0);
    std::fill(session.columnMovementMasks.begin(), session.columnMovementMasks.end(), 0);
    std::fill(session.boardMovementMask.begin(), session.boardMovementMask.end(), 0);
    std::fill(session.dirtyMovementRows.begin(), session.dirtyMovementRows.end(), 0);
    std::fill(session.dirtyMovementColumns.begin(), session.dirtyMovementColumns.end(), 0);
    session.dirtyMovementBoard = false;
    clearRigidState(session);

    const bool objectDirty = session.dirtyObjectBoard
        || std::any_of(session.dirtyObjectRows.begin(), session.dirtyObjectRows.end(), [](uint8_t value) { return value != 0; })
        || std::any_of(session.dirtyObjectColumns.begin(), session.dirtyObjectColumns.end(), [](uint8_t value) { return value != 0; });
    if (!objectDirty) {
        session.anyMasksDirty = false;
    }
}

void setShiftedMask5(MaskVector& value, int32_t shift, int32_t bits) {
    const int32_t word = shift / static_cast<int32_t>(kMaskWordBits);
    const int32_t bit = shift & static_cast<int32_t>(kMaskWordBitMask);
    if (word >= static_cast<int32_t>(value.size())) {
        return;
    }
    const MaskWordUnsigned packedBits = static_cast<MaskWordUnsigned>(bits & 0x1F);
    value[static_cast<size_t>(word)] |= static_cast<MaskWord>(packedBits << bit);
    if (bit > static_cast<int32_t>(kMaskWordBits - 5U) && word + 1 < static_cast<int32_t>(value.size())) {
        value[static_cast<size_t>(word + 1)] |= static_cast<MaskWord>(packedBits >> (kMaskWordBits - bit));
    }
}

std::vector<int32_t> findLayersInMask(const Session& session, const MaskVector& cellMask) {
    std::vector<int32_t> layers;
    for (int32_t objectId = 0; objectId < session.game->objectCount; ++objectId) {
        const uint32_t word = maskWordIndex(static_cast<uint32_t>(objectId));
        if (word < cellMask.size() && (cellMask[static_cast<size_t>(word)] & maskBit(static_cast<uint32_t>(objectId))) != 0) {
            if (static_cast<size_t>(objectId) < session.game->objectsById.size()) {
                const auto& object = session.game->objectsById[static_cast<size_t>(objectId)];
                if (object.layer >= 0) {
                    layers.push_back(object.layer);
                }
            }
        }
    }
    std::sort(layers.begin(), layers.end());
    layers.erase(std::unique(layers.begin(), layers.end()), layers.end());
    return layers;
}

int32_t inputToDirectionMask(ps_input input) {
    switch (input) {
        case PS_INPUT_UP: return 1;
        case PS_INPUT_LEFT: return 4;
        case PS_INPUT_DOWN: return 2;
        case PS_INPUT_RIGHT: return 8;
        case PS_INPUT_ACTION: return 16;
        default: return 0;
    }
}

std::pair<int32_t, int32_t> directionMaskToDelta(int32_t directionMask) {
    switch (directionMask) {
        case 1: return {0, -1};
        case 2: return {0, 1};
        case 4: return {-1, 0};
        case 8: return {1, 0};
        case 16: return {0, 0};
        default: return {0, 0};
    }
}

bool seedPlayerMovements(Session& session, int32_t directionMask) {
    const Game& game = *session.game;
    if (directionMask == 0 || game.playerMask == kNullMaskOffset) {
        return false;
    }
    const MaskWord* playerMask = maskPtr(game, game.playerMask);
    const uint32_t wordCount = game.wordCount;

    bool changed = false;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    const uint32_t objectStride = static_cast<uint32_t>(game.strideObject);
    const uint32_t movementStride = static_cast<uint32_t>(game.strideMovement);
    MaskVector playerCellMask;
    MaskVector movementMask;
    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        const MaskWord* cellMaskPtr = getCellObjectsPtr(session, tileIndex);
        if (!game.playerMaskAggregate) {
            if (!anyBitsInCommon(cellMaskPtr, objectStride, playerMask, wordCount)) {
                continue;
            }
        } else {
            bool containsAll = true;
            for (uint32_t word = 0; word < wordCount; ++word) {
                if ((cellMaskPtr[word] & playerMask[word]) != playerMask[word]) {
                    containsAll = false;
                    break;
                }
            }
            if (!containsAll) {
                continue;
            }
        }

        playerCellMask.resize(objectStride);
        for (uint32_t word = 0; word < objectStride; ++word) {
            const MaskWord pm = word < wordCount ? playerMask[word] : 0;
            playerCellMask[word] = cellMaskPtr[word] & pm;
        }
        const auto layers = findLayersInMask(session, playerCellMask);
        if (layers.empty()) {
            continue;
        }

        const MaskWord* movementSrc = getCellMovementsPtr(session, tileIndex);
        movementMask.assign(movementSrc, movementSrc + movementStride);
        bool tileChanged = false;
        for (const int32_t layer : layers) {
            const int32_t shift = 5 * layer;
            const int32_t before = getShiftedMask5(movementMask, shift);
            setShiftedMask5(movementMask, shift, directionMask);
            const int32_t after = getShiftedMask5(movementMask, shift);
            tileChanged = tileChanged || before != after;
        }
        if (tileChanged) {
            setCellMovements(session, tileIndex, movementMask);
            changed = true;
        }
    }

    return changed;
}

bool cellContainsPlayer(const Session& session, int32_t tileIndex) {
    const Game& game = *session.game;
    if (game.playerMask == kNullMaskOffset) {
        return false;
    }
    const MaskWord* playerMask = maskPtr(game, game.playerMask);
    const uint32_t wordCount = game.wordCount;
    const MaskWord* cellMask = getCellObjectsPtr(session, tileIndex);
    if (!game.playerMaskAggregate) {
        return anyBitsInCommon(cellMask, wordCount, playerMask, wordCount);
    }
    return bitsSetInArray(playerMask, wordCount, cellMask, wordCount);
}

std::vector<int32_t> collectPlayerPositions(const Session& session) {
    std::vector<int32_t> positions;
    if (session.game->playerMask == kNullMaskOffset) {
        return positions;
    }
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        if (cellContainsPlayer(session, tileIndex)) {
            positions.push_back(tileIndex);
        }
    }
    return positions;
}

bool resolveOneLayerMovement(Session& session, int32_t tileIndex, int32_t layer, int32_t directionMask, bool emitAudio) {
    const auto [dx, dy] = directionMaskToDelta(directionMask);
    const int32_t x = tileIndex / session.liveLevel.height;
    const int32_t y = tileIndex % session.liveLevel.height;
    const int32_t targetX = x + dx;
    const int32_t targetY = y + dy;
    if (targetX < 0 || targetX >= session.liveLevel.width || targetY < 0 || targetY >= session.liveLevel.height) {
        return false;
    }

    const int32_t targetIndex = tileIndex + dy + dx * session.liveLevel.height;
    const Game& game = *session.game;
    const MaskWord* layerMask = maskPtr(game, game.layerMaskOffsets[static_cast<size_t>(layer)]);
    const uint32_t wordCount = game.wordCount;
    MaskVector sourceMask = getCellObjects(session, tileIndex);
    const MaskVector sourceMaskBeforeMove = sourceMask;
    MaskVector targetMask = getCellObjects(session, targetIndex);
    if (directionMask != 16 && anyBitsInCommon(targetMask.data(), targetMask.size(), layerMask, wordCount)) {
        return false;
    }

    MaskVector movingEntities = sourceMask;
    for (size_t word = 0; word < movingEntities.size() && word < wordCount; ++word) {
        movingEntities[word] &= layerMask[word];
        sourceMask[word] &= ~layerMask[word];
        targetMask[word] |= movingEntities[word];
    }

    if (emitAudio && static_cast<size_t>(layer) < game.sfxMovementMasks.size()) {
        for (const auto& entry : game.sfxMovementMasks[static_cast<size_t>(layer)]) {
            const MaskWord* entryObjectMask = maskPtr(game, entry.objectMask);
            if (entryObjectMask == nullptr) continue;
            if (!anyBitsInCommon(sourceMaskBeforeMove.data(), sourceMaskBeforeMove.size(),
                                 entryObjectMask, wordCount)) {
                continue;
            }
            const MaskWord* entryDirectionMask = maskPtr(game, entry.directionMask);
            if (entryDirectionMask == nullptr) continue;
            const uint32_t wIdx = movementWordIndexForLayer(static_cast<uint32_t>(layer));
            const uint32_t bIdx = movementBitShiftForLayer(static_cast<uint32_t>(layer));
            const int32_t dirBits = (wIdx < entry.directionMaskWidth)
                ? static_cast<int32_t>((entryDirectionMask[wIdx] >> bIdx) & 0x1F) : 0;
            if ((dirBits & directionMask) == 0) {
                continue;
            }
            appendAudioEvent(session, entry.seed, "canmove");
        }
    }

    if (movementDebugEnabled()) {
        std::ostringstream stream;
        stream << "resolve tile=(" << x << "," << y << ")"
               << " target=(" << targetX << "," << targetY << ")"
               << " layer=" << layer
               << " objects=" << describeObjects(session, movingEntities)
               << " direction=" << describeMovements(session, getCellMovements(session, tileIndex));
        movementDebugLog(stream.str());
    }

    setCellObjects(session, tileIndex, sourceMask);
    setCellObjects(session, targetIndex, targetMask);
    return true;
}

MovementResolveOutcome resolveMovements(Session& session, std::vector<bool>* bannedGroups, bool emitAudio) {
    MovementResolveOutcome outcome;
    bool moved = true;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    while (moved) {
        moved = false;
        const uint32_t movementStride = static_cast<uint32_t>(session.game->strideMovement);
        for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            const MaskWord* movementMaskPtr = getCellMovementsPtr(session, tileIndex);
            if (!anyBitsSet(movementMaskPtr, movementStride)) {
                continue;
            }
            MaskVector movementMask(movementMaskPtr, movementMaskPtr + movementStride);
            bool changedTile = false;
            bool preventAggregateSplit = false;

            // Aggregate player movement must be atomic across all player layers present in the cell.
            // If one layer is blocked, none of the player layers should move.
            if (session.game->playerMaskAggregate && cellContainsPlayer(session, tileIndex)) {
                const Game& game = *session.game;
                const size_t aggregatePlayerCount = collectPlayerPositions(session).size();
                const MaskVector cellMask = getCellObjects(session, tileIndex);
                MaskVector playerCellMask = cellMask;
                const MaskWord* playerMaskWords = maskPtr(game, game.playerMask);
                for (uint32_t word = 0; word < game.wordCount && word < playerCellMask.size(); ++word) {
                    playerCellMask[word] &= playerMaskWords[word];
                }
                const auto playerLayers = findLayersInMask(session, playerCellMask);
                if (!playerLayers.empty()) {
                    int32_t playerDirection = 0;
                    bool playerHasMovement = false;
                    for (const int32_t layer : playerLayers) {
                        const int32_t layerMovement = getShiftedMask5(movementMask, 5 * layer);
                        if (layerMovement == 0) {
                            continue;
                        }
                        playerHasMovement = true;
                        if (playerDirection == 0) {
                            playerDirection = layerMovement;
                        } else if (playerDirection != layerMovement) {
                            // Mixed directions across the aggregate player: fall back to per-layer behavior.
                            playerHasMovement = false;
                            break;
                        }
                    }

                    if (playerHasMovement && playerDirection != 0) {
                        const auto [pdx, pdy] = directionMaskToDelta(playerDirection);
                        const int32_t x = tileIndex / session.liveLevel.height;
                        const int32_t y = tileIndex % session.liveLevel.height;
                        const int32_t targetX = x + pdx;
                        const int32_t targetY = y + pdy;
                        bool canMoveAll = true;
                        if (targetX < 0 || targetX >= session.liveLevel.width || targetY < 0 || targetY >= session.liveLevel.height) {
                            canMoveAll = false;
                            preventAggregateSplit = true;
                        } else {
                            const int32_t targetIndex = tileIndex + pdy + pdx * session.liveLevel.height;
                            const MaskVector targetMaskAll = getCellObjects(session, targetIndex);
                            bool blockedByPlayerConstituent = false;
                            for (const int32_t layer : playerLayers) {
                                const int32_t layerMovement = getShiftedMask5(movementMask, 5 * layer);
                                if (layerMovement == 0) {
                                    continue;
                                }
                                const MaskWord* layerMask = maskPtr(game, game.layerMaskOffsets[static_cast<size_t>(layer)]);
                                if (playerDirection != 16 && anyBitsInCommon(targetMaskAll.data(), targetMaskAll.size(), layerMask, game.wordCount)) {
                                    canMoveAll = false;
                                    blockedByPlayerConstituent = blockedByPlayerConstituent
                                        || anyBitsInCommon(targetMaskAll.data(), targetMaskAll.size(), playerMaskWords, game.wordCount);
                                    break;
                                }
                            }
                            preventAggregateSplit = blockedByPlayerConstituent || aggregatePlayerCount <= 1;

                            if (canMoveAll) {
                                MaskVector sourceMask = getCellObjects(session, tileIndex);
                                MaskVector targetMask = getCellObjects(session, targetIndex);
                                for (const int32_t layer : playerLayers) {
                                    const int32_t layerMovement = getShiftedMask5(movementMask, 5 * layer);
                                    if (layerMovement == 0) {
                                        continue;
                                    }
                                    const MaskWord* layerMask = maskPtr(game, game.layerMaskOffsets[static_cast<size_t>(layer)]);
                                    for (size_t word = 0; word < sourceMask.size() && word < game.wordCount; ++word) {
                                        const MaskWord moving = sourceMask[word] & layerMask[word];
                                        sourceMask[word] &= ~layerMask[word];
                                        targetMask[word] |= moving;
                                    }
                                    clearShiftedMask5(movementMask, 5 * layer);
                                }
                                setCellObjects(session, tileIndex, sourceMask);
                                setCellObjects(session, targetIndex, targetMask);
                                moved = true;
                                outcome.moved = true;
                                changedTile = true;
                            }
                        }
                    }
                }
            }

            for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
                const int32_t layerMovement = getShiftedMask5(movementMask, 5 * layer);
                if (layerMovement == 0) {
                    continue;
                }
                // Prevent per-layer movement from splitting aggregate player pieces.
                if (preventAggregateSplit && session.game->playerMaskAggregate && cellContainsPlayer(session, tileIndex)) {
                    const Game& game = *session.game;
                    const MaskWord* cellMask = getCellObjectsPtr(session, tileIndex);
                    const MaskWord* playerMaskWords = maskPtr(game, game.playerMask);
                    const MaskWord* layerMask = maskPtr(game, game.layerMaskOffsets[static_cast<size_t>(layer)]);
                    bool blocked = false;
                    for (uint32_t word = 0; word < game.wordCount; ++word) {
                        if (((cellMask[word] & playerMaskWords[word]) & layerMask[word]) != 0) {
                            blocked = true;
                            break;
                        }
                    }
                    if (blocked) {
                        continue;
                    }
                }
                if (resolveOneLayerMovement(session, tileIndex, layer, layerMovement, emitAudio)) {
                    clearShiftedMask5(movementMask, 5 * layer);
                    moved = true;
                    outcome.moved = true;
                    changedTile = true;
                }
            }
            if (changedTile) {
                setCellMovements(session, tileIndex, movementMask);
            }
        }
    }

    const uint32_t failureMovementStride = static_cast<uint32_t>(session.game->strideMovement);
    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        const MaskWord* movementMaskPtr2 = getCellMovementsPtr(session, tileIndex);
        if (!anyBitsSet(movementMaskPtr2, failureMovementStride)) {
            continue;
        }
        MaskVector movementMask(movementMaskPtr2, movementMaskPtr2 + failureMovementStride);

        if (session.game->rigid) {
            MaskVector rigidMovementAppliedMask = getCellRigidMovementAppliedMask(session, tileIndex);
            if (anyBitsSet(rigidMovementAppliedMask)) {
                for (size_t word = 0; word < movementMask.size() && word < rigidMovementAppliedMask.size(); ++word) {
                    movementMask[word] &= rigidMovementAppliedMask[word];
                }
                if (anyBitsSet(movementMask)) {
                    for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
                        if (getShiftedMask5(movementMask, 5 * layer) == 0) {
                            continue;
                        }
                        const MaskVector rigidGroupIndexMask = getCellRigidGroupIndexMask(session, tileIndex);
                        const int32_t rigidGroupIndex = getShiftedMask5(rigidGroupIndexMask, 5 * layer) - 1;
                        if (rigidGroupIndex >= 0
                            && static_cast<size_t>(rigidGroupIndex) < session.game->rigidGroupIndexToGroupIndex.size()) {
                            const int32_t groupIndex = session.game->rigidGroupIndexToGroupIndex[static_cast<size_t>(rigidGroupIndex)];
                            if (bannedGroups != nullptr && groupIndex >= 0) {
                                if (static_cast<size_t>(groupIndex) >= bannedGroups->size()) {
                                    bannedGroups->resize(static_cast<size_t>(groupIndex + 1), false);
                                }
                                if (!(*bannedGroups)[static_cast<size_t>(groupIndex)]) {
                                    (*bannedGroups)[static_cast<size_t>(groupIndex)] = true;
                                    outcome.shouldUndo = true;
                                    std::ostringstream stream;
                                    stream << "unresolved movement at tile=" << tileIndex
                                           << " layer=" << layer
                                           << " rigid_group_index=" << rigidGroupIndex
                                           << " bans_group=" << groupIndex;
                                    rigidDebugLog(stream.str());
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (emitAudio && !session.game->sfxMovementFailureMasks.empty()) {
            const Game& game = *session.game;
            const MaskVector cellMask = getCellObjects(session, tileIndex);
            for (const auto& entry : game.sfxMovementFailureMasks) {
                const MaskWord* entryObjectMask = maskPtr(game, entry.objectMask);
                if (entryObjectMask == nullptr) continue;
                if (!anyBitsInCommon(cellMask.data(), cellMask.size(), entryObjectMask, game.wordCount)) {
                    continue;
                }
                const MaskWord* entryDirectionMask = maskPtr(game, entry.directionMask);
                if (entryDirectionMask == nullptr) continue;
                if (!anyBitsInCommon(entryDirectionMask, entry.directionMaskWidth,
                                     movementMask.data(), movementMask.size())) {
                    continue;
                }
                appendAudioEvent(session, entry.seed, "cantmove");
            }
        }
    }

    clearMovementState(session);
    return outcome;
}

bool matchesPatternAt(const Session& session, const Pattern& pattern, int32_t tileIndex) {
    addCounter(gRuntimeCounters.patternTests);
    if (pattern.kind != Pattern::Kind::CellPattern) {
        return false;
    }
    const Game& game = *session.game;
    const uint32_t objectWordCount   = game.wordCount;
    const uint32_t movementWordCount = game.movementWordCount;
    const MaskWord* objects   = getCellObjectsPtr(session, tileIndex);
    const MaskWord* movements = getCellMovementsPtr(session, tileIndex);
    const MaskWord* arena    = game.maskArena.data();

    if (pattern.hasObjectsPresent) {
        const MaskWord* objectsPresent = arena + pattern.objectsPresent;
        for (uint32_t w = 0; w < objectWordCount; ++w) {
            if ((objects[w] & objectsPresent[w]) != objectsPresent[w]) {
                return false;
            }
        }
    }
    if (pattern.hasObjectsMissing) {
        const MaskWord* objectsMissing = arena + pattern.objectsMissing;
        for (uint32_t w = 0; w < objectWordCount; ++w) {
            if ((objects[w] & objectsMissing[w]) != 0) {
                return false;
            }
        }
    }
    for (uint32_t i = 0; i < pattern.anyObjectsCount; ++i) {
        const MaskOffset offset = game.anyObjectOffsets[pattern.anyObjectsFirst + i];
        const MaskWord* anyMask = arena + offset;
        bool found = false;
        for (uint32_t w = 0; w < objectWordCount; ++w) {
            if ((objects[w] & anyMask[w]) != 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    if (pattern.hasMovementsPresent) {
        const MaskWord* movementsPresent = arena + pattern.movementsPresent;
        for (uint32_t w = 0; w < movementWordCount; ++w) {
            if ((movements[w] & movementsPresent[w]) != movementsPresent[w]) {
                return false;
            }
        }
    }
    if (pattern.hasMovementsMissing) {
        const MaskWord* movementsMissing = arena + pattern.movementsMissing;
        for (uint32_t w = 0; w < movementWordCount; ++w) {
            if ((movements[w] & movementsMissing[w]) != 0) {
                return false;
            }
        }
    }
    addCounter(gRuntimeCounters.patternMatches);
    return true;
}

bool applyReplacementAt(Session& session, const Rule& rule, const Pattern& pattern, int32_t tileIndex) {
    addCounter(gRuntimeCounters.replacementsAttempted);
    if (!pattern.replacement.has_value()) {
        return false;
    }
    const auto& replacement = *pattern.replacement;
    const Game& game = *session.game;
    const uint32_t objectWordCount = game.wordCount;
    const uint32_t movementWordCount = game.movementWordCount;

    if (!rule.rigid
        && !replacement.hasRandomEntityMask
        && !replacement.hasRandomDirMask
        && !ruleDebugEnabled()) {
        const MaskWord* oldObjects = getCellObjectsPtr(session, tileIndex);
        const MaskWord* oldMovements = getCellMovementsPtr(session, tileIndex);
        const MaskWord* objectsClear = maskPtr(game, replacement.objectsClear);
        const MaskWord* objectsSet = maskPtr(game, replacement.objectsSet);
        const MaskWord* movementsClear = maskPtr(game, replacement.movementsClear);
        const MaskWord* movementsSet = maskPtr(game, replacement.movementsSet);
        const MaskWord* movementsLayerMask = replacement.hasMovementsLayerMask
            ? maskPtr(game, replacement.movementsLayerMask)
            : nullptr;

        MaskVector& newObjects = session.replacementObjectsScratch;
        MaskVector& newMovements = session.replacementMovementsScratch;
        MaskVector& created = session.replacementCreatedScratch;
        MaskVector& destroyed = session.replacementDestroyedScratch;
        newObjects.resize(objectWordCount);
        created.resize(objectWordCount);
        destroyed.resize(objectWordCount);
        newMovements.resize(movementWordCount);

        bool objectsChanged = false;
        bool movementsChanged = false;
        for (uint32_t word = 0; word < objectWordCount; ++word) {
            const MaskWord clearWord = objectsClear != nullptr ? objectsClear[word] : 0;
            const MaskWord setWord = objectsSet != nullptr ? objectsSet[word] : 0;
            const MaskWord before = oldObjects[word];
            const MaskWord after = (before & ~clearWord) | setWord;
            newObjects[word] = after;
            created[word] = after & ~before;
            destroyed[word] = before & ~after;
            objectsChanged = objectsChanged || after != before;
        }
        for (uint32_t word = 0; word < movementWordCount; ++word) {
            MaskWord clearWord = movementsClear != nullptr ? movementsClear[word] : 0;
            if (movementsLayerMask != nullptr) {
                clearWord |= movementsLayerMask[word];
            }
            const MaskWord setWord = movementsSet != nullptr ? movementsSet[word] : 0;
            const MaskWord before = oldMovements[word];
            const MaskWord after = (before & ~clearWord) | setWord;
            newMovements[word] = after;
            movementsChanged = movementsChanged || after != before;
        }

        if (!objectsChanged && !movementsChanged) {
            return false;
        }
        if (objectsChanged) {
            setCellObjectsFromWords(session, tileIndex, newObjects.data());
            accumulateMaskWords(session.pendingCreateMask, created.data(), created.size());
            accumulateMaskWords(session.pendingDestroyMask, destroyed.data(), destroyed.size());
        }
        if (movementsChanged) {
            setCellMovementsFromWords(session, tileIndex, newMovements.data());
        }
        addCounter(gRuntimeCounters.replacementsApplied);
        return true;
    }

    auto copyIntoScratchPair = [](MaskVector& current,
                                  MaskVector& old,
                                  const MaskWord* src,
                                  size_t n) {
        current.resize(n);
        old.resize(n);
        if (n > 0) {
            std::memcpy(current.data(), src, n * sizeof(MaskWord));
            std::memcpy(old.data(), src, n * sizeof(MaskWord));
        }
    };
    copyIntoScratchPair(session.replacementObjectsScratch,
                        session.replacementOldObjectsScratch,
                        getCellObjectsPtr(session, tileIndex),
                        static_cast<size_t>(game.strideObject));
    copyIntoScratchPair(session.replacementMovementsScratch,
                        session.replacementOldMovementsScratch,
                        getCellMovementsPtr(session, tileIndex),
                        static_cast<size_t>(game.strideMovement));
    MaskVector& objects      = session.replacementObjectsScratch;
    MaskVector& movements    = session.replacementMovementsScratch;
    const MaskVector& oldObjects   = session.replacementOldObjectsScratch;
    const MaskVector& oldMovements = session.replacementOldMovementsScratch;
    MaskVector rigidGroupIndexMask;
    MaskVector rigidMovementAppliedMask;
    bool rigidChange = false;
    // Reuse per-session scratch buffers instead of allocating a fresh
    // MaskVector per invocation. Width is stable across the session.
    auto initScratch = [](MaskVector& scratch,
                          const MaskWord* source,
                          uint32_t wordCount) {
        scratch.resize(wordCount);
        if (source != nullptr) {
            if (wordCount > 0) {
                std::memcpy(scratch.data(), source, static_cast<size_t>(wordCount) * sizeof(MaskWord));
            }
        } else {
            std::fill(scratch.begin(), scratch.end(), 0);
        }
    };
    initScratch(session.replacementObjectsClearScratch,  maskPtr(game, replacement.objectsClear),   objectWordCount);
    initScratch(session.replacementObjectsSetScratch,    maskPtr(game, replacement.objectsSet),     objectWordCount);
    initScratch(session.replacementMovementsClearScratch,maskPtr(game, replacement.movementsClear), movementWordCount);
    initScratch(session.replacementMovementsSetScratch,  maskPtr(game, replacement.movementsSet),   movementWordCount);
    MaskVector& objectsClear   = session.replacementObjectsClearScratch;
    MaskVector& objectsSet     = session.replacementObjectsSetScratch;
    MaskVector& movementsClear = session.replacementMovementsClearScratch;
    MaskVector& movementsSet   = session.replacementMovementsSetScratch;

    const MaskWord* movementsLayerMask = replacement.hasMovementsLayerMask ? maskPtr(game, replacement.movementsLayerMask) : nullptr;
    const MaskWord* randomEntityMask   = replacement.hasRandomEntityMask ? maskPtr(game, replacement.randomEntityMask) : nullptr;
    const uint32_t randomEntityMaskWidth = replacement.randomEntityMaskWidth;
    const MaskWord* randomDirMask      = replacement.hasRandomDirMask ? maskPtr(game, replacement.randomDirMask) : nullptr;
    const uint32_t randomDirMaskWidth    = replacement.randomDirMaskWidth;

    if (movementsLayerMask != nullptr) {
        for (size_t word = 0; word < movementsClear.size() && word < movementWordCount; ++word) {
            movementsClear[word] |= movementsLayerMask[word];
        }
    }

    if (replacement.hasRandomEntityMask) {
        const std::vector<int32_t>& choices = replacement.randomEntityChoices;
        if (!choices.empty()) {
            const double randomValue = randomUniform(session.randomState);
            const size_t chosen = std::min(
                choices.size() - 1,
                static_cast<size_t>(std::floor(randomValue * static_cast<double>(choices.size())))
            );
            const int32_t objectId = choices[chosen];
            const uint32_t word = maskWordIndex(static_cast<uint32_t>(objectId));
            if (randomDebugEnabled()) {
                std::ostringstream stream;
                stream << "replacement_random_entity"
                       << " line=" << rule.lineNumber
                       << " tile=" << tileIndex
                       << " choice_count=" << choices.size()
                       << " random=" << randomValue
                       << " chosen_index=" << chosen
                       << " chosen_id=" << objectId;
                if (static_cast<size_t>(objectId) < session.game->idDict.size()) {
                    stream << " chosen_name=" << session.game->idDict[static_cast<size_t>(objectId)];
                }
                stream << " choices=";
                for (size_t choiceIndex = 0; choiceIndex < choices.size(); ++choiceIndex) {
                    if (choiceIndex > 0) {
                        stream << ",";
                    }
                    stream << choices[choiceIndex];
                    if (static_cast<size_t>(choices[choiceIndex]) < session.game->idDict.size()) {
                        stream << ":" << session.game->idDict[static_cast<size_t>(choices[choiceIndex])];
                    }
                }
                randomDebugLog(stream.str());
            }
            objectsSet[static_cast<size_t>(word)] |= maskBit(static_cast<uint32_t>(objectId));
            if (static_cast<size_t>(objectId) < session.game->objectsById.size()) {
                const int32_t layer = session.game->objectsById[static_cast<size_t>(objectId)].layer;
                if (layer >= 0 && static_cast<size_t>(layer) < game.layerMaskOffsets.size()) {
                    const MaskWord* layerMask = maskPtr(game, game.layerMaskOffsets[static_cast<size_t>(layer)]);
                    for (size_t idx = 0; idx < objectsClear.size() && idx < game.wordCount; ++idx) {
                        objectsClear[idx] |= layerMask[idx];
                    }
                    clearShiftedMask5(movementsClear, 5 * layer);
                    const uint32_t moveWord = movementWordIndexForLayer(static_cast<uint32_t>(layer));
                    const uint32_t moveBit = movementBitShiftForLayer(static_cast<uint32_t>(layer));
                    if (moveWord < movementsClear.size()) {
                        movementsClear[moveWord] |= static_cast<MaskWord>(MaskWordUnsigned{0x1F} << moveBit);
                    }
                }
            }
        }
    }

    if (replacement.hasRandomDirMask) {
        for (const int32_t layer : replacement.randomDirLayers) {
            const int32_t shift = 5 * layer;
            const uint32_t wordIdx = movementWordIndexForLayer(static_cast<uint32_t>(layer));
            const uint32_t bitIdx = movementBitShiftForLayer(static_cast<uint32_t>(layer));
            const int32_t dirBits = (wordIdx < randomDirMaskWidth)
                ? static_cast<int32_t>((randomDirMask[wordIdx] >> bitIdx) & 0x1F) : 0;
            if (dirBits != 0) {
                const double randomValue = randomUniform(session.randomState);
                const int32_t randomDir = static_cast<int32_t>(std::floor(randomValue * 4.0));
                if (randomDebugEnabled()) {
                    std::ostringstream stream;
                    stream << "replacement_random_dir"
                           << " line=" << rule.lineNumber
                           << " tile=" << tileIndex
                           << " layer=" << layer
                           << " random=" << randomValue
                           << " dir=" << randomDir;
                    randomDebugLog(stream.str());
                }
                const int32_t word = (shift + randomDir) / static_cast<int32_t>(kMaskWordBits);
                const int32_t bit = (shift + randomDir) & static_cast<int32_t>(kMaskWordBitMask);
                if (static_cast<size_t>(word) < movementsSet.size()) {
                    movementsSet[static_cast<size_t>(word)] |= static_cast<MaskWord>(MaskWordUnsigned{1} << bit);
                }
            }
        }
    }

    for (size_t word = 0; word < objects.size(); ++word) {
        objects[word] = (objects[word] & ~objectsClear[word]) | objectsSet[word];
    }
    for (size_t word = 0; word < movements.size(); ++word) {
        movements[word] = (movements[word] & ~movementsClear[word]) | movementsSet[word];
    }

    session.replacementCreatedScratch.resize(objects.size());
    session.replacementDestroyedScratch.resize(objects.size());
    MaskVector& created   = session.replacementCreatedScratch;
    MaskVector& destroyed = session.replacementDestroyedScratch;
    for (size_t word = 0; word < objects.size(); ++word) {
        created[word] = objects[word] & ~oldObjects[word];
        destroyed[word] = oldObjects[word] & ~objects[word];
    }

    if (rule.rigid && movementsLayerMask != nullptr && movementWordCount > 0) {
        const int32_t rigidGroupIndex = (rule.groupNumber >= 0
            && static_cast<size_t>(rule.groupNumber) < session.game->groupNumberToRigidGroupIndex.size())
            ? session.game->groupNumberToRigidGroupIndex[static_cast<size_t>(rule.groupNumber)] + 1
            : 0;
        if (rigidGroupIndex > 0) {
            session.replacementRigidMaskScratch.assign(static_cast<size_t>(session.game->strideMovement), 0);
            MaskVector& rigidMask = session.replacementRigidMaskScratch;
            for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
                const uint32_t wIdx = movementWordIndexForLayer(static_cast<uint32_t>(layer));
                const uint32_t bIdx = movementBitShiftForLayer(static_cast<uint32_t>(layer));
                const int32_t layerBits = (wIdx < movementWordCount)
                    ? static_cast<int32_t>((movementsLayerMask[wIdx] >> bIdx) & 0x1F) : 0;
                if (layerBits != 0) {
                    setShiftedMask5(rigidMask, 5 * layer, rigidGroupIndex);
                }
            }

            rigidGroupIndexMask = getCellRigidGroupIndexMask(session, tileIndex);
            rigidMovementAppliedMask = getCellRigidMovementAppliedMask(session, tileIndex);
            if (!bitsSetInArray(rigidMask.data(), rigidMask.size(), rigidGroupIndexMask.data(), rigidGroupIndexMask.size())
                && !bitsSetInArray(movementsLayerMask, movementWordCount, rigidMovementAppliedMask.data(), rigidMovementAppliedMask.size())) {
                for (size_t word = 0; word < rigidGroupIndexMask.size() && word < rigidMask.size(); ++word) {
                    rigidGroupIndexMask[word] |= rigidMask[word];
                }
                for (size_t word = 0; word < rigidMovementAppliedMask.size() && word < movementWordCount; ++word) {
                    rigidMovementAppliedMask[word] |= movementsLayerMask[word];
                }
                rigidChange = true;
            }
        }
    }

    if (objects == oldObjects && movements == oldMovements && !rigidChange) {
        return false;
    }
    if (ruleDebugLineFilterMatches(rule.lineNumber)) {
        std::ostringstream stream;
        stream << "line=" << rule.lineNumber
               << " apply tile=" << tileIndex
               << "@(" << (tileIndex / session.liveLevel.height) << "," << (tileIndex % session.liveLevel.height) << ")"
               << " objects_before=" << describeObjects(session, oldObjects)
               << " objects_after=" << describeObjects(session, objects)
               << " movements_before=" << describeMovements(session, oldMovements)
               << " movements_after=" << describeMovements(session, movements);
        ruleDebugLog(stream.str());
    }
    setCellObjects(session, tileIndex, objects);
    setCellMovements(session, tileIndex, movements);
    accumulateMask(session.pendingCreateMask, created);
    accumulateMask(session.pendingDestroyMask, destroyed);
    if (rigidChange) {
        setCellRigidGroupIndexMask(session, tileIndex, rigidGroupIndexMask);
        setCellRigidMovementAppliedMask(session, tileIndex, rigidMovementAppliedMask);
    }
    addCounter(gRuntimeCounters.replacementsApplied);
    return true;
}

struct RowAnchor {
    int32_t patternIndex = -1;
    const std::vector<int32_t>* objectIds = nullptr;
    uint64_t cellCount = 0;
};

uint64_t objectPresenceCount(const Session& session, int32_t objectId) {
    if (objectId < 0
        || objectId >= session.game->objectCount
        || static_cast<size_t>(objectId) >= session.objectCellCounts.size()) {
        return 0;
    }
    return session.objectCellCounts[static_cast<size_t>(objectId)];
}

std::optional<RowAnchor> chooseRowAnchor(const Session& session, const std::vector<Pattern>& row) {
    if (session.objectCellIndexDirty || session.objectCellBits.empty()) {
        return std::nullopt;
    }
    const Game& game = *session.game;
    std::optional<RowAnchor> best;
    auto consider = [&](int32_t patternIndex, const std::vector<int32_t>& objectIds) {
        if (objectIds.empty()) {
            return;
        }
        uint64_t count = 0;
        for (const int32_t objectId : objectIds) {
            if (objectId < game.objectCount) {
                count += objectPresenceCount(session, objectId);
            }
        }
        if (count > 0 && (!best.has_value() || count < best->cellCount)) {
            best = RowAnchor{patternIndex, &objectIds, count};
        }
    };
    for (int32_t patternIndex = 0; patternIndex < static_cast<int32_t>(row.size()); ++patternIndex) {
        const Pattern& pattern = row[static_cast<size_t>(patternIndex)];
        if (pattern.kind != Pattern::Kind::CellPattern) {
            continue;
        }
        consider(patternIndex, pattern.objectAnchorIds);
        for (const auto& anyIds : pattern.anyObjectAnchorIds) {
            consider(patternIndex, anyIds);
        }
    }
    return best;
}

bool collectAnchoredRowMatchesInto(
    const Session& session,
    const std::vector<Pattern>& row,
    int32_t direction,
    const MaskWord* rowObjectMask,
    uint32_t rowObjectMaskWords,
    const MaskWord* rowMovementMask,
    uint32_t rowMovementMaskWords,
    int32_t xmin,
    int32_t xmax,
    int32_t ymin,
    int32_t ymax,
    int32_t delta,
    std::vector<int32_t>& matches
) {
    const auto anchor = chooseRowAnchor(session, row);
    if (!anchor.has_value()) {
        return false;
    }

    const bool horizontal = direction > 2;
    const int32_t validStartCount = std::max(0, xmax - xmin) * std::max(0, ymax - ymin);
    if (validStartCount <= 0 || anchor->cellCount >= static_cast<uint64_t>(std::max(8, validStartCount))) {
        return false;
    }
    if (anchor->objectIds == nullptr || anchor->objectIds->empty()) {
        return false;
    }

    const auto [dx, dy] = directionMaskToDelta(direction);
    const size_t cellWordCount = objectCellWordCount(session);

    const int32_t height = session.liveLevel.height;
    const int32_t width = session.liveLevel.width;
    const int32_t tileCount = width * height;
    for (const int32_t objectId : *anchor->objectIds) {
        if (objectId < 0 || objectId >= session.game->objectCount) {
            continue;
        }
        const size_t objectBase = static_cast<size_t>(objectId) * cellWordCount;
        if (objectBase + cellWordCount > session.objectCellBits.size()) {
            continue;
        }
        for (size_t wordIndex = 0; wordIndex < cellWordCount; ++wordIndex) {
            uint64_t bits = session.objectCellBits[objectBase + wordIndex];
            while (bits != 0) {
                const int32_t bit = __builtin_ctzll(bits);
                const int32_t anchorTile = static_cast<int32_t>(wordIndex * 64 + static_cast<size_t>(bit));
                bits &= bits - 1;
                if (anchorTile >= tileCount) {
                    continue;
                }

                const int32_t anchorX = anchorTile / height;
                const int32_t anchorY = anchorTile % height;
                const int32_t startX = anchorX - anchor->patternIndex * dx;
                const int32_t startY = anchorY - anchor->patternIndex * dy;
                if (startX < xmin || startX >= xmax || startY < ymin || startY >= ymax) {
                    continue;
                }

                const MaskWord* lineObjects = horizontal
                    ? session.rowMasks.data() + static_cast<size_t>(startY * session.game->strideObject)
                    : session.columnMasks.data() + static_cast<size_t>(startX * session.game->strideObject);
                const MaskWord* lineMovements = horizontal
                    ? session.rowMovementMasks.data() + static_cast<size_t>(startY * session.game->strideMovement)
                    : session.columnMovementMasks.data() + static_cast<size_t>(startX * session.game->strideMovement);
                if (!bitsSetInArray(rowObjectMask, rowObjectMaskWords, lineObjects, static_cast<size_t>(session.game->strideObject))
                    || !bitsSetInArray(rowMovementMask, rowMovementMaskWords, lineMovements, static_cast<size_t>(session.game->strideMovement))) {
                    continue;
                }

                addCounter(gRuntimeCounters.candidateCellsTested);
                const int32_t startIndex = startX * height + startY;
                if (rowStillMatchesAt(session, row, startIndex, delta)) {
                    matches.push_back(startIndex);
                }
            }
        }
    }

    const bool needsSortAndUnique = horizontal
        || anchor->objectIds == nullptr
        || anchor->objectIds->size() > 1;
    if (needsSortAndUnique && matches.size() > 1) {
        std::sort(matches.begin(), matches.end(), [horizontal, height](int32_t lhs, int32_t rhs) {
            if (!horizontal) {
                return lhs < rhs;
            }
            const int32_t lhsX = lhs / height;
            const int32_t lhsY = lhs % height;
            const int32_t rhsX = rhs / height;
            const int32_t rhsY = rhs % height;
            return lhsY == rhsY ? lhsX < rhsX : lhsY < rhsY;
        });
        matches.erase(std::unique(matches.begin(), matches.end()), matches.end());
    }
    return true;
}

void collectRowMatchesInto(
    const Session& session,
    const std::vector<Pattern>& row,
    int32_t direction,
    const MaskWord* rowObjectMask,
    uint32_t rowObjectMaskWords,
    const MaskWord* rowMovementMask,
    uint32_t rowMovementMaskWords,
    std::vector<int32_t>& matches
) {
    matches.clear();
    if (row.empty()) {
        return;
    }

    const int32_t len = static_cast<int32_t>(row.size());
    int32_t xmin = 0;
    int32_t xmax = session.liveLevel.width;
    int32_t ymin = 0;
    int32_t ymax = session.liveLevel.height;
    switch (direction) {
        case 1:
            ymin += (len - 1);
            break;
        case 2:
            ymax -= (len - 1);
            break;
        case 4:
            xmin += (len - 1);
            break;
        case 8:
            xmax -= (len - 1);
            break;
        default:
            return;
    }

    const bool horizontal = direction > 2;
    const auto [dx, dy] = directionMaskToDelta(direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        return;
    }

    if (!bitsSetInArray(rowObjectMask, rowObjectMaskWords, session.boardMask.data(), session.boardMask.size())
        || !bitsSetInArray(rowMovementMask, rowMovementMaskWords, session.boardMovementMask.data(), session.boardMovementMask.size())) {
        return;
    }

    if (collectAnchoredRowMatchesInto(session, row, direction,
                                      rowObjectMask, rowObjectMaskWords,
                                      rowMovementMask, rowMovementMaskWords,
                                      xmin, xmax, ymin, ymax, delta,
                                      matches)) {
        return;
    }

    if (horizontal) {
        for (int32_t y = ymin; y < ymax; ++y) {
            addCounter(gRuntimeCounters.rowScans);
            const MaskWord* rowObjects = session.rowMasks.data() + static_cast<size_t>(y * session.game->strideObject);
            const MaskWord* rowMovements = session.rowMovementMasks.data() + static_cast<size_t>(y * session.game->strideMovement);
            if (!bitsSetInArray(rowObjectMask, rowObjectMaskWords, rowObjects, static_cast<size_t>(session.game->strideObject))
                || !bitsSetInArray(rowMovementMask, rowMovementMaskWords, rowMovements, static_cast<size_t>(session.game->strideMovement))) {
                continue;
            }
            for (int32_t x = xmin; x < xmax; ++x) {
                addCounter(gRuntimeCounters.candidateCellsTested);
                const int32_t startIndex = x * session.liveLevel.height + y;
                bool matched = true;
                for (int32_t cellIndex = 0; cellIndex < len; ++cellIndex) {
                    if (!matchesPatternAt(session, row[static_cast<size_t>(cellIndex)], startIndex + cellIndex * delta)) {
                        matched = false;
                        break;
                    }
                }
                if (matched) {
                    matches.push_back(startIndex);
                }
            }
        }
    } else {
        for (int32_t x = xmin; x < xmax; ++x) {
            addCounter(gRuntimeCounters.rowScans);
            const MaskWord* columnObjects = session.columnMasks.data() + static_cast<size_t>(x * session.game->strideObject);
            const MaskWord* columnMovements = session.columnMovementMasks.data() + static_cast<size_t>(x * session.game->strideMovement);
            if (!bitsSetInArray(rowObjectMask, rowObjectMaskWords, columnObjects, static_cast<size_t>(session.game->strideObject))
                || !bitsSetInArray(rowMovementMask, rowMovementMaskWords, columnMovements, static_cast<size_t>(session.game->strideMovement))) {
                continue;
            }
            for (int32_t y = ymin; y < ymax; ++y) {
                addCounter(gRuntimeCounters.candidateCellsTested);
                const int32_t startIndex = x * session.liveLevel.height + y;
                bool matched = true;
                for (int32_t cellIndex = 0; cellIndex < len; ++cellIndex) {
                    if (!matchesPatternAt(session, row[static_cast<size_t>(cellIndex)], startIndex + cellIndex * delta)) {
                        matched = false;
                        break;
                    }
                }
                if (matched) {
                    matches.push_back(startIndex);
                }
            }
        }
    }
}

std::vector<int32_t> collectRowMatches(
    const Session& session,
    const std::vector<Pattern>& row,
    int32_t direction,
    const MaskWord* rowObjectMask,
    uint32_t rowObjectMaskWords,
    const MaskWord* rowMovementMask,
    uint32_t rowMovementMaskWords
) {
    std::vector<int32_t> matches;
    collectRowMatchesInto(
        session,
        row,
        direction,
        rowObjectMask,
        rowObjectMaskWords,
        rowMovementMask,
        rowMovementMaskWords,
        matches
    );
    return matches;
}

bool rowStillMatchesAt(const Session& session, const std::vector<Pattern>& row, int32_t startIndex, int32_t delta) {
    for (int32_t cellIndex = 0; cellIndex < static_cast<int32_t>(row.size()); ++cellIndex) {
        if (!matchesPatternAt(session, row[static_cast<size_t>(cellIndex)], startIndex + cellIndex * delta)) {
            return false;
        }
    }
    return true;
}

bool applyRowAt(Session& session, const Rule& rule, const std::vector<Pattern>& row, int32_t startIndex, int32_t delta) {
    bool changed = false;
    for (int32_t cellIndex = 0; cellIndex < static_cast<int32_t>(row.size()); ++cellIndex) {
        changed = applyReplacementAt(session, rule, row[static_cast<size_t>(cellIndex)], startIndex + cellIndex * delta) || changed;
    }
    return changed;
}

using RowMatch = std::vector<int32_t>;
using RuleMatch = std::vector<RowMatch>;

std::vector<RowMatch> collectEllipsisRowMatches(
    Session& session,
    const std::vector<Pattern>& row,
    int32_t direction,
    const MaskWord* rowObjectMask,
    uint32_t rowObjectMaskWords,
    const MaskWord* rowMovementMask,
    uint32_t rowMovementMaskWords
) {
    std::vector<RowMatch> matches;
    int32_t concreteCount = 0;
    for (const auto& pattern : row) {
        if (pattern.kind != Pattern::Kind::Ellipsis) {
            ++concreteCount;
        }
    }
    if (concreteCount == static_cast<int32_t>(row.size())) {
        return matches;
    }

    const auto [dx, dy] = directionMaskToDelta(direction);
    const int32_t parallelDelta = dx * session.liveLevel.height + dy;
    if (parallelDelta == 0) {
        return matches;
    }
    const bool horizontal = direction > 2;
    const int32_t lineCount = horizontal ? session.liveLevel.height : session.liveLevel.width;
    std::vector<uint8_t>& linePossible = session.ellipsisLinePossibleScratch;
    linePossible.assign(static_cast<size_t>(std::max(lineCount, 0)), 0);
    for (int32_t line = 0; line < lineCount; ++line) {
        const MaskWord* lineObjects = horizontal
            ? session.rowMasks.data() + static_cast<size_t>(line * session.game->strideObject)
            : session.columnMasks.data() + static_cast<size_t>(line * session.game->strideObject);
        const MaskWord* lineMovements = horizontal
            ? session.rowMovementMasks.data() + static_cast<size_t>(line * session.game->strideMovement)
            : session.columnMovementMasks.data() + static_cast<size_t>(line * session.game->strideMovement);
        linePossible[static_cast<size_t>(line)] =
            bitsSetInArray(rowObjectMask, rowObjectMaskWords,
                           lineObjects, static_cast<size_t>(session.game->strideObject))
            && bitsSetInArray(rowMovementMask, rowMovementMaskWords,
                              lineMovements, static_cast<size_t>(session.game->strideMovement));
    }

    auto availableAlongDirection = [&](int32_t x, int32_t y) {
        switch (direction) {
            case 1: return y + 1;
            case 2: return session.liveLevel.height - y;
            case 4: return x + 1;
            case 8: return session.liveLevel.width - x;
            default: return 0;
        }
    };

    std::vector<int32_t>& minConcreteSuffix = session.ellipsisMinConcreteSuffixScratch;
    minConcreteSuffix.assign(row.size() + 1, 0);
    for (int32_t rowIndex = static_cast<int32_t>(row.size()) - 1; rowIndex >= 0; --rowIndex) {
        minConcreteSuffix[static_cast<size_t>(rowIndex)] = minConcreteSuffix[static_cast<size_t>(rowIndex + 1)]
            + (row[static_cast<size_t>(rowIndex)].kind == Pattern::Kind::Ellipsis ? 0 : 1);
    }

    RowMatch& positions = session.ellipsisPositionsScratch;
    positions.clear();
    positions.reserve(static_cast<size_t>(concreteCount));
    for (int32_t x = 0; x < session.liveLevel.width; ++x) {
        if (!horizontal && (x < 0 || x >= lineCount || !linePossible[static_cast<size_t>(x)])) {
            continue;
        }
        for (int32_t y = 0; y < session.liveLevel.height; ++y) {
            if (horizontal && (y < 0 || y >= lineCount || !linePossible[static_cast<size_t>(y)])) {
                continue;
            }
            addCounter(gRuntimeCounters.ellipsisScans);
            const int32_t available = availableAlongDirection(x, y);
            if (available < concreteCount) {
                continue;
            }
            const int32_t tileIndex = x * session.liveLevel.height + y;

            positions.clear();
            auto search = [&](auto&& self, int32_t rowIndex, int32_t offset) -> void {
                if (rowIndex >= static_cast<int32_t>(row.size())) {
                    matches.push_back(positions);
                    return;
                }

                const Pattern& pattern = row[static_cast<size_t>(rowIndex)];
                if (pattern.kind == Pattern::Kind::Ellipsis) {
                    const int32_t maxSkip = available - offset - minConcreteSuffix[static_cast<size_t>(rowIndex + 1)];
                    for (int32_t skip = 0; skip <= maxSkip; ++skip) {
                        self(self, rowIndex + 1, offset + skip);
                    }
                    return;
                }

                if (offset >= available) {
                    return;
                }
                const int32_t matchIndex = tileIndex + offset * parallelDelta;
                if (!matchesPatternAt(session, pattern, matchIndex)) {
                    return;
                }
                positions.push_back(matchIndex);
                self(self, rowIndex + 1, offset + 1);
                positions.pop_back();
            };
            search(search, 0, 0);
        }
    }

    return matches;
}

bool applyEllipsisRowAt(Session& session, const Rule& rule, const std::vector<Pattern>& row, const RowMatch& positions) {
    if (positions.empty()) {
        return false;
    }

    bool changed = false;
    int32_t positionIndex = 0;
    for (int32_t cellIndex = 0; cellIndex < static_cast<int32_t>(row.size()); ++cellIndex) {
        if (row[static_cast<size_t>(cellIndex)].kind == Pattern::Kind::Ellipsis) {
            continue;
        }
        changed = applyReplacementAt(session, rule, row[static_cast<size_t>(cellIndex)], positions[static_cast<size_t>(positionIndex++)]) || changed;
    }
    return changed;
}

bool applyRowMatchAt(
    Session& session,
    const Rule& rule,
    const std::vector<Pattern>& row,
    int32_t ellipsisCount,
    const RowMatch& match,
    int32_t delta
) {
    if (ellipsisCount == 0) {
        if (match.empty()) {
            return false;
        }
        return applyRowAt(session, rule, row, match.front(), delta);
    }
    if (ellipsisCount >= 1) {
        return applyEllipsisRowAt(session, rule, row, match);
    }
    return false;
}

bool rowMatchStillMatches(
    const Session& session,
    const std::vector<Pattern>& row,
    int32_t ellipsisCount,
    const RowMatch& match,
    int32_t delta
) {
    if (ellipsisCount == 0) {
        return !match.empty() && rowStillMatchesAt(session, row, match.front(), delta);
    }
    int32_t positionIndex = 0;
    for (int32_t cellIndex = 0; cellIndex < static_cast<int32_t>(row.size()); ++cellIndex) {
        if (row[static_cast<size_t>(cellIndex)].kind == Pattern::Kind::Ellipsis) {
            continue;
        }
        if (positionIndex >= static_cast<int32_t>(match.size())
            || !matchesPatternAt(session, row[static_cast<size_t>(cellIndex)], match[static_cast<size_t>(positionIndex)])) {
            return false;
        }
        ++positionIndex;
    }
    return positionIndex == static_cast<int32_t>(match.size());
}

bool ruleCanPossiblyMatch(const Session& session, const Rule& rule) {
    const Game& game = *session.game;
    const MaskWord* required = game.maskArena.data() + rule.ruleMask;
    if (!bitsSetInArray(required, game.wordCount, session.boardMask.data(), session.boardMask.size())) {
        return false;
    }
    if (rule.hasRuleMovementMask) {
        const MaskWord* requiredMovements = game.maskArena.data() + rule.ruleMovementMask;
        if (!bitsSetInArray(requiredMovements, game.movementWordCount,
                            session.boardMovementMask.data(), session.boardMovementMask.size())) {
            return false;
        }
    }
    return true;
}

RuleApplyOutcome tryApplySimpleRule(Session& session, const Rule& rule, CommandState& commands, bool maskPrechecked) {
    const bool logRule = ruleDebugLineFilterMatches(rule.lineNumber);
    if (logRule) {
        std::ostringstream stream;
        stream << "line=" << rule.lineNumber
               << " begin direction=" << rule.direction
               << " pattern_rows=" << rule.patterns.size()
               << " is_random=" << (rule.isRandom ? 1 : 0);
        ruleDebugLog(stream.str());
    }
    if (rule.isRandom || rule.patterns.empty()) {
        if (logRule) {
            std::ostringstream stream;
            stream << "line=" << rule.lineNumber << " skip reason="
                   << (rule.isRandom ? "random" : "empty-patterns");
            ruleDebugLog(stream.str());
        }
        return {};
    }
    if (!maskPrechecked && !ruleCanPossiblyMatch(session, rule)) {
        addCounter(gRuntimeCounters.rulesSkippedByMask);
        if (logRule) {
            std::ostringstream stream;
            stream << "line=" << rule.lineNumber
                   << " skip reason=rule-mask"
                   << " rule_mask=" << describeObjects(session, arenaCopy(*session.game, rule.ruleMask, session.game->wordCount))
                   << " board_mask=" << describeObjects(session, session.boardMask);
            ruleDebugLog(stream.str());
        }
        return {};
    }
    const auto [dx, dy] = directionMaskToDelta(rule.direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        if (logRule) {
            std::ostringstream stream;
            stream << "line=" << rule.lineNumber << " skip reason=delta-zero";
            ruleDebugLog(stream.str());
        }
        return {};
    }

    if (rule.patterns.size() == 1) {
        const size_t rowIndex = 0;
        if (rule.ellipsisCount.empty() || rule.patterns[rowIndex].empty()) {
            return {};
        }
        const auto& row = rule.patterns[rowIndex];
        const int32_t ellipsisCount = rule.ellipsisCount[rowIndex];
        bool matched = false;
        bool changed = false;

        if (ellipsisCount == 0) {
            const Game& game = *session.game;
            const MaskOffset rowObjectOffset = rule.cellRowMasksCount > 0
                ? game.cellRowMaskOffsets[rule.cellRowMasksFirst]
                : rule.ruleMask;
            const MaskWord* rowObjectMask = maskPtr(game, rowObjectOffset);
            const MaskOffset rowMovementOffset = rule.cellRowMasksMovementsCount > 0
                ? game.cellRowMaskMovementsOffsets[rule.cellRowMasksMovementsFirst]
                : kNullMaskOffset;
            const MaskWord* rowMovementMask = maskPtr(game, rowMovementOffset);
            const uint32_t rowMovementMaskWords = rowMovementMask != nullptr ? game.movementWordCount : 0;
            std::vector<int32_t>& matches = session.singleRowMatchScratch;
            collectRowMatchesInto(session, row, rule.direction,
                                  rowObjectMask, game.wordCount,
                                  rowMovementMask, rowMovementMaskWords,
                                  matches);
            if (matches.empty()) {
                if (logRule) {
                    const MaskVector rowObjectMaskCopy = arenaCopy(game, rowObjectOffset, game.wordCount);
                    const MaskVector rowMovementMaskCopy = rowMovementMask != nullptr
                        ? arenaCopy(game, rowMovementOffset, game.movementWordCount)
                        : MaskVector{};
                    std::ostringstream stream;
                    stream << "line=" << rule.lineNumber
                           << " row=0 matches=0"
                           << " object_mask=" << describeObjects(session, rowObjectMaskCopy)
                           << " movement_mask=" << describeMovements(session, rowMovementMaskCopy);
                    ruleDebugLog(stream.str());
                }
                return {};
            }
            matched = true;
            if (logRule) {
                std::ostringstream stream;
                stream << "line=" << rule.lineNumber
                       << " row=0 matches=" << matches.size()
                       << " starts=" << formatMatchList(matches, session.liveLevel.height);
                ruleDebugLog(stream.str());
            }
            for (size_t matchIndex = 0; matchIndex < matches.size(); ++matchIndex) {
                const int32_t startIndex = matches[matchIndex];
                if (matchIndex > 0 && !rowStillMatchesAt(session, row, startIndex, delta)) {
                    continue;
                }
                changed = applyRowAt(session, rule, row, startIndex, delta) || changed;
            }
        } else {
            const Game& game = *session.game;
            const MaskOffset rowObjectOffset = rule.cellRowMasksCount > 0
                ? game.cellRowMaskOffsets[rule.cellRowMasksFirst]
                : rule.ruleMask;
            const MaskWord* rowObjectMask = maskPtr(game, rowObjectOffset);
            const MaskOffset rowMovementOffset = rule.cellRowMasksMovementsCount > 0
                ? game.cellRowMaskMovementsOffsets[rule.cellRowMasksMovementsFirst]
                : kNullMaskOffset;
            const MaskWord* rowMovementMask = maskPtr(game, rowMovementOffset);
            const uint32_t rowMovementMaskWords = rowMovementMask != nullptr ? game.movementWordCount : 0;
            auto matches = collectEllipsisRowMatches(session, row, rule.direction,
                                                     rowObjectMask, game.wordCount,
                                                     rowMovementMask, rowMovementMaskWords);
            if (matches.empty()) {
                if (logRule) {
                    std::ostringstream stream;
                    stream << "line=" << rule.lineNumber
                           << " row=0 matches=0 ellipsis=" << ellipsisCount;
                    ruleDebugLog(stream.str());
                }
                return {};
            }
            matched = true;
            if (logRule) {
                std::ostringstream stream;
                stream << "line=" << rule.lineNumber
                       << " row=0 matches=" << matches.size()
                       << " ellipsis=" << ellipsisCount;
                ruleDebugLog(stream.str());
            }
            for (size_t matchIndex = 0; matchIndex < matches.size(); ++matchIndex) {
                const RowMatch& match = matches[matchIndex];
                if (matchIndex > 0 && !rowMatchStillMatches(session, row, ellipsisCount, match, delta)) {
                    continue;
                }
                changed = applyEllipsisRowAt(session, rule, row, match) || changed;
            }
        }

        if (matched) {
            queueRuleCommands(rule, commands);
        }
        if (ruleDebugEnabled()) {
            std::ostringstream stream;
            stream << "line=" << rule.lineNumber
                   << " matched=1 changed=" << (changed ? 1 : 0)
                   << " row_count=1";
            ruleDebugLog(stream.str());
        }
        return RuleApplyOutcome{changed, changed};
    }

    std::vector<std::vector<RowMatch>> rowMatches;
    rowMatches.reserve(rule.patterns.size());
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        if (rowIndex >= rule.ellipsisCount.size() || rule.patterns[rowIndex].empty()) {
            return {};
        }
        const auto& row = rule.patterns[rowIndex];
        const int32_t ellipsisCount = rule.ellipsisCount[rowIndex];
        if (ellipsisCount == 0) {
            const Game& game = *session.game;
            const MaskOffset rowObjectOffset = rowIndex < rule.cellRowMasksCount
                ? game.cellRowMaskOffsets[rule.cellRowMasksFirst + rowIndex]
                : rule.ruleMask;
            const MaskWord* rowObjectMask = maskPtr(game, rowObjectOffset);
            const uint32_t rowObjectMaskWords = game.wordCount;
            const MaskOffset rowMovementOffset = rowIndex < rule.cellRowMasksMovementsCount
                ? game.cellRowMaskMovementsOffsets[rule.cellRowMasksMovementsFirst + rowIndex]
                : kNullMaskOffset;
            const MaskWord* rowMovementMask = maskPtr(game, rowMovementOffset);
            const uint32_t rowMovementMaskWords = rowMovementMask != nullptr ? game.movementWordCount : 0;
            auto matches = collectRowMatches(session, row, rule.direction,
                                             rowObjectMask, rowObjectMaskWords,
                                             rowMovementMask, rowMovementMaskWords);
            if (matches.empty()) {
                if (logRule) {
                    const MaskVector rowObjectMaskCopy = arenaCopy(game, rowObjectOffset, game.wordCount);
                    const MaskVector rowMovementMaskCopy = rowMovementMask != nullptr
                        ? arenaCopy(game, rowMovementOffset, game.movementWordCount)
                        : MaskVector{};
                    std::ostringstream stream;
                    stream << "line=" << rule.lineNumber
                           << " row=" << rowIndex
                           << " matches=0"
                           << " object_mask=" << describeObjects(session, rowObjectMaskCopy)
                           << " movement_mask=" << describeMovements(session, rowMovementMaskCopy);
                    ruleDebugLog(stream.str());
                }
                return {};
            }
            if (logRule) {
                std::ostringstream stream;
                stream << "line=" << rule.lineNumber
                       << " row=" << rowIndex
                       << " matches=" << matches.size()
                       << " starts=" << formatMatchList(matches, session.liveLevel.height);
                ruleDebugLog(stream.str());
            }
            std::vector<RowMatch> wrappedMatches;
            wrappedMatches.reserve(matches.size());
            for (const int32_t startIndex : matches) {
                wrappedMatches.push_back(RowMatch{startIndex});
            }
            rowMatches.push_back(std::move(wrappedMatches));
        } else {
            const Game& game = *session.game;
            const MaskOffset rowObjectOffset = rowIndex < rule.cellRowMasksCount
                ? game.cellRowMaskOffsets[rule.cellRowMasksFirst + rowIndex]
                : rule.ruleMask;
            const MaskWord* rowObjectMask = maskPtr(game, rowObjectOffset);
            const MaskOffset rowMovementOffset = rowIndex < rule.cellRowMasksMovementsCount
                ? game.cellRowMaskMovementsOffsets[rule.cellRowMasksMovementsFirst + rowIndex]
                : kNullMaskOffset;
            const MaskWord* rowMovementMask = maskPtr(game, rowMovementOffset);
            const uint32_t rowMovementMaskWords = rowMovementMask != nullptr ? game.movementWordCount : 0;
            auto matches = collectEllipsisRowMatches(session, row, rule.direction,
                                                     rowObjectMask, game.wordCount,
                                                     rowMovementMask, rowMovementMaskWords);
            if (matches.empty()) {
                if (logRule) {
                    std::ostringstream stream;
                    stream << "line=" << rule.lineNumber
                           << " row=" << rowIndex
                           << " matches=0 ellipsis=" << ellipsisCount;
                    ruleDebugLog(stream.str());
                }
                return {};
            }
            if (logRule) {
                std::ostringstream stream;
                stream << "line=" << rule.lineNumber
                       << " row=" << rowIndex
                       << " matches=" << matches.size()
                       << " ellipsis=" << ellipsisCount;
                ruleDebugLog(stream.str());
            }
            rowMatches.push_back(std::move(matches));
        }
    }

    std::vector<RuleMatch> tuples(1);
    for (const auto& matches : rowMatches) {
        std::vector<RuleMatch> newTuples;
        newTuples.reserve(tuples.size() * matches.size());
        for (const auto& match : matches) {
            for (const auto& tuple : tuples) {
                RuleMatch newTuple = tuple;
                newTuple.push_back(match);
                newTuples.push_back(std::move(newTuple));
            }
        }
        tuples = std::move(newTuples);
    }

    if (tuples.empty()) {
        return {};
    }

    bool changed = false;
    for (size_t tupleIndex = 0; tupleIndex < tuples.size(); ++tupleIndex) {
        const auto& tuple = tuples[tupleIndex];
        if (tupleIndex > 0) {
            bool stillMatches = true;
            for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
                const int32_t ellipsisCount = rowIndex < rule.ellipsisCount.size()
                    ? rule.ellipsisCount[rowIndex]
                    : 0;
                if (!rowMatchStillMatches(session, rule.patterns[rowIndex], ellipsisCount, tuple[rowIndex], delta)) {
                    stillMatches = false;
                    break;
                }
            }
            if (!stillMatches) {
                continue;
            }
        }
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            const int32_t ellipsisCount = rowIndex < rule.ellipsisCount.size()
                ? rule.ellipsisCount[rowIndex]
                : 0;
            changed = applyRowMatchAt(session, rule, rule.patterns[rowIndex], ellipsisCount, tuple[rowIndex], delta) || changed;
        }
    }
    // Mirror JS `Rule.prototype.tryApply`: queue commands after replacement attempts,
    // but still when the row matched (matches.length > 0 there).
    queueRuleCommands(rule, commands);

    if (ruleDebugEnabled()) {
        std::ostringstream stream;
        stream << "line=" << rule.lineNumber
               << " matched=1 changed=" << (changed ? 1 : 0)
               << " row_count=" << rule.patterns.size();
        ruleDebugLog(stream.str());
    }
    // JS returns whether any replacement ran; `matched` is used only for logging here.
    return RuleApplyOutcome{changed, changed};
}

bool collectRandomRuleMatches(Session& session, const Rule& rule, std::vector<RuleMatch>& outMatches) {
    // In JS, a "random rule group" runs random selection across *all* rules in the group,
    // regardless of whether individual rules are marked random.
    if (rule.patterns.empty()) {
        return false;
    }
    if (!ruleCanPossiblyMatch(session, rule)) {
        outMatches.clear();
        return true;
    }

    const auto [dx, dy] = directionMaskToDelta(rule.direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        return false;
    }

    std::vector<std::vector<RowMatch>> rowMatches;
    rowMatches.reserve(rule.patterns.size());
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        if (rowIndex >= rule.ellipsisCount.size() || rule.patterns[rowIndex].empty()) {
            return false;
        }
        const auto& row = rule.patterns[rowIndex];
        if (rule.ellipsisCount[rowIndex] == 0) {
            const Game& game = *session.game;
            const MaskOffset rowObjectOffset = rowIndex < rule.cellRowMasksCount
                ? game.cellRowMaskOffsets[rule.cellRowMasksFirst + rowIndex]
                : rule.ruleMask;
            const MaskWord* rowObjectMask = maskPtr(game, rowObjectOffset);
            const MaskOffset rowMovementOffset = rowIndex < rule.cellRowMasksMovementsCount
                ? game.cellRowMaskMovementsOffsets[rule.cellRowMasksMovementsFirst + rowIndex]
                : kNullMaskOffset;
            const MaskWord* rowMovementMask = maskPtr(game, rowMovementOffset);
            const uint32_t rowMovementMaskWords = rowMovementMask != nullptr ? game.movementWordCount : 0;
            auto matches = collectRowMatches(session, row, rule.direction,
                                             rowObjectMask, game.wordCount,
                                             rowMovementMask, rowMovementMaskWords);
            if (matches.empty()) {
                outMatches.clear();
                return true;
            }
            std::vector<RowMatch> wrappedMatches;
            wrappedMatches.reserve(matches.size());
            for (const int32_t startIndex : matches) {
                wrappedMatches.push_back(RowMatch{startIndex});
            }
            rowMatches.push_back(std::move(wrappedMatches));
        } else {
            const Game& game = *session.game;
            const MaskOffset rowObjectOffset = rowIndex < rule.cellRowMasksCount
                ? game.cellRowMaskOffsets[rule.cellRowMasksFirst + rowIndex]
                : rule.ruleMask;
            const MaskWord* rowObjectMask = maskPtr(game, rowObjectOffset);
            const MaskOffset rowMovementOffset = rowIndex < rule.cellRowMasksMovementsCount
                ? game.cellRowMaskMovementsOffsets[rule.cellRowMasksMovementsFirst + rowIndex]
                : kNullMaskOffset;
            const MaskWord* rowMovementMask = maskPtr(game, rowMovementOffset);
            const uint32_t rowMovementMaskWords = rowMovementMask != nullptr ? game.movementWordCount : 0;
            auto matches = collectEllipsisRowMatches(session, row, rule.direction,
                                                     rowObjectMask, game.wordCount,
                                                     rowMovementMask, rowMovementMaskWords);
            if (matches.empty()) {
                outMatches.clear();
                return true;
            }
            rowMatches.push_back(std::move(matches));
        }
    }

    outMatches.assign(1, {});
    for (const auto& matches : rowMatches) {
        std::vector<RuleMatch> newTuples;
        newTuples.reserve(outMatches.size() * matches.size());
        for (const auto& match : matches) {
            for (const auto& tuple : outMatches) {
                RuleMatch newTuple = tuple;
                newTuple.push_back(match);
                newTuples.push_back(std::move(newTuple));
            }
        }
        outMatches = std::move(newTuples);
    }
    return true;
}

bool applyRandomRuleGroup(Session& session, const std::vector<Rule>& group, CommandState& commands) {
    struct Candidate {
        const Rule* rule = nullptr;
        RuleMatch tuple;
    };

    std::vector<Candidate> candidates;
    std::vector<RuleMatch> matches;
    for (const auto& rule : group) {
        if (!collectRandomRuleMatches(session, rule, matches)) {
            continue;
        }
        for (const auto& tuple : matches) {
            candidates.push_back(Candidate{&rule, tuple});
        }
    }

    if (candidates.empty()) {
        return false;
    }

    const auto sessionHashFilter = randomDebugSessionHashFilter();
    const auto boardHashFilter = randomDebugBoardHashFilter();
    const uint64_t sessionHashBeforeRandom = (randomDebugEnabled() && sessionHashFilter.has_value())
        ? hashSession64(session)
        : 0;
    std::optional<std::string> serializedBeforeRandom;
    if (randomDebugEnabled() && (boardHashFilter.has_value() || !randomDebugSubstringFilter().empty())) {
        serializedBeforeRandom = serializeTestString(session);
    }

    bool shouldLogRandom = randomDebugEnabled();
    if (shouldLogRandom && sessionHashFilter.has_value()) {
        shouldLogRandom = sessionHashBeforeRandom == *sessionHashFilter;
    }
    if (shouldLogRandom && boardHashFilter.has_value()) {
        const uint64_t serializedHash = fnv1a64(
            reinterpret_cast<const uint8_t*>(serializedBeforeRandom->data()),
            serializedBeforeRandom->size()
        );
        shouldLogRandom = serializedHash == *boardHashFilter;
        if (shouldLogRandom) {
            const std::string_view substring = randomDebugSubstringFilter();
            if (!substring.empty()) {
                shouldLogRandom = serializedBeforeRandom->find(substring) != std::string::npos;
            }
        }
    } else if (shouldLogRandom) {
        const std::string_view substring = randomDebugSubstringFilter();
        if (!substring.empty()) {
            shouldLogRandom = serializedBeforeRandom->find(substring) != std::string::npos;
        }
    }

    const double randomValue = randomUniform(session.randomState);
    const size_t chosenIndex = std::min(
        candidates.size() - 1,
        static_cast<size_t>(std::floor(randomValue * static_cast<double>(candidates.size())))
    );
    const Candidate& chosen = candidates[chosenIndex];
    if (shouldLogRandom) {
        std::ostringstream stream;
        stream << "group_line=" << group[0].lineNumber
               << " candidate_count=" << candidates.size()
               << " random=" << randomValue
               << " chosen_index=" << chosenIndex;
        for (size_t index = 0; index < candidates.size(); ++index) {
            stream << " candidate[" << index << "]={line=" << candidates[index].rule->lineNumber << ",tuple=";
            for (size_t tupleIndex = 0; tupleIndex < candidates[index].tuple.size(); ++tupleIndex) {
                if (tupleIndex > 0) {
                    stream << ",";
                }
                stream << "[";
                for (size_t rowMatchIndex = 0; rowMatchIndex < candidates[index].tuple[tupleIndex].size(); ++rowMatchIndex) {
                    if (rowMatchIndex > 0) {
                        stream << "/";
                    }
                    stream << candidates[index].tuple[tupleIndex][rowMatchIndex];
                }
                stream << "]";
            }
            stream << "}";
        }
        randomDebugLog(stream.str());
    }
    queueRuleCommands(*chosen.rule, commands);
    const auto [dx, dy] = directionMaskToDelta(chosen.rule->direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        return false;
    }

    bool changed = false;
    for (size_t rowIndex = 0; rowIndex < chosen.tuple.size() && rowIndex < chosen.rule->patterns.size(); ++rowIndex) {
        const int32_t ellipsisCount = rowIndex < chosen.rule->ellipsisCount.size()
            ? chosen.rule->ellipsisCount[rowIndex]
            : 0;
        changed = applyRowMatchAt(
            session,
            *chosen.rule,
            chosen.rule->patterns[rowIndex],
            ellipsisCount,
            chosen.tuple[rowIndex],
            delta
        ) || changed;
    }
    if (changed) {
        std::ostringstream stream;
        stream << "line=" << chosen.rule->lineNumber
               << " matched=1 changed=1 random=1 candidate_count=" << candidates.size();
        ruleDebugLog(stream.str());
    }
    return changed;
}

bool applyRuleGroup(Session& session, const std::vector<Rule>& group, CommandState& commands) {
    if (group.empty()) {
        return false;
    }
    if (group[0].isRandom) {
        const bool changed = applyRandomRuleGroup(session, group, commands);
        if (changed) {
            rebuildMasks(session);
        }
        return changed;
    }
    bool hasChanges = false;
    bool madeChange = true;
    int loopCount = 0;
    rebuildMasks(session);
    while (madeChange && loopCount++ < 200) {
        madeChange = false;
        for (const auto& rule : group) {
            addCounter(gRuntimeCounters.rulesVisited);
            if (!ruleCanPossiblyMatch(session, rule)) {
                addCounter(gRuntimeCounters.rulesSkippedByMask);
                if (ruleDebugLineFilterMatches(rule.lineNumber)) {
                    std::ostringstream stream;
                    stream << "line=" << rule.lineNumber
                           << " skip reason=rule-mask"
                           << " rule_mask=" << describeObjects(session, arenaCopy(*session.game, rule.ruleMask, session.game->wordCount))
                           << " board_mask=" << describeObjects(session, session.boardMask);
                    ruleDebugLog(stream.str());
                }
                continue;
            }
            const RuleApplyOutcome outcome = tryApplySimpleRule(session, rule, commands, true);
            madeChange = outcome.changed || madeChange;
            if (outcome.changed) {
                rebuildMasks(session);
            }
        }
        hasChanges = hasChanges || madeChange;
    }
    return hasChanges;
}

std::optional<int32_t> lookupLoopPoint(const LoopPointTable& loopPoint, int32_t index) {
    if (index < 0 || static_cast<size_t>(index) >= loopPoint.entries.size()) {
        return std::nullopt;
    }
    return loopPoint.entries[static_cast<size_t>(index)];
}

bool applyRuleGroups(
    Session& session,
    const std::vector<std::vector<Rule>>& groups,
    const LoopPointTable& loopPoint,
    CommandState& commands,
    const std::vector<bool>* bannedGroups
) {
    bool loopPropagated = false;
    bool hasChanges = false;
    int32_t loopCount = 0;
    int32_t groupIndex = 0;
    const int32_t groupCount = static_cast<int32_t>(groups.size());
    while (groupIndex < groupCount) {
        bool groupChanged = false;
        if (bannedGroups == nullptr
            || static_cast<size_t>(groupIndex) >= bannedGroups->size()
            || !(*bannedGroups)[static_cast<size_t>(groupIndex)]) {
            groupChanged = applyRuleGroup(session, groups[static_cast<size_t>(groupIndex)], commands);
        }
        loopPropagated = groupChanged || loopPropagated;
        hasChanges = groupChanged || hasChanges;

        if (loopPropagated) {
            if (const auto target = lookupLoopPoint(loopPoint, groupIndex); target.has_value()) {
                groupIndex = *target;
                loopPropagated = false;
                if (++loopCount > 200) {
                    break;
                }
                continue;
            }
        }

        ++groupIndex;
        if (groupIndex == groupCount && loopPropagated) {
            if (const auto target = lookupLoopPoint(loopPoint, groupIndex); target.has_value()) {
                groupIndex = *target;
                loopPropagated = false;
                if (++loopCount > 200) {
                    break;
                }
            }
        }
    }
    return hasChanges;
}

size_t countNonZeroWords(const std::vector<int32_t>& values) {
    return std::count_if(values.begin(), values.end(), [](int32_t value) { return value != 0; });
}

#if PS_MASK_WORD_BITS == 64
size_t countNonZeroWords(const MaskVector& values) {
    return std::count_if(values.begin(), values.end(), [](MaskWord value) { return value != 0; });
}
#endif

void rebuildObjectCellIndex(Session& session) {
    const int32_t objectCount = session.game->objectCount;
    const int32_t stride = session.game->strideObject;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    session.objectCellBitTileCount = tileCount;
    session.objectCellBits.assign(static_cast<size_t>(objectCount) * cellWordCount, 0);
    session.objectCellCounts.assign(static_cast<size_t>(std::max(objectCount, 0)), 0);
    if (objectCount <= 0 || stride <= 0 || tileCount <= 0 || cellWordCount == 0) {
        session.objectCellIndexDirty = false;
        return;
    }

    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        const size_t cellBase = static_cast<size_t>(tileIndex * stride);
        const size_t bitWord = static_cast<size_t>(tileIndex >> 6);
        const uint64_t bitMask = uint64_t{1} << static_cast<uint32_t>(tileIndex & 63);
        for (int32_t word = 0; word < stride; ++word) {
            MaskWordUnsigned bits = static_cast<MaskWordUnsigned>(session.liveLevel.objects[cellBase + static_cast<size_t>(word)]);
            while (bits != 0) {
                const int32_t bit = maskWordCountTrailingZeros(bits);
                const int32_t objectId = word * static_cast<int32_t>(kMaskWordBits) + bit;
                if (objectId < objectCount) {
                    session.objectCellBits[static_cast<size_t>(objectId) * cellWordCount + bitWord] |= bitMask;
                    ++session.objectCellCounts[static_cast<size_t>(objectId)];
                }
                bits &= bits - 1;
            }
        }
    }
    session.objectCellIndexDirty = false;
}

// Incremental rebuildMasks: setCellObjects/setCellMovements already OR new
// bits into the row/col/board masks on the write path, so the only case a
// rebuild is needed is when bits were *cleared*. The set-paths mark those
// rows/columns dirty. This function rebuilds exactly the dirty slices from
// scratch and leaves the rest untouched. On a clean session (anyMasksDirty
// == false) this is a branch and return.
void rebuildMasks(Session& session) {
    addCounter(gRuntimeCounters.maskRebuildCalls);
    const int32_t objectStride = session.game->strideObject;
    const int32_t movementStride = session.game->strideMovement;
    const int32_t width = session.liveLevel.width;
    const int32_t height = session.liveLevel.height;
    const int32_t tileCount = width * height;
    const size_t cellWordCount = static_cast<size_t>((tileCount + 63) / 64);
    if (!session.anyMasksDirty
        && !session.objectCellIndexDirty
        && session.objectCellBitTileCount == tileCount
        && session.objectCellBits.size() == static_cast<size_t>(session.game->objectCount) * cellWordCount
        && session.objectCellCounts.size() == static_cast<size_t>(session.game->objectCount)) {
        return;
    }

    // Reshape storage on first call / level-dimension change. We compare
    // against the expected sizes and (re)allocate uniformly if anything is
    // off — this also drives "mark everything dirty on first use" via the
    // fact that we populate a fresh row/col mask set below.
    const size_t rowObjectSize = static_cast<size_t>(height * objectStride);
    const size_t columnObjectSize = static_cast<size_t>(width * objectStride);
    const size_t rowMovementSize = static_cast<size_t>(height * movementStride);
    const size_t columnMovementSize = static_cast<size_t>(width * movementStride);
    bool sizeChanged = false;
    auto ensureSize = [&sizeChanged](MaskVector& v, size_t n) {
        if (v.size() != n) { v.assign(n, 0); sizeChanged = true; }
    };
    ensureSize(session.rowMasks, rowObjectSize);
    ensureSize(session.columnMasks, columnObjectSize);
    ensureSize(session.boardMask, static_cast<size_t>(objectStride));
    ensureSize(session.rowMovementMasks, rowMovementSize);
    ensureSize(session.columnMovementMasks, columnMovementSize);
    ensureSize(session.boardMovementMask, static_cast<size_t>(movementStride));
    if (session.dirtyObjectRows.size() != static_cast<size_t>(height)) {
        session.dirtyObjectRows.assign(static_cast<size_t>(height), 1);
        sizeChanged = true;
    }
    if (session.dirtyObjectColumns.size() != static_cast<size_t>(width)) {
        session.dirtyObjectColumns.assign(static_cast<size_t>(width), 1);
        sizeChanged = true;
    }
    if (session.dirtyMovementRows.size() != static_cast<size_t>(height)) {
        session.dirtyMovementRows.assign(static_cast<size_t>(height), 1);
        sizeChanged = true;
    }
    if (session.dirtyMovementColumns.size() != static_cast<size_t>(width)) {
        session.dirtyMovementColumns.assign(static_cast<size_t>(width), 1);
        sizeChanged = true;
    }
    if (sizeChanged) {
        std::fill(session.dirtyObjectRows.begin(), session.dirtyObjectRows.end(), 1);
        std::fill(session.dirtyObjectColumns.begin(), session.dirtyObjectColumns.end(), 1);
        std::fill(session.dirtyMovementRows.begin(), session.dirtyMovementRows.end(), 1);
        std::fill(session.dirtyMovementColumns.begin(), session.dirtyMovementColumns.end(), 1);
        session.dirtyObjectBoard = true;
        session.dirtyMovementBoard = true;
        session.objectCellIndexDirty = true;
        session.anyMasksDirty = true;
    }

    if (session.objectCellIndexDirty
        || session.objectCellBitTileCount != width * height
        || session.objectCellBits.size() != static_cast<size_t>(session.game->objectCount) * objectCellWordCount(session)
        || session.objectCellCounts.size() != static_cast<size_t>(session.game->objectCount)) {
        rebuildObjectCellIndex(session);
    }

    if (!session.anyMasksDirty) {
        return;
    }
    addCounter(gRuntimeCounters.maskRebuildDirtyCalls);

    // ---- Object masks ---------------------------------------------------
    // Rebuild each dirty row: zero its slice, then OR every tile in that row.
    for (int32_t y = 0; y < height; ++y) {
        if (!session.dirtyObjectRows[static_cast<size_t>(y)]) continue;
        addCounter(gRuntimeCounters.maskRebuildRows);
        MaskWord* rowStart = session.rowMasks.data() + static_cast<size_t>(y * objectStride);
        std::fill(rowStart, rowStart + objectStride, 0);
        for (int32_t x = 0; x < width; ++x) {
            const size_t objectBase = static_cast<size_t>((x * height + y) * objectStride);
            const MaskWord* cell = session.liveLevel.objects.data() + objectBase;
            for (int32_t word = 0; word < objectStride; ++word) {
                rowStart[word] |= cell[word];
            }
        }
        session.dirtyObjectRows[static_cast<size_t>(y)] = 0;
    }
    for (int32_t x = 0; x < width; ++x) {
        if (!session.dirtyObjectColumns[static_cast<size_t>(x)]) continue;
        addCounter(gRuntimeCounters.maskRebuildColumns);
        MaskWord* colStart = session.columnMasks.data() + static_cast<size_t>(x * objectStride);
        std::fill(colStart, colStart + objectStride, 0);
        for (int32_t y = 0; y < height; ++y) {
            const size_t objectBase = static_cast<size_t>((x * height + y) * objectStride);
            const MaskWord* cell = session.liveLevel.objects.data() + objectBase;
            for (int32_t word = 0; word < objectStride; ++word) {
                colStart[word] |= cell[word];
            }
        }
        session.dirtyObjectColumns[static_cast<size_t>(x)] = 0;
    }
    if (session.dirtyObjectBoard) {
        // boardMask = OR over all rowMasks slices.
        std::fill(session.boardMask.begin(), session.boardMask.end(), 0);
        for (int32_t y = 0; y < height; ++y) {
            const MaskWord* rowStart = session.rowMasks.data() + static_cast<size_t>(y * objectStride);
            for (int32_t word = 0; word < objectStride; ++word) {
                session.boardMask[static_cast<size_t>(word)] |= rowStart[word];
            }
        }
        session.dirtyObjectBoard = false;
    }

    // ---- Movement masks -------------------------------------------------
    for (int32_t y = 0; y < height; ++y) {
        if (!session.dirtyMovementRows[static_cast<size_t>(y)]) continue;
        addCounter(gRuntimeCounters.maskRebuildRows);
        MaskWord* rowStart = session.rowMovementMasks.data() + static_cast<size_t>(y * movementStride);
        std::fill(rowStart, rowStart + movementStride, 0);
        for (int32_t x = 0; x < width; ++x) {
            const size_t movementBase = static_cast<size_t>((x * height + y) * movementStride);
            const MaskWord* cell = session.liveMovements.data() + movementBase;
            for (int32_t word = 0; word < movementStride; ++word) {
                rowStart[word] |= cell[word];
            }
        }
        session.dirtyMovementRows[static_cast<size_t>(y)] = 0;
    }
    for (int32_t x = 0; x < width; ++x) {
        if (!session.dirtyMovementColumns[static_cast<size_t>(x)]) continue;
        addCounter(gRuntimeCounters.maskRebuildColumns);
        MaskWord* colStart = session.columnMovementMasks.data() + static_cast<size_t>(x * movementStride);
        std::fill(colStart, colStart + movementStride, 0);
        for (int32_t y = 0; y < height; ++y) {
            const size_t movementBase = static_cast<size_t>((x * height + y) * movementStride);
            const MaskWord* cell = session.liveMovements.data() + movementBase;
            for (int32_t word = 0; word < movementStride; ++word) {
                colStart[word] |= cell[word];
            }
        }
        session.dirtyMovementColumns[static_cast<size_t>(x)] = 0;
    }
    if (session.dirtyMovementBoard) {
        std::fill(session.boardMovementMask.begin(), session.boardMovementMask.end(), 0);
        for (int32_t y = 0; y < height; ++y) {
            const MaskWord* rowStart = session.rowMovementMasks.data() + static_cast<size_t>(y * movementStride);
            for (int32_t word = 0; word < movementStride; ++word) {
                session.boardMovementMask[static_cast<size_t>(word)] |= rowStart[word];
            }
        }
        session.dirtyMovementBoard = false;
    }

    session.anyMasksDirty = false;
}

std::vector<uint8_t> buildSessionHashBytes(const Session& session) {
    std::vector<uint8_t> bytes;
    auto append = [&bytes](const auto& value) {
        const auto* data = reinterpret_cast<const uint8_t*>(&value);
        bytes.insert(bytes.end(), data, data + sizeof(value));
    };

    append(session.preparedSession.currentLevelIndex);
    append(session.preparedSession.titleScreen);
    append(session.preparedSession.textMode);
    append(session.preparedSession.winning);
    append(session.pendingAgain);
    append(session.randomState.i);
    append(session.randomState.j);
    append(session.randomState.valid);
    const auto* randomBytes = reinterpret_cast<const uint8_t*>(session.randomState.s.data());
    bytes.insert(bytes.end(), randomBytes, randomBytes + session.randomState.s.size() * sizeof(uint8_t));

    const auto& objects = session.liveLevel.objects;
    const auto* objectBytes = reinterpret_cast<const uint8_t*>(objects.data());
    bytes.insert(bytes.end(), objectBytes, objectBytes + objects.size() * sizeof(MaskWord));
    const auto& movements = session.liveMovements;
    const auto* movementBytes = reinterpret_cast<const uint8_t*>(movements.data());
    bytes.insert(bytes.end(), movementBytes, movementBytes + movements.size() * sizeof(MaskWord));
    bytes.insert(bytes.end(), session.preparedSession.loadedLevelSeed.begin(), session.preparedSession.loadedLevelSeed.end());
    return bytes;
}

void appendHashBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ull;
    }
}

template <typename T>
void appendHashValue(uint64_t& hash, const T& value) {
    appendHashBytes(hash, &value, sizeof(value));
}

uint64_t hashSession64NoAlloc(const Session& session, uint64_t seed) {
    uint64_t hash = seed;

    appendHashValue(hash, session.preparedSession.currentLevelIndex);
    appendHashValue(hash, session.preparedSession.titleScreen);
    appendHashValue(hash, session.preparedSession.textMode);
    appendHashValue(hash, session.preparedSession.winning);
    appendHashValue(hash, session.pendingAgain);
    appendHashValue(hash, session.randomState.i);
    appendHashValue(hash, session.randomState.j);
    appendHashValue(hash, session.randomState.valid);
    appendHashBytes(hash, session.randomState.s.data(), session.randomState.s.size() * sizeof(uint8_t));

    const auto& objects = session.liveLevel.objects;
    appendHashBytes(hash, objects.data(), objects.size() * sizeof(MaskWord));
    const auto& movements = session.liveMovements;
    appendHashBytes(hash, movements.data(), movements.size() * sizeof(MaskWord));
    appendHashBytes(hash, session.preparedSession.loadedLevelSeed.data(), session.preparedSession.loadedLevelSeed.size());

    return hash;
}

ps_hash128 hashSession128NoAlloc(const Session& session) {
    ps_hash128 result{};
    result.lo = hashSession64NoAlloc(session, 1469598103934665603ull);
    result.hi = hashSession64NoAlloc(session, 7809847782465536322ull);
    return result;
}

std::string escapeJson(std::string_view input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char ch : input) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out.push_back(ch); break;
        }
    }
    return out;
}

bool showContinueOptionOnTitleScreen(const PreparedSession& prepared) {
    return prepared.currentLevelIndex > 0 || prepared.currentLevelTarget.has_value();
}

void markAllMasksDirty(Session& session) {
    std::fill(session.dirtyObjectRows.begin(), session.dirtyObjectRows.end(), 1);
    std::fill(session.dirtyObjectColumns.begin(), session.dirtyObjectColumns.end(), 1);
    std::fill(session.dirtyMovementRows.begin(), session.dirtyMovementRows.end(), 1);
    std::fill(session.dirtyMovementColumns.begin(), session.dirtyMovementColumns.end(), 1);
    session.dirtyObjectBoard = true;
    session.dirtyMovementBoard = true;
    session.objectCellIndexDirty = true;
    session.anyMasksDirty = true;
}

// Variant for callers that bulk-zero liveMovements without going through
// setCellMovements. Row/col/board movement masks retain stale OR'd bits
// until the next rebuild; mark them all dirty so the next rebuildMasks()
// recomputes from the current (zeroed or just-seeded) movement state.
void markAllMovementMasksDirty(Session& session) {
    std::fill(session.dirtyMovementRows.begin(), session.dirtyMovementRows.end(), 1);
    std::fill(session.dirtyMovementColumns.begin(), session.dirtyMovementColumns.end(), 1);
    session.dirtyMovementBoard = true;
    session.anyMasksDirty = true;
}

void restoreSnapshot(Session& session, const Session::UndoSnapshot& snapshot, bool restoreRandomState) {
    session.preparedSession = snapshot.preparedSession;
    session.liveLevel = snapshot.liveLevel;
    session.liveMovements = snapshot.liveMovements;
    session.rigidGroupIndexMasks = snapshot.rigidGroupIndexMasks;
    session.rigidMovementAppliedMasks = snapshot.rigidMovementAppliedMasks;
    if (restoreRandomState) {
        session.randomState = snapshot.randomState;
    }
    session.pendingAgain = false;
    markAllMasksDirty(session);
    rebuildMasks(session);
}

Session::UndoSnapshot makeUndoSnapshot(const Session& session) {
    return Session::UndoSnapshot{
        session.preparedSession,
        session.liveLevel,
        session.liveMovements,
        session.rigidGroupIndexMasks,
        session.rigidMovementAppliedMasks,
        session.randomState,
    };
}

void pushUndoSnapshot(Session& session) {
    session.undoStack.push_back(makeUndoSnapshot(session));
    session.canUndo = true;
}

void restoreRestartTarget(Session& session) {
    session.liveLevel = session.preparedSession.level;
    if (!session.preparedSession.restart.objects.empty()) {
        session.liveLevel.width = session.preparedSession.restart.width;
        session.liveLevel.height = session.preparedSession.restart.height;
        session.liveLevel.objects = session.preparedSession.restart.objects;
        session.preparedSession.oldFlickscreenDat = session.preparedSession.restart.oldFlickscreenDat;
    }
    session.liveMovements.assign(static_cast<size_t>(session.liveLevel.width * session.liveLevel.height * session.game->strideMovement), 0);
    session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
    session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
    session.pendingAgain = false;
    markAllMasksDirty(session);
    rebuildMasks(session);
}

// Raw-pointer variant used after WinCondition mask migration: filter lives
// in Game::maskArena with width wordCount.
bool matchesFilter(const MaskWord* filter, uint32_t filterCount,
                   bool aggregate,
                   const MaskWord* cell, uint32_t cellCount) {
    return aggregate
        ? bitsSetInArray(filter, filterCount, cell, cellCount)
        : anyBitsInCommon(filter, filterCount, cell, cellCount);
}

bool evaluateWinConditions(const Session& session) {
    if (session.game->winConditions.empty()) {
        return false;
    }

    const Game& game = *session.game;
    const uint32_t wordCount = game.wordCount;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    for (const auto& condition : game.winConditions) {
        const MaskWord* filter1 = maskPtr(game, condition.filter1);
        const MaskWord* filter2 = maskPtr(game, condition.filter2);
        bool rulePassed = true;
        switch (condition.quantifier) {
            case -1: {
                for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
                    const MaskWord* cell = getCellObjectsPtr(session, tileIndex);
                    if (matchesFilter(filter1, wordCount, condition.aggr1, cell, wordCount)
                        && matchesFilter(filter2, wordCount, condition.aggr2, cell, wordCount)) {
                        rulePassed = false;
                        break;
                    }
                }
                break;
            }
            case 0: {
                bool passedTest = false;
                for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
                    const MaskWord* cell = getCellObjectsPtr(session, tileIndex);
                    if (matchesFilter(filter1, wordCount, condition.aggr1, cell, wordCount)
                        && matchesFilter(filter2, wordCount, condition.aggr2, cell, wordCount)) {
                        passedTest = true;
                        break;
                    }
                }
                rulePassed = passedTest;
                break;
            }
            case 1: {
                for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
                    const MaskWord* cell = getCellObjectsPtr(session, tileIndex);
                    if (matchesFilter(filter1, wordCount, condition.aggr1, cell, wordCount)
                        && !matchesFilter(filter2, wordCount, condition.aggr2, cell, wordCount)) {
                        rulePassed = false;
                        break;
                    }
                }
                break;
            }
            default:
                rulePassed = false;
                break;
        }
        if (!rulePassed) {
            return false;
        }
    }

    return true;
}

bool advanceToNextLevel(Session& session) {
    if (session.game->levels.empty()) {
        return false;
    }

    if (session.preparedSession.currentLevelIndex < static_cast<int32_t>(session.game->levels.size()) - 1) {
        session.preparedSession.currentLevelIndex += 1;
        session.preparedSession.currentLevelTarget.reset();
        session.preparedSession.titleScreen = false;
        session.preparedSession.level = session.game->levels[static_cast<size_t>(session.preparedSession.currentLevelIndex)];
        session.preparedSession.textMode = session.preparedSession.level.isMessage;
        if (!session.preparedSession.textMode) {
        session.preparedSession.titleMode = 0;
        session.preparedSession.titleSelection = showContinueOptionOnTitleScreen(session.preparedSession) ? 1 : 0;
    }
    session.preparedSession.titleSelected = false;
    session.preparedSession.messageSelected = false;
    session.preparedSession.messageText.clear();
    session.preparedSession.winning = false;
        if (session.preparedSession.textMode) {
            session.liveMovements.assign(static_cast<size_t>(session.liveLevel.width * session.liveLevel.height * session.game->strideMovement), 0);
            session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
            session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
            session.pendingAgain = false;
            markAllMovementMasksDirty(session);
            rebuildMasks(session);
            session.undoStack.clear();
            session.canUndo = false;
            return true;
        }
        session.preparedSession.restart.width = session.preparedSession.level.width;
        session.preparedSession.restart.height = session.preparedSession.level.height;
        session.preparedSession.restart.objects = session.preparedSession.level.objects;
        session.preparedSession.restart.oldFlickscreenDat = session.preparedSession.oldFlickscreenDat;
        restoreRestartTarget(session);
        ::puzzlescript::runRulesOnLevelStart(session);
        session.undoStack.clear();
        session.canUndo = false;
        return true;
    }

    session.preparedSession.currentLevelIndex = 0;
    session.preparedSession.currentLevelTarget.reset();
    session.preparedSession.titleScreen = true;
    session.preparedSession.textMode = true;
    session.preparedSession.titleMode = showContinueOptionOnTitleScreen(session.preparedSession) ? 1 : 0;
    session.preparedSession.titleSelection = showContinueOptionOnTitleScreen(session.preparedSession) ? 1 : 0;
    session.preparedSession.titleSelected = false;
    session.preparedSession.messageSelected = false;
    session.preparedSession.messageText.clear();
    session.preparedSession.winning = false;
    session.liveMovements.assign(static_cast<size_t>(session.liveLevel.width * session.liveLevel.height * session.game->strideMovement), 0);
    session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
    session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
    session.pendingAgain = false;
    markAllMovementMasksDirty(session);
    rebuildMasks(session);
    session.undoStack.clear();
    session.canUndo = false;
    return true;
}

void resetToPrepared(Session& session) {
    session.liveLevel = session.preparedSession.level;
    session.liveMovements.assign(static_cast<size_t>(session.liveLevel.width * session.liveLevel.height * session.game->strideMovement), 0);
    session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
    session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
    session.canUndo = false;
    session.undoStack.clear();
    session.pendingAgain = false;
    markAllMasksDirty(session);
    if (session.preparedSession.hasRandomState
        && session.preparedSession.randomStateS.size() == session.randomState.s.size()) {
        session.randomState.valid = session.preparedSession.randomStateValid;
        session.randomState.i = session.preparedSession.randomStateI;
        session.randomState.j = session.preparedSession.randomStateJ;
        std::copy(
            session.preparedSession.randomStateS.begin(),
            session.preparedSession.randomStateS.end(),
            session.randomState.s.begin()
        );
    } else {
        seedRandomState(session.randomState, session.preparedSession.loadedLevelSeed);
    }
    rebuildMasks(session);
}

} // namespace

void runRulesOnLevelStart(Session& session);
void runRulesOnLevelStart(Session& session, RuntimeStepOptions options);
bool wouldAgainChange(Session& session, bool* outWouldModify = nullptr, bool emitAudio = true);
void settlePendingAgain(Session& session);

std::unique_ptr<Error> loadGameFromJson(std::string_view jsonText, std::shared_ptr<const Game>& outGame) {
    try {
        json::Value root = json::parse(jsonText);
        const auto& rootObject = root.asObject();
        const auto& gameValue = requireField(rootObject, "game");
        const auto& gameObject = gameValue.asObject();

        auto game = std::make_shared<Game>();
        game->schemaVersion = toInt(requireField(rootObject, "schema_version"));

        const auto& strides = requireField(gameObject, "strides").asObject();
        game->strideObject = toInt(requireField(strides, "object"));
        game->strideMovement = toInt(requireField(strides, "movement"));
        game->wordCount = static_cast<uint32_t>(game->strideObject);
        game->movementWordCount = static_cast<uint32_t>(game->strideMovement);
        game->maskArena.reserve(1024);
        game->layerCount = toInt(requireField(strides, "layers"));
        game->objectCount = toInt(requireField(gameObject, "object_count"));

        const auto& colors = requireField(gameObject, "colors").asObject();
        game->foregroundColor = toString(requireField(colors, "foreground"));
        game->backgroundColor = toString(requireField(colors, "background"));

        const auto& background = requireField(gameObject, "background").asObject();
        game->backgroundId = toInt(requireField(background, "id"));
        game->backgroundLayer = toInt(requireField(background, "layer"));

        if (const auto metadataPairs = gameObject.find("metadata_pairs"); metadataPairs != gameObject.end()) {
            game->metadataPairs = parseStringVector(metadataPairs->second);
        }
        if (const auto metadataMap = gameObject.find("metadata_map"); metadataMap != gameObject.end()) {
            game->metadataMap = parseStringMap(metadataMap->second);
        }
        if (const auto metadataLines = gameObject.find("metadata_lines"); metadataLines != gameObject.end()) {
            game->metadataLines = parseIntMap(metadataLines->second);
        }

        game->idDict = parseStringVector(requireField(gameObject, "id_dict"));
        if (const auto glyphOrder = gameObject.find("glyph_order"); glyphOrder != gameObject.end()) {
            game->glyphOrder = parseStringVector(glyphOrder->second);
        }
        if (const auto glyphDict = gameObject.find("glyph_dict"); glyphDict != gameObject.end()) {
            game->glyphMaskTable = parseNamedMaskTable(*game, glyphDict->second);
        }
        game->objectsById.resize(static_cast<size_t>(game->objectCount));
        const auto& objectsArray = requireField(gameObject, "objects").asArray();
        for (size_t objectIndex = 0; objectIndex < objectsArray.size(); ++objectIndex) {
            try {
                const auto& object = objectsArray[objectIndex].asObject();
                ObjectDef entry;
                entry.name = toString(requireField(object, "name"));
                entry.id = toInt(requireField(object, "id"));
                entry.layer = toInt(requireField(object, "layer"));
                entry.colors = parseStringVector(requireField(object, "colors"));
                entry.sprite = parseSprite(requireField(object, "spritematrix"));
                if (entry.id >= 0 && static_cast<size_t>(entry.id) < game->objectsById.size()) {
                    game->objectsById[static_cast<size_t>(entry.id)] = std::move(entry);
                }
            } catch (const std::exception& error) {
                throw json::ParseError("Failed parsing object[" + std::to_string(objectIndex) + "]: " + error.what());
            }
        }

        if (const auto collisionLayers = gameObject.find("collision_layers"); collisionLayers != gameObject.end()) {
            for (const auto& layerValue : collisionLayers->second.asArray()) {
                game->collisionLayers.push_back(parseStringVector(layerValue));
            }
        }
        if (const auto layerMasks = gameObject.find("layer_masks"); layerMasks != gameObject.end()) {
            for (const auto& maskValue : layerMasks->second.asArray()) {
                game->layerMaskOffsets.push_back(storeMaskWords(*game, parseMaskVector(maskValue)));
            }
        }
        if (const auto objectMasks = gameObject.find("object_masks"); objectMasks != gameObject.end()) {
            game->objectMaskTable = parseNamedMaskTable(*game, objectMasks->second);
        }
        if (const auto aggregateMasks = gameObject.find("aggregate_masks"); aggregateMasks != gameObject.end()) {
            game->aggregateMaskTable = parseNamedMaskTable(*game, aggregateMasks->second);
        }
        if (const auto playerMask = gameObject.find("player_mask"); playerMask != gameObject.end()) {
            if (playerMask->second.isObject()) {
                const auto& playerMaskObject = playerMask->second.asObject();
                if (const auto aggregate = playerMaskObject.find("aggregate"); aggregate != playerMaskObject.end()) {
                    game->playerMaskAggregate = toBool(aggregate->second);
                }
                if (const auto mask = playerMaskObject.find("mask"); mask != playerMaskObject.end()) {
                    game->playerMask = storeMaskWords(*game, parseMaskVector(mask->second));
                }
            } else {
                game->playerMask = storeMaskWords(*game, parseMaskVector(playerMask->second));
            }
        }
        if (const auto rigid = gameObject.find("rigid"); rigid != gameObject.end()) {
            game->rigid = toBool(rigid->second);
        }
        if (const auto rigidGroups = gameObject.find("rigid_groups"); rigidGroups != gameObject.end()) {
            game->rigidGroups = parseBoolVector(rigidGroups->second);
        }
        if (const auto indexMap = gameObject.find("rigid_group_index_to_group_index"); indexMap != gameObject.end()) {
            game->rigidGroupIndexToGroupIndex = parseIntVector(indexMap->second);
        }
        if (const auto indexMap = gameObject.find("group_index_to_rigid_group_index"); indexMap != gameObject.end()) {
            game->groupIndexToRigidGroupIndex = parseIntVector(indexMap->second);
        }
        if (const auto indexMap = gameObject.find("group_number_to_rigid_group_index"); indexMap != gameObject.end()) {
            game->groupNumberToRigidGroupIndex = parseIntVector(indexMap->second);
        }
        if (const auto rules = gameObject.find("rules"); rules != gameObject.end()) {
            game->rules = parseRuleGroups(*game, rules->second);
        }
        if (const auto lateRules = gameObject.find("late_rules"); lateRules != gameObject.end()) {
            game->lateRules = parseRuleGroups(*game, lateRules->second);
        }
        if (const auto loopPoint = gameObject.find("loop_point"); loopPoint != gameObject.end()) {
            game->loopPoint = parseLoopPointTable(loopPoint->second);
        }
        if (const auto lateLoopPoint = gameObject.find("late_loop_point"); lateLoopPoint != gameObject.end()) {
            game->lateLoopPoint = parseLoopPointTable(lateLoopPoint->second);
        }
        if (const auto winconditions = gameObject.find("winconditions"); winconditions != gameObject.end()) {
            for (const auto& conditionValue : winconditions->second.asArray()) {
                game->winConditions.push_back(parseWinCondition(*game, conditionValue));
            }
        }

        const auto& levelsArray = requireField(gameObject, "levels").asArray();
        for (size_t levelIndex = 0; levelIndex < levelsArray.size(); ++levelIndex) {
            try {
                game->levels.push_back(parseLevelTemplate(levelsArray[levelIndex]));
            } catch (const std::exception& error) {
                throw json::ParseError("Failed parsing level[" + std::to_string(levelIndex) + "]: " + error.what());
            }
        }

        if (const auto sfxEvents = gameObject.find("sfx_events"); sfxEvents != gameObject.end()) {
            game->sfxEvents = parseSoundEventMap(sfxEvents->second);
        }
        if (const auto sfxCreationMasks = gameObject.find("sfx_creation_masks"); sfxCreationMasks != gameObject.end()) {
            game->sfxCreationMasks = parseSoundMaskEntries(*game, sfxCreationMasks->second);
        }
        if (const auto sfxDestructionMasks = gameObject.find("sfx_destruction_masks"); sfxDestructionMasks != gameObject.end()) {
            game->sfxDestructionMasks = parseSoundMaskEntries(*game, sfxDestructionMasks->second);
        }
        if (const auto sfxMovementMasks = gameObject.find("sfx_movement_masks"); sfxMovementMasks != gameObject.end()) {
            game->sfxMovementMasks = parseLayeredSoundMaskEntries(*game, sfxMovementMasks->second);
        }
        if (const auto sfxMovementFailureMasks = gameObject.find("sfx_movement_failure_masks"); sfxMovementFailureMasks != gameObject.end()) {
            game->sfxMovementFailureMasks = parseSoundMaskEntries(*game, sfxMovementFailureMasks->second);
        }

        if (const auto prepared = rootObject.find("prepared_session"); prepared != rootObject.end()) {
            try {
                game->preparedSession = parsePreparedSession(prepared->second);
                if (game->preparedSession.level.width == 0
                    && game->preparedSession.level.height == 0
                    && game->preparedSession.level.objects.empty()
                    && game->preparedSession.currentLevelIndex >= 0
                    && static_cast<size_t>(game->preparedSession.currentLevelIndex) < game->levels.size()) {
                    game->preparedSession.level = game->levels[static_cast<size_t>(game->preparedSession.currentLevelIndex)];
                }
            } catch (const std::exception& error) {
                throw json::ParseError("Failed parsing prepared_session: " + std::string(error.what()));
            }
        }
        outGame = std::move(game);
        return nullptr;
    } catch (const std::exception& error) {
        return std::make_unique<Error>(error.what());
    }
}

std::unique_ptr<Session> createSession(std::shared_ptr<const Game> game) {
    auto session = std::make_unique<Session>();
    session->game = std::move(game);
    session->preparedSession = session->game->preparedSession;
    session->backend = detectBestBackend();
    resetToPrepared(*session);
    return session;
}

std::unique_ptr<Session> createSessionWithLoadedLevelSeed(std::shared_ptr<const Game> game, std::string loadedLevelSeed) {
    auto session = std::make_unique<Session>();
    session->game = std::move(game);
    session->preparedSession = session->game->preparedSession;
    session->preparedSession.loadedLevelSeed = std::move(loadedLevelSeed);
    session->preparedSession.hasRandomState = false;
    session->preparedSession.randomStateValid = false;
    session->preparedSession.randomStateI = 0;
    session->preparedSession.randomStateJ = 0;
    session->preparedSession.randomStateS.clear();
    session->backend = detectBestBackend();
    resetToPrepared(*session);
    return session;
}

std::unique_ptr<Error> loadLevel(Session& session, int32_t levelIndex) {
    if (levelIndex < 0 || static_cast<size_t>(levelIndex) >= session.game->levels.size()) {
        return std::make_unique<Error>("Level index out of range");
    }

    session.preparedSession.currentLevelIndex = levelIndex;
    session.preparedSession.currentLevelTarget.reset();
    session.preparedSession.titleScreen = false;
    session.preparedSession.level = session.game->levels[static_cast<size_t>(levelIndex)];
    session.preparedSession.textMode = session.preparedSession.level.isMessage;
    session.preparedSession.titleMode = session.preparedSession.textMode
        ? (showContinueOptionOnTitleScreen(session.preparedSession) ? 1 : 0)
        : 0;
    session.preparedSession.titleSelection = showContinueOptionOnTitleScreen(session.preparedSession) ? 1 : 0;
    session.preparedSession.titleSelected = false;
    session.preparedSession.messageSelected = false;
    session.preparedSession.messageText.clear();
    session.preparedSession.winning = false;
    session.preparedSession.restart.width = session.preparedSession.level.width;
    session.preparedSession.restart.height = session.preparedSession.level.height;
    session.preparedSession.restart.objects = session.preparedSession.level.objects;
    session.preparedSession.restart.oldFlickscreenDat = session.preparedSession.oldFlickscreenDat;
    resetToPrepared(session);
    runRulesOnLevelStart(session);
    settlePendingAgain(session);
    return nullptr;
}

std::unique_ptr<Error> advanceLevel(Session& session) {
    if (session.game->levels.empty()) {
        return std::make_unique<Error>("No levels available");
    }
    (void)advanceToNextLevel(session);
    return nullptr;
}

bool restart(Session& session, RuntimeStepOptions options) {
    if (session.game->metadataMap.find("norestart") != session.game->metadataMap.end()) {
        return true;
    }
    if (options.playableUndo) {
        pushUndoSnapshot(session);
    }
    restoreRestartTarget(session);
    runRulesOnLevelStart(session, options);
    return true;
}

bool restart(Session& session) {
    return restart(session, RuntimeStepOptions{});
}

bool undo(Session& session) {
    if (session.game->metadataMap.find("noundo") != session.game->metadataMap.end()) {
        return true;
    }
    while (!session.undoStack.empty()) {
        const auto& top = session.undoStack.back();
        if (top.liveLevel.width != session.liveLevel.width
            || top.liveLevel.height != session.liveLevel.height
            || top.liveLevel.objects != session.liveLevel.objects) {
            break;
        }
        session.undoStack.pop_back();
    }
    if (session.undoStack.empty()) {
        session.canUndo = false;
        return false;
    }
    const auto snapshot = std::move(session.undoStack.back());
    session.undoStack.pop_back();
    restoreSnapshot(session, snapshot, false);
    session.canUndo = !session.undoStack.empty();
    return true;
}

uint64_t hashSession64(const Session& session) {
    return hashSession64NoAlloc(session, 1469598103934665603ull);
}

ps_hash128 hashSession128(const Session& session) {
    return hashSession128NoAlloc(session);
}

std::string serializeTestString(const Session& session) {
    std::string output;
    std::map<std::string, int32_t> seenCells;
    int32_t nextIndex = 0;
    const int32_t stride = session.game->strideObject;

    for (int32_t y = 0; y < session.liveLevel.height; ++y) {
        for (int32_t x = 0; x < session.liveLevel.width; ++x) {
            const size_t cellOffset = static_cast<size_t>((y * session.liveLevel.width + x) * stride);
            std::vector<std::string> objects;
            for (int32_t bit = 0; bit < static_cast<int32_t>(kMaskWordBits) * stride; ++bit) {
                const uint32_t word = maskWordIndex(static_cast<uint32_t>(bit));
                if ((session.liveLevel.objects[cellOffset + static_cast<size_t>(word)] & maskBit(static_cast<uint32_t>(bit))) != 0) {
                    if (static_cast<size_t>(bit) < session.game->idDict.size()) {
                        objects.push_back(session.game->idDict[static_cast<size_t>(bit)]);
                    }
                }
            }
            std::sort(objects.begin(), objects.end());
            std::string cellKey;
            for (size_t index = 0; index < objects.size(); ++index) {
                if (index > 0) {
                    cellKey.push_back(' ');
                }
                cellKey += objects[index];
            }

            auto [it, inserted] = seenCells.emplace(cellKey, nextIndex);
            if (inserted) {
                output += cellKey;
                output.push_back(':');
                ++nextIndex;
            }
            output += std::to_string(it->second);
            output.push_back(',');
        }
        output.push_back('\n');
    }

    return output;
}

std::string exportSnapshot(const Session& session) {
    const uint64_t hash64 = hashSession64(session);
    const ps_hash128 hash128 = hashSession128(session);
    std::ostringstream stream;
    stream << "{"
           << "\"current_level_index\":" << session.preparedSession.currentLevelIndex << ","
           << "\"current_level_target\":";
    if (session.preparedSession.currentLevelTarget.has_value()) {
        stream << *session.preparedSession.currentLevelTarget;
    } else {
        stream << "null";
    }
    stream << ","
           << "\"title_screen\":" << (session.preparedSession.titleScreen ? "true" : "false") << ","
           << "\"text_mode\":" << (session.preparedSession.textMode ? "true" : "false") << ","
           << "\"title_mode\":" << session.preparedSession.titleMode << ","
           << "\"title_selection\":" << session.preparedSession.titleSelection << ","
           << "\"title_selected\":" << (session.preparedSession.titleSelected ? "true" : "false") << ","
           << "\"message_selected\":" << (session.preparedSession.messageSelected ? "true" : "false") << ","
           << "\"winning\":" << (session.preparedSession.winning ? "true" : "false") << ","
           << "\"movement_word_count_nonzero\":" << countNonZeroWords(session.liveMovements) << ","
           << "\"random_state_valid\":" << (session.randomState.valid ? "true" : "false") << ","
           << "\"random_state_i\":" << static_cast<int32_t>(session.randomState.i) << ","
           << "\"random_state_j\":" << static_cast<int32_t>(session.randomState.j) << ","
           << "\"loaded_level_seed\":\"" << escapeJson(session.preparedSession.loadedLevelSeed) << "\","
           << "\"hash64\":" << hash64 << ","
           << "\"hash128\":{\"lo\":" << hash128.lo << ",\"hi\":" << hash128.hi << "},"
           << "\"movement_board_mask\":[";
    for (size_t index = 0; index < session.boardMovementMask.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << session.boardMovementMask[index];
    }
    stream << "],"
           << "\"random_state_preview_bytes\":[";
    const std::vector<int32_t> previewBytes = previewRandomBytes(session.randomState, 8);
    for (size_t index = 0; index < previewBytes.size(); ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << previewBytes[index];
    }
    stream << "],"
           << "\"serialized_level\":\"" << escapeJson(serializeTestString(session)) << "\""
           << "}";
    return stream.str();
}

void discardTopUndoSnapshot(Session& session) {
    if (!session.undoStack.empty()) {
        session.undoStack.pop_back();
    }
    session.canUndo = !session.undoStack.empty();
}

ps_step_result executeTurn(Session& session, int32_t directionMask, ExecuteTurnOptions options);

void runRulesOnLevelStart(Session& session) {
    runRulesOnLevelStart(session, RuntimeStepOptions{});
}

void runRulesOnLevelStart(Session& session, RuntimeStepOptions options) {
    if (session.game->metadataMap.find("run_rules_on_level_start") == session.game->metadataMap.end()) {
        return;
    }

    session.pendingAgain = false;
    (void)executeTurn(session, 0, ExecuteTurnOptions{
        .pushUndo = false,
        .recordRestartUndo = options.playableUndo,
        .emitAudio = options.emitAudio,
        .ignoreRestartCommand = true,
        .ignoreWin = true,
    });
}

bool wouldAgainChange(Session& session, bool* outWouldModify, bool emitAudio) {
    const uint64_t beforeHash = hashSession64(session);
    session.pendingAgain = false;
    bool wouldModify = false;
    const ps_step_result result = executeTurn(session, 0, ExecuteTurnOptions{
        .pushUndo = false,
        .emitAudio = emitAudio,
        .ignoreWin = true,
        .dontModify = true,
        .observedModification = &wouldModify,
    });
    const bool changed = result.changed || result.transitioned || result.won;
    const int iterations = 1;
    if (outWouldModify != nullptr) {
        *outWouldModify = wouldModify;
    }

    if (againDebugEnabled()) {
        std::ostringstream stream;
        stream << "probe changed=" << (changed ? 1 : 0)
               << " modified=" << (wouldModify ? 1 : 0)
               << " iterations=" << iterations
               << " before_hash=" << beforeHash
               << " after_hash=" << hashSession64(session);
        againDebugLog(stream.str());
    }
    return changed;
}

ps_step_result executeTurn(Session& session, int32_t directionMask, ExecuteTurnOptions options) {
    ps_step_result result{};
    session.lastAudioEvents.clear();
    session.lastUiAudioEvents.clear();
    session.pendingCreateMask.assign(static_cast<size_t>(session.game->strideObject), 0);
    session.pendingDestroyMask.assign(static_cast<size_t>(session.game->strideObject), 0);

    std::optional<Session::UndoSnapshot> localTurnStart;
    const Session::UndoSnapshot* turnStartPtr = nullptr;
    if (options.pushUndo) {
        pushUndoSnapshot(session);
        turnStartPtr = &session.undoStack.back();
    } else {
        localTurnStart = makeUndoSnapshot(session);
        turnStartPtr = &*localTurnStart;
    }
    const Session::UndoSnapshot& turnStart = *turnStartPtr;
    const std::vector<int32_t> startPlayerPositions = directionMask != 0
        ? collectPlayerPositions(session)
        : std::vector<int32_t>{};

    session.pendingAgain = false;
    bool seeded = false;
    bool ruleChanged = false;
    bool moved = false;
    bool lateRuleChanged = false;
    CommandState commands;
    std::vector<bool> bannedGroups;
    int rigidLoopCount = 0;
    while (true) {
        commands = CommandState{};
        restoreSnapshot(session, turnStart, false);
        clearMovementState(session);
        if (directionMask != 0) {
            seeded = seedPlayerMovements(session, directionMask);
        }
        rebuildMasks(session);
        const bool ruleChangedThisPass = applyRuleGroups(session, session.game->rules, session.game->loopPoint, commands, &bannedGroups);
        dumpActiveMovements(session, "pre-resolve");
        const MovementResolveOutcome movementOutcome = resolveMovements(session, &bannedGroups, options.emitAudio);
        rebuildMasks(session);
        if (rigidDebugEnabled()) {
            std::ostringstream stream;
            stream << "turn_pass=" << (rigidLoopCount + 1)
                   << " direction=" << directionMask
                   << " rules_changed=" << (ruleChangedThisPass ? 1 : 0)
                   << " moved=" << (movementOutcome.moved ? 1 : 0)
                   << " should_undo=" << (movementOutcome.shouldUndo ? 1 : 0)
                   << " banned_groups=";
            bool emitted = false;
            for (size_t index = 0; index < bannedGroups.size(); ++index) {
                if (!bannedGroups[index]) {
                    continue;
                }
                if (emitted) {
                    stream << ",";
                }
                stream << index;
                emitted = true;
            }
            if (!emitted) {
                stream << "-";
            }
            rigidDebugLog(stream.str());
        }
        if (movementOutcome.shouldUndo && rigidLoopCount < 49) {
            clearAudioEventsByKind(session, "canmove");
            ++rigidLoopCount;
            continue;
        }
        ruleChanged = ruleChangedThisPass;
        moved = movementOutcome.moved;
        lateRuleChanged = applyRuleGroups(session, session.game->lateRules, session.game->lateLoopPoint, commands, nullptr);
        break;
    }
    const bool modified = session.liveLevel.objects != turnStart.liveLevel.objects;
    if (options.observedModification != nullptr) {
        *options.observedModification = modified;
    }

    if (!startPlayerPositions.empty() && session.game->metadataMap.find("require_player_movement") != session.game->metadataMap.end()) {
        bool someMoved = false;
        for (const int32_t tileIndex : startPlayerPositions) {
            if (!cellContainsPlayer(session, tileIndex)) {
                someMoved = true;
                break;
            }
        }
        if (!someMoved) {
            restoreSnapshot(session, turnStart, false);
            if (options.pushUndo) {
                discardTopUndoSnapshot(session);
            }
            rebuildMasks(session);
            return result;
        }
    }

    if (commandQueueContains(commands, "cancel")) {
        restoreSnapshot(session, turnStart, false);
        if (options.pushUndo) {
            discardTopUndoSnapshot(session);
        }
        session.lastAudioEvents.clear();
        if (!options.dontModify && options.emitAudio) {
            tryPlaySimpleSound(session, "cancel");
        }
        result.changed = options.dontModify
            ? commands.queue.size() > 1
            : (modified || !commands.queue.empty());
        if (options.emitAudio) {
            sortAudioEvents(session);
        }
        result.audio_event_count = session.lastAudioEvents.size();
        result.audio_events = session.lastAudioEvents.empty() ? nullptr : session.lastAudioEvents.data();
        result.ui_audio_event_count = session.lastUiAudioEvents.size();
        result.ui_audio_events = session.lastUiAudioEvents.empty() ? nullptr : session.lastUiAudioEvents.data();
        rebuildMasks(session);
        return result;
    }

    if (options.dontModify) {
        if (modified || commandQueueContains(commands, "win") || commandQueueContains(commands, "restart")) {
            restoreSnapshot(session, turnStart, false);
            rebuildMasks(session);
            result.changed = true;
            return result;
        }
        restoreSnapshot(session, turnStart, false);
        rebuildMasks(session);
        return result;
    }

    if (options.emitAudio) {
        tryPlayMaskSounds(session, session.game->sfxCreationMasks, session.pendingCreateMask, "create");
        tryPlayMaskSounds(session, session.game->sfxDestructionMasks, session.pendingDestroyMask, "destroy");
    }
    processOutputCommands(session, commands, commandQueueContains(commands, "restart"), options.emitAudio);

    if (commandQueueContains(commands, "restart") && !options.ignoreRestartCommand) {
        if (!options.pushUndo && options.recordRestartUndo) {
            pushUndoSnapshot(session);
        }
        restoreRestartTarget(session);
        runRulesOnLevelStart(session, RuntimeStepOptions{
            .playableUndo = options.recordRestartUndo,
            .emitAudio = options.emitAudio,
        });
        if (options.emitAudio) {
            tryPlaySimpleSound(session, "restart");
        }
    }

    const bool won = !options.ignoreWin && (commandQueueContains(commands, "win") || evaluateWinConditions(session));
    const bool transitioned = won && advanceToNextLevel(session);
    if (won) {
        (void)transitioned;
    }
    if (!won && commandQueueContains(commands, "checkpoint")) {
        session.preparedSession.restart.width = session.liveLevel.width;
        session.preparedSession.restart.height = session.liveLevel.height;
        session.preparedSession.restart.objects = session.liveLevel.objects;
        session.preparedSession.restart.oldFlickscreenDat = session.preparedSession.oldFlickscreenDat;
    }

    const bool hasAgain = commandQueueContains(commands, "again");
    bool againWouldChange = false;
    bool againWouldModify = false;
    if (!won && hasAgain && modified) {
        const auto audioBeforeAgainProbe = session.lastAudioEvents;
        againWouldChange = wouldAgainChange(session, &againWouldModify, options.emitAudio);
        if (options.emitAudio) {
            session.lastAudioEvents = audioBeforeAgainProbe;
        }
    }
    if (ruleDebugEnabled()) {
        std::ostringstream stream;
        stream << "turn_summary direction=" << directionMask
               << " seeded=" << (seeded ? 1 : 0)
               << " rule_changed=" << (ruleChanged ? 1 : 0)
               << " moved=" << (moved ? 1 : 0)
               << " late_rule_changed=" << (lateRuleChanged ? 1 : 0)
               << " modified=" << (modified ? 1 : 0)
               << " won=" << (won ? 1 : 0)
               << " transitioned=" << (transitioned ? 1 : 0)
               << " has_again=" << (hasAgain ? 1 : 0)
               << " again_probe_modified=" << (againWouldModify ? 1 : 0)
               << " schedule_again=" << (againWouldChange ? 1 : 0);
        ruleDebugLog(stream.str());
    }
    if (againDebugEnabled() && (hasAgain || session.pendingAgain || modified)) {
        std::ostringstream stream;
        stream << "post-turn modified=" << (modified ? 1 : 0)
               << " won=" << (won ? 1 : 0)
               << " has_again=" << (hasAgain ? 1 : 0)
               << " probe_modified=" << (againWouldModify ? 1 : 0)
               << " schedule_again=" << (againWouldChange ? 1 : 0)
               << " commands=";
        for (size_t index = 0; index < commands.queue.size(); ++index) {
            if (index > 0) {
                stream << ",";
            }
            stream << commands.queue[index];
        }
        againDebugLog(stream.str());
    }
    if (againWouldChange) {
        session.pendingAgain = true;
    }

    result.changed = seeded || ruleChanged || moved || lateRuleChanged || modified || transitioned || !commands.queue.empty();
    result.transitioned = transitioned;
    result.won = won;
    if (options.emitAudio) {
        sortAudioEvents(session);
    }
    result.audio_event_count = session.lastAudioEvents.size();
    result.audio_events = session.lastAudioEvents.empty() ? nullptr : session.lastAudioEvents.data();
    result.ui_audio_event_count = session.lastUiAudioEvents.size();
    result.ui_audio_events = session.lastUiAudioEvents.empty() ? nullptr : session.lastUiAudioEvents.data();
    rebuildMasks(session);
    return result;
}

size_t listInputs(ps_input* output, size_t capacity) {
    static constexpr ps_input kInputs[] = {
        PS_INPUT_UP,
        PS_INPUT_LEFT,
        PS_INPUT_DOWN,
        PS_INPUT_RIGHT,
        PS_INPUT_ACTION,
        PS_INPUT_TICK,
    };
    const size_t total = sizeof(kInputs) / sizeof(kInputs[0]);
    if (output) {
        const size_t count = std::min(capacity, total);
        std::copy_n(kInputs, count, output);
    }
    return total;
}

ps_step_result step(Session& session, ps_input input, RuntimeStepOptions options) {
    ps_step_result result{};
    session.lastAudioEvents.clear();
    session.lastUiAudioEvents.clear();
    if (session.preparedSession.textMode && !session.preparedSession.titleScreen && input == PS_INPUT_ACTION) {
        if (session.preparedSession.level.isMessage) {
            result.transitioned = advanceToNextLevel(session);
        } else {
            session.preparedSession.textMode = false;
            session.preparedSession.messageText.clear();
            session.preparedSession.messageSelected = false;
            result.transitioned = true;
            rebuildMasks(session);
        }
        if (options.emitAudio) {
            tryPlaySimpleSound(session, "closemessage");
        }
        result.changed = true;
        result.ui_audio_event_count = session.lastUiAudioEvents.size();
        result.ui_audio_events = session.lastUiAudioEvents.empty() ? nullptr : session.lastUiAudioEvents.data();
        return result;
    }
    if (session.preparedSession.titleScreen && input == PS_INPUT_ACTION) {
        session.preparedSession.titleScreen = false;
        result.changed = true;
        result.transitioned = true;
        result.audio_event_count = 0;
        result.audio_events = nullptr;
        result.ui_audio_event_count = 0;
        result.ui_audio_events = nullptr;
        rebuildMasks(session);
        return result;
    }

    if (input == PS_INPUT_TICK) {
        return tick(session, options);
    }

    return executeTurn(session, inputToDirectionMask(input), ExecuteTurnOptions{
        .pushUndo = options.playableUndo,
        .recordRestartUndo = options.playableUndo,
        .emitAudio = options.emitAudio,
    });
}

ps_step_result step(Session& session, ps_input input) {
    return step(session, input, RuntimeStepOptions{});
}

ps_step_result tick(Session& session, RuntimeStepOptions options) {
    return executeTurn(session, 0, ExecuteTurnOptions{
        .pushUndo = false,
        .recordRestartUndo = options.playableUndo,
        .emitAudio = options.emitAudio,
    });
}

ps_step_result tick(Session& session) {
    return tick(session, RuntimeStepOptions{});
}

void settlePendingAgain(Session& session, RuntimeStepOptions options) {
    constexpr int kMaxAgainIterations = 500;
    for (int iteration = 0; iteration < kMaxAgainIterations && session.pendingAgain; ++iteration) {
        (void)tick(session, options);
    }
}

void settlePendingAgain(Session& session) {
    settlePendingAgain(session, RuntimeStepOptions{});
}

std::unique_ptr<Error> benchmarkCloneHash(const Session& session, uint32_t iterations, uint32_t threads, ps_benchmark_result& outResult) {
    if (iterations == 0) {
        return std::make_unique<Error>("Iterations must be greater than zero");
    }
    if (threads == 0) {
        threads = 1;
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<std::future<uint64_t>> workers;
    workers.reserve(threads);

    for (uint32_t threadIndex = 0; threadIndex < threads; ++threadIndex) {
        workers.push_back(std::async(std::launch::async, [&session, iterations, threads, threadIndex]() {
            uint64_t hashAccumulator = 0;
            for (uint32_t iteration = threadIndex; iteration < iterations; iteration += threads) {
                auto clone = std::make_unique<Session>(session);
                const auto hash = hashSession64(*clone);
                hashAccumulator ^= hash + static_cast<uint64_t>(iteration);
            }
            return hashAccumulator;
        }));
    }

    uint64_t combinedHash = 0;
    for (auto& worker : workers) {
        combinedHash ^= worker.get();
    }

    const auto finish = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = finish - start;

    outResult.iterations = iterations;
    outResult.threads = threads;
    outResult.elapsed_seconds = elapsed.count();
    outResult.iterations_per_second = elapsed.count() > 0.0 ? static_cast<double>(iterations) / elapsed.count() : 0.0;
    outResult.hash_accumulator = combinedHash;
    return nullptr;
}

void setRuntimeCountersEnabled(bool enabled) {
    gRuntimeCountersEnabled = enabled;
}

void resetRuntimeCounters() {
    gRuntimeCounters.rulesVisited.store(0, std::memory_order_relaxed);
    gRuntimeCounters.rulesSkippedByMask.store(0, std::memory_order_relaxed);
    gRuntimeCounters.candidateCellsTested.store(0, std::memory_order_relaxed);
    gRuntimeCounters.patternTests.store(0, std::memory_order_relaxed);
    gRuntimeCounters.patternMatches.store(0, std::memory_order_relaxed);
    gRuntimeCounters.replacementsAttempted.store(0, std::memory_order_relaxed);
    gRuntimeCounters.replacementsApplied.store(0, std::memory_order_relaxed);
    gRuntimeCounters.rowScans.store(0, std::memory_order_relaxed);
    gRuntimeCounters.ellipsisScans.store(0, std::memory_order_relaxed);
    gRuntimeCounters.maskRebuildCalls.store(0, std::memory_order_relaxed);
    gRuntimeCounters.maskRebuildDirtyCalls.store(0, std::memory_order_relaxed);
    gRuntimeCounters.maskRebuildRows.store(0, std::memory_order_relaxed);
    gRuntimeCounters.maskRebuildColumns.store(0, std::memory_order_relaxed);
}

ps_runtime_counters snapshotRuntimeCounters() {
    ps_runtime_counters counters{};
    counters.rules_visited = gRuntimeCounters.rulesVisited.load(std::memory_order_relaxed);
    counters.rules_skipped_by_mask = gRuntimeCounters.rulesSkippedByMask.load(std::memory_order_relaxed);
    counters.candidate_cells_tested = gRuntimeCounters.candidateCellsTested.load(std::memory_order_relaxed);
    counters.pattern_tests = gRuntimeCounters.patternTests.load(std::memory_order_relaxed);
    counters.pattern_matches = gRuntimeCounters.patternMatches.load(std::memory_order_relaxed);
    counters.replacements_attempted = gRuntimeCounters.replacementsAttempted.load(std::memory_order_relaxed);
    counters.replacements_applied = gRuntimeCounters.replacementsApplied.load(std::memory_order_relaxed);
    counters.row_scans = gRuntimeCounters.rowScans.load(std::memory_order_relaxed);
    counters.ellipsis_scans = gRuntimeCounters.ellipsisScans.load(std::memory_order_relaxed);
    counters.mask_rebuild_calls = gRuntimeCounters.maskRebuildCalls.load(std::memory_order_relaxed);
    counters.mask_rebuild_dirty_calls = gRuntimeCounters.maskRebuildDirtyCalls.load(std::memory_order_relaxed);
    counters.mask_rebuild_rows = gRuntimeCounters.maskRebuildRows.load(std::memory_order_relaxed);
    counters.mask_rebuild_columns = gRuntimeCounters.maskRebuildColumns.load(std::memory_order_relaxed);
    return counters;
}

} // namespace puzzlescript
