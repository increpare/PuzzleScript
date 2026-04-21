#include "core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <map>
#include <numeric>
#include <sstream>
#include <thread>

namespace puzzlescript {
namespace {

void rebuildMasks(Session& session);
std::string toString(const json::Value& value);

struct CommandState {
    std::vector<std::string> queue;
    std::string messageText;
};

struct RuleApplyOutcome {
    bool matched = false;
    bool changed = false;
};

struct MovementResolveOutcome {
    bool moved = false;
    bool shouldUndo = false;
};

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

bool toBool(const json::Value& value) {
    return value.isBool() ? value.asBool() : toInt(value) != 0;
}

bool anyBitsSet(const BitVector& value) {
    return std::any_of(value.begin(), value.end(), [](int32_t word) { return word != 0; });
}

bool commandQueueContains(const CommandState& state, std::string_view command) {
    return std::find(state.queue.begin(), state.queue.end(), std::string(command)) != state.queue.end();
}

void appendAudioEvent(Session& session, int32_t seed, const char* kind) {
    const auto duplicate = std::find_if(session.lastAudioEvents.begin(), session.lastAudioEvents.end(), [seed](const ps_audio_event& event) {
        return event.seed == seed;
    });
    if (duplicate == session.lastAudioEvents.end()) {
        session.lastAudioEvents.push_back(ps_audio_event{seed, kind});
    }
}

void tryPlaySimpleSound(Session& session, std::string_view soundName) {
    if (!session.game->sfxEvents.isObject()) {
        return;
    }
    const auto& events = session.game->sfxEvents.asObject();
    const auto it = events.find(std::string(soundName));
    if (it == events.end()) {
        return;
    }
    appendAudioEvent(session, toInt(it->second), "");
}

void processOutputCommands(Session& session, const CommandState& commands) {
    (void)session;
    (void)commands;
}

void queueRuleCommands(const Rule& rule, CommandState& state) {
    if (!rule.commands.isArray() || rule.commands.asArray().empty()) {
        return;
    }

    const bool preexistingCancel = commandQueueContains(state, "cancel");
    const bool preexistingRestart = commandQueueContains(state, "restart");
    bool currentRuleCancel = false;
    bool currentRuleRestart = false;

    for (const auto& commandValue : rule.commands.asArray()) {
        if (!commandValue.isArray() || commandValue.asArray().empty()) {
            continue;
        }
        const std::string commandName = toString(commandValue.asArray()[0]);
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

    for (const auto& commandValue : rule.commands.asArray()) {
        if (!commandValue.isArray() || commandValue.asArray().empty()) {
            continue;
        }
        const auto& commandArray = commandValue.asArray();
        const std::string commandName = toString(commandArray[0]);
        if (!commandQueueContains(state, commandName)) {
            state.queue.push_back(commandName);
        }
        if (commandName == "message" && commandArray.size() > 1) {
            state.messageText = toString(commandArray[1]);
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

bool anyBitsInCommon(const BitVector& lhs, const BitVector& rhs) {
    const size_t count = std::min(lhs.size(), rhs.size());
    for (size_t index = 0; index < count; ++index) {
        if ((lhs[index] & rhs[index]) != 0) {
            return true;
        }
    }
    return false;
}

bool bitsSetInArray(const BitVector& required, const int32_t* actual, size_t actualCount) {
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

bool bitsSetInArray(const BitVector& required, const BitVector& actual) {
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

std::map<std::string, BitVector> parseBitVectorMap(const json::Value& value) {
    std::map<std::string, BitVector> result;
    for (const auto& [key, entry] : value.asObject()) {
        result.emplace(key, parseIntVector(entry));
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

Replacement parseReplacement(const json::Value& value) {
    const auto& object = value.asObject();
    Replacement replacement;
    replacement.objectsClear = parseIntVector(requireField(object, "objects_clear"));
    replacement.objectsSet = parseIntVector(requireField(object, "objects_set"));
    replacement.movementsClear = parseIntVector(requireField(object, "movements_clear"));
    replacement.movementsSet = parseIntVector(requireField(object, "movements_set"));
    replacement.movementsLayerMask = parseIntVector(requireField(object, "movements_layer_mask"));
    replacement.randomEntityMask = parseIntVector(requireField(object, "random_entity_mask"));
    replacement.randomDirMask = parseIntVector(requireField(object, "random_dir_mask"));
    return replacement;
}

Pattern parsePattern(const json::Value& value) {
    const auto& object = value.asObject();
    Pattern pattern;
    const std::string kind = toString(requireField(object, "kind"));
    pattern.kind = kind == "ellipsis" ? Pattern::Kind::Ellipsis : Pattern::Kind::CellPattern;
    if (pattern.kind == Pattern::Kind::Ellipsis) {
        return pattern;
    }

    pattern.objectsPresent = parseIntVector(requireField(object, "objects_present"));
    pattern.objectsMissing = parseIntVector(requireField(object, "objects_missing"));
    for (const auto& anyMask : requireField(object, "any_objects_present").asArray()) {
        pattern.anyObjectsPresent.push_back(parseIntVector(anyMask));
    }
    pattern.movementsPresent = parseIntVector(requireField(object, "movements_present"));
    pattern.movementsMissing = parseIntVector(requireField(object, "movements_missing"));
    if (const auto* replacement = value.find("replacement"); replacement && !replacement->isNull()) {
        pattern.replacement = parseReplacement(*replacement);
    }
    return pattern;
}

Rule parseRule(const json::Value& value) {
    const auto& object = value.asObject();
    Rule rule;
    rule.direction = toInt(requireField(object, "direction"));
    rule.hasReplacements = toBool(requireField(object, "has_replacements"));
    rule.lineNumber = toInt(requireField(object, "line_number"));
    rule.ellipsisCount = parseIntVector(requireField(object, "ellipsis_count"));
    rule.groupNumber = toInt(requireField(object, "group_number"));
    rule.rigid = toBool(requireField(object, "rigid"));
    rule.commands = requireField(object, "commands");
    rule.isRandom = toBool(requireField(object, "is_random"));
    for (const auto& rowMask : requireField(object, "cell_row_masks").asArray()) {
        rule.cellRowMasks.push_back(parseIntVector(rowMask));
    }
    for (const auto& rowMask : requireField(object, "cell_row_masks_movements").asArray()) {
        rule.cellRowMasksMovements.push_back(parseIntVector(rowMask));
    }
    rule.ruleMask = parseIntVector(requireField(object, "rule_mask"));
    for (const auto& patternRowValue : requireField(object, "patterns").asArray()) {
        std::vector<Pattern> patternRow;
        for (const auto& patternValue : patternRowValue.asArray()) {
            patternRow.push_back(parsePattern(patternValue));
        }
        rule.patterns.push_back(std::move(patternRow));
    }
    return rule;
}

std::vector<std::vector<Rule>> parseRuleGroups(const json::Value& value) {
    std::vector<std::vector<Rule>> groups;
    for (const auto& groupValue : value.asArray()) {
        std::vector<Rule> group;
        for (const auto& ruleValue : groupValue.asArray()) {
            group.push_back(parseRule(ruleValue));
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

WinCondition parseWinCondition(const json::Value& value) {
    const auto& object = value.asObject();
    WinCondition condition;
    condition.quantifier = toInt(requireField(object, "quantifier"));
    condition.filter1 = parseIntVector(requireField(object, "filter1"));
    condition.filter2 = parseIntVector(requireField(object, "filter2"));
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
    level.lineNumber = toInt(requireField(object, "line_number"));
    level.width = toInt(requireField(object, "width"));
    level.height = toInt(requireField(object, "height"));
    level.layerCount = toInt(requireField(object, "layer_count"));
    level.objects = parseIntVector(requireField(object, "objects"));
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
    if (const auto* seed = value.find("loaded_level_seed"); seed && !seed->isNull()) {
        prepared.loadedLevelSeed = toString(*seed);
    }
    prepared.oldFlickscreenDat = parseIntVector(requireField(object, "old_flickscreen_dat"));
    prepared.level = parseLevelTemplate(requireField(object, "level"));
    prepared.serializedLevel = requireField(object, "serialized_level").asString();

    if (const auto* restart = value.find("restart_target"); restart && restart->isObject()) {
        const auto& restartObject = restart->asObject();
        prepared.restart.width = toInt(requireField(restartObject, "width"));
        prepared.restart.height = toInt(requireField(restartObject, "height"));
        prepared.restart.objects = parseIntVector(requireField(restartObject, "objects"));
        prepared.restart.oldFlickscreenDat = parseIntVector(requireField(restartObject, "old_flickscreen_dat"));
    }

    return prepared;
}

BitVector getCellObjects(const Session& session, int32_t tileIndex) {
    BitVector result(static_cast<size_t>(session.game->strideObject), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideObject);
    for (int32_t word = 0; word < session.game->strideObject; ++word) {
        result[static_cast<size_t>(word)] = session.liveLevel.objects[base + static_cast<size_t>(word)];
    }
    return result;
}

const int32_t* getCellObjectsPtr(const Session& session, int32_t tileIndex) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideObject);
    return session.liveLevel.objects.data() + base;
}

void setCellObjects(Session& session, int32_t tileIndex, const BitVector& objects) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideObject);
    for (int32_t word = 0; word < session.game->strideObject; ++word) {
        session.liveLevel.objects[base + static_cast<size_t>(word)] = objects[static_cast<size_t>(word)];
    }
}

BitVector getCellMovements(const Session& session, int32_t tileIndex) {
    BitVector result(static_cast<size_t>(session.game->strideMovement), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        result[static_cast<size_t>(word)] = session.liveMovements[base + static_cast<size_t>(word)];
    }
    return result;
}

const int32_t* getCellMovementsPtr(const Session& session, int32_t tileIndex) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    return session.liveMovements.data() + base;
}

BitVector getCellRigidGroupIndexMask(const Session& session, int32_t tileIndex) {
    BitVector result(static_cast<size_t>(session.game->strideMovement), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        result[static_cast<size_t>(word)] = session.rigidGroupIndexMasks[base + static_cast<size_t>(word)];
    }
    return result;
}

BitVector getCellRigidMovementAppliedMask(const Session& session, int32_t tileIndex) {
    BitVector result(static_cast<size_t>(session.game->strideMovement), 0);
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        result[static_cast<size_t>(word)] = session.rigidMovementAppliedMasks[base + static_cast<size_t>(word)];
    }
    return result;
}

int32_t getShiftedMask5(const BitVector& value, int32_t shift) {
    const int32_t word = shift >> 5;
    const int32_t bit = shift & 31;
    uint32_t result = 0;
    if (word < static_cast<int32_t>(value.size())) {
        result = static_cast<uint32_t>(value[static_cast<size_t>(word)]) >> bit;
    }
    if (bit > 27 && word + 1 < static_cast<int32_t>(value.size())) {
        result |= static_cast<uint32_t>(value[static_cast<size_t>(word + 1)]) << (32 - bit);
    }
    return static_cast<int32_t>(result & 0x1F);
}

void clearShiftedMask5(BitVector& value, int32_t shift) {
    const int32_t word = shift >> 5;
    const int32_t bit = shift & 31;
    if (word >= static_cast<int32_t>(value.size())) {
        return;
    }
    const uint32_t lowMask = 0x1Fu << bit;
    value[static_cast<size_t>(word)] &= ~static_cast<int32_t>(lowMask);
    if (bit > 27 && word + 1 < static_cast<int32_t>(value.size())) {
        const uint32_t highMask = 0x1Fu >> (32 - bit);
        value[static_cast<size_t>(word + 1)] &= ~static_cast<int32_t>(highMask);
    }
}

void setCellMovements(Session& session, int32_t tileIndex, const BitVector& movements) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        session.liveMovements[base + static_cast<size_t>(word)] = movements[static_cast<size_t>(word)];
    }
}

void setCellRigidGroupIndexMask(Session& session, int32_t tileIndex, const BitVector& masks) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        session.rigidGroupIndexMasks[base + static_cast<size_t>(word)] = masks[static_cast<size_t>(word)];
    }
}

void setCellRigidMovementAppliedMask(Session& session, int32_t tileIndex, const BitVector& masks) {
    const size_t base = static_cast<size_t>(tileIndex * session.game->strideMovement);
    for (int32_t word = 0; word < session.game->strideMovement; ++word) {
        session.rigidMovementAppliedMasks[base + static_cast<size_t>(word)] = masks[static_cast<size_t>(word)];
    }
}

void clearRigidState(Session& session) {
    std::fill(session.rigidGroupIndexMasks.begin(), session.rigidGroupIndexMasks.end(), 0);
    std::fill(session.rigidMovementAppliedMasks.begin(), session.rigidMovementAppliedMasks.end(), 0);
}

void setShiftedMask5(BitVector& value, int32_t shift, int32_t bits) {
    const int32_t word = shift >> 5;
    const int32_t bit = shift & 31;
    if (word >= static_cast<int32_t>(value.size())) {
        return;
    }
    const uint32_t packedBits = static_cast<uint32_t>(bits & 0x1F);
    value[static_cast<size_t>(word)] |= static_cast<int32_t>(packedBits << bit);
    if (bit > 27 && word + 1 < static_cast<int32_t>(value.size())) {
        value[static_cast<size_t>(word + 1)] |= static_cast<int32_t>(packedBits >> (32 - bit));
    }
}

std::vector<int32_t> findLayersInMask(const Session& session, const BitVector& cellMask) {
    std::vector<int32_t> layers;
    for (int32_t objectId = 0; objectId < session.game->objectCount; ++objectId) {
        const int32_t word = objectId >> 5;
        const int32_t bit = objectId & 31;
        if (word < static_cast<int32_t>(cellMask.size()) && (cellMask[static_cast<size_t>(word)] & (1 << bit)) != 0) {
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
    if (directionMask == 0 || session.game->playerMask.empty()) {
        return false;
    }

    bool changed = false;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        BitVector cellMask = getCellObjects(session, tileIndex);
        if (!session.game->playerMaskAggregate) {
            if (!anyBitsInCommon(cellMask, session.game->playerMask)) {
                continue;
            }
        } else {
            bool containsAll = true;
            for (size_t word = 0; word < session.game->playerMask.size(); ++word) {
                if ((cellMask[word] & session.game->playerMask[word]) != session.game->playerMask[word]) {
                    containsAll = false;
                    break;
                }
            }
            if (!containsAll) {
                continue;
            }
        }

        for (size_t word = 0; word < cellMask.size(); ++word) {
            cellMask[word] &= session.game->playerMask[word];
        }
        const auto layers = findLayersInMask(session, cellMask);
        if (layers.empty()) {
            continue;
        }

        BitVector movementMask = getCellMovements(session, tileIndex);
        bool tileChanged = false;
        for (const int32_t layer : layers) {
            const int32_t shift = 5 * layer;
            const int32_t word = shift >> 5;
            const int32_t bit = shift & 31;
            const int32_t dirBits = directionMask << bit;
            const int32_t oldValue = movementMask[static_cast<size_t>(word)];
            movementMask[static_cast<size_t>(word)] |= dirBits;
            tileChanged = tileChanged || movementMask[static_cast<size_t>(word)] != oldValue;
        }
        if (tileChanged) {
            setCellMovements(session, tileIndex, movementMask);
            changed = true;
        }
    }

    return changed;
}

bool cellContainsPlayer(const Session& session, int32_t tileIndex) {
    if (session.game->playerMask.empty()) {
        return false;
    }
    const BitVector cellMask = getCellObjects(session, tileIndex);
    if (!session.game->playerMaskAggregate) {
        return anyBitsInCommon(cellMask, session.game->playerMask);
    }
    return bitsSetInArray(session.game->playerMask, cellMask);
}

std::vector<int32_t> collectPlayerPositions(const Session& session) {
    std::vector<int32_t> positions;
    if (session.game->playerMask.empty()) {
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

bool resolveOneLayerMovement(Session& session, int32_t tileIndex, int32_t layer, int32_t directionMask) {
    const auto [dx, dy] = directionMaskToDelta(directionMask);
    const int32_t x = tileIndex / session.liveLevel.height;
    const int32_t y = tileIndex % session.liveLevel.height;
    const int32_t targetX = x + dx;
    const int32_t targetY = y + dy;
    if (targetX < 0 || targetX >= session.liveLevel.width || targetY < 0 || targetY >= session.liveLevel.height) {
        return false;
    }

    const int32_t targetIndex = tileIndex + dy + dx * session.liveLevel.height;
    const BitVector& layerMask = session.game->layerMasks[static_cast<size_t>(layer)];
    BitVector sourceMask = getCellObjects(session, tileIndex);
    const BitVector sourceMaskBeforeMove = sourceMask;
    BitVector targetMask = getCellObjects(session, targetIndex);
    if (directionMask != 16 && anyBitsInCommon(targetMask, layerMask)) {
        return false;
    }

    BitVector movingEntities = sourceMask;
    for (size_t word = 0; word < movingEntities.size(); ++word) {
        movingEntities[word] &= layerMask[word];
        sourceMask[word] &= ~layerMask[word];
        targetMask[word] |= movingEntities[word];
    }

    if (session.game->sfxMovementMasks.isArray()) {
        const auto& movementLayers = session.game->sfxMovementMasks.asArray();
        if (static_cast<size_t>(layer) < movementLayers.size() && movementLayers[static_cast<size_t>(layer)].isArray()) {
            for (const auto& entryValue : movementLayers[static_cast<size_t>(layer)].asArray()) {
                if (!entryValue.isObject()) {
                    continue;
                }
                const auto& entry = entryValue.asObject();
                const auto* objectMaskValue = entryValue.find("objectMask");
                const auto* directionMaskValue = entryValue.find("directionMask");
                const auto* seedValue = entryValue.find("seed");
                if (!objectMaskValue || !directionMaskValue || !seedValue) {
                    continue;
                }
                const BitVector objectMask = parseIntVector(*objectMaskValue);
                const BitVector directionMaskBits = parseIntVector(*directionMaskValue);
                if (!anyBitsInCommon(sourceMaskBeforeMove, objectMask)) {
                    continue;
                }
                BitVector layerDirectionMask(static_cast<size_t>(session.game->strideMovement), 0);
                const int32_t shift = layer * 5;
                const int32_t word = shift >> 5;
                const int32_t bit = shift & 31;
                layerDirectionMask[static_cast<size_t>(word)] = directionMask << bit;
                if (!anyBitsInCommon(directionMaskBits, layerDirectionMask)) {
                    continue;
                }
                const int32_t seed = toInt(*seedValue);
                const auto duplicate = std::find_if(session.lastAudioEvents.begin(), session.lastAudioEvents.end(), [seed](const ps_audio_event& event) {
                    return event.seed == seed;
                });
                if (duplicate == session.lastAudioEvents.end()) {
                    session.lastAudioEvents.push_back(ps_audio_event{seed, ""});
                }
            }
        }
    }

    setCellObjects(session, tileIndex, sourceMask);
    setCellObjects(session, targetIndex, targetMask);
    return true;
}

MovementResolveOutcome resolveMovements(Session& session, std::vector<bool>* bannedGroups) {
    MovementResolveOutcome outcome;
    bool moved = true;
    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    while (moved) {
        moved = false;
        for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
            BitVector movementMask = getCellMovements(session, tileIndex);
            if (!anyBitsSet(movementMask)) {
                continue;
            }
            bool changedTile = false;
            for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
                const int32_t layerMovement = getShiftedMask5(movementMask, 5 * layer);
                if (layerMovement == 0) {
                    continue;
                }
                if (resolveOneLayerMovement(session, tileIndex, layer, layerMovement)) {
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

    for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        BitVector movementMask = getCellMovements(session, tileIndex);
        if (!anyBitsSet(movementMask)) {
            continue;
        }

        if (session.game->rigid) {
            BitVector rigidMovementAppliedMask = getCellRigidMovementAppliedMask(session, tileIndex);
            if (anyBitsSet(rigidMovementAppliedMask)) {
                for (size_t word = 0; word < movementMask.size() && word < rigidMovementAppliedMask.size(); ++word) {
                    movementMask[word] &= rigidMovementAppliedMask[word];
                }
                if (anyBitsSet(movementMask)) {
                    for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
                        if (getShiftedMask5(movementMask, 5 * layer) == 0) {
                            continue;
                        }
                        const BitVector rigidGroupIndexMask = getCellRigidGroupIndexMask(session, tileIndex);
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
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    std::fill(session.liveMovements.begin(), session.liveMovements.end(), 0);
    clearRigidState(session);
    return outcome;
}

bool matchesPatternAt(const Session& session, const Pattern& pattern, int32_t tileIndex) {
    if (pattern.kind != Pattern::Kind::CellPattern) {
        return false;
    }
    const int32_t* objects = getCellObjectsPtr(session, tileIndex);
    const int32_t* movements = getCellMovementsPtr(session, tileIndex);

    for (size_t index = 0; index < pattern.objectsPresent.size(); ++index) {
        if ((objects[index] & pattern.objectsPresent[index]) != pattern.objectsPresent[index]) {
            return false;
        }
    }
    for (size_t index = 0; index < pattern.objectsMissing.size(); ++index) {
        if ((objects[index] & pattern.objectsMissing[index]) != 0) {
            return false;
        }
    }
    for (const auto& anyMask : pattern.anyObjectsPresent) {
        bool found = false;
        for (size_t index = 0; index < anyMask.size(); ++index) {
            if ((objects[index] & anyMask[index]) != 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    for (size_t index = 0; index < pattern.movementsPresent.size(); ++index) {
        if ((movements[index] & pattern.movementsPresent[index]) != pattern.movementsPresent[index]) {
            return false;
        }
    }
    for (size_t index = 0; index < pattern.movementsMissing.size(); ++index) {
        if ((movements[index] & pattern.movementsMissing[index]) != 0) {
            return false;
        }
    }
    return true;
}

bool applyReplacementAt(Session& session, const Rule& rule, const Pattern& pattern, int32_t tileIndex) {
    if (!pattern.replacement.has_value()) {
        return false;
    }
    const auto& replacement = *pattern.replacement;
    BitVector objects = getCellObjects(session, tileIndex);
    BitVector movements = getCellMovements(session, tileIndex);
    const BitVector oldObjects = objects;
    const BitVector oldMovements = movements;
    BitVector rigidGroupIndexMask;
    BitVector rigidMovementAppliedMask;
    bool rigidChange = false;
    BitVector objectsClear = replacement.objectsClear;
    BitVector objectsSet = replacement.objectsSet;
    BitVector movementsSet = replacement.movementsSet;
    BitVector movementsClear = replacement.movementsClear;

    for (size_t word = 0; word < movementsClear.size(); ++word) {
        movementsClear[word] |= replacement.movementsLayerMask[word];
    }

    if (anyBitsSet(replacement.randomEntityMask)) {
        std::vector<int32_t> choices;
        for (int32_t objectId = 0; objectId < session.game->objectCount; ++objectId) {
            const int32_t word = objectId >> 5;
            const int32_t bit = objectId & 31;
            if (word < static_cast<int32_t>(replacement.randomEntityMask.size())
                && (replacement.randomEntityMask[static_cast<size_t>(word)] & (1 << bit)) != 0) {
                choices.push_back(objectId);
            }
        }
        if (!choices.empty()) {
            const size_t chosen = std::min(
                choices.size() - 1,
                static_cast<size_t>(std::floor(randomUniform(session.randomState) * static_cast<double>(choices.size())))
            );
            const int32_t objectId = choices[chosen];
            const int32_t word = objectId >> 5;
            const int32_t bit = objectId & 31;
            objectsSet[static_cast<size_t>(word)] |= (1 << bit);
            if (static_cast<size_t>(objectId) < session.game->objectsById.size()) {
                const int32_t layer = session.game->objectsById[static_cast<size_t>(objectId)].layer;
                if (layer >= 0 && static_cast<size_t>(layer) < session.game->layerMasks.size()) {
                    const BitVector& layerMask = session.game->layerMasks[static_cast<size_t>(layer)];
                    for (size_t idx = 0; idx < objectsClear.size() && idx < layerMask.size(); ++idx) {
                        objectsClear[idx] |= layerMask[idx];
                    }
                    clearShiftedMask5(movementsClear, 5 * layer);
                    const int32_t shift = 5 * layer;
                    const int32_t moveWord = shift >> 5;
                    const int32_t moveBit = shift & 31;
                    if (static_cast<size_t>(moveWord) < movementsClear.size()) {
                        movementsClear[static_cast<size_t>(moveWord)] |= (0x1F << moveBit);
                    }
                }
            }
        }
    }

    if (anyBitsSet(replacement.randomDirMask)) {
        for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
            const int32_t shift = 5 * layer;
            if (getShiftedMask5(replacement.randomDirMask, shift) != 0) {
                const int32_t randomDir = static_cast<int32_t>(std::floor(randomUniform(session.randomState) * 4.0));
                const int32_t word = (shift + randomDir) >> 5;
                const int32_t bit = (shift + randomDir) & 31;
                if (static_cast<size_t>(word) < movementsSet.size()) {
                    movementsSet[static_cast<size_t>(word)] |= (1 << bit);
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

    if (rule.rigid && !replacement.movementsLayerMask.empty()) {
        const int32_t rigidGroupIndex = (rule.groupNumber >= 0
            && static_cast<size_t>(rule.groupNumber) < session.game->groupNumberToRigidGroupIndex.size())
            ? session.game->groupNumberToRigidGroupIndex[static_cast<size_t>(rule.groupNumber)] + 1
            : 0;
        if (rigidGroupIndex > 0) {
            BitVector rigidMask(static_cast<size_t>(session.game->strideMovement), 0);
            for (int32_t layer = 0; layer < session.game->layerCount; ++layer) {
                const int32_t shift = 5 * layer;
                if (getShiftedMask5(replacement.movementsLayerMask, shift) != 0) {
                    setShiftedMask5(rigidMask, shift, rigidGroupIndex);
                }
            }

            rigidGroupIndexMask = getCellRigidGroupIndexMask(session, tileIndex);
            rigidMovementAppliedMask = getCellRigidMovementAppliedMask(session, tileIndex);
            if (!bitsSetInArray(rigidMask, rigidGroupIndexMask)
                && !bitsSetInArray(replacement.movementsLayerMask, rigidMovementAppliedMask)) {
                for (size_t word = 0; word < rigidGroupIndexMask.size() && word < rigidMask.size(); ++word) {
                    rigidGroupIndexMask[word] |= rigidMask[word];
                }
                for (size_t word = 0; word < rigidMovementAppliedMask.size() && word < replacement.movementsLayerMask.size(); ++word) {
                    rigidMovementAppliedMask[word] |= replacement.movementsLayerMask[word];
                }
                rigidChange = true;
            }
        }
    }

    if (objects == oldObjects && movements == oldMovements && !rigidChange) {
        return false;
    }
    setCellObjects(session, tileIndex, objects);
    setCellMovements(session, tileIndex, movements);
    if (rigidChange) {
        setCellRigidGroupIndexMask(session, tileIndex, rigidGroupIndexMask);
        setCellRigidMovementAppliedMask(session, tileIndex, rigidMovementAppliedMask);
    }
    return true;
}

std::vector<int32_t> collectRowMatches(
    const Session& session,
    const std::vector<Pattern>& row,
    int32_t direction,
    const BitVector& rowObjectMask,
    const BitVector& rowMovementMask
) {
    std::vector<int32_t> matches;
    if (row.empty()) {
        return matches;
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
            return matches;
    }

    const bool horizontal = direction > 2;
    const auto [dx, dy] = directionMaskToDelta(direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        return matches;
    }

    if (!bitsSetInArray(rowObjectMask, session.boardMask) || !bitsSetInArray(rowMovementMask, session.boardMovementMask)) {
        return matches;
    }

    if (horizontal) {
        for (int32_t y = ymin; y < ymax; ++y) {
            const int32_t* rowObjects = session.rowMasks.data() + static_cast<size_t>(y * session.game->strideObject);
            const int32_t* rowMovements = session.rowMovementMasks.data() + static_cast<size_t>(y * session.game->strideMovement);
            if (!bitsSetInArray(rowObjectMask, rowObjects, static_cast<size_t>(session.game->strideObject))
                || !bitsSetInArray(rowMovementMask, rowMovements, static_cast<size_t>(session.game->strideMovement))) {
                continue;
            }
            for (int32_t x = xmin; x < xmax; ++x) {
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
            const int32_t* columnObjects = session.columnMasks.data() + static_cast<size_t>(x * session.game->strideObject);
            const int32_t* columnMovements = session.columnMovementMasks.data() + static_cast<size_t>(x * session.game->strideMovement);
            if (!bitsSetInArray(rowObjectMask, columnObjects, static_cast<size_t>(session.game->strideObject))
                || !bitsSetInArray(rowMovementMask, columnMovements, static_cast<size_t>(session.game->strideMovement))) {
                continue;
            }
            for (int32_t y = ymin; y < ymax; ++y) {
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

bool ruleCanPossiblyMatch(const Session& session, const Rule& rule) {
    return bitsSetInArray(rule.ruleMask, session.boardMask);
}

RuleApplyOutcome tryApplySimpleRule(Session& session, const Rule& rule, CommandState& commands) {
    if (rule.isRandom || rule.patterns.empty()) {
        return {};
    }
    if (!ruleCanPossiblyMatch(session, rule)) {
        return {};
    }
    if (rule.patterns.size() == 1 && rule.ellipsisCount.size() == 1 && rule.ellipsisCount[0] == 1) {
        const auto& row = rule.patterns[0];
        int32_t ellipsisIndex = -1;
        for (int32_t i = 0; i < static_cast<int32_t>(row.size()); ++i) {
            if (row[static_cast<size_t>(i)].kind == Pattern::Kind::Ellipsis) {
                ellipsisIndex = i;
                break;
            }
        }
        if (ellipsisIndex < 0) {
            return {};
        }

        const int32_t prefixLength = ellipsisIndex;
        const int32_t suffixLength = static_cast<int32_t>(row.size()) - ellipsisIndex - 1;
        const auto [dx, dy] = directionMaskToDelta(rule.direction);
        const int32_t parallelDelta = dx * session.liveLevel.height + dy;
        if (parallelDelta == 0) {
            return {};
        }

        auto availableAlongDirection = [&](int32_t startIndex) {
            const int32_t x = startIndex / session.liveLevel.height;
            const int32_t y = startIndex % session.liveLevel.height;
            switch (rule.direction) {
                case 1: return y + 1;
                case 2: return session.liveLevel.height - y;
                case 4: return x + 1;
                case 8: return session.liveLevel.width - x;
                default: return 0;
            }
        };

        std::vector<std::vector<int32_t>> matches;
        for (int32_t tileIndex = 0; tileIndex < session.liveLevel.width * session.liveLevel.height; ++tileIndex) {
            const int32_t available = availableAlongDirection(tileIndex);
            const int32_t maxGap = available - (prefixLength + suffixLength);
            if (maxGap < 0) {
                continue;
            }
            for (int32_t gap = 0; gap <= maxGap; ++gap) {
                std::vector<int32_t> positions;
                positions.reserve(static_cast<size_t>(prefixLength + suffixLength));
                bool matched = true;
                for (int32_t cellIndex = 0; cellIndex < prefixLength; ++cellIndex) {
                    const int32_t matchIndex = tileIndex + cellIndex * parallelDelta;
                    if (!matchesPatternAt(session, row[static_cast<size_t>(cellIndex)], matchIndex)) {
                        matched = false;
                        break;
                    }
                    positions.push_back(matchIndex);
                }
                for (int32_t cellIndex = 0; matched && cellIndex < suffixLength; ++cellIndex) {
                    const int32_t rowIndex = ellipsisIndex + 1 + cellIndex;
                    const int32_t matchIndex = tileIndex + (ellipsisIndex + gap + cellIndex) * parallelDelta;
                    if (!matchesPatternAt(session, row[static_cast<size_t>(rowIndex)], matchIndex)) {
                        matched = false;
                        break;
                    }
                    positions.push_back(matchIndex);
                }
                if (matched) {
                    matches.push_back(std::move(positions));
                }
            }
        }

        if (matches.empty()) {
            return {};
        }

        queueRuleCommands(rule, commands);

        bool changed = false;
        for (size_t matchIndex = 0; matchIndex < matches.size(); ++matchIndex) {
            const auto& positions = matches[matchIndex];
            if (matchIndex > 0) {
                bool stillMatches = true;
                int32_t positionIndex = 0;
                for (int32_t cellIndex = 0; cellIndex < static_cast<int32_t>(row.size()); ++cellIndex) {
                    if (cellIndex == ellipsisIndex) {
                        continue;
                    }
                    if (!matchesPatternAt(session, row[static_cast<size_t>(cellIndex)], positions[static_cast<size_t>(positionIndex++)])) {
                        stillMatches = false;
                        break;
                    }
                }
                if (!stillMatches) {
                    continue;
                }
            }

            int32_t positionIndex = 0;
            for (int32_t cellIndex = 0; cellIndex < static_cast<int32_t>(row.size()); ++cellIndex) {
                if (cellIndex == ellipsisIndex) {
                    continue;
                }
                changed = applyReplacementAt(session, rule, row[static_cast<size_t>(cellIndex)], positions[static_cast<size_t>(positionIndex++)]) || changed;
            }
        }

        return RuleApplyOutcome{true, changed};
    }
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        if (rowIndex >= rule.ellipsisCount.size() || rule.ellipsisCount[rowIndex] != 0 || rule.patterns[rowIndex].empty()) {
            return {};
        }
    }
    const auto [dx, dy] = directionMaskToDelta(rule.direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        return {};
    }

    std::vector<std::vector<int32_t>> rowMatches;
    rowMatches.reserve(rule.patterns.size());
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const auto& row = rule.patterns[rowIndex];
        const BitVector& rowObjectMask = rowIndex < rule.cellRowMasks.size()
            ? rule.cellRowMasks[rowIndex]
            : rule.ruleMask;
        const BitVector& rowMovementMask = rowIndex < rule.cellRowMasksMovements.size()
            ? rule.cellRowMasksMovements[rowIndex]
            : BitVector{};
        auto matches = collectRowMatches(session, row, rule.direction, rowObjectMask, rowMovementMask);
        if (matches.empty()) {
            return {};
        }
        rowMatches.push_back(std::move(matches));
    }

    std::vector<std::vector<int32_t>> tuples(1);
    for (const auto& matches : rowMatches) {
        std::vector<std::vector<int32_t>> newTuples;
        newTuples.reserve(tuples.size() * matches.size());
        for (const int32_t match : matches) {
            for (const auto& tuple : tuples) {
                std::vector<int32_t> newTuple = tuple;
                newTuple.push_back(match);
                newTuples.push_back(std::move(newTuple));
            }
        }
        tuples = std::move(newTuples);
    }

    if (tuples.empty()) {
        return {};
    }

    queueRuleCommands(rule, commands);

    bool changed = false;
    for (size_t tupleIndex = 0; tupleIndex < tuples.size(); ++tupleIndex) {
        const auto& tuple = tuples[tupleIndex];
        if (tupleIndex > 0) {
            bool stillMatches = true;
            for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
                if (!rowStillMatchesAt(session, rule.patterns[rowIndex], tuple[rowIndex], delta)) {
                    stillMatches = false;
                    break;
                }
            }
            if (!stillMatches) {
                continue;
            }
        }
        for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
            changed = applyRowAt(session, rule, rule.patterns[rowIndex], tuple[rowIndex], delta) || changed;
        }
    }
    return RuleApplyOutcome{true, changed};
}

bool collectRandomRuleMatches(const Session& session, const Rule& rule, std::vector<std::vector<int32_t>>& outMatches) {
    if (!rule.isRandom || rule.patterns.empty()) {
        return false;
    }
    if (!ruleCanPossiblyMatch(session, rule)) {
        outMatches.clear();
        return true;
    }

    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        if (rowIndex >= rule.ellipsisCount.size() || rule.ellipsisCount[rowIndex] != 0 || rule.patterns[rowIndex].empty()) {
            return false;
        }
    }

    const auto [dx, dy] = directionMaskToDelta(rule.direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        return false;
    }

    std::vector<std::vector<int32_t>> rowMatches;
    rowMatches.reserve(rule.patterns.size());
    for (size_t rowIndex = 0; rowIndex < rule.patterns.size(); ++rowIndex) {
        const auto& row = rule.patterns[rowIndex];
        const BitVector& rowObjectMask = rowIndex < rule.cellRowMasks.size()
            ? rule.cellRowMasks[rowIndex]
            : rule.ruleMask;
        const BitVector& rowMovementMask = rowIndex < rule.cellRowMasksMovements.size()
            ? rule.cellRowMasksMovements[rowIndex]
            : BitVector{};
        auto matches = collectRowMatches(session, row, rule.direction, rowObjectMask, rowMovementMask);
        if (matches.empty()) {
            outMatches.clear();
            return true;
        }
        rowMatches.push_back(std::move(matches));
    }

    outMatches.assign(1, {});
    for (const auto& matches : rowMatches) {
        std::vector<std::vector<int32_t>> newTuples;
        newTuples.reserve(outMatches.size() * matches.size());
        for (const int32_t match : matches) {
            for (const auto& tuple : outMatches) {
                std::vector<int32_t> newTuple = tuple;
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
        std::vector<int32_t> tuple;
    };

    std::vector<Candidate> candidates;
    std::vector<std::vector<int32_t>> matches;
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

    const size_t chosenIndex = std::min(
        candidates.size() - 1,
        static_cast<size_t>(std::floor(randomUniform(session.randomState) * static_cast<double>(candidates.size())))
    );
    const Candidate& chosen = candidates[chosenIndex];
    queueRuleCommands(*chosen.rule, commands);
    const auto [dx, dy] = directionMaskToDelta(chosen.rule->direction);
    const int32_t delta = dx * session.liveLevel.height + dy;
    if (delta == 0) {
        return false;
    }

    bool changed = false;
    for (size_t rowIndex = 0; rowIndex < chosen.tuple.size() && rowIndex < chosen.rule->patterns.size(); ++rowIndex) {
        changed = applyRowAt(session, *chosen.rule, chosen.rule->patterns[rowIndex], chosen.tuple[rowIndex], delta) || changed;
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
            const RuleApplyOutcome outcome = tryApplySimpleRule(session, rule, commands);
            madeChange = outcome.changed || madeChange;
            if (outcome.changed) {
                rebuildMasks(session);
            }
        }
        hasChanges = hasChanges || madeChange;
    }
    return hasChanges;
}

std::optional<int32_t> lookupLoopPoint(const json::Value& loopPoint, int32_t index) {
    if (loopPoint.isObject()) {
        const auto& object = loopPoint.asObject();
        const auto it = object.find(std::to_string(index));
        if (it != object.end() && !it->second.isNull()) {
            return toInt(it->second);
        }
    } else if (loopPoint.isArray()) {
        const auto& array = loopPoint.asArray();
        if (index >= 0 && static_cast<size_t>(index) < array.size() && !array[static_cast<size_t>(index)].isNull()) {
            return toInt(array[static_cast<size_t>(index)]);
        }
    }
    return std::nullopt;
}

bool applyRuleGroups(
    Session& session,
    const std::vector<std::vector<Rule>>& groups,
    const json::Value& loopPoint,
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

void rebuildMasks(Session& session) {
    const int32_t objectStride = session.game->strideObject;
    const int32_t movementStride = session.game->strideMovement;
    const int32_t width = session.liveLevel.width;
    const int32_t height = session.liveLevel.height;

    session.rowMasks.assign(static_cast<size_t>(height * objectStride), 0);
    session.columnMasks.assign(static_cast<size_t>(width * objectStride), 0);
    session.boardMask.assign(static_cast<size_t>(objectStride), 0);
    session.rowMovementMasks.assign(static_cast<size_t>(height * movementStride), 0);
    session.columnMovementMasks.assign(static_cast<size_t>(width * movementStride), 0);
    session.boardMovementMask.assign(static_cast<size_t>(movementStride), 0);

    for (int32_t x = 0; x < width; ++x) {
        for (int32_t y = 0; y < height; ++y) {
            const int32_t tileIndex = x * height + y;
            const size_t objectBase = static_cast<size_t>(tileIndex * objectStride);
            for (int32_t word = 0; word < objectStride; ++word) {
                const int32_t value = session.liveLevel.objects[objectBase + static_cast<size_t>(word)];
                session.rowMasks[static_cast<size_t>(y * objectStride + word)] |= value;
                session.columnMasks[static_cast<size_t>(x * objectStride + word)] |= value;
                session.boardMask[static_cast<size_t>(word)] |= value;
            }
            const size_t movementBase = static_cast<size_t>(tileIndex * movementStride);
            for (int32_t word = 0; word < movementStride; ++word) {
                const int32_t value = session.liveMovements[movementBase + static_cast<size_t>(word)];
                session.rowMovementMasks[static_cast<size_t>(y * movementStride + word)] |= value;
                session.columnMovementMasks[static_cast<size_t>(x * movementStride + word)] |= value;
                session.boardMovementMask[static_cast<size_t>(word)] |= value;
            }
        }
    }
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
    bytes.insert(bytes.end(), objectBytes, objectBytes + objects.size() * sizeof(int32_t));
    const auto& movements = session.liveMovements;
    const auto* movementBytes = reinterpret_cast<const uint8_t*>(movements.data());
    bytes.insert(bytes.end(), movementBytes, movementBytes + movements.size() * sizeof(int32_t));
    bytes.insert(bytes.end(), session.preparedSession.loadedLevelSeed.begin(), session.preparedSession.loadedLevelSeed.end());
    return bytes;
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

void restoreSnapshot(Session& session, const Session::UndoSnapshot& snapshot) {
    session.preparedSession = snapshot.preparedSession;
    session.liveLevel = snapshot.liveLevel;
    session.liveMovements = snapshot.liveMovements;
    session.rigidGroupIndexMasks = snapshot.rigidGroupIndexMasks;
    session.rigidMovementAppliedMasks = snapshot.rigidMovementAppliedMasks;
    session.randomState = snapshot.randomState;
    session.pendingAgain = false;
    rebuildMasks(session);
}

void pushUndoSnapshot(Session& session) {
    session.undoStack.push_back(Session::UndoSnapshot{
        session.preparedSession,
        session.liveLevel,
        session.liveMovements,
        session.rigidGroupIndexMasks,
        session.rigidMovementAppliedMasks,
        session.randomState,
    });
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
    rebuildMasks(session);
}

bool matchesFilter(const BitVector& filter, bool aggregate, const BitVector& cell) {
    return aggregate ? bitsSetInArray(filter, cell) : anyBitsInCommon(filter, cell);
}

bool evaluateWinConditions(const Session& session) {
    if (session.game->winConditions.empty()) {
        return false;
    }

    const int32_t tileCount = session.liveLevel.width * session.liveLevel.height;
    for (const auto& condition : session.game->winConditions) {
        bool rulePassed = true;
        switch (condition.quantifier) {
            case -1: {
                for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
                    const BitVector cell = getCellObjects(session, tileIndex);
                    if (matchesFilter(condition.filter1, condition.aggr1, cell)
                        && matchesFilter(condition.filter2, condition.aggr2, cell)) {
                        rulePassed = false;
                        break;
                    }
                }
                break;
            }
            case 0: {
                bool passedTest = false;
                for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
                    const BitVector cell = getCellObjects(session, tileIndex);
                    if (matchesFilter(condition.filter1, condition.aggr1, cell)
                        && matchesFilter(condition.filter2, condition.aggr2, cell)) {
                        passedTest = true;
                        break;
                    }
                }
                rulePassed = passedTest;
                break;
            }
            case 1: {
                for (int32_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
                    const BitVector cell = getCellObjects(session, tileIndex);
                    if (matchesFilter(condition.filter1, condition.aggr1, cell)
                        && !matchesFilter(condition.filter2, condition.aggr2, cell)) {
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
        session.preparedSession.winning = false;
        if (session.preparedSession.textMode) {
            session.liveMovements.assign(static_cast<size_t>(session.liveLevel.width * session.liveLevel.height * session.game->strideMovement), 0);
            session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
            session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
            session.pendingAgain = false;
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
    session.preparedSession.winning = false;
    session.liveMovements.assign(static_cast<size_t>(session.liveLevel.width * session.liveLevel.height * session.game->strideMovement), 0);
    session.rigidGroupIndexMasks.assign(session.liveMovements.size(), 0);
    session.rigidMovementAppliedMasks.assign(session.liveMovements.size(), 0);
    session.pendingAgain = false;
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
    seedRandomState(session.randomState, session.preparedSession.loadedLevelSeed);
    rebuildMasks(session);
}

} // namespace

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
            game->glyphDict = parseBitVectorMap(glyphDict->second);
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
                game->layerMasks.push_back(parseIntVector(maskValue));
            }
        }
        if (const auto objectMasks = gameObject.find("object_masks"); objectMasks != gameObject.end()) {
            game->objectMasks = parseBitVectorMap(objectMasks->second);
        }
        if (const auto aggregateMasks = gameObject.find("aggregate_masks"); aggregateMasks != gameObject.end()) {
            game->aggregateMasks = parseBitVectorMap(aggregateMasks->second);
        }
        if (const auto playerMask = gameObject.find("player_mask"); playerMask != gameObject.end()) {
            if (playerMask->second.isObject()) {
                const auto& playerMaskObject = playerMask->second.asObject();
                if (const auto aggregate = playerMaskObject.find("aggregate"); aggregate != playerMaskObject.end()) {
                    game->playerMaskAggregate = toBool(aggregate->second);
                }
                if (const auto mask = playerMaskObject.find("mask"); mask != playerMaskObject.end()) {
                    game->playerMask = parseIntVector(mask->second);
                }
            } else {
                game->playerMask = parseIntVector(playerMask->second);
            }
        }
        if (const auto propertiesSingleLayer = gameObject.find("properties_single_layer"); propertiesSingleLayer != gameObject.end()) {
            game->propertiesSingleLayer = propertiesSingleLayer->second;
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
            game->rules = parseRuleGroups(rules->second);
        }
        if (const auto lateRules = gameObject.find("late_rules"); lateRules != gameObject.end()) {
            game->lateRules = parseRuleGroups(lateRules->second);
        }
        if (const auto loopPoint = gameObject.find("loop_point"); loopPoint != gameObject.end()) {
            game->loopPoint = loopPoint->second;
        }
        if (const auto lateLoopPoint = gameObject.find("late_loop_point"); lateLoopPoint != gameObject.end()) {
            game->lateLoopPoint = lateLoopPoint->second;
        }
        if (const auto winconditions = gameObject.find("winconditions"); winconditions != gameObject.end()) {
            for (const auto& conditionValue : winconditions->second.asArray()) {
                game->winConditions.push_back(parseWinCondition(conditionValue));
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
            game->sfxEvents = sfxEvents->second;
        }
        if (const auto sfxCreationMasks = gameObject.find("sfx_creation_masks"); sfxCreationMasks != gameObject.end()) {
            game->sfxCreationMasks = sfxCreationMasks->second;
        }
        if (const auto sfxDestructionMasks = gameObject.find("sfx_destruction_masks"); sfxDestructionMasks != gameObject.end()) {
            game->sfxDestructionMasks = sfxDestructionMasks->second;
        }
        if (const auto sfxMovementMasks = gameObject.find("sfx_movement_masks"); sfxMovementMasks != gameObject.end()) {
            game->sfxMovementMasks = sfxMovementMasks->second;
        }
        if (const auto sfxMovementFailureMasks = gameObject.find("sfx_movement_failure_masks"); sfxMovementFailureMasks != gameObject.end()) {
            game->sfxMovementFailureMasks = sfxMovementFailureMasks->second;
        }
        if (const auto sounds = gameObject.find("sounds"); sounds != gameObject.end()) {
            game->sounds = sounds->second;
        }

        if (const auto prepared = rootObject.find("prepared_session"); prepared != rootObject.end()) {
            try {
                game->preparedSession = parsePreparedSession(prepared->second);
            } catch (const std::exception& error) {
                throw json::ParseError("Failed parsing prepared_session: " + std::string(error.what()));
            }
        }

        game->root = std::move(root);
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
    session.preparedSession.winning = false;
    session.preparedSession.restart.width = session.preparedSession.level.width;
    session.preparedSession.restart.height = session.preparedSession.level.height;
    session.preparedSession.restart.objects = session.preparedSession.level.objects;
    session.preparedSession.restart.oldFlickscreenDat = session.preparedSession.oldFlickscreenDat;
    resetToPrepared(session);
    return nullptr;
}

bool restart(Session& session) {
    pushUndoSnapshot(session);
    restoreRestartTarget(session);
    return true;
}

bool undo(Session& session) {
    if (session.undoStack.empty()) {
        return false;
    }
    const auto snapshot = std::move(session.undoStack.back());
    session.undoStack.pop_back();
    restoreSnapshot(session, snapshot);
    session.canUndo = !session.undoStack.empty();
    return true;
}

uint64_t hashSession64(const Session& session) {
    const auto bytes = buildSessionHashBytes(session);
    return fnv1a64(bytes.data(), bytes.size());
}

ps_hash128 hashSession128(const Session& session) {
    return dualHash128(buildSessionHashBytes(session));
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
            for (int32_t bit = 0; bit < 32 * stride; ++bit) {
                const int32_t word = bit >> 5;
                const int32_t mask = 1 << (bit & 31);
                if (session.liveLevel.objects[cellOffset + static_cast<size_t>(word)] & mask) {
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

ps_step_result executeTurn(Session& session, int32_t directionMask, bool pushUndo) {
    ps_step_result result{};
    session.lastAudioEvents.clear();

    const Session::UndoSnapshot turnStart{
        session.preparedSession,
        session.liveLevel,
        session.liveMovements,
        session.rigidGroupIndexMasks,
        session.rigidMovementAppliedMasks,
        session.randomState,
    };
    const std::vector<int32_t> startPlayerPositions = collectPlayerPositions(session);

    if (pushUndo) {
        pushUndoSnapshot(session);
    }

    session.pendingAgain = false;
    std::fill(session.liveMovements.begin(), session.liveMovements.end(), 0);
    clearRigidState(session);
    const bool seeded = directionMask != 0 && seedPlayerMovements(session, directionMask);
    bool ruleChanged = false;
    bool moved = false;
    bool lateRuleChanged = false;
    CommandState commands;
    std::vector<bool> bannedGroups;
    int rigidLoopCount = 0;
    while (true) {
        commands = CommandState{};
        restoreSnapshot(session, turnStart);
        std::fill(session.liveMovements.begin(), session.liveMovements.end(), 0);
        clearRigidState(session);
        if (directionMask != 0) {
            (void)seedPlayerMovements(session, directionMask);
        }
        rebuildMasks(session);
        const bool ruleChangedThisPass = applyRuleGroups(session, session.game->rules, session.game->loopPoint, commands, &bannedGroups);
        const MovementResolveOutcome movementOutcome = resolveMovements(session, &bannedGroups);
        rebuildMasks(session);
        if (movementOutcome.shouldUndo && rigidLoopCount < 49) {
            ++rigidLoopCount;
            continue;
        }
        ruleChanged = ruleChangedThisPass;
        moved = movementOutcome.moved;
        lateRuleChanged = applyRuleGroups(session, session.game->lateRules, session.game->lateLoopPoint, commands, nullptr);
        break;
    }
    const bool modified = session.liveLevel.objects != turnStart.liveLevel.objects;

    if (!startPlayerPositions.empty() && session.game->metadataMap.find("require_player_movement") != session.game->metadataMap.end()) {
        bool someMoved = false;
        for (const int32_t tileIndex : startPlayerPositions) {
            if (!cellContainsPlayer(session, tileIndex)) {
                someMoved = true;
                break;
            }
        }
        if (!someMoved) {
            restoreSnapshot(session, turnStart);
            if (pushUndo) {
                discardTopUndoSnapshot(session);
            }
            rebuildMasks(session);
            return result;
        }
    }

    processOutputCommands(session, commands);

    if (commandQueueContains(commands, "cancel")) {
        restoreSnapshot(session, turnStart);
        if (pushUndo) {
            discardTopUndoSnapshot(session);
        }
        tryPlaySimpleSound(session, "cancel");
        result.changed = modified || !commands.queue.empty();
        result.audio_event_count = session.lastAudioEvents.size();
        result.audio_events = session.lastAudioEvents.empty() ? nullptr : session.lastAudioEvents.data();
        rebuildMasks(session);
        return result;
    }

    if (commandQueueContains(commands, "restart")) {
        restoreRestartTarget(session);
        tryPlaySimpleSound(session, "restart");
    }

    const bool won = commandQueueContains(commands, "win") || evaluateWinConditions(session);
    const bool transitioned = won && advanceToNextLevel(session);
    if (won) {
        (void)transitioned;
    }

    if (!won && commandQueueContains(commands, "again") && modified) {
        session.pendingAgain = true;
    }

    result.changed = seeded || ruleChanged || moved || lateRuleChanged || modified || transitioned || !commands.queue.empty();
    result.transitioned = transitioned;
    result.won = won;
    result.audio_event_count = session.lastAudioEvents.size();
    result.audio_events = session.lastAudioEvents.empty() ? nullptr : session.lastAudioEvents.data();
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

ps_step_result step(Session& session, ps_input input) {
    ps_step_result result{};
    session.lastAudioEvents.clear();
    if (session.preparedSession.titleScreen && input == PS_INPUT_ACTION) {
        session.preparedSession.titleScreen = false;
        result.changed = true;
        result.transitioned = true;
        result.audio_event_count = 0;
        result.audio_events = nullptr;
        rebuildMasks(session);
        return result;
    }

    if (input == PS_INPUT_TICK) {
        return tick(session);
    }

    return executeTurn(session, inputToDirectionMask(input), true);
}

ps_step_result tick(Session& session) {
    if (session.pendingAgain) {
        return executeTurn(session, 0, false);
    }
    session.lastAudioEvents.clear();
    std::fill(session.liveMovements.begin(), session.liveMovements.end(), 0);
    rebuildMasks(session);
    return ps_step_result{};
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

} // namespace puzzlescript
